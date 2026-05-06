"""Batch evaluation harness for PandaSourceNavigator.

Feeds a collection of errorDialog strings through the navigator, deduplicates
them by normalised pattern, and reports which patterns the navigator struggles
with (no_candidates, too_many_candidates, irrelevant) together with diagnostic
flags (term_fallback, low_term_quality, rounds_exhausted).

Usage::

    # Terms-only (LLM extraction + grep, no navigation rounds):
    python -m bamboo.scripts.eval_source_navigator --from-file errors.json --terms-only

    # Full pipeline on a file of task IDs (auto-detected, fetched from PanDA):
    python -m bamboo.scripts.eval_source_navigator --from-file task_ids.json --output report.json

    # With LLM judge on suspicious results (identifier coverage < 0.3):
    python -m bamboo.scripts.eval_source_navigator --from-file errors.json --judge

    # Single string smoke-test:
    python -m bamboo.scripts.eval_source_navigator "scout_ramCount threshold exceeded"
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from bamboo.agents.panda_source_navigator import (
    PandaSourceNavigator,
    _get_pkg_roots,
    _grep_sliding_window,
)
from bamboo.llm.prompts import SOURCE_GREP_TERMS_PROMPT
from bamboo.utils.narrator import set_narrator


# ── normalisation ────────────────────────────────────────────────────────────

_HTML_TAG = re.compile(r"<[^>]+>")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_FLOAT = re.compile(r"\b\d+\.\d+(?:[eE][+-]?\d+)?\b")
_INT = re.compile(r"\b\d+\b")
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
_HOST = re.compile(r"\b(?:[a-zA-Z0-9][\w-]*\.){2,}[a-zA-Z]{2,}\b")


def normalize_error_dialog(s: str) -> str:
    s = _HTML_TAG.sub(" ", s)
    s = _TIMESTAMP.sub("<TIMESTAMP>", s)
    s = _FLOAT.sub("<FLOAT>", s)
    s = _INT.sub("<NUM>", s)
    s = _UUID.sub("<UUID>", s)
    s = _HOST.sub("<HOST>", s)
    return " ".join(s.split())


# ── clustering ───────────────────────────────────────────────────────────────

@dataclass
class Cluster:
    key: str
    count: int = 0
    representative: str = ""


def cluster_strings(strings: list[str]) -> list[Cluster]:
    clusters: dict[str, Cluster] = {}
    for s in strings:
        key = normalize_error_dialog(s)
        if key not in clusters:
            clusters[key] = Cluster(key=key, count=0, representative=s)
        c = clusters[key]
        c.count += 1
        if len(s) < len(c.representative):
            c.representative = s
    return sorted(clusters.values(), key=lambda c: c.count, reverse=True)


# ── input loading ────────────────────────────────────────────────────────────

def _looks_like_task_id(v: Any) -> bool:
    if isinstance(v, int):
        return True
    if isinstance(v, str) and v.strip().isdigit():
        return True
    return False


async def load_strings_from_file(path: str, console: Console) -> list[str]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        strings: list[str] = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "errorDialog" in row:
                    strings.append(row["errorDialog"] or "")
                elif "jediTaskID" in row:
                    strings.append(row["jediTaskID"] or "")
                elif "taskID" in row:
                    strings.append(row["taskID"] or "")
        if strings and all(_looks_like_task_id(s) for s in strings[:5]):
            return await _fetch_from_task_ids(strings, console)
        return strings

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    first = raw[0]
    if _looks_like_task_id(first):
        return await _fetch_from_task_ids([str(v) for v in raw], console)
    if isinstance(first, dict):
        return [str(item.get("errorDialog") or "") for item in raw]
    return [str(item) for item in raw]


async def _fetch_from_task_ids(task_ids: list[str], console: Console) -> list[str]:
    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415
    console.print(f"[cyan]Fetching {len(task_ids)} task(s) from PanDA...[/cyan]")
    results: list[str] = []
    for tid in task_ids:
        try:
            data = await fetch_task_data(tid)
            results.append(str(data.get("errorDialog") or ""))
        except Exception as exc:
            console.print(f"[yellow]  task {tid}: fetch failed ({exc})[/yellow]")
            results.append("")
    return results


# ── identifier coverage ──────────────────────────────────────────────────────

_IDENTIFIER = re.compile(
    r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b"  # snake_case
    r"|\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+\b"       # camelCase
)


def _extract_identifiers(terms: list[str]) -> list[str]:
    ids: list[str] = []
    for t in terms:
        ids.extend(_IDENTIFIER.findall(t))
    return list(dict.fromkeys(ids))


def identifier_coverage(grep_terms: list[str], nav_result: str) -> float:
    ids = _extract_identifiers(grep_terms)
    if not ids:
        return 0.0
    result_lower = nav_result.lower()
    return sum(1 for ident in ids if ident.lower() in result_lower) / len(ids)


# ── flags ────────────────────────────────────────────────────────────────────

def compute_flags(
    grep_terms: list[str],
    term_extraction_succeeded: bool,
    rounds_used: int,
) -> dict[str, bool]:
    id_fraction = (
        len([t for t in grep_terms if _IDENTIFIER.search(t)]) / len(grep_terms)
        if grep_terms else 0.0
    )
    return {
        "term_fallback": not term_extraction_succeeded,
        "low_term_quality": id_fraction < 0.5,
        "rounds_exhausted": rounds_used >= PandaSourceNavigator.MAX_ROUNDS,
    }


# ── categorisation ───────────────────────────────────────────────────────────

_NO_CANDIDATES_PREFIXES = (
    "No methods found",
    "Neither pandaserver",
    "Found candidate methods but could not",
)


def categorize(
    error_dialog: str,
    nav_result: str | None,
    candidates_count: int,
    max_overlap: int,
    judge_verdict: str | None,
) -> str:
    if not error_dialog.strip():
        return "empty_error_dialog"
    if nav_result is None:
        return "error"
    if any(nav_result.startswith(p) for p in _NO_CANDIDATES_PREFIXES) or candidates_count == 0:
        return "no_candidates"
    if max_overlap <= 1 and candidates_count >= 15:
        return "too_many_candidates"
    if judge_verdict == "irrelevant":
        return "irrelevant"
    if judge_verdict == "unclear":
        return "unclear"
    return "relevant"


# ── LLM judge ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating whether a source-code analysis addresses the right code.

Error dialog (original error message):
{error_dialog}

Source-code analysis (what the navigator returned):
{nav_result}

Does this analysis address code that is plausibly responsible for the error?
Answer with exactly one of: relevant / irrelevant / unclear
Then one sentence of reasoning.

Format:
verdict: <relevant|irrelevant|unclear>
reason: <one sentence>"""


async def judge_relevance(
    error_dialog: str, nav_result: str, llm
) -> tuple[str, str]:
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    prompt = _JUDGE_PROMPT.format(
        error_dialog=error_dialog[:600],
        nav_result=nav_result[:1200],
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.content.strip()
        verdict = "unclear"
        reason = text
        for line in text.splitlines():
            low = line.lower()
            if low.startswith("verdict:"):
                v = low.split(":", 1)[1].strip()
                if v in ("relevant", "irrelevant", "unclear"):
                    verdict = v
            elif low.startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
        return verdict, reason
    except Exception as exc:
        return "unclear", f"judge failed: {exc}"


# ── terms-only evaluation ────────────────────────────────────────────────────

async def eval_one_terms_only(
    error_dialog: str, pkg_roots: dict, llm
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    t0 = time.monotonic()

    resp = await llm.ainvoke(
        [HumanMessage(content=SOURCE_GREP_TERMS_PROMPT.format(question=error_dialog))]
    )
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```")).strip()
    try:
        grep_terms: list[str] = list(
            dict.fromkeys(t for t in json.loads(raw) if isinstance(t, str) and t)
        )
        term_ok = True
    except Exception:
        grep_terms = [w for w in error_dialog.split() if len(w) >= 6]
        term_ok = False

    loop = asyncio.get_event_loop()
    total_candidates = 0
    term_stats: dict[str, Any] = {}
    for fragment in grep_terms:
        cands, stats, term_used = await loop.run_in_executor(
            None, _grep_sliding_window, pkg_roots, fragment
        )
        total_candidates += len(cands)
        term_stats[fragment] = {**stats, "term_used": term_used, "candidates": len(cands)}

    flags = compute_flags(grep_terms, term_ok, 0)
    nav_result = "" if total_candidates > 0 else "No methods found"
    cat = categorize(error_dialog, nav_result, total_candidates, 1, None)

    return {
        "grep_terms": grep_terms,
        "term_stats": term_stats,
        "candidates_count": total_candidates,
        "max_overlap": 0,
        "nav_result": None,
        "identifier_coverage": 0.0,
        "judge_verdict": None,
        "judge_reasoning": None,
        "category": cat,
        "flags": flags,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "error": None,
    }


# ── full evaluation ──────────────────────────────────────────────────────────

async def eval_one_full(
    error_dialog: str,
    use_judge: bool,
    judge_all: bool,
    llm,
) -> dict[str, Any]:
    t0 = time.monotonic()
    nav = PandaSourceNavigator()
    nav_result: str | None = None
    exc_msg: str | None = None

    try:
        nav_result = await nav.navigate(error_dialog)
    except Exception as exc:
        exc_msg = str(exc)

    grep_terms = nav.last_grep_terms
    candidates_count = nav.last_candidates_count
    max_overlap = nav.last_max_overlap

    coverage = identifier_coverage(grep_terms, nav_result or "") if nav_result else 0.0
    flags = compute_flags(grep_terms, nav.last_term_extraction_succeeded, nav.last_rounds_used)

    suspicious = (
        coverage < 0.3
        and nav_result is not None
        and not any(nav_result.startswith(p) for p in _NO_CANDIDATES_PREFIXES)
    )
    judge_verdict: str | None = None
    judge_reasoning: str | None = None
    if nav_result and (judge_all or (use_judge and suspicious)):
        judge_verdict, judge_reasoning = await judge_relevance(error_dialog, nav_result, llm)

    cat = categorize(
        error_dialog,
        nav_result if exc_msg is None else None,
        candidates_count,
        max_overlap,
        judge_verdict,
    )

    return {
        "grep_terms": grep_terms,
        "term_stats": {},
        "candidates_count": candidates_count,
        "max_overlap": max_overlap,
        "nav_result": nav_result,
        "identifier_coverage": round(coverage, 3),
        "judge_verdict": judge_verdict,
        "judge_reasoning": judge_reasoning,
        "category": cat,
        "flags": flags,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "error": exc_msg,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

_CATEGORY_STYLE: dict[str, str] = {
    "relevant": "green",
    "irrelevant": "yellow",
    "unclear": "yellow",
    "no_candidates": "red",
    "too_many_candidates": "red",
    "empty_error_dialog": "dim",
    "error": "bold red",
}

_ALL_CATEGORIES = (
    "relevant",
    "irrelevant",
    "unclear",
    "no_candidates",
    "too_many_candidates",
    "empty_error_dialog",
    "error",
)


@click.command()
@click.argument("query", required=False)
@click.option("--from-file", "from_file", default=None, metavar="FILE",
              help="JSON or CSV file with errorDialog strings or task IDs.")
@click.option("--terms-only", is_flag=True, default=False,
              help="LLM term extraction + grep only; skip navigation and synthesis.")
@click.option("--judge", "use_judge", is_flag=True, default=False,
              help="Run LLM-as-judge on suspicious results (identifier coverage < 0.3).")
@click.option("--judge-all", is_flag=True, default=False,
              help="Run LLM-as-judge on every result regardless of coverage.")
@click.option("--limit", default=0, metavar="N",
              help="Cap number of distinct clusters to evaluate (0 = unlimited).")
@click.option("--output", default=None, metavar="FILE",
              help="Write full JSON report to FILE.")
@click.option("--verbose", is_flag=True, default=False,
              help="Show navigator internals (grep terms, candidates, each round). "
                   "Useful when testing a single string.")
def main(
    query: str | None,
    from_file: str | None,
    terms_only: bool,
    use_judge: bool,
    judge_all: bool,
    limit: int,
    output: str | None,
    verbose: bool,
) -> None:
    """Batch-evaluate PandaSourceNavigator over a collection of errorDialog strings."""
    console = Console()
    set_narrator(console, verbose=verbose)
    asyncio.run(_run(console, query, from_file, terms_only, use_judge, judge_all, limit, output))


async def _run(
    console: Console,
    query: str | None,
    from_file: str | None,
    terms_only: bool,
    use_judge: bool,
    judge_all: bool,
    limit: int,
    output: str | None,
) -> None:
    from bamboo.llm import get_extraction_llm  # noqa: PLC0415

    pkg_roots = _get_pkg_roots()
    if not pkg_roots:
        console.print("[red]Neither pandaserver nor pandajedi is installed.[/red]")
        return

    if from_file:
        raw_strings = await load_strings_from_file(from_file, console)
    elif query:
        raw_strings = [query]
    else:
        console.print("[red]Provide a QUERY argument or --from-file FILE.[/red]")
        return

    console.print(f"Loaded [bold]{len(raw_strings)}[/bold] raw string(s).")

    clusters = cluster_strings(raw_strings)
    console.print(
        f"Reduced to [bold]{len(clusters)}[/bold] distinct pattern(s) after normalisation."
    )

    if limit and limit < len(clusters):
        clusters = clusters[:limit]
        console.print(f"Capped to [bold]{limit}[/bold] cluster(s) via --limit.")

    llm = get_extraction_llm()
    results: list[dict[str, Any]] = []
    summary: dict[str, int] = {"raw_inputs": len(raw_strings), "distinct_patterns": len(clusters)}
    for cat in _ALL_CATEGORIES:
        summary[cat] = 0

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Evaluating...", total=len(clusters))
        for idx, cluster in enumerate(clusters):
            rep = cluster.representative
            if terms_only:
                r = await eval_one_terms_only(rep, pkg_roots, llm)
            else:
                r = await eval_one_full(rep, use_judge, judge_all, llm)
            summary[r["category"]] = summary.get(r["category"], 0) + 1
            results.append({
                "index": idx,
                "count": cluster.count,
                "normalized_pattern": cluster.key[:200],
                "representative": rep,
                **r,
            })
            progress.advance(task)

    _print_table(console, results)
    _print_summary(console, summary)

    if output:
        Path(output).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"\nReport written to [bold]{output}[/bold]")


def _print_table(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="Source Navigator Evaluation", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("N", width=4)
    table.add_column("errorDialog", width=58, no_wrap=True)
    table.add_column("category", width=22)
    table.add_column("terms", width=30, no_wrap=True)
    table.add_column("cands", width=6)
    table.add_column("s", width=6)

    for r in results:
        flags: dict[str, bool] = r.get("flags") or {}
        flag_str = "".join([
            "F" if flags.get("term_fallback") else "",
            "Q" if flags.get("low_term_quality") else "",
            "R" if flags.get("rounds_exhausted") else "",
        ])
        cat = r["category"]
        cat_display = f"{cat} [{flag_str}]" if flag_str else cat
        style = _CATEGORY_STYLE.get(cat, "")
        terms_display = ", ".join(r.get("grep_terms") or [])[:28]
        table.add_row(
            str(r["index"]),
            str(r["count"]),
            r["representative"][:56],
            cat_display,
            terms_display,
            str(r.get("candidates_count", "")),
            str(r.get("elapsed_s", "")),
            style=style,
        )

    console.print(table)


def _print_summary(console: Console, summary: dict[str, int]) -> None:
    console.rule("Summary")
    console.print(f"  Raw inputs:        {summary['raw_inputs']}")
    console.print(f"  Distinct patterns: {summary['distinct_patterns']}")
    for cat in _ALL_CATEGORIES:
        n = summary.get(cat, 0)
        if n:
            style = _CATEGORY_STYLE.get(cat, "")
            console.print(f"  [{style}]{cat}[/{style}]: {n}")


if __name__ == "__main__":
    main()
