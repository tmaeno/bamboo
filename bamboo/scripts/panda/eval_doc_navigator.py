"""Batch evaluation harness for PanDA doc context-prefetch.

Feeds a collection of task IDs through the production ``prefetch_panda_docs``
pipeline and reports result quality (section count, query generated, LLM-judge
verdict).

Usage::

    # Single task ID smoke-test:
    python -m bamboo.scripts.panda.eval_doc_navigator 28026850

    # Batch from a JSON array of task IDs:
    python -m bamboo.scripts.panda.eval_doc_navigator --from-file task_ids.json --judge --output report.json

    # Batch from a CSV with a jediTaskID or taskID column:
    python -m bamboo.scripts.panda.eval_doc_navigator --from-file tasks.csv

    # Verbose — print retrieved hint text per task:
    python -m bamboo.scripts.panda.eval_doc_navigator 28026850 --verbose
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from bamboo.utils.narrator import counting, set_narrator

logger = logging.getLogger(__name__)


# ── LLM judge ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating documentation search results.

Query / error context: {query}

Top doc sections returned:
{results_text}

Do these sections directly answer the query, or at least provide clearly relevant context?
Answer with exactly one of: relevant / irrelevant / unclear
Then one sentence of reasoning.

Format:
verdict: <relevant|irrelevant|unclear>
reason: <one sentence>"""


async def judge_relevance(
    error_dialog: str, hints_text: str, llm
) -> tuple[str, str]:
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    prompt = _JUDGE_PROMPT.format(
        query=error_dialog[:400], results_text=hints_text[:2000]
    )
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.content.strip()
        verdict, reason = "unclear", text
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


# ── task ID helpers ───────────────────────────────────────────────────────────

def _looks_like_task_id(v: Any) -> bool:
    if isinstance(v, int):
        return True
    if isinstance(v, str) and v.strip().isdigit():
        return True
    return False


async def load_task_ids_from_file(path: str) -> list[str]:
    """Load task IDs from a JSON array or CSV file."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        ids: list[str] = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = (
                    row.get("jediTaskID")
                    or row.get("taskID")
                    or row.get("task_id")
                    or ""
                )
                if tid and _looks_like_task_id(tid):
                    ids.append(tid.strip())
        return ids
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    if not all(_looks_like_task_id(v) for v in raw[:5]):
        raise ValueError(f"Expected a JSON array of task IDs (integers) in {path}")
    return [str(v) for v in raw]


# ── categorisation ────────────────────────────────────────────────────────────

def categorize(
    has_error_dialog: bool,
    result_count: int,
    nav_error: str | None,
    judge_verdict: str | None,
) -> str:
    if nav_error is not None:
        return "error"
    if not has_error_dialog:
        return "empty_query"
    if result_count == 0:
        return "no_results"
    if judge_verdict == "irrelevant":
        return "irrelevant"
    if judge_verdict == "unclear":
        return "unclear"
    return "relevant"


# ── evaluation ────────────────────────────────────────────────────────────────

async def eval_one(
    task_id: str,
    use_judge: bool,
    llm,
) -> dict[str, Any]:
    from bamboo.agents.context_prefetch import prefetch_panda_docs  # noqa: PLC0415
    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

    exc_msg: str | None = None
    task_data: dict[str, Any] = {}
    doc_hints: dict[str, str] = {}
    meta: dict[str, str] = {}

    try:
        task_data = await fetch_task_data(task_id)
    except Exception as exc:
        exc_msg = f"fetch failed: {exc}"

    t0 = time.monotonic()
    if exc_msg is None:
        try:
            doc_hints, meta = await prefetch_panda_docs(task_data)
        except Exception as exc:
            exc_msg = str(exc)

    error_dialog = (task_data.get("errorDialog") or "").strip()
    status = task_data.get("status", "")

    nl_query = meta.get("nl_query", "")
    keyword_query: str | None = meta.get("keyword_query") or None

    # Re-fetch raw results using the exact same queries prefetch used,
    # so we can count hits and break down by source strategy.
    source_dist: dict[str, int] = {"semantic": 0, "llm_traversal": 0, "bm25": 0, "multi": 0}
    result_count = 0
    if nl_query and exc_msg is None:
        try:
            from bamboo.mcp.panda_mcp_client import PandaMcpClient  # noqa: PLC0415
            raw = await PandaMcpClient().execute(
                "search_panda_docs", query=nl_query, keyword_query=keyword_query
            ) or []
            result_count = len(raw)
            for r in raw:
                src = r.get("source", "semantic")
                source_dist[src] = source_dist.get(src, 0) + 1
        except Exception as exc:
            logger.warning("eval_one: search_panda_docs failed: %s", exc)

    top_query = nl_query[:42] if nl_query else ""

    judge_verdict: str | None = None
    judge_reasoning: str | None = None
    if use_judge and doc_hints:
        combined = "\n\n".join(
            f"[Query: {q}]\n{text}" for q, text in list(doc_hints.items())[:3]
        )
        judge_verdict, judge_reasoning = await judge_relevance(
            error_dialog, combined, llm
        )

    cat = categorize(bool(error_dialog), result_count, exc_msg, judge_verdict)

    return {
        "task_id": task_id,
        "status": status,
        "error_dialog": error_dialog[:120],
        "nl_query": nl_query,
        "keyword_query": keyword_query or "",
        "top_query": top_query,
        "result_count": result_count,
        "source_dist": source_dist,
        "judge_verdict": judge_verdict,
        "judge_reasoning": judge_reasoning,
        "category": cat,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "error": exc_msg,
        "doc_hints": doc_hints,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

_CATEGORY_STYLE: dict[str, str] = {
    "relevant": "green",
    "irrelevant": "yellow",
    "unclear": "yellow",
    "no_results": "red",
    "empty_query": "dim",
    "error": "bold red",
}

_ALL_CATEGORIES = (
    "relevant",
    "irrelevant",
    "unclear",
    "no_results",
    "empty_query",
    "error",
)


@click.command()
@click.argument("task_id", required=False)
@click.option("--from-file", "from_file", default=None, metavar="FILE",
              help="JSON array or CSV file with task IDs.")
@click.option("--judge", "use_judge", is_flag=True, default=False,
              help="Run LLM-as-judge on every result set with at least one hit.")
@click.option("--limit", default=0, metavar="N",
              help="Cap number of tasks to evaluate (0 = unlimited).")
@click.option("--output", default=None, metavar="FILE",
              help="Write full JSON report to FILE.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print retrieved hint text for each task.")
@click.option("--rebuild-docs", is_flag=True, default=False,
              help="Force a full rebuild of the doc index (clears cached metadata).")
def main(
    task_id: str | None,
    from_file: str | None,
    use_judge: bool,
    limit: int,
    output: str | None,
    verbose: bool,
    rebuild_docs: bool,
) -> None:
    """Evaluate PanDA doc context-prefetch over a collection of task IDs."""
    console = Console()
    set_narrator(console, verbose=verbose)

    if rebuild_docs:
        from bamboo.agents.panda_doc_navigator import invalidate_doc_cache  # noqa: PLC0415
        deleted = invalidate_doc_cache()
        click.echo(
            "✓ Doc index cache cleared — will rebuild on next use." if deleted
            else "Doc index cache was already empty."
        )

    asyncio.run(_run(console, task_id, from_file, use_judge, limit, output, verbose))
    os._exit(0)


async def _run(
    console: Console,
    task_id: str | None,
    from_file: str | None,
    use_judge: bool,
    limit: int,
    output: str | None,
    verbose: bool,
) -> None:
    from bamboo.llm import get_extraction_llm  # noqa: PLC0415

    if from_file:
        task_ids = await load_task_ids_from_file(from_file)
    elif task_id:
        if not _looks_like_task_id(task_id):
            console.print(f"[red]'{task_id}' does not look like a task ID (expected integer).[/red]")
            return
        task_ids = [task_id]
    else:
        console.print("[red]Provide a TASK_ID argument or --from-file FILE.[/red]")
        return

    console.print(f"Loaded [bold]{len(task_ids)}[/bold] task ID(s).")

    if limit and limit < len(task_ids):
        task_ids = task_ids[:limit]
        console.print(f"Capped to [bold]{limit}[/bold] via --limit.")

    llm = get_extraction_llm()

    results: list[dict[str, Any]] = []
    summary: dict[str, int] = {"total_tasks": len(task_ids)}
    for cat in _ALL_CATEGORIES:
        summary[cat] = 0

    with counting(f"Evaluating {len(task_ids)} task(s)", total=len(task_ids)) as advance:
        for idx, tid in enumerate(task_ids):
            r = await eval_one(tid, use_judge, llm)
            summary[r["category"]] = summary.get(r["category"], 0) + 1
            results.append({"index": idx, **r})
            if verbose and r["doc_hints"]:
                for q, text in r["doc_hints"].items():
                    console.print(f"\n[bold]Query:[/bold] {q}")
                    for line in text.splitlines()[:8]:
                        console.print(f"  {line}")
            advance()

    _print_table(console, results)
    _print_summary(console, summary)

    if output:
        serializable = [
            {k: v for k, v in r.items() if k != "doc_hints"} for r in results
        ]
        Path(output).write_text(
            json.dumps(
                {"summary": summary, "results": serializable},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        console.print(f"\nReport written to [bold]{output}[/bold]")


def _format_source_dist(sd: dict[str, int]) -> str:
    parts = []
    if sd.get("semantic"):
        parts.append(f"S:{sd['semantic']}")
    if sd.get("llm_traversal"):
        parts.append(f"L:{sd['llm_traversal']}")
    if sd.get("bm25"):
        parts.append(f"B:{sd['bm25']}")
    if sd.get("multi"):
        parts.append(f"M:{sd['multi']}")
    return " ".join(parts) if parts else "-"


def _print_table(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="Doc Context-Prefetch Evaluation", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("task_id", width=12)
    table.add_column("status", width=12)
    table.add_column("n", width=3)
    table.add_column("S/L/B/M", width=14, no_wrap=True)
    table.add_column("query", width=42, no_wrap=True)
    table.add_column("category", width=14)
    table.add_column("s", width=6)

    for r in results:
        cat = r["category"]
        style = _CATEGORY_STYLE.get(cat, "")
        table.add_row(
            str(r["index"]),
            str(r["task_id"]),
            r.get("status", ""),
            str(r.get("result_count", "")),
            _format_source_dist(r.get("source_dist", {})),
            r.get("top_query", "")[:42],
            cat,
            str(r.get("elapsed_s", "")),
            style=style,
        )

    console.print(table)


def _print_summary(console: Console, summary: dict[str, int]) -> None:
    console.rule("Summary")
    console.print(f"  Total tasks: {summary['total_tasks']}")
    for cat in _ALL_CATEGORIES:
        n = summary.get(cat, 0)
        if n:
            style = _CATEGORY_STYLE.get(cat, "")
            console.print(f"  [{style}]{cat}[/{style}]: {n}")


if __name__ == "__main__":
    main()
