"""Batch evaluation harness for PandaDocNavigator.

Feeds a collection of natural-language queries through the navigator and
reports result quality, source-strategy distribution (semantic vs LLM
traversal vs both), and optional LLM-as-judge verdicts.

Usage::

    # Single query smoke-test:
    python -m bamboo.scripts.panda.eval_doc_navigator "how to set memory limit for jobs"

    # Batch from a JSON file of query strings:
    python -m bamboo.scripts.panda.eval_doc_navigator --from-file queries.json --judge --output report.json

    # Verbose — print full result content per query:
    python -m bamboo.scripts.panda.eval_doc_navigator --verbose "exhausted task retry limit"
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from bamboo.agents.panda_doc_navigator import PandaDocNavigator
from bamboo.utils.narrator import set_narrator


# ── LLM judge ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating documentation search results.

Query: {query}

Top results returned:
{results_text}

Do these results directly answer the query, or at least provide clearly relevant context?
Answer with exactly one of: relevant / irrelevant / unclear
Then one sentence of reasoning.

Format:
verdict: <relevant|irrelevant|unclear>
reason: <one sentence>"""


async def judge_relevance(
    query: str, results_text: str, llm
) -> tuple[str, str]:
    from langchain_core.messages import HumanMessage  # noqa: PLC0415
    prompt = _JUDGE_PROMPT.format(query=query[:400], results_text=results_text[:2000])
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


def _results_text(results) -> str:
    parts = []
    for i, r in enumerate(results[:5], 1):
        snippet = r.content[:300].replace("\n", " ").strip()
        parts.append(f"{i}. [{r.title}] {snippet}")
    return "\n".join(parts)


# ── input loading ─────────────────────────────────────────────────────────────

async def load_queries_from_file(path: str) -> list[str]:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        queries: list[str] = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("query") or row.get("Query") or ""
                if q:
                    queries.append(q)
        return queries
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    return [str(item) for item in raw if item]


# ── categorisation ────────────────────────────────────────────────────────────

def categorize(
    query: str,
    result_count: int,
    nav_error: str | None,
    judge_verdict: str | None,
) -> str:
    if not query.strip():
        return "empty_query"
    if nav_error is not None:
        return "error"
    if result_count == 0:
        return "no_results"
    if judge_verdict == "irrelevant":
        return "irrelevant"
    if judge_verdict == "unclear":
        return "unclear"
    return "relevant"


# ── evaluation ────────────────────────────────────────────────────────────────

async def eval_one(
    query: str,
    navigator: PandaDocNavigator,
    top_k: int,
    use_judge: bool,
    llm,
) -> dict[str, Any]:
    t0 = time.monotonic()
    results = []
    exc_msg: str | None = None

    try:
        results = await navigator.search(query, top_k=top_k)
    except Exception as exc:
        exc_msg = str(exc)

    source_dist = {"semantic": 0, "llm_traversal": 0, "both": 0}
    top_titles: list[str] = []
    top_urls: list[str] = []
    top_score = 0.0

    for r in results:
        source_dist[r.source] = source_dist.get(r.source, 0) + 1
        if len(top_titles) < 3:
            top_titles.append(r.title)
            top_urls.append(r.url)
    if results:
        top_score = results[0].score

    judge_verdict: str | None = None
    judge_reasoning: str | None = None
    if use_judge and results:
        judge_verdict, judge_reasoning = await judge_relevance(
            query, _results_text(results), llm
        )

    cat = categorize(query, len(results), exc_msg, judge_verdict)

    return {
        "query": query,
        "result_count": len(results),
        "top_score": round(top_score, 3),
        "source_dist": source_dist,
        "top_titles": top_titles,
        "top_urls": top_urls,
        "judge_verdict": judge_verdict,
        "judge_reasoning": judge_reasoning,
        "category": cat,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "error": exc_msg,
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
@click.argument("query", required=False)
@click.option("--from-file", "from_file", default=None, metavar="FILE",
              help="JSON or CSV file with query strings.")
@click.option("--top-k", default=5, metavar="N",
              help="Number of results to request per query (default: 5).")
@click.option("--judge", "use_judge", is_flag=True, default=False,
              help="Run LLM-as-judge on every result set with at least one hit.")
@click.option("--limit", default=0, metavar="N",
              help="Cap number of queries to evaluate (0 = unlimited).")
@click.option("--output", default=None, metavar="FILE",
              help="Write full JSON report to FILE.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print full result content for each query.")
def main(
    query: str | None,
    from_file: str | None,
    top_k: int,
    use_judge: bool,
    limit: int,
    output: str | None,
    verbose: bool,
) -> None:
    """Batch-evaluate PandaDocNavigator over a collection of query strings."""
    console = Console()
    set_narrator(console, verbose=verbose)
    asyncio.run(_run(console, query, from_file, top_k, use_judge, limit, output, verbose))


async def _run(
    console: Console,
    query: str | None,
    from_file: str | None,
    top_k: int,
    use_judge: bool,
    limit: int,
    output: str | None,
    verbose: bool,
) -> None:
    from bamboo.llm import get_extraction_llm  # noqa: PLC0415

    if from_file:
        queries = await load_queries_from_file(from_file)
    elif query:
        queries = [query]
    else:
        console.print("[red]Provide a QUERY argument or --from-file FILE.[/red]")
        return

    console.print(f"Loaded [bold]{len(queries)}[/bold] query/queries.")

    if limit and limit < len(queries):
        queries = queries[:limit]
        console.print(f"Capped to [bold]{limit}[/bold] via --limit.")

    llm = get_extraction_llm() if use_judge else None
    navigator = PandaDocNavigator()

    results: list[dict[str, Any]] = []
    summary: dict[str, int] = {"total_queries": len(queries)}
    for cat in _ALL_CATEGORIES:
        summary[cat] = 0
    total_source: dict[str, int] = {"semantic": 0, "llm_traversal": 0, "both": 0}

    with Progress(console=console, transient=True) as progress:
        task = progress.add_task("Evaluating...", total=len(queries))
        for idx, q in enumerate(queries):
            r = await eval_one(q, navigator, top_k, use_judge, llm)
            summary[r["category"]] = summary.get(r["category"], 0) + 1
            for k in ("semantic", "llm_traversal", "both"):
                total_source[k] += r["source_dist"].get(k, 0)
            results.append({"index": idx, **r})
            if verbose and r["top_titles"]:
                console.print(f"\n[bold]{q}[/bold]")
                for title, url in zip(r["top_titles"], r["top_urls"]):
                    console.print(f"  • {title} — {url}")
            progress.advance(task)

    _print_table(console, results)
    _print_summary(console, summary, total_source)

    if output:
        Path(output).write_text(
            json.dumps(
                {"summary": summary, "source_totals": total_source, "results": results},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        console.print(f"\nReport written to [bold]{output}[/bold]")


def _print_table(console: Console, results: list[dict[str, Any]]) -> None:
    table = Table(title="Doc Navigator Evaluation", show_lines=False)
    table.add_column("#", style="dim", width=4)
    table.add_column("query", width=50, no_wrap=True)
    table.add_column("category", width=14)
    table.add_column("n", width=3)
    table.add_column("score", width=6)
    table.add_column("S/L/B", width=7)
    table.add_column("s", width=6)

    for r in results:
        sd = r.get("source_dist") or {}
        sources = f"{sd.get('semantic', 0)}/{sd.get('llm_traversal', 0)}/{sd.get('both', 0)}"
        cat = r["category"]
        style = _CATEGORY_STYLE.get(cat, "")
        table.add_row(
            str(r["index"]),
            r["query"][:48],
            cat,
            str(r.get("result_count", "")),
            str(r.get("top_score", "")),
            sources,
            str(r.get("elapsed_s", "")),
            style=style,
        )

    console.print(table)


def _print_summary(
    console: Console,
    summary: dict[str, int],
    total_source: dict[str, int],
) -> None:
    console.rule("Summary")
    console.print(f"  Total queries: {summary['total_queries']}")
    for cat in _ALL_CATEGORIES:
        n = summary.get(cat, 0)
        if n:
            style = _CATEGORY_STYLE.get(cat, "")
            console.print(f"  [{style}]{cat}[/{style}]: {n}")

    total_results = sum(total_source.values())
    if total_results:
        console.rule("Source strategy breakdown")
        for key, label in (("semantic", "semantic only"), ("llm_traversal", "llm traversal"), ("both", "both")):
            n = total_source.get(key, 0)
            pct = round(100 * n / total_results)
            console.print(f"  {label}: {n} ({pct}%)")


if __name__ == "__main__":
    main()
