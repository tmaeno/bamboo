"""Analyze many tasks in one process, amortizing service + model startup.

``bamboo analyze`` builds the dependency graph, connects Neo4j + Qdrant, and loads
the in-process embedding/reranker models on **every** invocation. In a batch
context that fixed cost dominates, so this command pays it **once**: it builds
deps a single time and loops every staged task through the same per-task pipeline
as ``bamboo analyze`` (via :func:`bamboo.scripts.analyze_task.analyze_one`),
keeping the services and in-process models warm across the whole batch.

Data flow
---------
- Input:  ``--input-dir`` (one ``*.json`` task-data file per task) and/or one or
          more ``--task-id`` (fetched live from PanDA when egress is allowed).
- Output: ``--output-dir`` — one ``<task>.json`` :class:`AnalysisResult` per task,
          plus a ``<task>.error.json`` sidecar for any task that fails.

A failing task is isolated (recorded, the batch continues); the process exits
non-zero if any task failed, so a scheduler can detect partial failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from bamboo.utils.logging import setup_logging

console = Console()
logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(value: Any) -> str:
    """Sanitise *value* into a safe output filename stem."""
    stem = _UNSAFE_FILENAME_RE.sub("_", str(value)).strip("_")
    return stem or "task"


def _load_task_files(input_dir: Path) -> list[tuple[str, dict]]:
    """Return ``(label, task_dict)`` for every readable ``*.json`` in *input_dir*."""
    tasks: list[tuple[str, dict]] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            tasks.append((path.stem, json.loads(path.read_text())))
        except Exception as exc:  # noqa: BLE001 — skip the bad file, keep the batch
            console.print(f"  [red]✗[/red] {path.name}: invalid JSON — {exc}")
    return tasks


async def _run(
    input_dir: str | None,
    task_ids: list[int],
    output_dir: str,
    external_data: str | None,
    drafts_dir: str,
    concurrency: int,
    verbose: bool,
) -> int:
    """Build deps once, analyze every task, write per-task results. Returns exit code."""
    setup_logging()
    from bamboo.utils.narrator import set_narrator  # noqa: PLC0415

    set_narrator(console, verbose=verbose)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    external_dict = json.loads(Path(external_data).read_text()) if external_data else None

    # Work list: file-backed tasks (task_dict known) + PanDA task-ids (resolved lazily).
    work: list[tuple[str, dict | None, int | None]] = []
    if input_dir:
        for label, task_dict in _load_task_files(Path(input_dir)):
            work.append((label, task_dict, None))
    for tid in task_ids:
        work.append((f"task_{tid}", None, tid))

    if not work:
        console.print("[yellow]No tasks found (empty --input-dir and no --task-id).[/yellow]")
        return 0

    console.print(f"Batch-analyzing [bold]{len(work)}[/bold] task(s) → {out}/")

    from bamboo.agents.helpers.deps import (  # noqa: PLC0415
        build_deps,
        resolve_task_data,
    )
    from bamboo.scripts.analyze_task import analyze_one  # noqa: PLC0415

    deps = build_deps()
    graph_db = deps.graph_db
    vector_db = deps.vector_db

    n_ok = n_novel = n_fail = 0
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _analyze(label: str, task_dict: dict | None, task_id: int | None) -> bool:
        nonlocal n_ok, n_novel, n_fail
        async with sem:
            try:
                if task_dict is None and task_id is not None:
                    task_dict = await resolve_task_data(task_id)

                result, _prescription, email_content, draft_path = await analyze_one(
                    deps, task_dict, external_dict, drafts_dir=drafts_dir
                )
                if email_content:
                    result.email_content = email_content

                rid = (task_dict or {}).get("jediTaskID", label)
                (out / f"{_safe_stem(rid)}.json").write_text(result.model_dump_json(indent=2))

                n_ok += 1
                if draft_path is not None:
                    n_novel += 1
                    console.print(f"  [yellow]◆[/yellow] {label}: novel incident → {draft_path}")
                else:
                    console.print(
                        f"  [green]✓[/green] {label}: {result.root_cause[:60]} "
                        f"({result.confidence:.0%})"
                    )
                return True
            except Exception as exc:  # noqa: BLE001 — isolate per-task failure
                n_fail += 1
                console.print(f"  [red]✗[/red] {label}: {exc}")
                logger.exception("batch-analyze failed for %s", label)
                try:
                    (out / f"{_safe_stem(label)}.error.json").write_text(
                        json.dumps(
                            {"label": label, "task_id": task_id, "error": str(exc)},
                            indent=2,
                        )
                    )
                except Exception:  # noqa: BLE001 — sidecar is best-effort
                    pass
                return False

    try:
        await graph_db.connect()
        await vector_db.connect()

        if concurrency > 1:
            await asyncio.gather(*(_analyze(*item) for item in work))
        else:
            for item in work:
                await _analyze(*item)
    finally:
        await graph_db.close()
        await vector_db.close()

    table = Table(title="batch-analyze summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Analyzed OK", str(n_ok))
    table.add_row("  of which novel incidents", str(n_novel))
    table.add_row("Failed", str(n_fail))
    console.print(table)

    if n_fail:
        console.print(f"\n[yellow]⚠  {n_fail} task(s) failed — see *.error.json in {out}/.[/yellow]")
        return 1
    console.print(f"\n[bold green]✓ All {n_ok} task(s) analyzed.[/bold green]")
    return 0


@click.command("batch-analyze")
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Directory of task-data JSON files (each *.json is one task). Combined with --task-id.",
)
@click.option(
    "--task-id",
    "task_ids",
    type=int,
    multiple=True,
    help="PanDA jediTaskID to fetch live (repeatable). Requires PanDA egress.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write one result JSON per task (created if missing).",
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    default=None,
    help="Optional external-data JSON applied to every task.",
)
@click.option(
    "--drafts-dir",
    default="drafts",
    show_default=True,
    type=click.Path(),
    help="Directory to write seed drafts for novel (unmatched) incidents.",
)
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=int,
    help="Max concurrent analyses. Ollama serializes inference, so 1 is a safe default.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(input_dir, task_ids, output_dir, external_data, drafts_dir, concurrency, verbose):
    """Analyze many tasks in one process, amortizing startup across the batch.

    Builds deps once (DB connections + in-process models stay warm) and runs the
    same per-task pipeline as ``bamboo analyze`` over every staged task. One result
    file is written per task; failures are isolated and the process exits non-zero
    if any task failed.

    Examples:

    \b
      bamboo batch-analyze --input-dir /in --output-dir /out
      bamboo batch-analyze --task-id 123 --task-id 456 --output-dir /out
      bamboo batch-analyze --input-dir tasks/ --output-dir out/ --concurrency 4
    """
    if not input_dir and not task_ids:
        raise click.UsageError("Provide --input-dir and/or one or more --task-id.")
    rc = asyncio.run(
        _run(
            input_dir=input_dir,
            task_ids=list(task_ids),
            output_dir=output_dir,
            external_data=external_data,
            drafts_dir=drafts_dir,
            concurrency=concurrency,
            verbose=verbose,
        )
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
