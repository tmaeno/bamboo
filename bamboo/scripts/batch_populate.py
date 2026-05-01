"""Batch-populate knowledge databases from reviewed draft files.

Reads every ``reviewed: true`` JSON file in the drafts directory, calls
``process_knowledge()`` for each pending task_id, and archives newly reviewed
drafts to the approved email library.

Data flow
---------
- Input:  ``drafts/`` (snapshot-only — no live PanDA fetch here)
- Output: Neo4j + Qdrant (via ``process_knowledge``),
          ``approved_email_drafts/`` (archive of reviewed drafts)

Archival rules
--------------
- ``matched_from: null``        → always archive (new knowledge)
- Pre-filled, email unchanged   → skip (source already in library)
- Pre-filled, email changed     → archive as a new entry (never overwrite source)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from bamboo.utils.logging import setup_logging

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _email_body_to_text(email_body: dict[str, Any]) -> str:
    """Serialise a structured email_body dict to plain text for process_knowledge."""
    parts: list[str] = []

    if email_body.get("background"):
        parts.append(f"Background:\n{email_body['background']}")

    if email_body.get("cause"):
        parts.append(f"Cause:\n{email_body['cause']}")

    if email_body.get("resolution"):
        parts.append(f"Resolution:\n{email_body['resolution']}")

    procedure = email_body.get("procedure") or []
    if procedure:
        steps = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(procedure))
        parts.append(f"Investigation procedure:\n{steps}")

    return "\n\n".join(parts)


def _emails_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if two email_body dicts have identical content."""
    for key in ("background", "cause", "resolution"):
        if a.get(key, "") != b.get(key, ""):
            return False
    if list(a.get("procedure") or []) != list(b.get("procedure") or []):
        return False
    return True


async def _compute_embedding(raw_error: str) -> list[float]:
    from bamboo.llm import get_embeddings

    model = get_embeddings()
    return await model.aembed_query(raw_error or "(empty)")


async def _archive_draft(
    draft: dict[str, Any],
    draft_path: Path,
    approved_dir: Path,
) -> bool:
    """Archive *draft* to *approved_dir* if applicable.  Returns True if archived."""
    matched_from = draft.get("matched_from")
    email_body = draft.get("email_body", {})

    if matched_from is None:
        # New draft — always archive
        pass
    else:
        # Pre-filled draft — archive only if email_body was changed vs. source
        try:
            source = json.loads(Path(matched_from).read_text())
            if _emails_equal(email_body, source.get("email_body", {})):
                logger.debug("pre-filled draft %s unchanged — skipping archival", draft_path.name)
                return False
        except Exception as exc:
            logger.warning("could not read matched_from %s: %s — archiving as new entry", matched_from, exc)

    task_data = draft.get("task_data", {})
    rep_id = task_data.get("jediTaskID", draft_path.stem.replace("task_", ""))
    out_file = approved_dir / f"task_{rep_id}.json"

    # Avoid clobbering an existing approved entry with a different task's data
    if out_file.exists():
        # Use a timestamped suffix to avoid collision
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_file = approved_dir / f"task_{rep_id}_{ts}.json"

    raw_error = task_data.get("errorDialog", "")
    embedding = await _compute_embedding(raw_error)

    approved_entry = {
        **draft,
        "reviewed": True,
        "errorDialog_embedding": embedding,
    }
    out_file.write_text(json.dumps(approved_entry, indent=2))
    logger.info("archived %s → %s", draft_path.name, out_file.name)
    return True


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------


async def _run(
    drafts_dir: str,
    save_to: str,
    dry_run: bool,
    yes: bool,
    concurrency: int,
    verbose: bool,
) -> None:
    setup_logging()
    from bamboo.utils.narrator import set_narrator

    set_narrator(console, verbose=verbose)

    drafts = Path(drafts_dir)
    approved_dir = Path(save_to)
    approved_dir.mkdir(parents=True, exist_ok=True)

    draft_files = sorted(drafts.glob("*.json"))
    if not draft_files:
        console.print(f"[yellow]No JSON files found in {drafts_dir}/[/yellow]")
        return

    reviewed = [
        f for f in draft_files
        if json.loads(f.read_text()).get("reviewed", False)
    ]
    if not reviewed:
        console.print(
            f"[yellow]Found {len(draft_files)} draft(s) but none are marked "
            f"[bold]\"reviewed\": true[/bold] — nothing to populate.[/yellow]"
        )
        console.print("Set [bold]\"reviewed\": true[/bold] in each file you have verified.")
        return

    # Count total pending task_ids
    total_tasks = sum(
        len(json.loads(f.read_text()).get("task_ids", []))
        for f in reviewed
    )
    console.print(
        f"Found [bold]{len(reviewed)}[/bold] reviewed draft(s) covering "
        f"[bold]{total_tasks}[/bold] pending task(s)."
    )

    if total_tasks == 0:
        console.print("[green]All tasks already populated — nothing to do.[/green]")
        return

    if dry_run:
        console.print("\n[bold yellow]Dry-run mode — no databases will be modified.[/bold yellow]")
    elif not yes:
        import click as _click
        if not _click.confirm(f"\nPopulate {total_tasks} task(s) into Neo4j + Qdrant?", default=True):
            console.print("[dim]Aborted.[/dim]")
            return

    # Set up KnowledgeAccumulator (shared across all tasks)
    from bamboo.agents.exploration_planner import ExplorationPlanner
    from bamboo.agents.context_enricher import ContextEnricher
    from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
    from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
    from bamboo.config import get_settings
    from bamboo.database.graph_database_client import GraphDatabaseClient
    from bamboo.database.vector_database_client import VectorDatabaseClient
    from bamboo.mcp.factory import build_mcp_client

    settings = get_settings()
    reviewer = KnowledgeReviewer()
    _mcp = build_mcp_client(settings)
    explorer = ContextEnricher(_mcp, planner=ExplorationPlanner(_mcp))

    neo4j = GraphDatabaseClient()
    qdrant = VectorDatabaseClient()

    n_populated = 0
    n_failed = 0
    n_archived = 0
    n_skipped_archive = 0

    sem = asyncio.Semaphore(concurrency)

    try:
        await neo4j.connect()
        await qdrant.connect()

        agent = KnowledgeAccumulator(neo4j, qdrant, reviewer=reviewer, explorer=explorer)

        for draft_file in reviewed:
            draft = json.loads(draft_file.read_text())
            task_ids: list[int] = list(draft.get("task_ids", []))
            if not task_ids:
                console.print(f"  [dim]{draft_file.name}: no pending task_ids — checking archival[/dim]")
            else:
                email_text = _email_body_to_text(draft.get("email_body", {}))
                base_task_data: dict[str, Any] = dict(draft.get("task_data", {}))

                for task_id in list(task_ids):
                    async def _populate_one(tid: int, base: dict[str, Any]) -> bool:
                        task_data = {**base, "jediTaskID": tid}
                        async with sem:
                            try:
                                if dry_run:
                                    console.print(
                                        f"  [dim][dry-run] would populate task {tid}[/dim]"
                                    )
                                    return True
                                result = await agent.process_knowledge(
                                    email_text=email_text,
                                    task_data=task_data,
                                )
                                console.print(
                                    f"  [green]✓[/green] task {tid}: "
                                    f"{len(result.graph.nodes)} nodes, "
                                    f"{len(result.graph.relationships)} relationships"
                                )
                                return True
                            except Exception as exc:
                                console.print(f"  [red]✗[/red] task {tid}: {exc}")
                                logger.exception("process_knowledge failed for task %s", tid)
                                return False

                    ok = await _populate_one(task_id, base_task_data)
                    if ok:
                        n_populated += 1
                        if not dry_run:
                            task_ids.remove(task_id)
                            draft["task_ids"] = task_ids
                            draft_file.write_text(json.dumps(draft, indent=2))
                    else:
                        n_failed += 1

            # Archive phase — only after all tasks in this draft are processed
            if not dry_run and not draft.get("task_ids"):
                archived = await _archive_draft(draft, draft_file, approved_dir)
                if archived:
                    n_archived += 1
                else:
                    n_skipped_archive += 1

    except Exception as exc:
        console.print(f"[red]Fatal error: {exc}[/red]")
        logger.exception("batch_populate failed")
        sys.exit(1)
    finally:
        await neo4j.close()
        await qdrant.close()

    # Summary table
    table = Table(title="batch-populate summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Tasks populated", str(n_populated))
    table.add_row("Tasks failed (still pending)", str(n_failed))
    table.add_row("Drafts archived to approved library", str(n_archived))
    table.add_row("Drafts skipped archival (pre-filled, unchanged)", str(n_skipped_archive))
    console.print(table)

    if dry_run:
        console.print("\n[bold yellow]Dry-run complete — no data was written.[/bold yellow]")
    elif n_failed:
        console.print(
            f"\n[yellow]⚠  {n_failed} task(s) failed. "
            f"They remain in task_ids for the next run.[/yellow]"
        )
    else:
        console.print("\n[bold green]✓ All tasks populated successfully.[/bold green]")


# ---------------------------------------------------------------------------
# Click entry point
# ---------------------------------------------------------------------------


@click.command("batch-populate")
@click.option(
    "--drafts",
    "drafts_dir",
    default="drafts",
    show_default=True,
    type=click.Path(),
    help="Directory containing reviewed draft JSON files.",
)
@click.option(
    "--save-to",
    default="approved_email_drafts",
    show_default=True,
    type=click.Path(),
    help="Directory to archive reviewed drafts to (approved email library).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview what would be populated without writing to any database.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Maximum concurrent process_knowledge calls.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(drafts_dir, save_to, dry_run, yes, concurrency, verbose):
    """Batch-populate knowledge databases from reviewed draft files.

    Reads all ``reviewed: true`` JSON files in the drafts directory and calls
    ``bamboo populate`` logic for each pending task_id.  Successfully populated
    tasks are removed from the draft's ``task_ids`` list.  Newly reviewed drafts
    are archived to the approved email library for future use.

    Examples:

    \b
      bamboo batch-populate
      bamboo batch-populate --drafts my_drafts/ --save-to approved/
      bamboo batch-populate --dry-run
      bamboo batch-populate --yes --concurrency 5
    """
    asyncio.run(
        _run(
            drafts_dir=drafts_dir,
            save_to=save_to,
            dry_run=dry_run,
            yes=yes,
            concurrency=concurrency,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
