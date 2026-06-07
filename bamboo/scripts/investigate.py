"""`bamboo investigate` — co-investigation mode CLI entry point.

Builds the collaborator wiring (PandaMcpClient, GraphDatabaseClient,
VectorDatabaseClient, KnowledgeAccumulator, ReasoningNavigator, the
ErrorCategoryClassifier) and hands them to
:class:`bamboo.agents.investigation_session.InvestigationOrchestrator` for
``start() → run() → finalize()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

import click

from bamboo.agents.investigation_session import InvestigationOrchestrator
from bamboo.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option("--task-id", type=int, default=None, help="PanDA jediTaskID (typical entry).")
@click.option(
    "--task-data",
    type=click.Path(exists=True),
    default=None,
    help="Path to a task_data JSON file (alternative to --task-id).",
)
@click.option(
    "--symptom",
    type=str,
    default=None,
    help="Free-text symptom for non-PanDA scenarios (used when no task-id/task-data is available).",
)
@click.option(
    "--save",
    type=click.Path(),
    default=None,
    help=(
        "Where to checkpoint session state after each turn. "
        "Default: ~/.bamboo/investigations/<session-id>.json"
    ),
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Resume a prior session from its --save JSON file.",
)
@click.option("--max-turns", type=int, default=30, help="Safety cap on dialog turns (default 30).")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Walk through the session but never commit at the end.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose log output.")
@click.option(
    "--allow-mutating-autorun",
    is_flag=True,
    default=False,
    help=(
        "Allow state-changing (read_only=False) procedures to also be durably "
        "auto-run in the interactive loop (default: read-only procedures only). "
        "Inert until a state-changing tool exists; the automatic analyze phase "
        "stays read-only regardless."
    ),
)
def main(
    task_id: int | None,
    task_data: str | None,
    symptom: str | None,
    save: str | None,
    resume: str | None,
    max_turns: int,
    dry_run: bool,
    verbose: bool,
    allow_mutating_autorun: bool,
) -> None:
    """Co-investigate a live incident with bamboo."""
    setup_logging()
    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    if task_id is None and task_data is None and symptom is None and resume is None:
        click.echo(
            "Need one of: --task-id, --task-data, --symptom, or --resume.",
            err=True,
        )
        sys.exit(2)

    task_data_dict = None
    if task_data:
        task_data_dict = json.loads(Path(task_data).read_text())

    save_path = Path(save) if save else None
    session_id_override = None

    if resume:
        # Rehydrate session id (and skip the start() call).
        session_blob = json.loads(Path(resume).read_text())
        session_id_override = session_blob.get("session_id")
        save_path = save_path or Path(resume)
    else:
        if save_path is None:
            sid = uuid.uuid4().hex[:12]
            save_path = Path.home() / ".bamboo" / "investigations" / f"{sid}.json"
            session_id_override = sid

    asyncio.run(
        _run(
            task_id=task_id,
            task_data=task_data_dict,
            symptom=symptom,
            save_path=save_path,
            resume_from=Path(resume) if resume else None,
            session_id=session_id_override,
            max_turns=max_turns,
            dry_run=dry_run,
            verbose=verbose,
            allow_mutating_autorun=allow_mutating_autorun,
        )
    )


async def _run(
    *,
    task_id: int | None,
    task_data: dict | None,
    symptom: str | None,
    save_path: Path,
    resume_from: Path | None,
    session_id: str | None,
    max_turns: int,
    dry_run: bool,
    verbose: bool = False,
    allow_mutating_autorun: bool = False,
) -> None:
    # Local imports keep module load fast for `bamboo --help`.
    from rich.console import Console

    from bamboo.agents.deps import build_deps
    from bamboo.frontends.cli import CliInteractionIO
    from bamboo.utils.narrator import set_narrator

    # Build the terminal IO up front so the shared factory can route interactive
    # MCP tools (e.g. request_human_input) through it, and the orchestrator reuses it.
    console = Console()
    # Wire the narrator on the SAME console so say()'s "→" lines render and the
    # thinking() spinner coordinates with the IO's prompts/panels — the setup every
    # other CLI command does (e.g. analyze). Without it investigate shows no narration.
    set_narrator(console, verbose=verbose)
    io = CliInteractionIO(console)
    deps = build_deps(io=io)
    deps.console = console

    # Connect the knowledge backends up front (best-effort), mirroring `analyze`,
    # so the session-start hypothesis (analyze_task) and the end-of-session commit
    # can query them. A missing DB degrades gracefully — it must not abort the
    # interactive session — so connect failures are warned, not fatal.
    for _db in (deps.graph_db, deps.vector_db):
        try:
            await _db.connect()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "investigate: %s connect failed (%s) — continuing degraded",
                type(_db).__name__,
                exc,
            )

    orch = InvestigationOrchestrator(
        deps=deps,
        session_id=session_id,
        max_turns=max_turns,
        save_path=save_path,
        dry_run=dry_run,
        allow_mutating_autorun=allow_mutating_autorun,
    )

    if resume_from is not None:
        from bamboo.agents.investigation_session import InvestigationSession

        blob = json.loads(resume_from.read_text())
        orch.session = InvestigationSession.model_validate(blob)
        click.echo(f"Resumed session {orch.session.session_id} at turn {orch.session.turn}.")
        # Re-build tool registry without re-running start().
        await orch._build_tool_registry()
    else:
        await orch.start(task_id=task_id, task_data=task_data, symptom=symptom)

    try:
        await orch.run()
    except KeyboardInterrupt:
        click.echo("\nInterrupted; session saved.")
    try:
        await orch.finalize()
    finally:
        # Close the backends we opened (best-effort) so the async drivers don't
        # leak / warn on process exit.
        for _db in (deps.graph_db, deps.vector_db):
            try:
                await _db.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
