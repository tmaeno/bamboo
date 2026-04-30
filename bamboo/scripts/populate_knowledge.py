"""Script to populate knowledge base from sources."""

import asyncio
import json
import sys
import traceback
from pathlib import Path

import click

from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.utils.logging import setup_logging


@click.command()
@click.option(
    "--email-thread",
    type=click.Path(exists=True),
    help="Path to email thread text file",
)
@click.option(
    "--task-data",
    type=click.Path(exists=True),
    default=None,
    help="Path to task data JSON file. Mutually exclusive with --task-id.",
)
@click.option(
    "--task-id",
    type=int,
    default=None,
    help=(
        "PanDA jediTaskID — fetch task data directly from PanDA instead of a file. "
        "Requires PANDA_URL / PANDA_URL_SSL to be set (or uses the CERN defaults). "
        "Mutually exclusive with --task-data."
    ),
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    help="Path to external data JSON file",
)
@click.option(
    "--require-procedures",
    is_flag=True,
    default=False,
    help=(
        "Reject graphs that contain no Procedure nodes. "
        "Exits with code 1 if no investigation procedure was captured."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Extract and preview without writing to any database. "
        "No Neo4j or Qdrant connections are made."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save the extracted graph as JSON to this file path (implies --dry-run preview output).",
)
@click.option(
    "--max-retries",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Maximum reviewer-rejection retries (0 = no retry after first review). "
        "Defaults to 2. Use --max-retries 1 to debug a single "
        "extraction→review→explorer chain."
    ),
)
@click.option(
    "--debug-report",
    type=click.Path(),
    default=None,
    help=(
        "Save a JSON trace of every intermediate pipeline step to this file. "
        "Useful for investigating unexpected extraction or storage behaviour."
    ),
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(
    email_thread, task_data, task_id, external_data, require_procedures,
    dry_run, output, max_retries, debug_report, verbose,
):
    """Populate knowledge base from various sources.

    Task data can be supplied either as a local JSON file (--task-data) or
    fetched live from PanDA by jediTaskID (--task-id).  The two options are
    mutually exclusive.

    Use ``--dry-run`` to preview extraction without writing to any database
    (equivalent to the former ``bamboo extract`` command).

    Examples:

    \b
      bamboo populate --task-id 12345
      bamboo populate --task-id 12345 --dry-run
      bamboo populate --task-id 12345 --dry-run --output graph_preview.json
      bamboo populate --task-id 12345 --dry-run -v --max-retries 1
      bamboo populate --task-id 12345 --debug-report debug.json
    """
    setup_logging()

    if task_data and task_id:
        raise click.UsageError("--task-data and --task-id are mutually exclusive.")

    email_text = ""
    if email_thread:
        email_text = Path(email_thread).read_text()

    task_dict = None
    if task_data:
        task_dict = json.loads(Path(task_data).read_text())

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    asyncio.run(
        _run_populate(
            email_text, task_dict, task_id, external_dict,
            require_procedures=require_procedures,
            dry_run=dry_run,
            output=output,
            max_retries=max_retries,
            debug_report=debug_report,
            verbose=verbose,
        )
    )


async def _run_populate(
    email_text, task_dict, task_id, external_dict,
    require_procedures=False,
    dry_run=False,
    output=None,
    max_retries=None,
    debug_report=None,
    verbose=False,
):
    """Core populate / dry-run logic."""
    from rich.console import Console

    from bamboo.utils.narrator import set_narrator

    set_narrator(Console(), verbose=verbose)

    if task_id is not None:
        from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

        try:
            task_dict = await fetch_task_data(task_id)
        except Exception as e:
            click.echo(f"Error fetching task data from PanDA: {e}", err=True)
            sys.exit(1)

    from bamboo.agents.exploration_planner import ExplorationPlanner
    from bamboo.agents.extra_source_explorer import ExtraSourceExplorer
    from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
    from bamboo.config import get_settings
    from bamboo.mcp.factory import build_mcp_client

    settings = get_settings()
    reviewer = KnowledgeReviewer()
    _mcp = build_mcp_client(settings)
    explorer = ExtraSourceExplorer(_mcp, planner=ExplorationPlanner(_mcp))
    accumulator_kwargs = {}
    if max_retries is not None:
        accumulator_kwargs["max_review_retries"] = max_retries

    if dry_run:
        # No DB connections needed for extraction preview.
        neo4j = None
        qdrant = None
    else:
        neo4j = GraphDatabaseClient()
        qdrant = VectorDatabaseClient()

    debug_trace: dict | None = {} if debug_report else None

    try:
        if not dry_run:
            await neo4j.connect()
            await qdrant.connect()

        agent = KnowledgeAccumulator(
            neo4j, qdrant, reviewer=reviewer, explorer=explorer, **accumulator_kwargs
        )

        msg = "Extracting knowledge (dry-run — nothing will be written)..." if dry_run else "Extracting knowledge..."
        click.echo(msg)

        result = await agent.process_knowledge(
            email_text=email_text,
            task_data=task_dict,
            external_data=external_dict,
            dry_run=dry_run,
            require_procedures=require_procedures,
            debug_trace=debug_trace,
        )

        if dry_run or verbose:
            from bamboo.utils.display import print_extraction_result  # noqa: PLC0415
            print_extraction_result(result, verbose=verbose, dry_run=dry_run)
        else:
            click.echo("\n✓ Knowledge extracted and stored successfully!")
            click.echo(f"\nSummary:\n{result.summary}")
            click.echo(f"\nNodes: {len(result.graph.nodes)}")
            click.echo(f"Relationships: {len(result.graph.relationships)}")

        if require_procedures and not result.stored:
            click.echo(
                "\n⚠  No Procedure nodes — graph not stored (--require-procedures).",
                err=True,
            )
            sys.exit(1)

        if output:
            from bamboo.utils.display import save_graph_json  # noqa: PLC0415
            save_graph_json(result, output)

        if debug_report and debug_trace is not None:
            Path(debug_report).write_text(
                json.dumps(debug_trace, indent=2, default=str)
            )
            click.echo(f"\n✓ Debug report saved to {debug_report}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\n--- Traceback ---", err=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not dry_run:
            if neo4j is not None:
                await neo4j.close()
            if qdrant is not None:
                await qdrant.close()


if __name__ == "__main__":
    main()
