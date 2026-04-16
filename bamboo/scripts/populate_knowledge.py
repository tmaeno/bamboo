"""Script to populate knowledge base from sources."""

import asyncio
import json
import sys
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
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(email_thread, task_data, task_id, external_data, require_procedures, verbose):
    """Populate knowledge base from various sources.

    Task data can be supplied either as a local JSON file (--task-data) or
    fetched live from PanDA by jediTaskID (--task-id).  The two options are
    mutually exclusive.
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

    asyncio.run(_extract_knowledge(email_text, task_dict, task_id, external_dict, require_procedures, verbose))


async def _extract_knowledge(email_text, task_dict, task_id, external_dict, require_procedures=False, verbose=False):
    """Extract and store knowledge."""
    from rich.console import Console

    from bamboo.utils.narrator import set_narrator, thinking

    set_narrator(Console(), verbose=verbose)

    if task_id is not None:
        from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

        try:
            task_dict = await fetch_task_data(task_id)
        except Exception as e:
            click.echo(f"Error fetching task data from PanDA: {e}", err=True)
            sys.exit(1)

    from bamboo.agents.extra_source_explorer import ExtraSourceExplorer
    from bamboo.agents.exploration_planner import ExplorationPlanner
    from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
    from bamboo.config import get_settings
    from bamboo.mcp.factory import build_mcp_client

    settings = get_settings()
    reviewer = KnowledgeReviewer()
    _mcp = build_mcp_client(settings)
    explorer = ExtraSourceExplorer(_mcp, planner=ExplorationPlanner(_mcp))

    neo4j = GraphDatabaseClient()
    qdrant = VectorDatabaseClient()

    try:
        await neo4j.connect()
        await qdrant.connect()

        agent = KnowledgeAccumulator(neo4j, qdrant, reviewer=reviewer, explorer=explorer)

        click.echo("Extracting knowledge...")
        result = await agent.process_knowledge(
            email_text=email_text,
            task_data=task_dict,
            external_data=external_dict,
            require_procedures=require_procedures,
        )

        if require_procedures and not result.stored:
            click.echo(
                "\n⚠  No Procedure nodes — graph not stored (--require-procedures).",
                err=True,
            )
            sys.exit(1)

        click.echo("\n✓ Knowledge extracted successfully!")
        click.echo("\nSummary:")
        click.echo(result.summary)
        click.echo(f"\nNodes: {len(result.graph.nodes)}")
        click.echo(f"Relationships: {len(result.graph.relationships)}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        await neo4j.close()
        await qdrant.close()


if __name__ == "__main__":
    main()
