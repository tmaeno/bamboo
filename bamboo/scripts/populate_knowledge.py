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
def main(email_thread, task_data, task_id, external_data):
    """Populate knowledge base from various sources.

    Task data can be supplied either as a local JSON file (--task-data) or
    fetched live from PanDA by jediTaskID (--task-id).  The two options are
    mutually exclusive.
    """
    setup_logging()

    if task_data and task_id:
        raise click.UsageError("--task-data and --task-id are mutually exclusive.")

    # Load data
    email_text = ""
    if email_thread:
        email_text = Path(email_thread).read_text()

    task_dict = None
    if task_data:
        task_dict = json.loads(Path(task_data).read_text())

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    # Run extraction (task_id is resolved inside the async function when provided)
    asyncio.run(extract_knowledge(email_text, task_dict, external_dict, task_id))


async def extract_knowledge(email_text, task_dict, external_dict, task_id=None):
    """Extract and store knowledge."""
    # Fetch task data from PanDA if a task_id was given instead of a file.
    if task_id is not None:
        from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

        click.echo(f"Fetching task data from PanDA for task_id={task_id}...")
        try:
            task_dict = await fetch_task_data(task_id)
        except Exception as e:
            click.echo(f"Error fetching task data from PanDA: {e}", err=True)
            sys.exit(1)

    neo4j = GraphDatabaseClient()
    qdrant = VectorDatabaseClient()

    try:
        await neo4j.connect()
        await qdrant.connect()

        agent = KnowledgeAccumulator(neo4j, qdrant)

        click.echo("Extracting knowledge...")
        result = await agent.process_knowledge(
            email_text=email_text,
            task_data=task_dict,
            external_data=external_dict,
        )

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
