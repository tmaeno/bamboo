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
    help="Path to task data JSON file",
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    help="Path to external data JSON file",
)
def main(email_thread, task_data, external_data):
    """Populate knowledge base from various sources."""
    setup_logging()

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

    # Run extraction
    asyncio.run(extract_knowledge(email_text, task_dict, external_dict))


async def extract_knowledge(email_text, task_dict, external_dict):
    """Extract and store knowledge."""
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
