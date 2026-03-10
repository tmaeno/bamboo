"""Extract knowledge graph and preview it without writing to any database.

This script runs the full extraction pipeline in dry-run mode — LLM calls,
error classification, and graph construction all happen normally, but nothing
is written to Neo4j or Qdrant.  It is the complement to ``populate_knowledge``
and is exposed as ``bamboo extract``.

Typical workflow::

    # 1. Preview what would be stored:
    bamboo extract --task-id 12345

    # 2. Inspect / approve, then commit:
    bamboo populate --task-id 12345
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path

import click

from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
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
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save the extracted graph as JSON to this file path.",
)
def main(email_thread, task_data, task_id, external_data, output):
    """Extract knowledge graph and preview it without writing to any database.

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

    if task_id is not None:
        from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

        click.echo(f"Fetching task data from PanDA for task_id={task_id}...")
        try:
            task_dict = fetch_task_data(task_id)
        except Exception as e:
            click.echo(f"Error fetching task data from PanDA: {e}", err=True)
            sys.exit(1)

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    asyncio.run(_run_extraction(email_text, task_dict, external_dict, output))


async def _run_extraction(email_text, task_dict, external_dict, output=None):
    """Run extraction in dry-run mode and print the result."""
    # Dry-run needs no database connections at all — skip Neo4j and Qdrant.
    # Pass None so KnowledgeAccumulator skips all DB calls.
    agent = KnowledgeAccumulator(graph_db=None, vector_db=None)

    try:
        click.echo("Extracting knowledge (dry-run — nothing will be written)...")
        result = await agent.process_knowledge(
            email_text=email_text,
            task_data=task_dict,
            external_data=external_dict,
            dry_run=True,
        )

        # --- Summary ---
        click.echo("\n" + "=" * 70)
        click.echo("EXTRACTION PREVIEW")
        click.echo("=" * 70)
        click.echo(f"\nNodes:         {len(result.graph.nodes)}")
        click.echo(f"Relationships: {len(result.graph.relationships)}")

        # Node breakdown by type
        from collections import Counter

        type_counts = Counter(n.node_type.value for n in result.graph.nodes)
        click.echo("\nNode types:")
        for node_type, count in sorted(type_counts.items()):
            click.echo(f"  {node_type:<30} {count}")

        click.echo(f"\nSummary:\n{result.summary}")
        click.echo("\n" + "=" * 70)
        click.echo("Nothing was written to Neo4j or Qdrant.")

        # --- Optional JSON output ---
        if output:
            graph_json = {
                "summary": result.summary,
                "key_insights": result.key_insights,
                "nodes": [
                    {
                        "id": n.id,
                        "type": n.node_type.value,
                        "name": n.name,
                        "description": n.description,
                        "metadata": n.metadata,
                    }
                    for n in result.graph.nodes
                ],
                "relationships": [
                    {
                        "source_id": r.source_id,
                        "target_id": r.target_id,
                        "type": r.relation_type.value,
                        "confidence": r.confidence,
                    }
                    for r in result.graph.relationships
                ],
            }
            Path(output).write_text(json.dumps(graph_json, indent=2, default=str))
            click.echo(f"\n✓ Graph preview saved to {output}")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\n--- Traceback ---", err=True)
        traceback.print_exc()
        # Give a specific hint for the most common cause: wrong PyTorch version
        msg = str(e)
        if "nn" in msg or "torch" in msg.lower() or "pytorch" in msg.lower():
            click.echo(
                "\nHint: This looks like a PyTorch version incompatibility.\n"
                "sentence-transformers requires PyTorch >= 2.4, but an older version\n"
                "is installed.  Fix with:\n\n"
                "  pip install --upgrade torch\n\n"
                "Or switch to OpenAI embeddings (no local PyTorch needed):\n"
                "  EMBEDDINGS_PROVIDER=openai in your .env",
                err=True,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
