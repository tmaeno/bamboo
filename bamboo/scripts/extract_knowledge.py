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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Stream live agent-style narration of each extraction step.",
)
def main(email_thread, task_data, task_id, external_data, output, verbose):
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

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    asyncio.run(
        _run_extraction(
            email_text, task_dict, task_id, external_dict, output, verbose=verbose
        )
    )


async def _run_extraction(
    email_text, task_dict, task_id, external_dict, output=None, verbose=False
):
    """Run extraction in dry-run mode and print the result."""
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

    # Dry-run needs no database connections at all — skip Neo4j and Qdrant.
    # Pass None so KnowledgeAccumulator skips all DB calls.
    from bamboo.agents.extra_source_explorer import ExtraSourceExplorer
    from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
    from bamboo.config import get_settings
    from bamboo.mcp.panda_mcp_client import PandaMcpClient

    settings = get_settings()
    reviewer = KnowledgeReviewer() if settings.enable_knowledge_review else None
    explorer = ExtraSourceExplorer(PandaMcpClient()) if settings.enable_knowledge_review else None
    agent = KnowledgeAccumulator(graph_db=None, vector_db=None, reviewer=reviewer, explorer=explorer)

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

        rel_counts = Counter(r.relation_type.value for r in result.graph.relationships)
        click.echo("\nRelationship types:")
        for rel_type, count in sorted(rel_counts.items()):
            click.echo(f"  {rel_type:<30} {count}")

        if verbose:
            click.echo("\n--- Node details ---")
            from collections import defaultdict

            nodes_by_type: dict = defaultdict(list)
            for n in result.graph.nodes:
                nodes_by_type[n.node_type.value].append(n)
            for type_name in sorted(nodes_by_type):
                click.echo(f"\n[{type_name}]")
                for n in nodes_by_type[type_name]:
                    click.echo(f"  • {n.name}")
                    if n.description:
                        # wrap long descriptions at 80 chars
                        desc = n.description.replace("\n", " ")
                        if len(desc) > 120:
                            desc = desc[:117] + "..."
                        click.echo(f"    {desc}")
                    # type-specific extra fields
                    extras = {}
                    if hasattr(n, "confidence") and n.confidence != 1.0:
                        extras["confidence"] = f"{n.confidence:.2f}"
                    if hasattr(n, "steps") and n.steps:
                        extras["steps"] = len(n.steps)
                    if hasattr(n, "attribute"):
                        extras["attribute"] = n.attribute
                    if hasattr(n, "severity") and n.severity:
                        extras["severity"] = n.severity
                    if extras:
                        click.echo(
                            "    " + "  ".join(f"{k}={v}" for k, v in extras.items())
                        )

            click.echo("\n--- Relationships ---")
            for r in sorted(
                result.graph.relationships,
                key=lambda r: (r.relation_type.value, r.source_id),
            ):
                conf = f"  [{r.confidence:.2f}]" if r.confidence != 1.0 else ""
                click.echo(
                    f"  {r.source_id}  -[{r.relation_type.value}]->  {r.target_id}{conf}"
                )

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
