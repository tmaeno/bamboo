"""Shared display helpers for extraction result output."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import click

from bamboo.models.knowledge_entity import ExtractedKnowledge


def print_extraction_result(result: ExtractedKnowledge, verbose: bool, dry_run: bool) -> None:
    """Print a summary of an extraction result to stdout.

    Args:
        result:   The extraction result returned by
                  :meth:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator.process_knowledge`.
        verbose:  When ``True``, print per-node details, isolated-node list,
                  and full relationship list.
        dry_run:  When ``True``, add a footer note that nothing was written.
    """
    click.echo("\n" + "=" * 70)
    click.echo("EXTRACTION PREVIEW" if dry_run else "EXTRACTION RESULT")
    click.echo("=" * 70)
    click.echo(f"\nNodes:         {len(result.graph.nodes)}")
    click.echo(f"Relationships: {len(result.graph.relationships)}")

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

        connected_names: set[str] = set()
        for r in result.graph.relationships:
            connected_names.add(r.source_id)
            connected_names.add(r.target_id)

        def _node_extras(n) -> dict:
            from bamboo.agents.extractors.panda_knowledge_extractor import _node_concepts  # noqa: PLC0415
            extras: dict = {}
            if hasattr(n, "confidence") and n.confidence != 1.0:
                extras["confidence"] = f"{n.confidence:.2f}"
            if hasattr(n, "steps") and n.steps:
                extras["steps"] = len(n.steps)
            if hasattr(n, "attribute"):
                extras["attribute"] = n.attribute
            if hasattr(n, "severity") and n.severity:
                extras["severity"] = n.severity
            concepts = _node_concepts(n)
            if concepts:
                extras["concept"] = ", ".join(concepts)
            return extras

        def _print_connected_node(n) -> None:
            click.echo(f"  • {n.name}")
            if n.description:
                desc = n.description.replace("\n", " ")
                if len(desc) > 120:
                    desc = desc[:117] + "..."
                click.echo(f"    {desc}")
            extras = _node_extras(n)
            if extras:
                click.echo("    " + "  ".join(f"{k}={v}" for k, v in extras.items()))

        def _print_isolated_node(n) -> None:
            click.echo(click.style(f"  • {n.name}", fg="bright_black"))
            if n.description:
                desc = n.description.replace("\n", " ")
                if len(desc) > 120:
                    desc = desc[:117] + "..."
                click.echo(click.style(f"    {desc}", fg="bright_black"))
            extras = _node_extras(n)
            if extras:
                click.echo(click.style(
                    "    " + "  ".join(f"{k}={v}" for k, v in extras.items()),
                    fg="bright_black",
                ))

        nodes_by_type: dict = defaultdict(list)
        for n in result.graph.nodes:
            nodes_by_type[n.node_type.value].append(n)
        for type_name in sorted(nodes_by_type):
            nodes = nodes_by_type[type_name]
            connected = [n for n in nodes if n.name in connected_names]
            isolated = [n for n in nodes if n.name not in connected_names]
            click.echo(f"\n[{type_name}]")
            if connected:
                click.echo("  connected:")
                for n in connected:
                    _print_connected_node(n)
            if isolated:
                click.echo(click.style("  isolated:", fg="bright_black"))
                for n in isolated:
                    _print_isolated_node(n)

        click.echo("\n--- Relationships ---")
        for r in sorted(
            result.graph.relationships,
            key=lambda r: (r.relation_type.value, r.source_id),
        ):
            conf = f"  [{r.confidence:.2f}]" if r.confidence != 1.0 else ""
            click.echo(f"  {r.source_id}  -[{r.relation_type.value}]->  {r.target_id}{conf}")

    click.echo(f"\nSummary:\n{result.summary}")
    click.echo("\n" + "=" * 70)
    if dry_run:
        click.echo("Nothing was written to Neo4j or Qdrant.")


def save_graph_json(result: ExtractedKnowledge, output: str) -> None:
    """Serialise an extraction result graph to a JSON file.

    Args:
        result: The extraction result.
        output: Destination file path.
    """
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
