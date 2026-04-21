"""Script to analyze a problematic task."""

import asyncio
import json
import sys
from pathlib import Path

import click

from bamboo.agents.reasoning_navigator import ReasoningNavigator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.utils.logging import setup_logging


@click.command()
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
        "PanDA jediTaskID — fetch task data directly from PanDA instead of a file. Requires PANDA_URL / PANDA_URL_SSL to be set (or uses the CERN defaults). Mutually exclusive with --task-data."
    ),
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    help="Path to external data JSON file",
)
@click.option("--output", type=click.Path(), help="Path to save analysis results")
@click.option(
    "--compare-task-id",
    "compare_task_ids",
    type=int,
    multiple=True,
    help=(
        "Additional jediTaskID(s) to compare against. When one or more are given, "
        "the common subgraph across all tasks (including --task-id) is displayed "
        "instead of the single-task analysis. Repeatable: --compare-task-id 456 "
        "--compare-task-id 789."
    ),
)
@click.option(
    "--min-occurrences",
    type=click.IntRange(min=2),
    default=2,
    show_default=True,
    help="Minimum number of tasks that must share an edge for it to appear in the pattern output.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(task_data, task_id, external_data, output, compare_task_ids, min_occurrences, verbose):
    """Analyze a problematic task and generate a resolution.

    Task data can be supplied either as a local JSON file (--task-data) or
    fetched live from PanDA by jediTaskID (--task-id).  The two options are
    mutually exclusive and at least one must be provided.

    When --compare-task-id is given, the command instead displays the common
    subgraph across all specified tasks (edges shared by at least
    --min-occurrences tasks).  This requires all tasks to have been previously
    accumulated with ``bamboo populate``.
    """
    setup_logging()

    if task_data and task_id:
        raise click.UsageError("--task-data and --task-id are mutually exclusive.")
    if not task_data and task_id is None:
        raise click.UsageError("One of --task-data or --task-id is required.")

    task_dict = None
    if task_data:
        task_dict = json.loads(Path(task_data).read_text())

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    if compare_task_ids:
        all_task_ids = ([task_id] if task_id is not None else []) + list(compare_task_ids)
        asyncio.run(_find_pattern(all_task_ids, min_occurrences))
        return

    result, prescription, email_content = asyncio.run(
        _analyze_task(task_dict, task_id, external_dict, verbose)
    )

    # Display results
    click.echo("\n" + "=" * 80)
    click.echo("TASK ANALYSIS RESULTS")
    click.echo("=" * 80)
    click.echo(f"\nTask ID: {result.task_id}")
    click.echo(f"Root Cause: {result.root_cause}")
    click.echo(f"Confidence: {result.confidence:.2%}")
    click.echo(f"\nResolution: {result.resolution}")
    click.echo(f"\nExplanation:\n{result.explanation}")

    if prescription and prescription.get("hints"):
        click.echo("\n" + "-" * 80)
        click.echo("PRESCRIPTION")
        click.echo("-" * 80)
        for hint in prescription["hints"]:
            click.echo(f"  • {hint}")
        if prescription.get("command_template"):
            click.echo(f"\n  Suggested options: {prescription['command_template']}")
        if prescription.get("notes"):
            click.echo(f"\n  Notes: {prescription['notes']}")

    click.echo("\n" + "-" * 80)
    click.echo("EMAIL DRAFT")
    click.echo("-" * 80)
    click.echo(email_content)
    click.echo("=" * 80)

    if output:
        output_path = Path(output)
        result.email_content = email_content
        output_path.write_text(result.model_dump_json(indent=2))
        click.echo(f"\n✓ Results saved to {output}")

    click.echo("\n")
    feedback = click.prompt(
        "Do you approve this analysis? (yes/no/edit)",
        type=click.Choice(["yes", "no", "edit"]),
        default="yes",
    )

    if feedback == "yes":
        click.echo("✓ Analysis approved!")
    elif feedback == "no":
        reason = click.prompt("Please provide feedback for improvement")
        click.echo(f"Feedback recorded: {reason}")
    else:
        click.echo("Please edit the results manually.")


async def _find_pattern(task_ids: list[int], min_occurrences: int) -> None:
    """Fetch graph_ids for the given task IDs and display the common subgraph."""
    import uuid

    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

    graph_db = GraphDatabaseClient()
    try:
        await graph_db.connect()

        graph_ids: list[str] = []
        for tid in task_ids:
            try:
                task_dict = await fetch_task_data(tid)
            except Exception as e:
                click.echo(f"Warning: could not fetch task {tid}: {e}", err=True)
                continue
            status = (task_dict or {}).get("status", "")
            if status:
                gid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"graph:{tid}:{status}"))
            else:
                gid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"graph:{tid}"))
            graph_ids.append(gid)
            click.echo(f"  Task {tid} (status={status or 'unknown'}) → graph_id {gid}")

        if len(graph_ids) < 2:
            click.echo("Need at least 2 resolvable task IDs to compute a pattern.", err=True)
            return

        click.echo(
            f"\nQuerying common subgraph across {len(graph_ids)} task(s) "
            f"(min_occurrences={min_occurrences})..."
        )
        edges = await graph_db.find_common_pattern(graph_ids, min_occurrences)

        click.echo("\n" + "=" * 70)
        click.echo("COMMON PATTERN")
        click.echo("=" * 70)
        click.echo(f"\nEdges matching threshold: {len(edges)}")

        if not edges:
            click.echo("No common edges found at this threshold.")
            return

        from collections import Counter
        node_types: Counter = Counter()
        seen_nodes: set[str] = set()
        for e in edges:
            for name, ntype in ((e["src_name"], e["src_type"]), (e["tgt_name"], e["tgt_type"])):
                if name not in seen_nodes:
                    node_types[ntype] += 1
                    seen_nodes.add(name)

        click.echo(f"Distinct nodes: {len(seen_nodes)}")
        click.echo("\nNode types:")
        for ntype, count in sorted(node_types.items()):
            click.echo(f"  {ntype:<30} {count}")

        click.echo("\nEdges (sorted by occurrence count):")
        for e in edges:
            conf = f"  [{e['confidence']:.2f}]" if e["confidence"] != 1.0 else ""
            click.echo(
                f"  {e['src_name']}  -[{e['rel_type']}]->  {e['tgt_name']}"
                f"  (shared by {e['occurrence_count']}/{len(graph_ids)} tasks){conf}"
            )
        click.echo("\n" + "=" * 70)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        await graph_db.close()


async def _analyze_task(task_dict, task_id, external_dict, verbose=False):
    """Run the async reasoning pipeline and return results."""
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

    from bamboo.agents.email_drafter import EmailDrafter
    from bamboo.agents.exploration_planner import ExplorationPlanner
    from bamboo.agents.extra_source_explorer import ExtraSourceExplorer
    from bamboo.agents.prescription_composer import PrescriptionComposer
    from bamboo.config import get_settings
    from bamboo.mcp.factory import build_mcp_client

    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()

    try:
        await graph_db.connect()
        await vector_db.connect()

        settings = get_settings()
        _mcp = build_mcp_client(settings)
        explorer = ExtraSourceExplorer(_mcp, planner=ExplorationPlanner(_mcp))
        agent = ReasoningNavigator(graph_db, vector_db, explorer=explorer)

        click.echo("Analyzing task...")
        result = await agent.analyze_task(
            task_data=task_dict,
            external_data=external_dict,
        )

        click.echo("Composing prescription...")
        prescription = await PrescriptionComposer(_mcp).compose(task_dict, result)

        click.echo("Drafting email...")
        email_content = await EmailDrafter().draft(task_dict, result, prescription)

        return result, prescription, email_content

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        await graph_db.close()
        await vector_db.close()


if __name__ == "__main__":
    main()
