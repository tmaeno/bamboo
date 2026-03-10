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
@click.option("--task-data", type=click.Path(exists=True), default=None, help="Path to task data JSON file. Mutually exclusive with --task-id.")
@click.option("--task-id", type=int, default=None, help=("PanDA jediTaskID — fetch task data directly from PanDA instead of a file. Requires PANDA_URL / PANDA_URL_SSL to be set (or uses the CERN defaults). Mutually exclusive with --task-data."))
@click.option("--external-data", type=click.Path(exists=True), help="Path to external data JSON file")
@click.option("--output", type=click.Path(), help="Path to save analysis results")
def main(task_data, task_id, external_data, output):
    """Analyze a problematic task and generate a resolution.

    Task data can be supplied either as a local JSON file (--task-data) or
    fetched live from PanDA by jediTaskID (--task-id).  The two options are
    mutually exclusive and at least one must be provided.
    """
    setup_logging()

    if task_data and task_id:
        raise click.UsageError("--task-data and --task-id are mutually exclusive.")
    if not task_data and task_id is None:
        raise click.UsageError("One of --task-data or --task-id is required.")

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

    result = asyncio.run(_analyze_task(task_dict, external_dict))

    # Display results
    click.echo("\n" + "=" * 80)
    click.echo("TASK ANALYSIS RESULTS")
    click.echo("=" * 80)
    click.echo(f"\nTask ID: {result.task_id}")
    click.echo(f"Root Cause: {result.root_cause}")
    click.echo(f"Confidence: {result.confidence:.2%}")
    click.echo(f"\nResolution: {result.resolution}")
    click.echo(f"\nExplanation:\n{result.explanation}")
    click.echo("\n" + "-" * 80)
    click.echo("EMAIL DRAFT")
    click.echo("-" * 80)
    click.echo(result.email_content)
    click.echo("=" * 80)

    if output:
        output_path = Path(output)
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


async def _analyze_task(task_dict, external_dict):
    """Run the async reasoning pipeline and return results."""
    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()

    try:
        await graph_db.connect()
        await vector_db.connect()

        agent = ReasoningNavigator(graph_db, vector_db)

        click.echo("Analyzing task...")
        return await agent.analyze_task(
            task_data=task_dict,
            external_data=external_dict,
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        await graph_db.close()
        await vector_db.close()


if __name__ == "__main__":
    main()
