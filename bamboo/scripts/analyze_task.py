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
    required=True,
    help="Path to task data JSON file",
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    help="Path to external data JSON file",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Path to save analysis results",
)
def main(task_data, external_data, output):
    """Analyze a problematic task and generate a resolution."""
    setup_logging()

    # Load data
    task_dict = json.loads(Path(task_data).read_text())

    external_dict = None
    if external_data:
        external_dict = json.loads(Path(external_data).read_text())

    # Run analysis
    result = asyncio.run(analyze_task(task_dict, external_dict))

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

    # Save results if requested
    if output:
        output_path = Path(output)
        output_path.write_text(result.model_dump_json(indent=2))
        click.echo(f"\n✓ Results saved to {output}")

    # Ask for feedback
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


async def analyze_task(task_dict, external_dict):
    """Analyze task and return results."""
    neo4j = GraphDatabaseClient()
    qdrant = VectorDatabaseClient()

    try:
        await neo4j.connect()
        await qdrant.connect()

        agent = ReasoningNavigator(neo4j, qdrant)

        click.echo("Analyzing task...")
        result = await agent.analyze_task(
            task_data=task_dict,
            external_data=external_dict,
        )

        return result

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        await neo4j.close()
        await qdrant.close()


if __name__ == "__main__":
    main()
