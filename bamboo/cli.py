"""Interactive CLI for Bamboo."""

import asyncio
import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.agents.reasoning_navigator import ReasoningAgent
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.utils.logging import setup_logging

console = Console()


@click.group()
def cli():
    """Bamboo - AI Agent System for Task Exhaustion Analysis."""
    setup_logging()


@cli.command()
def interactive():
    """Start interactive mode."""
    console.print(
        Panel.fit(
            "[bold blue]Bamboo Interactive Mode[/bold blue]\n"
            "AI Agent System for Task Exhaustion Analysis",
            border_style="blue",
        )
    )

    while True:
        console.print("\n[bold]Main Menu:[/bold]")
        console.print("1. Populate knowledge base")
        console.print("2. Analyze exhausted task")
        console.print("3. Query knowledge graph")
        console.print("4. Exit")

        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4"])

        if choice == "1":
            asyncio.run(populate_knowledge_interactive())
        elif choice == "2":
            asyncio.run(analyze_task_interactive())
        elif choice == "3":
            asyncio.run(query_knowledge_interactive())
        elif choice == "4":
            console.print("[green]Goodbye![/green]")
            break


async def populate_knowledge_interactive():
    """Interactive knowledge population."""
    console.print("\n[bold cyan]Populate Knowledge Base[/bold cyan]")

    email_text = ""
    if Confirm.ask("Do you have an email thread?"):
        email_path = Prompt.ask("Enter path to email text file")
        try:
            with open(email_path) as f:
                email_text = f.read()
        except Exception as e:
            console.print(f"[red]Error reading email file: {e}[/red]")
            return

    task_dict = None
    if Confirm.ask("Do you have task data?"):
        task_path = Prompt.ask("Enter path to task JSON file")
        try:
            with open(task_path) as f:
                task_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading task file: {e}[/red]")
            return

    external_dict = None
    if Confirm.ask("Do you have external data?"):
        external_path = Prompt.ask("Enter path to external JSON file")
        try:
            with open(external_path) as f:
                external_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading external file: {e}[/red]")
            return

    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()

    try:
        await graph_db.connect()
        await vector_db.connect()

        agent = KnowledgeAccumulator(graph_db, vector_db)

        with console.status("[bold green]Extracting knowledge..."):
            result = await agent.process_knowledge(
                email_text=email_text,
                task_data=task_dict,
                external_data=external_dict,
            )

        console.print("\n[bold green]✓ Knowledge extracted successfully![/bold green]")
        console.print(f"\n[bold]Summary:[/bold]\n{result.summary}")
        console.print("\n[bold]Statistics:[/bold]")
        console.print(f"  Nodes: {len(result.graph.nodes)}")
        console.print(f"  Relationships: {len(result.graph.relationships)}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await graph_db.close()
        await vector_db.close()


async def analyze_task_interactive():
    """Interactive task analysis."""
    console.print("\n[bold cyan]Analyze Exhausted Task[/bold cyan]")

    task_path = Prompt.ask("Enter path to task JSON file")
    try:
        with open(task_path) as f:
            task_dict = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading task file: {e}[/red]")
        return

    external_dict = None
    if Confirm.ask("Do you have external data?"):
        external_path = Prompt.ask("Enter path to external JSON file")
        try:
            with open(external_path) as f:
                external_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading external file: {e}[/red]")

    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()

    try:
        await graph_db.connect()
        await vector_db.connect()

        agent = ReasoningAgent(graph_db, vector_db)

        with console.status("[bold green]Analyzing task..."):
            result = await agent.analyze_task(
                task_data=task_dict,
                external_data=external_dict,
            )

        console.print("\n" + "=" * 80)
        console.print(
            Panel(
                f"[bold]Task ID:[/bold] {result.task_id}\n"
                f"[bold]Root Cause:[/bold] {result.root_cause}\n"
                f"[bold]Confidence:[/bold] {result.confidence:.2%}\n"
                f"[bold]Resolution:[/bold] {result.resolution}",
                title="Analysis Results",
                border_style="green",
            )
        )

        console.print(f"\n[bold]Explanation:[/bold]\n{result.explanation}")

        console.print("\n" + "-" * 80)
        console.print(
            Panel(result.email_content, title="Email Draft", border_style="blue")
        )

        if Confirm.ask("\nDo you approve this analysis?"):
            console.print("[green]✓ Analysis approved![/green]")
        else:
            feedback = Prompt.ask("Please provide feedback for improvement")
            console.print(f"[yellow]Feedback recorded: {feedback}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await graph_db.close()
        await vector_db.close()


async def query_knowledge_interactive():
    """Interactive knowledge graph querying."""
    console.print("\n[bold cyan]Query Knowledge Graph[/bold cyan]")

    query_type = Prompt.ask(
        "Query type",
        choices=["error", "features"],
        default="error",
    )

    graph_db = GraphDatabaseClient()

    try:
        await graph_db.connect()

        if query_type == "error":
            error_name = Prompt.ask("Enter error message or keyword")
            results = await graph_db.find_causes(errors=[error_name], limit=10)
        else:
            features_str = Prompt.ask("Enter features (comma-separated)")
            features = [f.strip() for f in features_str.split(",")]
            results = await graph_db.find_causes(task_features=features, limit=10)

        if results:
            table = Table(title="Query Results")
            table.add_column("Cause", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Frequency", style="green")
            table.add_column("Confidence", style="yellow")

            for result in results:
                table.add_row(
                    result.get("cause_name", ""),
                    result.get("cause_description", "")[:50] + "...",
                    str(result.get("frequency", 0)),
                    f"{result.get('confidence', 0):.2f}",
                )

            console.print(table)
        else:
            console.print("[yellow]No results found[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await graph_db.close()


if __name__ == "__main__":
    cli()
