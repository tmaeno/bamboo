"""Interactive CLI for Bamboo."""

import asyncio
import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.agents.reasoning_navigator import ReasoningNavigator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.utils.logging import setup_logging

console = Console()


@click.group()
def cli():
    """Bamboo - Bolstered Assistance for Managing and Building Operations and Oversight."""
    setup_logging()


@cli.command()
def interactive():
    """Start interactive mode."""
    console.print(
        Panel.fit(
            "[bold blue]Bamboo Interactive Mode[/bold blue]\n"
            "Bolstered Assistance for Managing and Building Operations and Oversight",
            border_style="blue",
        )
    )

    while True:
        console.print("\n[bold]Main Menu:[/bold]")
        console.print("1. Populate knowledge base")
        console.print("2. Analyze problematic task")
        console.print("3. Query knowledge graph (graph database)")
        console.print("4. Query knowledge base (vector database)")
        console.print("5. Exit")

        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])

        if choice == "1":
            asyncio.run(populate_knowledge_interactive())
        elif choice == "2":
            asyncio.run(analyze_task_interactive())
        elif choice == "3":
            asyncio.run(query_knowledge_interactive())
        elif choice == "4":
            asyncio.run(query_vector_interactive())
        elif choice == "5":
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
        use_panda = Confirm.ask(
            "Fetch task data directly from PanDA by task ID?", default=False
        )
        if use_panda:
            task_id_str = Prompt.ask("Enter PanDA jediTaskID")
            try:
                from bamboo.utils.panda_client import fetch_task_data

                with console.status(
                    f"[bold green]Fetching task {task_id_str} from PanDA..."
                ):
                    task_dict = await fetch_task_data(int(task_id_str))
                console.print(
                    f"[green]✓ Fetched {len(task_dict)} fields for task {task_id_str}[/green]"
                )
            except Exception as e:
                console.print(f"[red]Error fetching task from PanDA: {e}[/red]")
                return
        else:
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

    dry_run = Confirm.ask(
        "\nRun in [bold]dry-run mode[/bold] (extract & preview without writing to databases)?",
        default=False,
    )

    try:
        await graph_db.connect()
        await vector_db.connect()

        agent = KnowledgeAccumulator(graph_db, vector_db)

        with console.status("[bold green]Extracting knowledge..."):
            result = await agent.process_knowledge(
                email_text=email_text,
                task_data=task_dict,
                external_data=external_dict,
                dry_run=True,  # always preview first
            )

        console.print("\n[bold green]✓ Extraction preview complete[/bold green]")
        console.print(f"\n[bold]Summary:[/bold]\n{result.summary}")
        console.print("\n[bold]Statistics:[/bold]")
        console.print(f"  Nodes: {len(result.graph.nodes)}")
        console.print(f"  Relationships: {len(result.graph.relationships)}")

        if dry_run:
            console.print(
                "\n[yellow]Dry-run mode: no data was written to the databases.[/yellow]"
            )
        else:
            if not Confirm.ask("\nCommit this data to the databases?", default=True):
                console.print("[yellow]Aborted — no data written.[/yellow]")
                return

            with console.status("[bold green]Writing to databases..."):
                await agent._store_graph(result.graph)
                await agent._store_in_vector_db(
                    result.graph,
                    result.summary,
                    await agent._extract_key_insights(result.graph),
                )
            console.print("[bold green]✓ Knowledge stored successfully![/bold green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await graph_db.close()
        await vector_db.close()


async def analyze_task_interactive():
    """Interactive task analysis."""
    console.print("\n[bold cyan]Analyze Problematic Task[/bold cyan]")

    task_dict = None
    use_panda = Confirm.ask(
        "Fetch task data directly from PanDA by task ID?", default=False
    )
    if use_panda:
        task_id_str = Prompt.ask("Enter PanDA jediTaskID")
        try:
            from bamboo.utils.panda_client import fetch_task_data

            with console.status(
                f"[bold green]Fetching task {task_id_str} from PanDA..."
            ):
                task_dict = await fetch_task_data(int(task_id_str))
            console.print(
                f"[green]✓ Fetched {len(task_dict)} fields for task {task_id_str}[/green]"
            )
        except Exception as e:
            console.print(f"[red]Error fetching task from PanDA: {e}[/red]")
            return
    else:
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

        agent = ReasoningNavigator(graph_db, vector_db)

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


@cli.command("fetch-task")
@click.argument("task_id", type=int)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Save the fetched task data as JSON to this file path.",
)
def fetch_task_cmd(task_id, output):
    """Fetch task details from PanDA and display or save them.

    TASK_ID is the PanDA jediTaskID to look up.

    Examples:

    \b
      bamboo fetch-task 12345
      bamboo fetch-task 12345 --output task_12345.json
    """

    async def _run():
        from bamboo.utils.panda_client import fetch_task_data

        with console.status(f"[bold green]Fetching task {task_id} from PanDA..."):
            data = await fetch_task_data(task_id)
        return data

    try:
        data = asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    if output:
        import json as _json
        from pathlib import Path

        Path(output).write_text(_json.dumps(data, indent=2))
        console.print(f"[green]✓ Task {task_id} data saved to {output}[/green]")
    else:
        import json as _json

        console.print_json(_json.dumps(data))


async def query_vector_interactive():
    """Interactive vector database (semantic similarity) search."""
    console.print("\n[bold cyan]Query Knowledge Base (Vector Database)[/bold cyan]")

    query_text = Prompt.ask("Enter search query (free-form text)")

    limit_str = Prompt.ask("Maximum number of results", default="5")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 5

    threshold_str = Prompt.ask("Minimum similarity score (0.0 – 1.0)", default="0.5")
    try:
        score_threshold = float(threshold_str)
    except ValueError:
        score_threshold = 0.5

    # Optional section filter (Summary, KeyInsight, canonical_node::*, etc.)
    section_filter = None
    if Confirm.ask("Filter by section?", default=False):
        section_filter = Prompt.ask(
            "Section name (e.g. Summary, KeyInsight, canonical_node::Cause)"
        ).strip()

    vector_db = VectorDatabaseClient()

    try:
        await vector_db.connect()

        from bamboo.llm import get_embeddings

        with console.status("[bold green]Embedding query..."):
            embeddings = get_embeddings()
            query_embedding = embeddings.embed_query(query_text)

        filter_conditions = {"section": section_filter} if section_filter else None

        with console.status("[bold green]Searching vector database..."):
            results = await vector_db.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions,
            )

        if not results:
            console.print(
                "[yellow]No results found above the similarity threshold.[/yellow]"
            )
            return

        console.print(f"\n[bold green]Found {len(results)} result(s):[/bold green]\n")
        for i, hit in enumerate(results, 1):
            score = hit.get("score", 0.0)
            content = hit.get("content", "").strip()
            metadata = hit.get("metadata", {})
            section = metadata.get("section", "—")
            graph_id = metadata.get("graph_id", "—")

            console.print(
                Panel(
                    content,
                    title=f"[bold]#{i}  score={score:.3f}  section={section}  graph_id={graph_id}[/bold]",
                    border_style="cyan",
                )
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await vector_db.close()


async def query_knowledge_interactive():
    """Interactive knowledge graph querying."""
    console.print("\n[bold cyan]Query Knowledge Graph (Graph Database)[/bold cyan]")

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


@cli.command("populate")
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
        "Mutually exclusive with --task-data."
    ),
)
@click.option(
    "--external-data",
    type=click.Path(exists=True),
    help="Path to external data JSON file",
)
def populate_cmd(email_thread, task_data, task_id, external_data):
    """Populate knowledge base from various sources."""
    from bamboo.scripts.populate_knowledge import main as _main

    ctx = click.get_current_context()
    ctx.invoke(
        _main,
        email_thread=email_thread,
        task_data=task_data,
        task_id=task_id,
        external_data=external_data,
    )


@cli.command("analyze")
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
    type=click.Path(),
    help="Path to save analysis results",
)
def analyze_cmd(task_data, task_id, external_data, output):
    """Analyze a problematic task and generate a resolution."""
    from bamboo.scripts.analyze_task import main as _main

    ctx = click.get_current_context()
    ctx.invoke(
        _main,
        task_data=task_data,
        task_id=task_id,
        external_data=external_data,
        output=output,
    )


@cli.command("verify")
def verify_cmd():
    """Verify that the Bamboo package is correctly installed."""
    import sys

    from bamboo.scripts.verify import main as _main

    sys.exit(_main())


if __name__ == "__main__":
    cli()
