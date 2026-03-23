"""Interactive CLI for Bamboo."""

import atexit
import os
import readline
import sys

import asyncio
import json

import re

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bamboo.utils.logging import setup_logging

console = Console()

# ---------------------------------------------------------------------------
# Shell completion — auto-installed on first run
# ---------------------------------------------------------------------------

# Sentinel written into rc files so we only ever add it once.
_COMPLETION_MARKER = "# bamboo shell completion (auto-installed)"

# NOTE: completion scripts are NOT defined here.
# The single source of truth is:
#   bamboo/data/completions/_bamboo       (zsh)
#   bamboo/data/completions/_bamboo.bash  (bash)
# cli.py reads them via importlib.resources — no hardcoded copy anywhere.


def _completion_script(filename: str) -> str:
    """Read a completion script from the installed package data.

    ``bamboo/data/completions/`` is the single source of truth.
    ``~/.zsh/completions/_bamboo`` is always derived from it, never the
    other way around — so they are guaranteed to be identical.
    """
    import importlib.resources as pkg_resources

    ref = pkg_resources.files("bamboo.data.completions").joinpath(filename)
    return ref.read_text(encoding="utf-8")


def _ensure_completion() -> None:
    """Silently install shell completion on first run.

    Writes the packaged completion script to the user's completion directory.
    No subprocess, no Python startup at Tab time — instant response.
    Does nothing if already installed, not a TTY, or shell is unsupported.
    """
    if os.environ.get("_BAMBOO_COMPLETE"):
        return
    if not sys.stdout.isatty():
        return

    shell = os.path.basename(os.environ.get("SHELL", "")).lower()

    if shell == "zsh":
        _install_zsh_completion()
    elif shell == "bash":
        _install_bash_completion()


def _install_zsh_completion() -> None:
    """Write _bamboo from package data to ~/.zsh/completions/ and wire ~/.zshrc."""
    comp_dir = os.path.expanduser("~/.zsh/completions")
    comp_file = os.path.join(comp_dir, "_bamboo")
    rc_path = os.path.expanduser("~/.zshrc")

    # compinit -u silently skips the "insecure directories" prompt.
    # chmod 755 on the directory is belt-and-suspenders: it ensures the dir
    # is not group/world-writable, which is what triggers the warning.
    COMPINIT_LINE = "autoload -Uz compinit && compinit -u"

    try:
        script = _completion_script("_bamboo")

        # Keep installed file in sync with package data — only write on change.
        os.makedirs(comp_dir, exist_ok=True)
        # Ensure the directory is not group/world-writable (zsh security check).
        os.chmod(comp_dir, 0o755)
        try:
            with open(comp_file) as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current != script:
            with open(comp_file, "w") as f:
                f.write(script)

        try:
            with open(rc_path) as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""

        if _COMPLETION_MARKER in existing:
            # Already installed — patch compinit to use -u if it doesn't already.
            # This fixes existing installs that got the old compinit without -u.
            if "compinit -u" not in existing:
                patched = existing.replace(
                    "autoload -Uz compinit && compinit\n",
                    f"{COMPINIT_LINE}\n",
                )
                with open(rc_path, "w") as f:
                    f.write(patched)
            return

        fpath_line = f"fpath=('{comp_dir}' $fpath)"
        new_content = (
            f"{_COMPLETION_MARKER}\n"
            f"{fpath_line}\n"
            f"{COMPINIT_LINE}\n"
            f"{existing}"
        )
        with open(rc_path, "w") as f:
            f.write(new_content)

        console.print(
            "[dim]✓ Tab-completion installed. "
            "Run [bold]exec zsh[/bold] to activate.[/dim]"
        )
    except Exception:
        pass


def _install_bash_completion() -> None:
    """Write static bash completion script to ~/.bash_completion.d/bamboo."""
    comp_dir = os.path.expanduser("~/.bash_completion.d")
    comp_file = os.path.join(comp_dir, "bamboo")
    rc_path = os.path.expanduser("~/.bashrc")

    try:
        os.makedirs(comp_dir, exist_ok=True)
        try:
            with open(comp_file) as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current != _completion_script("_bamboo.bash"):
            with open(comp_file, "w") as f:
                f.write(_completion_script("_bamboo.bash"))

        try:
            with open(rc_path) as f:
                existing = f.read()
        except FileNotFoundError:
            existing = ""

        if _COMPLETION_MARKER in existing:
            return

        source_line = f"source '{comp_file}'"
        with open(rc_path, "a") as f:
            f.write(f"\n{_COMPLETION_MARKER}\n{source_line}\n")

        console.print(
            "[dim]✓ Tab-completion installed. "
            "Run [bold]exec bash[/bold] to activate.[/dim]"
        )
    except Exception:
        pass


@click.group()
def cli():
    """Bamboo - Bolstered Assistance for Managing and Building Operations and Oversight."""
    setup_logging()
    _ensure_completion()


_HISTORY_FILE = os.path.expanduser("~/.bamboo_history")
_readline_initialized = False


def _setup_readline() -> None:
    """Load readline history and arrange for it to be saved on exit."""
    global _readline_initialized
    if _readline_initialized:
        return
    _readline_initialized = True
    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)
    atexit.register(readline.write_history_file, _HISTORY_FILE)


_RICH_MARKUP_RE = re.compile(r"\[/?[^\]]*\]")


def _strip_markup(text: str) -> str:
    return _RICH_MARKUP_RE.sub("", text)


def _ask(
    prompt: str,
    *,
    default: "str | None" = None,
    choices: "list[str] | None" = None,
) -> str:
    """Readline-compatible replacement for rich.prompt.Prompt.ask.

    Calls input(prompt) directly so readline knows the cursor position and
    up/down arrow history recall works without garbling the prompt.
    """
    plain = _strip_markup(prompt)
    if choices:
        plain += f" [{'/'.join(choices)}]"
    if default is not None:
        plain += f" ({default})"
    plain += ": "
    while True:
        try:
            answer = input(plain).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        if not answer:
            if default is not None:
                return default
            continue
        if choices and answer not in choices:
            console.print(
                f"[yellow]  Please select one of: {', '.join(choices)}[/yellow]"
            )
            continue
        return answer


def _confirm(prompt: str, *, default: "bool | None" = None) -> bool:
    """Readline-compatible replacement for rich.prompt.Confirm.ask."""
    plain = _strip_markup(prompt)
    hint = "[Y/n]" if default is True else "[y/N]" if default is False else "[y/n]"
    full = f"{plain} {hint}: "
    while True:
        try:
            answer = input(full).strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        if not answer:
            if default is not None:
                return default
            continue
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        console.print("[yellow]  Please enter y or n.[/yellow]")


@cli.command()
def interactive():
    """Start interactive mode."""
    _setup_readline()
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
        console.print("5. [red]Cleanup databases[/red]")
        console.print("6. Preview log filter")
        console.print("7. Exit")

        choice = _ask("Select an option", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            asyncio.run(populate_knowledge_interactive())
        elif choice == "2":
            asyncio.run(analyze_task_interactive())
        elif choice == "3":
            asyncio.run(query_knowledge_interactive())
        elif choice == "4":
            asyncio.run(query_vector_interactive())
        elif choice == "5":
            asyncio.run(cleanup_databases_interactive())
        elif choice == "6":
            asyncio.run(preview_log_filter_interactive())
        elif choice == "7":
            console.print("[green]Goodbye![/green]")
            break


async def populate_knowledge_interactive():
    """Interactive knowledge population."""
    from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
    from bamboo.database.graph_database_client import GraphDatabaseClient
    from bamboo.database.vector_database_client import VectorDatabaseClient

    console.print("\n[bold cyan]Populate Knowledge Base[/bold cyan]")

    email_text = ""
    if _confirm("Do you have an email thread?"):
        email_path = _ask("Enter path to email text file")
        try:
            with open(email_path) as f:
                email_text = f.read()
        except Exception as e:
            console.print(f"[red]Error reading email file: {e}[/red]")
            return

    task_dict = None
    if _confirm("Do you have task data?"):
        use_panda = _confirm(
            "Fetch task data directly from PanDA by task ID?", default=False
        )
        if use_panda:
            task_id_str = _ask("Enter PanDA jediTaskID")
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
            task_path = _ask("Enter path to task JSON file")
            try:
                with open(task_path) as f:
                    task_dict = json.load(f)
            except Exception as e:
                console.print(f"[red]Error reading task file: {e}[/red]")
                return

    external_dict = None
    if _confirm("Do you have external data?"):
        external_path = _ask("Enter path to external JSON file")
        try:
            with open(external_path) as f:
                external_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading external file: {e}[/red]")
            return

    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()

    dry_run = _confirm(
        "\nRun in [bold]dry-run mode[/bold] (extract & preview without writing to databases)?",
        default=False,
    )

    try:
        await graph_db.connect()
        await vector_db.connect()

        from bamboo.agents.knowledge_reviewer import KnowledgeReviewer
        from bamboo.agents.extra_source_explorer import ExtraSourceExplorer
        from bamboo.config import get_settings
        from bamboo.mcp.panda_mcp_client import PandaMcpClient
        settings = get_settings()
        reviewer = KnowledgeReviewer() if settings.enable_knowledge_review else None
        explorer = ExtraSourceExplorer(PandaMcpClient()) if settings.enable_knowledge_review else None
        agent = KnowledgeAccumulator(graph_db, vector_db, reviewer=reviewer, explorer=explorer)

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
            if not _confirm("\nCommit this data to the databases?", default=True):
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
    from bamboo.agents.reasoning_navigator import ReasoningNavigator
    from bamboo.database.graph_database_client import GraphDatabaseClient
    from bamboo.database.vector_database_client import VectorDatabaseClient

    console.print("\n[bold cyan]Analyze Problematic Task[/bold cyan]")

    task_dict = None
    use_panda = _confirm(
        "Fetch task data directly from PanDA by task ID?", default=False
    )
    if use_panda:
        task_id_str = _ask("Enter PanDA jediTaskID")
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
        task_path = _ask("Enter path to task JSON file")
        try:
            with open(task_path) as f:
                task_dict = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading task file: {e}[/red]")
            return

    external_dict = None
    if _confirm("Do you have external data?"):
        external_path = _ask("Enter path to external JSON file")
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

        if _confirm("\nDo you approve this analysis?"):
            console.print("[green]✓ Analysis approved![/green]")
        else:
            feedback = _ask("Please provide feedback for improvement")
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
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print every curl command and raw server response (useful for debugging auth/network issues).",
)
def fetch_task_cmd(task_id, output, verbose):
    """Fetch task details from PanDA and display or save them.

    TASK_ID is the PanDA jediTaskID to look up.

    Examples:

    \b
      bamboo fetch-task 12345
      bamboo fetch-task 12345 --output task_12345.json
      bamboo fetch-task 12345 --verbose
    """
    import json as _json
    from bamboo.utils.panda_client import (
        extract_log_urls,
        fetch_log_url,
        fetch_task_data,
    )

    try:
        with console.status(f"[bold green]Fetching task {task_id} from PanDA..."):
            data = asyncio.run(fetch_task_data(task_id, verbose=verbose))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    if output:
        from pathlib import Path

        Path(output).write_text(_json.dumps(data, indent=2, default=str))
        console.print(f"[green]✓ Task {task_id} data saved to {output}[/green]")
    else:
        console.print_json(_json.dumps(data, default=str))

    # When --verbose, extract any log URLs embedded in errorDialog and print them.
    if verbose:
        error_dialog = data.get("errorDialog", "") or ""
        log_urls = extract_log_urls(str(error_dialog))
        if log_urls:
            console.print(f"\n[bold]Linked log files in errorDialog:[/bold]")
            for url in log_urls:
                console.print(f"  [cyan]{url}[/cyan]")
                content = fetch_log_url(url)
                if content:
                    # Show first 100 lines to avoid flooding the terminal.
                    lines = content.splitlines()
                    preview = "\n".join(lines[:100])
                    if len(lines) > 100:
                        preview += f"\n[dim]… ({len(lines) - 100} more lines)[/dim]"
                    console.print(
                        f"\n[dim]── {url} ({len(lines)} lines) ──[/dim]\n{preview}\n"
                    )
                else:
                    console.print(f"  [yellow]Could not fetch log from {url}[/yellow]")


async def query_vector_interactive():
    """Interactive vector database (semantic similarity) search."""
    from bamboo.database.vector_database_client import VectorDatabaseClient

    console.print("\n[bold cyan]Query Knowledge Base (Vector Database)[/bold cyan]")

    query_text = _ask("Enter search query (free-form text)")

    limit_str = _ask("Maximum number of results", default="5")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 5

    threshold_str = _ask("Minimum similarity score (0.0 – 1.0)", default="0.5")
    try:
        score_threshold = float(threshold_str)
    except ValueError:
        score_threshold = 0.5

    # Optional section filter (Summary, KeyInsight, canonical_node::*, etc.)
    section_filter = None
    if _confirm("Filter by section?", default=False):
        section_filter = _ask(
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
    from bamboo.database.graph_database_client import GraphDatabaseClient

    console.print("\n[bold cyan]Query Knowledge Graph (Graph Database)[/bold cyan]")

    query_type = _ask(
        "Query type",
        choices=["error", "features"],
        default="error",
    )

    graph_db = GraphDatabaseClient()

    try:
        await graph_db.connect()

        if query_type == "error":
            error_name = _ask("Enter error message or keyword")
            results = await graph_db.find_causes(errors=[error_name], limit=10)
        else:
            features_str = _ask("Enter features (comma-separated)")
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


async def preview_log_filter_interactive():
    """Fetch a log and show how the filter reduces it."""
    from bamboo.utils.log_filters import (
        SOURCE_FILTER_REGISTRY,
        filter_log,
        filter_log_auto,
        source_name_for_task,
    )
    from bamboo.utils.panda_client import async_fetch_log_content, extract_log_urls

    console.print("\n[bold cyan]Preview Log Filter[/bold cyan]")

    # --- source ---
    source = _ask(
        "Log source",
        choices=["task-id", "url", "file"],
        default="task-id",
    )

    raw: str | None = None
    label = ""
    suggested_filter: str | None = None  # set when prodSourceLabel is known

    if source == "task-id":
        task_id_str = _ask("Enter PanDA jediTaskID")
        try:
            task_data = await fetch_task_data_for_filter(task_id_str)
        except Exception as e:
            console.print(f"[red]Error fetching task: {e}[/red]")
            return

        suggested_filter = source_name_for_task(
            task_data.get("prodSourceLabel", "") or ""
        )

        error_dialog = task_data.get("errorDialog", "") or ""
        log_urls = extract_log_urls(str(error_dialog))
        if not log_urls:
            console.print("[yellow]No log URLs found in errorDialog.[/yellow]")
            return

        if len(log_urls) == 1:
            chosen_url = log_urls[0]
        else:
            console.print("\n[bold]Available log URLs:[/bold]")
            for i, u in enumerate(log_urls, 1):
                console.print(f"  {i}. {u}")
            idx_str = _ask(
                "Select URL number",
                choices=[str(i) for i in range(1, len(log_urls) + 1)],
                default="1",
            )
            chosen_url = log_urls[int(idx_str) - 1]

        label = chosen_url
        with console.status(f"[bold green]Fetching {chosen_url} ..."):
            raw = await async_fetch_log_content(chosen_url)
    elif source == "url":
        url = _ask("Enter log URL")
        label = url
        with console.status(f"[bold green]Fetching {url} ..."):
            raw = await async_fetch_log_content(url)
    else:  # file
        path = _ask("Enter path to log file")
        label = path
        try:
            with open(path) as f:
                raw = f.read()
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            return

    if not raw:
        console.print("[yellow]Could not retrieve log content.[/yellow]")
        return

    # --- filter choice ---
    filter_choices = ["auto", "generic"] + list(SOURCE_FILTER_REGISTRY.keys())
    default_idx = (
        str(filter_choices.index(suggested_filter) + 1)
        if suggested_filter and suggested_filter in filter_choices
        else "1"
    )
    console.print("\n[bold]Available filters:[/bold]")
    for i, name in enumerate(filter_choices, 1):
        suffix = (
            "  [dim](try each specialised filter, fall back to generic)[/dim]"
            if name == "auto"
            else ""
        )
        suffix = (
            "  [dim](head + tail + signal lines)[/dim]" if name == "generic" else suffix
        )
        tag = "  [green](suggested)[/green]" if name == suggested_filter else ""
        console.print(f"  {i}. {name}{suffix}{tag}")

    idx_str = _ask(
        "Select filter",
        choices=[str(i) for i in range(1, len(filter_choices) + 1)],
        default=default_idx,
    )
    chosen_filter = filter_choices[int(idx_str) - 1]

    with console.status("[bold green]Filtering..."):
        if chosen_filter == "auto":
            filtered = filter_log_auto(raw)
        elif chosen_filter == "generic":
            filtered = filter_log(raw)
        else:
            filtered = SOURCE_FILTER_REGISTRY[chosen_filter](raw) or filter_log(raw)

    raw_lines = raw.count("\n")
    filtered_lines = filtered.count("\n")
    reduction = 100 * (1 - filtered_lines / max(raw_lines, 1))

    console.print(
        f"\n[dim]── {label} ──[/dim]\n"
        f"[bold]Filter:[/bold] {chosen_filter}  "
        f"[bold]Original:[/bold] {raw_lines} lines  "
        f"[bold]Filtered:[/bold] {filtered_lines} lines  "
        f"[bold]Reduction:[/bold] {reduction:.0f}%\n"
    )
    console.print(
        Panel(filtered, title=f"Filtered log [{chosen_filter}]", border_style="green")
    )

    if _confirm("\nShow raw log?", default=False):
        console.print(Panel(raw, title="Raw log", border_style="dim"))


async def fetch_task_data_for_filter(task_id_str: str) -> dict:
    """Thin wrapper so preview_log_filter_interactive can fetch task data."""
    from bamboo.utils.panda_client import fetch_task_data

    with console.status(f"[bold green]Fetching task {task_id_str} from PanDA..."):
        return await fetch_task_data(int(task_id_str))


async def cleanup_databases_interactive():
    """Interactive database cleanup — lets the user wipe graph DB, vector DB, or both."""
    from bamboo.database.graph_database_client import GraphDatabaseClient
    from bamboo.database.vector_database_client import VectorDatabaseClient

    console.print("\n[bold red]Cleanup Databases[/bold red]")
    console.print(
        "[yellow]Warning: this permanently deletes data and cannot be undone.[/yellow]"
    )

    target = _ask(
        "What do you want to clear?",
        choices=["graph", "vector", "both", "cancel"],
        default="cancel",
    )

    if target == "cancel":
        console.print("[dim]Cancelled.[/dim]")
        return

    confirmed = _confirm(
        f"Are you sure you want to clear the {target} database(s)?",
        default=False,
    )
    if not confirmed:
        console.print("[dim]Cancelled.[/dim]")
        return

    if target in ("graph", "both"):
        graph_db = GraphDatabaseClient()
        try:
            await graph_db.connect()
            await graph_db.clear_all()
            console.print("[green]Graph database cleared.[/green]")
        except Exception as e:
            console.print(f"[red]Graph DB error: {e}[/red]")
        finally:
            await graph_db.close()

    if target in ("vector", "both"):
        vector_db = VectorDatabaseClient()
        try:
            await vector_db.connect()
            await vector_db.clear_all()
            console.print("[green]Vector database cleared.[/green]")
        except Exception as e:
            console.print(f"[red]Vector DB error: {e}[/red]")
        finally:
            await vector_db.close()


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


@cli.command("extract")
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
def extract_cmd(email_thread, task_data, task_id, external_data, output, verbose):
    """Extract knowledge graph and preview it without writing to any database.

    Runs the full extraction pipeline (LLM calls, error classification, graph
    construction) and prints a summary of what *would* be stored by
    ``bamboo populate``.  Nothing is written to Neo4j or Qdrant.

    Examples:

    \b
      bamboo extract --task-id 12345
      bamboo extract --task-data task.json --email-thread email.txt
      bamboo extract --task-data task.json --output graph_preview.json
      bamboo extract --task-id 12345 --verbose
    """
    from bamboo.scripts.extract_knowledge import main as _main

    ctx = click.get_current_context()
    ctx.invoke(
        _main,
        email_thread=email_thread,
        task_data=task_data,
        task_id=task_id,
        external_data=external_data,
        output=output,
        verbose=verbose,
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
