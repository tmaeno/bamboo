"""Standalone debug script for PandaSourceNavigator.

Usage:
    # Grep only — no LLM, instant, verify body search works:
    python -m bamboo.scripts.test_source_navigator "scout_ramCount" --grep-only

    # Full navigation with LLM:
    python -m bamboo.scripts.test_source_navigator "scout_ramCount threshold exhaustion"
"""

import asyncio
import os
import sys

import click
from rich.console import Console

from bamboo.agents.panda_source_navigator import (
    PandaSourceNavigator,
    _get_pkg_roots,
    _grep_candidates,
)
from bamboo.utils.narrator import set_narrator


@click.command()
@click.argument("query")
@click.option("--grep-only", is_flag=True, default=False,
              help="Run body grep with raw query words only (no LLM calls).")
def main(query: str, grep_only: bool) -> None:
    """Test PandaSourceNavigator with QUERY."""
    console = Console()
    set_narrator(console, verbose=True)

    pkg_roots = _get_pkg_roots()
    if not pkg_roots:
        console.print("[red]Neither pandaserver nor pandajedi is installed[/red]")
        sys.exit(1)

    console.print(f"[bold]query:[/bold] {query!r}")
    for pkg_name, pkg_root in pkg_roots.items():
        console.print(f"[bold]{pkg_name}:[/bold] {pkg_root}")

    if grep_only:
        terms = [w for w in query.split() if len(w) >= 4]
        console.print(f"\n[bold]grep terms (raw):[/bold] {terms}")
        hits = _grep_candidates(pkg_roots, terms)
        console.print(f"[bold]{len(hits)} candidate(s):[/bold]")
        for h in hits:
            console.print(f"  {h['module']}::{h['qualname']}  |  {h['docstring_first_line'][:80]}")
        os._exit(0)

    result = asyncio.run(_navigate(query))
    console.rule("result")
    console.print(result)
    os._exit(0)


async def _navigate(query: str) -> str:
    nav = PandaSourceNavigator()
    return await nav.navigate(query)


if __name__ == "__main__":
    main()
