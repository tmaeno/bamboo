"""Terminal frontend — :class:`CliInteractionIO`.

The default :class:`~bamboo.frontends.base.InteractionIO` implementation, used by
the CLI commands.  Input goes through the readline-compatible helpers in
:mod:`bamboo.utils.prompts`; output reproduces the Rich ``Panel``/``Syntax``/
``Table`` rendering the orchestrator used to emit inline, so CLI behavior is
preserved byte-for-byte after the frontend refactor.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from bamboo.frontends.base import Column, InteractionIO
from bamboo.utils.prompts import ask as _ask, confirm as _confirm


def _parse_editor_buffer(
    text: str,
    *,
    fallback_strategy: str,
    fallback_summary: str,
    fallback_triggers: list[str],
) -> tuple[str, str, str, list[str]]:
    """Parse the editor temp-file format back into (strategy, code, summary, triggers).

    Header lines are ``# <key>: <value>`` (and ``#   - <item>`` for list items).
    Lines starting with ``## `` are treated as literal code comments and stay
    in the code body. The first non-header line terminates the header.
    """
    strategy = fallback_strategy
    summary = fallback_summary
    triggers: list[str] = []
    triggers_started = False
    body_lines: list[str] = []
    in_body = False

    for raw in text.splitlines():
        line = raw.rstrip()
        if not in_body:
            stripped = line.lstrip()
            # `## ` is a literal code comment: treat as body start signal.
            if stripped.startswith("## "):
                in_body = True
                body_lines.append(line)
                continue
            # Header lines start with a single '#' (but not '##').
            if stripped.startswith("#") and not stripped.startswith("##"):
                payload = stripped.lstrip("#").lstrip()
                if not payload:
                    # blank comment — keep parsing headers
                    continue
                if payload.startswith("strategy_type:"):
                    strategy = payload.split(":", 1)[1].strip() or fallback_strategy
                    triggers_started = False
                elif payload.startswith("code_summary:"):
                    summary = payload.split(":", 1)[1].strip()
                    triggers_started = False
                elif payload.startswith("trigger_signals"):
                    triggers_started = True
                elif payload.startswith("---"):
                    # Separator comment — keep skipping headers.
                    triggers_started = False
                elif payload.startswith("- ") and triggers_started:
                    item = payload[2:].strip()
                    if item:
                        triggers.append(item)
                # Unknown header — ignore.
                continue
            # First non-header, non-blank line → body.
            if line.strip() == "":
                # Blank lines before body are skipped.
                continue
            in_body = True
            body_lines.append(line)
        else:
            body_lines.append(line)

    code = "\n".join(body_lines).strip()
    if not triggers:
        triggers = list(fallback_triggers)
    return strategy, code, summary, triggers


class CliInteractionIO(InteractionIO):
    """Rich-terminal frontend.  Wraps a :class:`rich.console.Console`."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    @property
    def supports_interaction(self) -> bool:
        """Interactive only on a real terminal (not when piped / under cron)."""
        return sys.stdout.isatty()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    async def ask(
        self,
        prompt: str,
        *,
        default: str | None = None,
        choices: list[str] | None = None,
    ) -> str:
        # Offload the blocking ``input()`` to a thread so the event loop is not
        # stalled; ``SystemExit`` raised on EOF/Ctrl-C propagates through the
        # executor future, preserving the CLI's abandon-on-interrupt behavior.
        return await asyncio.to_thread(
            _ask, prompt, default=default, choices=choices, console=self.console
        )

    async def confirm(self, prompt: str, *, default: bool | None = None) -> bool:
        return await asyncio.to_thread(
            _confirm, prompt, default=default, console=self.console
        )

    async def edit(
        self,
        *,
        strategy_type: str,
        code: str,
        summary: str,
        triggers: list[str],
    ) -> tuple[str, str, str, list[str]]:
        return await asyncio.to_thread(
            self._edit_in_editor,
            strategy_type=strategy_type,
            code=code,
            summary=summary,
            triggers=triggers,
        )

    def _edit_in_editor(
        self,
        *,
        strategy_type: str,
        code: str,
        summary: str,
        triggers: list[str],
    ) -> tuple[str, str, str, list[str]]:
        """Open all five replayability fields in $EDITOR; return the edited values.

        Format of the temp file::

            # strategy_type: <slug>
            # code_summary: <one-line summary>
            # trigger_signals:
            #   - <signal 1>
            #   - <signal 2>
            # --- code below (lines starting with `#` are header metadata; `## ` is a literal comment) ---

            <python source>

        Header lines (starting with ``#`` but not ``## ``) are parsed back into
        the three structured fields; the remaining lines are the code body.
        This lets the human edit ``code_summary`` and ``trigger_signals``
        without re-running the LLM — important when the LLM's rationale is
        slightly off but the code itself is fine.

        Returns ``(strategy_type, code, code_summary, trigger_signals)``.
        Fields not present in the edited file fall back to the original
        values passed in.
        """
        import os
        import subprocess
        import tempfile

        editor = os.environ.get("EDITOR", "vi")
        trigger_lines = "\n".join(f"#   - {t}" for t in triggers) or "#   - "
        header = (
            f"# strategy_type: {strategy_type}\n"
            f"# code_summary: {summary}\n"
            f"# trigger_signals:\n"
            f"{trigger_lines}\n"
            "# --- code below (lines starting with `#` are header metadata; `## ` is a literal comment) ---\n\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as f:
            f.write(header + code)
            path = f.name
        try:
            subprocess.call([editor, path])
            with open(path) as f:
                edited = f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        return _parse_editor_buffer(
            edited,
            fallback_strategy=strategy_type,
            fallback_summary=summary,
            fallback_triggers=triggers,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def notice(self, text: str) -> None:
        self.console.print(text)

    def panel(
        self,
        body: str,
        *,
        title: str | None = None,
        style: str | None = None,
        fit: bool = False,
    ) -> None:
        kwargs: dict = {}
        if title is not None:
            kwargs["title"] = title
        if style is not None:
            kwargs["border_style"] = style
        panel = Panel.fit(body, **kwargs) if fit else Panel(body, **kwargs)
        self.console.print(panel)

    def code(self, code: str, *, lang: str = "python") -> None:
        self.console.print(Syntax(code, lang, theme="monokai", line_numbers=False))

    def table(self, *, title: str, columns: list[Column], rows: list[list[str]]) -> None:
        table = Table(title=title)
        for col in columns:
            table.add_column(col.header, justify=col.justify)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def result(self, summary: str, *, title: str | None = None) -> None:
        self.console.print(Panel(summary, title=title))

    def diff(
        self,
        rows: list[tuple[str, str, str]],
        *,
        edge_count: int,
        edges: list[tuple[str, str, str]] | None = None,
    ) -> None:
        # The terminal frontend renders the node table; *edges* (used for the
        # Mattermost Mermaid diagram) are not shown here.
        table = Table(
            title=f"commit diff ({len(rows)} node(s), {edge_count} relationship(s))"
        )
        table.add_column("type")
        table.add_column("name")
        table.add_column("action", justify="center")
        new_count = 0
        merge_count = 0
        for ntype, name, action in rows:
            if action == "new":
                new_count += 1
                table.add_row(ntype, name, "[green]new[/green]")
            else:
                merge_count += 1
                table.add_row(ntype, name, "[dim]merge[/dim]")
        self.console.print(table)
        self.console.print(
            f"[bold]summary:[/bold] {new_count} new, {merge_count} will merge — "
            f"{edge_count} edge(s) to write."
        )
