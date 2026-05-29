"""Readline-compatible interactive prompts shared between CLI and orchestrators.

Originally these lived inside :mod:`bamboo.cli`. They were extracted here so
:class:`bamboo.agents.investigation_session.InvestigationOrchestrator` can use
the same input helpers without importing from ``bamboo.cli`` (which itself
imports many heavy modules). ``bamboo.cli`` re-exports the same names for
back-compat.

Why custom helpers instead of ``rich.prompt.Prompt.ask`` / ``Confirm.ask``?
Rich's prompts disable readline integration on some terminals, which breaks
up/down arrow history recall. These wrappers call ``input(prompt)`` directly
so readline knows the cursor position and history works as users expect.
"""

from __future__ import annotations

import re

from rich.console import Console


# A module-level Console instance shared by all callers — keeps colour output
# consistent across the CLI and the orchestrator. Importers may also pass
# their own Console via the optional ``console`` keyword if they want to
# redirect output (e.g. for tests).
_default_console = Console()


_RICH_MARKUP_RE = re.compile(r"\[/?[^\]]*\]")


def _strip_markup(text: str) -> str:
    """Strip rich-style markup tags like ``[bold]`` for prompts that go through ``input()``."""
    return _RICH_MARKUP_RE.sub("", text)


def ask(
    prompt: str,
    *,
    default: str | None = None,
    choices: list[str] | None = None,
    console: Console | None = None,
) -> str:
    """Readline-compatible replacement for ``rich.prompt.Prompt.ask``.

    Calls ``input(prompt)`` directly so readline knows the cursor position and
    up/down arrow history recall works without garbling the prompt.

    Args:
        prompt:   Prompt text (may contain rich markup; stripped for display).
        default:  Returned when the user enters an empty line. When None, the
                  prompt re-asks instead of returning empty.
        choices:  Optional list of allowed answers; re-asks on mismatch.
        console:  Console used for re-ask error messages. Defaults to the
                  module-level shared instance.

    Returns:
        The user's answer (or ``default`` on empty input). Raises ``SystemExit``
        on EOF or Ctrl-C — same behavior as the original cli.py helper.
    """
    c = console or _default_console
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
            c.print(
                f"[yellow]  Please select one of: {', '.join(choices)}[/yellow]"
            )
            continue
        return answer


def confirm(
    prompt: str,
    *,
    default: bool | None = None,
    console: Console | None = None,
) -> bool:
    """Readline-compatible replacement for ``rich.prompt.Confirm.ask``.

    Args:
        prompt:   Prompt text (may contain rich markup; stripped for display).
        default:  Returned when the user enters an empty line. When None, the
                  prompt re-asks instead of returning a default.
        console:  Console used for re-ask error messages. Defaults to the
                  module-level shared instance.

    Returns:
        ``True`` for ``y``/``yes`` (case-insensitive), ``False`` for ``n``/``no``,
        or ``default`` on empty input. Raises ``SystemExit`` on EOF/Ctrl-C.
    """
    c = console or _default_console
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
        c.print("[yellow]  Please enter y or n.[/yellow]")
