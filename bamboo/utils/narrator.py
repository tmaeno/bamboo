"""Lightweight narrative output for verbose agent-style pipeline tracing.

Usage::

    from bamboo.utils.narrator import say, set_narrator, thinking

    # At the CLI / script entry point — pass a Rich Console:
    set_narrator(Console())

    # Plain message:
    say("Found a link to a log. Downloading...")

    # Spinner while waiting for something slow (e.g. an LLM call):
    with thinking("Generating summary..."):
        result = await llm.ainvoke(...)

A ``ContextVar`` is used so that the narrator is scoped to an asyncio task /
coroutine tree; multiple parallel extractions can have independent narrators
without any global state.
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.highlighter import RegexHighlighter
from rich.theme import Theme

_console: ContextVar[Console | None] = ContextVar("narrator_console", default=None)
_verbose: ContextVar[bool] = ContextVar("narrator_verbose", default=False)

# Shared Progress instance for all concurrent thinking() callers.
# _progress_lock is only held briefly (no await inside), so no deadlock risk.
_progress: Progress | None = None
_progress_ref_count: int = 0
_progress_lock = threading.Lock()

# custom highlighter
class CustomHighlighter(RegexHighlighter):
    base_style = "custom."
    highlights = [r"(?P<number>(?<=[ (:/])\d+\.?\d*|^\d+\.?\d*)",
                  r"(?P<string>\"[^\"]*\"|'[^']*')"]


def set_narrator(console: Console, verbose: bool = False) -> Token:
    """Activate *console* as the narrator for the current task context.

    When *verbose* is ``True``, :func:`say` prints messages.  The spinner
    from :func:`thinking` is always active regardless of *verbose*.

    Returns the :class:`~contextvars.Token` so the caller can restore the
    previous state with ``_console.reset(token)`` if needed.
    """
    _verbose.set(verbose)
    custom_theme = Theme({"custom.number": "bold magenta", "custom.string": "bold green"})
    console.highlighter = CustomHighlighter()
    console.push_theme(custom_theme)
    return _console.set(console)


def say(msg: str) -> None:
    """Print *msg* to the narrator console (no-op when none is set or not verbose)."""
    if _verbose.get():
        c = _console.get()
        if c is not None:
            c.print(f"[dim cyan]  →[/dim cyan] {msg}")


def show_block(title: str, content: str, max_lines: int = 60) -> None:
    """Print *content* inside a labelled panel (no-op when not verbose).

    Long content is truncated to *max_lines* lines with a note showing how
    many lines were omitted, so the terminal is not flooded.

    Args:
        title:     Panel border title, e.g. ``"brokerage_log (filtered)"``.
        content:   The text to display.
        max_lines: Maximum lines before truncation (default 60).
    """
    if not _verbose.get():
        return
    c = _console.get()
    if c is None:
        return
    lines = content.splitlines()
    n_lines = len(lines)
    if n_lines > max_lines:
        body = (
            "\n".join(lines[:max_lines])
            + f"\n[dim]... {n_lines - max_lines} more lines omitted ...[/dim]"
        )
    else:
        body = "\n".join(lines)
    c.print(
        Panel(body, title=f"[bold cyan]{title}[/bold cyan]", border_style="dim cyan")
    )


@contextmanager
def thinking(msg: str) -> Generator[None, None, None]:
    """Show a spinner row labelled *msg* while a slow operation runs.

    Multiple concurrent callers each get their own row in a shared
    :class:`~rich.progress.Progress` display; all rows disappear together
    when the last one exits (``transient=True``).

    The label is prefixed with the caller's module name, e.g.
    ``[knowledge_accumulator] Summarizing the graph…``

    No-op when no narrator console is active.
    """
    global _progress, _progress_ref_count

    c = _console.get()
    if c is None:
        yield
        return

    caller = inspect.stack()[2]
    module = inspect.getmodule(caller[0])
    module_name = module.__name__.rsplit(".", 1)[-1] if module else "?"
    description = f"[dim cyan]\\[{module_name}][/dim cyan] [cyan]{msg}[/cyan]"

    with _progress_lock:
        if _progress is None:
            _progress = Progress(
                SpinnerColumn("dots"),
                TextColumn("{task.description}"),
                console=c,
                transient=True,
            )
            _progress.start()
        _progress_ref_count += 1
        task_id = _progress.add_task(description, total=None)

    try:
        yield
    finally:
        with _progress_lock:
            if _progress is not None:
                _progress.remove_task(task_id)
                _progress_ref_count -= 1
                if _progress_ref_count == 0:
                    _progress.stop()
                    _progress = None


@contextmanager
def counting(msg: str, total: int) -> Generator[Callable[[], None], None, None]:
    """Like :func:`thinking`, but yields an ``advance()`` callable that shows X/total progress.

    Usage::

        with counting("Summarising nodes", total=len(nodes)) as advance:
            for node in nodes:
                process(node)
                advance()

    No-op (yields a no-op callable) when no narrator console is active.
    """
    global _progress, _progress_ref_count

    c = _console.get()
    if c is None:
        yield lambda: None
        return

    caller = inspect.stack()[2]
    module = inspect.getmodule(caller[0])
    module_name = module.__name__.rsplit(".", 1)[-1] if module else "?"
    done = 0

    def _desc(n: int) -> str:
        return f"[dim cyan]\\[{module_name}][/dim cyan] [cyan]{msg} ({n}/{total})[/cyan]"

    with _progress_lock:
        if _progress is None:
            _progress = Progress(
                SpinnerColumn("dots"),
                TextColumn("{task.description}"),
                console=c,
                transient=True,
            )
            _progress.start()
        _progress_ref_count += 1
        task_id = _progress.add_task(_desc(0), total=None)

    def advance() -> None:
        nonlocal done
        done += 1
        with _progress_lock:
            if _progress is not None:
                _progress.update(task_id, description=_desc(done))

    try:
        yield advance
    finally:
        with _progress_lock:
            if _progress is not None:
                _progress.remove_task(task_id)
                _progress_ref_count -= 1
                if _progress_ref_count == 0:
                    _progress.stop()
                    _progress = None
