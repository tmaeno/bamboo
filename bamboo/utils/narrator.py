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

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from rich.console import Console

_console: ContextVar[Console | None] = ContextVar("narrator_console", default=None)


def set_narrator(console: Console) -> Token:
    """Activate *console* as the narrator for the current task context.

    Returns the :class:`~contextvars.Token` so the caller can restore the
    previous state with ``_console.reset(token)`` if needed.
    """
    return _console.set(console)


def say(msg: str) -> None:
    """Print *msg* to the narrator console (no-op when none is set)."""
    c = _console.get()
    if c is not None:
        c.print(f"[dim cyan]  →[/dim cyan] {msg}")


@contextmanager
def thinking(msg: str) -> Generator[None, None, None]:
    """Show a spinner labelled *msg* while a slow operation runs.

    The spinner renders in a background thread so ``await`` inside the
    ``with`` block works normally::

        with thinking("Summarizing the graph..."):
            response = await llm.ainvoke(prompt)

    No-op when no narrator console is active.
    """
    c = _console.get()
    if c is None:
        yield
        return
    with c.status(f"[dim cyan]{msg}[/dim cyan]", spinner="dots"):
        yield
