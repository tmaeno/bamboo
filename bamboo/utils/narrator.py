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
from contextlib import contextmanager
from contextvars import ContextVar, Token
import time
from typing import Generator
import threading

from rich.live import Live
from rich.spinner import Spinner
from rich.console import Console
from rich.text import Text
from rich.columns import Columns

_console: ContextVar[Console | None] = ContextVar("narrator_console", default=None)
_verbose: ContextVar[bool] = ContextVar("narrator_verbose", default=False)


def set_narrator(console: Console, verbose: bool = False) -> Token:
    """Activate *console* as the narrator for the current task context.

    When *verbose* is ``True``, :func:`say` prints messages.  The spinner
    from :func:`thinking` is always active regardless of *verbose*.

    Returns the :class:`~contextvars.Token` so the caller can restore the
    previous state with ``_console.reset(token)`` if needed.
    """
    _verbose.set(verbose)
    return _console.set(console)


def say(msg: str) -> None:
    """Print *msg* to the narrator console (no-op when none is set or not verbose)."""
    if _verbose.get():
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

    The label is prefixed with the caller's module name, e.g.
    ``[knowledge_accumulator] Summarizing the graph...``

    No-op when no narrator console is active.
    """
    c = _console.get()
    if c is None:
        yield
        return
    caller = inspect.stack()[2]
    module = inspect.getmodule(caller[0])
    module_name = module.__name__.rsplit(".", 1)[-1] if module else "?"

    spinner = Spinner("dots")
    stop_event = threading.Event()

    # This function runs in a background thread to update the spinner label with animated dots
    def update_loop(live):
        n_dots = 10
        counter = 0
        while not stop_event.is_set():
            dots = "." * (counter % n_dots + 1)
            label = Text.from_markup(
                f"[dim cyan]\\[{module_name}][/dim cyan] [cyan]{msg}{dots}[/cyan]"
            )
            live.update(Columns([spinner, label]))
            counter += 1
            time.sleep(0.3)  # Faster dots feel more responsive

    with Live("", refresh_per_second=10, transient=True) as live:
        # Start the "dots" animation in the background
        thread = threading.Thread(target=update_loop, args=(live,), daemon=True)
        thread.start()

        try:
            yield  # The code inside the 'with' block runs HERE
        finally:
            # Tell the background thread to stop when the 'with' block finishes
            stop_event.set()
            thread.join(timeout=0.1)
