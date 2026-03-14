"""Lightweight narrative output for verbose agent-style pipeline tracing.

Usage::

    from bamboo.utils.narrator import say, set_narrator

    # At the CLI / script entry point — set up the narrator once:
    token = set_narrator(lambda msg: print(f"  → {msg}"))

    # Deep inside the pipeline — just call say():
    say("Found a link to a log. Downloading...")   # printed only if narrator is active

A ``ContextVar`` is used so that the narrator is scoped to an asyncio task /
coroutine tree and multiple parallel extractions can have independent narrators
without any global state.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Callable

_narrator: ContextVar[Callable[[str], None] | None] = ContextVar(
    "narrator", default=None
)


def set_narrator(fn: Callable[[str], None]) -> Token:
    """Activate *fn* as the narrator for the current task context.

    Returns the :class:`~contextvars.Token` so the caller can restore the
    previous narrator with ``_narrator.reset(token)`` if needed.
    """
    return _narrator.set(fn)


def say(msg: str) -> None:
    """Emit *msg* via the active narrator (no-op when none is set)."""
    fn = _narrator.get()
    if fn is not None:
        fn(msg)
