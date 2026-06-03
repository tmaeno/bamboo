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

Convention — narration vs. logging
----------------------------------
These functions emit on the ``bamboo.narration`` logger and are the
**operator-facing** progress channel: they render on the CLI Rich console *and*
are surfaced by interactive frontends (the Mattermost bot shows them in its live
post). Use them for things a person watching a run wants to see:

* ``say(msg)``                  — a progress milestone / result ("found 3 similar
                                  tasks", "fetched payload.stdout (2.5 MB)").
* ``thinking(msg)``/``counting``— the current step (drives a spinner / status head).
* ``warn(msg)`` / ``error(msg)``— an operator-significant warning / error (highlighted).
* ``show_block(title, body)``   — verbose detail (LLM prompts, logs); logged but
                                  **not** sent to chat.

Plain ``logging`` (``logger = logging.getLogger(__name__)``; ``logger.info`` /
``debug`` / ``warning`` …) is the **developer/diagnostic** channel — console/log
file only, never chat. Use it for low-level detail, per-item / hot-loop logs, and
internal diagnostics. Rule of thumb: if an operator should see it, narrate it;
otherwise ``logger.*`` it. Don't emit the same milestone via both (the CLI would
print it twice).
"""

from __future__ import annotations

import inspect
import logging
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

# Narration is emitted as records on this single logger — the one progress
# stream.  Interactive frontends (e.g. the Mattermost bot) attach a logging
# Handler to it; the console/file see the same records, so a frontend view is
# always a subset of the console.  ``extra={"narration_kind": ...}`` tags a
# record as a ``"step"`` (drives a frontend status head) or ``"block"`` (verbose
# detail kept out of chat); untagged records are ordinary progress lines.
_NARRATION_LOGGER = logging.getLogger("bamboo.narration")

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class _ConsoleLogHandler(logging.Handler):
    """Render log records through a Rich ``Console`` instead of raw stdout.

    Routing logging through the *same* Console as the ``thinking()`` spinner makes
    Rich coordinate the two (it clears the live spinner region, prints the line,
    then redraws), so log lines no longer corrupt an in-progress spinner.
    """

    def __init__(self, console: Console) -> None:
        super().__init__()
        self._console = console
        self.setFormatter(logging.Formatter(_LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._console.print(self.format(record), markup=False, highlight=False)
        except Exception:  # noqa: BLE001
            self.handleError(record)


def _route_logging_through_console(console: Console) -> None:
    """Make the root logger render through *console* (CLI), not raw stdout.

    Replaces the plain stdout/stderr ``StreamHandler``(s) installed by
    ``setup_logging`` with a single :class:`_ConsoleLogHandler` so all logging
    coordinates with the narrator's live spinner. Idempotent.
    """
    root = logging.getLogger()
    if any(isinstance(h, _ConsoleLogHandler) for h in root.handlers):
        return
    for h in list(root.handlers):
        # Drop the plain stream handlers (stdout/stderr); leave anything else
        # (e.g. pytest's capture handler, file handlers) intact.
        if type(h) is logging.StreamHandler:
            root.removeHandler(h)
    root.addHandler(_ConsoleLogHandler(console))


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
    # The Rich console is now the renderer for narration on this (CLI) process,
    # so keep the narration logger from *also* echoing to the root stdout handler
    # (which would double every line). The bot never sets a console, so there the
    # narration logger keeps propagating (→ server console) + its MM handler.
    _NARRATION_LOGGER.propagate = False
    # Route module logging through this Console too, so log lines coordinate with
    # the live thinking() spinner instead of corrupting it (raw stdout doesn't).
    _route_logging_through_console(console)
    return _console.set(console)


def say(msg: str, *, level: int = logging.INFO) -> None:
    """Emit a progress line on the narration logger; render on the CLI console.

    The record flows to the console/file (via logging) and to any installed
    frontend handler (e.g. Mattermost) that filters by level. ``level`` is a
    stdlib logging level (``logging.INFO`` for a milestone, ``logging.DEBUG`` for
    verbose detail). On a CLI with a Rich console it also pretty-prints when
    verbose.
    """
    _NARRATION_LOGGER.log(level, "%s", msg)
    if _verbose.get():
        c = _console.get()
        if c is not None:
            prefix = "[yellow]  ⚠[/yellow]" if level >= logging.WARNING else "[dim cyan]  →[/dim cyan]"
            c.print(f"{prefix} {msg}")


def warn(msg: str) -> None:
    """Narrate an operator-significant warning (WARNING level; surfaced + highlighted)."""
    say(msg, level=logging.WARNING)


def error(msg: str) -> None:
    """Narrate an operator-significant error (ERROR level; surfaced + highlighted)."""
    say(msg, level=logging.ERROR)


def show_block(title: str, content: str, max_lines: int = 60) -> None:
    """Log *content* as a verbose detail block; render as a panel on the CLI.

    Logged at INFO tagged ``narration_kind="block"`` so it appears in the
    console/file firehose but is **excluded** from chat frontends. Long content
    is truncated to *max_lines* lines on the CLI panel.

    Args:
        title:     Panel border title, e.g. ``"brokerage_log (filtered)"``.
        content:   The text to display.
        max_lines: Maximum lines before truncation (default 60).
    """
    _NARRATION_LOGGER.info("[%s]\n%s", title, content, extra={"narration_kind": "block"})
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

    caller = inspect.stack()[2]
    module = inspect.getmodule(caller[0])
    module_name = module.__name__.rsplit(".", 1)[-1] if module else "?"

    # A "step" record → frontends use it as the status head; console sees it too.
    _NARRATION_LOGGER.info("[%s] %s", module_name, msg, extra={"narration_kind": "step"})

    c = _console.get()
    if c is None:
        yield
        return

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

    caller = inspect.stack()[2]
    module = inspect.getmodule(caller[0])
    module_name = module.__name__.rsplit(".", 1)[-1] if module else "?"
    done = 0

    def _emit_step(n: int) -> None:
        _NARRATION_LOGGER.info(
            "[%s] %s (%d/%d)", module_name, msg, n, total, extra={"narration_kind": "step"}
        )

    _emit_step(0)

    c = _console.get()
    if c is None:
        # No Rich progress bar, but still emit step records as work advances.
        def advance_log() -> None:
            nonlocal done
            done += 1
            _emit_step(done)
        yield advance_log
        return

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
        _emit_step(done)
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
