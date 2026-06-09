"""Frontend interaction port — the contract between bamboo's interactive
engines and a concrete presentation surface (terminal, Mattermost, ...).

:class:`~bamboo.agents.investigation_session.InvestigationOrchestrator` and the
capture/analyze flows talk to the human exclusively through an
:class:`InteractionIO`.  Concrete frontends live alongside this module and follow
the same ``base.py`` + implementations + ``factory.py`` shape the repo already
uses for :mod:`bamboo.mcp` clients and :mod:`bamboo.database` backends:

* :class:`~bamboo.frontends.cli.CliInteractionIO` — Rich terminal (default).
* ``bamboo.frontends.mattermost`` — Mattermost bot (added later).

**Input** methods are ``async`` so a chat frontend can suspend a turn until the
next message arrives over its socket; the terminal frontend simply blocks on
``input()``.  **Output** methods are synchronous "render this now" calls — the
adapter decides whether that means printing to a console or posting a message.
Output strings may carry Rich-style markup (e.g. ``[bold]x[/bold]``); the
terminal adapter renders it, other adapters strip it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class DetailSink(Protocol):
    """A live region for one turn's verbose detail (streamed LLM output).

    Obtained from :meth:`InteractionIO.detail_stream`.  The engine feeds it the
    model's reasoning/answer deltas as they stream and a couple of summary lines
    (intent, strategy); each frontend renders it however suits — a transient
    terminal panel, a live-updating chat message, or nothing.

    ``active`` lets the caller skip the (slower, reasoning-enabled) streaming path
    entirely when the sink would render nothing — e.g. a non-verbose session.
    """

    active: bool

    def feed(self, text: str, *, reasoning: bool = False) -> None:
        """Append a streamed delta — answer text, or model reasoning when *reasoning*."""
        ...

    def meta(self, line: str) -> None:
        """Record a one-line summary fact for the turn (e.g. ``intent → tool``)."""
        ...


class _NullDetailSink:
    """A do-nothing :class:`DetailSink` — the default when a frontend doesn't render detail."""

    active = False

    def feed(self, text: str, *, reasoning: bool = False) -> None:  # noqa: D102
        pass

    def meta(self, line: str) -> None:  # noqa: D102
        pass


@asynccontextmanager
async def _null_detail_stream() -> AsyncIterator[DetailSink]:
    """Default :meth:`InteractionIO.detail_stream` — yields an inactive no-op sink."""
    yield _NullDetailSink()


@dataclass
class Column:
    """One column spec for :meth:`InteractionIO.table`.

    Attributes:
        header:  Column header text.
        justify: Cell justification — ``"left"`` (default), ``"right"`` or
                 ``"center"``.  Maps directly to Rich's ``Table.add_column``.
    """

    header: str
    justify: str = "left"


@dataclass(frozen=True)
class ReviewOption:
    """One choice in a :meth:`InteractionIO.review_orchestration` prompt.

    Attributes:
        key:   The canonical, human-readable answer (e.g. ``"run"``, ``"reject"``)
               — what the operator types and what the engine compares against.
        label: Short description shown next to the key (e.g. ``"run once"``).
        alias: Optional one-letter shortcut also accepted (e.g. ``"y"``).
    """

    key: str
    label: str
    alias: str | None = None


def match_choice(text: str, options: list[ReviewOption]) -> str | None:
    """Resolve a typed answer to a :class:`ReviewOption` key (case-insensitive).

    Matches the answer against each option's ``key`` or ``alias``. A blank answer
    selects the **last** option (the safe default — ``reject``). Returns the
    matched key, or ``None`` if the answer matches nothing.
    """
    s = (text or "").strip().lower()
    if not s:
        return options[-1].key if options else None
    for o in options:
        if s == o.key.lower() or (o.alias and s == o.alias.lower()):
            return o.key
    return None


class InteractionIO(ABC):
    """Frontend-agnostic interaction surface for the interactive engines.

    A single instance is injected into an engine (via ``_Deps``) and used for
    every human turn: asking for input, confirming side-effecting actions,
    editing generated code, and rendering panels/tables/diffs.  One concrete
    subclass exists per frontend; a single instance serves the investigate,
    capture-from-thread, and analyze flows.
    """

    @property
    def supports_interaction(self) -> bool:
        """Whether this frontend can prompt the operator and get a reply.

        Gates interactive MCP tools (e.g. ``request_human_input``).  Defaults to
        ``False``; concrete frontends override (terminal: a TTY; chat: always).
        """
        return False

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @abstractmethod
    async def ask(
        self,
        prompt: str,
        *,
        default: str | None = None,
        choices: list[str] | None = None,
    ) -> str:
        """Prompt the human for a line of text and return it.

        Args:
            prompt:  Prompt text (may contain Rich markup).
            default: Returned on empty input; when ``None`` the frontend re-asks
                     instead of returning empty.
            choices: Optional allowed answers; the frontend re-asks on mismatch.

        The terminal frontend raises ``SystemExit`` on EOF/Ctrl-C (preserving the
        CLI's existing abandon-on-interrupt behavior).
        """
        ...

    async def prompt_turn(self) -> str:
        """Get the operator's next free-form turn input (the main-loop prompt).

        Default delegates to :meth:`ask` with a ``>`` prompt — the terminal's
        recurring cue (and its ``SystemExit``-on-EOF behavior). A chat frontend
        overrides this to simply await the next message without posting a prompt
        (the session's one-time call-to-action already told the user to type).
        """
        return await self.ask("[bold cyan]>[/bold cyan]")

    @abstractmethod
    async def confirm(self, prompt: str, *, default: bool | None = None) -> bool:
        """Ask a yes/no question and return the boolean answer."""
        ...

    @abstractmethod
    async def edit(
        self,
        *,
        strategy_type: str,
        code: str,
        summary: str,
        triggers: list[str],
    ) -> tuple[str, str, str, list[str]]:
        """Let the human edit a proposed orchestration block.

        Returns the (possibly edited) ``(strategy_type, code, summary, triggers)``.
        The terminal frontend opens ``$EDITOR``; a chat frontend uses a modal.
        """
        ...

    # ------------------------------------------------------------------
    # Output (render-now)
    # ------------------------------------------------------------------

    @abstractmethod
    def notice(self, text: str) -> None:
        """Render a short status/info line (may contain Rich markup)."""
        ...

    @abstractmethod
    def panel(
        self,
        body: str,
        *,
        title: str | None = None,
        style: str | None = None,
        fit: bool = False,
    ) -> None:
        """Render a bordered panel of text.

        Args:
            body:  Panel body (may contain Rich markup).
            title: Optional panel title.
            style: Optional border style/color name.
            fit:   When ``True`` the terminal sizes the panel to its content
                   (Rich ``Panel.fit``); chat frontends ignore this.
        """
        ...

    @abstractmethod
    def code(self, code: str, *, lang: str = "python") -> None:
        """Render a syntax-highlighted code block."""
        ...

    @abstractmethod
    def table(self, *, title: str, columns: list[Column], rows: list[list[str]]) -> None:
        """Render a table.  Cell strings may contain Rich markup."""
        ...

    @abstractmethod
    def result(self, summary: str, *, title: str | None = None) -> None:
        """Render a tool/analysis result block."""
        ...

    @abstractmethod
    def diff(
        self,
        rows: list[tuple[str, str, str]],
        *,
        edge_count: int,
        edges: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Render a commit diff.

        Args:
            rows:       ``(node_type, name, action)`` triples where *action* is
                        ``"new"`` or ``"merge"``.
            edge_count: Number of relationships that will be written.
            edges:      Optional ``(source_name, target_name, relation_type)``
                        triples.  Frontends that draw a graph (e.g. Mermaid) use
                        them; the terminal frontend ignores them.
        """
        ...

    # ------------------------------------------------------------------
    # Streaming detail (live region for one turn's verbose detail)
    # ------------------------------------------------------------------

    def detail_stream(self, *, title: str):
        """Open a live region for one turn's verbose detail; an async context manager.

        Yields a :class:`DetailSink` the engine feeds streamed LLM reasoning/answer
        deltas and summary lines.  Concrete frontends override to render it (a
        transient terminal panel, a live-updating chat message, …).  The default
        is a no-op that yields an **inactive** sink, so frontends that don't render
        streaming detail — and test fakes — need no changes.

        Returns:
            An ``async with``-compatible context manager yielding a ``DetailSink``.
        """
        return _null_detail_stream()

    # ------------------------------------------------------------------
    # Higher-level interactions
    # ------------------------------------------------------------------

    async def review_orchestration(
        self,
        *,
        strategy_type: str,
        summary: str,
        triggers: list[str],
        code: str,
        options: list[ReviewOption],
        sink: DetailSink | None = None,
    ) -> str:
        """Show a proposed orchestration + its code, and return the chosen option key.

        Bundles the proposal display and the review prompt into one logical step so
        each frontend can render it optimally (a chat frontend can consolidate it
        into a single message, reusing *sink*'s live card). The default reproduces
        the terminal rendering — a ``proposed orchestration`` panel, the code, then
        an :meth:`ask` over the option keys — and **ignores** *sink*.

        Args:
            strategy_type/summary/triggers/code: the proposal to display.
            options: the allowed choices (key/label/alias); the last is the default.
            sink:    the turn's live detail sink, if any (used by chat frontends to
                     reuse the same message; ignored here).

        Returns:
            The chosen option ``key`` (e.g. ``"run"``, ``"edit"``, ``"reject"``).
        """
        triggers_str = "\n  - ".join(triggers) if triggers else "(none)"
        header = (
            f"[bold]strategy:[/bold]  {strategy_type}\n"
            f"[bold]summary:[/bold]  {summary or '(none)'}\n"
            f"[bold]trigger:[/bold]\n  - {triggers_str}"
        )
        self.panel(header, title="proposed orchestration", style="yellow")
        self.code(code, lang="python")
        prompt = "[bold]Review[/bold] — " + " / ".join(
            f"[{o.key}] {o.label}" for o in options
        )
        keys = [o.key for o in options]
        default = options[-1].key if options else None
        while True:
            answer = await self.ask(prompt, default=default, choices=None)
            choice = match_choice(answer, options)
            if choice is not None:
                return choice
            self.notice(f"[dim]Please answer one of: {', '.join(keys)}[/dim]")
