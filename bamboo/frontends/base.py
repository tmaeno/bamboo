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
from dataclasses import dataclass


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


class InteractionIO(ABC):
    """Frontend-agnostic interaction surface for the interactive engines.

    A single instance is injected into an engine (via ``_Deps``) and used for
    every human turn: asking for input, confirming side-effecting actions,
    editing generated code, and rendering panels/tables/diffs.  One concrete
    subclass exists per frontend; a single instance serves the investigate,
    capture-from-thread, and analyze flows.
    """

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
