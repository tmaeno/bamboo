"""Unit tests for the shared `build_deps` factory and the IO-driven interactive path.

Covers:
- `build_deps()` returns a fully-populated `_Deps` (offline construction) with a
  `source_navigator`-equipped explorer — i.e. the `extractor=` crash is gone.
- `InteractionIO.supports_interaction` per frontend.
- `request_human_input` routes through the active IO (chat/terminal) when present,
  and falls back to stdin otherwise.
- `ContextEnricher` offers/hides interactive tools based on the IO's interactivity.
"""

from __future__ import annotations

import builtins

import pytest

from bamboo.agents.context_enricher import ContextEnricher
from bamboo.agents.deps import build_deps
from bamboo.mcp.interactive_mcp_client import InteractiveMcpClient


class _IO:
    """Minimal InteractionIO stand-in: declares interactivity and answers `ask`."""

    def __init__(self, *, interactive: bool = True, answer: str = "answer") -> None:
        self._interactive = interactive
        self._answer = answer

    @property
    def supports_interaction(self) -> bool:
        return self._interactive

    async def ask(self, prompt, *, default=None, choices=None) -> str:
        return self._answer


# ---------------------------------------------------------------------------
# build_deps
# ---------------------------------------------------------------------------


def test_build_deps_returns_full_bundle():
    deps = build_deps(io=_IO())
    # No TypeError, and every collaborator the orchestrator needs is present.
    assert deps.mcp_client is not None
    assert deps.graph_db is not None and deps.vector_db is not None
    assert deps.reasoning_navigator is not None
    assert deps.knowledge_accumulator is not None
    assert deps.error_classifier is not None
    assert deps.extractor is not None  # active extraction strategy
    assert deps.io is not None
    # Explorer carries the strategy's source_navigator (default "panda" → non-None).
    assert deps.reasoning_navigator._explorer._source_navigator is not None


# ---------------------------------------------------------------------------
# supports_interaction
# ---------------------------------------------------------------------------


def test_cli_supports_interaction_follows_tty(monkeypatch):
    import bamboo.frontends.cli as climod

    class _Out:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(climod.sys, "stdout", _Out(True))
    assert climod.CliInteractionIO().supports_interaction is True
    monkeypatch.setattr(climod.sys, "stdout", _Out(False))
    assert climod.CliInteractionIO().supports_interaction is False


def test_mattermost_supports_interaction_is_true():
    from bamboo.frontends.mattermost.io import MattermostInteractionIO, ThreadTransport

    class _T(ThreadTransport):
        def send(self, text, *, props=None):  # pragma: no cover - trivial
            pass

        async def next_reply(self):  # pragma: no cover - trivial
            return ""

    assert MattermostInteractionIO(_T()).supports_interaction is True


# ---------------------------------------------------------------------------
# InteractiveMcpClient routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_human_input_routes_through_io():
    client = InteractiveMcpClient(_IO(answer="  hello  "))
    out = await client.execute("request_human_input", prompt="what happened?")
    assert out == "hello"  # stripped, via io.ask — not stdin


@pytest.mark.asyncio
async def test_request_human_input_stdin_fallback_on_eof(monkeypatch):
    def _eof(*_a, **_k):
        raise EOFError

    monkeypatch.setattr(builtins, "input", _eof)
    client = InteractiveMcpClient(io=None)
    out = await client.execute("request_human_input", prompt="?")
    assert out == ""  # no IO → stdin path → EOF → empty string


# ---------------------------------------------------------------------------
# ContextEnricher interactive-tool gating
# ---------------------------------------------------------------------------


def test_context_enricher_offers_interactive_tool_when_io_interactive():
    enr = ContextEnricher(InteractiveMcpClient(), io=_IO(interactive=True))
    names = [t.name for t in enr._filtered_tools()]
    assert "request_human_input" in names


def test_context_enricher_hides_interactive_tool_when_io_not_interactive():
    enr = ContextEnricher(InteractiveMcpClient(), io=_IO(interactive=False))
    names = [t.name for t in enr._filtered_tools()]
    assert "request_human_input" not in names
