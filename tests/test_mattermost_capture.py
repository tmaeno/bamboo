"""Unit tests for Phase 3 capture-from-thread.

No live server/client: the accumulator is mocked (``process_knowledge`` returns
an extracted graph; ``store_extracted`` records the commit), and the thread is a
fake transport with scripted replies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bamboo.frontends.mattermost.capture import build_email_text, run_capture
from bamboo.frontends.mattermost.io import MattermostInteractionIO, ThreadTransport
from bamboo.models.graph_element import CauseNode, ResolutionNode
from bamboo.models.knowledge_entity import ExtractedKnowledge, KnowledgeGraph


class FakeTransport(ThreadTransport):
    def __init__(self, replies):
        self.sent = []
        self._replies = iter(replies)

    def send(self, text, *, props=None):
        self.sent.append(text)

    async def next_reply(self):
        return next(self._replies)


def _accumulator_with_graph(nodes):
    graph = KnowledgeGraph(nodes=nodes, relationships=[])
    graph.metadata = {"graph_id": "graph:test"}
    acc = MagicMock()
    acc.process_knowledge = AsyncMock(
        return_value=ExtractedKnowledge(graph=graph, summary="sum", key_insights=[])
    )
    acc.store_extracted = AsyncMock(return_value=("sum", []))
    return acc


def _graph_db():
    db = MagicMock()
    db.get_node_description = AsyncMock(return_value=None)  # everything "new"
    return db


# ---------------------------------------------------------------------------
# build_email_text
# ---------------------------------------------------------------------------


def test_build_email_text_appends_cause_and_resolution():
    txt = build_email_text("ops discussed the OOM", "memory pressure", "bigger queue")
    assert "ops discussed the OOM" in txt
    assert "Cause: memory pressure" in txt
    assert "Resolution: bigger queue" in txt


def test_build_email_text_omits_blank_resolution():
    txt = build_email_text("transcript", "the cause", None)
    assert "Resolution:" not in txt


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_capture_extracts_reviews_and_stores_on_confirm():
    acc = _accumulator_with_graph([CauseNode(name="memory pressure", description="oom")])
    io = MattermostInteractionIO(FakeTransport(["memory pressure", "bigger queue", "yes"]))

    stored = await run_capture(
        io, transcript="we saw OOMs", task_id=None, accumulator=acc, graph_db=_graph_db()
    )

    assert stored is True
    # Extracted via dry-run with the transcript + ops summary.
    acc.process_knowledge.assert_awaited_once()
    kwargs = acc.process_knowledge.call_args.kwargs
    assert kwargs["dry_run"] is True
    assert "we saw OOMs" in kwargs["email_text"]
    assert "Cause: memory pressure" in kwargs["email_text"]
    # Committed via the shared path.
    acc.store_extracted.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_capture_aborts_when_declined():
    acc = _accumulator_with_graph([CauseNode(name="c", description="d")])
    io = MattermostInteractionIO(FakeTransport(["the cause", "", "no"]))

    stored = await run_capture(
        io, transcript="t", task_id=None, accumulator=acc, graph_db=_graph_db()
    )

    assert stored is False
    acc.store_extracted.assert_not_called()


@pytest.mark.asyncio
async def test_run_capture_handles_empty_extraction():
    acc = _accumulator_with_graph([])  # nothing extracted
    io = MattermostInteractionIO(FakeTransport(["cause", ""]))

    stored = await run_capture(
        io, transcript="t", task_id=None, accumulator=acc, graph_db=_graph_db()
    )

    assert stored is False
    acc.store_extracted.assert_not_called()


@pytest.mark.asyncio
async def test_run_capture_fetches_task_data_when_task_id_given(monkeypatch):
    acc = _accumulator_with_graph([ResolutionNode(name="r", description="d")])
    # Capture fetches via the shared seam (resolve_task_data), not an MCP tool.
    import bamboo.agents.deps as deps

    monkeypatch.setattr(
        deps,
        "resolve_task_data",
        AsyncMock(return_value={"jediTaskID": 42, "status": "broken"}),
    )
    io = MattermostInteractionIO(FakeTransport(["the cause", "the fix", "yes"]))

    await run_capture(io, transcript="t", task_id=42, accumulator=acc, graph_db=_graph_db())

    deps.resolve_task_data.assert_awaited_once_with(42)
    # task_data flowed into extraction.
    assert acc.process_knowledge.call_args.kwargs["task_data"] == {
        "jediTaskID": 42,
        "status": "broken",
    }


@pytest.mark.asyncio
async def test_run_capture_continues_when_task_data_fetch_fails(monkeypatch):
    acc = _accumulator_with_graph([CauseNode(name="c", description="d")])
    import bamboo.agents.deps as deps

    monkeypatch.setattr(
        deps, "resolve_task_data", AsyncMock(side_effect=RuntimeError("boom"))
    )
    io = MattermostInteractionIO(FakeTransport(["the cause", "", "yes"]))

    stored = await run_capture(
        io, transcript="t", task_id=42, accumulator=acc, graph_db=_graph_db()
    )

    # Fetch failed → a notice is posted and capture continues with task_data=None.
    assert any("Could not fetch task_data" in s for s in io.transport.sent)
    assert acc.process_knowledge.call_args.kwargs["task_data"] is None
    assert stored is True
