"""Unit tests for the in-chat ``analyze`` command.

No live server/PanDA: the task-data fetch is injected, and ``deps`` is a mock
whose ``reasoning_navigator.analyze_task`` returns a stub ``AnalysisResult``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bamboo.frontends.mattermost.analyze import run_analyze
from bamboo.frontends.mattermost.io import MattermostInteractionIO, ThreadTransport
from bamboo.models.knowledge_entity import AnalysisResult


class _CaptureTransport(ThreadTransport):
    def __init__(self):
        self.sent = []
        self.props = []

    def send(self, text, *, props=None):
        self.sent.append(text)
        self.props.append(props)

    async def next_reply(self):  # pragma: no cover - analyze never reads replies
        return ""


def _io():
    return MattermostInteractionIO(_CaptureTransport())


def _deps(result=None, analyze_exc=None):
    deps = MagicMock()
    deps.graph_db.connect = AsyncMock()
    deps.vector_db.connect = AsyncMock()
    deps.mcp_client.connect = AsyncMock()
    if analyze_exc is not None:
        deps.reasoning_navigator.analyze_task = AsyncMock(side_effect=analyze_exc)
    else:
        deps.reasoning_navigator.analyze_task = AsyncMock(return_value=result)
    return deps


def _result(task_id="12345"):
    return AnalysisResult(
        task_id=task_id,
        root_cause="scout jobs OOMed",
        confidence=0.8,
        resolution="resubmit to high-memory queue",
        explanation="the scout phase failed under memory pressure",
    )


@pytest.mark.asyncio
async def test_run_analyze_posts_result_card():
    io = _io()
    deps = _deps(result=_result("12345"))

    ok = await run_analyze(io, task_id=12345, deps=deps, fetch=AsyncMock(return_value={"jediTaskID": 12345}))

    assert ok is True
    deps.reasoning_navigator.analyze_task.assert_awaited_once()
    # An attachment card with the task id in its title was posted.
    attachments = [
        p["attachments"][0] for p in io.transport.props if isinstance(p, dict) and "attachments" in p
    ]
    assert len(attachments) == 1
    assert "12345" in attachments[0]["title"]


@pytest.mark.asyncio
async def test_run_analyze_requires_task_id():
    io = _io()
    deps = _deps(result=_result())

    ok = await run_analyze(io, task_id=None, deps=deps, fetch=AsyncMock())

    assert ok is False
    deps.reasoning_navigator.analyze_task.assert_not_awaited()
    assert any("Usage" in t for t in io.transport.sent)


@pytest.mark.asyncio
async def test_run_analyze_handles_fetch_error():
    io = _io()
    deps = _deps(result=_result())

    ok = await run_analyze(
        io, task_id=999, deps=deps, fetch=AsyncMock(side_effect=RuntimeError("boom"))
    )

    assert ok is False
    deps.reasoning_navigator.analyze_task.assert_not_awaited()
    assert any("Could not fetch task data" in t for t in io.transport.sent)


@pytest.mark.asyncio
async def test_run_analyze_handles_analysis_error():
    io = _io()
    deps = _deps(analyze_exc=RuntimeError("kaboom"))

    ok = await run_analyze(
        io, task_id=42, deps=deps, fetch=AsyncMock(return_value={"jediTaskID": 42})
    )

    assert ok is False
    assert any("Analysis failed" in t for t in io.transport.sent)
    # No attachment card was posted on failure.
    assert all(not (isinstance(p, dict) and "attachments" in p) for p in io.transport.props)
