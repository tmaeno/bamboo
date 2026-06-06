"""Phase-1 code-execution trust model (docs/EXECUTION_TRUST.md).

Covers the cross-cutting runtime boundary and the read-only automatic phase:

* ``ToolProxy``/``run_orchestration_code`` ``allowed_tools`` refusal (alias-proof),
* per-tool DEBUG narration,
* ``ContextEnricher`` read-only generation + side-effect accessor,
* ``ReasoningNavigator`` skipping + suggesting side-effecting stored procedures.

The interactive review-and-policy gate lives in ``test_investigation_session.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from bamboo.agents.context_enricher import ContextEnricher, ExplorationResult
from bamboo.agents.orchestration import run_orchestration_code
from bamboo.agents.reasoning_navigator import ReasoningNavigator
from bamboo.mcp.base import McpTool


class _FakeClient:
    """Minimal MCP client: records executed tool names, echoes the name."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def task_data_tools(self):  # noqa: D401
        return frozenset()

    async def execute(self, name, **kwargs):
        self.calls.append(name)
        return {"ran": name}


class _NarrationCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# ---------------------------------------------------------------------------
# Runtime allow-set boundary (alias-proof)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowlist_refuses_disallowed_tool_including_alias():
    """A side-effecting tool reached via an alias is refused at the call site."""
    client = _FakeClient()
    code = "m = tools.kill_job\nreturn await m()"  # aliased — static analysis would miss it
    result, call_log = await run_orchestration_code(
        code,
        client=client,
        task_data={},
        task_data_tool_names=frozenset(),
        allowed_tools=frozenset({"get_status"}),
    )
    assert result == {}  # fail-closed
    assert "kill_job" not in client.calls  # never dispatched
    assert call_log == []


@pytest.mark.asyncio
async def test_allowlist_permits_listed_tool():
    client = _FakeClient()
    result, call_log = await run_orchestration_code(
        "return await tools.get_status()",
        client=client,
        task_data={},
        task_data_tool_names=frozenset(),
        allowed_tools=frozenset({"get_status"}),
    )
    assert result == {"ran": "get_status"}
    assert client.calls == ["get_status"]


@pytest.mark.asyncio
async def test_no_allowlist_permits_anything():
    """Interactive callers pass allowed_tools=None → no restriction (gated by review)."""
    client = _FakeClient()
    result, _ = await run_orchestration_code(
        "return await tools.anything()",
        client=client,
        task_data={},
        task_data_tool_names=frozenset(),
        allowed_tools=None,
    )
    assert result == {"ran": "anything"}


@pytest.mark.asyncio
async def test_per_tool_debug_narration():
    lg = logging.getLogger("bamboo.narration")
    cap = _NarrationCapture()
    lg.addHandler(cap)
    old_level = lg.level
    lg.setLevel(logging.DEBUG)
    try:
        await run_orchestration_code(
            "return await tools.get_status(task_id=5)",
            client=_FakeClient(),
            task_data={},
            task_data_tool_names=frozenset(),
        )
    finally:
        lg.removeHandler(cap)
        lg.setLevel(old_level)
    assert any("→ get_status(" in m and "task_id=5" in m for m in cap.messages)
    assert any("↳ get_status returned" in m for m in cap.messages)


# ---------------------------------------------------------------------------
# ContextEnricher: read-only by construction
# ---------------------------------------------------------------------------


class _ToolListClient:
    def __init__(self, tools: list[McpTool]) -> None:
        self._tools = tools

    def list_tools(self) -> list[McpTool]:
        return self._tools


def _mixed_tools() -> list[McpTool]:
    # read_a: external PanDA read (read_only=True, external_access=True — both defaults).
    # write_b: state-changing (read_only=False) — should be excluded from automatic phases.
    return [
        McpTool(name="read_a", description="d", parameters_schema={"type": "object"}),
        McpTool(name="write_b", description="d", parameters_schema={"type": "object"}, read_only=False),
    ]


def test_context_enricher_filtered_tools_keeps_reads_drops_state_changing():
    enr = ContextEnricher(_ToolListClient(_mixed_tools()), io=None)
    names = [t.name for t in enr._filtered_tools()]
    assert "read_a" in names  # external PanDA read is allowed in automatic phases
    assert "write_b" not in names  # state-changing tool excluded


def test_context_enricher_non_read_only_tool_names():
    enr = ContextEnricher(_ToolListClient(_mixed_tools()), io=None)
    assert enr.non_read_only_tool_names() == frozenset({"write_b"})


# ---------------------------------------------------------------------------
# ReasoningNavigator: skip + suggest state-changing stored procedures
# ---------------------------------------------------------------------------


class _FakeExplorer:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def non_read_only_tool_names(self) -> frozenset[str]:
        return frozenset({"kill_job"})

    async def run_stored_code(self, code, task_data):
        self.ran.append(code)
        return {"data": 1}, ["get_status"]

    async def explore(self, *args, **kwargs):
        return ExplorationResult()


@pytest.mark.asyncio
async def test_navigator_skips_and_suggests_side_effecting_stored_procedure():
    explorer = _FakeExplorer()
    nav = ReasoningNavigator(graph_db=MagicMock(), vector_db=MagicMock(), explorer=explorer)
    procedures = [
        {
            "strategy_type": "read_only_check",
            "cause_name": "c1",
            "code_summary": "look",
            "orchestration_code": "return await tools.get_status()",
            "procedure_name": "p1",
            "frequency": 1,
        },
        {
            "strategy_type": "kill_the_jobs",
            "cause_name": "c2",
            "code_summary": "kill",
            "orchestration_code": "return await tools.kill_job(job_id=1)",
            "procedure_name": "p2",
            "frequency": 1,
        },
    ]
    result = await nav._run_investigation({"jediTaskID": 1}, procedures)

    # Only the read-only procedure was replayed; the side-effecting one was not.
    assert len(explorer.ran) == 1
    assert "get_status" in explorer.ran[0]
    assert all("kill_job" not in c for c in explorer.ran)
    # The side-effecting procedure is surfaced as a suggestion.
    note = result["investigation_note"]
    assert "kill_the_jobs" in note
    assert "suggest" in note.lower()
