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
from bamboo.agents.orchestration import referenced_tool_names, run_orchestration_code
from bamboo.agents.procedure_tools import (
    build_procedure_tools_registry,
    procedure_signature,
    procedure_tool_name,
)
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


# ---------------------------------------------------------------------------
# Phase 2a: procedure signature/identity + reusable procedure-tools
# ---------------------------------------------------------------------------


def test_referenced_tool_names():
    assert referenced_tool_names("a=await tools.get_jobs()\nb=await tools.fetch_logs()\nreturn {}") == frozenset(
        {"get_jobs", "fetch_logs"}
    )
    assert referenced_tool_names("return await tools.get_status()") == frozenset({"get_status"})
    # aliased (non-call attribute access) is still seen
    assert referenced_tool_names("m = tools.kill_job\nreturn await m()") == frozenset({"kill_job"})
    assert referenced_tool_names("def (:") == frozenset()  # unparseable → empty


def test_procedure_signature_and_name():
    code = "a=await tools.get_jobs()\nb=await tools.fetch_logs()\nreturn {}"
    sig = procedure_signature(code)
    assert sig == ["fetch_logs", "get_jobs"]  # sorted
    name = procedure_tool_name(sig)
    assert name == "proc__fetch_logs__get_jobs"
    assert name.isidentifier()  # callable as tools.<name>
    # Same tools (any order/phrasing) → same identity; different tools → different.
    assert procedure_tool_name(procedure_signature("b=await tools.fetch_logs()\nreturn await tools.get_jobs()")) == name
    assert procedure_tool_name(["get_jobs"]) != name
    assert procedure_tool_name([]) == ""  # no-tool block → caller falls back to legacy naming


_PROCS = [
    {  # non-trivial, read-only → exposed
        "tool_name": "proc__fetch_logs__get_jobs",
        "signature": ["fetch_logs", "get_jobs"],
        "orchestration_code": "a=await tools.get_jobs(task_id=task_id)\nb=await tools.fetch_logs()\nreturn {'a': a, 'b': b}",
        "strategy_type": "check failed jobs",
        "code_summary": "fetch jobs + logs",
        "external_access": True,
        "cause_names": ["memory pressure"],
    },
    {  # single-tool → skipped (raw tool covers it)
        "tool_name": "proc__get_jobs",
        "signature": ["get_jobs"],
        "orchestration_code": "return await tools.get_jobs()",
        "external_access": True,
        "cause_names": ["x"],
    },
    {  # state-changing → skipped
        "tool_name": "proc__get_jobs__kill_job",
        "signature": ["get_jobs", "kill_job"],
        "orchestration_code": "await tools.kill_job()\nreturn await tools.get_jobs()",
        "external_access": True,
        "cause_names": ["y"],
    },
]


def test_build_procedure_tools_registry_filters():
    client = _FakeClient()
    descs, calls = build_procedure_tools_registry(
        _PROCS,
        client=client,
        task_data={"jediTaskID": 42},
        task_id=42,
        task_data_tool_names=frozenset(),
        non_read_only_tool_names=frozenset({"kill_job"}),
    )
    assert set(descs) == {"proc__fetch_logs__get_jobs"}  # only non-trivial + read-only
    d = descs["proc__fetch_logs__get_jobs"]
    assert d.read_only is True and d.external_access is True
    assert "fetch_logs" in d.description and "get_jobs" in d.description


@pytest.mark.asyncio
async def test_procedure_tool_callable_runs_via_sandbox():
    client = _FakeClient()
    _descs, calls = build_procedure_tools_registry(
        _PROCS[:1],
        client=client,
        task_data={"jediTaskID": 42},
        task_id=42,
        task_data_tool_names=frozenset(),
        non_read_only_tool_names=frozenset({"kill_job"}),
    )
    result = await calls["proc__fetch_logs__get_jobs"]()
    # The stored code ran through the sandbox: both tools were dispatched, task_id injected.
    assert client.calls == ["get_jobs", "fetch_logs"]
    assert result["a"] == {"ran": "get_jobs"}


# ---------------------------------------------------------------------------
# Phase 2b: durable auto_run surfacing + allow_mutating exposure
# ---------------------------------------------------------------------------


def test_registry_surfaces_auto_run_and_respects_allow_mutating():
    procs = [
        {  # read-only, durably granted
            "tool_name": "proc__a__b",
            "signature": ["a", "b"],
            "orchestration_code": "x=await tools.a()\ny=await tools.b()\nreturn {}",
            "external_access": True,
            "auto_run": True,
            "cause_names": ["c"],
        },
        {  # state-changing (calls kill_job) + granted
            "tool_name": "proc__get__kill_job",
            "signature": ["get", "kill_job"],
            "orchestration_code": "await tools.kill_job()\nreturn await tools.get()",
            "external_access": True,
            "auto_run": True,
            "cause_names": ["c"],
        },
    ]
    common = {
        "client": _FakeClient(),
        "task_data": {},
        "task_id": None,
        "task_data_tool_names": frozenset(),
        "non_read_only_tool_names": frozenset({"kill_job"}),
    }
    # Default: state-changing excluded; auto_run surfaced on the read-only one.
    d, _ = build_procedure_tools_registry(procs, **common)
    assert set(d) == {"proc__a__b"}
    assert d["proc__a__b"].metadata["auto_run"] is True
    assert d["proc__a__b"].read_only is True
    # allow_mutating: state-changing procedure also exposed, flagged read_only=False.
    d2, _ = build_procedure_tools_registry(procs, allow_mutating=True, **common)
    assert set(d2) == {"proc__a__b", "proc__get__kill_job"}
    assert d2["proc__get__kill_job"].read_only is False
