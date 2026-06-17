"""Unit tests for `bamboo.agents.investigation_session`.

Covers the parts of the InvestigationOrchestrator that don't require a live
LLM / Neo4j / Qdrant / MCP server: the meta-command parser, the binary
intent-classification disambiguation gate, the static side-effects gate, the
narration-merge logic, the /undo snapshot, and the end-of-session edge
wiring. The LLM, MCP client, graph DB, and KnowledgeAccumulator are all
mocked.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bamboo.agents.investigation_session import (
    IntentGap,
    InvestigationOrchestrator,
    InvestigationSession,
    OrchestrationRun,
    Turn,
    _code_hash,
    _Deps,
    _format_running_graph_summary,
    _parse_json_response,
)
from bamboo.frontends.base import InteractionIO, ReviewOption, match_choice
from bamboo.mcp.base import McpTool
from bamboo.models.graph_element import (
    CauseNode,
    NodeType,
    ProcedureNode,
    RelationType,
    ResolutionNode,
    SymptomNode,
    TaskFeatureNode,
)
from bamboo.models.knowledge_entity import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedIO(InteractionIO):
    """Headless :class:`InteractionIO` for tests.

    ``ask`` answers context-appropriately (cause/resolution text for the
    finalize form, ``"y"`` otherwise); ``confirm`` returns ``True``.  All render
    methods are no-ops.  Replaces the old practice of monkeypatching the
    module-level ``_ask``/``_confirm`` globals, which no longer exist.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def ask(self, prompt, *, default=None, choices=None) -> str:
        self.asked.append(prompt)
        plow = (prompt or "").lower()
        if "cause" in plow and "what" in plow:
            return "memory pressure"
        if "resolution" in plow and "what" in plow:
            return "use larger queue"
        return "y"

    async def confirm(self, prompt, *, default=None) -> bool:
        return True

    async def edit(self, *, strategy_type, code, summary, triggers):
        return strategy_type, code, summary, triggers

    def notice(self, text) -> None:  # noqa: D401
        pass

    def panel(self, body, *, title=None, style=None, fit=False) -> None:
        pass

    def code(self, code, *, lang="python") -> None:
        pass

    def table(self, *, title, columns, rows) -> None:
        pass

    def result(self, summary, *, title=None) -> None:
        pass

    def diff(self, rows, *, edge_count, edges=None) -> None:
        pass


class _QueueIO(_ScriptedIO):
    """`_ScriptedIO` whose ``ask`` returns answers from a queue (then ``"N"``).

    Used to script the review-and-policy prompt choices (``y``/``a``/``k``/``N``).
    """

    def __init__(self, answers: list[str]) -> None:
        super().__init__()
        self.answers = list(answers)

    async def ask(self, prompt, *, default=None, choices=None) -> str:
        self.asked.append(prompt)
        return self.answers.pop(0) if self.answers else "N"


def _make_mcp_client(tools: list[McpTool] | None = None) -> Any:
    client = MagicMock()
    client.connect = AsyncMock()
    client.list_tools = MagicMock(return_value=tools or [])
    client.task_data_tools = MagicMock(return_value=frozenset())
    client.execute = AsyncMock(return_value={"ok": True})
    return client


def _make_graph_db() -> Any:
    db = MagicMock()
    db.find_causes = AsyncMock(return_value=[])
    db.find_procedures_for_causes = AsyncMock(return_value=[])
    db.find_all_procedures = AsyncMock(return_value=[])
    db.get_node_description = AsyncMock(return_value=None)
    return db


def _build_orch(
    *, mcp_tools: list[McpTool] | None = None, console=None, io: InteractionIO | None = None
) -> InvestigationOrchestrator:
    deps = _Deps(
        mcp_client=_make_mcp_client(mcp_tools),
        graph_db=_make_graph_db(),
        vector_db=None,
        extractor=None,
        reasoning_navigator=None,
        knowledge_accumulator=None,
        error_classifier=None,
        console=MagicMock() if console is None else console,
        io=io if io is not None else _ScriptedIO(),
    )
    return InvestigationOrchestrator(deps=deps, session_id="t-session", max_turns=10)


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_response_strips_fences():
    assert _parse_json_response('```json\n{"x": 1}\n```') == {"x": 1}
    assert _parse_json_response('```\n{"y": 2}\n```') == {"y": 2}
    assert _parse_json_response('{"z": 3}') == {"z": 3}


def test_parse_json_response_returns_empty_on_invalid():
    assert _parse_json_response("not json") == {}
    assert _parse_json_response("") == {}
    assert _parse_json_response("[1, 2]") == {}  # list, not dict


# ---------------------------------------------------------------------------
# /undo snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_restores_partial_graph():
    orch = _build_orch()
    # Snapshot is a deep copy: mutating the original graph after snapshotting
    # must NOT mutate the snapshot.
    orch.session.partial_graph.nodes.append(
        SymptomNode(name="A", description="a")
    )
    orch.session.last_turn_snapshot = orch.session.partial_graph.model_copy(deep=True)
    orch.session.partial_graph.nodes.append(
        SymptomNode(name="B", description="b")
    )
    assert len(orch.session.partial_graph.nodes) == 2
    assert len(orch.session.last_turn_snapshot.nodes) == 1, "deep copy was broken"

    handled = await orch._handle_meta_command("/undo")
    assert handled is True
    assert len(orch.session.partial_graph.nodes) == 1
    assert orch.session.partial_graph.nodes[0].name == "A"
    assert any(g.kind == "undo" for g in orch.session.intent_gap_log)


@pytest.mark.asyncio
async def test_done_and_abandon_terminate():
    orch = _build_orch()
    assert await orch._handle_meta_command("/done") == "terminate"
    assert orch.session.status == "resolved"

    orch2 = _build_orch()
    assert await orch2._handle_meta_command("/abandon") == "terminate"
    assert orch2.session.status == "abandoned"


@pytest.mark.asyncio
async def test_unknown_meta_command_returns_false():
    orch = _build_orch()
    assert await orch._handle_meta_command("hello world") is False


@pytest.mark.asyncio
async def test_skip_returns_true_without_mutating_state():
    orch = _build_orch()
    orch.session.partial_graph.nodes.append(SymptomNode(name="A"))
    pre_nodes = len(orch.session.partial_graph.nodes)
    pre_turns = len(orch.session.turn_history)
    assert await orch._handle_meta_command("/skip") is True
    assert len(orch.session.partial_graph.nodes) == pre_nodes
    assert len(orch.session.turn_history) == pre_turns


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_classifier_returns_tool_on_high_confidence(monkeypatch):
    orch = _build_orch()

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return '{"intent": "tool", "confidence": 0.95, "is_close_call": false, "rationale": "imperative"}'

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    intent = await orch._classify_intent("show me the failed scout jobs")
    assert intent == "tool"


@pytest.mark.asyncio
async def test_intent_classifier_falls_back_to_narration_on_llm_failure(monkeypatch):
    orch = _build_orch()

    async def boom(_self, system, user, **_kwargs):  # noqa: ARG001
        raise RuntimeError("LLM down")

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", boom, raising=True)
    # Default-safe: never trigger an external tool on a classifier failure.
    intent = await orch._classify_intent("anything")
    assert intent == "narration"


# ---------------------------------------------------------------------------
# Narration merge
# ---------------------------------------------------------------------------


def test_narration_merge_adds_nodes_and_relationships():
    orch = _build_orch()
    parsed = {
        "nodes": [
            {"node_type": "Cause", "name": "memory pressure", "description": "high mem"},
            {"node_type": "Resolution", "name": "use larger queue", "description": "switch site"},
        ],
        "relationships": [
            {
                "source_name": "memory pressure",
                "target_name": "use larger queue",
                "relation_type": "solved_by",
                "confidence": 0.9,
            }
        ],
    }
    added = orch._merge_narration_extraction(parsed)
    assert added == 3
    names = {n.name for n in orch.session.partial_graph.nodes}
    assert "memory pressure" in names and "use larger queue" in names
    assert any(r.relation_type == RelationType.SOLVED_BY for r in orch.session.partial_graph.relationships)


def test_narration_merge_dedups_by_name():
    orch = _build_orch()
    # Pre-existing node with the same name.
    orch.session.partial_graph.nodes.append(
        CauseNode(name="X", description="initial")
    )
    parsed = {
        "nodes": [
            {"node_type": "Cause", "name": "X", "description": "second-time"},
        ],
        "relationships": [],
    }
    added = orch._merge_narration_extraction(parsed)
    # Still only one "X" node — canonical-name dedup.
    cause_xs = [n for n in orch.session.partial_graph.nodes if n.name == "X"]
    assert len(cause_xs) == 1
    # `added` counts what was passed in (1 node), but dedup keeps only one in graph.
    assert added == 1


def test_narration_merge_drops_relationships_with_unknown_endpoints():
    orch = _build_orch()
    parsed = {
        "nodes": [{"node_type": "Cause", "name": "A", "description": "a"}],
        "relationships": [
            {"source_name": "A", "target_name": "MISSING", "relation_type": "solved_by"},
        ],
    }
    orch._merge_narration_extraction(parsed)
    assert orch.session.partial_graph.relationships == []


# ---------------------------------------------------------------------------
# Atomic action recording
# ---------------------------------------------------------------------------


def test_record_atomic_action_creates_procedure_and_task_context():
    orch = _build_orch()
    run = OrchestrationRun(
        strategy_type="inspect_failed_scout_jobs",
        code='return {"jobs": []}',
        code_summary="Fetch scout jobs.",
        trigger_signals=["scout phase failed"],
        external_access=True,
        call_log=["get_scout_job_details"],
        result_summary="0 jobs",
        atomic_action_id="abc123",
    )
    orch._record_atomic_action(run, raw_result={"jobs": []})

    procs = [n for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)]
    ctxs = [n for n in orch.session.partial_graph.nodes if n.node_type == NodeType.TASK_CONTEXT]
    assert len(procs) == 1
    assert len(ctxs) == 1
    proc = procs[0]
    assert proc.metadata["orchestration_code"] == 'return {"jobs": []}'
    assert proc.metadata["code_summary"] == "Fetch scout jobs."
    assert proc.metadata["external_access"] is True
    # The per-edge fields are pending until finalize() wires them onto an edge.
    assert proc.metadata["_pending_edge"]["trigger_signals"] == ["scout phase failed"]


# ---------------------------------------------------------------------------
# Finalize / edge wiring
# ---------------------------------------------------------------------------


def test_wire_finalization_creates_cause_resolution_and_investigated_by():
    orch = _build_orch()
    # Set up: a Symptom + a Task_Feature + a tentative Procedure.
    orch.session.partial_graph.nodes.extend(
        [
            SymptomNode(name="TimeoutSymptom", description="timed out"),
            TaskFeatureNode(name="ramCount=4-6GB", attribute="ramCount", value="4-6GB"),
        ]
    )
    run = OrchestrationRun(
        strategy_type="inspect_logs",
        code="return {}",
        code_summary="Look at logs.",
        trigger_signals=["sig"],
        atomic_action_id="aa1",
    )
    orch._record_atomic_action(run, raw_result={})

    orch._wire_finalization_edges(cause_text="low memory pressure", resolution_text="use larger queue")

    names = {n.name for n in orch.session.partial_graph.nodes}
    assert "low memory pressure" in names
    assert "use larger queue" in names
    # Procedure was promoted to canonical name strategy_type:cause.
    proc_names = [
        n.name for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)
    ]
    assert "inspect_logs:low memory pressure" in proc_names
    assert not any("tentative" in name for name in proc_names)

    rel_types = {r.relation_type for r in orch.session.partial_graph.relationships}
    assert RelationType.INVESTIGATED_BY in rel_types
    assert RelationType.SOLVED_BY in rel_types
    assert RelationType.INDICATE in rel_types
    assert RelationType.CONTRIBUTE_TO in rel_types

    # The investigated_by edge must carry trigger_signals on its metadata.
    inv_edge = next(
        r for r in orch.session.partial_graph.relationships
        if r.relation_type == RelationType.INVESTIGATED_BY
    )
    edge_meta = (inv_edge.properties or {}).get("metadata", {})
    assert edge_meta.get("trigger_signals") == ["sig"]
    assert "executed_at" in edge_meta


def test_wire_finalization_abandon_keeps_procedure_tentative():
    orch = _build_orch()
    run = OrchestrationRun(strategy_type="probe", code="return {}", atomic_action_id="aa2")
    orch._record_atomic_action(run, raw_result={})

    # No cause supplied = abandon path.
    orch._wire_finalization_edges(cause_text=None, resolution_text=None)

    proc = next(n for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode))
    assert proc.metadata.get("status") == "tentative"
    assert proc.metadata.get("session_status") == "ongoing"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_tool_registry_merges_mcp_and_internal_tools():
    tools = [
        McpTool(
            name="get_x",
            description="external",
            parameters_schema={"type": "object", "properties": {}},
        )
    ]
    orch = _build_orch(mcp_tools=tools)
    await orch._build_tool_registry()
    unified = orch._unified_tool_descriptors()
    names = {t["name"] for t in unified}
    assert "get_x" in names
    assert "query_past_causes_for_symptom" in names
    assert "query_past_procedures_for_cause" in names
    # External MCP tool defaults to external_access=True; internal tools are False.
    by_name = {t["name"]: t for t in unified}
    assert by_name["get_x"]["external_access"] is True
    assert by_name["query_past_causes_for_symptom"]["external_access"] is False
    # Both are read_only=True (no current tool changes state).
    assert by_name["get_x"]["read_only"] is True
    assert by_name["query_past_causes_for_symptom"]["read_only"] is True


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def test_format_running_graph_summary_groups_by_type():
    g = KnowledgeGraph(
        nodes=[
            SymptomNode(name="S1"),
            SymptomNode(name="S2"),
            CauseNode(name="C1"),
        ]
    )
    out = _format_running_graph_summary(g)
    assert "Symptom" in out or "SYMPTOM" in out or "symptom" in out.lower()
    assert "S1" in out and "C1" in out


def test_format_running_graph_summary_empty():
    assert _format_running_graph_summary(KnowledgeGraph()) == "(empty)"


# ---------------------------------------------------------------------------
# Session resume (round-trip)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_resume_round_trip(tmp_path):
    """Persist mid-session state via --save then rehydrate via --resume.

    Round-trips the whole InvestigationSession dataclass through pydantic JSON.
    Verifies: turn count, status, partial_graph nodes (including a tentative
    Procedure with metadata), turn_history with an embedded OrchestrationRun,
    and the IntentGap log. Catches any field that pydantic can't serialise.
    """
    orch = _build_orch()
    orch.session.turn = 3
    orch.session.status = "ongoing"
    orch.session.initial_inputs = {"task_id": 12345}
    orch.session.doc_hints = {"panda": "ATLAS PanDA workflow manager"}
    orch.session.partial_graph.nodes.extend(
        [
            SymptomNode(name="LowEfficiency", description="cpu low"),
            ProcedureNode(
                name="inspect_x:tentative:aa1",
                description="probe x",
                strategy_type="inspect_x",
                metadata={
                    "orchestration_code": 'return {"ok": True}',
                    "code_summary": "probe x",
                    "external_access": True,
                    "atomic_action_id": "aa1",
                    "_pending_edge": {
                        "trigger_signals": ["low cpu seen"],
                        "result_summary": "got ok",
                        "executed_at": "2026-05-29T00:00:00Z",
                    },
                },
            ),
        ]
    )
    orch.session.turn_history.append(
        Turn(
            role="system",
            kind="tool",
            text="inspect_x",
            orchestration=OrchestrationRun(
                strategy_type="inspect_x",
                code='return {"ok": True}',
                code_summary="probe x",
                trigger_signals=["low cpu seen"],
                external_access=True,
                call_log=["foo"],
                result_summary='{"ok": True}',
                atomic_action_id="aa1",
                executed_at="2026-05-29T00:00:00Z",
            ),
        )
    )
    orch.session.intent_gap_log.append(
        IntentGap(utterance="ambiguous", classifier_guess="tool", kind="disambiguation", turn=2)
    )

    save_file = tmp_path / "session.json"
    orch.save_path = save_file
    orch._persist()

    # Rehydrate.
    from bamboo.agents.investigation_session import InvestigationSession

    blob = save_file.read_text()
    restored = InvestigationSession.model_validate_json(blob)

    assert restored.turn == 3
    assert restored.status == "ongoing"
    assert restored.initial_inputs["task_id"] == 12345
    assert restored.doc_hints["panda"].startswith("ATLAS")

    names = {n.name for n in restored.partial_graph.nodes}
    assert "LowEfficiency" in names
    assert "inspect_x:tentative:aa1" in names

    proc = next(n for n in restored.partial_graph.nodes if n.name == "inspect_x:tentative:aa1")
    assert proc.metadata["orchestration_code"] == 'return {"ok": True}'
    assert proc.metadata["_pending_edge"]["trigger_signals"] == ["low cpu seen"]

    assert len(restored.turn_history) == 1
    th = restored.turn_history[0]
    assert th.orchestration is not None
    assert th.orchestration.strategy_type == "inspect_x"
    assert th.orchestration.call_log == ["foo"]

    assert len(restored.intent_gap_log) == 1
    assert restored.intent_gap_log[0].kind == "disambiguation"


# ---------------------------------------------------------------------------
# Commit diff classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_nodes_for_diff_marks_existing_as_merge():
    """The diff helper should use NodeType.value (the Neo4j label) when looking up existing nodes."""
    orch = _build_orch()

    # Pre-stage: SymptomNode "Existing" already in Neo4j; "Fresh" not.
    async def fake_get(label, name):
        # The label MUST be the canonical "Symptom" (NodeType.value), not "NodeType.SYMPTOM".
        if label != "Symptom":
            raise AssertionError(f"got label {label!r}; expected 'Symptom'")
        return "stored description" if name == "Existing" else None

    orch.deps.graph_db.get_node_description = AsyncMock(side_effect=fake_get)

    graph = KnowledgeGraph(
        nodes=[
            SymptomNode(name="Existing"),
            SymptomNode(name="Fresh"),
        ]
    )
    rows = await orch._classify_nodes_for_diff(graph)
    by_name = {name: action for _, name, action in rows}
    assert by_name["Existing"] == "merge"
    assert by_name["Fresh"] == "new"


@pytest.mark.asyncio
async def test_classify_nodes_for_diff_treats_db_error_as_new():
    """graph_db errors per node are non-fatal — that node falls through to 'new'."""
    orch = _build_orch()
    orch.deps.graph_db.get_node_description = AsyncMock(side_effect=RuntimeError("conn down"))

    graph = KnowledgeGraph(nodes=[SymptomNode(name="X")])
    rows = await orch._classify_nodes_for_diff(graph)
    assert rows == [("Symptom", "X", "new")]


# ---------------------------------------------------------------------------
# Editor buffer parsing
# ---------------------------------------------------------------------------


def test_parse_editor_buffer_round_trips_all_fields():
    from bamboo.frontends.cli import _parse_editor_buffer

    text = """\
# strategy_type: inspect_failed_scout_jobs
# code_summary: Fetch the scout-job details for the failing task.
# trigger_signals:
#   - symptom indicates scout phase failed
#   - scout-job inspection not yet performed in this session
# --- code below (lines starting with `#` are header metadata; `## ` is a literal comment) ---

## this is a real code comment that should be preserved
jobs = await tools.get_scout_job_details(task_id=task_id)
return {"jobs": jobs}
"""
    strategy, code, summary, triggers = _parse_editor_buffer(
        text,
        fallback_strategy="fb_strategy",
        fallback_summary="fb_summary",
        fallback_triggers=["fb"],
    )
    assert strategy == "inspect_failed_scout_jobs"
    assert summary == "Fetch the scout-job details for the failing task."
    assert triggers == [
        "symptom indicates scout phase failed",
        "scout-job inspection not yet performed in this session",
    ]
    # Body preserves the literal '## ' code comment.
    assert "## this is a real code comment" in code
    assert "get_scout_job_details" in code
    # Body does NOT contain the header lines.
    assert "strategy_type:" not in code
    assert "code_summary:" not in code


def test_parse_editor_buffer_falls_back_when_fields_missing():
    from bamboo.frontends.cli import _parse_editor_buffer

    text = """\
# only a separator
return {}
"""
    strategy, code, summary, triggers = _parse_editor_buffer(
        text,
        fallback_strategy="fb_strategy",
        fallback_summary="fb_summary",
        fallback_triggers=["fb_trig"],
    )
    # Headers all missing → fallbacks used.
    assert strategy == "fb_strategy"
    assert summary == "fb_summary"
    assert triggers == ["fb_trig"]
    assert code == "return {}"


def test_parse_editor_buffer_handles_empty_trigger_block():
    from bamboo.frontends.cli import _parse_editor_buffer

    text = """\
# strategy_type: x
# code_summary: y
# trigger_signals:
#   -
# ---
return {}
"""
    strategy, code, summary, triggers = _parse_editor_buffer(
        text,
        fallback_strategy="fb",
        fallback_summary="fb",
        fallback_triggers=["fb"],
    )
    # Empty item rows produce no triggers; falls back to fallback list.
    assert triggers == ["fb"]
    assert strategy == "x"


# ---------------------------------------------------------------------------
# Tool-gap-log path (LLM returns code=null)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_turn_logs_gap_when_llm_returns_null_code(monkeypatch):
    """When the orchestration LLM says no tool fits, log to tool_gap_log and don't crash."""
    orch = _build_orch()
    await orch._build_tool_registry()

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return '{"code": null, "reason": "no tool can open external links"}'

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    await orch._tool_turn("open the GitLab MR for this task")

    assert len(orch.session.tool_gap_log) == 1
    gap = orch.session.tool_gap_log[0]
    assert gap.human_request_text == "open the GitLab MR for this task"
    assert "external links" in gap.llm_reason
    # No atomic_action recorded.
    procs = [n for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)]
    assert procs == []


# ---------------------------------------------------------------------------
# End-to-end integration (mocked LLM + DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_session_tool_narration_finalize_commit(monkeypatch):
    """Drive a full session: 1 tool turn → 1 narration turn → /done → commit.

    No live services: LLM is scripted, MCP client is in-memory, graph_db /
    KnowledgeAccumulator are mocks. Verifies the orchestrator wires every
    piece together: tool branch records a ProcedureNode with the right
    replayability metadata, narration branch merges extracted Causes,
    finalize() wires investigated_by + solved_by edges, commit calls
    KnowledgeAccumulator's storage methods, gap logs land where expected.
    """
    # --- Mocks ---
    mcp_client = MagicMock()
    mcp_client.connect = AsyncMock()
    mcp_client.list_tools = MagicMock(
        return_value=[
            McpTool(
                name="get_scout_job_details",
                description="fetch scout job info",
                parameters_schema={"type": "object", "properties": {"task_id": {"type": "integer"}}},
                external_access=True,
            ),
        ]
    )
    mcp_client.task_data_tools = MagicMock(return_value=frozenset())
    mcp_client.execute = AsyncMock(
        return_value=[{"PandaID": 1001, "status": "failed"}, {"PandaID": 1002, "status": "failed"}]
    )

    graph_db = MagicMock()
    graph_db.find_causes = AsyncMock(return_value=[])
    graph_db.find_procedures_for_causes = AsyncMock(return_value=[])
    graph_db.get_node_description = AsyncMock(return_value=None)  # all nodes are "new"

    accumulator = MagicMock()
    accumulator.store_extracted = AsyncMock(return_value=("summary text", ["insight 1"]))

    deps = _Deps(
        mcp_client=mcp_client,
        graph_db=graph_db,
        vector_db=None,
        extractor=None,
        reasoning_navigator=None,  # skip the proactive-hypothesis path
        knowledge_accumulator=accumulator,
        error_classifier=None,
        console=MagicMock(),
        io=_ScriptedIO(),
    )
    orch = InvestigationOrchestrator(deps=deps, session_id="integ-test", max_turns=10)
    orch.session.initial_inputs = {"task_id": 12345, "task_data": {"status": "exhausted"}, "symptom": None}
    await orch._build_tool_registry()

    # --- Scripted LLM responses ---
    llm_queue = [
        # 1: tool-turn orchestration call → emits a single-tool code block that calls get_scout_job_details
        '{"strategy_type":"inspect_failed_scout_jobs",'
        '"code":"jobs = await tools.get_scout_job_details(task_id=task_id)\\nreturn {\\"jobs\\": jobs}",'
        '"code_summary":"Fetch the failing scout jobs for this task.",'
        '"trigger_signals":["scout phase has failed jobs","no scout inspection yet"]}',
        # 2: narration extraction call → returns a Cause + Resolution
        '{"nodes":['
        '{"node_type":"Cause","name":"memory pressure","description":"the failed jobs all OOMed"},'
        '{"node_type":"Resolution","name":"use larger queue","description":"switch the task to a high-memory queue"}'
        '],"relationships":['
        '{"source_name":"memory pressure","target_name":"use larger queue","relation_type":"solved_by","confidence":0.95}'
        ']}',
    ]

    async def scripted_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return llm_queue.pop(0)

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", scripted_invoke, raising=True)
    # The injected _ScriptedIO answers the confirmation prompts ("y") and the
    # finalize form (cause/resolution text) — no module-global monkeypatching.

    # --- Turn 1: tool ---
    await orch._tool_turn("show me the failed scout jobs")

    procs = [n for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)]
    assert len(procs) == 1
    proc = procs[0]
    assert proc.strategy_type == "inspect_failed_scout_jobs"
    assert proc.metadata["external_access"] is True
    assert "get_scout_job_details" in proc.metadata["orchestration_code"]
    # The tool was actually invoked through the proxy.
    mcp_client.execute.assert_called_once()
    # OrchestrationRun is captured on the turn_history entry.
    last_turn = orch.session.turn_history[-1]
    assert last_turn.orchestration is not None
    assert last_turn.orchestration.call_log == ["get_scout_job_details"]

    # --- Turn 2: narration ---
    await orch._narration_turn("the failed jobs all OOMed — looks like memory pressure")

    names = {n.name for n in orch.session.partial_graph.nodes}
    assert "memory pressure" in names
    assert "use larger queue" in names
    assert any(
        r.relation_type == RelationType.SOLVED_BY for r in orch.session.partial_graph.relationships
    )

    # --- Finalize: cause already extracted; supply final cause + resolution via the form ---
    orch.session.status = "resolved"
    await orch.finalize()

    # Procedure was promoted to its stable, signature-based canonical name (Phase 2a):
    # the cause is carried by the investigated_by edge, not the node name.
    proc_names = [n.name for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)]
    assert "proc__get_scout_job_details" in proc_names, proc_names
    # investigated_by edge wired between Cause and Procedure.
    rel_types = {r.relation_type for r in orch.session.partial_graph.relationships}
    assert RelationType.INVESTIGATED_BY in rel_types

    # Commit delegated to the shared KnowledgeAccumulator.store_extracted path.
    accumulator.store_extracted.assert_called_once()

    # graph_id was set on the graph for resume idempotency.
    stored_graph = accumulator.store_extracted.call_args.args[0]
    assert stored_graph.metadata.get("graph_id") == "investigate:integ-test"


# ---------------------------------------------------------------------------
# Phase-1 review-and-policy gate (docs/EXECUTION_TRUST.md)
# ---------------------------------------------------------------------------


def test_code_hash_ignores_whitespace_but_not_logic():
    assert _code_hash("x = 1\n\n  y = 2  ") == _code_hash("x = 1\n  y = 2")
    assert _code_hash("x = 1") != _code_hash("x = 2")


def test_code_policies_round_trips():
    s = InvestigationSession(session_id="s")
    s.code_policies["abc123"] = "auto_run"
    s2 = InvestigationSession.model_validate_json(s.model_dump_json())
    assert s2.code_policies == {"abc123": "auto_run"}


def _readonly_tool() -> McpTool:
    # read_only=True is the default; a plain read tool.
    return McpTool(
        name="get_x",
        description="read x",
        parameters_schema={"type": "object"},
    )


def _orch_plan(code: str) -> str:
    return json.dumps(
        {"strategy_type": "s", "code": code, "code_summary": "sum", "trigger_signals": []}
    )


@pytest.mark.asyncio
async def test_tool_turn_auto_run_policy_persists_and_skips_prompt(monkeypatch):
    """Choosing `a` persists auto_run; the same code re-runs with no further prompt."""
    io = _QueueIO(["a"])  # turn 1: review → auto-run
    orch = _build_orch(mcp_tools=[_readonly_tool()], io=io)
    await orch._build_tool_registry()
    code = 'r = await tools.get_x()\nreturn {"r": r}'

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return _orch_plan(code)

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)

    await orch._tool_turn("do x")
    assert orch.session.code_policies.get(_code_hash(code)) == "auto_run"
    asked_after_first = len(io.asked)

    # Turn 2: identical code → auto-run, no new prompt.
    await orch._tool_turn("do x again")
    assert len(io.asked) == asked_after_first  # no additional ask
    tool_turns = [t for t in orch.session.turn_history if t.kind == "tool"]
    assert len(tool_turns) == 2


@pytest.mark.asyncio
async def test_tool_turn_reject_records_no_procedure(monkeypatch):
    io = _QueueIO(["N"])
    orch = _build_orch(mcp_tools=[_readonly_tool()], io=io)
    await orch._build_tool_registry()

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return _orch_plan('return {"r": await tools.get_x()}')

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    await orch._tool_turn("do x")

    procs = [n for n in orch.session.partial_graph.nodes if isinstance(n, ProcedureNode)]
    assert procs == []
    assert any(g.kind == "reject_code" for g in orch.session.intent_gap_log)


@pytest.mark.asyncio
async def test_tool_turn_run_once_does_not_persist_policy(monkeypatch):
    io = _QueueIO(["y"])  # run once, no persistence
    orch = _build_orch(mcp_tools=[_readonly_tool()], io=io)
    await orch._build_tool_registry()
    code = 'return {"r": await tools.get_x()}'

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return _orch_plan(code)

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    await orch._tool_turn("do x")
    assert orch.session.code_policies == {}  # nothing remembered


@pytest.mark.asyncio
async def test_build_tool_registry_exposes_reusable_procedure_tools():
    """Phase 2a: approved non-trivial read-only procedures become callable proc__ tools."""
    orch = _build_orch()
    orch.deps.graph_db.find_all_procedures = AsyncMock(
        return_value=[
            {
                "tool_name": "proc__fetch_logs__get_jobs",
                "signature": ["fetch_logs", "get_jobs"],
                "orchestration_code": "a=await tools.get_jobs(task_id=task_id)\nb=await tools.fetch_logs()\nreturn {}",
                "strategy_type": "check jobs",
                "code_summary": "fetch jobs + logs",
                "external_access": True,
                "cause_names": ["memory pressure"],
            },
        ]
    )
    await orch._build_tool_registry()
    assert "proc__fetch_logs__get_jobs" in orch._procedure_tool_descriptors
    assert "proc__fetch_logs__get_jobs" in orch._procedure_tool_callables
    # And it's surfaced in the unified descriptor list the planner sees, as read-only.
    by_name = {d["name"]: d for d in orch._unified_tool_descriptors()}
    assert "proc__fetch_logs__get_jobs" in by_name
    assert by_name["proc__fetch_logs__get_jobs"]["read_only"] is True


def _durable_proc_row(auto_run: bool) -> dict:
    return {
        "tool_name": "proc__fetch_logs__get_jobs",
        "signature": ["fetch_logs", "get_jobs"],
        "orchestration_code": "a=await tools.get_jobs(task_id=task_id)\nb=await tools.fetch_logs()\nreturn {}",
        "strategy_type": "check jobs",
        "code_summary": "fetch jobs + logs",
        "external_access": True,
        "auto_run": auto_run,
        "cause_names": ["memory pressure"],
    }


@pytest.mark.asyncio
async def test_durable_autorun_skips_prompt_for_granted_procedure(monkeypatch):
    """Phase 2b: a turn that calls only a durably-granted procedure runs with NO prompt."""
    io = _QueueIO([])  # would return "N" if asked — we assert it is NOT asked
    orch = _build_orch(io=io)
    orch.deps.graph_db.find_all_procedures = AsyncMock(return_value=[_durable_proc_row(auto_run=True)])
    await orch._build_tool_registry()
    assert "proc__fetch_logs__get_jobs" in orch._durable_autorun_procs

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return _orch_plan("return await tools.proc__fetch_logs__get_jobs()")

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    asked_before = len(io.asked)
    await orch._tool_turn("reuse the saved procedure")
    assert len(io.asked) == asked_before  # auto-ran without a review prompt
    assert [t for t in orch.session.turn_history if t.kind == "tool"]  # the turn executed


@pytest.mark.asyncio
async def test_grant_durable_autorun_on_procedure_reuse(monkeypatch):
    """Choosing `a` on a procedure-reuse turn persists a durable (cross-session) grant."""
    io = _QueueIO(["a"])
    orch = _build_orch(io=io)
    orch.deps.graph_db.set_procedure_auto_run = AsyncMock(return_value=True)
    orch.deps.graph_db.find_all_procedures = AsyncMock(return_value=[_durable_proc_row(auto_run=False)])
    await orch._build_tool_registry()
    assert "proc__fetch_logs__get_jobs" not in orch._durable_autorun_procs  # not granted yet

    async def fake_invoke(_self, system, user, **_kwargs):  # noqa: ARG001
        return _orch_plan("return await tools.proc__fetch_logs__get_jobs()")

    monkeypatch.setattr(InvestigationOrchestrator, "_invoke_llm", fake_invoke, raising=True)
    await orch._tool_turn("reuse and trust it")
    orch.deps.graph_db.set_procedure_auto_run.assert_awaited_with("proc__fetch_logs__get_jobs", True)
    assert "proc__fetch_logs__get_jobs" in orch._durable_autorun_procs


@pytest.mark.asyncio
async def test_revoke_durable_procedure_grant():
    orch = _build_orch()
    orch.deps.graph_db.set_procedure_auto_run = AsyncMock(return_value=True)
    orch._durable_autorun_procs = {"proc__x__y"}
    assert await orch._handle_meta_command("/revoke proc__x__y") is True
    orch.deps.graph_db.set_procedure_auto_run.assert_awaited_with("proc__x__y", False)
    assert "proc__x__y" not in orch._durable_autorun_procs


@pytest.mark.asyncio
async def test_approvals_and_revoke_meta_commands():
    orch = _build_orch()
    orch.session.code_policies["deadbeef0001"] = "auto_run"
    orch.session.code_policies["cafebabe0002"] = "always_ask"

    assert await orch._handle_meta_command("/approvals") is True
    # Revoke by hash-prefix.
    assert await orch._handle_meta_command("/revoke deadbeef") is True
    assert "deadbeef0001" not in orch.session.code_policies
    assert "cafebabe0002" in orch.session.code_policies
    # Revoke all.
    assert await orch._handle_meta_command("/revoke all") is True
    assert orch.session.code_policies == {}


# ---------------------------------------------------------------------------
# _invoke_llm streaming path (strategy step)
# ---------------------------------------------------------------------------


class _RecordingSink:
    """An active DetailSink that records what it was fed."""

    active = True

    def __init__(self) -> None:
        self.answer: list[str] = []
        self.reasoning: list[str] = []
        self.meta_lines: list[str] = []

    def feed(self, text: str, *, reasoning: bool = False) -> None:
        (self.reasoning if reasoning else self.answer).append(text)

    def meta(self, line: str) -> None:
        self.meta_lines.append(line)


class _InactiveSink(_RecordingSink):
    active = False


class _FakeStreamLLM:
    """Fake chat model whose ``astream`` yields the given chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.astream_kwargs: list[dict] = []

    async def astream(self, _messages, **kwargs):
        self.astream_kwargs.append(kwargs)
        for c in self._chunks:
            yield c

    async def ainvoke(self, _messages, **_kwargs):
        raise AssertionError("ainvoke must not be called when the sink is active")


class _FakeInvokeLLM:
    """Fake chat model with only ``ainvoke`` (the non-streaming path)."""

    async def ainvoke(self, _messages, **_kwargs):
        return SimpleNamespace(content="plain result")


@pytest.mark.asyncio
async def test_invoke_llm_streams_answer_and_reasoning_and_forces_ollama_reasoning(monkeypatch):
    from bamboo.agents import investigation_session as sess

    monkeypatch.setattr(sess, "get_settings", lambda: SimpleNamespace(llm_provider="ollama"))
    orch = _build_orch()
    orch._llm = _FakeStreamLLM(
        [
            SimpleNamespace(content="", additional_kwargs={"reasoning_content": "let me "}),
            SimpleNamespace(content='{"strategy', additional_kwargs={"reasoning_content": "think"}),
            SimpleNamespace(content='_type": "x"}', additional_kwargs={}),
        ]
    )
    sink = _RecordingSink()
    out = await orch._invoke_llm("sys", "user", stream_sink=sink)

    # Returned text is the concatenated *answer* only — clean for JSON parsing.
    assert out == '{"strategy_type": "x"}'
    assert _parse_json_response(out) == {"strategy_type": "x"}
    assert "".join(sink.answer) == '{"strategy_type": "x"}'
    assert "".join(sink.reasoning) == "let me think"
    # Reasoning is forced on for this call only (Ollama provider).
    assert orch._llm.astream_kwargs[0].get("reasoning") is True


@pytest.mark.asyncio
async def test_invoke_llm_uses_ainvoke_when_sink_absent_or_inactive():
    orch = _build_orch()
    orch._llm = _FakeInvokeLLM()
    assert await orch._invoke_llm("sys", "user") == "plain result"
    assert await orch._invoke_llm("sys", "user", stream_sink=_InactiveSink()) == "plain result"


# ---------------------------------------------------------------------------
# review choices: match_choice + review_orchestration default
# ---------------------------------------------------------------------------

_OPTS = [
    ReviewOption("run", "run once", "y"),
    ReviewOption("auto", "auto-run", "a"),
    ReviewOption("edit", "edit", "e"),
    ReviewOption("reject", "reject", "n"),
]


def test_match_choice_keys_aliases_and_default():
    assert match_choice("run", _OPTS) == "run"
    assert match_choice("AUTO", _OPTS) == "auto"        # case-insensitive
    assert match_choice("y", _OPTS) == "run"            # single-letter alias
    assert match_choice("n", _OPTS) == "reject"
    assert match_choice("", _OPTS) == "reject"          # blank → last (safe default)
    assert match_choice("nonsense", _OPTS) is None


@pytest.mark.asyncio
async def test_review_orchestration_default_resolves_choice():
    io = _QueueIO(["auto"])
    choice = await io.review_orchestration(
        strategy_type="s", summary="sum", triggers=["t1"], code="x = 1", options=_OPTS
    )
    assert choice == "auto"
    assert any("Review" in p for p in io.asked)  # the proposal prompt was shown


@pytest.mark.asyncio
async def test_review_orchestration_default_reprompts_then_accepts_alias():
    io = _QueueIO(["bogus", "y"])  # invalid, then the "run" alias
    choice = await io.review_orchestration(
        strategy_type="s", summary="", triggers=[], code="x = 1", options=_OPTS
    )
    assert choice == "run"


# ---------------------------------------------------------------------------
# Session-start root-cause hypothesis panel (_show_past_similar)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_show_past_similar_stashes_full_analysis_card():
    from bamboo.models.knowledge_entity import AnalysisResult

    orch = _build_orch()
    orch.deps.reasoning_navigator = SimpleNamespace(
        analyze_task=AsyncMock(
            return_value=AnalysisResult(
                task_id="t1",
                root_cause="Input dataset metadata mismatch",
                confidence=0.95,
                resolution="Verify the dataset integrity in Rucio.",
                explanation="file list len=0 != meta=1 — Rucio listing empty vs metadata.",
            )
        )
    )

    await orch._show_past_similar({"status": "exhausted"}, None)

    # The hypothesis is stashed (posted with the CTA in run()), not rendered here.
    card = orch._hypothesis_card
    assert card is not None
    assert "root cause analysis" in (card.title or "").lower()
    assert "Input dataset metadata mismatch" in card.body  # cause
    assert "0.95" in card.body  # confidence
    assert "Verify the dataset integrity in Rucio." in card.body  # resolution
    assert "file list len=0 != meta=1" in card.body  # reasoning included


def test_display_kickoff_panel_builds_fields_card():
    from bamboo.frontends.base import Card

    io = _ScriptedIO()
    captured: list[Card] = []
    io.cards = lambda cards: captured.extend(cards)
    orch = _build_orch(io=io)

    orch._display_kickoff_panel(
        {
            "status": "failed",
            "taskType": "anal",
            "site": "DESY-HH_TEST",
            "errorDialog": "failed since no file was successfully processed",
        },
        None,
    )

    assert len(captured) == 1
    card = captured[0]
    assert card.title == "Task under investigation"
    assert card.style == "red"  # failed → red accent
    assert "failed since no file" in card.body  # errorDialog as body
    assert ("status", "failed", True) in card.fields
    assert ("taskType", "anal", True) in card.fields
    assert ("site", "DESY-HH_TEST", True) in card.fields


# ---------------------------------------------------------------------------
# Tool-selection budget gating (_render_available_tools)
# ---------------------------------------------------------------------------


def _tool_descs(n: int) -> list[dict]:
    return [
        {
            "name": f"t{i}",
            "description": "d" * 60,
            "parameters_schema": {"properties": {"x": {}}},
            "external_access": True,
            "read_only": True,
        }
        for i in range(n)
    ]


def _small_settings():
    # tiny context window forces the over-budget retrieval path
    return SimpleNamespace(
        llm_context_window=200,
        tool_budget_margin=0,
        tool_max_full_schemas=5,
        mcp_servers_config="",
        llm_provider="ollama",
    )


async def test_render_available_tools_under_budget_renders_all_no_selector(monkeypatch):
    import bamboo.agents.investigation_session as ist

    orch = _build_orch()
    orch._tok = len
    # generous window => everything fits => selector never consulted
    monkeypatch.setattr(
        ist, "get_settings",
        lambda: SimpleNamespace(llm_context_window=100000, tool_budget_margin=0,
                                tool_max_full_schemas=100, mcp_servers_config="",
                                llm_provider="ollama"),
    )
    selector = MagicMock()
    selector.select = AsyncMock()
    orch.deps.tool_selector = selector

    descs = _tool_descs(5)
    text, shown, dropped = await orch._render_available_tools(
        descs, base_text="", utterance="why", error_dialog="err"
    )
    assert shown == {d["name"] for d in descs}
    assert dropped == []
    selector.select.assert_not_awaited()


async def test_render_available_tools_over_budget_uses_selector(monkeypatch):
    import bamboo.agents.investigation_session as ist
    from bamboo.agents.helpers.tool_selection import Selection

    orch = _build_orch()
    orch._tok = len
    monkeypatch.setattr(ist, "get_settings", _small_settings)

    descs = _tool_descs(10)
    names = [d["name"] for d in descs][:5]
    selector = MagicMock()
    selector.ensure_index = AsyncMock()
    selector.select = AsyncMock(
        return_value=Selection(ordered=names, full_schema_names=set(names[:2]))
    )
    orch.deps.tool_selector = selector

    text, shown, dropped = await orch._render_available_tools(
        descs, base_text="", utterance="why", error_dialog="err"
    )
    selector.ensure_index.assert_awaited()
    selector.select.assert_awaited()
    assert shown  # a subset is shown
    assert set(shown) <= set(names)  # only retrieved candidates
    assert dropped  # some of the 10 omitted under the tiny budget


async def test_render_available_tools_fail_hard_on_retrieval_unavailable(monkeypatch):
    import bamboo.agents.investigation_session as ist
    from bamboo.agents.helpers.tool_selection import RetrievalUnavailable

    orch = _build_orch()
    orch._tok = len
    monkeypatch.setattr(ist, "get_settings", _small_settings)

    selector = MagicMock()
    selector.ensure_index = AsyncMock()
    selector.select = AsyncMock(side_effect=RetrievalUnavailable("vector store down"))
    orch.deps.tool_selector = selector

    with pytest.raises(RetrievalUnavailable):
        await orch._render_available_tools(
            _tool_descs(10), base_text="", utterance="q", error_dialog="e"
        )


async def test_index_approved_run_indexes_and_logs(monkeypatch):
    orch = _build_orch()
    selector = MagicMock()
    selector.index_procedure_run = AsyncMock()
    orch.deps.tool_selector = selector

    run = OrchestrationRun(
        strategy_type="s",
        code="x = await tools.get_parent_task(task_id=task_id)\nreturn {'x': x}",
        utterance="who is the parent",
        call_log=["get_parent_task"],
    )
    await orch._index_approved_run(run, error_dialog="boom", shown_names={"get_parent_task"})
    selector.index_procedure_run.assert_awaited_once()
    kwargs = selector.index_procedure_run.await_args.kwargs
    assert "get_parent_task" in kwargs["tool_names"]
    assert "who is the parent" in kwargs["prompt_text"]
    # the approved (prompt -> tools) example is stashed for finalize upweight
    assert orch._approved_run_prompts


async def test_index_approved_run_skips_errored_runs(monkeypatch):
    orch = _build_orch()
    selector = MagicMock()
    selector.index_procedure_run = AsyncMock()
    orch.deps.tool_selector = selector
    run = OrchestrationRun(strategy_type="s", code="...", error="boom", utterance="q")
    await orch._index_approved_run(run, error_dialog="e", shown_names=set())
    selector.index_procedure_run.assert_not_awaited()
