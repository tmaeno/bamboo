"""Unit tests for `bamboo.agents.helpers.tool_selection`.

Phase 0 covers the shared ``render_tools`` renderer: that it reproduces the
historical per-path output verbatim (the safe-refactor guarantee), that the two
public call sites delegate to it, and that the greedy token-budget fill never
exceeds the budget and reports what it omitted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bamboo.agents.context_enricher import _build_tools_description
from bamboo.agents.helpers.deps import derive_selector_params
from bamboo.agents.helpers.tool_selection import (
    RetrievalUnavailable,
    Selection,
    ToolSelector,
    _content_hash,
    _merge_sources,
    render_tools,
)
from bamboo.agents.investigation_session import _format_available_tools
from bamboo.mcp.base import McpTool

# --- reference (legacy) implementations, frozen here as a regression oracle ---


def _legacy_investigate(tools: list[dict]) -> str:
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        access = "external" if t.get("external_access") else "internal"
        rw = "read-only" if t.get("read_only", True) else "MODIFIES-STATE"
        schema = json.dumps(t.get("parameters_schema") or {}, indent=None)
        lines.append(
            f"- {t['name']} [{access}, {rw}]: {t.get('description', '').strip()}\n"
            f"    args schema: {schema}"
        )
    return "\n".join(lines)


def _legacy_explorer(tools: list[McpTool]) -> str:
    parts = []
    for t in tools:
        params = ", ".join(t.parameters_schema.get("properties", {}).keys())
        parts.append(f"- {t.name}({params})\n  {t.description}")
    return "\n\n".join(parts)


def _dicts() -> list[dict]:
    return [
        {
            "name": "get_parent_task",
            "description": "  Fetch the parent task.  ",
            "parameters_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}},
            "external_access": True,
            "read_only": True,
        },
        {
            "name": "kill_task",
            "description": "Kill a task.",
            "parameters_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}},
            "external_access": True,
            "read_only": False,
        },
        {
            "name": "query_graph",
            "description": "Internal graph lookup.",
            "parameters_schema": {"properties": {}},
            "external_access": False,
            "read_only": True,
        },
    ]


def _mcp_tools() -> list[McpTool]:
    return [
        McpTool(
            name="get_parent_task",
            description="Fetch the parent task.",
            parameters_schema={"type": "object", "properties": {"task_id": {}}},
        ),
        McpTool(
            name="search_docs",
            description="Search the docs.",
            parameters_schema={"properties": {"query": {}, "limit": {}}},
        ),
    ]


# --- behaviour-preserving equivalence ---------------------------------------


def test_investigate_render_matches_legacy():
    tools = _dicts()
    text, omitted = render_tools(tools, style="investigate")
    assert omitted == []
    assert text == _legacy_investigate(tools)
    # and the public call site delegates identically
    assert _format_available_tools(tools) == _legacy_investigate(tools)


def test_explorer_render_matches_legacy():
    tools = _mcp_tools()
    text, omitted = render_tools(tools, full_schema_for=set(), style="explorer")
    assert omitted == []
    assert text == _legacy_explorer(tools)
    assert _build_tools_description(tools) == _legacy_explorer(tools)


def test_empty_inputs_preserve_sentinels():
    assert render_tools([], style="investigate") == ("(none)", [])
    assert render_tools([], full_schema_for=set(), style="explorer") == ("", [])
    assert _format_available_tools([]) == "(none)"
    assert _build_tools_description([]) == ""


# --- schema-on-demand --------------------------------------------------------


def test_full_schema_for_subset_renders_schema_only_for_named():
    tools = _dicts()
    text, _ = render_tools(tools, full_schema_for={"get_parent_task"}, style="investigate")
    # named tool keeps its full JSON schema
    assert "args schema:" in text
    assert '"task_id"' in text.split("kill_task")[0]
    # an un-named tool is rendered compact (args names only, no schema line)
    kill_block = text.split("- kill_task")[1].splitlines()
    assert kill_block[1].strip().startswith("args: (")


# --- greedy token budget -----------------------------------------------------


def test_budget_none_renders_everything():
    tools = _dicts()
    text, omitted = render_tools(tools, style="investigate", token_budget=None)
    assert omitted == []
    assert text == _legacy_investigate(tools)


def test_budget_omits_overflow_and_never_exceeds():
    tools = _dicts()
    full = render_tools(tools, style="investigate")[0]
    # budget that fits the first block but not all three (count_tokens = char count)
    first_len = len(render_tools(tools[:1], style="investigate")[0])
    budget = first_len + 5
    text, omitted = render_tools(
        tools, style="investigate", token_budget=budget, count_tokens=len
    )
    assert len(text) <= budget
    assert omitted  # something was dropped
    assert len(text) < len(full)
    # dropped names are reported and absent from the rendered text
    for name in omitted:
        assert name not in text


def test_budget_keeps_at_least_first_even_if_oversized():
    tools = _dicts()
    text, omitted = render_tools(
        tools, style="investigate", token_budget=1, count_tokens=len
    )
    assert tools[0]["name"] in text
    assert set(omitted) == {t["name"] for t in tools[1:]}


def test_base_tokens_counts_against_budget():
    tools = _dicts()
    first_len = len(render_tools(tools[:1], style="investigate")[0])
    # with base_tokens consuming the budget, only the mandatory first tool fits
    text, omitted = render_tools(
        tools,
        style="investigate",
        token_budget=first_len + 5,
        base_tokens=first_len,
        count_tokens=len,
    )
    assert tools[0]["name"] in text
    assert set(omitted) == {t["name"] for t in tools[1:]}


# --- ToolSelector: deterministic fakes --------------------------------------


class _FakeEmbeddings:
    """Cheap deterministic stand-in (the vector is irrelevant to the fake DB)."""

    def embed_query(self, text):
        return [float(len(text)), 1.0]

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


class _FakeVectorDB:
    def __init__(self, search_results=None, fail_sections=()):
        self.docs: dict = {}
        self.search_results = search_results or {}
        self.fail_sections = set(fail_sections)
        self.upserts: list = []

    async def get_document(self, vid):
        return self.docs.get(vid)

    async def upsert_section_vector(self, vid, emb, content, section, metadata):
        self.upserts.append((vid, content, section, dict(metadata)))
        self.docs[vid] = {"id": vid, "content": content, "metadata": dict(metadata)}
        return vid

    async def search_similar(self, q, limit=10, score_threshold=0.7, filter_conditions=None):
        section = (filter_conditions or {}).get("section")
        if section in self.fail_sections:
            raise RuntimeError("vector store down")
        return list(self.search_results.get(section, []))[:limit]


def _selector(vdb, **kw):
    return ToolSelector(vdb, _FakeEmbeddings(), **kw)


def _cat_row(name, score=0.9):
    return {"id": name, "score": score, "metadata": {"tool_name": name}}


def _trig_row(tools, score=0.9, weight=1.0):
    return {"id": ",".join(tools), "score": score, "metadata": {"tools": tools, "weight": weight}}


def _many_tools(n):
    return [
        {
            "name": f"tool_{i}",
            "description": f"does thing {i}",
            "parameters_schema": {"properties": {"x": {}}},
            "external_access": True,
            "read_only": True,
        }
        for i in range(n)
    ]


# --- _merge_sources ---


def test_merge_sources_interleaves_reserved_explore():
    out = _merge_sources(["a", "b", "c", "d"], ["x", "y", "z"], reserved_explore=2)
    assert out[:4] == ["a", "x", "b", "y"]
    assert set(out) == {"a", "b", "c", "d", "x", "y", "z"}


def test_merge_sources_cold_start_uses_source2():
    out = _merge_sources([], ["x", "y", "z"], reserved_explore=2)
    assert out == ["x", "y", "z"]


# --- ensure_index ---


async def test_ensure_index_only_embeds_new_or_changed():
    tools = _dicts()
    vdb = _FakeVectorDB()
    sel = _selector(vdb)
    seeded = tools[0]
    vid = ToolSelector._vid("toolcat", "ns", seeded["name"])
    vdb.docs[vid] = {"id": vid, "metadata": {"tool_hash": _content_hash(seeded)}}

    await sel.ensure_index(tools, config_namespace="ns")

    upserted = {m["tool_name"] for (_v, _c, _s, m) in vdb.upserts}
    assert seeded["name"] not in upserted  # unchanged -> skipped
    assert upserted == {tools[1]["name"], tools[2]["name"]}
    assert all(section == "ToolCatalogue" for (_v, _c, section, _m) in vdb.upserts)


async def test_ensure_index_noop_when_all_current():
    tools = _dicts()
    vdb = _FakeVectorDB()
    sel = _selector(vdb)
    for t in tools:
        vid = ToolSelector._vid("toolcat", "ns", t["name"])
        vdb.docs[vid] = {"id": vid, "metadata": {"tool_hash": _content_hash(t)}}
    await sel.ensure_index(tools, config_namespace="ns")
    assert vdb.upserts == []


# --- index_procedure_run ---


async def test_index_procedure_run_records_sorted_signature_and_weight():
    vdb = _FakeVectorDB()
    sel = _selector(vdb)
    await sel.index_procedure_run(
        tool_names=["get_retry_chain", "get_parent_task"],
        prompt_text="  why did the task fail?  ",
        config_namespace="ns",
        weight=2.0,
    )
    assert len(vdb.upserts) == 1
    _v, content, section, meta = vdb.upserts[0]
    assert section == "ProcedureTriggers"
    assert content == "why did the task fail?"
    assert meta["tools"] == ["get_parent_task", "get_retry_chain"]
    assert meta["weight"] == 2.0


async def test_index_procedure_run_noop_on_empty():
    vdb = _FakeVectorDB()
    sel = _selector(vdb)
    await sel.index_procedure_run(tool_names=[], prompt_text="x", config_namespace="ns")
    await sel.index_procedure_run(tool_names=["a"], prompt_text="   ", config_namespace="ns")
    assert vdb.upserts == []


# --- select ---


async def test_select_fail_hard_when_catalogue_retrieval_unavailable():
    tools = _many_tools(5)
    vdb = _FakeVectorDB(fail_sections={"ToolCatalogue"})
    sel = _selector(vdb)
    with pytest.raises(RetrievalUnavailable):
        await sel.select("symptom", tools, budget=1000, config_namespace="ns", count_tokens=len)


async def test_select_source1_failure_is_non_fatal():
    tools = _many_tools(5)
    vdb = _FakeVectorDB(
        search_results={"ToolCatalogue": [_cat_row("tool_1"), _cat_row("tool_2")]},
        fail_sections={"ProcedureTriggers"},
    )
    sel = _selector(vdb)
    out = await sel.select("symptom", tools, budget=10000, config_namespace="ns", count_tokens=len)
    assert isinstance(out, Selection)
    assert set(out.ordered) == {"tool_1", "tool_2"}


async def test_select_intersects_with_live_tools():
    tools = _many_tools(3)
    vdb = _FakeVectorDB(
        search_results={"ToolCatalogue": [_cat_row("tool_1"), _cat_row("ghost_tool")]},
    )
    sel = _selector(vdb)
    out = await sel.select("symptom", tools, budget=10000, config_namespace="ns", count_tokens=len)
    assert "ghost_tool" not in out.ordered
    assert "tool_1" in out.ordered


async def test_select_reserves_explore_slots_for_source2():
    tools = _many_tools(10)
    s1_tools = [f"tool_{i}" for i in range(8)]
    vdb = _FakeVectorDB(
        search_results={
            "ProcedureTriggers": [_trig_row(s1_tools)],
            "ToolCatalogue": [_cat_row("tool_9"), _cat_row("tool_0")],
        },
    )
    sel = _selector(vdb, reserved_explore=2)
    out = await sel.select("symptom", tools, budget=10000, config_namespace="ns", count_tokens=len)
    # the brand-new source-#2 tool is guaranteed an early (full-schema) slot
    assert "tool_9" in out.full_schema_names
    assert out.ordered.index("tool_9") < 4


async def test_select_budget_limits_full_schemas_and_drops_overflow():
    tools = _many_tools(20)
    rows = [_cat_row(f"tool_{i}", score=1.0 - i * 0.01) for i in range(20)]
    vdb = _FakeVectorDB(search_results={"ToolCatalogue": rows})
    sel = _selector(vdb)
    out = await sel.select("symptom", tools, budget=300, config_namespace="ns", count_tokens=len)
    assert out.full_schema_names  # at least one full
    assert len(out.full_schema_names) < 20  # not all full
    assert out.dropped  # overflow dropped
    assert set(out.ordered).isdisjoint(out.dropped)


async def test_select_max_full_schemas_caps_full_tier_under_generous_budget():
    # The relevance cap binds even when the token budget would allow far more full
    # schemas: only `max_full_schemas` get full, the rest fall through to compact.
    tools = _many_tools(20)
    rows = [_cat_row(f"tool_{i}", score=1.0 - i * 0.01) for i in range(20)]
    vdb = _FakeVectorDB(search_results={"ToolCatalogue": rows})
    sel = _selector(vdb, max_full_schemas=3)
    out = await sel.select("symptom", tools, budget=1_000_000, config_namespace="ns", count_tokens=len)
    assert len(out.full_schema_names) == 3  # capped, not budget-limited
    compact = [n for n in out.ordered if n not in out.full_schema_names]
    assert compact  # candidates beyond the cap are compact, not dropped
    assert not out.dropped  # generous budget => nothing dropped


# --- derive_selector_params (the single primary-knob → pool/clamp derivation) ---


def test_derive_selector_params_auto_pool_and_clamp():
    s = SimpleNamespace(tool_max_full_schemas=25, tool_retrieval_candidate_k=0, tool_reserved_explore=4)
    candidate_k, reserved_explore, max_full = derive_selector_params(s)
    assert (candidate_k, reserved_explore, max_full) == (75, 4, 25)  # 3×25, clamp no-op


def test_derive_selector_params_small_cap_hits_floor_and_clamps_reserved():
    s = SimpleNamespace(tool_max_full_schemas=2, tool_retrieval_candidate_k=0, tool_reserved_explore=4)
    candidate_k, reserved_explore, max_full = derive_selector_params(s)
    assert candidate_k == 40  # max(40, 3×2) floor
    assert reserved_explore == 2  # clamped to <= max_full_schemas


def test_derive_selector_params_explicit_candidate_k_overrides():
    s = SimpleNamespace(tool_max_full_schemas=25, tool_retrieval_candidate_k=120, tool_reserved_explore=4)
    candidate_k, _, _ = derive_selector_params(s)
    assert candidate_k == 120
