"""ContextEnricher orchestration-code path: failure contract & no-op.

Since the single-wave ``_select_tools`` fallback was removed, ``explore()`` has
one path. These tests pin its two outcomes:

* a genuine code-generation failure **raises** :class:`ExplorationError`
  (fail-hard, no silent degradation), and the MCP client is still closed; and
* a legitimate "nothing to do" outcome (no actionable gaps, or only capability
  gaps) returns an empty / runnable-code-free :class:`ExplorationResult`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bamboo.agents.context_enricher import (
    ContextEnricher,
    ExplorationError,
    ExplorationResult,
)
from bamboo.mcp.base import McpTool

# A resolvable gap, in the shape _parse_gaps expects.
_GAP_JSON = json.dumps([{"gap": "fetch parent task data", "resolvable": True}])


class _FakeLLM:
    """Returns queued ``.content`` strings on successive ``ainvoke`` calls."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return SimpleNamespace(content=self._responses.pop(0))


class _FakeClient:
    """Minimal MCP client tracking connect/close and exposing one read tool."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    def list_tools(self) -> list[McpTool]:
        return [McpTool(name="read_a", description="d", parameters_schema={"type": "object"})]

    def task_data_tools(self) -> frozenset[str]:
        return frozenset()


def _enricher(llm: _FakeLLM) -> ContextEnricher:
    enr = ContextEnricher(_FakeClient(), io=None)
    enr._llm = llm  # inject; bypasses get_extraction_llm()

    async def _no_render(*_a, **_k) -> str:
        # Avoid the tokenizer/settings dependency — irrelevant to these tests.
        return ""

    enr._render_tools_description = _no_render  # type: ignore[method-assign]
    return enr


# ---------------------------------------------------------------------------
# _generate_orchestration_code: failure vs. legitimate no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_code_raises_on_unparseable_response():
    """Gaps found, but the code-gen response is garbage (no code, no gaps) → raise."""
    enr = _enricher(_FakeLLM(_GAP_JSON, "this is not JSON at all"))
    tools = enr._client.list_tools()
    with pytest.raises(ExplorationError):
        await enr._generate_orchestration_code({"jediTaskID": 1}, ["vague symptom"], tools)


@pytest.mark.asyncio
async def test_generate_code_no_actionable_gaps_returns_none():
    """Gap analysis returns nothing actionable → legitimate no-op (None), no raise."""
    enr = _enricher(_FakeLLM(json.dumps([])))
    tools = enr._client.list_tools()
    result = await enr._generate_orchestration_code({"jediTaskID": 1}, ["vague symptom"], tools)
    assert result is None


@pytest.mark.asyncio
async def test_generate_code_empty_code_with_capability_gaps_is_legitimate():
    """Empty code *with* capability gaps is legitimate (no tool can help) — not a failure."""
    codegen = json.dumps(
        {
            "orchestration_code": "",
            "capability_gaps": [
                {"investigation": "inspect site config", "suggested_tool_capability": "site_cfg"}
            ],
        }
    )
    enr = _enricher(_FakeLLM(_GAP_JSON, codegen))
    tools = enr._client.list_tools()
    result = await enr._generate_orchestration_code({"jediTaskID": 1}, ["vague symptom"], tools)
    assert result is not None
    code, gaps = result
    assert code == ""
    assert gaps and gaps[0]["investigation"] == "inspect site config"


# ---------------------------------------------------------------------------
# explore(): propagation & graceful no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explore_propagates_error_and_closes_client():
    """A code-gen failure surfaces out of explore(); the client is still closed."""
    enr = _enricher(_FakeLLM(_GAP_JSON, "garbage"))
    with pytest.raises(ExplorationError):
        await enr.explore({"jediTaskID": 1}, ["vague symptom — no marker"])
    assert enr._client.closed is True


@pytest.mark.asyncio
async def test_explore_returns_empty_result_when_no_gaps():
    """No actionable gaps → empty ExplorationResult, no raise, client closed."""
    enr = _enricher(_FakeLLM(json.dumps([])))
    result = await enr.explore({"jediTaskID": 1}, ["vague symptom — no marker"])
    assert isinstance(result, ExplorationResult)
    assert result.external_data == {}
    assert result.task_logs == {}
    assert result.tool_calls == []
    assert enr._client.closed is True
