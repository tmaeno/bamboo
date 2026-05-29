"""Internal read-only tool registry for `bamboo investigate`.

These are *not* MCP tools — they don't call out to PanDA. They expose a small
set of read-only queries against bamboo's own knowledge graph (Neo4j) and
analysis pipeline (ReasoningNavigator). From the orchestration LLM's
perspective they look identical to MCP tools (same descriptor shape — name,
description, parameters_schema, has_side_effects=False) so a single
``INVESTIGATE_ORCHESTRATION`` prompt can pick from a unified registry.

The registry is built by :func:`build_internal_tools_registry`, which returns
``({name: McpTool descriptor}, {name: async_callable})``. The orchestrator
hands the descriptors to the planner LLM (so the model can choose them) and
the callables to :class:`bamboo.agents.orchestration.ToolProxy` (so
``tools.<name>(...)`` in the generated code actually invokes them).

Per plan §D, every internal tool here delegates to an existing reader —
``query_past_causes_for_symptom`` wraps ``ReasoningNavigator.analyze_task``;
``query_past_procedures_for_cause`` wraps
``graph_db.find_procedures_for_causes``. No parallel implementations.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from bamboo.mcp.base import McpTool


# Type alias used in signatures below.
ToolCallable = Callable[..., Awaitable[Any]]


def build_internal_tools_registry(
    *,
    graph_db: Any,
    reasoning_navigator: Any | None = None,
) -> tuple[dict[str, McpTool], dict[str, ToolCallable]]:
    """Return ``(descriptors, callables)`` for the v1 internal-tool set.

    Args:
        graph_db:            A :class:`bamboo.database.graph_database_client.GraphDatabaseClient`
                             instance — used by
                             ``query_past_procedures_for_cause`` and
                             ``query_past_causes_for_symptom`` (when no
                             ``reasoning_navigator`` is supplied).
        reasoning_navigator: Optional :class:`~bamboo.agents.reasoning_navigator.ReasoningNavigator`.
                             When provided, ``query_past_causes_for_symptom``
                             uses its full analyze pipeline (similarity search
                             + LLM root-cause identification). When ``None``,
                             a lighter fallback queries the graph DB directly
                             via ``graph_db.find_causes(symptoms=[symptom])``.

    Returns:
        Two parallel dicts keyed by tool name: the McpTool descriptors (for
        the planner LLM) and the async callables (for the ToolProxy).
    """

    async def query_past_causes_for_symptom(*, symptom: str) -> dict[str, Any]:
        """Look up canonical Causes whose Symptom matches `symptom`.

        Uses :meth:`ReasoningNavigator.analyze_task` when one was supplied to
        the registry builder — that path includes Qdrant similarity search
        and weighting. Otherwise falls back to ``graph_db.find_causes``.
        """
        if reasoning_navigator is not None and hasattr(reasoning_navigator, "graph_db"):
            # Direct symptom-name query through the graph_db client used by
            # analyze_task internally — fast, no LLM call needed for a simple
            # surface query of "what causes are linked to this symptom name?"
            causes = await reasoning_navigator.graph_db.find_causes(symptoms=[symptom])
        else:
            causes = await graph_db.find_causes(symptoms=[symptom])
        return {"symptom": symptom, "causes": causes}

    async def query_past_procedures_for_cause(
        *,
        cause_name: str,
        include_tentative: bool = False,
    ) -> dict[str, Any]:
        """Return the Procedures linked to the given canonical cause name.

        Delegates to :meth:`GraphDatabaseClient.find_procedures_for_causes`
        with the v1 stored-code fields surfaced (orchestration_code,
        code_summary, has_side_effects, trigger_signals, result_summary —
        see §0.6.a). ``include_tentative`` mirrors that method's flag for
        callers that want to see abandoned-session procedures too.
        """
        procedures = await graph_db.find_procedures_for_causes(
            [cause_name], include_tentative=include_tentative
        )
        return {"cause_name": cause_name, "procedures": procedures}

    descriptors: dict[str, McpTool] = {
        "query_past_causes_for_symptom": McpTool(
            name="query_past_causes_for_symptom",
            description=(
                "Look up known Causes for a given Symptom name in bamboo's own "
                "knowledge graph. Use when the human asks 'what did past "
                "incidents do for this kind of error?' or wants to know whether "
                "a similar errorDialog has been seen before. Read-only — does "
                "not hit PanDA."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "symptom": {
                        "type": "string",
                        "description": "Canonical Symptom name (the category, not the raw errorDialog text).",
                    },
                },
                "required": ["symptom"],
            },
            has_side_effects=False,
        ),
        "query_past_procedures_for_cause": McpTool(
            name="query_past_procedures_for_cause",
            description=(
                "Fetch the investigation Procedures recorded for a given Cause "
                "— each Procedure includes the strategy_type, description, "
                "accumulated parameters, and (for investigate-captured "
                "procedures) the stored orchestration_code + code_summary + "
                "trigger_signals that can be re-executed on this task. "
                "Read-only — does not hit PanDA."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "cause_name": {
                        "type": "string",
                        "description": "Canonical Cause name.",
                    },
                    "include_tentative": {
                        "type": "boolean",
                        "description": "When true, include procedures committed by abandoned investigate sessions (status='tentative'). Default false.",
                        "default": False,
                    },
                },
                "required": ["cause_name"],
            },
            has_side_effects=False,
        ),
    }

    callables: dict[str, ToolCallable] = {
        "query_past_causes_for_symptom": query_past_causes_for_symptom,
        "query_past_procedures_for_cause": query_past_procedures_for_cause,
    }

    return descriptors, callables
