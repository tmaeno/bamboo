"""Two-phase exploration planner for :class:`~bamboo.agents.ExtraSourceExplorer`.

:class:`ExplorationPlanner` sits between :class:`~bamboo.agents.KnowledgeReviewer`
and :class:`~bamboo.agents.ExtraSourceExplorer`.  Given the reviewer's issues it
runs two sequential LLM calls:

1. **Gap analysis** — convert raw reviewer issues into precise, tool-neutral
   descriptions of what specific information is missing and why it matters.
2. **Step planning** — map each resolvable gap to one or more MCP tool calls,
   grouped into sequential *steps* (tools within a step run concurrently; steps
   run in order so a later step can rely on earlier steps having populated the
   extraction context).

The planner is **fail-open**: any error in either phase causes :meth:`plan` to
return ``None``, which signals :class:`ExtraSourceExplorer` to fall back to its
existing single-LLM-call ``_select_tools`` path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from bamboo.llm import (
    EXPLORATION_GAP_ANALYSIS_PROMPT,
    EXPLORATION_PLAN_PROMPT,
    get_extraction_llm,
)
from bamboo.mcp.base import McpClient, McpTool
from bamboo.utils.narrator import say, show_block, thinking

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """One sequential step in an :class:`ExplorationPlan`.

    All tool calls within a step run concurrently (``asyncio.gather``).
    Steps run in order; a later step may depend on earlier steps having
    populated ``task_logs`` / ``external_data``, but does not receive their
    return values directly.

    Attributes:
        reason:     Plain-English description of what this step fetches and
                    which gap it closes.
        tool_calls: List of ``{"tool": str, "args": dict}`` dicts — same
                    format as :func:`~bamboo.agents.extra_source_explorer._parse_tool_calls`.
    """

    reason: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExplorationPlan:
    """An ordered list of :class:`PlanStep` objects produced by :class:`ExplorationPlanner`.

    Attributes:
        gaps:  Structured gap descriptions from the gap-analysis phase,
               preserved for logging and observability.
        steps: Ordered execution groups.  Steps run sequentially; tool calls
               within each step run concurrently.
    """

    gaps: list[str] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)

    @property
    def total_tool_calls(self) -> int:
        """Total number of tool calls across all steps."""
        return sum(len(s.tool_calls) for s in self.steps)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class ExplorationPlanner:
    """Two-phase LLM planner: gap analysis then sequential step construction.

    Args:
        mcp_client: Used only to list available tools via :meth:`list_tools`.
                    The planner never executes tools.
    """

    def __init__(self, mcp_client: McpClient) -> None:
        self._client = mcp_client
        self._llm = None  # lazy — same pattern as KnowledgeAccumulator

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_extraction_llm()
        return self._llm

    async def plan(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        tools: list[McpTool],
    ) -> ExplorationPlan | None:
        """Run both planning phases and return a structured plan.

        Returns ``None`` on any failure so the caller can fall back to the
        existing single-LLM-call tool-selection path.

        Args:
            task_data:     Structured task fields from PanDA.
            review_issues: Issue strings from the previous
                           :class:`~bamboo.agents.KnowledgeReviewer` result.
            tools:         Available MCP tools (already listed by the caller).
        """
        if not review_issues or not tools:
            return None
        try:
            gaps = await self._analyse_gaps(task_data, review_issues, tools)
            if not gaps:
                say("Planner found no actionable gaps — falling back to direct tool selection.")
                return None
            steps = await self._build_plan(gaps, task_data, tools)
            plan = ExplorationPlan(gaps=gaps, steps=steps)
            say(
                f"Exploration plan: {len(plan.steps)} step(s), "
                f"{plan.total_tool_calls} tool call(s) total."
            )
            return plan
        except Exception as exc:
            logger.warning(
                "ExplorationPlanner: planning failed (%s) — falling back to _select_tools",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Private: phase 1 — gap analysis
    # ------------------------------------------------------------------

    async def _analyse_gaps(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        tools: list[McpTool],
    ) -> list[str]:
        """LLM call 1: convert reviewer issues into structured gap descriptions.

        Returns a list of plain-English gap strings (the ``"gap"`` field from
        each JSON object), filtering to only those marked ``"resolvable": true``.
        Returns ``[]`` on parse error.
        """
        from bamboo.agents.knowledge_reviewer import _build_task_summary  # noqa: PLC0415
        from bamboo.agents.extra_source_explorer import _build_tools_description  # noqa: PLC0415

        task_summary = _build_task_summary(task_data)
        tools_description = _build_tools_description(tools)
        issues_text = "\n".join(f"- {i}" for i in review_issues)

        prompt = EXPLORATION_GAP_ANALYSIS_PROMPT.format(
            review_issues=issues_text,
            task_summary=task_summary,
            tools_description=tools_description,
        )

        say("Analysing information gaps...")
        with thinking("Analysing gaps"):
            response = await self.llm.ainvoke(prompt)

        gaps = _parse_gaps(response.content)
        if gaps:
            show_block(
                "planner: gap analysis",
                "\n".join(f"• {g}" for g in gaps),
            )
        return gaps

    # ------------------------------------------------------------------
    # Private: phase 2 — step planning
    # ------------------------------------------------------------------

    async def _build_plan(
        self,
        gaps: list[str],
        task_data: dict[str, Any],
        tools: list[McpTool],
    ) -> list[PlanStep]:
        """LLM call 2: map gaps to ordered PlanSteps.

        Returns ``[]`` on parse error (triggers fallback in :meth:`plan`).
        """
        from bamboo.agents.knowledge_reviewer import _build_task_summary  # noqa: PLC0415
        from bamboo.agents.extra_source_explorer import _build_tools_description  # noqa: PLC0415

        task_summary = _build_task_summary(task_data)
        tools_description = _build_tools_description(tools)
        gaps_text = "\n".join(f"- {g}" for g in gaps)

        prompt = EXPLORATION_PLAN_PROMPT.format(
            gaps=gaps_text,
            task_summary=task_summary,
            tools_description=tools_description,
        )

        say("Building exploration plan...")
        with thinking("Planning"):
            response = await self.llm.ainvoke(prompt)

        return _parse_plan_steps(response.content)


# ---------------------------------------------------------------------------
# Private parse helpers
# ---------------------------------------------------------------------------


def _parse_gaps(response: str) -> list[str]:
    """Parse the gap-analysis JSON response; return gap strings for resolvable gaps."""
    text = _strip_fences(response)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning("ExplorationPlanner: gap-analysis response is not a list — ignored")
            return []
        return [
            str(item["gap"])
            for item in data
            if isinstance(item, dict) and item.get("resolvable", True) and item.get("gap")
        ]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("ExplorationPlanner: failed to parse gap-analysis response: %s", exc)
        return []


def _parse_plan_steps(response: str) -> list[PlanStep]:
    """Parse the plan JSON response into a list of :class:`PlanStep` objects."""
    text = _strip_fences(response)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning("ExplorationPlanner: plan response is not a list — ignored")
            return []
        steps = []
        for item in data:
            if not isinstance(item, dict):
                continue
            tool_calls = [
                tc for tc in item.get("tool_calls", [])
                if isinstance(tc, dict) and "tool" in tc
            ]
            if tool_calls:
                steps.append(PlanStep(
                    reason=str(item.get("reason", "")),
                    tool_calls=tool_calls,
                ))
        return steps
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("ExplorationPlanner: failed to parse plan response: %s", exc)
        return []


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from an LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text[text.find("```json") + 7: text.rfind("```")].strip()
    elif "```" in text:
        text = text[text.find("```") + 3: text.rfind("```")].strip()
    return text
