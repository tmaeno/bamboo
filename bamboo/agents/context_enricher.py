"""Source exploration agent: fetches additional data on reviewer rejection.

:class:`ContextEnricher` sits between the first reviewer rejection and the
second extraction attempt.  It asks an LLM which MCP tools to invoke (given
the reviewer's issues and available tools), executes them concurrently, and
returns an :class:`ExplorationResult` ready to be merged into the next
extraction pass.

The explorer fires **at most once** per accumulation run, and only when a
:class:`~bamboo.agents.knowledge_reviewer.KnowledgeReviewer` is configured.
Any tool failure is handled defensively: exceptions are logged and skipped so
the pipeline never stalls.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from bamboo.agents.knowledge_reviewer import _build_task_summary
from bamboo.llm import EXPLORER_TOOL_SELECTION_PROMPT, get_extraction_llm
from bamboo.mcp.base import McpClient, McpTool  # noqa: F401 (McpTool used in type hints)
from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Data fetched by the explorer to augment the next extraction pass.

    Attributes:
        task_logs:     New log content keyed by a stable source label
                       (``"explorer:error_dialog:<url>"``).
                       Merged into ``task_logs`` in ``process_knowledge()``.
        external_data: Structured data returned by non-log tools.
                       Keys: ``"parent_task"``, ``"retry_chain"``,
                       ``"jobs_summary"``.
                       Merged (shallow) into ``external_data``.
        tool_calls:    Record of which tools were selected and why, for
                       observability (logging / metadata).
    """

    task_logs: dict[str, str] = field(default_factory=dict)
    external_data: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ContextEnricher:
    """LLM-driven single-pass source explorer.

    Given the reviewer's issues list and the current task data, either follows
    an :class:`~bamboo.agents.exploration_planner.ExplorationPlan` (when a
    planner is configured) or falls back to asking the LLM directly which tools
    to call.  In both cases the selected tools are executed and their results
    merged into an :class:`ExplorationResult`.

    Args:
        mcp_client: The :class:`~bamboo.mcp.base.McpClient` that provides
                    available tools and executes them.
        planner:    Optional :class:`~bamboo.agents.exploration_planner.ExplorationPlanner`.
                    When provided, the explorer uses its two-phase plan
                    (gap analysis → sequential steps) instead of the single
                    ``_select_tools`` LLM call.  Falls back to ``_select_tools``
                    if the planner returns ``None``.
    """

    def __init__(
        self,
        mcp_client: McpClient,
        planner=None,
        source_navigator=None,
    ) -> None:
        self._client = mcp_client
        self._planner = planner
        self._source_navigator = source_navigator
        self._llm = None  # lazy — same pattern as KnowledgeAccumulator

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_extraction_llm()  # temperature=0, deterministic JSON
        return self._llm

    @property
    def planner(self):
        """The configured :class:`~bamboo.agents.exploration_planner.ExplorationPlanner`, or ``None``."""
        return self._planner

    def _filtered_tools(self) -> list[McpTool]:
        """Return tools from the client, excluding interactive tools when not on a tty."""
        import sys
        tools = self._client.list_tools()
        if not sys.stdout.isatty():
            tools = [t for t in tools if not t.requires_interaction]
        return tools

    def available_tools(self) -> list[McpTool]:
        """Return the filtered tool list for passing to the reviewer.

        The reviewer uses this to annotate issues with
        ``"→ resolvable with <tool_name>"``.
        """
        return self._filtered_tools()

    async def explore(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        doc_hints: dict[str, str] | None = None,
        skip_gap_analysis: bool = False,
        plan=None,
    ) -> ExplorationResult:
        """Single select-and-fetch pass.

        1. List available tools from the MCP client.
        2. One LLM call to select which tools to invoke.
        3. Execute selected tools concurrently with ``asyncio.gather``.
        4. Merge results into an :class:`ExplorationResult`.

        Returns an empty :class:`ExplorationResult` (no tool calls) if the
        LLM returns an empty selection, if no issues are provided, or if any
        internal error occurs (fail-open).

        Args:
            task_data:     Structured task fields from PanDA.
            review_issues: Issue strings from the previous
                           :class:`~bamboo.agents.knowledge_reviewer.ReviewResult`.
            plan:          Optional pre-built :class:`~bamboo.agents.exploration_planner.ExplorationPlan`.
                           When provided, skips all planning phases and executes the
                           plan steps directly.  ``review_issues`` is ignored.
        """
        if (not review_issues and plan is None) or not task_data:
            return ExplorationResult()

        await self._client.connect()
        try:
            tools = self._filtered_tools()
            task_data_tool_names = self._client.task_data_tools()

            # ── Pre-built plan: skip all planning, execute steps directly ──────────────
            if plan is not None:
                out = ExplorationResult()
                if plan.steps:
                    all_tool_calls = [tc for step in plan.steps for tc in step.tool_calls]
                    logger.info(
                        "ContextEnricher: executing pre-built plan — %d step(s), %d tool call(s)",
                        len(plan.steps),
                        len(all_tool_calls),
                    )
                    out.tool_calls.extend(all_tool_calls)
                    for i, step in enumerate(plan.steps, 1):
                        say(f"  [step {i}/{len(plan.steps)}] {step.reason}")
                        coros = [
                            self._tool_coro(tc, task_data, task_data_tool_names)
                            for tc in step.tool_calls
                        ]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        for tc, result in zip(step.tool_calls, results):
                            if isinstance(result, BaseException):
                                say(f"    {tc['tool']}: failed — {result}")
                            else:
                                say(f"    {tc['tool']}: done.")
                                if isinstance(result, str) and result:
                                    show_block(tc["tool"], result)
                            self._merge_tool_result(tc["tool"], result, out)
                return out

            # ── Hard-route: "No investigation procedure captured" → request_human_input ──
            # Deterministic code routing — the LLM planner consistently mis-routes this
            # gap to data-fetching tools.  Handle it in Python before the planner runs.
            _PROCEDURE_MARKER = "no investigation procedure captured"
            procedure_issues = [i for i in review_issues if _PROCEDURE_MARKER in i.lower()]
            other_issues = [i for i in review_issues if _PROCEDURE_MARKER not in i.lower()]
            human_input_tool = next((t for t in tools if t.name == "request_human_input"), None)

            out = ExplorationResult()

            if procedure_issues and human_input_tool:
                say("Requesting human input for missing investigation procedure...")
                _prompt = (
                    "Please describe the investigation steps that were actually performed "
                    "to diagnose this task failure — what was checked, what commands were "
                    "run, and what confirmed the root cause."
                )
                tc = {"tool": "request_human_input", "args": {"prompt": _prompt}}
                out.tool_calls.append(tc)
                try:
                    result = await self._client.execute("request_human_input", prompt=_prompt)
                    say("  request_human_input: done.")
                except Exception as exc:
                    say(f"  request_human_input: failed — {exc}")
                    result = exc
                self._merge_tool_result("request_human_input", result, out)
            elif procedure_issues:
                logger.info(
                    "ContextEnricher: procedure gap found but request_human_input "
                    "not available — skipping"
                )
                say("  Procedure gap: request_human_input not available.")

            if not other_issues:
                return out

            # ── Primary path: planner ────
            if self._planner is not None:
                plan = await self._planner.plan(
                    task_data, other_issues, tools,
                    doc_hints=doc_hints, skip_gap_analysis=skip_gap_analysis,
                )
                if plan is not None and plan.steps:
                    all_tool_calls = [tc for step in plan.steps for tc in step.tool_calls]
                    logger.info(
                        "ContextEnricher: executing plan — %d step(s), %d tool call(s)",
                        len(plan.steps),
                        len(all_tool_calls),
                    )
                    out.tool_calls.extend(all_tool_calls)
                    for i, step in enumerate(plan.steps, 1):
                        say(f"  [step {i}/{len(plan.steps)}] {step.reason}")
                        coros = [
                            self._tool_coro(tc, task_data, task_data_tool_names)
                            for tc in step.tool_calls
                        ]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        for tc, result in zip(step.tool_calls, results):
                            if isinstance(result, BaseException):
                                say(f"    {tc['tool']}: failed — {result}")
                            else:
                                say(f"    {tc['tool']}: done.")
                                if isinstance(result, str) and result:
                                    show_block(tc["tool"], result)
                            self._merge_tool_result(tc["tool"], result, out)
                    return out
                if plan is not None and not plan.steps:
                    logger.info(
                        "ContextEnricher: planner found no steps — falling back to _select_tools"
                    )
                    say("Explorer: planner found no steps — falling back to direct tool selection.")

            # ── Fallback: single-wave concurrent path (no planner / plan=None) ─
            tool_calls = await self._select_tools(task_data, other_issues, tools)
            if not tool_calls:
                logger.info("ContextEnricher: LLM selected no tools — nothing to explore")
                say("Explorer selected no additional tools.")
                return out

            tool_names = [tc["tool"] for tc in tool_calls]
            logger.info(
                "ContextEnricher: executing %d tool call(s): %s",
                len(tool_calls),
                tool_names,
            )
            say(f"Explorer fetching from {len(tool_calls)} tool(s): {', '.join(tool_names)}.")

            coros = [self._tool_coro(tc, task_data, task_data_tool_names) for tc in tool_calls]
            results = await asyncio.gather(*coros, return_exceptions=True)

            out.tool_calls.extend(tool_calls)
            for tc, result in zip(tool_calls, results):
                if isinstance(result, BaseException):
                    say(f"  {tc['tool']}: failed — {result}")
                else:
                    say(f"  {tc['tool']}: done.")
                    if isinstance(result, str) and result:
                        show_block(tc["tool"], result)
                self._merge_tool_result(tc["tool"], result, out)

            return out
        finally:
            await self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _select_tools(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        tools: list[McpTool],
    ) -> list[dict[str, Any]]:
        """Ask the LLM which tools to call; return parsed list or [] on error."""
        task_summary = _build_task_summary(task_data)
        tools_description = _build_tools_description(tools)
        issues_text = "\n".join(f"- {i}" for i in review_issues)

        prompt = EXPLORER_TOOL_SELECTION_PROMPT.format(
            review_issues=issues_text,
            task_summary=task_summary,
            tools_description=tools_description,
        )

        try:
            say("Selecting additional data sources to fetch...")
            with thinking("Working"):
                response = await self.llm.ainvoke(prompt)
            return _parse_tool_calls(response.content)
        except Exception:
            logger.exception("ContextEnricher: LLM tool-selection call failed — failing open")
            return []

    def _tool_coro(
        self,
        tc: dict,
        task_data: dict[str, Any],
        task_data_tool_names: frozenset[str],
    ):
        """Return the coroutine for one planned tool call.

        Routes ``search_panda_server_source`` to :attr:`_source_navigator` when
        one is configured, otherwise falls through to the MCP client.
        """
        if tc["tool"] == "search_panda_server_source" and self._source_navigator is not None:
            query = tc.get("args", {}).get("query", "")
            return self._source_navigator.navigate(query)
        args = (
            {**tc.get("args", {}), "task_data": task_data}
            if tc["tool"] in task_data_tool_names
            else tc.get("args", {})
        )
        return self._client.execute(tc["tool"], **args)

    def _merge_tool_result(
        self,
        tool_name: str,
        result: Any,
        out: ExplorationResult,
    ) -> None:
        """Route one tool result into the correct :class:`ExplorationResult` field."""
        if isinstance(result, BaseException):
            logger.warning(
                "ContextEnricher: tool %r raised %s — skipped", tool_name, result
            )
            return

        if tool_name == "fetch_linked_log_files":
            if isinstance(result, dict):
                for url, content in result.items():
                    key = f"explorer:error_dialog:{url}"
                    out.task_logs[key] = content
        elif tool_name == "get_parent_task":
            if result is not None:
                out.external_data["parent_task"] = result
        elif tool_name == "get_retry_chain":
            if isinstance(result, list):
                out.external_data["retry_chain"] = result
        elif tool_name == "get_task_jobs_summary":
            if isinstance(result, dict):
                out.external_data["jobs_summary"] = result
        elif tool_name == "get_scout_job_details":
            if isinstance(result, list):
                out.external_data["representative_jobs"] = result
        elif tool_name == "get_task_input_datasets":
            if isinstance(result, list) and result:
                out.task_logs["jedi:input_datasets"] = json.dumps(result, indent=2, default=str)
        elif tool_name == "search_panda_server_source":
            if isinstance(result, str) and result:
                out.task_logs["panda_server:source_search"] = result
            elif isinstance(result, list) and result:
                out.task_logs["panda_server:source_search"] = json.dumps(result, indent=2)
        elif tool_name == "search_panda_docs":
            if isinstance(result, list) and result:
                out.task_logs["panda_docs:search"] = json.dumps(result, indent=2)
        elif tool_name == "request_human_input":
            if result and isinstance(result, str):
                out.task_logs["human_input:procedures"] = result
        else:
            # Unknown tool — likely from an external MCP server.
            # Store raw result in external_data so the LLM extractor receives
            # it as additional unstructured context on the next attempt.
            logger.debug(
                "ContextEnricher: storing raw result for unrecognised tool %r",
                tool_name,
            )
            out.external_data[f"tool:{tool_name}"] = result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------



def _build_tools_description(tools: list[McpTool]) -> str:
    """Format tool descriptors for the LLM tool-selection prompt."""
    parts = []
    for t in tools:
        params = ", ".join(t.parameters_schema.get("properties", {}).keys())
        parts.append(f"- {t.name}({params})\n  {t.description}")
    return "\n\n".join(parts)


def _parse_tool_calls(response: str) -> list[dict[str, Any]]:
    """Parse the LLM's JSON tool-selection response, failing open on any error."""
    text = response.strip()
    if "```json" in text:
        text = text[text.find("```json") + 7 : text.rfind("```")].strip()
    elif "```" in text:
        text = text[text.find("```") + 3 : text.rfind("```")].strip()
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning("ContextEnricher: LLM returned non-list JSON — ignoring")
            return []
        valid = []
        for item in data:
            if isinstance(item, dict) and "tool" in item:
                valid.append(item)
        return valid
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("ContextEnricher: failed to parse tool-selection response: %s", exc)
        return []
