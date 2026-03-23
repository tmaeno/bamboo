"""Source exploration agent: fetches additional data on reviewer rejection.

:class:`ExtraSourceExplorer` sits between the first reviewer rejection and the
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

from bamboo.llm import EXPLORER_TOOL_SELECTION_PROMPT, get_extraction_llm
from bamboo.mcp.base import McpClient, McpTool
from bamboo.utils.narrator import say, thinking

logger = logging.getLogger(__name__)

# Task fields forwarded to the LLM for tool-selection context.
_TASK_SUMMARY_KEYS = (
    "taskID",
    "status",
    "errorDialog",
    "retryID",
    "transUses",
    "prodSourceLabel",
    "taskName",
    "taskType",
    "nJobs",
    "nJobsFinished",
    "nJobsFailed",
)


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


class ExtraSourceExplorer:
    """LLM-driven single-pass source explorer.

    Given the reviewer's issues list and the current task data, asks the LLM
    which MCP tools to invoke, executes them concurrently, and returns an
    :class:`ExplorationResult` ready to be merged into the next extraction
    attempt.

    Args:
        mcp_client: The :class:`~bamboo.mcp.base.McpClient` that provides
                    available tools and executes them.
    """

    def __init__(self, mcp_client: McpClient) -> None:
        self._client = mcp_client
        self._llm = None  # lazy — same pattern as KnowledgeAccumulator

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_extraction_llm()  # temperature=0, deterministic JSON
        return self._llm

    async def explore(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
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
        """
        if not review_issues or not task_data:
            return ExplorationResult()

        tools = self._client.list_tools()
        tool_calls = await self._select_tools(task_data, review_issues, tools)

        if not tool_calls:
            logger.info("ExtraSourceExplorer: LLM selected no tools — nothing to explore")
            say("Explorer selected no additional tools.")
            return ExplorationResult()

        tool_names = [tc["tool"] for tc in tool_calls]
        logger.info(
            "ExtraSourceExplorer: executing %d tool call(s): %s",
            len(tool_calls),
            tool_names,
        )
        say(f"Explorer fetching from {len(tool_calls)} tool(s): {', '.join(tool_names)}.")

        coros = [
            self._client.execute(tc["tool"], **tc.get("args", {}))
            for tc in tool_calls
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        out = ExplorationResult(tool_calls=tool_calls)
        for tc, result in zip(tool_calls, results):
            if isinstance(result, BaseException):
                say(f"  {tc['tool']}: failed — {result}")
            else:
                say(f"  {tc['tool']}: done.")
            self._merge_tool_result(tc["tool"], result, out)

        return out

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
            logger.exception("ExtraSourceExplorer: LLM tool-selection call failed — failing open")
            return []

    def _merge_tool_result(
        self,
        tool_name: str,
        result: Any,
        out: ExplorationResult,
    ) -> None:
        """Route one tool result into the correct :class:`ExplorationResult` field."""
        if isinstance(result, BaseException):
            logger.warning(
                "ExtraSourceExplorer: tool %r raised %s — skipped", tool_name, result
            )
            return

        if tool_name == "fetch_error_dialog_logs":
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
        else:
            logger.warning("ExtraSourceExplorer: unrecognised tool result for %r — skipped", tool_name)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_task_summary(task_data: dict[str, Any]) -> str:
    """Return a compact JSON string of only the fields relevant to tool selection."""
    subset: dict[str, Any] = {k: task_data[k] for k in _TASK_SUMMARY_KEYS if k in task_data}
    # Truncate errorDialog — just enough to see if log URLs exist.
    if subset.get("errorDialog"):
        subset["errorDialog"] = str(subset["errorDialog"])[:500]
    return json.dumps(subset, indent=2, default=str)


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
            logger.warning("ExtraSourceExplorer: LLM returned non-list JSON — ignoring")
            return []
        valid = []
        for item in data:
            if isinstance(item, dict) and "tool" in item:
                valid.append(item)
        return valid
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("ExtraSourceExplorer: failed to parse tool-selection response: %s", exc)
        return []
