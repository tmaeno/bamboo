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
import builtins as _builtins_module
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

_SAFE_BUILTINS = {
    name: getattr(_builtins_module, name)
    for name in (
        "len", "isinstance", "issubclass", "type",
        "dict", "list", "tuple", "set", "str", "int", "float", "bool",
        "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
        "any", "all", "min", "max", "sum", "abs", "round",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "None", "True", "False", "repr",
    )
    if hasattr(_builtins_module, name)
}

from bamboo.agents.knowledge_reviewer import _build_task_summary, _join_doc_hints
from bamboo.llm import (
    EXPLORATION_GAP_ANALYSIS_SYSTEM,
    EXPLORATION_GAP_ANALYSIS_USER,
    EXPLORER_TOOL_SELECTION_SYSTEM,
    EXPLORER_TOOL_SELECTION_USER,
    PROCEDURE_ORCHESTRATION_CODE_SYSTEM,
    PROCEDURE_ORCHESTRATION_CODE_USER,
    TOOL_ORCHESTRATION_CODE_SYSTEM,
    TOOL_ORCHESTRATION_CODE_USER,
    get_extraction_llm,
)
from bamboo.mcp.base import McpClient, McpTool  # noqa: F401 (McpTool used in type hints)
from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """Data fetched by the explorer to augment the next extraction pass.

    Attributes:
        task_logs:        New log content keyed by a stable source label
                          (``"explorer:error_dialog:<url>"``).
                          Merged into ``task_logs`` in ``process_knowledge()``.
        external_data:    Structured data returned by non-log tools.
                          Keys: ``"parent_task"``, ``"retry_chain"``,
                          ``"jobs_summary"``.
                          Merged (shallow) into ``external_data``.
        tool_calls:       Record of which tools were selected and why, for
                          observability (logging / metadata).
        capability_gaps:  Investigation directions the planner identified that
                          no available tool can address. Each entry has
                          ``"investigation"`` and ``"suggested_tool_capability"``.
                          Populated only when the orchestration-code path runs.
    """

    task_logs: dict[str, str] = field(default_factory=dict)
    external_data: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    capability_gaps: list[dict] = field(default_factory=list)


class ToolProxy:
    """Exposes MCP tools as async methods for LLM-generated orchestration code.

    ``task_data`` is pre-injected for tools that accept it, so the generated
    code never handles it directly.  Any tool name the LLM produces is
    forwarded to the MCP client; unknown names fail at runtime (caught by
    :meth:`ContextEnricher._run_orchestration_code`).
    """

    def __init__(
        self,
        client,
        task_data: dict[str, Any],
        task_data_tool_names: frozenset[str],
        call_log: list[str],
    ) -> None:
        self._client = client
        self._task_data = task_data
        self._td_names = task_data_tool_names
        self._log = call_log

    def __getattr__(self, name: str):
        async def call(**kwargs):
            self._log.append(name)
            if name in self._td_names:
                kwargs["task_data"] = self._task_data
            return await self._client.execute(name, **kwargs)
        return call


class ContextEnricher:
    """LLM-driven single-pass source explorer.

    Given the reviewer's issues list and the current task data, the explorer:

    1. Identifies information gaps from the issues (LLM call 1, exploratory path
       only — procedure-driven path treats issues as gaps directly).
    2. Generates Python orchestration code that calls the appropriate MCP tools,
       plus a list of ``capability_gaps`` for investigations no tool can address
       (LLM call 2).
    3. Executes the code in a sandboxed namespace via
       :meth:`_run_orchestration_code`.

    The orchestration-code path supports dependent tool chains natively (e.g.
    ``find_similar_successful_tasks`` → ``get_successful_job_logs(task_id=…)``).
    Falls back to :meth:`_select_tools` (single-wave LLM-driven selection) if
    code generation fails.

    Args:
        mcp_client:       The :class:`~bamboo.mcp.base.McpClient` that provides
                          available tools and executes them.
        source_navigator: Optional source-code navigator (used for the
                          ``source_navigator`` virtual tool).
    """

    def __init__(
        self,
        mcp_client: McpClient,
        source_navigator=None,
    ) -> None:
        self._client = mcp_client
        self._source_navigator = source_navigator
        self._llm = None  # lazy — same pattern as KnowledgeAccumulator

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_extraction_llm()  # temperature=0, deterministic JSON
        return self._llm

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
    ) -> ExplorationResult:
        """Single select-and-fetch pass.

        1. List available tools from the MCP client.
        2. One LLM call to select which tools to invoke (or generate
           orchestration code, or build a static plan, depending on path).
        3. Execute selected tools concurrently with ``asyncio.gather``.
        4. Merge results into an :class:`ExplorationResult`.

        Returns an empty :class:`ExplorationResult` (no tool calls) if the
        LLM returns an empty selection, if no issues are provided, or if any
        internal error occurs (fail-open).

        Args:
            task_data:         Structured task fields from PanDA.
            review_issues:     Issue strings from the previous
                               :class:`~bamboo.agents.knowledge_reviewer.ReviewResult`.
            doc_hints:         Domain documentation passed to the planner.
            skip_gap_analysis: When True, treats ``review_issues`` as concrete
                               procedure instructions and uses the static
                               plan path. Otherwise uses the orchestration-code
                               path which natively supports dependent chains.
        """
        if not review_issues or not task_data:
            return ExplorationResult()

        await self._client.connect()
        try:
            tools = self._filtered_tools()
            task_data_tool_names = self._client.task_data_tools()

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
                self._merge_tool_result("request_human_input", result, out, task_data=task_data)
            elif procedure_issues:
                logger.info(
                    "ContextEnricher: procedure gap found but request_human_input "
                    "not available — skipping"
                )
                say("  Procedure gap: request_human_input not available.")

            if not other_issues:
                return out

            # ── Primary path: orchestration code (handles both exploratory and
            #     procedure-driven flows; the mode parameter switches prompts).
            result = await self._generate_orchestration_code(
                task_data, other_issues, tools,
                doc_hints=doc_hints,
                mode="procedure" if skip_gap_analysis else "exploratory",
                skip_gap_analysis=skip_gap_analysis,
            )
            if result is not None:
                code, capability_gaps = result
                out.capability_gaps = capability_gaps
                raw, called = await self._run_orchestration_code(
                    code, task_data, task_data_tool_names
                )
                for name in called:
                    out.tool_calls.append({"tool": name, "via": "orchestration"})
                for key, value in raw.items():
                    label = f"orchestration:{key}"
                    out.external_data[label] = value
                    if isinstance(value, str) and value:
                        show_block(label, value[:2000])
                return out
            say("Explorer: code generation failed — falling back to direct tool selection.")

            # ── Fallback: single-wave concurrent path ─
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
                self._merge_tool_result(tc["tool"], result, out, task_data=task_data)

            return out
        finally:
            await self._client.close()

    # ------------------------------------------------------------------
    # Private: planning (gap analysis + orchestration-code generation)
    # ------------------------------------------------------------------

    async def _analyse_gaps(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        tools: list[McpTool],
        doc_hints: dict[str, str] | None = None,
    ) -> list[str]:
        """LLM call 1: convert reviewer issues into structured gap descriptions.

        Returns a list of plain-English gap strings (the ``"gap"`` field from
        each JSON object), filtering to only those marked ``"resolvable": true``.
        Returns ``[]`` on parse error.
        """
        task_summary = _build_task_summary(task_data)
        tools_description = _build_tools_description(tools)
        issues_text = "\n".join(f"- {i}" for i in review_issues)

        user_content = EXPLORATION_GAP_ANALYSIS_USER.format(
            review_issues=issues_text,
            task_summary=task_summary,
            tools_description=tools_description,
            domain_hints=_join_doc_hints(doc_hints),
        )
        messages = [
            SystemMessage(content=EXPLORATION_GAP_ANALYSIS_SYSTEM),
            HumanMessage(content=user_content),
        ]

        say("Analysing information gaps...")
        with thinking("Analysing gaps"):
            response = await self.llm.ainvoke(messages)

        gaps = _parse_gaps(response.content)
        if gaps:
            show_block(
                "planner: gap analysis",
                "\n".join(f"• {g}" for g in gaps),
            )
        return gaps

    async def _generate_orchestration_code(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        tools: list[McpTool],
        doc_hints: dict[str, str] | None = None,
        *,
        mode: str = "exploratory",
        skip_gap_analysis: bool = False,
    ) -> tuple[str, list[dict]] | None:
        """Generate orchestration code plus capability_gaps.

        When ``skip_gap_analysis`` is False (exploratory path), phase 1 runs
        gap analysis. When True (procedure-driven path), ``review_issues`` are
        treated as the gaps directly. Phase 2 asks the LLM to emit a JSON
        object with ``orchestration_code`` and ``capability_gaps``. The prompt
        template differs by ``mode``: ``"exploratory"`` uses
        :data:`TOOL_ORCHESTRATION_CODE_SYSTEM` (creative investigation);
        ``"procedure"`` uses :data:`PROCEDURE_ORCHESTRATION_CODE_SYSTEM`
        (execute procedure verbatim, no speculation).

        Returns ``(code, capability_gaps)`` on success, or ``None`` on any
        failure (fail-open — caller falls back to :meth:`_select_tools`).
        """
        if not review_issues or not tools:
            say("Explorer: no review issues or no tools available — skipping orchestration.")
            return None
        try:
            if skip_gap_analysis:
                gaps = list(review_issues)
            else:
                gaps = await self._analyse_gaps(task_data, review_issues, tools, doc_hints)
            if not gaps:
                say("Planner found no actionable gaps.")
                return None

            if mode == "procedure":
                system_content = PROCEDURE_ORCHESTRATION_CODE_SYSTEM
                user_template = PROCEDURE_ORCHESTRATION_CODE_USER
            else:
                system_content = TOOL_ORCHESTRATION_CODE_SYSTEM
                user_template = TOOL_ORCHESTRATION_CODE_USER
            user_content = user_template.format(
                gaps="\n".join(f"- {g}" for g in gaps),
                task_summary=_build_task_summary(task_data),
                tools_description=_build_tools_description(tools),
            )
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
            say(
                f"Generating {'procedure-driven' if mode == 'procedure' else 'exploratory'} "
                "orchestration code..."
            )
            with thinking("Generating code"):
                response = await self.llm.ainvoke(messages)

            code, capability_gaps = _parse_orchestration_response(response.content)
            if not code:
                return None
            show_block("planner: orchestration code", code)
            if capability_gaps:
                show_block(
                    "planner: capability gaps",
                    "\n".join(
                        f"• {g.get('investigation', '')} "
                        f"[needs: {g.get('suggested_tool_capability', '')}]"
                        for g in capability_gaps
                    ),
                )
            return code, capability_gaps
        except Exception as exc:
            logger.warning("ContextEnricher._generate_orchestration_code: failed (%s)", exc)
            say(f"Explorer: code generation raised {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_orchestration_code(
        self,
        code: str,
        task_data: dict[str, Any],
        task_data_tool_names: frozenset[str],
    ) -> tuple[dict[str, Any], list[str]]:
        """Execute LLM-generated orchestration code in a sandboxed namespace.

        The code runs as the body of ``async def _fn(tools, asyncio)`` with
        only :data:`_SAFE_BUILTINS` available.  Any exception (syntax, runtime,
        or timeout) is logged and an empty result is returned (fail-open).

        Returns:
            Tuple ``(result, call_log)`` where ``result`` is the dict the
            orchestration code returned (or ``{}`` on any failure) and
            ``call_log`` is the list of tool names that were invoked through
            the proxy. ``call_log`` may be partially populated if a runtime
            exception interrupted execution after some calls succeeded.
        """
        call_log: list[str] = []
        proxy = ToolProxy(self._client, task_data, task_data_tool_names, call_log)
        namespace: dict[str, Any] = {"asyncio": asyncio, "__builtins__": _SAFE_BUILTINS}
        indented = "\n".join(f"    {line}" for line in code.splitlines())
        full_code = f"async def _fn(tools, asyncio):\n{indented}"
        try:
            exec(full_code, namespace)  # noqa: S102
        except SyntaxError as exc:
            logger.warning("ContextEnricher: orchestration code has syntax error: %s", exc)
            return {}, call_log
        try:
            result = await asyncio.wait_for(namespace["_fn"](proxy, asyncio), timeout=600)
        except asyncio.TimeoutError:
            logger.warning("ContextEnricher: orchestration code timed out after 600 s")
            return {}, call_log
        except Exception as exc:
            logger.warning("ContextEnricher: orchestration code raised: %s", exc)
            return {}, call_log
        if call_log:
            say(f"  orchestration called: {', '.join(call_log)}")
        if not isinstance(result, dict):
            logger.warning(
                "ContextEnricher: orchestration code returned %r — expected dict",
                type(result).__name__,
            )
            return {}, call_log
        return result, call_log

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

        user_content = EXPLORER_TOOL_SELECTION_USER.format(
            review_issues=issues_text,
            task_summary=task_summary,
            tools_description=tools_description,
        )
        messages = [
            SystemMessage(content=EXPLORER_TOOL_SELECTION_SYSTEM),
            HumanMessage(content=user_content),
        ]

        try:
            say("Selecting additional data sources to fetch...")
            with thinking("Selecting tools"):
                response = await self.llm.ainvoke(messages)
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
        task_data: dict[str, Any] | None = None,
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
        elif tool_name == "fetch_brokerage_context":
            if isinstance(result, dict):
                out.external_data["tool:fetch_brokerage_context"] = result
                # Parse logs into a structured summary and surface at top level so the
                # final analysis LLM sees the key facts (terminal filter, per-site fates)
                # without having to synthesize them from dense raw log text.
                from bamboo.utils.log_filters import parse_brokerage_summary  # noqa: PLC0415
                sites_of_interest: list[str] = []
                td = task_data or {}
                site = (td.get("site") or "").strip() if isinstance(td.get("site"), str) else ""
                if site:
                    sites_of_interest.append(site)
                included = td.get("includedSite") or []
                if isinstance(included, str):
                    included = [s.strip() for s in included.split(",") if s.strip()]
                for s in included:
                    if s and s not in sites_of_interest:
                        sites_of_interest.append(s)
                summaries: dict[str, dict] = {}
                for url, log_text in (result.get("logs") or {}).items():
                    if not isinstance(log_text, str):
                        continue
                    parsed = parse_brokerage_summary(
                        log_text, sites_of_interest=sites_of_interest
                    )
                    if parsed:
                        summaries[url] = parsed
                if summaries:
                    out.external_data["brokerage_summary"] = summaries
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
    text = _strip_fences(response)
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


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from an LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text[text.find("```json") + 7: text.rfind("```")].strip()
    elif "```" in text:
        text = text[text.find("```") + 3: text.rfind("```")].strip()
    return text


def _parse_gaps(response: str) -> list[str]:
    """Parse the gap-analysis JSON response; return gap strings for resolvable gaps."""
    text = _strip_fences(response)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            logger.warning("ContextEnricher: gap-analysis response is not a list — ignored")
            return []
        return [
            str(item["gap"])
            for item in data
            if isinstance(item, dict) and item.get("resolvable", True) and item.get("gap")
        ]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("ContextEnricher: failed to parse gap-analysis response: %s", exc)
        return []


def _parse_orchestration_response(response: str) -> tuple[str, list[dict]]:
    """Parse the orchestration-code response (works for both prompt variants).

    Expects a JSON object with ``"orchestration_code"`` (string) and
    ``"capability_gaps"`` (list of dicts). Returns ``(code, capability_gaps)``;
    on parse error returns ``("", [])`` so the caller can fail open.
    """
    text = _strip_fences(response).strip()
    preview = text[:300] + ("…" if len(text) > 300 else "")
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            logger.warning(
                "ContextEnricher: orchestration response is not a dict — got %s. Raw: %r",
                type(data).__name__, preview,
            )
            say(
                f"Explorer: orchestration response is not a JSON object "
                f"(got {type(data).__name__})."
            )
            return "", []
        # Accept the canonical key plus known LLM-hallucinated abbreviations.
        # The first alias found wins. When a non-canonical alias is used we
        # warn but still honor the value, so a single hallucinated key name
        # doesn't burn the whole code-generation pass.
        _CODE_KEY_ALIASES = ("orchestration_code", "orchest_code", "code")
        code_field = None
        used_key = None
        for k in _CODE_KEY_ALIASES:
            if k in data:
                code_field = data[k]
                used_key = k
                break
        if used_key and used_key != "orchestration_code":
            logger.warning(
                "ContextEnricher: LLM used alias %r instead of 'orchestration_code' — accepting",
                used_key,
            )
            say(
                f"Explorer: LLM used alias {used_key!r} instead of "
                f"'orchestration_code' — accepting."
            )
        raw_gaps = data.get("capability_gaps", []) or []
        capability_gaps = [
            g for g in raw_gaps
            if isinstance(g, dict) and g.get("investigation")
        ]
        code = code_field if isinstance(code_field, str) else ""

        if not code:
            if used_key is None:
                reason = f"field omitted (looked for any of {list(_CODE_KEY_ALIASES)})"
            elif code_field is None:
                reason = f"field {used_key!r} is null"
            elif isinstance(code_field, str):
                reason = f"field {used_key!r} is empty string"
            else:
                reason = f"field {used_key!r} is {type(code_field).__name__}"
            keys = sorted(data.keys())
            logger.warning(
                "ContextEnricher: orchestration_code empty (%s). keys=%s gaps=%d raw=%r",
                reason, keys, len(capability_gaps), preview,
            )
            say(
                f"Explorer: orchestration_code empty ({reason}). "
                f"keys={keys}, capability_gaps={len(capability_gaps)}, "
                f"raw: {preview!r}"
            )
            if capability_gaps:
                show_block(
                    "planner: capability gaps (no code generated)",
                    "\n".join(
                        f"• {g.get('investigation', '')} "
                        f"[needs: {g.get('suggested_tool_capability', '')}]"
                        for g in capability_gaps
                    ),
                )

        if code:
            explanation = data.get("explanation")
            if isinstance(explanation, str) and explanation.strip():
                show_block(
                    "orchestration: argument-source explanation",
                    explanation,
                )
                logger.info(
                    "ContextEnricher: orchestration explanation: %s",
                    explanation,
                )

        return code, capability_gaps
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "ContextEnricher: failed to parse orchestration response: %s. Raw: %r",
            exc, preview,
        )
        say(
            f"Explorer: orchestration JSON parse failed ({exc}); "
            f"first 300 chars: {preview!r}"
        )
        return "", []
