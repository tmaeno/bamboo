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

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.agents.knowledge_reviewer import _build_task_summary, _join_doc_hints
from bamboo.agents.helpers.orchestration import (
    SAFE_BUILTINS as _SAFE_BUILTINS,  # noqa: F401  (re-exported for any back-compat callers)
    ToolProxy as _SharedToolProxy,
    run_orchestration_code as _shared_run_orchestration_code,
)
from bamboo.agents.helpers.tool_selection import (
    RetrievalUnavailable,
    config_namespace,
    render_tools,
)
from bamboo.llm import (
    EXPLORATION_GAP_ANALYSIS_SYSTEM,
    EXPLORATION_GAP_ANALYSIS_USER,
    PROCEDURE_ORCHESTRATION_CODE_SYSTEM,
    PROCEDURE_ORCHESTRATION_CODE_USER,
    TOOL_ORCHESTRATION_CODE_SYSTEM,
    TOOL_ORCHESTRATION_CODE_USER,
    get_extraction_llm,
)
from bamboo.mcp.base import McpClient, McpTool  # noqa: F401 (McpTool used in type hints)
from bamboo.utils.narrator import error, say, show_block, thinking, warn

logger = logging.getLogger(__name__)


class ExplorationError(RuntimeError):
    """Raised when orchestration-code generation fails outright.

    Distinct from a legitimate "nothing to explore" outcome (no review issues, no
    available tools, or no actionable gaps), which returns an empty
    :class:`ExplorationResult`. The explorer raises this so a genuine
    planning/LLM/parse failure surfaces loudly — failing the current task —
    rather than silently continuing with incomplete data. Every ``explore()``
    caller catches it at a per-invocation boundary (e.g. ``bamboo populate`` marks
    the task failed and continues with the rest of the batch).
    """


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


# Re-export the shared ToolProxy under the historical name so any external
# importer of ``bamboo.agents.context_enricher.ToolProxy`` keeps working.
ToolProxy = _SharedToolProxy


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
    A legitimate "nothing to explore" outcome (no issues, no tools, or no
    actionable gaps) returns an empty :class:`ExplorationResult`; a genuine
    code-generation failure raises :class:`ExplorationError` (fail-hard, no
    silent degradation).

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
        io=None,
        tool_selector=None,
    ) -> None:
        self._client = mcp_client
        self._source_navigator = source_navigator
        # Optional InteractionIO. Decides whether interactive tools (e.g.
        # request_human_input) are offered; falls back to TTY detection when None.
        self._io = io
        self._llm = None  # lazy — same pattern as KnowledgeAccumulator
        # Optional ToolSelector for budget-gating the prompt on large catalogues
        # (None => render all tools, the historical behaviour). The explorer is
        # automatic, so it *reads* the validated source #1 index but never writes it.
        self._tool_selector = tool_selector
        self._tok = None
        self._cfg_ns = None
        self._tool_index_built = False

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_extraction_llm()  # temperature=0, deterministic JSON
        return self._llm

    def _tokenizer(self):
        if self._tok is None:
            from bamboo.llm.llm_client import get_token_counter  # noqa: PLC0415

            self._tok = get_token_counter()
        return self._tok

    def _config_namespace(self) -> str:
        if self._cfg_ns is None:
            from bamboo.config import get_settings  # noqa: PLC0415

            self._cfg_ns = config_namespace(get_settings())
        return self._cfg_ns

    async def _ensure_tool_index_once(self, tools) -> None:
        if self._tool_index_built or self._tool_selector is None:
            return
        try:
            await self._tool_selector.ensure_index(
                tools, config_namespace=self._config_namespace()
            )
        except Exception as exc:  # noqa: BLE001 — degrade rather than break exploration
            warn(f"tool-catalogue index build failed: {exc}")
        finally:
            self._tool_index_built = True

    async def _render_tools_description(self, tools, *, base_text: str, query: str) -> str:
        """Budget-gated compact tool list for the explorer's codegen prompts.

        Mirrors the investigate path but explorer-style (compact, schema-free) and
        source-#2-led. The explorer is automatic/best-effort, so on retrieval
        failure it **degrades** to a budget-truncated compact list rather than
        aborting (unlike the human-facing investigate path).
        """
        from bamboo.config import get_settings  # noqa: PLC0415
        from bamboo.llm.llm_client import resolve_context_window  # noqa: PLC0415

        tok = self._tokenizer()
        settings = get_settings()
        budget = max(
            0, resolve_context_window(settings) - tok(base_text) - settings.tool_budget_margin
        )
        all_names = {t.name for t in tools}
        text, omitted = render_tools(
            tools, full_schema_for=set(), style="explorer", token_budget=budget, count_tokens=tok
        )
        if not omitted:
            return text
        if self._tool_selector is None:
            # Budget-truncated with no selector — surface it instead of dropping silently.
            say(
                f"explorer tools: {len(all_names) - len(omitted)}/{len(all_names)} shown, "
                f"{len(omitted)} truncated to fit budget={budget} tok (no selector)",
                level=logging.DEBUG,
            )
            return text
        try:
            await self._ensure_tool_index_once(tools)
            selection = await self._tool_selector.select(
                query,
                tools,
                budget=budget,
                count_tokens=tok,
                config_namespace=self._config_namespace(),
                style="explorer",
            )
            by_name = {t.name: t for t in tools}
            ordered = [by_name[n] for n in selection.ordered if n in by_name]
            text, omitted = render_tools(
                ordered, full_schema_for=set(), style="explorer", token_budget=budget, count_tokens=tok
            )
            shown = {d.name for d in ordered} - set(omitted)
            dropped = sorted(all_names - shown)
            say(
                f"explorer tool selection: {len(shown)}/{len(all_names)} shown, "
                f"{len(dropped)} omitted (budget={budget} tok)"
                + (f" dropped={dropped}" if dropped else ""),
                level=logging.DEBUG,
            )
        except RetrievalUnavailable:
            warn("tool retrieval unavailable — using a budget-truncated tool list")
        return text

    def _filtered_tools(self) -> list[McpTool]:
        """Return the read-only tools the explorer's planner may use.

        The explorer only ever does automatic, unattended data-gathering (no
        operator turn-by-turn), so **state-changing tools are dropped** (those
        with ``read_only=False``) — the planner never even sees them, so it
        generates read-only code by construction (the ``ToolProxy`` allow-set in
        ``_run_orchestration_code`` is the runtime backstop). External PanDA
        *reads* (``read_only=True, external_access=True``) are kept — fetching
        data is exactly the explorer's job. Interactive tools
        (``request_human_input``) are also dropped when the frontend can't gather
        human input, using the injected ``InteractionIO.supports_interaction``
        when available (so the chat bot can offer them via thread replies); falls
        back to TTY detection for IO-less callers.
        """
        tools = [t for t in self._client.list_tools() if t.read_only]
        if self._io is not None:
            interactive = self._io.supports_interaction
        else:
            import sys
            interactive = sys.stdout.isatty()
        if not interactive:
            tools = [t for t in tools if not t.requires_interaction]
        return tools

    def available_tools(self) -> list[McpTool]:
        """Return the filtered tool list for passing to the reviewer.

        The reviewer uses this to annotate issues with
        ``"→ resolvable with <tool_name>"``.
        """
        return self._filtered_tools()

    def non_read_only_tool_names(self) -> frozenset[str]:
        """Names of the client's state-changing tools (``read_only=False``).

        For callers (e.g. :class:`ReasoningNavigator`) that statically pre-screen
        stored procedure code before unattended replay, so a state-changing
        procedure can be skipped + surfaced as a suggestion rather than attempted
        and refused at the ``ToolProxy`` boundary.
        """
        return frozenset(t.name for t in self._client.list_tools() if not t.read_only)

    async def run_stored_code(
        self,
        code: str,
        task_data: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Execute a pre-existing orchestration code block (no LLM regeneration).

        Used by ``ReasoningNavigator._run_investigation_via_procedures`` to
        prefer ``orchestration_code`` stored on Procedure nodes (captured by
        ``bamboo investigate``) over the explorer's regenerate-from-description
        path. Reuses the same sandbox / proxy / timeout as
        :meth:`_run_orchestration_code`.

        Returns:
            ``(result, call_log)`` — the dict the code returned (or ``{}`` on
            any failure) and the list of tool names invoked through the proxy.
        """
        await self._client.connect()
        task_data_tool_names = self._client.task_data_tools()
        return await self._run_orchestration_code(code, task_data, task_data_tool_names)

    async def explore(
        self,
        task_data: dict[str, Any],
        review_issues: list[str],
        doc_hints: dict[str, str] | None = None,
        skip_gap_analysis: bool = False,
    ) -> ExplorationResult:
        """Single generate-and-fetch pass via the orchestration-code path.

        1. List available tools from the MCP client.
        2. Generate orchestration code (gap analysis + code generation).
        3. Execute the code in a sandbox; merge results into an
           :class:`ExplorationResult`.

        Returns an empty :class:`ExplorationResult` (no tool calls) when there is
        legitimately nothing to explore — no issues/task data, no available tools,
        or the planner finds no actionable gaps. Raises :class:`ExplorationError`
        on a genuine code-generation failure (malformed/empty LLM response or an
        exception during planning), so callers fail hard rather than continue with
        incomplete data.

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
                    say("  request_human_input: done.", level=logging.DEBUG)
                except Exception as exc:
                    warn(f"request_human_input failed — {exc}")
                    result = exc
                if result and isinstance(result, str):
                    out.task_logs["human_input:procedures"] = result
            elif procedure_issues:
                say("  Procedure gap: request_human_input not available.", level=logging.DEBUG)

            if not other_issues:
                return out

            # ── Orchestration-code path (handles both exploratory and
            #     procedure-driven flows; the mode parameter switches prompts).
            #     Raises ExplorationError on a genuine generation failure.
            result = await self._generate_orchestration_code(
                task_data, other_issues, tools,
                doc_hints=doc_hints,
                mode="procedure" if skip_gap_analysis else "exploratory",
                skip_gap_analysis=skip_gap_analysis,
            )
            if result is None:
                # Legitimate no-op: the planner found nothing actionable to fetch.
                return out
            code, capability_gaps = result
            out.capability_gaps = capability_gaps
            if code:
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

        Returns:
            ``(code, capability_gaps)`` on success. ``code`` may be ``""`` when
            the planner produced only capability gaps (every gap needs a tool
            that does not exist) — a legitimate outcome with nothing to run.
            ``None`` when there is legitimately nothing to do (no issues/tools,
            or no actionable gaps).

        Raises:
            ExplorationError: on a genuine generation failure — the LLM returned
                neither runnable code nor any capability gap (malformed/empty
                response), or an exception occurred during planning. Fail-hard so
                the caller surfaces it rather than continuing with incomplete data.
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
            gaps_text = "\n".join(f"- {g}" for g in gaps)
            task_summary = _build_task_summary(task_data)
            base_text = system_content + "\n" + user_template.format(
                gaps=gaps_text, task_summary=task_summary, tools_description=""
            )
            tools_description = await self._render_tools_description(
                tools, base_text=base_text, query=gaps_text
            )
            user_content = user_template.format(
                gaps=gaps_text,
                task_summary=task_summary,
                tools_description=tools_description,
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
            if not code and not capability_gaps:
                # Genuine generation failure: neither runnable code nor a single
                # capability gap came back (malformed / empty response or parse
                # failure). Fail hard rather than silently degrade. (Empty code
                # *with* capability gaps is a legitimate "no tool can help"
                # outcome and falls through below.)
                raise ExplorationError(
                    "orchestration code generation produced no code and no "
                    "capability gaps"
                )
            if code:
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
        except ExplorationError:
            raise
        except Exception as exc:
            warn(f"code generation failed — {type(exc).__name__}: {exc}")
            raise ExplorationError(
                f"orchestration code generation failed: {type(exc).__name__}: {exc}"
            ) from exc

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

        Thin delegate to
        :func:`bamboo.agents.helpers.orchestration.run_orchestration_code` so the
        sandbox/proxy/exec mechanics are shared with ``bamboo investigate``.

        The explorer runs in **automatic** phases (populate, and the navigator's
        ``analyze_task`` hypothesis) with no operator watching turn-by-turn, so it
        is confined to **read-only** tools: ``allowed_tools`` is the read-only
        subset of the client's tools, and any state-changing call — whether from
        regenerated code, an aliased reference, or a replayed stored procedure —
        is refused at the :class:`~bamboo.agents.helpers.orchestration.ToolProxy`
        boundary. External PanDA *reads* are allowed (fetching data is the job);
        only ``read_only=False`` tools are blocked. (State changes only ever
        happen in investigate's interactive turn loop. See docs/EXECUTION_TRUST.md.)
        """
        read_only_tool_names = frozenset(
            t.name for t in self._client.list_tools() if t.read_only
        )
        return await _shared_run_orchestration_code(
            code,
            client=self._client,
            task_data=task_data,
            task_data_tool_names=task_data_tool_names,
            internal_tools=None,  # ContextEnricher has no internal-tool registry
            allowed_tools=read_only_tool_names,
            timeout=600.0,
            log_prefix="orchestration",
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_tools_description(tools: list[McpTool]) -> str:
    """Format tool descriptors for the gap-analysis prompt.

    Delegates to the shared :func:`~bamboo.agents.helpers.tool_selection.render_tools`
    with ``full_schema_for=set()`` (every tool rendered compact, schema-free — the
    historical explorer behaviour). The code-generation prompt instead uses
    :meth:`_render_tools_description`, which budget-gates the list for large
    catalogues.
    """
    text, _ = render_tools(tools, full_schema_for=set(), style="explorer")
    return text


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
