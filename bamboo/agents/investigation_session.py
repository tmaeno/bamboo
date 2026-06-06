"""`bamboo investigate` — co-investigation session orchestrator.

Lives between the human and the bamboo knowledge graph for one live-incident
investigation session. See plan §C-§F for the design rationale:

* Auto-fetches task_data at session start, builds the initial graph skeleton
  via the §0.1 shared bootstrap helper, and calls
  :meth:`bamboo.agents.reasoning_navigator.ReasoningNavigator.analyze_task` to
  surface past similar causes (§B).
* Per-turn dialog loop with a **binary intent classifier** (tool vs narration)
  driven by a small LLM call. ``/tool`` prefix forces tool intent; ``/undo``
  rolls back the last turn's mutation; ``/done`` and ``/abandon`` terminate
  (§D).
* The **tool branch** generates a sandboxed orchestration code block via the
  ``INVESTIGATE_ORCHESTRATION`` prompt. **Every new code block is reviewed** by the
  human before it runs (``_review_code``), who sets a per-code session policy
  (run-once / auto-run / always-ask); approved code executes via the shared
  :func:`bamboo.agents.orchestration.run_orchestration_code`. Each block becomes
  ONE atomic_action with stored code + summary + signals on the Procedure (per the
  §G replayability schema). See docs/EXECUTION_TRUST.md for the trust model.
* The **narration branch** runs the parameterised email-extraction prompt
  (per §0.3) on the utterance + recent turn history; merges the extracted
  entities into ``partial_graph`` with canonical-name dedup.
* End-of-session form (§E) shows the procedure list, asks the cause +
  resolution, wires edges according to the node-vs-edge split in §G, and
  commits via the existing :class:`KnowledgeAccumulator` storage methods
  (§F).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.console import Console

from bamboo.agents.orchestration import analyze_code_side_effects, run_orchestration_code
from bamboo.agents.task_data_bootstrap import bootstrap_initial_graph
from bamboo.utils.narrator import say
from bamboo.frontends.base import Column, InteractionIO
from bamboo.frontends.cli import CliInteractionIO
from bamboo.llm import (
    EMAIL_EXTRACTION_SYSTEM,
    INVESTIGATE_INTENT_SYSTEM,
    INVESTIGATE_INTENT_USER,
    INVESTIGATE_KICKOFF_SYSTEM,
    INVESTIGATE_KICKOFF_USER,
    INVESTIGATE_NARRATION_USER,
    INVESTIGATE_ORCHESTRATION_SYSTEM,
    INVESTIGATE_ORCHESTRATION_USER,
    get_extraction_llm,
)
from bamboo.models.graph_element import (
    CauseNode,
    GraphRelationship,
    NodeType,
    ProcedureNode,
    RelationType,
    ResolutionNode,
    SymptomNode,
    TaskContextNode,
    TaskFeatureNode,
)
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class ToolGap(BaseModel):
    human_request_text: str
    llm_reason: str = ""
    turn: int = 0


class IntentGap(BaseModel):
    utterance: str
    classifier_guess: str = ""
    classifier_confidence: float = 0.0
    human_correction: str = ""
    kind: str  # "disambiguation" | "undo" | "reject_code"
    turn: int = 0


class OrchestrationRun(BaseModel):
    """One executed (or attempted) orchestration block — the unit of capture."""

    strategy_type: str
    code: str
    code_summary: str = ""
    trigger_signals: list[str] = Field(default_factory=list)
    external_access: bool = True  # code calls an external (PanDA) tool; formerly has_side_effects
    call_log: list[str] = Field(default_factory=list)
    result_summary: str = ""
    atomic_action_id: str = ""
    error: Optional[str] = None
    executed_at: str = ""


class Turn(BaseModel):
    role: str  # "human" | "system"
    kind: str  # "narration" | "tool" | "meta" | "disambiguation" | ...
    text: str = ""
    orchestration: Optional[OrchestrationRun] = None


class InvestigationSession(BaseModel):
    """Pydantic-serialisable session state. Persisted to ``--save`` each turn."""

    session_id: str
    turn: int = 0
    max_turns: int = 30
    initial_inputs: dict[str, Any] = Field(default_factory=dict)
    doc_hints: dict[str, str] = Field(default_factory=dict)
    turn_history: list[Turn] = Field(default_factory=list)
    partial_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    last_turn_snapshot: Optional[KnowledgeGraph] = None
    similar_past: list[dict[str, Any]] = Field(default_factory=list)
    tool_gap_log: list[ToolGap] = Field(default_factory=list)
    intent_gap_log: list[IntentGap] = Field(default_factory=list)
    status: str = "ongoing"  # "ongoing" | "resolved" | "abandoned"
    # Session-scoped review decisions: code_hash → "auto_run" | "always_ask".
    # A human reviews each new code block once and chooses its policy; this is
    # honored for the rest of the session (persists with the session JSON, so it
    # survives --resume). NOT shared across sessions/users — see docs/EXECUTION_TRUST.md.
    code_policies: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_response(text: str) -> dict[str, Any]:
    """Tolerantly parse a JSON object from an LLM response, stripping fences."""
    s = (text or "").strip()
    match = _JSON_FENCE_RE.search(s)
    if match:
        s = match.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("investigate: failed to parse JSON LLM response: %s", exc)
        return {}


def _format_doc_hints(doc_hints: dict[str, str]) -> str:
    return "\n\n".join(v for v in (doc_hints or {}).values() if v) or "(none)"


def _format_available_tools(tools: list[dict[str, Any]]) -> str:
    """Render a unified-tool descriptor list for the orchestration prompt."""
    if not tools:
        return "(none)"
    lines = []
    for t in tools:
        access = "external" if t.get("external_access") else "internal"
        # read_only defaults True (a tool only modifies state when explicitly tagged).
        rw = "read-only" if t.get("read_only", True) else "MODIFIES-STATE"
        schema = json.dumps(t.get("parameters_schema") or {}, indent=None)
        lines.append(
            f"- {t['name']} [{access}, {rw}]: {t.get('description', '').strip()}\n"
            f"    args schema: {schema}"
        )
    return "\n".join(lines)


def _format_initial_signals(task_data: dict[str, Any]) -> str:
    if not task_data:
        return "(no task_data)"
    keys = ("status", "taskType", "processingType", "site", "gshare", "ramCount", "coreCount")
    parts = []
    for k in keys:
        v = task_data.get(k)
        if v:
            parts.append(f"  {k}: {v}")
    if not parts:
        return "(no recognised signals)"
    return "\n".join(parts)


def _format_running_graph_summary(graph: KnowledgeGraph) -> str:
    if not graph.nodes:
        return "(empty)"
    by_type: dict[str, list[str]] = {}
    for n in graph.nodes:
        by_type.setdefault(str(getattr(n, "node_type", "Node")), []).append(n.name)
    parts = []
    for k, names in by_type.items():
        preview = ", ".join(names[:5]) + ("..." if len(names) > 5 else "")
        parts.append(f"  {k} ({len(names)}): {preview}")
    return "\n".join(parts)


def _format_turn_history_tail(turns: list[Turn], tail: int = 5) -> str:
    if not turns:
        return "(none)"
    parts = []
    for t in turns[-tail:]:
        text = (t.text or "")[:200]
        parts.append(f"  ({t.role}/{t.kind}) {text}")
    return "\n".join(parts)


def _format_prior_results(turns: list[Turn], limit: int = 5) -> str:
    """Render the recent OrchestrationRun results so the planner can chain on them."""
    runs = [t.orchestration for t in turns if t.orchestration is not None]
    if not runs:
        return "(no prior orchestration results in this session)"
    parts = []
    for r in runs[-limit:]:
        called = ", ".join(r.call_log) or "(no calls logged)"
        parts.append(
            f"  - strategy='{r.strategy_type}' summary='{r.code_summary}' "
            f"called=[{called}] result_summary='{r.result_summary}'"
        )
    return "\n".join(parts)


def _summarise_result(result: Any, max_len: int = 200) -> str:
    """Cheap deterministic one-line gist of a tool result for storage."""
    try:
        s = repr(result)
    except Exception:  # noqa: BLE001
        s = "<unrepr-able result>"
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _code_hash(code: str) -> str:
    """Stable identity for a code block under whitespace-only normalization.

    Strips per-line trailing whitespace and drops blank lines (leading
    indentation is preserved — it is significant in Python), then sha256. Two
    blocks that differ only in formatting share a hash; a logic change yields a
    new hash, so it is re-reviewed (the safe direction). See docs/EXECUTION_TRUST.md.
    """
    normalized = "\n".join(line.rstrip() for line in (code or "").splitlines() if line.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class _Deps:
    """Bundle of collaborator objects the orchestrator needs.

    Grouped into one dataclass so callers (CLI / tests) pass a single
    structure and don't have to thread many arguments.
    """

    mcp_client: Any  # PandaMcpClient
    graph_db: Any  # GraphDatabaseClient
    vector_db: Optional[Any] = None
    extractor: Any = None  # PandaKnowledgeExtractor (for prefetch_hints)
    reasoning_navigator: Optional[Any] = None
    knowledge_accumulator: Optional[Any] = None  # for commit
    error_classifier: Any = None
    console: Optional[Console] = None
    io: Optional[InteractionIO] = None  # frontend; defaults to the Rich terminal


class InvestigationOrchestrator:
    """Drives one live-incident investigation session end-to-end."""

    def __init__(
        self,
        deps: _Deps,
        *,
        session_id: Optional[str] = None,
        max_turns: int = 30,
        save_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> None:
        self.deps = deps
        self.console = deps.console or Console()
        # Frontend interaction surface. Defaults to the Rich terminal so the CLI
        # and tests that don't supply one keep working unchanged.
        self.io: InteractionIO = deps.io or CliInteractionIO(self.console)
        self.save_path = save_path
        self.dry_run = dry_run
        sid = session_id or uuid.uuid4().hex[:12]
        self.session = InvestigationSession(session_id=sid, max_turns=max_turns)
        self._llm = None  # lazy
        self._internal_tools_descriptors: dict = {}
        self._internal_tools_callables: dict = {}
        self._mcp_tool_descriptors: list = []
        self._task_data_tool_names: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        task_id: Optional[int] = None,
        task_data: Optional[dict[str, Any]] = None,
        symptom: Optional[str] = None,
    ) -> None:
        """Plan §B — auto-prefetch + proactive hypothesis."""
        self.io.panel(
            f"[bold blue]bamboo investigate[/bold blue] — session {self.session.session_id}",
            style="blue",
            fit=True,
        )

        # 1) Fetch task_data if a task_id was supplied — via the shared seam, the
        #    same one analyze/capture/populate use (input acquisition lives at the
        #    entry point; the agent's contract is just "give me a task_data dict").
        if task_id is not None and task_data is None:
            from bamboo.agents.deps import resolve_task_data  # noqa: PLC0415

            try:
                task_data = await resolve_task_data(task_id)
            except Exception as exc:  # noqa: BLE001
                self.io.notice(f"[yellow]Failed to fetch task_data: {exc}[/yellow]")
                task_data = None

        self.session.initial_inputs = {
            "task_id": task_id,
            "task_data": task_data,
            "symptom": symptom,
        }

        # 2) Prefetch domain hints.
        if self.deps.extractor is not None:
            try:
                error_dialog = (task_data or {}).get("errorDialog") or symptom or ""
                self.session.doc_hints = await self.deps.extractor.prefetch_hints(
                    task_data or {}, email_text=error_dialog
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("prefetch_hints failed: %s", exc)
                self.session.doc_hints = {}

        # 3) Build initial graph skeleton.
        if task_data:
            try:
                nodes, rels = await bootstrap_initial_graph(
                    task_data=task_data,
                    external_data=None,
                    error_classifier=self.deps.error_classifier,
                    extract_embedded_logs=False,
                )
                self._merge_into_partial_graph(nodes, rels)
            except Exception as exc:  # noqa: BLE001
                logger.warning("bootstrap_initial_graph failed: %s", exc)
                self.io.notice(f"[yellow]bootstrap failed: {exc}[/yellow]")

        # 4) Display signals.
        self._display_kickoff_panel(task_data, symptom)

        # 5) Proactive hypothesis from past graph (analyze_task).
        await self._show_past_similar(task_data, symptom)

        # 6) Build the unified tool registry (PanDA MCP + internal).
        await self._build_tool_registry()

        # 7) Persist initial state.
        self._persist()

    async def run(self) -> None:
        """Plan §D — per-turn dialog loop until /done /abandon or max_turns."""
        self.io.notice(
            "[dim]Type your request or finding. Slash commands: /done /abandon /undo /tool /show-graph /show-tools[/dim]"
        )
        while self.session.turn < self.session.max_turns and self.session.status == "ongoing":
            try:
                utterance = await self.io.ask("[bold cyan]>[/bold cyan]")
            except SystemExit:
                self.session.status = "abandoned"
                break

            # Snapshot for /undo (deep copy — shallow would silently share node lists).
            self.session.last_turn_snapshot = self.session.partial_graph.model_copy(deep=True)

            # Meta-commands first (no LLM).
            meta_handled = await self._handle_meta_command(utterance)
            if meta_handled is True:
                self._persist()
                continue
            if meta_handled == "terminate":
                break

            # /tool prefix forces tool intent.
            forced_intent: Optional[str] = None
            text = utterance
            if utterance.startswith("/tool "):
                forced_intent = "tool"
                text = utterance[len("/tool ") :].strip()

            self.session.turn_history.append(Turn(role="human", kind="raw", text=text))

            if forced_intent == "tool":
                intent = "tool"
            else:
                intent = await self._classify_intent(text)

            if intent == "tool":
                await self._tool_turn(text)
            else:
                await self._narration_turn(text)

            self.session.turn += 1
            self._persist()

        if self.session.status == "ongoing" and self.session.turn >= self.session.max_turns:
            self.io.notice(f"[yellow]Reached max_turns={self.session.max_turns}; ending.[/yellow]")
            self.session.status = "abandoned"

    async def finalize(self) -> None:
        """Plan §E — end-of-session form: list → cause → resolution → wire edges."""
        if self.session.status == "abandoned" and not self._has_any_atomic_actions():
            self.io.notice("[dim]Nothing to commit (abandoned with no captured actions).[/dim]")
            return

        self._display_procedure_list()

        if self.session.status == "abandoned":
            self.io.notice("[dim]Abandoned session — committing Symptom + Procedures as tentative.[/dim]")
            cause_text = None
            resolution_text = None
        else:
            cause_text = await self.io.ask("[bold]What was the cause?[/bold]")
            resolution_text = await self.io.ask(
                "[bold]What was the resolution? (optional, blank to skip)[/bold]",
                default="",
            )
            resolution_text = resolution_text or None

        self._wire_finalization_edges(cause_text, resolution_text)
        await self._show_diff_and_commit()

    # ------------------------------------------------------------------
    # Turn-level handlers
    # ------------------------------------------------------------------

    async def _handle_meta_command(self, utterance: str) -> bool | str:
        """Return True if handled (continue loop), 'terminate' to break, False otherwise."""
        cmd = utterance.strip()
        if cmd == "/done":
            self.session.status = "resolved"
            return "terminate"
        if cmd == "/abandon":
            self.session.status = "abandoned"
            return "terminate"
        if cmd == "/undo":
            if self.session.last_turn_snapshot is not None:
                self.session.partial_graph = self.session.last_turn_snapshot
                self.session.intent_gap_log.append(
                    IntentGap(utterance="(prior turn)", kind="undo", turn=self.session.turn)
                )
                self.io.notice("[yellow]Reverted last turn.[/yellow]")
            else:
                self.io.notice("[dim]Nothing to undo.[/dim]")
            return True
        if cmd == "/show-graph":
            self.io.panel(_format_running_graph_summary(self.session.partial_graph), title="partial_graph")
            return True
        if cmd == "/show-tools":
            tool_list = self._unified_tool_descriptors()
            self.io.notice(_format_available_tools(tool_list))
            return True
        if cmd == "/paste":
            self.io.notice("[dim]Use the request_human_input tool by phrasing your request to bamboo.[/dim]")
            return True
        if cmd == "/skip":
            # Drops the prior turn's snapshot and treats this turn as a no-op —
            # used after a failed/rejected tool call when the human wants to
            # move on without recording anything for this turn.
            self.io.notice("[dim]Skipped this turn (no action recorded).[/dim]")
            return True
        if cmd == "/approvals":
            policies = self.session.code_policies
            if not policies:
                self.io.notice("[dim]No code-execution policies set this session.[/dim]")
            else:
                lines = "\n".join(f"  {h[:12]} → {p}" for h, p in policies.items())
                self.io.notice(f"[bold]Code policies (this session):[/bold]\n{lines}")
            return True
        if cmd.startswith("/revoke"):
            arg = cmd[len("/revoke"):].strip()
            if not arg:
                self.io.notice("[dim]Usage: /revoke <hash-prefix|all>[/dim]")
            elif arg == "all":
                n = len(self.session.code_policies)
                self.session.code_policies.clear()
                self.io.notice(f"[dim]Cleared {n} code polic(y/ies).[/dim]")
            else:
                matched = [h for h in self.session.code_policies if h.startswith(arg)]
                for h in matched:
                    del self.session.code_policies[h]
                self.io.notice(
                    f"[dim]Revoked {len(matched)} polic(y/ies).[/dim]"
                    if matched
                    else f"[dim]No policy matches '{arg}'.[/dim]"
                )
            return True
        return False

    async def _classify_intent(self, utterance: str) -> str:
        """Plan §D step 4 — binary classifier with disambiguation gate."""
        prompt_user = INVESTIGATE_INTENT_USER.format(
            domain_hints=_format_doc_hints(self.session.doc_hints),
            running_graph_summary=_format_running_graph_summary(self.session.partial_graph),
            utterance=utterance,
        )
        try:
            response = await self._invoke_llm(INVESTIGATE_INTENT_SYSTEM, prompt_user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("intent classifier failed: %s", exc)
            return "narration"  # safe default — won't trigger external tools
        parsed = _parse_json_response(response)
        intent = str(parsed.get("intent", "narration")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
        is_close = bool(parsed.get("is_close_call", False))
        if intent not in ("tool", "narration"):
            intent = "narration"

        say(
            f"intent → {intent} (confidence {confidence:.2f}"
            f"{'; close call' if is_close else ''})",
            level=logging.DEBUG,
        )

        if confidence < 0.7 or is_close:
            chosen = await self.io.ask(
                "[dim]I'm not sure — did you want me to[/dim] [bold](t)[/bold]ool [dim]or are you[/dim] [bold](s)[/bold]haring a finding?",
                choices=["t", "s"],
            )
            self.session.intent_gap_log.append(
                IntentGap(
                    utterance=utterance,
                    classifier_guess=intent,
                    classifier_confidence=confidence,
                    human_correction=chosen,
                    kind="disambiguation",
                    turn=self.session.turn,
                )
            )
            intent = "tool" if chosen == "t" else "narration"
        return intent

    async def _tool_turn(self, utterance: str) -> None:
        """Plan §D tool branch — generate code, review (per-code policy), execute, record."""
        task_id = (self.session.initial_inputs.get("task_id"))
        task_data = self.session.initial_inputs.get("task_data") or {}
        error_dialog = (task_data.get("errorDialog") if task_data else None) or self.session.initial_inputs.get("symptom") or ""

        tool_descriptors_full = self._unified_tool_descriptors()
        user_msg = INVESTIGATE_ORCHESTRATION_USER.format(
            domain_hints=_format_doc_hints(self.session.doc_hints),
            task_id=task_id or "(none)",
            error_dialog=error_dialog or "(none)",
            initial_signals=_format_initial_signals(task_data),
            prior_turn_results=_format_prior_results(self.session.turn_history),
            available_tools=_format_available_tools(tool_descriptors_full),
            utterance=utterance,
        )
        try:
            response = await self._invoke_llm(INVESTIGATE_ORCHESTRATION_SYSTEM, user_msg)
        except Exception as exc:  # noqa: BLE001
            self.io.notice(f"[red]Orchestration LLM call failed: {exc}[/red]")
            return
        plan = _parse_json_response(response)

        code = plan.get("code")
        if not code or not str(code).strip():
            reason = plan.get("reason", "(no reason returned)")
            self.io.notice(f"[yellow]No tool fits this request: {reason}[/yellow]")
            self.io.notice("[dim]Tip: rephrase, or call request_human_input via a follow-up turn to paste the info.[/dim]")
            self.session.tool_gap_log.append(
                ToolGap(human_request_text=utterance, llm_reason=str(reason), turn=self.session.turn)
            )
            self.session.turn_history.append(Turn(role="system", kind="tool_gap", text=str(reason)))
            return

        strategy_type = str(plan.get("strategy_type") or "unnamed_strategy").strip()
        code_summary = str(plan.get("code_summary") or "").strip()
        trigger_signals = list(plan.get("trigger_signals") or [])

        say(f"strategy → {strategy_type}: {code_summary}", level=logging.DEBUG)

        # Every new code block is reviewed (not just state-changing ones); the human
        # sets a per-code session policy. external_tool_names → recorded as Procedure
        # metadata (does this code hit PanDA), not the gate. See docs/EXECUTION_TRUST.md.
        external_tool_names = {t["name"] for t in tool_descriptors_full if t.get("external_access")}

        reviewed = await self._review_code(strategy_type, code_summary, trigger_signals, code)
        if reviewed is None:
            self.session.intent_gap_log.append(
                IntentGap(
                    utterance=utterance,
                    classifier_guess="tool",
                    kind="reject_code",
                    turn=self.session.turn,
                )
            )
            self.session.turn_history.append(Turn(role="system", kind="reject_code", text=strategy_type))
            self.io.notice("[dim]Aborted. Re-prompt to try a different request.[/dim]")
            return
        strategy_type, code, code_summary, trigger_signals = reviewed

        # Recorded as Procedure metadata; computed on the final (possibly edited) code.
        external_access = analyze_code_side_effects(code, external_tool_names)

        # Execute.
        result, call_log, error = await self._execute_orchestration(code, task_id, task_data)

        # Build an OrchestrationRun + ProcedureNode (tentative — no Cause yet).
        atomic_action_id = uuid.uuid4().hex[:12]
        run = OrchestrationRun(
            strategy_type=strategy_type,
            code=code,
            code_summary=code_summary,
            trigger_signals=trigger_signals,
            external_access=external_access,
            call_log=call_log,
            result_summary=_summarise_result(result),
            atomic_action_id=atomic_action_id,
            error=error,
            executed_at=_now_iso(),
        )
        self._record_atomic_action(run, raw_result=result)
        self.session.turn_history.append(Turn(role="system", kind="tool", text=strategy_type, orchestration=run))
        if error:
            self.io.notice(f"[red]Tool execution error: {error}[/red]")
        else:
            self.io.result(_summarise_result(result, max_len=1000), title=f"result · {strategy_type}")

    async def _review_code(
        self,
        strategy_type: str,
        code_summary: str,
        trigger_signals: list[str],
        code: str,
    ) -> Optional[tuple[str, str, str, list[str]]]:
        """Review-and-policy gate (Phase 1, session-scoped).

        Every code block not yet approved this session is shown to the operator,
        who picks a per-code policy: ``run-once`` (no persistence), ``auto-run``
        (skip future prompts for identical code this session), or ``always-ask``
        (re-prompt each time). ``auto_run`` code runs with no prompt. The decision
        is keyed on :func:`_code_hash` and stored on ``session.code_policies``.
        Returns the (possibly edited) ``(strategy_type, code, code_summary,
        trigger_signals)`` to execute, or ``None`` if declined.
        See docs/EXECUTION_TRUST.md.
        """
        while True:
            h = _code_hash(code)
            policy = self.session.code_policies.get(h)
            if policy == "auto_run":
                say(f"↻ auto-running approved code: {strategy_type}", level=logging.DEBUG)
                return strategy_type, code, code_summary, trigger_signals

            self._display_confirmation_panel(strategy_type, code_summary, trigger_signals, code)
            if policy == "always_ask":
                choice = await self.io.ask(
                    "[bold]Proceed?[/bold]", default="N", choices=["y", "N", "edit"]
                )
            else:
                choice = await self.io.ask(
                    "[bold]Review[/bold] — [y] run once / [a] auto-run / [k] always ask / "
                    "[edit] / [N] reject",
                    default="N",
                    choices=["y", "a", "k", "edit", "N"],
                )

            if choice == "N":
                return None
            if choice == "edit":
                strategy_type, code, code_summary, trigger_signals = await self.io.edit(
                    strategy_type=strategy_type,
                    code=code,
                    summary=code_summary,
                    triggers=trigger_signals,
                )
                continue  # re-review the edited code (recompute hash + re-prompt)
            if choice == "a":
                self.session.code_policies[h] = "auto_run"
                self.io.notice("[dim]✓ will auto-run this exact code for the rest of the session.[/dim]")
            elif choice == "k":
                self.session.code_policies[h] = "always_ask"
            # choice == "y" → run once, persist nothing.
            return strategy_type, code, code_summary, trigger_signals

    async def _narration_turn(self, utterance: str) -> None:
        """Plan §D narration branch — extract entities, merge into partial_graph."""
        user_msg = INVESTIGATE_NARRATION_USER.format(
            domain_hints=_format_doc_hints(self.session.doc_hints),
            running_graph_summary=_format_running_graph_summary(self.session.partial_graph),
            turn_history_tail=_format_turn_history_tail(self.session.turn_history),
            utterance=utterance,
        )
        try:
            response = await self._invoke_llm(EMAIL_EXTRACTION_SYSTEM, user_msg)
        except Exception as exc:  # noqa: BLE001
            self.io.notice(f"[red]Narration extraction failed: {exc}[/red]")
            return
        parsed = _parse_json_response(response)
        added = self._merge_narration_extraction(parsed)
        self.session.turn_history.append(Turn(role="system", kind="narration", text=f"extracted {added} entities"))
        if added == 0:
            self.io.notice("[dim]Noted (no new graph entities extracted).[/dim]")
        else:
            self.io.notice(f"[dim]Extracted {added} graph entit(y/ies) from your statement.[/dim]")

    # ------------------------------------------------------------------
    # Tool registry + execution
    # ------------------------------------------------------------------

    async def _build_tool_registry(self) -> None:
        """Build the unified registry (PanDA MCP + internal-tools)."""
        await self.deps.mcp_client.connect()
        mcp_tools = self.deps.mcp_client.list_tools()
        self._mcp_tool_descriptors = mcp_tools
        self._task_data_tool_names = self.deps.mcp_client.task_data_tools()

        from bamboo.agents.internal_tools import build_internal_tools_registry  # noqa: PLC0415

        descs, calls = build_internal_tools_registry(
            graph_db=self.deps.graph_db,
            reasoning_navigator=self.deps.reasoning_navigator,
        )
        self._internal_tools_descriptors = descs
        self._internal_tools_callables = calls

    def _unified_tool_descriptors(self) -> list[dict[str, Any]]:
        """Render unified descriptors for the orchestration prompt."""
        out: list[dict[str, Any]] = []
        for t in self._mcp_tool_descriptors:
            out.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters_schema": t.parameters_schema,
                    "external_access": getattr(t, "external_access", True),
                    "read_only": getattr(t, "read_only", True),
                }
            )
        for name, t in self._internal_tools_descriptors.items():
            out.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters_schema": t.parameters_schema,
                    "external_access": getattr(t, "external_access", False),
                    "read_only": getattr(t, "read_only", True),
                }
            )
        return out

    async def _execute_orchestration(
        self,
        code: str,
        task_id: Optional[int],
        task_data: dict[str, Any],
    ) -> tuple[Any, list[str], Optional[str]]:
        """Run code via the shared run_orchestration_code with task_id/task_data in scope.

        The ORCHESTRATION prompt promises the LLM that ``task_id`` and
        ``task_data`` are in scope. Pass them through ``extra_globals`` so they
        become real local names inside the sandboxed function — safer than
        embedding ``repr(task_data)`` into the source (which would fail or
        silently corrupt on values whose repr isn't a valid Python literal).
        """
        try:
            result, call_log = await run_orchestration_code(
                code,
                client=self.deps.mcp_client,
                task_data=task_data,
                task_data_tool_names=self._task_data_tool_names,
                internal_tools=self._internal_tools_callables,
                extra_globals={"task_id": task_id, "task_data": task_data},
                timeout=600.0,
                log_prefix=f"investigate:turn{self.session.turn}",
            )
            return result, call_log, None
        except Exception as exc:  # noqa: BLE001
            return {}, [], f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_kickoff_panel(self, task_data: Optional[dict[str, Any]], symptom: Optional[str]) -> None:
        lines: list[str] = []
        if task_data:
            status = task_data.get("status")
            err = task_data.get("errorDialog")
            if status:
                lines.append(f"[bold]status:[/bold] {status}")
            if err:
                short = re.sub(r"\s+", " ", str(err))[:300]
                lines.append(f"[bold]errorDialog:[/bold] {short}{'…' if len(str(err)) > 300 else ''}")
            top = _format_initial_signals(task_data)
            if top and top != "(no task_data)" and top != "(no recognised signals)":
                lines.append(f"[bold]signals:[/bold]\n{top}")
        if symptom:
            lines.append(f"[bold]symptom:[/bold] {symptom}")
        body = "\n".join(lines) if lines else "(no task_data or symptom)"
        self.io.panel(body, title="task under investigation", style="cyan")

    async def _show_past_similar(self, task_data: Optional[dict[str, Any]], symptom: Optional[str]) -> None:
        if self.deps.reasoning_navigator is None:
            return
        if not task_data:
            return
        try:
            # Reuse the hints already fetched in start() step 2 so analyze_task
            # doesn't prefetch a second time (its doc_hints= hook skips the
            # internal prefetch when given). None ⇒ let it self-prefetch.
            result = await self.deps.reasoning_navigator.analyze_task(
                task_data, doc_hints=self.session.doc_hints or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_task in session-start failed: %s", exc)
            return
        if result is None:
            return
        root_cause = getattr(result, "root_cause", None)
        confidence = getattr(result, "confidence", 0.0)
        if root_cause:
            self.io.panel(
                f"[bold]most-similar past root cause:[/bold] {root_cause}\n"
                f"[bold]confidence:[/bold] {confidence:.2f}\n"
                f"[dim](this is a hypothesis from past incidents — confirm or chase a different lead.)[/dim]",
                title="past similar incidents",
                style="magenta",
            )
        self.session.similar_past = [
            {"root_cause": root_cause, "confidence": confidence}
        ] if root_cause else []

    def _display_confirmation_panel(
        self,
        strategy_type: str,
        summary: str,
        triggers: list[str],
        code: str,
    ) -> None:
        triggers_str = "\n  - ".join(triggers) if triggers else "(none)"
        header = (
            f"[bold]strategy:[/bold]  {strategy_type}\n"
            f"[bold]summary:[/bold]  {summary or '(none)'}\n"
            f"[bold]trigger:[/bold]\n  - {triggers_str}"
        )
        self.io.panel(header, title="proposed orchestration", style="yellow")
        self.io.code(code, lang="python")

    def _display_procedure_list(self) -> None:
        runs = [t.orchestration for t in self.session.turn_history if t.orchestration is not None]
        if not runs:
            self.io.notice("[dim]No tool calls were recorded this session.[/dim]")
            return
        rows: list[list[str]] = []
        for i, r in enumerate(runs, 1):
            called = ", ".join(r.call_log) or "-"
            res = r.result_summary[:60] + ("…" if len(r.result_summary) > 60 else "")
            rows.append(
                [
                    str(i),
                    r.strategy_type,
                    r.code_summary or "-",
                    called,
                    res,
                    "y" if r.external_access else "n",
                ]
            )
        self.io.table(
            title=f"procedure list ({len(runs)} step(s))",
            columns=[
                Column("#", justify="right"),
                Column("strategy"),
                Column("summary"),
                Column("called"),
                Column("result"),
                Column("external?", justify="center"),
            ],
            rows=rows,
        )

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def _merge_into_partial_graph(self, nodes: list, rels: list) -> None:
        """Append nodes/rels with canonical-name dedup against existing partial_graph."""
        existing_keys = {(str(getattr(n, "node_type", "")), n.name) for n in self.session.partial_graph.nodes}
        for n in nodes:
            key = (str(getattr(n, "node_type", "")), n.name)
            if key not in existing_keys:
                self.session.partial_graph.nodes.append(n)
                existing_keys.add(key)
        for r in rels:
            self.session.partial_graph.relationships.append(r)

    def _record_atomic_action(self, run: OrchestrationRun, raw_result: Any) -> None:
        """Append a tentative ProcedureNode + a Task_Context node for the result."""
        proc_metadata: dict[str, Any] = {
            "orchestration_code": run.code,
            "code_summary": run.code_summary,
            "external_access": run.external_access,
            "atomic_action_id": run.atomic_action_id,
            # The per-edge fields (trigger_signals, result_summary, executed_at)
            # are kept here too during the session; finalize splits them onto the
            # eventual investigated_by edge per §G.
            "_pending_edge": {
                "trigger_signals": list(run.trigger_signals),
                "result_summary": run.result_summary,
                "executed_at": run.executed_at,
            },
        }
        if run.error:
            proc_metadata["error"] = run.error
        proc = ProcedureNode(
            name=f"{run.strategy_type}:tentative:{run.atomic_action_id}",
            description=run.code_summary or run.strategy_type,
            strategy_type=run.strategy_type,
            metadata=proc_metadata,
        )
        # Task_Context node carrying the raw result for downstream review.
        try:
            ctx_text = json.dumps(raw_result, default=str)[:4000]
        except (TypeError, ValueError):
            ctx_text = str(raw_result)[:4000]
        ctx = TaskContextNode(
            name=f"result:{run.atomic_action_id}",
            description=ctx_text,
            metadata={"source": "investigate:tool_result", "atomic_action_id": run.atomic_action_id},
        )
        self._merge_into_partial_graph([proc, ctx], [])

    def _merge_narration_extraction(self, parsed: dict[str, Any]) -> int:
        """Convert EMAIL_EXTRACTION JSON into nodes/edges; return count added."""
        if not parsed:
            return 0
        nodes_data = parsed.get("nodes") or []
        rels_data = parsed.get("relationships") or []
        nodes: list[Any] = []
        for nd in nodes_data:
            ntype = nd.get("node_type", "")
            base = {
                "name": nd.get("name", ""),
                "description": nd.get("description"),
                "metadata": nd.get("metadata") or {},
            }
            if not base["name"]:
                continue
            if ntype == "Cause":
                nodes.append(CauseNode(**base))
            elif ntype == "Resolution":
                nodes.append(ResolutionNode(**base, steps=nd.get("steps") or []))
            elif ntype == "Task_Context":
                nodes.append(TaskContextNode(**base))
        # Build relationships only when both endpoints are present in nodes (or partial_graph).
        existing_names = {n.name for n in self.session.partial_graph.nodes} | {n.name for n in nodes}
        rels: list[GraphRelationship] = []
        for rd in rels_data:
            src = rd.get("source_name")
            tgt = rd.get("target_name")
            if not src or not tgt or src not in existing_names or tgt not in existing_names:
                continue
            try:
                rtype = RelationType(rd.get("relation_type", ""))
            except ValueError:
                continue
            rels.append(
                GraphRelationship(
                    source_id=src,
                    target_id=tgt,
                    relation_type=rtype,
                    confidence=float(rd.get("confidence", 1.0)),
                )
            )
        self._merge_into_partial_graph(nodes, rels)
        return len(nodes) + len(rels)

    def _wire_finalization_edges(self, cause_text: Optional[str], resolution_text: Optional[str]) -> None:
        """Plan §E step 5 — promote tentative Procedures into Cause-attached edges."""
        # Promote / create Cause node when provided.
        cause_node: Optional[CauseNode] = None
        if cause_text:
            cause_node = CauseNode(
                name=cause_text.strip(),
                description=cause_text.strip(),
                metadata={"source": "investigate:final_form", "confirmed": True},
            )
            self._merge_into_partial_graph([cause_node], [])
        resolution_node: Optional[ResolutionNode] = None
        if resolution_text:
            resolution_node = ResolutionNode(
                name=resolution_text.strip(),
                description=resolution_text.strip(),
                metadata={"source": "investigate:final_form"},
            )
            self._merge_into_partial_graph([resolution_node], [])

        # Cause -[solved_by]-> Resolution.
        if cause_node is not None and resolution_node is not None:
            self.session.partial_graph.relationships.append(
                GraphRelationship(
                    source_id=cause_node.name,
                    target_id=resolution_node.name,
                    relation_type=RelationType.SOLVED_BY,
                    confidence=1.0,
                )
            )

        # Promote tentative Procedures: canonicalise name to strategy_type:<cause>
        # (per existing schema) and create investigated_by edge carrying per-edge fields.
        is_abandoned = cause_node is None
        for n in list(self.session.partial_graph.nodes):
            if not isinstance(n, ProcedureNode):
                continue
            if not n.name.startswith("") or "tentative" not in n.name:
                continue  # already finalised
            pending = (n.metadata or {}).pop("_pending_edge", {}) or {}
            if is_abandoned:
                n.metadata.setdefault("status", "tentative")
                n.metadata.setdefault("session_status", "ongoing")
                # Without a Cause we can't materialise the investigated_by edge.
                continue
            # Strip the per-session uniquifier and re-canonicalise.
            canonical_name = f"{n.strategy_type}:{cause_node.name}"
            n.name = canonical_name
            self.session.partial_graph.relationships.append(
                GraphRelationship(
                    source_id=cause_node.name,
                    target_id=canonical_name,
                    relation_type=RelationType.INVESTIGATED_BY,
                    confidence=1.0,
                    properties={
                        "metadata": {
                            "trigger_signals": pending.get("trigger_signals", []),
                            "result_summary": pending.get("result_summary", ""),
                            "executed_at": pending.get("executed_at", ""),
                        },
                    },
                )
            )

        # Wire Symptom -[indicate]-> Cause, Task_Features -[contribute_to]-> Cause.
        if cause_node is not None:
            for n in self.session.partial_graph.nodes:
                if isinstance(n, SymptomNode):
                    self.session.partial_graph.relationships.append(
                        GraphRelationship(
                            source_id=n.name,
                            target_id=cause_node.name,
                            relation_type=RelationType.INDICATE,
                            confidence=1.0,
                        )
                    )
                elif isinstance(n, TaskFeatureNode):
                    self.session.partial_graph.relationships.append(
                        GraphRelationship(
                            source_id=n.name,
                            target_id=cause_node.name,
                            relation_type=RelationType.CONTRIBUTE_TO,
                            confidence=0.5,
                        )
                    )

    async def _show_diff_and_commit(self) -> None:
        graph = self.session.partial_graph
        rows = await self._classify_nodes_for_diff(graph)

        # Render the per-node diff (the frontend decides table vs. Mermaid etc.).
        edges = [
            (
                r.source_id,
                r.target_id,
                r.relation_type.value if hasattr(r.relation_type, "value") else str(r.relation_type),
            )
            for r in graph.relationships
        ]
        self.io.diff(rows, edge_count=len(graph.relationships), edges=edges)

        if self.dry_run:
            self.io.notice("[yellow]--dry-run set; not committing.[/yellow]")
            return

        if not await self.io.confirm("commit this investigation?", default=True):
            self.io.notice("[yellow]Commit cancelled.[/yellow]")
            return

        await self._commit()

    async def _classify_nodes_for_diff(
        self, graph: KnowledgeGraph
    ) -> list[tuple[str, str, str]]:
        """Return per-node ``(node_type_label, name, "new"|"merge")`` for the diff display.

        Looks up each node by its canonical ``NodeType`` value (the same string
        used as the Neo4j label, e.g. ``"Symptom"``) — earlier drafts used
        ``str(NodeType.SYMPTOM)`` which produces ``"NodeType.SYMPTOM"`` (wrong
        format) and made every node look new. Errors per node are non-fatal
        (treated as "new"); a broken graph_db connection doesn't abort the
        whole commit flow.
        """
        rows: list[tuple[str, str, str]] = []
        for n in graph.nodes:
            nt = getattr(n, "node_type", None)
            label = nt.value if hasattr(nt, "value") else str(nt or "Node")
            try:
                existing = await self.deps.graph_db.get_node_description(label, n.name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("get_node_description(%s, %s) failed: %s", label, n.name, exc)
                existing = None
            rows.append((label, n.name, "merge" if existing is not None else "new"))
        return rows

    async def _commit(self) -> None:
        """Plan §F — delegate to KnowledgeAccumulator's existing storage methods."""
        if self.deps.knowledge_accumulator is None:
            self.io.notice("[red]No KnowledgeAccumulator wired; cannot commit.[/red]")
            return

        graph = self.session.partial_graph
        graph.metadata = dict(getattr(graph, "metadata", {}) or {})
        graph.metadata["graph_id"] = f"investigate:{self.session.session_id}"

        agent = self.deps.knowledge_accumulator
        try:
            transcript = self._render_transcript()
            await agent.store_extracted(
                graph, doc_hints=self.session.doc_hints, email_text=transcript
            )
            self.io.notice("[green]Committed.[/green]")
        except Exception as exc:  # noqa: BLE001
            logger.exception("investigate commit failed")
            self.io.notice(f"[red]Commit failed: {exc}[/red]")

        # Persist tool_gap_log to disk for developer mining.
        try:
            gap_path = Path.home() / ".bamboo" / "investigations" / "gaps.jsonl"
            gap_path.parent.mkdir(parents=True, exist_ok=True)
            with gap_path.open("a") as fh:
                for gap in self.session.tool_gap_log:
                    fh.write(json.dumps(gap.model_dump()) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not persist tool_gap_log: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_any_atomic_actions(self) -> bool:
        return any(t.orchestration is not None for t in self.session.turn_history)

    def _render_transcript(self) -> str:
        parts = []
        for i, t in enumerate(self.session.turn_history, 1):
            parts.append(f"Turn {i} ({t.role}/{t.kind}): {t.text}")
        return "\n".join(parts)

    def _persist(self) -> None:
        if not self.save_path:
            return
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(self.session.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not persist session to %s: %s", self.save_path, exc)

    async def _invoke_llm(self, system_message: str, user_message: str) -> str:
        """Single LLM round-trip helper; returns the raw content string."""
        if self._llm is None:
            self._llm = get_extraction_llm()
        response = await self._llm.ainvoke(
            [SystemMessage(content=system_message), HumanMessage(content=user_message)]
        )
        return getattr(response, "content", "") or ""
