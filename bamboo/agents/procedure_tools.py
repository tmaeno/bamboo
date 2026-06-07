"""Approved investigation Procedures exposed to the orchestration planner as tools.

Phase 2a of the execution-trust model (see docs/EXECUTION_TRUST.md). A captured
Procedure is identified by a **stable tool-call signature** (the sorted set of
``tools.<name>`` it calls — not the free-text ``strategy_type`` that made
near-duplicates accumulate). Non-trivial (≥2-tool), read-only procedures are
registered as callable tools so the planner can *reuse* prior work by emitting
``tools.<procedure>()`` instead of re-deriving the logic; each runs its stored,
already-reviewed code through the same sandbox (:func:`run_orchestration_code`),
so the read-only boundary composes.

This iteration adds **no durable auto-run**: the outer generated code that calls a
procedure-tool is still reviewed per turn (Phase 1C). Single-tool blocks are *not*
exposed (the raw tool already covers them); they are still captured + replayable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Awaitable, Callable

from bamboo.agents.orchestration import (
    analyze_code_side_effects,
    referenced_tool_names,
    run_orchestration_code,
)
from bamboo.mcp.base import McpTool

logger = logging.getLogger(__name__)

ToolCallable = Callable[..., Awaitable[Any]]


def procedure_signature(code: str) -> list[str]:
    """Stable identity for a procedure: the sorted set of tools its code calls.

    Replaces the free-text ``strategy_type`` as the dedup/identity key — same tools
    ⇒ same procedure, regardless of how the LLM phrased the strategy.
    """
    return sorted(referenced_tool_names(code))


def procedure_tool_name(signature: list[str]) -> str:
    """Identifier-safe, signature-derived name for a procedure (and its reuse-tool).

    Empty string for a no-tool block (caller falls back to legacy naming). The name
    embeds the composed tools (a good signal for the planner) and is bounded in
    length with a hash suffix when long. Always a valid Python identifier so the
    planner can write ``tools.<name>(...)``.
    """
    if not signature:
        return ""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", "proc__" + "__".join(signature))
    if len(safe) > 60:
        h = hashlib.sha256("::".join(signature).encode("utf-8")).hexdigest()[:8]
        safe = safe[:51] + "_" + h
    return safe


def _make_procedure_callable(
    code: str,
    *,
    client: Any,
    task_data: dict[str, Any],
    task_id: Any,
    task_data_tool_names: frozenset[str],
) -> ToolCallable:
    """A ToolProxy callable that replays a procedure's stored code in the sandbox.

    Runs through :func:`run_orchestration_code` (same proxy/sandbox), so an inner
    state-changing call would be refused by the runtime allow-set just like any
    other code — the read-only boundary composes. ``task_id``/``task_data`` are
    bound (the stored code expects them in scope); any LLM-supplied kwargs are
    ignored (a procedure is parameterised on the current task, not on args).
    """

    async def _run(**_kwargs: Any) -> Any:
        result, _call_log = await run_orchestration_code(
            code,
            client=client,
            task_data=task_data,
            task_data_tool_names=task_data_tool_names,
            extra_globals={"task_id": task_id, "task_data": task_data},
            log_prefix="procedure",
        )
        return result

    return _run


def build_procedure_tools_registry(
    procedures: list[dict[str, Any]],
    *,
    client: Any,
    task_data: dict[str, Any],
    task_id: Any,
    task_data_tool_names: frozenset[str],
    non_read_only_tool_names: frozenset[str],
) -> tuple[dict[str, McpTool], dict[str, ToolCallable]]:
    """Build ``(descriptors, callables)`` for the reusable-procedure toolkit.

    Exposes only procedures that are **code-bearing**, **non-trivial (≥2 distinct
    tools)**, and **read-only** (their code calls no ``read_only=False`` tool).
    Single-tool blocks are skipped (redundant with the raw tool). The procedures
    list is expected pre-ordered by reuse frequency (cause-agnostic), so the cap is
    applied by the caller via ``find_all_procedures(limit=...)``.

    Args:
        procedures:               Rows from ``graph_db.find_all_procedures`` (carry
                                  ``orchestration_code``, ``signature``,
                                  ``tool_name``, ``external_access``, ``cause_names``).
        client:                   MCP client for the sandboxed replay.
        task_data / task_id:      Current task, bound into each procedure callable.
        task_data_tool_names:     Tools that accept auto-injected ``task_data``.
        non_read_only_tool_names: State-changing tool names — a procedure whose code
                                  references any is excluded (state changes need a
                                  human in the loop; see docs/EXECUTION_TRUST.md).

    Returns:
        Two parallel dicts keyed by the procedure's identifier-safe tool name.
    """
    descriptors: dict[str, McpTool] = {}
    callables: dict[str, ToolCallable] = {}
    for p in procedures:
        code = (p.get("orchestration_code") or "").strip()
        if not code:
            continue
        signature = list(p.get("signature") or []) or procedure_signature(code)
        if len(signature) < 2:
            continue  # single-tool block — the raw tool already covers it
        if analyze_code_side_effects(code, non_read_only_tool_names):
            continue  # references a state-changing tool — not auto-exposed
        name = p.get("tool_name") or procedure_tool_name(signature)
        if not name or name in descriptors:
            continue
        causes = ", ".join(c for c in (p.get("cause_names") or []) if c) or "(various)"
        strategy = (p.get("strategy_type") or "").strip()
        summary = (p.get("code_summary") or "").strip()
        description = (
            f"Saved investigation procedure — reuses prior work by composing "
            f"{', '.join(signature)}. {strategy} {summary}".strip()
            + f" (used for cause(s): {causes})."
        )
        descriptors[name] = McpTool(
            name=name,
            description=description,
            parameters_schema={"type": "object", "properties": {}},
            read_only=True,
            external_access=bool(p.get("external_access")),
        )
        callables[name] = _make_procedure_callable(
            code,
            client=client,
            task_data=task_data,
            task_id=task_id,
            task_data_tool_names=task_data_tool_names,
        )
    return descriptors, callables
