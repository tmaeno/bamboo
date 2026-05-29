"""Sandboxed orchestration-code execution shared by ContextEnricher and investigate.

LLM-generated orchestration code is short async-function bodies that call
``tools.<name>(...)`` to compose one or more MCP tool invocations into a single
unit of work. This module owns the sandboxed ``exec`` mechanics, the
``ToolProxy`` that resolves ``tools.<name>`` to a backing callable, and a
static-analysis helper that determines whether a code block will call any tool
flagged as having side effects.

The orchestration-code primitive originated in :class:`ContextEnricher` for
populate's reviewer-driven exploration. ``bamboo investigate`` adopts the same
primitive — its tool branch generates one orchestration block per human
turn — so this module is the single place where the sandbox, proxy, and
side-effects classifier live. Both callers go through it.

Public API
----------
* :data:`SAFE_BUILTINS` — the restricted ``__builtins__`` mapping the sandbox
  exposes to LLM-generated code.
* :class:`ToolProxy` — exposes ``tools.<name>(...)`` over the unified registry
  (external MCP tools + optional internal read-only callables).
* :func:`run_orchestration_code` — async-exec the code, return
  ``(result, call_log)``.
* :func:`analyze_code_side_effects` — AST-walk the code and decide whether
  any ``tools.<name>`` call references a tool with ``has_side_effects=True``.
"""

from __future__ import annotations

import ast
import asyncio
import builtins as _builtins_module
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# Restricted builtins exposed to LLM-generated orchestration code. Same set as
# the ContextEnricher path used historically — intentionally small to make
# sandbox escape difficult while leaving enough primitives for realistic
# orchestration logic (filtering, mapping, aggregation).
SAFE_BUILTINS: dict[str, Any] = {
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


class ToolProxy:
    """Exposes MCP tools (and optional internal tools) as async methods for LLM-generated code.

    ``task_data`` is pre-injected for external MCP tools whose names appear in
    ``task_data_tool_names``, so generated code never handles it directly. The
    optional ``internal_tools`` registry lets investigate expose read-only
    graph queries (``query_past_causes_for_symptom`` etc.) alongside MCP
    tools — looked up first; falls through to ``client.execute(...)``
    otherwise.

    Every call (internal or external) is appended to ``call_log`` so the
    caller can correlate the orchestration's effects with which tools were
    actually invoked.

    Args:
        client:                MCP client with ``execute(name, **kwargs)``.
        task_data:             Current task fields — auto-injected for MCP
                               tools listed in ``task_data_tool_names``.
        task_data_tool_names:  Names of MCP tools that accept ``task_data``.
        call_log:              Mutable list the proxy appends each invoked
                               tool name to.
        internal_tools:        Optional ``{name: async_callable}`` registry of
                               internal read-only tools (no client routing).
                               Default empty preserves ContextEnricher's
                               historical behavior.
    """

    def __init__(
        self,
        client: Any,
        task_data: dict[str, Any],
        task_data_tool_names: frozenset[str],
        call_log: list[str],
        internal_tools: dict[str, Callable[..., Awaitable[Any]]] | None = None,
    ) -> None:
        self._client = client
        self._task_data = task_data
        self._td_names = task_data_tool_names
        self._log = call_log
        self._internal = internal_tools or {}

    def __getattr__(self, name: str):
        async def call(**kwargs):
            self._log.append(name)
            if name in self._internal:
                return await self._internal[name](**kwargs)
            if name in self._td_names:
                kwargs["task_data"] = self._task_data
            return await self._client.execute(name, **kwargs)
        return call


async def run_orchestration_code(
    code: str,
    *,
    client: Any,
    task_data: dict[str, Any],
    task_data_tool_names: frozenset[str],
    internal_tools: dict[str, Callable[..., Awaitable[Any]]] | None = None,
    extra_globals: dict[str, Any] | None = None,
    timeout: float = 600.0,
    log_prefix: str = "orchestration",
) -> tuple[dict[str, Any], list[str]]:
    """Execute LLM-generated orchestration code in a sandboxed namespace.

    The code is wrapped as the body of ``async def _fn(tools, asyncio, **kwargs)``
    with only :data:`SAFE_BUILTINS` available. Any exception (syntax, runtime,
    or timeout) is logged and an empty result is returned (fail-open).

    Args:
        code:                  Function-body source. Lines are indented by 4
                               spaces before being wrapped.
        client:                MCP client passed to :class:`ToolProxy`.
        task_data:             Auto-injected for tools whose names appear in
                               ``task_data_tool_names``.
        task_data_tool_names:  See :class:`ToolProxy`.
        internal_tools:        See :class:`ToolProxy`.
        extra_globals:         Optional ``{name: value}`` of additional names
                               to bind as local variables inside ``_fn``.
                               Used by ``bamboo investigate`` to expose
                               ``task_id`` and ``task_data`` as in-scope
                               names without string-repr'ing them (which
                               would be fragile for arbitrary task_data
                               values). Names must be valid Python
                               identifiers — invalid names are dropped with
                               a warning.
        timeout:               Seconds before forced cancellation (default 600).
        log_prefix:            Prefix for the narrator hint when calls were made
                               (default ``"orchestration"``).

    Returns:
        ``(result, call_log)`` — ``result`` is the dict the code returned (or
        ``{}`` on any failure); ``call_log`` is the list of tool names invoked
        through the proxy (may be partial if execution was interrupted).
    """
    # Local import to avoid pulling rich/narrator at module load time and to
    # keep this module dependency-light.
    from bamboo.utils.narrator import say  # noqa: PLC0415

    call_log: list[str] = []
    proxy = ToolProxy(
        client=client,
        task_data=task_data,
        task_data_tool_names=task_data_tool_names,
        call_log=call_log,
        internal_tools=internal_tools,
    )
    namespace: dict[str, Any] = {"asyncio": asyncio, "__builtins__": SAFE_BUILTINS}

    # Wrap user code into an async function. extra_globals (e.g. task_id,
    # task_data) become explicit parameters of _fn so they're available as
    # plain names inside the body — much safer than serialising into source.
    safe_extras: dict[str, Any] = {}
    if extra_globals:
        for name, value in extra_globals.items():
            if not name.isidentifier() or name in ("tools", "asyncio", "_fn"):
                logger.warning("orchestration: ignoring invalid extra_globals name %r", name)
                continue
            safe_extras[name] = value
    extra_params = ", ".join(safe_extras.keys())
    sig = "tools, asyncio" + (f", {extra_params}" if extra_params else "")

    indented = "\n".join(f"    {line}" for line in code.splitlines())
    full_code = f"async def _fn({sig}):\n{indented}"
    try:
        exec(full_code, namespace)  # noqa: S102
    except SyntaxError as exc:
        logger.warning("orchestration code has syntax error: %s", exc)
        return {}, call_log
    try:
        result = await asyncio.wait_for(
            namespace["_fn"](proxy, asyncio, **safe_extras), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning("orchestration code timed out after %s s", timeout)
        return {}, call_log
    except Exception as exc:
        logger.warning("orchestration code raised: %s", exc)
        return {}, call_log
    if call_log:
        say(f"  {log_prefix} called: {', '.join(call_log)}")
    if not isinstance(result, dict):
        logger.warning(
            "orchestration code returned %r — expected dict",
            type(result).__name__,
        )
        return {}, call_log
    return result, call_log


def analyze_code_side_effects(
    code: str,
    side_effect_tool_names: set[str] | frozenset[str],
) -> bool:
    """Return True if ``code`` calls any ``tools.<name>`` where name is side-effecting.

    Used by investigate's pre-execution confirmation gate (§D) — only orchestration
    blocks that touch side-effect-ful tools require human ``[y/N/edit]``;
    purely-internal blocks (all calls hit read-only graph queries) auto-execute.

    Args:
        code:                     Source of the orchestration function body.
        side_effect_tool_names:   Names of tools tagged ``has_side_effects=True``
                                  in the caller's unified registry.

    Returns:
        True if any AST node matches ``await? tools.<name>(...)`` where ``name``
        is in ``side_effect_tool_names``. False if the code parses but contains
        no such call, OR if the code is syntactically invalid (the caller's
        ``run_orchestration_code`` will surface the syntax error at execution
        time). Defensive default: if static analysis cannot prove the code is
        side-effect-free, treat it as side-effect-ful by returning True.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Conservatively assume side-effects when we cannot analyze.
        return True

    for node in ast.walk(tree):
        # Match `tools.<name>(...)` and `await tools.<name>(...)`.
        call_node: ast.Call | None = None
        if isinstance(node, ast.Call):
            call_node = node
        if call_node is None:
            continue
        func = call_node.func
        # We're looking for Attribute access of the form `tools.<name>` —
        # either directly (`tools.foo(...)`) or via await (the Call is the
        # operand of an Await elsewhere in the tree; either way we'll visit
        # the inner Call via ast.walk).
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "tools"
            and func.attr in side_effect_tool_names
        ):
            return True
    return False
