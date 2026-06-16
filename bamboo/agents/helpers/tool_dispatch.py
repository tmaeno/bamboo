"""Single-point dispatcher for one MCP tool call.

This module exists so :class:`bamboo.agents.context_enricher.ContextEnricher`
and :class:`bamboo.agents.investigation_session.InvestigationOrchestrator`
share the same tool-routing layer — special-cases (``search_panda_server_source``
→ source navigator), the ``task_data`` argument auto-injection for tools that
declare a need, and the underlying ``client.execute(...)`` fall-through all live
in one place.

The function returns a coroutine (or another awaitable) so callers can either
``await`` it directly or hand it to ``asyncio.gather`` for concurrent execution
— matching how ``ContextEnricher`` uses it today.
"""

from __future__ import annotations

from typing import Any, Awaitable


def dispatch_tool_call(
    tool_call: dict,
    *,
    client: Any,
    source_navigator: Any | None,
    task_data: dict[str, Any],
    task_data_tool_names: frozenset[str],
) -> Awaitable[Any]:
    """Return the awaitable for one planned tool call.

    Routes ``search_panda_server_source`` to ``source_navigator.navigate(query)``
    when one is configured; otherwise dispatches to ``client.execute(tool, **args)``.
    For tools whose name is in ``task_data_tool_names``, ``task_data`` is merged
    into the args before execution — matching ``ContextEnricher._tool_coro``'s
    pre-extraction behavior exactly.

    Args:
        tool_call:             ``{"tool": <name>, "args": {...}}`` planned call.
        client:                MCP client with ``execute(name, **kwargs)``.
        source_navigator:      Optional source-code navigator with ``navigate(query)``.
        task_data:             Current task fields — injected into args for
                               tools listed in ``task_data_tool_names``.
        task_data_tool_names:  Set of tool names that accept ``task_data`` as
                               an argument (from ``client.task_data_tools()``).

    Returns:
        The unawaited coroutine. Caller is responsible for awaiting (or
        gathering).
    """
    if tool_call["tool"] == "search_panda_server_source" and source_navigator is not None:
        query = tool_call.get("args", {}).get("query", "")
        return source_navigator.navigate(query)
    args = (
        {**tool_call.get("args", {}), "task_data": task_data}
        if tool_call["tool"] in task_data_tool_names
        else tool_call.get("args", {})
    )
    return client.execute(tool_call["tool"], **args)
