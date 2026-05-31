"""Analyze-a-task — run a one-shot root-cause analysis from a Mattermost command.

The MM-native form of ``bamboo analyze``: fetch the task data from PanDA, run the
:class:`~bamboo.agents.reasoning_navigator.ReasoningNavigator` over it, and post the
resulting :class:`~bamboo.models.knowledge_entity.AnalysisResult` back as the same
attachment card the CLI's ``--post-to-mattermost`` uses (:func:`render.analysis_message`).
Read-only: it queries the knowledge base and PanDA but writes nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from bamboo.frontends.mattermost import render
from bamboo.frontends.mattermost.io import MattermostInteractionIO

logger = logging.getLogger(__name__)

# Injectable task-data fetcher (channel-of-record: PanDA). Async ``(task_id) -> dict``.
FetchFn = Callable[[int], Awaitable[dict]]


async def _default_fetch(task_id: int) -> dict:
    """Fetch task data from PanDA (lazy import keeps pandaclient off the hot path)."""
    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415

    return await fetch_task_data(task_id)


async def run_analyze(
    io: MattermostInteractionIO,
    *,
    task_id: Optional[int],
    deps: Any,
    fetch: Optional[FetchFn] = None,
) -> bool:
    """Drive a one-shot analysis for *task_id* and post the result card.

    Returns True when an analysis card was posted, False on a usage/error notice.
    """
    if task_id is None:
        io.notice("Usage: `analyze <taskID>`")
        return False

    fetch = fetch or _default_fetch
    io.notice(f"Analyzing task {task_id}… (this can take a minute)")

    try:
        task_data = await fetch(task_id)
    except Exception as exc:  # noqa: BLE001
        io.notice(f"Could not fetch task data for {task_id}: {exc}")
        return False

    # The reasoning pipeline queries the graph/vector DBs and (for phase 2) MCP;
    # connect is idempotent — mirror the CLI which connects before analyzing.
    try:
        await deps.graph_db.connect()
        await deps.vector_db.connect()
        await deps.mcp_client.connect()
        result = await deps.reasoning_navigator.analyze_task(task_data=task_data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze failed for task %s", task_id)
        io.notice(f"Analysis failed: {exc}")
        return False

    io.transport.send("", props=render.analysis_message(result))
    return True
