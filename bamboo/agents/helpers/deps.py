"""Shared collaborator-construction factory.

:func:`build_deps` is the single place that wires the engine's collaborators
(graph/vector DBs, MCP client, extractor strategy, explorer, reasoning navigator,
knowledge accumulator) into a :class:`~bamboo.agents.investigation_session._Deps`
bundle.  Every entry point — the ``bamboo investigate``/``analyze`` CLIs and the
Mattermost bot — calls this so the **only** difference between them is the injected
:class:`~bamboo.frontends.base.InteractionIO`.

Returns *unconnected* clients: callers open connections (``graph_db.connect()`` …)
when they need them, exactly as before.  Imports are done inside the function to
keep ``bamboo --help`` fast and to avoid import cycles.
"""

from __future__ import annotations

from typing import Any, Optional


def build_deps(*, io: Any = None, settings: Optional[Any] = None) -> Any:
    """Build and return a ``_Deps`` collaborator bundle.

    Args:
        io:       Optional :class:`~bamboo.frontends.base.InteractionIO` to drive
                  the session (Rich terminal or Mattermost).  Threaded into the
                  MCP interactive client and the explorer so interactive tools
                  (e.g. ``request_human_input``) route through it.  ``None`` keeps
                  the legacy stdin/TTY behavior for one-shot/automation callers.
        settings: Optional :class:`~bamboo.config.Settings`; defaults to
                  :func:`~bamboo.config.get_settings`.
    """
    from bamboo.agents.context_enricher import ContextEnricher  # noqa: PLC0415
    from bamboo.agents.extractors import get_extraction_strategy  # noqa: PLC0415
    from bamboo.agents.extractors.panda_knowledge_extractor import (  # noqa: PLC0415
        ErrorCategoryClassifier,
    )
    from bamboo.agents.investigation_session import _Deps  # noqa: PLC0415
    from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator  # noqa: PLC0415
    from bamboo.agents.reasoning_navigator import ReasoningNavigator  # noqa: PLC0415
    from bamboo.config import get_settings  # noqa: PLC0415
    from bamboo.database.graph_database_client import GraphDatabaseClient  # noqa: PLC0415
    from bamboo.database.vector_database_client import VectorDatabaseClient  # noqa: PLC0415
    from bamboo.mcp.factory import build_mcp_client  # noqa: PLC0415

    settings = settings or get_settings()
    strategy = get_extraction_strategy()  # strategy-aware (EXTRACTION_STRATEGY, default "panda")

    mcp_client = build_mcp_client(settings, strategy, io=io)
    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()
    explorer = ContextEnricher(
        mcp_client, source_navigator=strategy.source_navigator(), io=io
    )
    # KnowledgeAccumulator/ReasoningNavigator self-create their own
    # KnowledgeGraphExtractor() — do NOT pass extractor= (it isn't a parameter).
    accumulator = KnowledgeAccumulator(
        graph_db=graph_db, vector_db=vector_db, reviewer=None, explorer=explorer
    )
    reasoning_navigator = ReasoningNavigator(
        graph_db=graph_db, vector_db=vector_db, explorer=explorer
    )
    return _Deps(
        mcp_client=mcp_client,
        graph_db=graph_db,
        vector_db=vector_db,
        extractor=strategy,  # active strategy; used by the orchestrator for prefetch_hints
        reasoning_navigator=reasoning_navigator,
        knowledge_accumulator=accumulator,
        error_classifier=ErrorCategoryClassifier(),
        io=io,
    )


async def resolve_task_data(
    task_id: Any = None,
    task_data: Optional[dict] = None,
    *,
    verbose: bool = False,
) -> Optional[dict]:
    """THE single way to obtain PanDA ``task_data`` for an entry point.

    Returns ``task_data`` unchanged when the caller already has it (from a
    ``--task-data`` file, a draft's embedded dict, chat context, …); otherwise
    fetches it from PanDA via :func:`bamboo.utils.panda_client.fetch_task_data`
    when only a ``task_id`` is given.  Returns ``None`` when neither is provided.

    This is input *acquisition*, which legitimately lives at the entry point
    (the only layer that knows the source) — unlike ``prefetch_hints``, the
    strategy-owned algorithm step that stays encapsulated inside the agents.
    Consolidating it here is what keeps the ``analyze`` / ``investigate`` /
    ``capture`` / ``populate`` fetch from drifting apart again.

    Raises on fetch failure — each caller keeps its own try/except for error UX.
    """
    if task_data is not None or task_id is None:
        return task_data
    from bamboo.utils.panda_client import fetch_task_data  # noqa: PLC0415  (keep pandaclient import lazy)

    return await fetch_task_data(task_id, verbose=verbose)
