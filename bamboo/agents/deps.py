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
