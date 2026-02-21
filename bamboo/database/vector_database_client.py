"""Vector database client: thin façade over the pluggable backend.

:class:`VectorDatabaseClient` is the single interface used by the rest of the
application to interact with the vector database.  It delegates every call to
the :class:`~bamboo.database.base.VectorDatabaseBackend` selected by
:func:`~bamboo.database.factory.get_vector_backend`.

Usage::

    client = VectorDatabaseClient()
    await client.connect()
    try:
        await client.upsert_section_vector(...)
        results = await client.search_similar(...)
    finally:
        await client.close()
"""

import logging
from typing import Any, Optional

from bamboo.database.factory import get_vector_backend

logger = logging.getLogger(__name__)


class VectorDatabaseClient:
    """Façade over the configured :class:`~bamboo.database.base.VectorDatabaseBackend`.

    All method signatures mirror those of the backend interface.  See
    :class:`~bamboo.database.base.VectorDatabaseBackend` for full
    documentation of each operation.
    """

    def __init__(self):
        self._backend = get_vector_backend()

    async def connect(self):
        """Open the backend connection."""
        await self._backend.connect()

    async def close(self):
        """Close the backend connection."""
        await self._backend.close()

    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update a vector point in the given section."""
        return await self._backend.upsert_section_vector(
            vector_id, embedding, content, section, metadata
        )

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return the most similar points above *score_threshold*."""
        return await self._backend.search_similar(
            query_embedding, limit, score_threshold, filter_conditions
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete the point with the given ID."""
        return await self._backend.delete_document(doc_id)

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single point by ID."""
        return await self._backend.get_document(doc_id)

    async def get_summaries_by_graph_ids(
        self, graph_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch ``Summary`` section entries for the given graph IDs.

        Second step of the two-step vector retrieval pattern used by
        :meth:`~bamboo.agents.reasoning_navigator.ReasoningNavigator._query_vector_database`.
        """
        return await self._backend.get_summaries_by_graph_ids(graph_ids)
