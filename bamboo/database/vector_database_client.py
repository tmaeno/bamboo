"""Vector database client with pluggable backend support."""

import logging
from typing import Any, Optional

from bamboo.database.factory import get_vector_backend

logger = logging.getLogger(__name__)


class VectorDatabaseClient:
    """Client for vector database operations with pluggable backend support."""

    def __init__(self):
        """Initialize vector database client."""
        self._backend = get_vector_backend()

    async def connect(self):
        """Establish connection to vector database backend."""
        await self._backend.connect()

    async def close(self):
        """Close connection."""
        await self._backend.close()

    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update a document in the vector database."""
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
        """Search for similar documents."""
        return await self._backend.search_similar(
            query_embedding, limit, score_threshold, filter_conditions
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        return await self._backend.delete_document(doc_id)

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific document by ID."""
        return await self._backend.get_document(doc_id)
