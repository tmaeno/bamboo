"""Qdrant vector database backend implementation."""

import logging
from typing import Any, Optional

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError as e:
    raise ImportError(
        "Qdrant backend requires 'qdrant-client' package. "
        "Install it with: pip install qdrant-client"
    ) from e

from bamboo.config import get_settings
from bamboo.database.base import VectorDatabaseBackend

logger = logging.getLogger(__name__)


class QdrantBackend(VectorDatabaseBackend):
    """Qdrant implementation of vector database backend."""

    def __init__(self):
        """Initialize Qdrant backend."""
        self.settings = get_settings()
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = self.settings.qdrant_collection_name

    async def connect(self):
        """Establish connection to Qdrant."""
        try:
            if self.settings.qdrant_api_key:
                self.client = AsyncQdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    check_compatibility=False,
                )
            else:
                self.client = AsyncQdrantClient(
                    url=self.settings.qdrant_url, check_compatibility=False
                )

            await self._ensure_collection()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self):
        """Close Qdrant connection."""
        if self.client:
            await self.client.close()
            logger.info("Qdrant connection closed")

    async def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""
        response = await self.client.get_collections()
        existing = {c.name for c in response.collections}
        if self.collection_name not in existing:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update a document in Qdrant."""
        point = PointStruct(
            id=vector_id,
            vector=embedding,
            payload={"content": content, "section": section, **metadata},
        )
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return vector_id

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents in Qdrant."""
        query_filter = None
        if filter_conditions:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "content": point.payload.get("content", ""),
                "entry": point.payload.get("entry", ""),
                "metadata": {
                    k: v
                    for k, v in point.payload.items()
                    if k not in ("content", "entry")
                },
            }
            for point in response.points
        ]

    async def get_summaries_by_graph_ids(
        self, graph_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch ``Summary`` section entries for the given graph IDs."""
        if not graph_ids:
            return []

        results = []
        for graph_id in graph_ids:
            try:
                points, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="section",
                                match=models.MatchValue(value="Summary"),
                            ),
                            models.FieldCondition(
                                key="graph_id",
                                match=models.MatchValue(value=graph_id),
                            ),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    results.append(
                        {
                            "id": point.id,
                            "score": 1.0,
                            "content": point.payload.get("content", ""),
                            "entry": point.payload.get("entry", ""),
                            "metadata": {
                                k: v
                                for k, v in point.payload.items()
                                if k not in ("content", "entry")
                            },
                        }
                    )
            except Exception as e:
                logger.warning(
                    "Failed to fetch summary for graph_id=%s: %s", graph_id, e
                )
        return results

    async def collection_exists(self) -> bool:
        """Return True if the Qdrant collection exists."""
        response = await self.client.get_collections()
        return self.collection_name in {c.name for c in response.collections}

    async def clear_all(self) -> None:
        """Drop and recreate the Qdrant collection (all vectors deleted)."""
        await self.client.delete_collection(self.collection_name)
        logger.info("Qdrant: collection '%s' dropped", self.collection_name)
        await self._ensure_collection()
        logger.info("Qdrant: collection '%s' recreated", self.collection_name)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[doc_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific document by ID."""
        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
                with_payload=True,
            )
            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "entry": point.payload.get("entry", ""),
                    "metadata": {
                        k: v
                        for k, v in point.payload.items()
                        if k not in ("content", "entry")
                    },
                }
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
        return None
