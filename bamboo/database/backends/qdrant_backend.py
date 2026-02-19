"""Qdrant vector database backend implementation."""

import logging
from typing import Any, Optional

try:
    from qdrant_client import QdrantClient as QdrantClientSDK
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
        self.client: Optional[QdrantClientSDK] = None
        self.collection_name = self.settings.qdrant_collection_name

    async def connect(self):
        """Establish connection to Qdrant."""
        try:
            if self.settings.qdrant_api_key:
                self.client = QdrantClientSDK(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                )
            else:
                self.client = QdrantClientSDK(url=self.settings.qdrant_url)

            # Create collection if it doesn't exist
            await self._ensure_collection()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self):
        """Close Qdrant connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant connection closed")

    async def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
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
            payload={
                "content": content,
                "section": section,
                **metadata,
            },
        )

        self.client.upsert(
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
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "entry": result.payload.get("entry", ""),
                "metadata": {
                    k: v
                    for k, v in result.payload.items()
                    if k not in ["content", "entry"]
                },
            }
            for result in results
        ]

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.client.delete(
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
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id],
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
                        if k not in ["content", "entry"]
                    },
                }
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
        return None


