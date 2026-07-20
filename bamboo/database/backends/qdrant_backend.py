"""Qdrant vector database backend implementation."""

import asyncio
import contextlib
import logging
from typing import Any, Optional

import httpx

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
        # Serialises the first connect so concurrent first queries connect once.
        self._connect_lock = asyncio.Lock()

    async def _ensure_connected(self):
        """Open the client on first use (idempotent, concurrency-guarded).

        Every public method calls this, so callers never have to remember to
        :meth:`connect`. A failed connect leaves the client unset (so a later call
        retries) and propagates, so the caller can degrade.
        """
        if self.client is not None:
            return
        async with self._connect_lock:
            if self.client is not None:  # another coroutine won the race
                return
            if self.settings.qdrant_api_key:
                client = AsyncQdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    check_compatibility=False,
                )
            else:
                client = AsyncQdrantClient(
                    url=self.settings.qdrant_url, check_compatibility=False
                )
            self.client = client  # set before _ensure_collection (which uses it)
            try:
                await self._ensure_collection()
            except Exception:  # noqa: BLE001 — don't leave a half-open client
                with contextlib.suppress(Exception):
                    await client.close()
                self.client = None
                raise
            logger.info("Successfully connected to Qdrant")

    async def connect(self):
        """Establish connection to Qdrant (idempotent). Explicit callers keep working."""
        try:
            await self._ensure_connected()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self):
        """Close Qdrant connection."""
        if self.client:
            await self.client.close()
            self.client = None  # allow a later call to re-connect lazily
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
        await self._ensure_connected()
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
        await self._ensure_connected()
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

        await self._ensure_connected()
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
        await self._ensure_connected()
        response = await self.client.get_collections()
        return self.collection_name in {c.name for c in response.collections}

    async def clear_all(self) -> None:
        """Drop and recreate the Qdrant collection (all vectors deleted)."""
        await self._ensure_connected()
        await self.client.delete_collection(self.collection_name)
        logger.info("Qdrant: collection '%s' dropped", self.collection_name)
        await self._ensure_collection()
        logger.info("Qdrant: collection '%s' recreated", self.collection_name)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        await self._ensure_connected()
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
        await self._ensure_connected()
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

    async def _download_collection_snapshot(
        self, collection_name: str, out_path: str
    ) -> str:
        """Create a snapshot of *collection_name*, stream it to *out_path*, then delete
        the server-side copy. Returns the server-side snapshot name.

        Uses only URL / API key via the Qdrant Snapshot API — no access to the
        server's on-disk storage dir — so it works against any reachable Qdrant.
        """
        snapshot = await self.client.create_snapshot(collection_name=collection_name)
        name = getattr(snapshot, "name", "")
        if not name:
            raise RuntimeError(
                f"Qdrant returned no snapshot for collection {collection_name!r}"
            )

        base = self.settings.qdrant_url.rstrip("/")
        url = f"{base}/collections/{collection_name}/snapshots/{name}"
        headers = {}
        if self.settings.qdrant_api_key:
            headers["api-key"] = self.settings.qdrant_api_key

        try:
            async with httpx.AsyncClient(timeout=None) as http:
                async with http.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    with open(out_path, "wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            fh.write(chunk)
        finally:
            with contextlib.suppress(Exception):
                await self.client.delete_snapshot(
                    collection_name=collection_name, snapshot_name=name
                )

        logger.info(
            "Exported Qdrant snapshot for collection '%s' → %s", collection_name, out_path
        )
        return name

    async def export_snapshot(
        self, out_path: str, collection_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Create a snapshot of a single collection and download it to *out_path*.

        Uses only the connection info in :attr:`settings` (URL / API key) via the
        Qdrant Snapshot API — no access to the server's on-disk storage dir — so it
        works against any reachable Qdrant (local Docker or a managed instance). The
        server-side snapshot is deleted after a successful download. *collection_name*
        defaults to the configured KB collection.

        Returns metadata for the KB manifest: the collection name, the snapshot
        filename, and the Qdrant server version.
        """
        await self._ensure_connected()
        server_version = await self._server_version()
        coll = collection_name or self.collection_name
        name = await self._download_collection_snapshot(coll, out_path)
        return {
            "qdrant_collection": coll,
            "qdrant_snapshot": name,
            "qdrant_version": server_version,
        }

    async def export_all_snapshots(self, out_dir: str) -> dict[str, Any]:
        """Snapshot *every* Qdrant collection into *out_dir*, one file per collection.

        Enumerates collections via the API (no hardcoded name) so all vector data —
        the KB collection plus auxiliaries like the doc-navigator's ``panda_docs`` /
        ``panda_docs_meta`` — travels with the batch KB. Each collection is written as
        ``qdrant-<collection>.snapshot``; the batch container recovers them all with a
        repeated ``qdrant --snapshot <file>:<collection>`` flag (see
        ``deploy/batch/entrypoint.sh``). Used by ``bamboo dump-kb``.

        Returns manifest metadata: the per-collection list (``qdrant_collections``),
        the primary KB collection, and the Qdrant server version.
        """
        from pathlib import Path  # noqa: PLC0415

        await self._ensure_connected()
        server_version = await self._server_version()
        names = sorted(c.name for c in (await self.client.get_collections()).collections)
        entries: list[dict[str, str]] = []
        for coll in names:
            snapshot_file = f"qdrant-{coll}.snapshot"
            await self._download_collection_snapshot(
                coll, str(Path(out_dir) / snapshot_file)
            )
            entries.append({"collection": coll, "snapshot_file": snapshot_file})
        logger.info("Exported %d Qdrant collection snapshot(s) → %s", len(entries), out_dir)
        return {
            "qdrant_collections": entries,
            "primary_collection": self.collection_name,
            "qdrant_version": server_version,
        }

    async def _server_version(self) -> str:
        """Return the Qdrant server version string (best-effort; ``""`` on failure)."""
        base = self.settings.qdrant_url.rstrip("/")
        headers = {}
        if self.settings.qdrant_api_key:
            headers["api-key"] = self.settings.qdrant_api_key
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.get(f"{base}/", headers=headers)
                resp.raise_for_status()
                return str(resp.json().get("version", ""))
        except Exception as exc:  # noqa: BLE001 — version is informational metadata
            logger.warning("Could not read Qdrant server version: %s", exc)
            return ""
