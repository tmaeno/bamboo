"""Tests for the batch KB dump: QdrantBackend snapshot export + `bamboo dump-kb`.

All are exercised with mocks — no live Qdrant/Neo4j — so they run in CI. They pin
the behaviours the batch pipeline depends on: a snapshot is created via the Snapshot
API (URL/collection only) and downloaded to disk, ``export_all_snapshots`` enumerates
*every* collection so auxiliaries (e.g. the doc-navigator's ``panda_docs``) travel with
the KB, and `dump-kb` stamps a metadata.json entirely from ``get_settings()`` + live
server versions.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bamboo.database.backends.qdrant_backend import QdrantBackend


class _FakeStream:
    """Async context manager standing in for httpx's streaming response."""

    def __init__(self, data: bytes):
        self._data = data

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeHTTPClient:
    """Async context manager standing in for httpx.AsyncClient."""

    def __init__(self, *args, **kwargs):
        self._data = b"SNAPSHOT-BYTES"

    def stream(self, method, url, headers=None):
        return _FakeStream(self._data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_export_snapshot_creates_downloads_and_cleans_up(tmp_path, monkeypatch):
    """export_snapshot creates a snapshot, downloads it, deletes the server copy."""
    backend = QdrantBackend()
    backend.collection_name = "bamboo_knowledge"
    # Pre-set client so _ensure_connected short-circuits (no real connection).
    backend.client = SimpleNamespace(
        create_snapshot=AsyncMock(return_value=SimpleNamespace(name="snap-1")),
        delete_snapshot=AsyncMock(),
    )
    monkeypatch.setattr(backend, "_server_version", AsyncMock(return_value="1.13.6"))
    monkeypatch.setattr(
        "bamboo.database.backends.qdrant_backend.httpx.AsyncClient", _FakeHTTPClient
    )

    out = tmp_path / "qdrant.snapshot"
    meta = asyncio.run(backend.export_snapshot(str(out)))

    # Downloaded bytes landed on disk.
    assert out.read_bytes() == b"SNAPSHOT-BYTES"
    # Metadata for the manifest.
    assert meta == {
        "qdrant_collection": "bamboo_knowledge",
        "qdrant_snapshot": "snap-1",
        "qdrant_version": "1.13.6",
    }
    # Created against the configured collection, then cleaned up server-side.
    backend.client.create_snapshot.assert_awaited_once_with(
        collection_name="bamboo_knowledge"
    )
    backend.client.delete_snapshot.assert_awaited_once_with(
        collection_name="bamboo_knowledge", snapshot_name="snap-1"
    )


def test_export_all_snapshots_enumerates_every_collection(tmp_path, monkeypatch):
    """export_all_snapshots snapshots each collection returned by the API, one file each."""
    backend = QdrantBackend()
    backend.collection_name = "bamboo_knowledge"
    backend.client = SimpleNamespace(
        get_collections=AsyncMock(
            return_value=SimpleNamespace(
                collections=[
                    SimpleNamespace(name="panda_docs"),
                    SimpleNamespace(name="bamboo_knowledge"),
                    SimpleNamespace(name="panda_docs_meta"),
                ]
            )
        ),
        create_snapshot=AsyncMock(return_value=SimpleNamespace(name="snap-1")),
        delete_snapshot=AsyncMock(),
    )
    monkeypatch.setattr(backend, "_server_version", AsyncMock(return_value="1.13.6"))
    monkeypatch.setattr(
        "bamboo.database.backends.qdrant_backend.httpx.AsyncClient", _FakeHTTPClient
    )

    meta = asyncio.run(backend.export_all_snapshots(str(tmp_path)))

    # One snapshot file per collection landed on disk, named qdrant-<collection>.snapshot.
    for coll in ("bamboo_knowledge", "panda_docs", "panda_docs_meta"):
        assert (tmp_path / f"qdrant-{coll}.snapshot").read_bytes() == b"SNAPSHOT-BYTES"
    # Manifest lists every collection (sorted) with its file; primary is the KB collection.
    assert meta["qdrant_collections"] == [
        {"collection": "bamboo_knowledge", "snapshot_file": "qdrant-bamboo_knowledge.snapshot"},
        {"collection": "panda_docs", "snapshot_file": "qdrant-panda_docs.snapshot"},
        {"collection": "panda_docs_meta", "snapshot_file": "qdrant-panda_docs_meta.snapshot"},
    ]
    assert meta["primary_collection"] == "bamboo_knowledge"
    assert meta["qdrant_version"] == "1.13.6"
    # Created once per collection, then cleaned up server-side.
    assert backend.client.create_snapshot.await_count == 3
    assert backend.client.delete_snapshot.await_count == 3


def test_export_snapshot_raises_when_no_snapshot(tmp_path, monkeypatch):
    """A snapshot response without a name is a hard error (nothing to download)."""
    backend = QdrantBackend()
    backend.client = SimpleNamespace(
        create_snapshot=AsyncMock(return_value=SimpleNamespace(name="")),
        delete_snapshot=AsyncMock(),
    )
    monkeypatch.setattr(backend, "_server_version", AsyncMock(return_value=""))

    try:
        asyncio.run(backend.export_snapshot(str(tmp_path / "x.snapshot")))
    except RuntimeError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected RuntimeError when snapshot has no name")


def test_dump_kb_writes_metadata_from_settings(tmp_path, monkeypatch):
    """`dump-kb` stamps metadata.json from get_settings() + live server versions."""
    from bamboo.config import get_settings
    from bamboo.scripts import dump_kb

    settings = get_settings()

    # Fake vector backend: real QdrantBackend (so the isinstance check passes) with the
    # snapshot export + close stubbed out. dump-kb exports *all* collections.
    collections = [
        {
            "collection": settings.qdrant_collection_name,
            "snapshot_file": f"qdrant-{settings.qdrant_collection_name}.snapshot",
        },
        {"collection": "panda_docs", "snapshot_file": "qdrant-panda_docs.snapshot"},
    ]
    backend = QdrantBackend()
    backend.export_all_snapshots = AsyncMock(
        return_value={
            "qdrant_collections": collections,
            "primary_collection": settings.qdrant_collection_name,
            "qdrant_version": "1.13.6",
        }
    )
    backend.close = AsyncMock()
    monkeypatch.setattr(
        "bamboo.database.factory.get_vector_backend", lambda: backend
    )
    monkeypatch.setattr(
        dump_kb, "_neo4j_version", AsyncMock(return_value="5.26.28")
    )

    rc = asyncio.run(dump_kb._run(str(tmp_path)))
    assert rc == 0

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta == {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "neo4j_database": settings.neo4j_database,
        "neo4j_version": "5.26.28",
        "qdrant_collection": settings.qdrant_collection_name,
        "qdrant_collections": collections,
        "qdrant_version": "1.13.6",
    }
    backend.export_all_snapshots.assert_awaited_once()
    backend.close.assert_awaited_once()
