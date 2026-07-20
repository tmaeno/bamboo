"""Unit tests for PandaDocNavigator staleness / index-meta behaviour.

No live Qdrant / GitHub: the navigator is built with ``object.__new__`` (skipping the
heavy ``__init__`` that loads embeddings + LLMs) and talks to an in-memory fake Qdrant.
These pin the two behaviours the batch pipeline relies on:

- ``DOC_INDEX_FREEZE`` short-circuits staleness so no rebuild / network call happens;
- index metadata round-trips through the dedicated ``panda_docs_meta`` collection
  (so it travels with the KB snapshot), and ``invalidate_doc_cache`` drops it.
"""

import asyncio
from types import SimpleNamespace

import pytest

from bamboo.agents import panda_doc_navigator as nav_mod
from bamboo.agents.panda_doc_navigator import (
    _META_COLLECTION,
    _META_POINT_ID,
    PandaDocNavigator,
)


class _FakeQdrant:
    """Minimal in-memory async stand-in for AsyncQdrantClient."""

    def __init__(self, collections=None):
        # name -> {point_id: payload}
        self._collections: dict[str, dict] = {c: {} for c in (collections or [])}

    async def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self._collections]
        )

    async def create_collection(self, collection_name, vectors_config=None):
        self._collections.setdefault(collection_name, {})

    async def upsert(self, collection_name, points):
        coll = self._collections.setdefault(collection_name, {})
        for p in points:
            coll[p.id] = dict(p.payload)

    async def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        coll = self._collections.get(collection_name, {})
        return [
            SimpleNamespace(id=i, payload=coll[i]) for i in ids if i in coll
        ]

    async def delete_collection(self, collection_name):
        self._collections.pop(collection_name, None)

    async def close(self):
        pass


def _bare_navigator(freeze: bool = False) -> PandaDocNavigator:
    """A navigator without the heavy __init__ (no embeddings/LLMs loaded)."""
    nav = object.__new__(PandaDocNavigator)
    nav._settings = SimpleNamespace(doc_index_freeze=freeze, embedding_dimension=8)
    return nav


# --------------------------------------------------------------------------- #
# Freeze short-circuit (A)
# --------------------------------------------------------------------------- #


def test_check_staleness_frozen_returns_false_without_network():
    nav = _bare_navigator(freeze=True)

    async def _boom():  # must never be reached when frozen
        raise AssertionError("_fetch_tree_sha called while frozen")

    nav._fetch_tree_sha = _boom
    assert asyncio.run(nav._check_staleness()) is False


def test_check_staleness_rebuilds_when_sha_changes():
    nav = _bare_navigator(freeze=False)

    async def _sha():
        return "new-sha"

    async def _meta():
        return {"sha": "old-sha"}

    nav._fetch_tree_sha = _sha
    nav._read_meta = _meta
    assert asyncio.run(nav._check_staleness()) is True


def test_check_staleness_no_rebuild_when_sha_matches():
    nav = _bare_navigator(freeze=False)

    async def _sha():
        return "same-sha"

    async def _meta():
        return {"sha": "same-sha"}

    nav._fetch_tree_sha = _sha
    nav._read_meta = _meta
    assert asyncio.run(nav._check_staleness()) is False


def test_check_staleness_github_unreachable_uses_existing_when_meta_present():
    nav = _bare_navigator(freeze=False)

    async def _sha():
        return None  # GitHub unreachable

    async def _meta():
        return {"sha": "whatever"}

    nav._fetch_tree_sha = _sha
    nav._read_meta = _meta
    # meta present → not stale (do not rebuild); meta absent → rebuild.
    assert asyncio.run(nav._check_staleness()) is False


# --------------------------------------------------------------------------- #
# Index-meta round-trip through Qdrant (B)
# --------------------------------------------------------------------------- #


def test_write_then_read_meta_roundtrips_via_qdrant():
    nav = _bare_navigator()
    fake = _FakeQdrant()
    nav._make_qdrant_client = lambda: fake

    asyncio.run(
        nav._write_meta("sha-123", system_summary="SUMMARY", file_shas={"a.rst": "h1"})
    )

    # Stored as a single sentinel point in the dedicated meta collection.
    assert _META_COLLECTION in fake._collections
    assert _META_POINT_ID in fake._collections[_META_COLLECTION]

    meta = asyncio.run(nav._read_meta())
    assert meta["sha"] == "sha-123"
    assert meta["system_summary"] == "SUMMARY"
    assert meta["file_shas"] == {"a.rst": "h1"}

    assert asyncio.run(nav.get_system_summary()) == "SUMMARY"


def test_read_meta_returns_none_when_collection_absent():
    nav = _bare_navigator()
    nav._make_qdrant_client = lambda: _FakeQdrant()  # no collections
    assert asyncio.run(nav._read_meta()) is None
    assert asyncio.run(nav.get_system_summary()) == ""


def test_invalidate_doc_cache_drops_doc_collections(monkeypatch, tmp_path):
    fake = _FakeQdrant(collections=["panda_docs", _META_COLLECTION, "bamboo_knowledge"])
    monkeypatch.setattr(nav_mod, "_make_qdrant_client_from_settings", lambda s: fake)
    monkeypatch.setattr(nav_mod, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(nav_mod, "_NODE_CACHE_FILE", tmp_path / "node_cache.json")

    removed = asyncio.run(nav_mod.invalidate_doc_cache())

    assert removed is True
    assert "panda_docs" not in fake._collections
    assert _META_COLLECTION not in fake._collections
    # Unrelated collections are left untouched.
    assert "bamboo_knowledge" in fake._collections


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
