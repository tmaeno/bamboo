"""Lazy/idempotent DB connect in the graph + vector backends.

The backends connect on first use (idempotent, concurrency-guarded) so no caller
ever has to remember to ``connect()`` — which is what previously made the
Mattermost ``investigate`` flow crash on a ``None`` Neo4j driver. These tests use
fake driver/client factories (no live Neo4j/Qdrant) to assert: auto-connect on
first use, connect-once under repeated/concurrent use, and reconnect after close.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------


class _FakeNeo4jSession:
    async def run(self, *a, **k):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeNeo4jDriver:
    def __init__(self) -> None:
        self.closed = False

    async def verify_connectivity(self):
        return None

    def session(self, **kwargs):
        return _FakeNeo4jSession()

    async def close(self):
        self.closed = True


def _patch_neo4j(monkeypatch):
    from bamboo.database.backends import neo4j_backend

    factory = MagicMock(side_effect=lambda *a, **k: _FakeNeo4jDriver())
    monkeypatch.setattr(neo4j_backend, "AsyncGraphDatabase", SimpleNamespace(driver=factory))
    return neo4j_backend, factory


@pytest.mark.asyncio
async def test_neo4j_lazy_connect_once_idempotent_and_reconnect(monkeypatch):
    neo4j_backend, factory = _patch_neo4j(monkeypatch)
    be = neo4j_backend.Neo4jBackend()
    assert be.driver is None

    await be._ensure_connected()
    assert be.driver is not None
    assert factory.call_count == 1

    await be._ensure_connected()  # idempotent — no second driver
    assert factory.call_count == 1

    await be.close()
    assert be.driver is None  # close resets so a later call reconnects
    await be._ensure_connected()
    assert factory.call_count == 2


@pytest.mark.asyncio
async def test_neo4j_session_connects_without_explicit_connect(monkeypatch):
    neo4j_backend, factory = _patch_neo4j(monkeypatch)
    be = neo4j_backend.Neo4jBackend()
    # A query path uses _session(), which must connect on its own.
    async with be._session(database="neo4j") as session:
        assert session is not None
    assert be.driver is not None and factory.call_count == 1


@pytest.mark.asyncio
async def test_neo4j_concurrent_first_use_connects_once(monkeypatch):
    neo4j_backend, factory = _patch_neo4j(monkeypatch)
    be = neo4j_backend.Neo4jBackend()
    await asyncio.gather(*[be._ensure_connected() for _ in range(5)])
    assert factory.call_count == 1


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.closed = False

    async def get_collections(self):
        return SimpleNamespace(collections=[])

    async def create_collection(self, **k):
        return None

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_qdrant_lazy_connect_once_idempotent_and_reconnect(monkeypatch):
    from bamboo.database.backends import qdrant_backend

    factory = MagicMock(side_effect=lambda *a, **k: _FakeQdrantClient())
    monkeypatch.setattr(qdrant_backend, "AsyncQdrantClient", factory)
    be = qdrant_backend.QdrantBackend()
    assert be.client is None

    await be._ensure_connected()
    assert be.client is not None
    assert factory.call_count == 1

    await be._ensure_connected()  # idempotent
    assert factory.call_count == 1

    await be.close()
    assert be.client is None
    await be._ensure_connected()
    assert factory.call_count == 2


# ---------------------------------------------------------------------------
# Mattermost serve releases the lazily-opened DBs at session end
# ---------------------------------------------------------------------------


class _MiniTransport:
    """Bot-less transport: stream_narration() no-ops (no live post)."""

    channel_id = None

    def send(self, text, *, props=None):
        pass

    async def next_reply(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_run_session_closes_dbs(monkeypatch):
    import bamboo.agents.investigation_session as sess
    from bamboo.frontends.mattermost import serve
    from bamboo.frontends.mattermost.bot import Command

    gdb = MagicMock()
    gdb.close = AsyncMock()
    vdb = MagicMock()
    vdb.close = AsyncMock()
    deps = SimpleNamespace(
        graph_db=gdb, vector_db=vdb,
        knowledge_accumulator=MagicMock(), mcp_client=MagicMock(), console=None,
    )
    monkeypatch.setattr(serve, "_build_deps", lambda io: deps)
    monkeypatch.setattr(serve, "resolve_user_credentials", AsyncMock(return_value=None))
    fake_orch = SimpleNamespace(start=AsyncMock(), run=AsyncMock(), finalize=AsyncMock())
    monkeypatch.setattr(sess, "InvestigationOrchestrator", lambda **k: fake_orch)

    await serve._run_session(_MiniTransport(), Command(kind="investigate", task_id=1, verbose=False))

    fake_orch.start.assert_awaited_once()
    gdb.close.assert_awaited_once()
    vdb.close.assert_awaited_once()
