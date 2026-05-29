"""Unit tests for per-user PanDA OIDC credential binding.

No pandaserver client needed: the override is exercised via a fake client. Also
checks that the ContextVar propagates across ``asyncio.to_thread`` (the
mechanism the Mattermost bot relies on, since HttpClient runs in a worker
thread).
"""

from __future__ import annotations

import asyncio

import pytest

from bamboo.utils.panda_client import (
    PandaCredentials,
    _apply_credentials,
    _panda_credentials,
    panda_credentials,
)


class FakeHttpClient:
    def __init__(self):
        self.overridden = None

    def override_oidc(self, oidc, id_token, auth_vo):
        self.overridden = (oidc, id_token, auth_vo)


def test_apply_credentials_noop_when_unbound():
    client = FakeHttpClient()
    _apply_credentials(client)
    assert client.overridden is None


def test_apply_credentials_overrides_when_bound():
    client = FakeHttpClient()
    with panda_credentials(PandaCredentials(id_token="tok-123", auth_vo="atlas")):
        _apply_credentials(client)
    assert client.overridden == (True, "tok-123", "atlas")


def test_panda_credentials_resets_after_block():
    with panda_credentials(PandaCredentials(id_token="x")):
        assert _panda_credentials.get() is not None
    assert _panda_credentials.get() is None


def test_panda_credentials_none_is_noop():
    client = FakeHttpClient()
    with panda_credentials(None):
        _apply_credentials(client)
    assert client.overridden is None


@pytest.mark.asyncio
async def test_credentials_propagate_across_to_thread():
    """The blocking HttpClient call runs via asyncio.to_thread; the bound token
    must be visible inside the worker thread (context is copied)."""
    client = FakeHttpClient()

    def in_worker_thread():
        _apply_credentials(client)

    with panda_credentials(PandaCredentials(id_token="thread-tok", auth_vo="vo")):
        await asyncio.to_thread(in_worker_thread)

    assert client.overridden == (True, "thread-tok", "vo")
