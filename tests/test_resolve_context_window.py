"""Unit tests for :func:`bamboo.llm.llm_client.resolve_context_window`.

Covers the override path, Ollama auto-detection via ``/api/ps`` (loaded, tag-tolerant
match, not-loaded, unreachable) and the cloud constants. ``httpx.get`` is monkeypatched
on the real module — the resolver imports it lazily, so patching the attribute works.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import bamboo.llm.llm_client as llm


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _settings(**kw):
    base = {
        "llm_context_window": 0,
        "llm_provider": "ollama",
        "llm_model": "qwen3.6:latest",
        "ollama_base_url": "http://localhost:11434",
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clear_cache():
    llm._CTX_CACHE.clear()
    yield
    llm._CTX_CACHE.clear()


def test_explicit_override_wins_without_probe(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not probe /api/ps when explicitly overridden")

    monkeypatch.setattr(httpx, "get", _boom)
    assert llm.resolve_context_window(_settings(llm_context_window=5000)) == 5000


def test_ollama_reads_served_context_length(monkeypatch):
    payload = {"models": [{"name": "qwen3.6:latest", "model": "qwen3.6:latest", "context_length": 262144}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    assert llm.resolve_context_window(_settings()) == 262144


def test_ollama_matches_model_ignoring_latest_tag(monkeypatch):
    payload = {"models": [{"name": "qwen3.6:latest", "model": "qwen3.6:latest", "context_length": 40960}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    assert llm.resolve_context_window(_settings(llm_model="qwen3.6")) == 40960


def test_ollama_model_not_loaded_falls_back(monkeypatch):
    payload = {"models": [{"name": "other:latest", "model": "other:latest", "context_length": 8000}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp(payload))
    assert llm.resolve_context_window(_settings()) == llm._OLLAMA_FALLBACK_CONTEXT


def test_ollama_unreachable_falls_back(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert llm.resolve_context_window(_settings()) == llm._OLLAMA_FALLBACK_CONTEXT


def test_cloud_uses_constant_without_probe(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("cloud providers must not probe /api/ps")

    monkeypatch.setattr(httpx, "get", _boom)
    assert llm.resolve_context_window(_settings(llm_provider="anthropic")) == 200_000
    assert llm.resolve_context_window(_settings(llm_provider="openai")) == 128_000
