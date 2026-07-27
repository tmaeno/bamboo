"""Regression tests for the Ollama branch of :func:`bamboo.llm.llm_client._build_llm`.

``base_url`` used to be omitted when constructing ``ChatOllama``. langchain-ollama then
builds ``ollama.Client(host=None)``, which silently resolves to ``OLLAMA_HOST`` or
``http://127.0.0.1:11434`` — so ``OLLAMA_BASE_URL`` governed the ``/api/ps`` probe and
``bamboo verify`` but *not* inference. In the batch container (Ollama on a random free
port) every ``ainvoke`` died with "All connection attempts failed" while ``bamboo verify``
reported a healthy server.

The assertions check the resolved httpx client host, not just the field, because the
field being ``None`` is exactly what the defaulting hid.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bamboo.llm.llm_client as llm

pytest.importorskip("langchain_ollama")

_BATCH_URL = "http://127.0.0.1:49221"


def _settings(**kw):
    base = {
        "llm_provider": "ollama",
        "llm_model": "qwen3.6:latest",
        "ollama_base_url": _BATCH_URL,
        "ollama_reasoning": True,
        "llm_api_key": "",
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _clear_llm_caches():
    """The public factories are lru_cached, so a model built by another test (or by
    import-time code) would otherwise leak across cases."""
    for factory in (llm.get_llm, llm.get_summary_llm, llm.get_extraction_llm):
        factory.cache_clear()
    yield
    for factory in (llm.get_llm, llm.get_summary_llm, llm.get_extraction_llm):
        factory.cache_clear()


def _client_host(model) -> str:
    return str(model._async_client._client.base_url)


def test_ollama_base_url_reaches_the_client(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    model = llm._build_llm(temperature=0.0)
    assert model.base_url == _BATCH_URL
    assert _client_host(model) == _BATCH_URL


def test_ollama_base_url_wins_over_ollama_host_env(monkeypatch):
    """OLLAMA_BASE_URL is bamboo's documented knob and must be authoritative; a stray
    OLLAMA_HOST (set by the batch container for the `ollama` CLI) must not win."""
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9999")
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    assert _client_host(llm._build_llm(temperature=0.0)) == _BATCH_URL


def test_blank_base_url_falls_back_to_library_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(ollama_base_url=""))
    model = llm._build_llm(temperature=0.0)
    assert model.base_url is None
    assert _client_host(model) == "http://127.0.0.1:11434"


def test_extraction_llm_uses_the_configured_base_url(monkeypatch):
    """The two call sites that surfaced the bug (context_prefetch, panda_source_navigator)
    both go through get_extraction_llm."""
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    model = llm.get_extraction_llm()
    assert _client_host(model) == _BATCH_URL
    assert model.temperature == 0.0
