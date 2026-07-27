"""Unit tests for the LLM checks in ``bamboo verify``.

Two false greens motivated these. The Ollama check used to ``GET /`` on the server
root, which proves only that *something* answers — not that ``LLM_MODEL`` was ever
pulled, and not that the client bamboo actually builds dials the same place. And no
check ever generated a token, so an invalid API key or an unloadable model passed
verification and failed at the first real call.

No network is touched: ``urllib.request.urlopen`` and ``get_extraction_llm`` are
monkeypatched, following ``tests/test_verify_tls.py`` (module-object import,
``monkeypatch`` only, assert on the returned bool).
"""

from __future__ import annotations

import io
import json
import urllib.request
from types import SimpleNamespace

import httpx
import pytest

from bamboo.scripts import verify

_BASE = "http://127.0.0.1:49221"


def _settings(**kw):
    base = {
        "llm_provider": "ollama",
        "llm_model": "qwen3.6:latest",
        "ollama_base_url": _BASE,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class _Resp(io.BytesIO):
    """Minimal stand-in for the urlopen context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serving(*model_names):
    def _urlopen(url, timeout=None):
        assert url == f"{_BASE}/api/tags", f"unexpected probe URL: {url}"
        payload = {"models": [{"model": n, "name": n} for n in model_names]}
        return _Resp(json.dumps(payload).encode())

    return _urlopen


# --------------------------------------------------------------------------- #
# _check_ollama_model
# --------------------------------------------------------------------------- #


def test_pulled_model_passes(monkeypatch, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", _serving("qwen3.6:latest", "llama3:8b"))
    assert verify._check_ollama_model(_settings()) is True
    assert "is pulled" in capsys.readouterr().out


def test_model_match_tolerates_an_implicit_latest_tag(monkeypatch):
    """`LLM_MODEL=qwen3.6` and a served `qwen3.6:latest` are the same model — the
    check reuses llm_client._ollama_model_matches so it can't disagree with the
    context-window probe."""
    monkeypatch.setattr(urllib.request, "urlopen", _serving("qwen3.6:latest"))
    assert verify._check_ollama_model(_settings(llm_model="qwen3.6")) is True


def test_server_up_but_model_not_pulled_fails(monkeypatch, capsys):
    """The false green this check exists for: `GET /` would have passed here."""
    monkeypatch.setattr(urllib.request, "urlopen", _serving("llama3:8b"))
    assert verify._check_ollama_model(_settings()) is False
    out = capsys.readouterr().out
    assert "is not pulled" in out
    assert "ollama pull qwen3.6:latest" in out
    assert "llama3:8b" in out  # tells the operator what *is* available


def test_unreachable_server_names_the_endpoint(monkeypatch, capsys):
    def _refused(url, timeout=None):
        raise OSError("All connection attempts failed")

    monkeypatch.setattr(urllib.request, "urlopen", _refused)
    assert verify._check_ollama_model(_settings()) is False
    out = capsys.readouterr().out
    assert _BASE in out
    assert "OLLAMA_BASE_URL" in out


def test_blank_base_url_falls_back_to_the_documented_default(monkeypatch):
    seen = {}

    def _urlopen(url, timeout=None):
        seen["url"] = url
        return _Resp(json.dumps({"models": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    verify._check_ollama_model(_settings(ollama_base_url=""))
    assert seen["url"] == "http://localhost:11434/api/tags"


# --------------------------------------------------------------------------- #
# check_llm_roundtrip
# --------------------------------------------------------------------------- #


@pytest.fixture
def _ollama(monkeypatch):
    monkeypatch.setattr("bamboo.config.get_settings", lambda: _settings())
    monkeypatch.setattr("bamboo.llm.errors.get_settings", lambda: _settings())


def _stub_llm(monkeypatch, result):
    """Install a get_extraction_llm whose invoke returns *result* or raises it."""

    def _invoke(messages):
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(content=result)

    monkeypatch.setattr(
        "bamboo.llm.get_extraction_llm", lambda: SimpleNamespace(invoke=_invoke)
    )


def test_roundtrip_ok(monkeypatch, capsys, _ollama):
    _stub_llm(monkeypatch, "ok")
    assert verify.check_llm_roundtrip() is True
    assert "round-trip OK" in capsys.readouterr().out


def test_roundtrip_reports_the_endpoint_on_a_connection_failure(monkeypatch, capsys, _ollama):
    _stub_llm(monkeypatch, httpx.ConnectError("All connection attempts failed"))
    assert verify.check_llm_roundtrip() is False
    out = capsys.readouterr().out
    assert "LLM round-trip failed" in out
    assert f"endpoint={_BASE}" in out
    assert "ollama serve" in out


def test_empty_response_is_a_failure(monkeypatch, capsys, _ollama):
    """Reachable but producing nothing is not a pass — that was the old bar."""
    _stub_llm(monkeypatch, "   ")
    assert verify.check_llm_roundtrip() is False
    assert "empty response" in capsys.readouterr().out
