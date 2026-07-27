"""Unit tests for :mod:`bamboo.llm.errors`.

The bug these lock in: a dead LLM endpoint surfaced as
``WARNING - prefetch_panda_docs: query extraction failed (All connection attempts
failed)`` — anyio's message for "every resolved address refused", naming no host,
no port, no provider, and logged as if it were a minor degradation. Diagnosing it
required reading anyio, httpx and langchain-ollama.

So the tests assert two things the old code got wrong: the message must name the
endpoint, and a connection failure must be ERROR rather than WARNING.
"""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace

import httpx
import pytest

import bamboo.llm.errors as errors

_BATCH_URL = "http://127.0.0.1:49221"


def _settings(**kw):
    base = {
        "llm_provider": "ollama",
        "llm_model": "qwen3.6:latest",
        "ollama_base_url": _BATCH_URL,
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def ollama_settings(monkeypatch):
    monkeypatch.setattr(errors, "get_settings", lambda: _settings())


def _anyio_style_error() -> OSError:
    """The exact exception anyio raises from ``connect_tcp`` — a plain OSError."""
    return OSError("All connection attempts failed")


def _wrapped(depth: int) -> Exception:
    """An anyio OSError re-wrapped ``depth`` times, as the provider SDKs do."""
    exc: BaseException = _anyio_style_error()
    for i in range(depth):
        try:
            raise (httpx.ConnectError("connect failed") if i == 0 else RuntimeError("sdk wrapper")) from exc
        except Exception as raised:  # noqa: BLE001
            exc = raised
    return exc  # type: ignore[return-value]


def test_is_llm_connection_error_is_the_generic_predicate():
    """The LLM-facing spelling of the shared detector — nothing about walking the
    __cause__ chain is LLM-specific, so it is tested once in test_endpoint_errors."""
    from bamboo.utils.errors import is_connection_error

    assert errors.is_llm_connection_error is is_connection_error


# --------------------------------------------------------------------------- #
# describe_llm_failure
# --------------------------------------------------------------------------- #


def test_message_names_provider_model_endpoint_and_hint(ollama_settings):
    msg = errors.describe_llm_failure(
        "prefetch_panda_docs: query extraction failed",
        _wrapped(2),
        fallback="falling back to raw errorDialog",
    )
    assert "prefetch_panda_docs: query extraction failed" in msg
    assert "provider=ollama" in msg
    assert "model=qwen3.6:latest" in msg
    assert f"endpoint={_BATCH_URL}" in msg
    assert "ollama serve" in msg
    assert "OLLAMA_BASE_URL" in msg
    assert "falling back to raw errorDialog" in msg
    assert "\n" not in msg  # one line, so it survives grep and a chat frontend


def test_no_hint_for_a_content_error(ollama_settings):
    msg = errors.describe_llm_failure("parse failed", ValueError("bad json"))
    assert "hint:" not in msg
    assert f"endpoint={_BATCH_URL}" in msg  # context is still useful


def test_blank_ollama_base_url_reports_the_library_default(monkeypatch):
    """The original bug exactly: base_url unset ⇒ langchain silently dials 11434."""
    monkeypatch.setattr(errors, "get_settings", lambda: _settings(ollama_base_url=""))
    msg = errors.describe_llm_failure("x", _anyio_style_error())
    assert "http://127.0.0.1:11434" in msg
    assert "library default" in msg


def test_cloud_provider_hint_mentions_proxy(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(
        errors, "get_settings", lambda: _settings(llm_provider="openai", llm_model="gpt-4o")
    )
    msg = errors.describe_llm_failure("x", _anyio_style_error())
    assert "https://api.openai.com/v1" in msg
    assert "HTTPS_PROXY" in msg
    assert "ollama serve" not in msg


def test_openai_base_url_env_override_is_reported(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.internal/v1")
    monkeypatch.setattr(errors, "get_settings", lambda: _settings(llm_provider="openai"))
    assert "http://gateway.internal/v1" in errors.describe_llm_failure("x", _anyio_style_error())


def test_never_raises_when_settings_are_broken(monkeypatch):
    """It runs inside except blocks — a config failure must not shadow the real error."""

    def _boom():
        raise RuntimeError("no .env")

    monkeypatch.setattr(errors, "get_settings", _boom)
    msg = errors.describe_llm_failure("ctx", ValueError("original problem"))
    assert "ctx" in msg
    assert "original problem" in msg


# --------------------------------------------------------------------------- #
# log_llm_failure — the severity rule
# --------------------------------------------------------------------------- #


def test_connection_failure_logs_at_error(ollama_settings, caplog):
    logger = logging.getLogger("bamboo.test.llm_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_llm_failure(logger, "ctx", _wrapped(2))
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert _BATCH_URL in caplog.records[0].getMessage()


def test_content_failure_stays_at_warning(ollama_settings, caplog):
    logger = logging.getLogger("bamboo.test.llm_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_llm_failure(logger, "ctx", ValueError("bad json"))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_exc_info_captures_the_llm_exception_not_the_ambient_one(ollama_settings, caplog):
    """Called outside any ``except`` block, so a bare ``exc_info=True`` would record
    ``(None, None, None)``. The traceback must come from the exception we passed."""
    logger = logging.getLogger("bamboo.test.llm_errors")
    exc = ValueError("x")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_llm_failure(logger, "ctx", exc, exc_info=True)
    assert caplog.records[0].exc_info[1] is exc


def test_exc_info_defaults_off(ollama_settings, caplog):
    logger = logging.getLogger("bamboo.test.llm_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_llm_failure(logger, "ctx", ValueError("x"))
    assert caplog.records[0].exc_info is None


# --------------------------------------------------------------------------- #
# Display safety — investigation_session pushes this string into Rich markup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc",
    [
        OSError("[Errno 61] Connection refused"),
        FileNotFoundError("No such file: [/tmp/socket]"),
    ],
)
def test_diagnostic_survives_rich_markup_when_escaped(ollama_settings, exc):
    """``InvestigationSession`` embeds this in ``[red]…[/red]`` for ``io.notice``.

    Un-escaped, Rich eats ``[Errno 61]`` as an unknown tag (the text silently
    vanishes) and ``[/tmp/socket]`` raises ``MarkupError`` outright, killing the
    turn with a traceback instead of showing the diagnostic.
    """
    from rich.console import Console
    from rich.markup import escape

    detail = errors.describe_llm_failure("Orchestration LLM call failed", exc)
    console = Console(file=io.StringIO(), width=200, no_color=True)
    console.print(f"[red]{escape(detail)}[/red]")  # must not raise
    rendered = console.file.getvalue()

    assert str(exc) in rendered  # bracketed text preserved, not swallowed
    assert f"endpoint={_BATCH_URL}" in rendered
