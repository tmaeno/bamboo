"""Unit tests for the TLS-trust-store check in ``bamboo verify``.

The real ``ssl`` trust store is never touched: ``_ca_cert_count`` is
monkeypatched to simulate an empty / populated store, and ``_find_env_file`` is
pointed at a temp file so the certifi-install path writes there.
"""

from __future__ import annotations

import sys

import pytest

from bamboo.scripts import verify


def test_trust_store_ok_does_not_write_env(monkeypatch, tmp_path):
    """A populated trust store passes and never touches the .env."""
    monkeypatch.setattr(verify, "_ca_cert_count", lambda: 137)
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\n")
    monkeypatch.setattr("bamboo.config._find_env_file", lambda: str(env))

    assert verify.check_tls_trust_store() is True
    # Untouched — no SSL_CERT_FILE injected.
    assert "SSL_CERT_FILE" not in env.read_text()


def test_empty_store_installs_certifi_into_env(monkeypatch, tmp_path):
    """An empty store is repaired by writing SSL_CERT_FILE to the active .env."""
    # 0 roots on the first probe, then populated after the install.
    counts = iter([0, 137])
    monkeypatch.setattr(verify, "_ca_cert_count", lambda: next(counts))

    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\n")
    monkeypatch.setattr("bamboo.config._find_env_file", lambda: str(env))

    assert verify.check_tls_trust_store() is True

    import certifi

    text = env.read_text()
    assert "SSL_CERT_FILE" in text
    assert certifi.where() in text
    assert "SSL_CERT_DIR" in text


def test_empty_store_without_certifi_fails(monkeypatch, tmp_path):
    """No trust store and no certifi → reported as a failure."""
    monkeypatch.setattr(verify, "_ca_cert_count", lambda: 0)
    # Force `import certifi` inside the check to raise ImportError.
    monkeypatch.setitem(sys.modules, "certifi", None)

    assert verify.check_tls_trust_store() is False


def test_empty_store_without_env_fails(monkeypatch, tmp_path):
    """No trust store and no .env to persist the fix → failure with guidance."""
    monkeypatch.setattr(verify, "_ca_cert_count", lambda: 0)
    monkeypatch.setattr("bamboo.config._find_env_file", lambda: None)

    assert verify.check_tls_trust_store() is False
