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


def test_ca_cert_count_falls_back_to_default_paths(monkeypatch):
    """An empty ``get_ca_certs()`` plus a populated default cafile is NOT "empty".

    Reproduces the Linux/Debian container case: the default context loads CAs lazily via
    a CApath/cafile, so ``get_ca_certs()`` returns ``[]`` even though the trust store works.
    ``_ca_cert_count`` must probe the on-disk default paths and report a positive count.
    """
    import ssl
    import types

    import certifi

    # Default context reports nothing eagerly loaded …
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        lambda *a, **k: types.SimpleNamespace(get_ca_certs=lambda: []),
    )
    # … but a real, non-empty default cafile is present on disk.
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: types.SimpleNamespace(
            cafile=None,
            capath=None,
            openssl_cafile=certifi.where(),
            openssl_capath=None,
        ),
    )

    assert verify._ca_cert_count() > 0


def test_empty_store_unwritable_env_does_not_crash(monkeypatch, tmp_path):
    """A repair on a read-only / bind-mounted .env must not crash verify.

    A single-file Docker bind mount can't be replaced by dotenv's temp-file + os.replace
    (OSError EBUSY); the check must apply the fix in-process and degrade gracefully.
    """
    import errno
    import os

    # 0 roots on the first probe, populated after the in-process fix.
    counts = iter([0, 137])
    monkeypatch.setattr(verify, "_ca_cert_count", lambda: next(counts))

    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\n")
    monkeypatch.setattr("bamboo.config._find_env_file", lambda: str(env))

    def _boom(*a, **k):
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr("dotenv.set_key", _boom)
    # Throwaway os.environ so the SSL_CERT_FILE the check sets does not leak to other tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))

    # Must not raise, and still reports success — the in-process fix stands.
    assert verify.check_tls_trust_store() is True
    assert os.environ.get("SSL_CERT_FILE")  # applied for this process
    assert "SSL_CERT_FILE" not in env.read_text()  # persist failed → .env untouched
