"""Unit tests for the TLS-trust-store check in ``bamboo verify``.

The real ``ssl`` trust store is never touched: ``_ca_cert_count`` is
monkeypatched to simulate an empty / populated store. The check is a pure
diagnostic — it must never mutate ``os.environ`` or write to ``.env`` (that
behaviour previously leaked a host-specific ``SSL_CERT_FILE`` into containers).
"""

from __future__ import annotations

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


def test_empty_store_is_ok_without_mutation(monkeypatch, tmp_path):
    """An empty stdlib store is no longer fatal and must not mutate env or .env.

    pandaclient's OIDC calls and every httpx client fall back to the bundled
    certifi roots, so the check only reports: it must not set ``SSL_CERT_FILE``
    (which used to leak a host path into containers) nor write to ``.env``.
    """
    import os

    monkeypatch.setattr(verify, "_ca_cert_count", lambda: 0)
    env = tmp_path / ".env"
    env.write_text("LLM_API_KEY=x\n")
    monkeypatch.setattr("bamboo.config._find_env_file", lambda: str(env))
    # Isolate os.environ so an accidental mutation can't leak to other tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    before = dict(os.environ)

    assert verify.check_tls_trust_store() is True
    assert os.environ == before  # no environment mutation
    assert "SSL_CERT_FILE" not in env.read_text()  # no .env write


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
