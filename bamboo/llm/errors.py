"""Endpoint-aware diagnostics for failed LLM calls.

The LLM-specific layer over :mod:`bamboo.utils.errors`: it resolves the provider,
model and *effective endpoint* that a failed call was dialled at, so a bare
``All connection attempts failed`` becomes something a reader can act on.

Call idiom, matching the rest of the package::

    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        log_llm_failure(logger, "prefetch_panda_docs: query extraction failed", exc,
                        fallback="falling back to raw errorDialog")

Kept free of the langchain provider imports (unlike ``llm_client``) so it is cheap
to import from an ``except`` block anywhere in the package.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from bamboo.config import get_settings
from bamboo.utils.errors import format_diagnostic, is_connection_error, log_diagnostic

# What langchain-ollama falls back to when ``base_url`` is unset — see the comment
# in ``llm_client._build_llm``.
_OLLAMA_LIBRARY_DEFAULT = "http://127.0.0.1:11434"
_DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}

# ``is_llm_connection_error`` is retained as the LLM-facing spelling of the generic
# predicate; there is nothing LLM-specific about the detection itself.
is_llm_connection_error = is_connection_error


def llm_endpoint(settings: Any = None) -> str:
    """Return the URL the configured provider will actually be dialled at.

    For ``openai``/``anthropic`` the SDKs read their base URL from the environment
    (bamboo has no setting for it), so those are resolved from the same env vars the
    SDK consults rather than guessed.
    """
    settings = settings or get_settings()
    provider = getattr(settings, "llm_provider", "") or ""
    if provider == "ollama":
        base = getattr(settings, "ollama_base_url", "") or ""
        return base or f"{_OLLAMA_LIBRARY_DEFAULT} (library default)"
    if provider == "openai":
        return (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or _DEFAULT_ENDPOINTS["openai"]
        )
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_BASE_URL") or _DEFAULT_ENDPOINTS["anthropic"]
    return "(unknown provider)"


def _connection_hint(provider: str, endpoint: str, model: str) -> str:
    """Actionable next step for a connection failure, per provider.

    Mirrors the wording ``bamboo verify`` already prints for the same conditions.
    """
    if provider == "ollama":
        return (
            f"no Ollama server at {endpoint} — start it with `ollama serve` "
            f"(and `ollama pull {model}`), or point OLLAMA_BASE_URL at the right host:port"
        )
    return (
        f"cannot reach {endpoint} — check network/proxy (HTTPS_PROXY / NO_PROXY) "
        "and firewall rules"
    )


def describe_llm_failure(
    context: str, exc: BaseException, *, fallback: str = ""
) -> str:
    """Build a one-line diagnostic for a failed LLM call.

    ``context`` says which call failed (``"prefetch_panda_docs: query extraction failed"``);
    ``fallback`` describes what the caller does instead. The provider, model and
    effective endpoint are always included, so the reader never has to guess which
    service was unreachable, and a ``hint:`` segment is appended for connection
    failures only.

    Never raises — it runs inside ``except`` blocks, so a failure to load settings
    degrades to the bare exception text rather than masking the original error.
    """
    target = hint = ""
    try:
        settings = get_settings()
        provider = getattr(settings, "llm_provider", "") or "?"
        model = getattr(settings, "llm_model", "") or "?"
        endpoint = llm_endpoint(settings)
        target = f"provider={provider} model={model} endpoint={endpoint}"
        if is_connection_error(exc):
            hint = _connection_hint(provider, endpoint, model)
    except Exception:  # noqa: BLE001 — diagnostics must never shadow the real error
        pass
    return format_diagnostic(context, exc, target=target, hint=hint, fallback=fallback)


def log_llm_failure(
    logger: logging.Logger,
    context: str,
    exc: BaseException,
    *,
    fallback: str = "",
    exc_info: bool = False,
) -> None:
    """Log a failed LLM call at the severity its cause deserves.

    Connection failures are ERROR (the endpoint is unreachable, so the run is
    doomed); content-level failures such as bad JSON stay at WARNING, which is what
    the surrounding best-effort fallbacks are designed for. See
    :func:`bamboo.utils.errors.log_diagnostic`.
    """
    log_diagnostic(
        logger,
        describe_llm_failure(context, exc, fallback=fallback),
        exc,
        exc_info=exc_info,
    )
