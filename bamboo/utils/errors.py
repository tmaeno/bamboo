"""Endpoint-aware diagnostics for failed calls to an external service.

Transport failures are unreadable on their own. anyio raises a bare
``OSError("All connection attempts failed")`` when every resolved address for a
host:port refuses, and that string reaches the log naming *no* host, port or
service — so a dead Ollama on 127.0.0.1:11434, an unreachable Qdrant and a blocked
api.github.com all produce byte-identical lines, and the reader has no idea which
knob to turn.

This module supplies the missing context for any service bamboo dials::

    try:
        meta = await client.retrieve(...)
    except Exception as exc:
        log_endpoint_failure(
            logger, "PandaDocNavigator: failed to read index meta", exc,
            service="qdrant", endpoint=settings.qdrant_url,
            fallback="treating the doc index as absent",
        )

:mod:`bamboo.llm.errors` layers the LLM-specific provider/model resolution on top
of the same primitives.
"""

from __future__ import annotations

import logging
import socket
from typing import Optional

import httpx

# Both families are needed. ``httpx.ConnectError`` derives from ``httpx.HTTPError``,
# *not* from ``OSError``; anyio's "All connection attempts failed" is a plain
# ``OSError``. ``ConnectionError`` and ``socket.gaierror`` are OSError subclasses,
# listed for documentation value.
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    ConnectionError,
    socket.gaierror,
    OSError,
)

# Default next step per service, used when the call site doesn't pass its own.
# Wording mirrors what ``bamboo verify`` prints for the same conditions.
_SERVICE_HINTS = {
    "qdrant": "start it with `docker compose up -d`, or point QDRANT_URL at the right host:port",
    "neo4j": "start the Neo4j server, or point NEO4J_URI at the right host:port",
    "github": "check network/proxy (HTTPS_PROXY / NO_PROXY) and firewall rules",
    "readthedocs": "check network/proxy (HTTPS_PROXY / NO_PROXY) and firewall rules",
}
_GENERIC_HINT = "check the endpoint, network/proxy (HTTPS_PROXY / NO_PROXY) and firewall rules"


def walk_causes(exc: BaseException, limit: int = 10):
    """Yield ``exc`` and its ``__cause__``/``__context__`` ancestors.

    Client libraries re-wrap transport errors (httpx → httpcore → the SDK), so the
    connection failure is usually several links down the chain. ``limit`` and the
    identity set guard against a cyclic chain.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    for _ in range(limit):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_connection_error(exc: BaseException) -> bool:
    """True when ``exc`` (or anything it wraps) is a failure to reach the endpoint.

    Distinguishes "the service is unreachable" — which no amount of retrying will
    fix, and which usually means the whole run is doomed — from a call that reached
    the service and came back with something unusable.
    """
    return any(isinstance(item, _CONNECTION_ERRORS) for item in walk_causes(exc))


def format_diagnostic(
    context: str,
    exc: BaseException,
    *,
    target: str = "",
    hint: str = "",
    fallback: str = "",
) -> str:
    """Assemble a one-line ``a | b | c`` diagnostic.

    Shared by :func:`describe_endpoint_failure` and
    :func:`bamboo.llm.errors.describe_llm_failure`, which differ only in how they
    render the ``target`` segment.
    """
    bits = [context]
    if target:
        bits.append(target)
    # Collapse whitespace in the exception text: neo4j's ServiceUnavailable (among
    # others) embeds newlines, which would split the record across log lines and
    # defeat grepping for it.
    bits.append(" ".join(f"{type(exc).__name__}: {exc}".split()))
    if hint:
        bits.append(f"hint: {hint}")
    if fallback:
        bits.append(fallback)
    return " | ".join(bits)


def describe_endpoint_failure(
    context: str,
    exc: BaseException,
    *,
    service: str,
    endpoint: str,
    hint: str = "",
    fallback: str = "",
) -> str:
    """Build a one-line diagnostic naming the service and URL that failed.

    ``hint`` overrides the per-service default and — like the default — is rendered
    only for connection failures, where "where do I point this" is the actual
    question. A 404 or a bad payload needs no such advice.
    """
    if is_connection_error(exc):
        hint = hint or _SERVICE_HINTS.get(service, _GENERIC_HINT)
    else:
        hint = ""
    return format_diagnostic(
        context, exc, target=f"{service}={endpoint}", hint=hint, fallback=fallback
    )


def log_diagnostic(
    logger: logging.Logger,
    message: str,
    exc: BaseException,
    *,
    exc_info: bool = False,
) -> None:
    """Log ``message`` at the severity ``exc``'s cause deserves.

    Connection failures are logged at ERROR: the service is unreachable, so every
    later call will fail the same way and whatever the run produces is worthless.
    Anything else stays at WARNING, which is what the surrounding best-effort
    fallbacks are designed for.

    This only changes what is *reported*; no caller's control flow changes.
    """
    level = logging.ERROR if is_connection_error(exc) else logging.WARNING
    logger.log(
        level,
        "%s",
        message,
        # Pass the exception itself rather than ``True``: the traceback is then the
        # failure's own, not whatever happens to be the active exception.
        exc_info=exc if exc_info else None,
    )


def log_endpoint_failure(
    logger: logging.Logger,
    context: str,
    exc: BaseException,
    *,
    service: str,
    endpoint: str,
    hint: str = "",
    fallback: str = "",
    exc_info: bool = False,
) -> None:
    """Format and log a failed call to an external service. See the module docstring."""
    log_diagnostic(
        logger,
        describe_endpoint_failure(
            context, exc, service=service, endpoint=endpoint, hint=hint, fallback=fallback
        ),
        exc,
        exc_info=exc_info,
    )
