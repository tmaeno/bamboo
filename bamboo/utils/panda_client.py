"""Utilities for fetching data from the PanDA server.

PanDA server configuration
--------------------------
``HttpClient`` reads the server URL from the environment:

* ``PANDA_API_URL_SSL`` – HTTPS API base URL (default: ``https://pandaserver.cern.ch:25443/api/v1``)
* ``PANDA_AUTH``        – set to ``oidc`` to use OIDC token auth instead of X.509
* ``PANDA_AUTH_VO``     – VO name when using OIDC
* ``PANDA_AUTH_ID_TOKEN`` – OIDC token (or ``file:<path>``)
* ``X509_USER_PROXY``   – path to X.509 proxy certificate

Set these variables (or place them in your ``.env`` file) to point at a
different PanDA instance (e.g. a development server).
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import re
from typing import Any
from bamboo.utils.narrator import thinking

logger = logging.getLogger(__name__)

# Matches <a href="URL">...</a> in HTML fragments (e.g. errorDialog values).
_HREF_RE = re.compile(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>', re.IGNORECASE)


def extract_log_urls(text: str) -> list[str]:
    """Return all href URLs found in HTML anchor tags within *text*."""
    return _HREF_RE.findall(text)


def fetch_log_url(url: str, timeout: float = 30.0) -> str | None:
    """Fetch a plain-text log file from *url* synchronously.

    Returns the content as a string, or ``None`` on any error.
    Used by ``bamboo fetch-task --verbose`` to display linked log files.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("fetch_log_url: httpx not installed — cannot fetch %s", url)
        return None
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        if response.status_code != 200:
            logger.warning(
                "fetch_log_url: %s returned HTTP %s", url, response.status_code
            )
            return None
        return response.text
    except Exception as exc:
        logger.warning("fetch_log_url: failed to fetch %s: %s", url, exc)
        return None


async def async_fetch_log_content(url: str, timeout: float = 30.0) -> str | None:
    """Async version of :func:`fetch_log_url` — used inside async extractors.

    Returns the response body as a string, or ``None`` on any error
    (network failure, non-200 status, non-text content-type).
    Failures are logged at WARNING level and never interrupt extraction.
    """
    try:
        import httpx
    except ImportError:
        logger.warning(
            "async_fetch_log_content: httpx not installed — cannot fetch %s", url
        )
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code != 200:
            logger.warning(
                "async_fetch_log_content: %s returned HTTP %s — skipping",
                url,
                response.status_code,
            )
            return None
        content_type = response.headers.get("content-type", "")
        content_encoding = response.headers.get("content-encoding", "")
        # httpx transparently decompresses Content-Encoding: gzip, so
        # response.content/.text is already plain text in that case.
        # Only manually decompress when the payload itself is a raw gzip
        # blob (indicated by Content-Type or a .gz URL suffix).
        needs_manual_decompress = (
            "application/gzip" in content_type or url.endswith(".gz")
        ) and "gzip" not in content_encoding
        if needs_manual_decompress:
            try:
                text = gzip.decompress(response.content).decode(
                    "utf-8", errors="replace"
                )
            except Exception as exc:
                logger.warning(
                    "async_fetch_log_content: failed to decompress gzip from %s: %s",
                    url,
                    exc,
                )
                return None
        else:
            text = response.text
        logger.info("async_fetch_log_content: fetched %d chars from %s", len(text), url)
        return text
    except Exception as exc:
        logger.warning("async_fetch_log_content: failed to fetch %s: %s", url, exc)
        return None


def _call_api(method: str, endpoint: str, data: dict) -> Any:
    """Call a PanDA REST API endpoint and return the ``data`` field of the response.

    Raises ``RuntimeError`` on connection failure or a non-success server response.
    This is a blocking function; wrap with ``asyncio.to_thread`` in async contexts.
    """
    from pandaserver.api.v1.http_client import HttpClient, api_url_ssl  # noqa: PLC0415

    client = HttpClient()
    url = f"{api_url_ssl}/{endpoint}"
    if method == "get":
        status, response = client.get(url, data)
    else:
        status, response = client.post(url, data)

    if status != 0:
        raise RuntimeError(
            f"PanDA connection error for {endpoint!r}: {response}. "
            "Check PANDA_API_URL_SSL."
        )
    if not isinstance(response, dict) or not response.get("success"):
        message = response.get("message", "unknown error") if isinstance(response, dict) else str(response)
        raise RuntimeError(
            f"PanDA returned error for {endpoint!r}: {message}"
        )
    return response.get("data")


async def fetch_task_data(task_id: int | str, verbose: bool = False) -> dict[str, Any]:
    """Fetch full task details from PanDA and return them as a ``dict``.

    The blocking ``HttpClient`` call is offloaded to a thread pool so
    the event loop stays responsive.

    Args:
        task_id: The PanDA ``jediTaskID`` to look up (int or numeric string).
        verbose: If ``True``, sets the bamboo logger to DEBUG level, which
            is useful for diagnosing network or authentication issues.

    Returns:
        A ``dict`` containing the task details as returned by the PanDA
        server, ready to be passed as ``task_data`` to
        :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`
        or any extraction strategy.

    Raises:
        RuntimeError: If the PanDA server returns a non-zero status code or
                      an error response.
        ValueError:   If *task_id* cannot be converted to ``int``.
    """
    task_id_int = int(task_id)

    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    logger.info("Fetching task details from PanDA for task_id=%s", task_id_int)

    with thinking("Working"):
        try:
            data = await asyncio.to_thread(
                _call_api, "get", "task/get_detailed_info", {"task_id": task_id_int}
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"PanDA returned error for task_id={task_id_int}: {exc}. "
                "Check PANDA_API_URL_SSL."
            ) from exc

    logger.info(
        "Successfully fetched task details for task_id=%s (%d fields)",
        task_id_int,
        len(data),
    )
    return data
