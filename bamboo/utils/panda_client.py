"""Synchronous wrapper around ``pandaclient.Client.get_task_details_json``.

This module provides :func:`fetch_task_data` — a helper that retrieves full
task details from a live PanDA server and returns them as a plain Python
``dict`` ready for use as ``task_data`` in the Bamboo pipeline.

PanDA server configuration
--------------------------
The underlying ``panda-client-light`` library reads the server URL from the
environment:

* ``PANDA_URL``      – plain HTTP base URL  (default: ``http://pandaserver.cern.ch:25080``)
* ``PANDA_URL_SSL``  – HTTPS base URL       (default: ``https://pandaserver.cern.ch``)

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

# Status code returned by panda-client-light on success.
_PANDA_OK = 0

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
        is_gzip = (
            "gzip" in content_encoding
            or "application/gzip" in content_type
            or url.endswith(".gz")
        )
        if is_gzip:
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
        elif "text" not in content_type and "json" not in content_type:
            logger.warning(
                "async_fetch_log_content: %s has non-text content-type %r — skipping",
                url,
                content_type,
            )
            return None
        else:
            text = response.text
        logger.info("async_fetch_log_content: fetched %d chars from %s", len(text), url)
        return text
    except Exception as exc:
        logger.warning("async_fetch_log_content: failed to fetch %s: %s", url, exc)
        return None


async def fetch_task_data(task_id: int | str, verbose: bool = False) -> dict[str, Any]:
    """Fetch full task details from PanDA and return them as a ``dict``.

    The blocking ``panda-client-light`` call is offloaded to a thread pool so
    the event loop stays responsive.

    Args:
        task_id: The PanDA ``jediTaskID`` to look up (int or numeric string).
        verbose: If ``True``, the underlying ``panda-client-light`` library
            will print every curl command it constructs and the raw server
            response to stdout, which is useful for diagnosing network or
            authentication issues.

    Returns:
        A ``dict`` containing the task details as returned by the PanDA
        server, ready to be passed as ``task_data`` to
        :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`
        or any extraction strategy.

    Raises:
        ImportError:  If ``panda-client-light`` is not installed.
        RuntimeError: If the PanDA server returns a non-zero status code or
                      an error response.
        ValueError:   If *task_id* cannot be converted to ``int``.2
    """
    try:
        from pandaclient import Client  # noqa: PLC0415  (conditional import)
    except ImportError as exc:
        raise ImportError(
            "panda-client-light is required to fetch task data from PanDA. "
            "Install it with: pip install panda-client-light"
        ) from exc

    task_id_int = int(task_id)

    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    logger.info("Fetching task details from PanDA for task_id=%s", task_id_int)

    with thinking("Working"):
        status, data = await asyncio.to_thread(
            Client.get_task_details_json, task_id_int, verbose=verbose
        )

    if status != _PANDA_OK:
        raise RuntimeError(
            f"PanDA returned status={status} for task_id={task_id_int}. "
            "Check that the task ID is valid and the PanDA server is reachable "
            "(PANDA_URL / PANDA_URL_SSL environment variables)."
        )

    if not isinstance(data, tuple):
        raise RuntimeError(
            f"Unexpected response type from PanDA: {type(data).__name__!r}. "
            f"Expected tuple, got: {data!r}"
        )

    if not data[0]:
        raise RuntimeError(
            f"PanDA gave the following error message for task_id={task_id_int}: {data[-1]}."
        )

    data = data[1]

    logger.info(
        "Successfully fetched task details for task_id=%s (%d fields)",
        task_id_int,
        len(data),
    )
    return data
