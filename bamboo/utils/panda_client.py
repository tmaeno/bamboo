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
import contextlib
import contextvars
import gzip
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from bamboo.utils.narrator import say, thinking

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore', category=InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-user PanDA identity (OIDC)
# ---------------------------------------------------------------------------
#
# By default ``HttpClient`` authenticates from process-global environment
# variables (PANDA_AUTH/PANDA_AUTH_ID_TOKEN/...). To let a multi-user frontend
# (e.g. the Mattermost bot) act as the *invoking* user, a per-call OIDC token can
# be bound for the duration of a request via :func:`panda_credentials`. It is
# stored in a ContextVar, which propagates across ``asyncio.to_thread`` (the
# context is copied), so the blocking ``HttpClient`` call running in a worker
# thread still sees the right token. When unset, behavior is unchanged (the
# client reads the environment as before).


@dataclass(frozen=True)
class PandaCredentials:
    """An OIDC token (and VO) to act as a specific PanDA user."""

    id_token: str
    auth_vo: str = ""


_panda_credentials: contextvars.ContextVar[Optional[PandaCredentials]] = contextvars.ContextVar(
    "panda_credentials", default=None
)


@contextlib.contextmanager
def panda_credentials(creds: Optional[PandaCredentials]):
    """Bind per-user PanDA OIDC credentials for the duration of the block.

    ``with panda_credentials(creds): await fetch_task_data(...)`` makes every
    PanDA API call inside act as that user. Passing ``None`` is a no-op (keeps
    the process-default identity).
    """
    token = _panda_credentials.set(creds)
    try:
        yield
    finally:
        _panda_credentials.reset(token)


def _apply_credentials(client: Any) -> None:
    """Apply the bound per-user OIDC credentials to *client*, if any.

    Uses ``HttpClient.override_oidc(oidc, id_token, auth_vo)`` to switch identity
    after construction. No-op when no credentials are bound.
    """
    creds = _panda_credentials.get()
    if creds is None:
        return
    override = getattr(client, "override_oidc", None)
    if override is None:  # pragma: no cover - older client without override
        logger.warning("HttpClient has no override_oidc; per-user identity ignored.")
        return
    override(True, creds.id_token, creds.auth_vo)

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
        say(f"fetched {len(text):,} chars from {url}")
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
    _apply_credentials(client)  # per-user OIDC override when bound (else env default)
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

    with thinking("Fetching task data from PanDA"):
        try:
            data = await asyncio.to_thread(
                _call_api, "get", "task/get_detailed_info", {"task_id": task_id_int}
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"PanDA returned error for task_id={task_id_int}: {exc}. "
                "Check PANDA_API_URL_SSL."
            ) from exc

    say(f"fetched task {task_id_int} details ({len(data)} fields)")
    return data


async def get_job_descriptions(
    task_id: int, unsuccessful_only: bool = False
) -> list[dict[str, Any]]:
    """Return job description dicts for *task_id*.

    Args:
        task_id: The PanDA ``jediTaskID``.
        unsuccessful_only: When ``True`` only failed, cancelled, and closed jobs
            are returned (server-side filter).

    Returns:
        List of job dicts.  Empty list on any error.
    """
    try:
        data = await asyncio.to_thread(
            _call_api,
            "get",
            "task/get_job_descriptions",
            {"task_id": task_id, "unsuccessful_only": unsuccessful_only},
        )
        return data if isinstance(data, list) else []
    except RuntimeError as exc:
        logger.warning("get_job_descriptions: failed for task_id=%s: %s", task_id, exc)
        return []


async def get_datasets_and_files(
    task_id: int,
    dataset_types: list[str] | None = None,
    dataset_only: bool = False,
) -> list[dict[str, Any]]:
    """Return dataset and file information for *task_id*.

    Each entry has the form ``{"dataset": {"name": ..., "id": ...}, "files": [...]}``.

    Args:
        task_id: The PanDA ``jediTaskID``.
        dataset_types: Dataset type filter (default: ``["input", "pseudo_input"]``).
        dataset_only: When ``True``, only return dataset information without associated files.

    Returns:
        List of dataset dicts.  Empty list on any error.
    """
    params: dict[str, Any] = {"task_id": task_id}
    if dataset_types:
        params["dataset_types"] = dataset_types
    if dataset_only:
        params["dataset_only"] = True
    try:
        data = await asyncio.to_thread(
            _call_api, "get", "task/get_datasets_and_files", params
        )
        return data if isinstance(data, list) else []
    except RuntimeError as exc:
        logger.warning("get_datasets_and_files: failed for task_id=%s: %s", task_id, exc)
        return []


def _parse_panda_datetime(value: Any) -> datetime | None:
    """Normalize a PanDA modificationTime-like value to a ``datetime``.

    Accepts a ``datetime`` (passthrough), a ``{"__datetime__": "<iso>"}`` wrapper
    dict (the encoding PanDA's wire protocol uses for datetime values when the
    receiving side has not applied ``pandaclient.Client.decode_special_cases``),
    or a string in either ISO 8601 form or the legacy ``YYYY-MM-DD HH:MM:SS``
    form. Returns ``None`` for anything else.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict):
        iso = value.get("__datetime__")
        if isinstance(iso, str):
            try:
                return datetime.fromisoformat(iso)
            except ValueError:
                return None
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


async def get_similar_successful_tasks(
    task_data: dict[str, Any],
    n_tasks: int = 50,
) -> list[dict[str, Any]]:
    """Find recently finished tasks similar to the failing task.

    Uses ``get_tasks_detailed_info_since`` with a ``filters`` dict.  Plain-value
    filters (``userName``, ``processingType``, ``transUses``, ``transHome``,
    ``architecture``) are pushed to SQL.  ``"finished|done"`` is applied via
    ``re.search`` server-side.

    Time window: the ``since`` cutoff is anchored on the failing task's
    ``modificationTime`` (``modificationTime - 14 days``) so the returned
    successful tasks are temporally comparable to the failure (same software
    release, queue conditions, etc.). If ``modificationTime`` is far enough
    in the past that the anchored cutoff would exceed the server's
    30-day-from-now hard limit, or if ``modificationTime`` is
    missing/unparseable, the function falls back to ``now - 30 days`` and
    logs a warning — the result in that case may not be temporally
    comparable.

    Args:
        task_data: The failing task's data dict.
        n_tasks: Maximum number of task IDs to retrieve before filtering.

    Returns:
        List of matching task detail dicts sorted by ``modificationTime`` descending.
        Empty list on any error.
    """
    WINDOW_DAYS = 14
    SERVER_LIMIT_DAYS = 30
    now = datetime.now()
    earliest_allowed = now - timedelta(days=SERVER_LIMIT_DAYS)

    raw_mtime = task_data.get("modificationTime")
    anchor = _parse_panda_datetime(raw_mtime)
    if raw_mtime and anchor is None:
        logger.warning(
            "get_similar_successful_tasks: cannot parse modificationTime=%r — "
            "falling back to now-anchored search (returned tasks may not be "
            "temporally comparable)",
            raw_mtime,
        )
    elif not raw_mtime:
        logger.warning(
            "get_similar_successful_tasks: task_data has no modificationTime — "
            "falling back to now-anchored search (returned tasks may not be "
            "temporally comparable)"
        )

    if anchor is None:
        since_dt = earliest_allowed
    else:
        desired = anchor - timedelta(days=WINDOW_DAYS)
        if desired < earliest_allowed:
            logger.warning(
                "get_similar_successful_tasks: failure modificationTime=%s is older "
                "than the server's %d-day window; falling back to now-anchored search "
                "(returned tasks may not be temporally comparable)",
                raw_mtime,
                SERVER_LIMIT_DAYS,
            )
            since_dt = earliest_allowed
        else:
            since_dt = desired
    since = since_dt.strftime("%Y-%m-%d %H:%M:%S")
    filters = {
        k: v
        for k, v in {
            "status": "finished|done",
            "userName": task_data.get("userName"),
            "prodSourceLabel": task_data.get("prodSourceLabel"),
            "processingType": task_data.get("processingType"),
            "transUses": task_data.get("transUses"),
            "transHome": task_data.get("transHome"),
            "architecture": task_data.get("architecture"),
        }.items()
        if v
    }
    params = {"since": since, "n_tasks": n_tasks, "filters": json.dumps(filters)}
    try:
        data = await asyncio.to_thread(
            _call_api, "get", "task/get_tasks_detailed_info_since", params
        )
        tasks = data if isinstance(data, list) else []
        return sorted(
            tasks,
            key=lambda t: _parse_panda_datetime(t.get("modificationTime")) or datetime.min,
            reverse=True,
        )
    except RuntimeError as exc:
        logger.warning("get_similar_successful_tasks: failed: %s", exc)
        return []


async def get_job_log(
    panda_id: int,
    filename: str = "payload.stdout",
) -> str | None:
    """Download a job log file from the PanDA monitor filebrowser.

    Args:
        panda_id: The ``PandaID`` of the job.
        filename: Log filename to fetch (default: ``payload.stdout``).

    Returns:
        Log content as a string, or ``None`` on any error.
    """
    base_url = os.getenv("PANDA_MONITOR_URL", "https://bigpanda.cern.ch")
    url = f"{base_url}/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    return await async_fetch_log_content(url)
