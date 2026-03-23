"""PanDA-backed MCP client exposing extended task and job data tools.

All tools delegate to :mod:`bamboo.utils.panda_client` helpers.  They are
designed to fill the gaps in the standard first-pass knowledge extraction:
log files embedded in ``errorDialog``, retry-chain context, and aggregated
job-level failure statistics.

Tool catalogue
--------------
``fetch_error_dialog_logs``
    Extracts and downloads log files linked in a task's ``errorDialog`` HTML.

``get_parent_task``
    Fetches full task data for the parent task identified by ``retryID``.

``get_retry_chain``
    Follows the ``retryID`` chain upward and returns a compact summary of
    each ancestor task.

``get_task_jobs_summary``
    Aggregates per-job status and error information for a task.
    Falls back gracefully if the pandaclient bulk-jobs endpoint is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bamboo.mcp.base import McpClient, McpTool
from bamboo.utils.panda_client import (
    async_fetch_log_content,
    extract_log_urls,
    fetch_task_data,
)

logger = logging.getLogger(__name__)

# Maximum number of errorDialog log URLs to fetch per call.
_MAX_ERROR_DIALOG_LOGS = 5

# Fields kept from parent task dicts in retry-chain results (keeps LLM context compact).
_RETRY_CHAIN_FIELDS = ("taskID", "status", "errorDialog", "retryID", "transUses")


class PandaMcpClient(McpClient):
    """MCP client backed by PanDA endpoints.

    Each tool is safe to call concurrently via ``asyncio.gather``.
    """

    def __init__(self) -> None:
        self._tools: list[McpTool] = [
            McpTool(
                name="fetch_error_dialog_logs",
                description=(
                    "Extracts all log file URLs from a task's errorDialog HTML and downloads "
                    "their content.  Use this when the reviewer reports that Symptom nodes are "
                    "too vague (e.g. 'UnknownError' instead of a specific error code), that log "
                    "evidence is absent, or that error messages are incomplete.  Requires the "
                    "errorDialog field to be non-empty and contain at least one <a href> link."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": ["integer", "string"],
                            "description": "PanDA jediTaskID",
                        },
                        "error_dialog": {
                            "type": "string",
                            "description": "Raw errorDialog field value from task_data",
                        },
                    },
                    "required": ["task_id", "error_dialog"],
                },
            ),
            McpTool(
                name="get_parent_task",
                description=(
                    "Fetches the full task data dict for the parent task identified by retryID.  "
                    "Use this when the reviewer notes that the root cause is unclear and the task "
                    "has a non-empty retryID field, suggesting the failure originated in an "
                    "earlier attempt that may have richer error context.  "
                    "Do NOT call if retryID is absent or null."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "retry_id": {
                            "type": ["integer", "string"],
                            "description": "The retryID value from task_data",
                        },
                    },
                    "required": ["retry_id"],
                },
            ),
            McpTool(
                name="get_retry_chain",
                description=(
                    "Follows the retryID chain upward and returns a compact list of ancestor "
                    "task summaries (taskID, status, errorDialog, retryID, transUses).  "
                    "Use this when the reviewer notes that the failure pattern spans multiple "
                    "retry attempts, or when a single parent lookup is insufficient to establish "
                    "the original failure context.  "
                    "Do NOT call if retryID is absent or null.  "
                    "Prefer get_parent_task for single-level lookups."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": ["integer", "string"],
                            "description": "PanDA jediTaskID to start from",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum number of ancestor levels to follow",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
            McpTool(
                name="get_task_jobs_summary",
                description=(
                    "Fetches and aggregates per-job status and error information for a task.  "
                    "Returns status counts, the top error diagnostics, and a sample of failed "
                    "job IDs.  Use this when the reviewer notes that job-level failure "
                    "distribution is missing, that it is unclear whether failures are systematic "
                    "or sporadic, or that job diagnostic messages have not been captured."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": ["integer", "string"],
                            "description": "PanDA jediTaskID",
                        },
                    },
                    "required": ["task_id"],
                },
            ),
        ]
        self._dispatch = {
            "fetch_error_dialog_logs": self._fetch_error_dialog_logs,
            "get_parent_task": self._get_parent_task,
            "get_retry_chain": self._get_retry_chain,
            "get_task_jobs_summary": self._get_task_jobs_summary,
        }

    def list_tools(self) -> list[McpTool]:
        return list(self._tools)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        handler = self._dispatch.get(tool_name)
        if handler is None:
            raise ValueError(f"PandaMcpClient: unknown tool {tool_name!r}")
        return await handler(**kwargs)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _fetch_error_dialog_logs(
        self,
        task_id: int | str,
        error_dialog: str,
    ) -> dict[str, str]:
        """Download log files linked in *error_dialog* HTML.

        Returns a dict mapping each URL to its text content.  URLs that
        return no content (network error, empty response) are excluded.
        Capped at :data:`_MAX_ERROR_DIALOG_LOGS` fetches.
        """
        urls = extract_log_urls(error_dialog or "")[:_MAX_ERROR_DIALOG_LOGS]
        if not urls:
            logger.debug(
                "PandaMcpClient.fetch_error_dialog_logs: no log URLs in errorDialog "
                "for task_id=%s",
                task_id,
            )
            return {}

        contents = await asyncio.gather(
            *[async_fetch_log_content(url) for url in urls],
            return_exceptions=True,
        )
        result: dict[str, str] = {}
        for url, content in zip(urls, contents):
            if isinstance(content, BaseException):
                logger.warning(
                    "PandaMcpClient: failed to fetch log %s: %s", url, content
                )
            elif content:
                result[url] = content
        logger.info(
            "PandaMcpClient.fetch_error_dialog_logs: fetched %d/%d log(s) for task_id=%s",
            len(result),
            len(urls),
            task_id,
        )
        return result

    async def _get_parent_task(
        self,
        retry_id: int | str,
    ) -> dict[str, Any] | None:
        """Fetch the parent task's full data dict.

        Returns ``None`` (not raises) if the fetch fails, so the explorer
        can continue with whatever else it found.
        """
        try:
            parent = await fetch_task_data(retry_id)
            logger.info(
                "PandaMcpClient.get_parent_task: fetched parent task_id=%s", retry_id
            )
            return parent
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_parent_task: failed to fetch retry_id=%s: %s",
                retry_id,
                exc,
            )
            return None

    async def _get_retry_chain(
        self,
        task_id: int | str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Follow retryID links upward from *task_id*, up to *max_depth* levels.

        Returns a list of compact task dicts (only the fields in
        :data:`_RETRY_CHAIN_FIELDS`), ordered from direct parent to oldest
        ancestor.  Stops early on fetch error or cycle detection.
        """
        chain: list[dict[str, Any]] = []
        seen: set[str] = {str(task_id)}
        current_id: int | str | None = task_id

        # We need the retryID of the starting task, not the task itself.
        # Fetch it first to get its retryID.
        try:
            current_task = await fetch_task_data(current_id)
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_retry_chain: failed to fetch task_id=%s: %s",
                current_id,
                exc,
            )
            return chain

        current_id = current_task.get("retryID")

        for _ in range(max_depth):
            if not current_id or str(current_id) in seen:
                break
            seen.add(str(current_id))
            try:
                parent = await fetch_task_data(current_id)
            except Exception as exc:
                logger.warning(
                    "PandaMcpClient.get_retry_chain: failed to fetch task_id=%s: %s",
                    current_id,
                    exc,
                )
                break
            compact = {k: parent.get(k) for k in _RETRY_CHAIN_FIELDS}
            # Truncate errorDialog to keep LLM context compact.
            if compact.get("errorDialog"):
                compact["errorDialog"] = str(compact["errorDialog"])[:500]
            chain.append(compact)
            current_id = parent.get("retryID")

        logger.info(
            "PandaMcpClient.get_retry_chain: found %d ancestor(s) for task_id=%s",
            len(chain),
            task_id,
        )
        return chain

    async def _get_task_jobs_summary(
        self,
        task_id: int | str,
    ) -> dict[str, Any]:
        """Aggregate per-job status and error data for *task_id*.

        Returns ``{"error": "unavailable"}`` when the pandaclient bulk-jobs
        endpoint is not accessible, so the explorer degrades gracefully.
        """
        try:
            from pandaclient import Client  # noqa: PLC0415
        except ImportError:
            return {"error": "panda-client-light not installed"}

        try:
            from collections import Counter

            status, job_list = await asyncio.to_thread(
                Client.getJobIDsInBulk, int(task_id)
            )
            if status != 0 or not isinstance(job_list, (list, dict)):
                return {"error": f"pandaclient returned status={status}"}

            # job_list may be a list of job dicts or a dict of jobID→status
            if isinstance(job_list, dict):
                # dict form: {jobID: status_string}
                status_counts = dict(Counter(job_list.values()))
                return {
                    "status_counts": status_counts,
                    "top_errors": [],
                    "failed_job_ids": [
                        jid
                        for jid, st in job_list.items()
                        if "failed" in str(st).lower()
                    ][:20],
                }

            # list form: list of job attribute dicts
            status_counts = dict(Counter(j.get("jobStatus", "unknown") for j in job_list))
            error_diags = [
                j.get("pilotErrorDiag") or j.get("ddmErrorDiag") or j.get("exeErrorDiag", "")
                for j in job_list
                if j.get("jobStatus") in ("failed", "cancelled")
            ]
            top_errors = [
                {"diag": diag, "count": count}
                for diag, count in Counter(error_diags).most_common(10)
                if diag
            ]
            failed_ids = [
                j.get("PandaID") or j.get("jobID")
                for j in job_list
                if j.get("jobStatus") in ("failed", "cancelled")
            ][:20]
            return {
                "status_counts": status_counts,
                "top_errors": top_errors,
                "failed_job_ids": failed_ids,
            }
        except AttributeError:
            # pandaclient version without getJobIDsInBulk
            return {"error": "bulk jobs endpoint unavailable in this pandaclient version"}
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_task_jobs_summary: failed for task_id=%s: %s",
                task_id,
                exc,
            )
            return {"error": str(exc)}
