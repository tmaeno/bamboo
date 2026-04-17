"""PanDA-backed MCP client exposing extended task and job data tools.

All tools delegate to :mod:`bamboo.utils.panda_client` helpers.  They are
designed to fill the gaps in the standard first-pass knowledge extraction:
log files embedded in ``errorDialog``, retry-chain context, aggregated
job-level failure statistics, and per-job failure details.

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

``get_scout_job_details``
    Fetches scout jobs (all statuses) plus a sample of failed non-scout jobs.
    Returns cpuConsumptionTime, wallTime, jobDuration alongside pilot error fields.

``get_task_input_datasets``
    Fetches input and pseudo-input dataset descriptions (file counts,
    availability) via ``get_files_in_datasets``.

``search_panda_server_source``
    Searches the locally installed panda-server source code for functions
    or logic matching a keyword.  Used for resource estimation analysis
    (scout → ramCount algorithm) and error message tracing.

``search_panda_docs``
    Searches the official PanDA WMS documentation (readthedocs) using the
    Sphinx client-side search index.  Returns matching page titles, URLs,
    and text snippets.  Used to retrieve domain-level context about task
    statuses, error patterns, and configuration parameters.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
from pathlib import Path
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
_RETRY_CHAIN_FIELDS = ("jediTaskID", "status", "errorDialog", "retryID", "transUses")

# PanDA WMS documentation GitHub repository.
_PANDA_DOCS_ORG = "tmaeno"
_PANDA_DOCS_RAW_BASE = f"https://raw.githubusercontent.com/{_PANDA_DOCS_ORG}/panda-docs/main"
_PANDA_DOCS_WEB_BASE = f"https://github.com/{_PANDA_DOCS_ORG}/panda-docs/blob/main"
_PANDA_DOCS_TREE_URL = (
    f"https://api.github.com/repos/{_PANDA_DOCS_ORG}/panda-docs/git/trees/main?recursive=1"
)

_TASK_PARAMS_RST_PATH = "docs/source/advanced/task_params.rst"
_BROKERAGE_RST_PATH = "docs/source/advanced/brokerage.rst"

# RST files excluded from the BM25 full-text index because they are handled
# by dedicated structured parsers (e.g. _fetch_task_params_table).
_BM25_EXCLUDE: frozenset[str] = frozenset({
    _TASK_PARAMS_RST_PATH,  # parsed by _fetch_task_params_table
})

# In-process BM25 index: None = not yet built.
# Built on the first search_panda_docs call from all docs/source/**/*.rst files
# except those listed in _BM25_EXCLUDE.
# Tuple layout: (BM25Okapi index, list of (path, paragraph_text) parallel to the corpus).
_bm25_data: tuple | None = None

# Parsed splitRule parameter table from docs/source/advanced/task_params.rst.
# Maps parameter name (e.g. "useExhausted") to its one-line description.
# None = not yet fetched; {} = fetch attempted but failed or table was empty.
_task_params_table: dict[str, str] | None = None


async def _fetch_task_params_table() -> dict[str, str]:
    """Fetch and parse the splitRule parameter table from task_params.rst.

    Returns a dict mapping parameter name → description sentence.
    Result is cached process-wide in ``_task_params_table``.
    Returns ``{}`` on any fetch or parse error (fail-open).
    """
    global _task_params_table
    if _task_params_table is not None:
        return _task_params_table

    url = f"{_PANDA_DOCS_RAW_BASE}/{_TASK_PARAMS_RST_PATH}"
    try:
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            r.raise_for_status()
            rst_text = r.text
    except Exception as exc:
        logger.warning("_fetch_task_params_table: failed to fetch %s: %s", url, exc)
        _task_params_table = {}
        return _task_params_table

    _task_params_table = _parse_task_params_table(rst_text)
    logger.info(
        "_fetch_task_params_table: parsed %d splitRule parameter(s)",
        len(_task_params_table),
    )
    return _task_params_table


def _parse_task_params_table(rst_text: str) -> dict[str, str]:
    """Parse ``.. list-table::`` entries from task_params.rst.

    Each entry has three columns: Parameter, Code, Description.
    Returns ``{parameter_name: description}`` for all entries found.
    """
    result: dict[str, str] = {}
    # State: 0=seeking entry start, 1=got param, 2=got code, 3=in description
    state = 0
    param = ""
    desc_parts: list[str] = []

    for line in rst_text.splitlines():
        stripped = line.strip()

        if state == 0:
            # New entry starts with "* - <word>" (first column = parameter name)
            if stripped.startswith("* - "):
                candidate = stripped[4:].strip()
                # Parameter names are camelCase identifiers (no spaces)
                if candidate and " " not in candidate:
                    param = candidate
                    desc_parts = []
                    state = 1
        elif state == 1:
            # Second column: code abbreviation — skip
            if stripped.startswith("- "):
                state = 2
            elif not stripped:
                state = 0  # blank line without column 2 → abort entry
        elif state == 2:
            # Third column: description starts here
            if stripped.startswith("- "):
                desc_parts = [stripped[2:].strip()]
                state = 3
            elif not stripped:
                state = 0
        elif state == 3:
            if stripped.startswith("* - "):
                # New entry — save previous and start fresh
                if param and desc_parts:
                    result[param] = " ".join(desc_parts)
                candidate = stripped[4:].strip()
                if candidate and " " not in candidate:
                    param = candidate
                    desc_parts = []
                    state = 1
                else:
                    state = 0
            elif stripped.startswith("- "):
                # Unexpected new column in same entry — end description
                if param and desc_parts:
                    result[param] = " ".join(desc_parts)
                param = ""
                desc_parts = []
                state = 0
            elif not stripped:
                # Blank line ends description
                if param and desc_parts:
                    result[param] = " ".join(desc_parts)
                param = ""
                desc_parts = []
                state = 0
            else:
                # Continuation line of description
                desc_parts.append(stripped)

    # Flush last entry if file ends without trailing blank line
    if state == 3 and param and desc_parts:
        result[param] = " ".join(desc_parts)

    return result


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
                    "properties": {},
                    "required": [],
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
                        "max_depth": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum number of ancestor levels to follow",
                        },
                    },
                    "required": [],
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
                    "properties": {},
                    "required": [],
                },
            ),
            McpTool(
                name="get_scout_job_details",
                description=(
                    "Fetches scout jobs (all statuses) and a sample of failed non-scout jobs "
                    "for this task.  Returns per-job resource metrics: cpuConsumptionTime, "
                    "wallTime, jobDuration; plus pilot error fields: pilotErrorCode, "
                    "pilotErrorDiag, transExitCode, computingSite, jobStatus, processingType."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "max_jobs": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of job records to return",
                        },
                    },
                    "required": [],
                },
            ),
            McpTool(
                name="get_task_input_datasets",
                description=(
                    "Fetches the input and pseudo-input dataset descriptions (including file "
                    "counts and availability) for a task.  Use this when the reviewer notes "
                    "symptoms consistent with input data issues — e.g. Symptom nodes like "
                    "'no input files', 'dataset not found', or 'STAGEIN_FAILED', or when "
                    "nInputFiles is non-zero but the graph has no dataset-level context.  "
                    "Do NOT call if pilot/payload error codes are already the primary symptom."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            McpTool(
                name="search_panda_server_source",
                description=(
                    "Searches the locally installed panda-server source code for functions "
                    "or logic matching a keyword or phrase.  Returns matching code snippets "
                    "with file paths and line numbers.  Useful in two scenarios:\n"
                    "1. RESOURCE ESTIMATION: task is in pending state due to high resource "
                    "requirements (memory, walltime, disk) that appear overestimated from "
                    "scout jobs — search for the estimation algorithm (e.g. query='ramCount "
                    "scout' or 'actualMemoryUsed') to explain how scout measurements were "
                    "translated into task resource requirements.\n"
                    "2. ERROR MESSAGE TRACING: errorDialog contains a vague or internal "
                    "error string with no clear cause node — search for the exact error "
                    "string to find which code path raised it and under what condition "
                    "(e.g. query='Brokerage for scout jobs failed').\n"
                    "Do NOT call if the errorDialog already contains a specific pilot or "
                    "payload error code that is self-explanatory.  "
                    "Requires panda-server: pip install \"bamboo[panda]\"."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Keyword or phrase to search for in the panda-server source, "
                                "e.g. 'ramCount scout', 'actualMemoryUsed', "
                                "'Brokerage for scout jobs failed'.  "
                                "For error messages from errorDialog, strip task/job-specific "
                                "values (IDs, site names, numbers) and search for the static "
                                "template portion only — e.g. for 'task 12345 at AGLT2 failed' "
                                "use 'scout jobs failed' or 'at site failed'."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of matching snippets to return",
                        },
                    },
                    "required": ["query"],
                },
            ),
            McpTool(
                name="search_panda_docs",
                description=(
                    "Searches the official PanDA WMS documentation for context about a "
                    "task status, error pattern, or configuration concept.  Use this when "
                    "a node name or error string requires domain-level explanation — what "
                    "it means, when it occurs, and what typically causes or resolves it.  "
                    "Examples: query='exhausted task status retry limit' to understand "
                    "TaskStatusExhausted; query='too many files input dataset limit' to "
                    "understand TooManyFilesInDataset; query='scout job memory overestimate' "
                    "for resource-related pending tasks.  "
                    "Returns up to 3 matching doc sections with page title, URL, and a "
                    "text snippet from the relevant passage.  "
                    "Do NOT call for errors that are already fully explained by the "
                    "errorDialog or pilot error code alone."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language search query, e.g. "
                                "'exhausted task status retry', "
                                "'cpu efficiency threshold task killed', "
                                "'scout job memory overestimate ramCount'.  "
                                "Use lowercase words; avoid CamelCase node names."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum number of doc sections to return (1–5)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            McpTool(
                name="fetch_brokerage_context",
                description=(
                    "Fetches the full unfiltered brokerage log from the task's "
                    "errorDialog link together with the PanDA brokerage "
                    "documentation page. Use this when investigating "
                    "'no suitable sites', placement throttling, or "
                    "'BrokerageNoCandidates' symptoms to get complete "
                    "site-selection details that the trimmed first-pass log omits."
                ),
                parameters_schema={"type": "object", "properties": {}, "required": []},
            ),
        ]
        self._dispatch = {
            "fetch_error_dialog_logs": self._fetch_error_dialog_logs,
            "get_parent_task": self._get_parent_task,
            "get_retry_chain": self._get_retry_chain,
            "get_task_jobs_summary": self._get_task_jobs_summary,
            "get_scout_job_details": self._get_scout_job_details,
            "get_task_input_datasets": self._get_task_input_datasets,
            "search_panda_server_source": self._search_panda_server_source,
            "search_panda_docs": self._search_panda_docs,
            "fetch_brokerage_context": self._fetch_brokerage_context,
        }

    def list_tools(self) -> list[McpTool]:
        return list(self._tools)

    def task_data_tools(self) -> frozenset[str]:
        """Return names of tools whose handler accepts a ``task_data`` parameter.

        Discovered dynamically from the dispatch table so newly added tools
        are included automatically without updating any static list.
        """
        import inspect
        return frozenset(
            name
            for name, handler in self._dispatch.items()
            if "task_data" in inspect.signature(handler).parameters
        )

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
        task_data: dict[str, Any],
    ) -> dict[str, str]:
        """Download log files linked in the task's errorDialog HTML.

        Returns a dict mapping each URL to its text content.  URLs that
        return no content (network error, empty response) are excluded.
        Capped at :data:`_MAX_ERROR_DIALOG_LOGS` fetches.
        """
        task_id = task_data.get("jediTaskID")
        error_dialog: str = task_data.get("errorDialog") or ""
        urls = extract_log_urls(error_dialog)[:_MAX_ERROR_DIALOG_LOGS]
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
        task_data: dict[str, Any],
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Follow retryID links upward from the task, up to *max_depth* levels.

        Returns a list of compact task dicts (only the fields in
        :data:`_RETRY_CHAIN_FIELDS`), ordered from direct parent to oldest
        ancestor.  Stops early on fetch error or cycle detection.
        """
        task_id = task_data.get("jediTaskID")
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
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Aggregate per-job status and error data for *task_id*.

        Fetches all job descriptions in a single call and computes status
        counts, top error diagnostics, and a sample of failed job IDs.

        Returns ``{"error": "<reason>"}`` on any failure so the explorer degrades
        gracefully.
        """
        try:
            from pandaclient import Client  # noqa: PLC0415
        except ImportError:
            return {"error": "panda-client-light not installed"}

        try:
            from collections import Counter

            task_id = int(task_data["jediTaskID"])
            status, jobs = await asyncio.to_thread(
                Client.get_job_descriptions, task_id
            )
            if status != 0 or not isinstance(jobs, list):
                return {"error": f"get_job_descriptions returned status={status}"}

            status_counts = dict(Counter(j.get("jobStatus", "unknown") for j in jobs))
            failed = [
                j for j in jobs
                if j.get("jobStatus") in ("failed", "cancelled", "closed")
            ]
            error_diags = [
                j.get("pilotErrorDiag") or j.get("ddmErrorDiag") or j.get("exeErrorDiag", "")
                for j in failed
            ]
            top_errors = [
                {"diag": diag, "count": count}
                for diag, count in Counter(error_diags).most_common(10)
                if diag
            ]
            failed_ids = [j.get("PandaID") for j in failed[:20]]
            logger.info(
                "PandaMcpClient.get_task_jobs_summary: %d total job(s), %d failed for task_id=%s",
                len(jobs),
                len(failed),
                task_id,
            )
            return {
                "status_counts": status_counts,
                "top_errors": top_errors,
                "failed_job_ids": failed_ids,
            }
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_task_jobs_summary: failed for task_id=%s: %s",
                task_data.get("jediTaskID", "unknown"),
                exc,
            )
            return {"error": str(exc)}

    async def _get_scout_job_details(
        self,
        task_data: dict[str, Any],
        max_jobs: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch details for representative failed jobs for *task_id*.

        Fetches all unsuccessful jobs (via ``get_job_descriptions`` with
        ``unsuccessful_only=True``) then splits by ``processingType``:
        failed scouts are prioritised first, then samples by distinct
        ``pilotErrorCode`` from the remaining jobs.

        Returns a list of compact dicts with fields: ``jobID``, ``computingSite``,
        ``jobStatus``, ``pilotErrorCode``, ``pilotErrorDiag``, ``transExitCode``,
        ``processingType``.  Returns an empty list on any error.
        """
        try:
            from pandaclient import Client  # noqa: PLC0415
        except ImportError:
            logger.warning("PandaMcpClient.get_scout_job_details: panda-client-light not installed")
            return []

        _COMPACT = (
            "jobID",
            "computingSite",
            "jobStatus",
            "pilotErrorCode",
            "pilotErrorDiag",
            "transExitCode",
            "processingType",
            "cpuConsumptionTime",
            "wallTime",
            "jobDuration",
            "extendedProdSourceLabel",
        )

        def _compact(j: dict[str, Any]) -> dict[str, Any]:
            d: dict[str, Any] = {k: j.get(k) for k in _COMPACT}
            d["jobID"] = j.get("PandaID") or j.get("jobID")
            return d

        try:
            task_id_int = int(task_data["jediTaskID"])

            # fetche jobs:
            all_status, all_raw = await asyncio.to_thread(
                Client.get_job_descriptions, task_id_int, False
                )

            scouts: list[dict[str, Any]] = []
            other_failed: list[dict[str, Any]] = []

            if all_status == 0 and isinstance(all_raw, list):
                for j in all_raw:
                    if isinstance(j, dict):
                        if str(j.get("extendedProdSourceLabel", "")).endswith("scout"):
                            scouts.append(_compact(j))
                        elif j.get("jobStatus") in ("failed", "cancelled", "closed"):
                            other_failed.append(_compact(j))

            # Scout-first (all statuses), then distinct pilotErrorCode from failed non-scouts
            selected: list[dict[str, Any]] = list(scouts[:max_jobs])
            seen_codes: set[str] = {str(d.get("pilotErrorCode") or "") for d in selected}
            for d in other_failed:
                if len(selected) >= max_jobs:
                    break
                code = str(d.get("pilotErrorCode") or "")
                if code and code not in seen_codes:
                    selected.append(d)
                    seen_codes.add(code)
            # Fill remaining slots without code restriction
            for d in other_failed:
                if len(selected) >= max_jobs:
                    break
                if d not in selected:
                    selected.append(d)

            logger.info(
                "PandaMcpClient.get_scout_job_details: selected %d job(s) for task_id=%s",
                len(selected),
                task_id_int,
            )
            return selected
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_scout_job_details: failed for task_id=%s: %s",
                task_data.get("jediTaskID", "unknown"),
                exc,
            )
            return []

    async def _get_task_input_datasets(
        self,
        task_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fetch input and pseudo-input dataset descriptions for *task_id*.

        Returns a list of dataset dicts (name, type, file counts, availability)
        from the ``get_files_in_datasets`` endpoint.  Returns an empty list on
        any failure so the explorer degrades gracefully.
        """
        try:
            from pandaclient import Client  # noqa: PLC0415
        except ImportError:
            logger.warning("PandaMcpClient.get_task_input_datasets: panda-client-light not installed")
            return []

        try:
            task_id_int = int(task_data["jediTaskID"])
            status, datasets = await asyncio.to_thread(
                Client.get_files_in_datasets, task_id_int, "input,pseudo_input"
            )
            if status != 0 or not isinstance(datasets, list):
                logger.warning(
                    "PandaMcpClient.get_task_input_datasets: returned status=%s for task_id=%s",
                    status,
                    task_id_int,
                )
                return []
            logger.info(
                "PandaMcpClient.get_task_input_datasets: fetched %d dataset(s) for task_id=%s",
                len(datasets),
                task_id,
            )
            return datasets
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_task_input_datasets: failed for task_id=%s: %s",
                task_id,
                exc,
            )
            return []

    async def _search_panda_server_source(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the locally installed panda-server source for *query*.

        Locates the ``pandaserver`` package in ``site-packages``, walks all
        ``.py`` files, and returns snippets (±10 lines) around every
        case-insensitive match, up to *max_results*.

        Returns an empty list if panda-server is not installed or on any error.
        Install with: ``pip install "bamboo[panda]"``.
        """
        spec = importlib.util.find_spec("pandaserver")
        if spec is None or spec.origin is None:
            logger.warning(
                "PandaMcpClient.search_panda_server_source: pandaserver not installed — "
                "run: pip install \"bamboo[panda]\""
            )
            return []

        pkg_root = Path(spec.origin).parent

        def _search() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            query_words = query.lower().split()
            window_size = 5
            for py_file in sorted(pkg_root.rglob("*.py")):
                if len(results) >= max_results:
                    break
                try:
                    lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                i = 0
                while i < len(lines) and len(results) < max_results:
                    window = " ".join(lines[i : i + window_size]).lower()
                    if all(w in window for w in query_words):
                        start = max(0, i - 10)
                        end = min(len(lines), i + window_size + 10)
                        results.append({
                            "file": str(py_file.relative_to(pkg_root)),
                            "line": i + 1,
                            "context": "\n".join(lines[start:end]),
                        })
                        i += window_size  # skip to avoid overlapping matches
                    else:
                        i += 1
            return results

        try:
            results = await asyncio.to_thread(_search)
            logger.info(
                "PandaMcpClient.search_panda_server_source: %d match(es) for query=%r",
                len(results),
                query,
            )
            return results
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.search_panda_server_source: failed for query=%r: %s",
                query,
                exc,
            )
            return []

    async def _search_panda_docs(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the PanDA WMS documentation using a local in-process RST cache.

        On the first call, fetches the repository file tree (Git Trees API —
        public, no token) and then downloads every ``docs/source/**/*.rst`` file
        concurrently from ``raw.githubusercontent.com``.  The content is cached
        in-process so subsequent calls are instant.

        Returns a list of ``{"title", "url", "snippet"}`` dicts, or an empty list
        on any network or parse error.
        """
        import httpx

        from rank_bm25 import BM25Okapi  # noqa: PLC0415

        bm25, paragraphs = await _build_bm25_index(httpx, BM25Okapi)
        if bm25 is None:
            return []

        tokens = re.findall(r"[a-z0-9]+", query.lower())
        bm25_scores = bm25.get_scores(tokens)

        # Rank all paragraphs and collect the top ones, grouped by source file.
        # We gather up to max_results * 3 candidate paragraphs so each file can
        # contribute more than one passage (adjacent sections may both be relevant).
        ranked = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )
        ranked = [i for i in ranked if bm25_scores[i] > 0]

        # Collect one anchor paragraph index per source file (the highest-scoring).
        # Paragraphs within a file are stored in document order, so we can walk
        # forward from the anchor to grab the surrounding section context.
        seen_paths: dict[str, int] = {}  # path → anchor paragraph index
        for idx in ranked:
            path = paragraphs[idx][0]
            if path not in seen_paths:
                seen_paths[path] = idx
            if len(seen_paths) >= max_results:
                break

        results: list[dict[str, Any]] = []
        for path, anchor_idx in seen_paths.items():
            title = path.split("/")[-1].replace(".rst", "").replace("_", " ").title()
            # Collect the anchor paragraph + up to 8 following paragraphs from
            # the same file to capture section bullets/prose that scored 0
            # (they don't repeat the query term but are part of the same section).
            context_parts: list[str] = []
            for j in range(anchor_idx, min(anchor_idx + 9, len(paragraphs))):
                if paragraphs[j][0] == path:
                    context_parts.append(paragraphs[j][1])
                else:
                    break
            snippet = "\n".join(context_parts)[:1500]
            results.append({
                "title": title,
                "url": f"{_PANDA_DOCS_WEB_BASE}/{path}",
                "snippet": snippet,
            })

        logger.info(
            "PandaMcpClient.search_panda_docs: %d result(s) for query=%r", len(results), query
        )
        return results

    async def _fetch_brokerage_context(
        self,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the full unfiltered brokerage log(s) and the brokerage RST page.

        Unlike ``fetch_error_dialog_logs``, no log filter is applied so the
        complete site-selection funnel is visible to the LLM.
        """
        error_dialog: str = (task_data or {}).get("errorDialog") or ""
        urls = extract_log_urls(error_dialog)[:_MAX_ERROR_DIALOG_LOGS]

        contents = await asyncio.gather(
            *[async_fetch_log_content(url) for url in urls],
            return_exceptions=True,
        )
        logs: dict[str, str] = {
            url: c
            for url, c in zip(urls, contents)
            if c and not isinstance(c, BaseException)
        }
        logger.info(
            "PandaMcpClient.fetch_brokerage_context: fetched %d/%d log(s)",
            len(logs),
            len(urls),
        )
        brokerage_doc, doc_path = await self._fetch_brokerage_doc()
        return {"logs": logs, "brokerage_doc": brokerage_doc, "doc_path": doc_path}

    async def _fetch_brokerage_doc(self) -> tuple[str, str]:
        """Fetch the full PanDA brokerage RST documentation page."""
        import httpx  # noqa: PLC0415
        url = f"{_PANDA_DOCS_RAW_BASE}/{_BROKERAGE_RST_PATH}"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                content = resp.text if resp.status_code == 200 else ""
                if content:
                    logger.info(
                        "PandaMcpClient: fetched brokerage doc (%d chars)", len(content)
                    )
                return content, _BROKERAGE_RST_PATH
        except Exception as exc:
            logger.warning("PandaMcpClient: failed to fetch brokerage doc: %s", exc)
            return "", _BROKERAGE_RST_PATH


# ---------------------------------------------------------------------------
# Module-level helpers for search_panda_docs
# ---------------------------------------------------------------------------

async def _build_bm25_index(httpx_module: Any, BM25Okapi: Any) -> tuple:
    """Build and cache the BM25 index over PanDA docs RST paragraphs.

    Phase 1: fetch the repository file tree (Git Trees API — no auth needed
    for public repos) to discover every ``docs/source/**/*.rst`` path.
    Phase 2: download all files concurrently from raw.githubusercontent.com.
    Phase 3: split each file into paragraphs, clean RST markup, tokenise, and
    build a BM25Okapi index.

    Returns ``(bm25, paragraphs)`` where *paragraphs* is a list of
    ``(path, text)`` tuples parallel to the BM25 corpus.  On any fatal error
    returns ``(None, [])``.
    """
    global _bm25_data
    if _bm25_data is not None:
        return _bm25_data

    _gh_headers = {
        "User-Agent": "bamboo-panda-docs-search",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # --- Phase 1: discover RST paths ---
    try:
        async with httpx_module.AsyncClient(timeout=15) as client:
            resp = await client.get(_PANDA_DOCS_TREE_URL, headers=_gh_headers)
            resp.raise_for_status()
            tree = resp.json().get("tree", [])
    except Exception as exc:
        logger.warning(
            "PandaMcpClient.search_panda_docs: failed to fetch doc tree: %s", exc
        )
        _bm25_data = (None, [])
        return _bm25_data

    rst_paths = [
        item["path"]
        for item in tree
        if item.get("type") == "blob"
        and item.get("path", "").startswith("docs/source/")
        and item["path"].endswith(".rst")
        and item["path"] not in _BM25_EXCLUDE
    ]
    logger.info(
        "PandaMcpClient.search_panda_docs: discovered %d RST file(s)", len(rst_paths)
    )

    # --- Phase 2: fetch all files concurrently ---
    async def _fetch(client: Any, path: str) -> tuple[str, str]:
        try:
            r = await client.get(f"{_PANDA_DOCS_RAW_BASE}/{path}")
            r.raise_for_status()
            return path, r.text
        except Exception as exc:
            logger.debug("PandaMcpClient.search_panda_docs: skipped %s: %s", path, exc)
            return path, ""

    async with httpx_module.AsyncClient(timeout=15) as client:
        fetched = await asyncio.gather(*[_fetch(client, p) for p in rst_paths])

    raw_files = {path: content for path, content in fetched if content}
    logger.info(
        "PandaMcpClient.search_panda_docs: fetched %d/%d RST file(s)",
        len(raw_files), len(rst_paths),
    )

    # --- Phase 3: split into paragraphs and build BM25 index ---
    def _clean(text: str) -> str:
        text = re.sub(r"\.\. \S+::.*", "", text)           # directives
        text = re.sub(r":[a-z]+:`([^`]*)`", r"\1", text)  # inline roles → bare text
        text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)  # **bold** / *italic*
        text = re.sub(r"^[=\-~^#*+]{3,}\s*$", "", text, flags=re.MULTILINE)  # underlines
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    paragraphs: list[tuple[str, str]] = []
    for path, content in raw_files.items():
        for raw_para in re.split(r"\n{2,}", content):
            cleaned = _clean(raw_para)
            if cleaned:
                paragraphs.append((path, cleaned))

    corpus = [re.findall(r"[a-z0-9]+", text.lower()) for _, text in paragraphs]
    bm25 = BM25Okapi(corpus)

    _bm25_data = (bm25, paragraphs)
    logger.info(
        "PandaMcpClient.search_panda_docs: BM25 index built over %d paragraph(s)",
        len(paragraphs),
    )
    return _bm25_data
