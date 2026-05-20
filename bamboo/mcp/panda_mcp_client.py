"""PanDA-backed MCP client exposing extended task and job data tools.

All tools delegate to :mod:`bamboo.utils.panda_client` helpers.  They are
designed to fill the gaps in the standard first-pass knowledge extraction:
log files embedded in ``errorDialog``, retry-chain context, aggregated
job-level failure statistics, and per-job failure details.
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
    get_datasets_and_files,
    get_job_descriptions,
    get_job_log,
    get_similar_successful_tasks,
)

logger = logging.getLogger(__name__)


# Maximum number of errorDialog log URLs to fetch per call.
_MAX_ERROR_DIALOG_LOGS = 5

# Fields kept from parent task dicts in retry-chain results (keeps LLM context compact).
_RETRY_CHAIN_FIELDS = ("jediTaskID", "status", "errorDialog", "retryID", "transUses")

# PanDA WMS documentation GitHub repository (source enumeration + task_params fetch).
_PANDA_DOCS_ORG = "tmaeno"
_PANDA_DOCS_RAW_BASE = f"https://raw.githubusercontent.com/{_PANDA_DOCS_ORG}/panda-docs/main"
_TASK_PARAMS_RST_PATH = "docs/source/advanced/task_params.rst"
_GDPCONFIG_RST_PATH = "docs/source/advanced/gdpconfig.rst"
_BROKERAGE_RST_PATH = "docs/source/advanced/brokerage.rst"

# Lazy-initialised doc navigator — built on the first search_panda_docs call.
_doc_navigator: Any = None

# Parsed splitRule parameter table from docs/source/advanced/task_params.rst.
# Maps parameter name (e.g. "useExhausted") to its one-line description.
# None = not yet fetched; {} = fetch attempted but failed or table was empty.
_task_params_table: dict[str, str] | None = None

# Parsed gdpconfig parameter table from docs/source/advanced/gdpconfig.rst.
# Maps UPPERCASE_KEY (and base form without wildcard suffixes) to description.
_gdpconfig_table: dict[str, str] | None = None


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


async def _fetch_gdpconfig_table() -> dict[str, str]:
    """Fetch and parse the gdpconfig parameter table from gdpconfig.rst.

    Returns a dict mapping parameter key → description.  Both the exact key
    (e.g. ``"SCOUT_MEM_LEAK_PER_CORE_<activity>"``) and its base form with
    wildcard suffixes stripped (e.g. ``"SCOUT_MEM_LEAK_PER_CORE"``) are stored
    so callers can match by either form.
    Result is cached process-wide.  Returns ``{}`` on any error (fail-open).
    """
    global _gdpconfig_table
    if _gdpconfig_table is not None:
        return _gdpconfig_table

    url = f"{_PANDA_DOCS_RAW_BASE}/{_GDPCONFIG_RST_PATH}"
    try:
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            r.raise_for_status()
            rst_text = r.text
    except Exception as exc:
        logger.warning("_fetch_gdpconfig_table: failed to fetch %s: %s", url, exc)
        _gdpconfig_table = {}
        return _gdpconfig_table

    _gdpconfig_table = _parse_gdpconfig_table(rst_text)
    logger.info(
        "_fetch_gdpconfig_table: parsed %d gdpconfig parameter(s)",
        len(_gdpconfig_table),
    )
    return _gdpconfig_table


def _parse_gdpconfig_table(rst_text: str) -> dict[str, str]:
    """Parse the two-column ``.. list-table::`` in gdpconfig.rst.

    Stores each key twice:
    - Exact form: ``"SCOUT_MEM_LEAK_PER_CORE_<activity>"``
    - Base form (wildcard suffixes stripped): ``"SCOUT_MEM_LEAK_PER_CORE"``

    Both map to the same description so lookup succeeds regardless of which
    form appears in doc_hints text.
    """
    result: dict[str, str] = {}
    # State: 0=seeking entry, 1=got key, 2=in description
    state = 0
    key = ""
    desc_parts: list[str] = []

    def _flush() -> None:
        if key and desc_parts:
            desc = " ".join(desc_parts)
            result[key] = desc
            # Also store the base form with wildcard suffixes removed.
            base = re.sub(r"\[_<[^>]+>\]$", "", key)   # KEY[_<x>] → KEY
            base = re.sub(r"_<[^>]+>$", "", base)       # KEY_<x>   → KEY
            if base and base != key:
                result.setdefault(base, desc)

    for line in rst_text.splitlines():
        stripped = line.strip()
        if state == 0:
            if stripped.startswith("* - "):
                candidate = stripped[4:].strip()
                # gdpconfig keys are UPPERCASE (allow [ < > _ as part of name)
                if candidate and candidate[0].isupper():
                    key = candidate
                    desc_parts = []
                    state = 1
        elif state == 1:
            if stripped.startswith("- "):
                desc_parts = [stripped[2:].strip()]
                state = 2
            elif not stripped:
                state = 0
        elif state == 2:
            if stripped.startswith("* - "):
                _flush()
                candidate = stripped[4:].strip()
                if candidate and candidate[0].isupper():
                    key = candidate
                    desc_parts = []
                    state = 1
                else:
                    key = ""
                    desc_parts = []
                    state = 0
            elif stripped.startswith("- "):
                _flush()
                key = ""
                desc_parts = []
                state = 0
            elif not stripped:
                _flush()
                key = ""
                desc_parts = []
                state = 0
            else:
                desc_parts.append(stripped)

    _flush()
    return result


class PandaMcpClient(McpClient):
    """MCP client backed by PanDA endpoints.

    Each tool is safe to call concurrently via ``asyncio.gather``.
    """

    def __init__(self) -> None:
        self._tools: list[McpTool] = [
            McpTool(
                name="fetch_linked_log_files",
                description=(
                    "Downloads task-level log files linked from the task's errorDialog HTML "
                    "(typically brokerage/dispatch logs from PanDA's site-selection layer). "
                    "NOT for fetching job execution logs — for that use get_failed_job_log_summary. "
                    "Use this when the reviewer reports that Symptom nodes are too vague "
                    "(e.g. 'UnknownError' instead of a specific error code), that log "
                    "evidence is absent, or that error messages are incomplete. Requires the "
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
                    "Returns AGGREGATE STATISTICS only (status counts, top error diagnostics, "
                    "and a sample of failed job IDs) — does NOT include actual log content. "
                    "For per-job log signal extraction use get_failed_job_log_summary. "
                    "Use this when the reviewer notes that job-level failure distribution is "
                    "missing, that it is unclear whether failures are systematic or sporadic, "
                    "or that job diagnostic messages have not been captured."
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
                    "Always phrase the query as a natural-language question.  "
                    "Examples: query='Why is a task set to exhausted after reaching the retry limit?' "
                    "to understand TaskStatusExhausted; "
                    "query='Why are there too many files in the input dataset?' "
                    "to understand TooManyFilesInDataset; "
                    "query='How does a scout job memory overestimate affect pending tasks?' "
                    "for resource-related pending tasks.  "
                    "Returns up to 3 matching doc sections, each with: title, URL, "
                    "full section content, parent_context (summary of the parent section), "
                    "breadcrumb (ancestor heading path), and source ('semantic', "
                    "'llm_traversal', or 'both').  "
                    "Do NOT call for errors that are already fully explained by the "
                    "errorDialog or pilot error code alone."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Phrase as a natural-language question — this improves "
                                "ranking quality.  Examples: "
                                "'Why is a task set to exhausted after reaching the retry limit?', "
                                "'When is a task killed due to low CPU efficiency?', "
                                "'How does scout job memory overestimate affect ramCount?', "
                                "'Why does brokerage find no candidate sites for a pending task?'  "
                                "Use lowercase words; avoid CamelCase node names."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum number of doc sections to return (1–5)",
                        },
                        "keyword_query": {
                            "type": "string",
                            "description": (
                                "Optional exact-term query for BM25 keyword search "
                                "(snake_case identifiers, ALL_CAPS constants). "
                                "If omitted, the main query is used for all strategies."
                            ),
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
            McpTool(
                name="fetch_cli_options",
                description=(
                    "Returns the --help output for the prun or pathena command. "
                    "Pass component='prun' or component='pathena' based on the "
                    "Component node in the task graph. Use when the resolution "
                    "involves resubmitting a task with different CLI options."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "component": {
                            "type": "string",
                            "enum": ["prun", "pathena"],
                            "description": "Which submission command to fetch help for",
                        }
                    },
                    "required": ["component"],
                },
            ),
            McpTool(
                name="find_similar_successful_tasks",
                description=(
                    "Finds recently finished tasks similar to the current failing task by "
                    "matching userName, processingType, software release (transUses/transHome), "
                    "and architecture.  Useful for differential diagnosis when the failure's "
                    "root cause is ambiguous from the failing task alone (could be code crash, "
                    "config error, transient infrastructure issue, etc.) — a similar successful "
                    "task confirms the workflow itself works and isolates the failure to this "
                    "task's specific inputs or environment.  Returns a ranked list of candidate "
                    "task dicts; pass the jediTaskID of the best match to "
                    "compare_failed_vs_successful_job_logs to obtain a compact "
                    "structured comparison."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "days_back": {
                            "type": "integer",
                            "default": 30,
                            "description": "How many days back to search (max 30)",
                        },
                        "n_tasks": {
                            "type": "integer",
                            "default": 50,
                            "description": "Maximum number of candidate tasks to retrieve",
                        },
                    },
                    "required": [],
                },
            ),
            McpTool(
                name="get_failed_job_log_summary",
                description=(
                    "Returns a compact heuristic summary (~3-10 KB) of the "
                    "current task's representative failed-job payload.stdout: "
                    "section markers seen, last meaningful action, last "
                    "timestamp, error lines (ERROR/FATAL/Traceback/etc.), and "
                    "the last ~50 raw tail lines. The primary tool for "
                    "inspecting WHY the failed job died — preferred for any "
                    "procedure that says 'check the failed log'. Resolves "
                    "the failed PandaID internally from task_data; no "
                    "parameters needed."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            McpTool(
                name="compare_failed_vs_successful_job_logs",
                description=(
                    "Fetches payload.stdout from a representative failed job of the "
                    "CURRENT task and a representative finished job of the REFERENCE "
                    "successful task, then runs structural alignment to identify "
                    "where the failed job diverged or was truncated. Returns a "
                    "structured comparison: whether the failed log appears truncated "
                    "mid-execution, a candidate problem step name, its duration in "
                    "the successful run, the next ~150 raw lines of the successful "
                    "log past the divergence point, and the last ~50 raw lines of "
                    "the failed log. Use this when the procedure says to 'compare "
                    "with a similar successful task' or to diagnose silent kills "
                    "(no error/traceback). Pass the jediTaskID returned by "
                    "find_similar_successful_tasks."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "reference_task_id": {
                            "type": ["integer", "string"],
                            "description": (
                                "jediTaskID of the reference successful task — "
                                "typically from find_similar_successful_tasks()[0]"
                                "['jediTaskID']."
                            ),
                        },
                    },
                    "required": ["reference_task_id"],
                },
            ),
            McpTool(
                name="summarize_job_log",
                description=(
                    "Returns a compact heuristic summary (~3-10 KB) of a "
                    "job's payload.stdout for a specific PandaID: total "
                    "lines/chars, section markers seen, last timestamp, "
                    "last meaningful action, error lines, and the last "
                    "~50 raw tail lines. Use this when you have a specific "
                    "PandaID (e.g. from get_scout_job_details). For the "
                    "current task's representative failed job, prefer the "
                    "convenience tool `get_failed_job_log_summary()`. For "
                    "differential diagnosis against a similar successful "
                    "task, use `compare_failed_vs_successful_job_logs`."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "panda_id": {
                            "type": ["integer", "string"],
                            "description": (
                                "PandaID of the job whose log to summarize."
                            ),
                        },
                    },
                    "required": ["panda_id"],
                },
            ),
        ]
        self._dispatch = {
            "fetch_linked_log_files": self._fetch_linked_log_files,
            "get_parent_task": self._get_parent_task,
            "get_retry_chain": self._get_retry_chain,
            "get_task_jobs_summary": self._get_task_jobs_summary,
            "get_scout_job_details": self._get_scout_job_details,
            "get_task_input_datasets": self._get_task_input_datasets,
            "search_panda_server_source": self._search_panda_server_source,
            "search_panda_docs": self._search_panda_docs,
            "fetch_brokerage_context": self._fetch_brokerage_context,
            "fetch_cli_options": self._fetch_cli_options,
            "find_similar_successful_tasks": self._find_similar_successful_tasks,
            "get_failed_job_log_summary": self._get_failed_job_log_summary,
            "compare_failed_vs_successful_job_logs": self._compare_failed_vs_successful_job_logs,
            "summarize_job_log": self._summarize_job_log,
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

    async def _fetch_linked_log_files(
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
                "PandaMcpClient.fetch_linked_log_files: no log URLs in errorDialog "
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
            "PandaMcpClient.fetch_linked_log_files: fetched %d/%d log(s) for task_id=%s",
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
        from collections import Counter  # noqa: PLC0415

        task_id = int(task_data["jediTaskID"])
        try:
            jobs = await get_job_descriptions(task_id)
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
                task_id,
                exc,
            )
            return {"error": str(exc)}

    async def _get_scout_job_details(
        self,
        task_data: dict[str, Any],
        max_jobs: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetch details for representative scout and failed jobs for *task_id*.

        Fetches all jobs (via ``get_job_descriptions``) then splits by
        ``extendedProdSourceLabel``: jobs whose label ends with ``scout`` are
        prioritised first (regardless of status), then samples by distinct
        ``pilotErrorCode`` from the remaining failed/cancelled/closed jobs.

        Returns a list of compact dicts with fields: ``jobID``, ``computingSite``,
        ``jobStatus``, ``pilotErrorCode``, ``pilotErrorDiag``, ``transExitCode``,
        ``processingType``, ``cpuConsumptionTime``, ``wallTime``, ``jobDuration``,
        ``extendedProdSourceLabel``, ``maxPSS`` (maximum proportional set size, kB),
        ``actualCoreCount``, ``memory_leak``.  Returns an empty list on any error.
        """
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
            "maxPSS",
            "actualCoreCount",
            "memory_leak",
        )

        def _compact(j: dict[str, Any]) -> dict[str, Any]:
            d: dict[str, Any] = {k: j.get(k) for k in _COMPACT}
            d["jobID"] = j.get("PandaID") or j.get("jobID")
            return d

        try:
            task_id_int = int(task_data["jediTaskID"])
            all_raw = await get_job_descriptions(task_id_int)

            scouts: list[dict[str, Any]] = []
            other_failed: list[dict[str, Any]] = []

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

        Returns a list of ``{"dataset": {"name": ..., "id": ...}, "files": [...]}``
        dicts.  Returns an empty list on any failure so the explorer degrades
        gracefully.
        """
        task_id_int = int(task_data["jediTaskID"])
        try:
            datasets = await get_datasets_and_files(task_id_int, dataset_only=True)
            logger.info(
                "PandaMcpClient.get_task_input_datasets: fetched %d dataset(s) for task_id=%s",
                len(datasets),
                task_id_int,
            )
            return datasets
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_task_input_datasets: failed for task_id=%s: %s",
                task_id_int,
                exc,
            )
            return []

    async def _find_similar_successful_tasks(
        self,
        task_data: dict[str, Any],
        days_back: int = 30,
        n_tasks: int = 50,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Return recently-finished tasks similar to the failing task."""
        try:
            results = await get_similar_successful_tasks(task_data, days_back, n_tasks)
            logger.info(
                "PandaMcpClient.find_similar_successful_tasks: found %d candidate(s)",
                len(results),
            )
            return results
        except Exception as exc:
            logger.warning("PandaMcpClient.find_similar_successful_tasks: failed: %s", exc)
            return {"error": str(exc)}

    async def _get_failed_job_log_summary(
        self,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Heuristic summary of the current task's representative failed-job log."""
        from bamboo.utils.log_filters import summarize_log_content  # noqa: PLC0415

        task_id = int(task_data["jediTaskID"])
        try:
            jobs = await get_job_descriptions(task_id, unsuccessful_only=True)
            if not jobs:
                return {"error": "no failed jobs found"}
            panda_id = jobs[0].get("PandaID") or jobs[0].get("jobID")
            content = await get_job_log(panda_id) or ""
            summary = summarize_log_content(content)
            summary["pandaID"] = panda_id
            summary["filename"] = "payload.stdout"
            logger.info(
                "PandaMcpClient.get_failed_job_log_summary: PandaID=%s, "
                "%d lines, %d error line(s), last_section=%r",
                panda_id,
                summary["total_lines"],
                len(summary["error_lines"]),
                summary["last_section_marker"],
            )
            return summary
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.get_failed_job_log_summary: failed for task_id=%s: %s",
                task_id,
                exc,
            )
            return {"error": str(exc)}

    async def _compare_failed_vs_successful_job_logs(
        self,
        task_data: dict[str, Any],
        reference_task_id: int | str,
    ) -> dict[str, Any]:
        """Fetch failed + reference-successful job logs and return alignment."""
        from bamboo.utils.log_filters import compare_job_logs  # noqa: PLC0415

        task_id = int(task_data["jediTaskID"])
        ref_id = int(reference_task_id)
        try:
            failed_jobs = await get_job_descriptions(task_id, unsuccessful_only=True)
            if not failed_jobs:
                return {"error": "no failed jobs found for current task"}
            failed_pid = failed_jobs[0].get("PandaID") or failed_jobs[0].get("jobID")
            failed_log = await get_job_log(failed_pid)

            ref_jobs = await get_job_descriptions(ref_id)
            finished = [j for j in ref_jobs if j.get("jobStatus") == "finished"]
            if not finished:
                return {"error": f"no finished jobs found for reference task {ref_id}"}
            succ_pid = finished[0].get("PandaID") or finished[0].get("jobID")
            successful_log = await get_job_log(succ_pid)

            result = compare_job_logs(failed_log or "", successful_log or "")
            result["failed_panda_id"] = failed_pid
            result["successful_panda_id"] = succ_pid
            result["reference_task_id"] = ref_id
            logger.info(
                "PandaMcpClient.compare_failed_vs_successful_job_logs: "
                "failed=%s successful=%s ref_task=%s truncated=%s step=%r",
                failed_pid, succ_pid, ref_id,
                result["failed_log_appears_truncated_mid_execution"],
                result["candidate_problem_step_from_alignment"],
            )
            return result
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.compare_failed_vs_successful_job_logs: "
                "failed (task_id=%s, ref=%s): %s",
                task_id, ref_id, exc,
            )
            return {"error": str(exc)}

    async def _summarize_job_log(
        self,
        panda_id: int | str,
    ) -> dict[str, Any]:
        """Fetch and heuristically summarise a job log by PandaID."""
        from bamboo.utils.log_filters import summarize_log_content  # noqa: PLC0415

        pid = int(panda_id)
        try:
            content = await get_job_log(pid) or ""
            summary = summarize_log_content(content)
            summary["pandaID"] = pid
            summary["filename"] = "payload.stdout"
            logger.info(
                "PandaMcpClient.summarize_job_log: PandaID=%s, %d lines, "
                "%d error line(s), last_section=%r",
                pid,
                summary["total_lines"],
                len(summary["error_lines"]),
                summary["last_section_marker"],
            )
            return summary
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.summarize_job_log: failed for PandaID=%s: %s",
                pid,
                exc,
            )
            return {"error": str(exc)}

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
        keyword_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search PanDA WMS documentation via the graph-based semantic navigator.

        On the first call, builds a heading-level graph from the ReadTheDocs HTML
        pages, LLM-summarises every node, and stores embeddings in a dedicated
        Qdrant collection.  Subsequent calls reuse the stored index, rebuilding
        only when the upstream GitHub tree SHA has changed.

        Two strategies run in parallel: flat semantic search (fast, good for
        paraphrase/synonym queries) and LLM-guided top-down traversal (catches
        non-obvious relevance).  Results are merged and ranked.

        Returns a list of ``{"title", "url", "content", "parent_context",
        "breadcrumb", "source"}`` dicts, or an empty list on any error.
        """
        global _doc_navigator
        from bamboo.agents.panda_doc_navigator import PandaDocNavigator  # noqa: PLC0415

        try:
            if _doc_navigator is None:
                _doc_navigator = PandaDocNavigator()
            results = await _doc_navigator.search(
                query, top_k=max_results, keyword_query=keyword_query
            )
        except Exception as exc:
            logger.warning(
                "PandaMcpClient.search_panda_docs: navigator failed for query=%r: %s",
                query, exc,
            )
            return []

        logger.info(
            "PandaMcpClient.search_panda_docs: %d result(s) for query=%r", len(results), query
        )
        return [
            {
                "title": r.title,
                "url": r.url,
                "content": r.content,
                "parent_context": r.parent_summary,
                "breadcrumb": r.breadcrumb,
                "source": r.source,
            }
            for r in results
        ]

    async def _fetch_brokerage_context(
        self,
        task_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the full unfiltered brokerage log(s) and the brokerage RST page.

        Unlike ``fetch_linked_log_files``, no log filter is applied so the
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

    async def _fetch_cli_options(self, component: str) -> dict[str, Any]:
        """Return the --help output for prun or pathena."""
        import io  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from contextlib import redirect_stderr, redirect_stdout  # noqa: PLC0415

        buf = io.StringIO()
        # Isolate panda-client from the parent process's sys.argv —
        # PrunScript.main reads sys.argv directly for its deprecated-option
        # scan, so any unrelated flags from the bamboo CLI (e.g. --task-id)
        # surface as "unrecognized argument" log errors.
        saved_argv = sys.argv[:]
        try:
            sys.argv = [component, "--help"]
            try:
                if component == "prun":
                    from pandaclient import PrunScript  # noqa: PLC0415
                    try:
                        with redirect_stdout(buf), redirect_stderr(buf):
                            PrunScript.main(ext_args=["--help"])
                    except SystemExit:
                        pass
                elif component == "pathena":
                    # PathenaScript defines optP at module level; importing
                    # it may itself trigger argparse (which sees the
                    # injected "--help" and raises SystemExit). The
                    # try/except must wrap the import too, not just the
                    # print_help call — SystemExit is a BaseException so it
                    # would otherwise propagate past the outer
                    # `except Exception` and kill the process.
                    try:
                        with redirect_stdout(buf), redirect_stderr(buf):
                            from pandaclient import PathenaScript  # noqa: PLC0415
                            PathenaScript.optP.print_help(buf)
                    except SystemExit:
                        pass
                else:
                    return {"error": f"Unknown component: {component}"}
            except ImportError as exc:
                return {"error": f"pandaclient not available: {exc}"}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}
        finally:
            sys.argv = saved_argv
        return {component: buf.getvalue()}

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

