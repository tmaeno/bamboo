"""Log pre-filtering utilities — reduce noise before sending to the LLM.

This module is intentionally separate from any extractor so it can be reused
by any pipeline stage that needs to compress raw log text.  New log-type
specialisations should be added here (new ``filter_<type>_log`` function +
constants) and registered in :func:`filter_log_auto`.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic signal extraction
# ---------------------------------------------------------------------------

# Lines containing any of these tokens (case-insensitive) are considered
# signal lines worth keeping.
_SIGNAL_PATTERNS: re.Pattern = re.compile(
    r"error|exception|traceback|fatal|critical|warning|warn|fail|abort|"
    r"killed|timeout|segfault|panic|denied|refused|missing|not found|"
    r"brokerage|no candidates|pilot|payload",
    re.IGNORECASE,
)

# Parts of a log line that are purely instance-specific and should be
# normalised away before deduplication (timestamps, hex addresses, UUIDs,
# numeric job/task IDs, hostnames).
_NOISE_RE: re.Pattern = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"  # ISO timestamps
    r"|\b\d{2}/\d{2}/\d{4}\b"  # dates MM/DD/YYYY
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"  # UUIDs
    r"|0x[0-9a-fA-F]+"  # hex addresses
    r"|\b\d{5,}\b"  # long numeric IDs (task/job IDs ≥5 digits)
    r"|\b[\w.-]+\.(cern\.ch|triumf\.ca|bnl\.gov|slac\.stanford\.edu)\b",  # hostnames
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Brokerage-log-specific patterns (AtlasBroker / AtlasAnalJobBroker)
# ---------------------------------------------------------------------------

# Minimum number of brokerage-specific markers required to treat a log as a
# brokerage log rather than a generic one.
_BROKERAGE_MIN_MARKERS = 3

# At least one of these must appear in the log to flag it as a brokerage log.
_BROKERAGE_LOG_MARKER_RE: re.Pattern = re.compile(
    r"Job brokerage summary"
    r"|candidates (?:passed|have input data|for final check)"
    r"|skip site=\S+.*?criteria=-",
    re.IGNORECASE,
)

# Opening line of the funnel summary block.
_BROKERAGE_SUMMARY_HEADER_RE: re.Pattern = re.compile(
    r"={3,}\s*Job brokerage summary\s*={3,}",
    re.IGNORECASE,
)

# A single row of the funnel summary: "1178 ->  54 candidates,  96% cut : input data check"
_BROKERAGE_SUMMARY_ENTRY_RE: re.Pattern = re.compile(
    r"\s*(\d+)\s*->\s*(\d+)\s*candidates,\s*([\d.]+)%\s*cut\s*:\s*(.+)"
)

# Progress milestone lines that mark the end of each filter stage.
# The optional non-capturing group handles parenthetical annotations such as
# "398 candidates (398 with AUTO, 0 with ANY) passed SW/HW check".
_BROKERAGE_PROGRESS_RE: re.Pattern = re.compile(
    r"(\d+)\s+candidates?(?:\s*\([^)]*\))?\s+(?:passed|have input data|for final check)",
    re.IGNORECASE,
)

# The very first line of a brokerage run.
_BROKERAGE_INITIAL_RE: re.Pattern = re.compile(
    r"initial\s+(\d+)\s+candidates",
    re.IGNORECASE,
)

# A site was excluded at some filter stage.
_BROKERAGE_SKIP_RE: re.Pattern = re.compile(r"skip site=", re.IGNORECASE)

# A site was selected for job submission.
_BROKERAGE_USE_SITE_RE: re.Pattern = re.compile(
    r"use site=\S+.*?criteria=\+use", re.IGNORECASE
)

# Data-locality / replica-availability diagnostic lines.
_BROKERAGE_DATA_AVAIL_RE: re.Pattern = re.compile(
    r"replica_availability"
    r"|is distributed"
    r"|is available at \d+ sites(?: on (?:DISK|TAPE))?",
    re.IGNORECASE,
)

# Lines that explain why a site was demoted to "problematic" (user-queue or
# bad-site throttle applied in the final weighting step).
_BROKERAGE_PROBLEMATIC_RE: re.Pattern = re.compile(
    r"getting rid of problematic site" r"|unsuitable for the user",
    re.IGNORECASE,
)


def _brokerage_filter_impl(
    log_text: str,
    top_cuts: int,
    max_skip_lines_per_section: int,
    include_data_avail: bool,
    fn_name: str,
) -> "str | None":
    """Shared implementation for analysis- and production-brokerage log filters.

    Both AtlasBroker and AtlasProdJobBroker produce logs with an identical
    funnel structure.  The only difference is that analysis logs include
    data-availability lines (``replica_availability``, ``is distributed``),
    which production logs do not.

    Returns the filtered string, or ``None`` if the log does not appear to be
    a brokerage log (caller should fall back to :func:`filter_log`).
    """
    if not log_text or not log_text.strip():
        return ""

    lines = log_text.splitlines()

    # --- Detect: require enough brokerage-specific markers ----------------
    n_markers = sum(1 for line in lines if _BROKERAGE_LOG_MARKER_RE.search(line))
    if n_markers < _BROKERAGE_MIN_MARKERS:
        return None

    # --- Locate the summary section ---------------------------------------
    summary_start_idx: "int | None" = None
    for i, line in enumerate(lines):
        if _BROKERAGE_SUMMARY_HEADER_RE.search(line):
            summary_start_idx = i
            break

    body_lines = lines[:summary_start_idx] if summary_start_idx is not None else lines
    summary_lines = lines[summary_start_idx:] if summary_start_idx is not None else []

    # --- Parse funnel summary entries ------------------------------------
    # Each entry: (n_before, n_after, cut_pct, check_name)
    summary_entries: "list[tuple[int, int, float, str]]" = []
    for line in summary_lines:
        m = _BROKERAGE_SUMMARY_ENTRY_RE.search(line)
        if m:
            summary_entries.append(
                (
                    int(m.group(1)),
                    int(m.group(2)),
                    float(m.group(3)),
                    m.group(4).strip(),
                )
            )

    # --- Build candidate-count → line-index map for the body -------------
    # Each "N candidates passed/have …" progress line marks the END of a
    # filter stage.  We record the index of the first occurrence of each
    # candidate count so we can slice the body into per-stage sections.
    n_to_idx: "dict[int, int]" = {}
    for i, line in enumerate(body_lines):
        m = _BROKERAGE_INITIAL_RE.search(line) or _BROKERAGE_PROGRESS_RE.search(line)
        if m and int(m.group(1)) not in n_to_idx:
            n_to_idx[int(m.group(1))] = i

    # --- Collect skip lines for the top-impact filter stages --------------
    top_section_details: "list[tuple[int, int, float, str, list[str]]]" = []
    if summary_entries:
        for n_before, n_after, cut_pct, check_name in sorted(
            summary_entries, key=lambda e: e[2], reverse=True
        )[:top_cuts]:
            start = n_to_idx.get(n_before, 0)
            # n_after may be 0 (final check eliminated all candidates) and
            # therefore absent from n_to_idx; fall back to end of body.
            end = n_to_idx.get(n_after, len(body_lines))
            section_skips = [
                line
                for line in body_lines[start : end + 1]
                if _BROKERAGE_SKIP_RE.search(line)
            ]
            top_section_details.append(
                (n_before, n_after, cut_pct, check_name, section_skips)
            )

    # --- Build output -----------------------------------------------------
    parts: "list[str]" = []

    # 1. Data-availability summary (analysis logs only)
    if include_data_avail:
        data_lines = [
            line for line in body_lines if _BROKERAGE_DATA_AVAIL_RE.search(line)
        ]
        if data_lines:
            parts.append("## Data availability")
            parts.extend(data_lines)

    # 2. Skip reasons for the highest-impact stages
    for n_before, n_after, cut_pct, check_name, skips in top_section_details:
        if not skips:
            continue
        parts.append(
            f"\n## Filter stage: {check_name}  "
            f"({cut_pct:.0f}% cut, {n_before}→{n_after} candidates)"
        )
        parts.extend(skips[:max_skip_lines_per_section])
        if len(skips) > max_skip_lines_per_section:
            parts.append(
                f"  ... ({len(skips) - max_skip_lines_per_section} more skip lines not shown)"
            )

    # 3. Problematic-site / user-queue-throttle lines
    problematic = [
        line for line in body_lines if _BROKERAGE_PROBLEMATIC_RE.search(line)
    ]
    if problematic:
        parts.append("\n## Problematic sites")
        parts.extend(problematic)

    # 4. Final outcome: selected sites or "no candidates"
    use_lines = [line for line in body_lines if _BROKERAGE_USE_SITE_RE.search(line)]
    # "no candidates" may appear either in the body (older log format) or after
    # the summary header (newer production log format) — search both.
    no_cand = [
        line
        for line in body_lines + summary_lines
        if re.search(r"\bno candidates\b", line, re.IGNORECASE)
    ]
    if use_lines or no_cand:
        parts.append("\n## Final selection")
        parts.extend(use_lines)
        parts.extend(no_cand)

    # 5. Full funnel summary (always)
    if summary_lines:
        parts.append("\n## " + "=" * 40)
        parts.extend(summary_lines)

    result = "\n".join(parts)
    logger.debug(
        "%s: %d raw lines → %d output lines (%.0f%% reduction)",
        fn_name,
        len(lines),
        len(result.splitlines()),
        100.0 * (1 - len(result.splitlines()) / max(len(lines), 1)),
    )
    return result


def filter_analysis_job_brokerage_log(
    log_text: str,
    top_cuts: int = 3,
    max_skip_lines_per_section: int = 10,
) -> "str | None":
    """Dedicated pre-filter for AtlasBroker-style analysis job brokerage logs.

    Brokerage logs follow a funnel structure: a sequence of filter stages
    progressively narrow a list of candidate sites, ending with a
    ``===== Job brokerage summary =====`` table.  The generic
    :func:`filter_log` treats this as a featureless text blob; this function
    understands the structure and extracts only what matters:

    1. **Data-availability lines** — replica/distribution status for the
       input dataset(s).
    2. **Skip-reason lines** for the ``top_cuts`` highest-impact filter stages
       (ranked by % of candidates removed).
    3. **Problematic-site lines** — user-queue throttle / bad-site notes that
       appear during the final weighting step.
    4. **Final-selection outcome** — ``use site=`` or ``no candidates`` lines.
    5. **Full summary table** — the ``N → M candidates, X% cut : <stage>``
       funnel printed at the end of every brokerage run.

    Args:
        log_text:                Raw brokerage log content.
        top_cuts:                Number of highest-% filter stages whose
                                 skip lines to include in the output.
        max_skip_lines_per_section: Hard cap on skip lines shown per stage
                                 (excess is replaced by an omitted-count note).

    Returns:
        Compact, structured string ready to send to the LLM, or ``None`` if
        the log doesn't look like a brokerage log — the caller should then
        fall back to :func:`filter_log`.
    """
    return _brokerage_filter_impl(
        log_text,
        top_cuts=top_cuts,
        max_skip_lines_per_section=max_skip_lines_per_section,
        include_data_avail=True,
        fn_name="filter_analysis_job_brokerage_log",
    )


def filter_prod_job_brokerage_log(
    log_text: str,
    top_cuts: int = 3,
    max_skip_lines_per_section: int = 10,
) -> "str | None":
    """Dedicated pre-filter for AtlasProdJobBroker-style production job brokerage logs.

    Identical in structure to :func:`filter_analysis_job_brokerage_log` but
    production logs do not contain data-availability lines, so section 1 is
    omitted.  The summary funnel, top-cut skip reasons, problematic sites, and
    final-selection outcome are all extracted in the same way.

    Args:
        log_text:                Raw production brokerage log content.
        top_cuts:                Number of highest-% filter stages whose
                                 skip lines to include in the output.
        max_skip_lines_per_section: Hard cap on skip lines shown per stage
                                 (excess is replaced by an omitted-count note).

    Returns:
        Compact, structured string ready to send to the LLM, or ``None`` if
        the log doesn't look like a brokerage log — the caller should then
        fall back to :func:`filter_log`.
    """
    return _brokerage_filter_impl(
        log_text,
        top_cuts=top_cuts,
        max_skip_lines_per_section=max_skip_lines_per_section,
        include_data_avail=False,
        fn_name="filter_prod_job_brokerage_log",
    )


# ---------------------------------------------------------------------------
# Source-name → specialised filter registry
# ---------------------------------------------------------------------------
# Maps a known log source name to its dedicated filter function.  When the
# source name is known at call time, ``filter_log_auto`` uses this mapping
# to skip the detection heuristics entirely — faster and more robust.
#
# Add new entries here when a new specialised filter is introduced.
# The value must be a callable with the same signature as
# ``filter_analysis_job_brokerage_log``: ``(log_text: str, **kwargs) -> str | None``.
SOURCE_FILTER_REGISTRY: dict[str, object] = {
    "analysis_job_brokerage_log": filter_analysis_job_brokerage_log,
    "prod_job_brokerage_log": filter_prod_job_brokerage_log,
}


def source_name_for_task(prod_source_label: str) -> str:
    """Return the :data:`SOURCE_FILTER_REGISTRY` key appropriate for a PanDA task.

    Args:
        prod_source_label: Value of ``task_data["prodSourceLabel"]``.

    Returns:
        ``"prod_job_brokerage_log"`` for managed (production) tasks,
        ``"analysis_job_brokerage_log"`` for everything else.
    """
    return (
        "prod_job_brokerage_log"
        if prod_source_label == "managed"
        else "analysis_job_brokerage_log"
    )


def filter_log_auto(
    log_text: str,
    source_name: "str | None" = None,
    context_lines: int = 3,
    max_lines: int = 300,
    head_lines: int = 20,
    tail_lines: int = 50,
) -> str:
    """Auto-dispatching log filter.

    If *source_name* is provided and matches an entry in
    :data:`SOURCE_FILTER_REGISTRY`, that specialised filter is called directly
    without running any detection heuristics.  If the specialised filter
    returns ``None`` (shouldn't happen for known sources, but guards against
    edge cases like empty/truncated content), the function falls back to
    :func:`filter_log`.

    When *source_name* is ``None`` or unknown, the function tries each
    registered specialised filter in insertion order, then falls back to
    :func:`filter_log`.

    To register a new log type, add it to :data:`SOURCE_FILTER_REGISTRY`
    without touching this function.
    """
    generic_kwargs = dict(
        context_lines=context_lines,
        max_lines=max_lines,
        head_lines=head_lines,
        tail_lines=tail_lines,
    )

    if source_name and source_name in SOURCE_FILTER_REGISTRY:
        fn = SOURCE_FILTER_REGISTRY[source_name]
        result = fn(log_text)
        if result is not None:
            return result
        # Edge case: known source but filter declined (e.g. empty log)
        return filter_log(log_text, **generic_kwargs)

    # Unknown source: try all specialised filters via detection heuristics
    for fn in SOURCE_FILTER_REGISTRY.values():
        result = fn(log_text)
        if result is not None:
            return result
    return filter_log(log_text, **generic_kwargs)


def filter_log(
    log_text: str,
    context_lines: int = 3,
    max_lines: int = 300,
    head_lines: int = 20,
    tail_lines: int = 50,
) -> str:
    """Reduce a raw log to a compact signal-only representation.

    The pipeline is:

    1. **Signal extraction** — keep only lines that match ``_SIGNAL_PATTERNS``
       plus ``context_lines`` before and after each match (like ``grep -B/-A``).
    2. **Deduplication** — normalise instance-specific tokens (timestamps,
       IDs, hostnames) and drop lines whose normalised form has already been
       seen.
    3. **Truncation** — if still over ``max_lines``, keep the first
       ``head_lines`` + last ``tail_lines`` of the filtered set, which
       preserves startup context and the final failure.

    Args:
        log_text:      Raw log content.
        context_lines: Lines of context to keep around each signal line.
        max_lines:     Maximum lines in the returned string.
        head_lines:    Lines to keep from the start if truncating.
        tail_lines:    Lines to keep from the end if truncating.

    Returns:
        Filtered log string ready to send to the LLM.  If the entire log
        is already below ``max_lines``, it is returned with only
        deduplication applied.
    """
    if not log_text or not log_text.strip():
        return ""

    lines = log_text.splitlines()

    # --- Step 1: mark signal lines and their context window ---------------
    signal_indices: set[int] = set()
    for i, line in enumerate(lines):
        if _SIGNAL_PATTERNS.search(line):
            for j in range(
                max(0, i - context_lines), min(len(lines), i + context_lines + 1)
            ):
                signal_indices.add(j)

    # If nothing matched (e.g. a pure debug log), fall back to the whole log.
    filtered = [lines[i] for i in sorted(signal_indices)] if signal_indices else lines

    # --- Step 2: deduplicate by normalised line ----------------------------
    seen: set[str] = set()
    deduped: list[str] = []
    for line in filtered:
        normalised = _NOISE_RE.sub("", line).strip()
        if normalised and normalised not in seen:
            seen.add(normalised)
            deduped.append(line)
        # blank / fully-normalised-away lines: keep at most one as separator
        elif not normalised and (not deduped or deduped[-1] != ""):
            deduped.append("")

    # --- Step 3: truncate if still too long --------------------------------
    if len(deduped) > max_lines:
        head = deduped[:head_lines]
        tail = deduped[-tail_lines:]
        omitted = len(deduped) - head_lines - tail_lines
        deduped = head + [f"... ({omitted} lines omitted) ..."] + tail

    result = "\n".join(deduped)
    logger.debug(
        "filter_log: %d raw lines → %d filtered lines (%.0f%% reduction)",
        len(lines),
        len(deduped),
        100.0 * (1 - len(deduped) / max(len(lines), 1)),
    )
    return result
