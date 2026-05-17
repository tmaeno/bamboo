"""Log pre-filtering utilities — reduce noise before sending to the LLM.

This module is intentionally separate from any extractor so it can be reused
by any pipeline stage that needs to compress raw log text.  New log-type
specialisations should be added here (new ``filter_<type>_log`` function +
constants) and registered in :func:`filter_log_auto`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

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

# Detailed skip-line parser: captures site name, free-text reason, and criteria token.
# Example match: "skip site=ANALY_CERN_T0_ART due to insufficient RAM less than ...
#                 criteria=-lowmemory"
_BROKERAGE_SKIP_DETAIL_RE: re.Pattern = re.compile(
    r"skip\s+site=(?P<site>\S+)\s+(?P<reason>.*?)\s*criteria=(?P<criteria>[-\w]+)",
    re.IGNORECASE,
)

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
    max_skipped_sites: int,
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

    # --- Collect skip lines for all filter stages -------------------------
    # All stages get a section header so the LLM sees the complete funnel
    # shape.  Skip lines are only collected for the top_cuts highest-impact
    # stages to keep the output concise; the rest receive a brief note.
    sorted_entries = sorted(summary_entries, key=lambda e: e[2], reverse=True)
    top_cut_names: "set[str]" = {
        check_name for _, _, _, check_name in sorted_entries[:top_cuts]
    }
    # Include=True: collect skip lines.  Include=False: header-only.
    all_section_details: "list[tuple[int, int, float, str, list[str], bool]]" = []
    for n_before, n_after, cut_pct, check_name in sorted_entries:
        start = n_to_idx.get(n_before, 0)
        # n_after may be 0 (final check eliminated all candidates) and
        # therefore absent from n_to_idx; fall back to end of body.
        end = n_to_idx.get(n_after, len(body_lines))
        include_skips = check_name in top_cut_names
        section_skips = (
            [
                line
                for line in body_lines[start : end + 1]
                if _BROKERAGE_SKIP_RE.search(line)
            ]
            if include_skips
            else []
        )
        all_section_details.append(
            (n_before, n_after, cut_pct, check_name, section_skips, include_skips)
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

    # 2. Filter stage sections — all stages get a header; skip lines only for top_cuts
    for (
        n_before,
        n_after,
        cut_pct,
        check_name,
        skips,
        include_skips,
    ) in all_section_details:
        parts.append(
            f"\n## Filter stage: {check_name}  "
            f"({cut_pct:.0f}% cut, {n_before}→{n_after} candidates)"
        )
        if skips:
            if n_before - n_after > max_skipped_sites:
                parts.append(
                    f"  ({n_before - n_after} sites skipped — too many to list individually)"
                )
            else:
                parts.extend(skips)
        elif include_skips:
            # We searched this stage but found no per-site skip lines —
            # sites were filtered silently (e.g. input data check).
            parts.append("  (no per-site skip lines — sites filtered implicitly)")
        else:
            # Stage is below the top_cuts threshold; skip lines not collected.
            parts.append(
                "  (skip lines not shown — not among the highest-impact stages)"
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
    max_skipped_sites: int = 20,
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
        max_skipped_sites: Hard cap on skip lines shown per stage
                                 (excess is replaced by an omitted-count note).

    Returns:
        Compact, structured string ready to send to the LLM, or ``None`` if
        the log doesn't look like a brokerage log — the caller should then
        fall back to :func:`filter_log`.
    """
    return _brokerage_filter_impl(
        log_text,
        top_cuts=top_cuts,
        max_skipped_sites=max_skipped_sites,
        include_data_avail=True,
        fn_name="filter_analysis_job_brokerage_log",
    )


def filter_prod_job_brokerage_log(
    log_text: str,
    top_cuts: int = 3,
    max_skipped_sites: int = 20,
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
        max_skipped_sites: Hard cap on skip lines shown per stage
                                 (excess is replaced by an omitted-count note).

    Returns:
        Compact, structured string ready to send to the LLM, or ``None`` if
        the log doesn't look like a brokerage log — the caller should then
        fall back to :func:`filter_log`.
    """
    return _brokerage_filter_impl(
        log_text,
        top_cuts=top_cuts,
        max_skipped_sites=max_skipped_sites,
        include_data_avail=False,
        fn_name="filter_prod_job_brokerage_log",
    )


def parse_brokerage_summary(
    log_text: str,
    sites_of_interest: "list[str] | None" = None,
) -> "dict | None":
    """Parse a PanDA brokerage log into a structured summary dict.

    Reads the funnel summary table at the bottom of the log and the per-site
    ``skip site=…`` lines in the body. Returns a dict with:

    - ``initial_candidates`` / ``final_candidates`` (int | None)
    - ``stages`` — list of {name, before, after, cut_pct, criteria_seen} in
      execution order (the summary table is already in execution order).
    - ``terminal_filter`` / ``terminal_cut_pct`` — last stage in execution order
      (= the stage that brought candidates to ``final_candidates``).
    - ``sites_of_interest`` — echoed input list.
    - ``sites_of_interest_fates`` — for each input site that has a matching
      skip line in the log: {site, filtered_at, criteria, reason}.

    Returns ``None`` when no ``===== Job brokerage summary =====`` section is
    found in the log; the caller should fall back to the raw log only.

    The function never raises and is safe to call on arbitrary text.
    """
    if not log_text or not _BROKERAGE_SUMMARY_HEADER_RE.search(log_text):
        return None
    sites_of_interest = list(sites_of_interest or [])

    lines = log_text.splitlines()

    # Locate the summary section: header line index.
    summary_start = next(
        (i for i, line in enumerate(lines) if _BROKERAGE_SUMMARY_HEADER_RE.search(line)),
        None,
    )
    if summary_start is None:
        return None
    body_lines = lines[:summary_start]
    summary_lines = lines[summary_start:]

    # Parse summary entries in their natural (execution) order.
    stages: list[dict] = []
    for line in summary_lines:
        m = _BROKERAGE_SUMMARY_ENTRY_RE.search(line)
        if m:
            stages.append({
                "name": m.group(4).strip(),
                "before": int(m.group(1)),
                "after": int(m.group(2)),
                "cut_pct": int(float(m.group(3))),
                "criteria_seen": [],
            })
    if not stages:
        return None

    initial_m = next(
        (_BROKERAGE_INITIAL_RE.search(line) for line in lines
         if _BROKERAGE_INITIAL_RE.search(line)),
        None,
    )
    # Final candidate count: also appears in the summary header block as
    # "the number of final candidates: N" or inferred from the last stage.
    _FINAL_RE = re.compile(r"final\s+candidates?:\s*(\d+)", re.IGNORECASE)
    final_m = next(
        (_FINAL_RE.search(line) for line in summary_lines if _FINAL_RE.search(line)),
        None,
    )

    # Map each candidate-count milestone to its line index in the body so we
    # can slice the body into per-stage sections. This mirrors the approach in
    # _brokerage_filter_impl.
    n_to_idx: dict[int, int] = {}
    for i, line in enumerate(body_lines):
        m = _BROKERAGE_INITIAL_RE.search(line) or _BROKERAGE_PROGRESS_RE.search(line)
        if m and int(m.group(1)) not in n_to_idx:
            n_to_idx[int(m.group(1))] = i

    # Walk skip lines once and attribute each to a stage. A skip line at body
    # index `j` belongs to the stage whose body slice contains `j`.
    seen_per_stage: dict[str, set[str]] = {s["name"]: set() for s in stages}
    fates: list[dict] = []
    for j, line in enumerate(body_lines):
        m = _BROKERAGE_SKIP_DETAIL_RE.search(line)
        if not m:
            continue
        # Find the stage owning this body index.
        owning_stage: dict | None = None
        for s in stages:
            start = n_to_idx.get(s["before"], 0)
            end = n_to_idx.get(s["after"], len(body_lines))
            if start <= j <= end:
                owning_stage = s
                break
        if owning_stage is None:
            continue
        criteria = m.group("criteria")
        seen_per_stage[owning_stage["name"]].add(criteria)
        site = m.group("site")
        if site in sites_of_interest and not any(f["site"] == site for f in fates):
            fates.append({
                "site": site,
                "filtered_at": owning_stage["name"],
                "criteria": criteria,
                "reason": m.group("reason").strip(),
            })

    for s in stages:
        s["criteria_seen"] = sorted(seen_per_stage[s["name"]])

    terminal = stages[-1]
    return {
        "initial_candidates": int(initial_m.group(1)) if initial_m else None,
        "final_candidates": int(final_m.group(1)) if final_m else terminal["after"],
        "stages": stages,
        "terminal_filter": terminal["name"],
        "terminal_cut_pct": terminal["cut_pct"],
        "sites_of_interest": sites_of_interest,
        "sites_of_interest_fates": fates,
    }


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


# ---------------------------------------------------------------------------
# Job-log pair comparison via structural alignment
# ---------------------------------------------------------------------------

# Verb hints suggesting a step-start banner.
_STEP_BANNER_RE: re.Pattern = re.compile(
    r"\b(start(?:ing)?|begin(?:ning)?|running|executing|launching|initiating)\b",
    re.IGNORECASE,
)

# Verb hints suggesting a step-end banner.
_STEP_END_RE: re.Pattern = re.compile(
    r"\b(end(?:ed)?|finish(?:ed)?|complete(?:d)?|done|exit(?:ed)?)\b",
    re.IGNORECASE,
)

# Timestamp at the start of a line, captured for duration computation.
_LEADING_TIMESTAMP_RE: re.Pattern = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
)

def _normalize_log_line(line: str) -> str:
    """Strip incident-specific tokens so two job-log lines compare structurally."""
    return _NOISE_RE.sub("", line).strip()


# Section-marker lines look like ``=== pre jobO ===`` or ``==== execute ====``.
# PanDA payload.stdout uses these to bracket workflow phases (pre/post jobO,
# CMake setup, execute, list in run dir, etc.). They're unique per log,
# stable across runs, and ideal as alignment anchors.
_SECTION_MARKER_RE: re.Pattern = re.compile(
    r"^\s*={3,}\s*(\S.*?\S|\S)\s*={3,}\s*$"
)


# Aggressive normalisation used ONLY for collapsing consecutive similar
# lines into a single representative before alignment (e.g. a MetaReader
# loop iterating over many input files). Strips URLs, paths, quoted
# strings, bracket lists, and file-extension tokens in addition to
# everything ``_NOISE_RE`` covers. Never applied to raw text fields
# surfaced to the LLM — those keep their original detail.
_COMPRESSION_NOISE_RE: re.Pattern = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\b\d{2}/\d{2}/\d{4}\b"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|0x[0-9a-fA-F]+"
    r"|\b\d{5,}\b"
    r"|\b[\w.-]+\.(cern\.ch|triumf\.ca|bnl\.gov|slac\.stanford\.edu|desy\.de)\b"
    r"|(?:root|https?|file|gsiftp)://[^\s'\"]+"
    r"|/(?:cvmfs|pnfs|eos|tmp|var|home|usr|opt)/[^\s'\"]+"
    r"|'[^']*'"
    r'|"[^"]*"'
    r"|\[[^\]]*\]"
    r"|\b[\w.-]+\.(?:data|json|xml|log|root|stdout|stderr|txt|py|so|cfg)\b",
    re.IGNORECASE,
)


def _compression_normalize_log_line(line: str) -> str:
    """Strong-normalise a log line for compression purposes (alignment only)."""
    return _COMPRESSION_NOISE_RE.sub("", line).strip()


def _compress_runs(
    lines: list[str],
) -> tuple[list[str], list[tuple[int, int]]]:
    """Run-length-encode consecutive lines that compression-normalise identically.

    A loop emitting the same log line for each iteration (e.g. file paths
    differ but get stripped by the compression noise regex) collapses to
    one entry, so alignment can bridge loops of different lengths.

    Note: more elaborate pattern detection (e.g. 2-cycle alternation)
    was tried but proved too aggressive — it let alignment leap across
    phase boundaries via text-identical but semantically-different lines
    in distant phases. Consecutive-duplicate compression is conservative
    and predictable.

    Returns ``(compressed_strings, compressed_to_orig)``. The k'th
    compressed entry covers original lines
    ``compressed_to_orig[k] = (start, end_inclusive)``.
    """
    compr_strings: list[str] = []
    compr_to_orig: list[tuple[int, int]] = []
    norms = [_compression_normalize_log_line(l) for l in lines]
    i, n = 0, len(norms)
    while i < n:
        j = i
        while j + 1 < n and norms[j + 1] == norms[i]:
            j += 1
        compr_strings.append(norms[i])
        compr_to_orig.append((i, j))
        i = j + 1
    return compr_strings, compr_to_orig


def _find_bottom_anchor(
    failed_norm: list[str],
    successful_norm: list[str],
    tail_k: int = 30,
    min_match_ratio: float = 0.8,
) -> "int | None":
    """Find the earliest position in ``successful_norm`` where the preceding
    ``tail_k`` lines closely match the last ``tail_k`` lines of
    ``failed_norm``.

    Used to refine the continuation start point when failed appears to be
    a *near-prefix* of successful (failed stopped mid-loop, successful
    continued). Top-down alignment may stop early inside the divergent
    loop body; the bottom-up anchor identifies where failed's actual
    content ends in successful's timeline.

    A high ``min_match_ratio`` keeps spurious matches in check: when
    failed has genuinely-different tail content (error / traceback that
    doesn't appear in successful), the score won't clear the bar and
    this function returns ``None``.

    Args:
        failed_norm, successful_norm: normalised line lists.
        tail_k: window size for the tail match (default 30).
        min_match_ratio: minimum fraction of matching lines (default 0.8).

    Returns:
        The anchor position (last matched line index in ``successful_norm``),
        or ``None`` if no high-confidence match is found.
    """
    n_fail, n_succ = len(failed_norm), len(successful_norm)
    if n_fail < tail_k or n_succ < tail_k:
        return None
    failed_tail = failed_norm[-tail_k:]
    last_failed = failed_tail[-1]
    for j in range(tail_k - 1, n_succ):
        if successful_norm[j] != last_failed:
            continue
        start = j - tail_k + 1
        match_count = sum(
            1
            for k in range(tail_k)
            if successful_norm[start + k] == failed_tail[k]
        )
        if match_count / tail_k >= min_match_ratio:
            return j
    return None


def _translate_markers_to_compressed(
    markers: list[tuple[int, str]],
    compressed_to_orig: list[tuple[int, int]],
) -> list[tuple[int, str]]:
    """Map ``(original_line_idx, marker_text)`` to the compressed-entry idx."""
    out: list[tuple[int, str]] = []
    cur = 0
    for orig_idx, text in markers:
        while (
            cur < len(compressed_to_orig)
            and compressed_to_orig[cur][1] < orig_idx
        ):
            cur += 1
        if cur >= len(compressed_to_orig):
            break
        start, end = compressed_to_orig[cur]
        if start <= orig_idx <= end:
            out.append((cur, text))
    return out


def _index_markers(lines: list[str]) -> list[tuple[int, str]]:
    """Return ``[(line_index, marker_text), ...]`` for every section-marker line.

    ``marker_text`` is the trimmed inner content (e.g. ``"pre jobO"`` from
    ``"=== pre jobO ==="``).
    """
    out: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _SECTION_MARKER_RE.match(line)
        if m:
            out.append((i, m.group(1).strip()))
    return out


def _parse_log_timestamp(ts: str) -> "datetime | None":
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts[:19], fmt)
        except ValueError:
            continue
    return None


def _align_prefix(
    failed_norm: list[str],
    successful_norm: list[str],
    failed_markers: list[tuple[int, str]],
    successful_markers: list[tuple[int, str]],
    window: int = 30,
    max_consecutive_skips: int = 20,
) -> tuple[int, int]:
    """Forward-only prefix alignment between two normalised log line sequences.

    Walks both sequences from index 0, advancing both pointers when the
    current lines match. On a mismatch:

      1. Window-search up to ``window`` lines ahead on each side. Whichever
         side finds the closer match wins; that pointer jumps to the match.
      2. If neither side has a window match, look for the next section
         marker (``=== text ===``) in each sequence at-or-after the current
         cursors. If the marker *text* matches on both sides, jump both
         pointers to those marker positions — this bridges divergent blocks
         like a PoolFileCatalog with different file lists.
      3. Otherwise tolerate a paired single-line skip. Once
         ``max_consecutive_skips`` paired skips accumulate without an
         anchor, roll back to the start of the run and stop alignment.

    Unlike :class:`difflib.SequenceMatcher`'s global longest-match, this
    can never leap past unmatched content speculatively. Phrases that
    recur later in ``successful`` (e.g. a generic "Stopping legacy file
    validation" line emitted in both INPUT and OUTPUT validation phases)
    cannot pull the alignment forward across a phase boundary.

    Returns ``(aligned_until_failed, aligned_until_successful)`` — the
    first index in each sequence that has NOT been aligned.
    """
    i, j = 0, 0
    n, m = len(failed_norm), len(successful_norm)
    consecutive_skips = 0
    fm_cursor = 0
    sm_cursor = 0
    while i < n and j < m:
        if failed_norm[i] == successful_norm[j]:
            i += 1
            j += 1
            consecutive_skips = 0
            continue
        found_j = next(
            (
                k for k in range(j + 1, min(j + window + 1, m))
                if successful_norm[k] == failed_norm[i]
            ),
            None,
        )
        found_i = next(
            (
                k for k in range(i + 1, min(i + window + 1, n))
                if failed_norm[k] == successful_norm[j]
            ),
            None,
        )
        if found_j is not None and (
            found_i is None or (found_j - j) <= (found_i - i)
        ):
            j = found_j
            consecutive_skips = 0
            continue
        if found_i is not None:
            i = found_i
            consecutive_skips = 0
            continue
        # Marker anchor: advance the cursors to the first marker
        # at-or-after the current alignment positions, then look for a
        # text match across the two remaining marker streams. We allow
        # one side to skip ahead (e.g. failed's first PoolFileCatalog is
        # 3× longer than successful's, putting failed's "GUIDs in PFC"
        # marker much later than successful's), but never backward.
        while (
            fm_cursor < len(failed_markers)
            and failed_markers[fm_cursor][0] < i
        ):
            fm_cursor += 1
        while (
            sm_cursor < len(successful_markers)
            and successful_markers[sm_cursor][0] < j
        ):
            sm_cursor += 1
        anchor_i: "int | None" = None
        anchor_j: "int | None" = None
        new_fm = fm_cursor
        new_sm = sm_cursor
        if (
            fm_cursor < len(failed_markers)
            and sm_cursor < len(successful_markers)
            and failed_markers[fm_cursor][1] == successful_markers[sm_cursor][1]
        ):
            anchor_i = failed_markers[fm_cursor][0]
            anchor_j = successful_markers[sm_cursor][0]
        elif sm_cursor < len(successful_markers):
            # Look for successful's next marker text later in failed.
            target = successful_markers[sm_cursor][1]
            for k in range(fm_cursor, len(failed_markers)):
                if failed_markers[k][1] == target:
                    anchor_i = failed_markers[k][0]
                    anchor_j = successful_markers[sm_cursor][0]
                    new_fm = k
                    break
        if anchor_i is None and fm_cursor < len(failed_markers):
            # Look for failed's next marker text later in successful.
            target = failed_markers[fm_cursor][1]
            for k in range(sm_cursor, len(successful_markers)):
                if successful_markers[k][1] == target:
                    anchor_i = failed_markers[fm_cursor][0]
                    anchor_j = successful_markers[k][0]
                    new_sm = k
                    break
        if anchor_i is not None and anchor_j is not None:
            i = anchor_i
            j = anchor_j
            fm_cursor = new_fm
            sm_cursor = new_sm
            consecutive_skips = 0
            continue
        # Neither side found an anchor — tolerate one paired skip.
        i += 1
        j += 1
        consecutive_skips += 1
        if consecutive_skips >= max_consecutive_skips:
            # Genuine divergence — roll back to the last confirmed match
            # so the caller sees the true alignment endpoint.
            i -= consecutive_skips
            j -= consecutive_skips
            break
    return i, j


def _truncate_long_lines(lines: list[str], max_line_chars: int) -> list[str]:
    """Cap each line to max_line_chars chars, marking the overflow.

    Lines under the cap pass through unchanged. Lines over the cap become
    ``<first max_line_chars chars> ...[N chars omitted]``. Used to keep raw
    log excerpts from including pathologically long single lines (serialised
    configs, base64 blobs) that bloat the LLM prompt without adding signal.
    """
    out: list[str] = []
    for line in lines:
        if len(line) > max_line_chars:
            omitted = len(line) - max_line_chars
            out.append(f"{line[:max_line_chars]} ...[{omitted:,} chars omitted]")
        else:
            out.append(line)
    return out


def _format_duration(start_ts: str, end_ts: str) -> "str | None":
    s = _parse_log_timestamp(start_ts)
    e = _parse_log_timestamp(end_ts)
    if s is None or e is None:
        return None
    delta = e - s
    sec = int(delta.total_seconds())
    if sec < 0:
        return None
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec // 60}m{sec % 60}s"
    return f"{sec // 3600}h{(sec % 3600) // 60}m"


_LOG_ERROR_RE: re.Pattern = re.compile(
    # Leading \b dropped so compound names like ``RuntimeError`` /
    # ``IOException`` match too. Trailing \b kept to avoid partial-word
    # matches like "errored" eating into the next char.
    r"(?:error|fatal|exception|traceback|critical|killed|segfault|aborted|panic)\b",
    re.IGNORECASE,
)


def summarize_log_content(
    log_text: str,
    *,
    tail_lines: int = 50,
    max_error_lines: int = 20,
    max_line_chars: int = 2000,
) -> dict:
    """Heuristic extraction of diagnostic signals from a job log.

    Returns a compact dict (~few KB) suitable for direct inclusion in an
    LLM prompt — vs the raw log which can be megabytes. Pure-function
    heuristic, no LLM call.

    Fields:
      - ``total_lines``, ``total_chars``: log size.
      - ``last_section_marker``: name of the last ``=== text ===`` line.
      - ``sections_seen``: ordered list of all marker names.
      - ``last_timestamp``: trailing ISO-like timestamp of the last
        timestamped line (or None).
      - ``last_meaningful_action``: trimmed text of the last non-blank
        line (truncated at ``max_line_chars``).
      - ``error_lines``: first ``max_error_lines`` lines matching common
        error patterns (ERROR/FATAL/Traceback/Exception/Killed/etc.).
      - ``tail_raw``: last ``tail_lines`` lines, individually capped at
        ``max_line_chars``.
    """
    lines = log_text.splitlines() if log_text else []
    total_lines = len(lines)
    total_chars = len(log_text) if log_text else 0

    sections_seen = [text for _, text in _index_markers(lines)]
    last_section_marker = sections_seen[-1] if sections_seen else None

    last_timestamp: "str | None" = None
    for line in reversed(lines):
        m = _LEADING_TIMESTAMP_RE.search(line)
        if m:
            last_timestamp = m.group(1)
            break

    last_meaningful_action: "str | None" = None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            last_meaningful_action = (
                stripped
                if len(stripped) <= max_line_chars
                else stripped[:max_line_chars] + " ...[truncated]"
            )
            break

    error_lines: list[str] = []
    for line in lines:
        if _LOG_ERROR_RE.search(line):
            text = (
                line
                if len(line) <= max_line_chars
                else line[:max_line_chars] + " ...[truncated]"
            )
            error_lines.append(text)
            if len(error_lines) >= max_error_lines:
                break

    tail_raw = "\n".join(
        _truncate_long_lines(lines[-tail_lines:], max_line_chars)
    )

    return {
        "total_lines": total_lines,
        "total_chars": total_chars,
        "last_section_marker": last_section_marker,
        "sections_seen": sections_seen,
        "last_timestamp": last_timestamp,
        "last_meaningful_action": last_meaningful_action,
        "error_lines": error_lines,
        "tail_raw": tail_raw,
    }


def compare_job_logs(
    failed: str,
    successful: str,
    *,
    tail_lines: int = 50,
    continuation_lines: int = 150,
    max_line_chars: int = 2000,
) -> dict:
    """Compare two payload.stdout logs via line-level structural alignment.

    Designed for the "failed job was killed mid-execution while a similar
    successful job completed" case. Normalises both logs (strips timestamps,
    UUIDs, host names, long numeric IDs) and walks them forward-only with
    :func:`_align_prefix` — never leaps past unmatched content, so a phrase
    that recurs late in ``successful`` cannot pull alignment across a phase
    boundary. Reports:

    - Whether the failed log appears to have been truncated mid-execution
      (alignment reaches the end of the failed log while the successful log
      still has content past the alignment).
    - A candidate "problem step" name — found either in the last ~30 aligned
      lines of the failed log (failed emitted a start banner before being
      killed) or in the first ~10 lines of the successful log past the
      alignment (failed was killed before emitting the banner).
    - The step's duration in the successful run if both a start banner
      (matched in the failed or successful log) and an end banner (matched
      in the successful log within ``continuation_lines``) carry parseable
      ISO timestamps.
    - The next ``continuation_lines`` raw lines of the successful log past
      the alignment point — what *should* have happened.
    - The last ``tail_lines`` raw lines of the failed log for context.

    Keys are ordered so the most actionable signals appear first in
    ``json.dumps(..., indent=2)`` output — the reasoning LLM reads top-down.

    Args:
        failed:             Raw payload.stdout text from the failed job.
        successful:         Raw payload.stdout text from a similar successful job.
        tail_lines:         Number of raw lines of ``failed`` to include in the
                            output (default 50).
        continuation_lines: Number of raw lines of ``successful`` past the
                            alignment point to include (default 150).
        max_line_chars:     Per-line length cap for the raw tail / continuation
                            text fields (default 2000). Lines longer than this
                            are truncated to the first ``max_line_chars`` chars
                            with a ``...[N chars omitted]`` marker. Bounds the
                            output size against pathologically long log lines
                            (serialised configs, base64 blobs).

    Returns:
        Dict with the schema documented above. Never raises; on malformed
        input returns a best-effort result with empty / zero fields.
    """
    failed_lines = failed.splitlines() if failed else []
    successful_lines = successful.splitlines() if successful else []

    # Forward-only prefix alignment over *run-length-compressed* sequences.
    # Consecutive lines that compression-normalise identically (e.g. a
    # MetaReader loop iterating over many input files) collapse into one
    # entry per side, so alignment can bridge loops of different lengths.
    # Section markers (``=== text ===``) serve as strong anchors for
    # structured divergences like PoolFileCatalog blocks.
    failed_markers = _index_markers(failed_lines)
    successful_markers = _index_markers(successful_lines)
    failed_compr, failed_c2o = _compress_runs(failed_lines)
    successful_compr, successful_c2o = _compress_runs(successful_lines)
    failed_compr_markers = _translate_markers_to_compressed(
        failed_markers, failed_c2o
    )
    successful_compr_markers = _translate_markers_to_compressed(
        successful_markers, successful_c2o
    )
    if failed_lines and successful_lines:
        ali_f, ali_s = _align_prefix(
            failed_compr,
            successful_compr,
            failed_compr_markers,
            successful_compr_markers,
        )
        aligned_until_failed = failed_c2o[ali_f - 1][1] + 1 if ali_f > 0 else 0
        aligned_until_successful = (
            successful_c2o[ali_s - 1][1] + 1 if ali_s > 0 else 0
        )
    else:
        aligned_until_failed, aligned_until_successful = 0, 0

    # Bottom-up anchor: when failed is a near-prefix of successful, this
    # finds the earliest position in successful where the preceding window
    # matches failed's tail with high confidence — accurately identifying
    # where failed's content stops in successful's timeline, even if
    # top-down alignment stopped earlier inside a divergent loop body.
    failed_compression_norm = [
        _compression_normalize_log_line(l) for l in failed_lines
    ]
    successful_compression_norm = [
        _compression_normalize_log_line(l) for l in successful_lines
    ]
    bottom_anchor = _find_bottom_anchor(
        failed_compression_norm, successful_compression_norm
    )
    # Continuation in successful starts at the bottom anchor when available,
    # otherwise at the top-down alignment endpoint.
    if bottom_anchor is not None:
        continuation_start_successful = bottom_anchor + 1
    else:
        continuation_start_successful = aligned_until_successful

    if bottom_anchor is not None:
        truncated = continuation_start_successful < len(successful_lines)
    else:
        successful_remaining = (
            len(successful_lines) - aligned_until_successful
        )
        truncated = (
            aligned_until_failed >= len(failed_lines)
            and successful_remaining > 0
            and len(failed_lines) > 0
        )

    # Candidate step name: prefer a start banner in the failed log's last ~30
    # aligned lines (failed emitted the banner before being killed); fall back
    # to the first ~10 lines of successful's continuation (failed was killed
    # before emitting the banner).
    candidate_step: "str | None" = None
    candidate_step_source: "str | None" = None  # "failed" or "successful"
    candidate_step_ts: "str | None" = None
    look_back = max(0, aligned_until_failed - 30)
    for i in range(aligned_until_failed - 1, look_back - 1, -1):
        if _STEP_BANNER_RE.search(failed_lines[i]):
            candidate_step = failed_lines[i].strip()
            candidate_step_source = "failed"
            m = _LEADING_TIMESTAMP_RE.search(failed_lines[i])
            if m:
                candidate_step_ts = m.group(1)
            break
    if candidate_step is None and truncated:
        look_forward = min(
            len(successful_lines), aligned_until_successful + 10
        )
        for j in range(aligned_until_successful, look_forward):
            if _STEP_BANNER_RE.search(successful_lines[j]):
                candidate_step = successful_lines[j].strip()
                candidate_step_source = "successful"
                m = _LEADING_TIMESTAMP_RE.search(successful_lines[j])
                if m:
                    candidate_step_ts = m.group(1)
                break

    # Step duration is always computed intra-successful-log so the result is
    # meaningful (failed-log timestamps belong to a different wall-clock run).
    # If the banner was found in the failed log, locate the equivalent banner
    # in the successful log near the alignment end.
    start_ts_in_successful: "str | None" = None
    if candidate_step_source == "successful":
        start_ts_in_successful = candidate_step_ts
    elif candidate_step_source == "failed":
        look_back_succ = max(0, aligned_until_successful - 30)
        for k in range(aligned_until_successful - 1, look_back_succ - 1, -1):
            if _STEP_BANNER_RE.search(successful_lines[k]):
                m = _LEADING_TIMESTAMP_RE.search(successful_lines[k])
                if m:
                    start_ts_in_successful = m.group(1)
                    break

    duration: "str | None" = None
    if candidate_step and start_ts_in_successful:
        search_start = (
            aligned_until_successful + 1
            if candidate_step_source == "successful"
            else aligned_until_successful
        )
        search_end = min(
            len(successful_lines),
            aligned_until_successful + continuation_lines,
        )
        for j in range(search_start, search_end):
            if _STEP_END_RE.search(successful_lines[j]):
                m = _LEADING_TIMESTAMP_RE.search(successful_lines[j])
                if m:
                    duration = _format_duration(
                        start_ts_in_successful, m.group(1)
                    )
                    if duration:
                        break

    failed_tail = "\n".join(
        _truncate_long_lines(failed_lines[-tail_lines:], max_line_chars)
    )
    cont_end = min(
        len(successful_lines),
        continuation_start_successful + continuation_lines,
    )
    successful_continuation = "\n".join(
        _truncate_long_lines(
            successful_lines[continuation_start_successful:cont_end],
            max_line_chars,
        )
    )

    return {
        "failed_log_appears_truncated_mid_execution": truncated,
        "candidate_problem_step_from_alignment": candidate_step,
        "candidate_step_duration_in_successful_run": duration,
        "successful_run_next_lines_after_failure_point": successful_continuation,
        "failed_log_tail_raw": failed_tail,
        "aligned_until_failed_line": aligned_until_failed,
        "aligned_until_successful_line": aligned_until_successful,
        "failed_log_total_lines": len(failed_lines),
        "successful_log_total_lines": len(successful_lines),
    }
