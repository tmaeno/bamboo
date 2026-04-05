"""PanDA job-data aggregation for the Bamboo knowledge pipeline.

:class:`PandaJobDataAggregator` converts a list of raw PanDA job attribute dicts
into a small number of stable, reusable graph nodes plus a set of raw signals
for further processing.

Design rationale
----------------
Storing one graph node per job would produce millions of ephemeral,
non-reusable nodes — a graph antipattern.  Instead, aggregation extracts the
*patterns* that are meaningful across many incidents:

* **Dominant discrete values** (site, transformation, queue) become
  :class:`~bamboo.models.graph_element.AggregatedJobFeatureNode` with the value holding
  the dominant item and its fraction, e.g. ``"computingSite=AGLT2(73%)"``.

* **Per-site failure rates** highlight whether failures are site-specific,
  e.g. ``"site_failure_rate=AGLT2:high(>50%)"``.

* **Continuous numeric values** (CPU time, actual memory) are bucketed using
  the same ``_BUCKETS`` thresholds as ``task_data`` continuous keys.

* **Error signals** from three distinct PanDA error channels:

  - *Pilot channel* (``pilotErrorCode`` / ``pilotErrorDiag``): failures
    detected by the pilot process (e.g. lost heartbeat, stage-in failure).
    Signals are prefixed ``"pilot:<code>"``.
  - *Payload channel* (``transExitCode``): non-zero exit codes from the
    transformation / user payload (e.g. Athena crash).
    Signals are prefixed ``"payload:<code>"``.
  - *DDM channel* (``ddmErrorCode`` / ``ddmErrorDiag``): data-management
    failures reported by Rucio/DDM (e.g. missing replicas, quota exceeded).
    Signals are prefixed ``"ddm:<code>"``.

  Both channels are collected as raw strings for the caller to pass through
  :class:`~bamboo.agents.extractors.panda_knowledge_extractor.ErrorCategoryClassifier`,
  which maps them to canonical :class:`~bamboo.models.graph_element.SymptomNode`
  names.

* **Representative error diag texts** (top-N distinct diagnostic strings,
  one per dominant error code per channel) are collected as ``context_texts``
  for :class:`~bamboo.models.graph_element.TaskContextNode` (vector DB only),
  preserving the semantic richness of raw messages for similarity search.

Output
------
:meth:`PandaJobDataAggregator.aggregate` returns a :class:`JobAggregationResult`
dataclass.  The caller (``PandaKnowledgeExtractor._extract_from_jobs``) is
responsible for:

1. Instantiating :class:`AggregatedJobFeatureNode` from ``feature_items``.
2. Classifying each string in ``error_signals`` through the error classifier
   to produce :class:`SymptomNode` instances.
3. Creating :class:`TaskContextNode` instances from ``context_texts`` for
   vector DB indexing.

The aggregator is intentionally stateless and has no database dependency —
it is a pure Python data transformation step.

Constants
---------
``JOB_DISCRETE_KEYS``
    Job-attribute keys whose dominant value is worth a ``AggregatedJobFeatureNode``.
``JOB_CONTINUOUS_KEYS``
    Job-attribute keys whose numeric value is bucketed before storage.
``_JOB_BUCKETS``
    Bucket definitions for ``JOB_CONTINUOUS_KEYS``, aligned with PanDA units.
``MAX_CONTEXT_TEXTS``
    Maximum number of representative ``errorDiag`` strings forwarded to the
    vector DB.  Capped to avoid overwhelming the LLM/embedding step.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bamboo.utils.sanitize import pseudonymise_dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys whose dominant value becomes a AggregatedJobFeatureNode via _aggregate_discrete.
# Keys handled by dedicated aggregation steps are excluded here:
#   - computingSite                          → _aggregate_site_failure_rates
#   - jobStatus                              → failure counting (not a feature node)
#   - pilotErrorCode, pilotErrorDiag         → _aggregate_error_signals (pilot channel)
#   - transExitCode                          → _aggregate_error_signals (payload channel)
#   - ddmErrorCode, ddmErrorDiag             → _aggregate_error_signals (ddm channel)
# ---------------------------------------------------------------------------
JOB_DISCRETE_KEYS: frozenset[str] = frozenset(
    {
        "gshare",
        "processingType",
        "prodSourceLabel",
        "resourceType",
        "workQueue",
    }
)

# ---------------------------------------------------------------------------
# Keys whose numeric value is bucketed into a range label.
# Units follow PanDA conventions (seconds for time, MB for memory).
# ---------------------------------------------------------------------------
JOB_CONTINUOUS_KEYS: frozenset[str] = frozenset(
    {
        "cpuConsumptionTime",
        "actualCoreCount",
        "maxRSS",
        "outputFileBytes",
        "inputFileBytes",
    }
)

# Bucket definitions: sorted list of (upper_bound_exclusive, label).
# The last entry's upper_bound must be math.inf.
_JOB_BUCKETS: dict[str, list[tuple[float, str]]] = {
    # CPU time: seconds
    "cpuConsumptionTime": [
        (3_600, "<1h"),
        (21_600, "1-6h"),
        (86_400, "6-24h"),
        (float("inf"), ">24h"),
    ],
    # Core count: integer
    "actualCoreCount": [
        (2, "1-core"),
        (5, "2-4-cores"),
        (9, "4-8-cores"),
        (17, "8-16-cores"),
        (float("inf"), ">16-cores"),
    ],
    # Memory: MB
    "maxRSS": [
        (512, "<512MB"),
        (2_048, "512MB-2GB"),
        (8_192, "2-8GB"),
        (float("inf"), ">8GB"),
    ],
    # File bytes: bytes → GB buckets
    "outputFileBytes": [
        (1_073_741_824, "<1GB"),
        (10_737_418_240, "1-10GB"),
        (float("inf"), ">10GB"),
    ],
    "inputFileBytes": [
        (1_073_741_824, "<1GB"),
        (10_737_418_240, "1-10GB"),
        (float("inf"), ">10GB"),
    ],
}

# Maximum number of representative error messages forwarded to vector DB.
MAX_CONTEXT_TEXTS: int = 5

# Failure-rate thresholds for the bucketed site_failure_rate feature.
_FAILURE_RATE_BUCKETS: list[tuple[float, str]] = [
    (0.2, "low(<20%)"),
    (0.5, "medium(20-50%)"),
    (0.8, "high(50-80%)"),
    (float("inf"), "very_high(>80%)"),
]


def _bucket_job_value(key: str, raw: Any) -> str:
    """Map a raw numeric value to a bucketed label.

    Args:
        key: The job attribute name (must be in ``JOB_CONTINUOUS_KEYS``).
        raw: The raw value (numeric or string representation of a number).

    Returns:
        A bucketed label string, e.g. ``"1-6h"``.  If *raw* cannot be parsed
        as a float, the original string representation is returned unchanged.

    Raises:
        ValueError: If *key* has no bucket definition.
    """
    buckets = _JOB_BUCKETS.get(key)
    if buckets is None:
        raise ValueError(f"No bucket definition for job key '{key}'")
    try:
        numeric = float(raw)
    except (ValueError, TypeError):
        return str(raw)
    for upper, label in buckets:
        if numeric < upper:
            return label
    return buckets[-1][1]


def _bucket_failure_rate(rate: float) -> str:
    """Map a failure rate in [0, 1] to a human-readable label."""
    for upper, label in _FAILURE_RATE_BUCKETS:
        if rate < upper:
            return label
    return _FAILURE_RATE_BUCKETS[-1][1]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class JobAggregationResult:
    """Output of :meth:`PandaJobDataAggregator.aggregate`.

    Attributes:
        feature_items:  List of ``(attribute, value, job_count)`` tuples
                        ready to be turned into
                        :class:`~bamboo.models.graph_element.AggregatedJobFeatureNode`.
        error_signals:  Raw error strings prefixed by source channel
                        (e.g. ``"pilot:1099"``, ``"payload:139"``) for the
                        caller to classify into
                        :class:`~bamboo.models.graph_element.SymptomNode`.
                        The prefix encodes the error origin so no separate
                        ComponentNode is needed.
        context_texts:  Representative diagnostic strings for
                        :class:`~bamboo.models.graph_element.TaskContextNode`
                        (vector DB only).
        total_jobs:     Total number of job records processed.
        failed_jobs:    Number of jobs with ``jobStatus != "finished"``.
    """

    feature_items: list[tuple[str, str, int]] = field(default_factory=list)
    error_signals: list[str] = field(default_factory=list)
    context_texts: list[str] = field(default_factory=list)
    total_jobs: int = 0
    failed_jobs: int = 0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class PandaJobDataAggregator:
    """Aggregates a list of raw PanDA job dicts into stable, reusable knowledge signals.

    The aggregator is stateless and has no database dependency.  Instantiate
    once and call :meth:`aggregate` for each batch of jobs.

    Args:
        max_context_texts:     Maximum number of representative diagnostic
                               strings forwarded to the vector DB.
        min_dominant_fraction: Minimum fraction a value must represent to be
                               emitted as a dominant ``AggregatedJobFeatureNode``.
                               Values below this are too scattered to be
                               meaningful.
    """

    def __init__(
        self,
        max_context_texts: int = MAX_CONTEXT_TEXTS,
        min_dominant_fraction: float = 0.10,
    ) -> None:
        self._max_context = max_context_texts
        self._min_fraction = min_dominant_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, jobs_data: list[dict[str, Any]]) -> JobAggregationResult:
        """Aggregate *jobs_data* into a :class:`JobAggregationResult`.

        Args:
            jobs_data: List of job attribute dicts.  Missing or ``None``
                       values are skipped gracefully.

        Returns:
            :class:`JobAggregationResult` with feature items, error signals,
            and representative context texts.
        """
        if not jobs_data:
            return JobAggregationResult()

        # Pseudonymise identity fields in every job record before any
        # aggregation step can store or forward them.  This mirrors the
        # same protection applied to task_data in PandaKnowledgeExtractor.
        jobs_data = [pseudonymise_dict(job) for job in jobs_data]

        total = len(jobs_data)
        failed = sum(1 for j in jobs_data if str(j.get("jobStatus", "")) != "finished")

        result = JobAggregationResult(total_jobs=total, failed_jobs=failed)

        self._aggregate_discrete(jobs_data, total, result)
        self._aggregate_continuous(jobs_data, total, result)
        self._aggregate_site_failure_rates(jobs_data, total, result)
        self._aggregate_error_signals(jobs_data, result)

        logger.debug(
            "PandaJobDataAggregator: %d jobs → %d feature items, %d error signals, "
            "%d context texts",
            total,
            len(result.feature_items),
            len(result.error_signals),
            len(result.context_texts),
        )
        return result

    # ------------------------------------------------------------------
    # Private aggregation steps
    # ------------------------------------------------------------------

    def _aggregate_discrete(
        self,
        jobs_data: list[dict[str, Any]],
        total: int,
        result: JobAggregationResult,
    ) -> None:
        """Emit a AggregatedJobFeatureNode for the dominant value of each discrete key.

        Only keys in :data:`JOB_DISCRETE_KEYS` are processed here.  Keys
        handled by dedicated aggregation steps (error channels, site failure
        rates) are intentionally absent from ``JOB_DISCRETE_KEYS`` and are
        therefore never reached here.
        """
        for key in JOB_DISCRETE_KEYS:
            counter: Counter[str] = Counter()
            for job in jobs_data:
                val = job.get(key)
                if val is not None and str(val).strip():
                    counter[str(val)] += 1
            if not counter:
                continue
            dominant, count = counter.most_common(1)[0]
            fraction = count / total
            if fraction < self._min_fraction:
                continue
            result.feature_items.append((key, dominant, count))

    def _aggregate_continuous(
        self,
        jobs_data: list[dict[str, Any]],
        total: int,
        result: JobAggregationResult,
    ) -> None:
        """Emit a AggregatedJobFeatureNode for the most common bucketed range per continuous key."""
        for key in JOB_CONTINUOUS_KEYS:
            counter: Counter[str] = Counter()
            for job in jobs_data:
                val = job.get(key)
                if val is None:
                    continue
                bucketed = _bucket_job_value(key, val)
                counter[bucketed] += 1
            if not counter:
                continue
            dominant, count = counter.most_common(1)[0]
            fraction = count / total
            if fraction < self._min_fraction:
                continue
            result.feature_items.append((key, dominant, count))

    def _aggregate_site_failure_rates(
        self,
        jobs_data: list[dict[str, Any]],
        total: int,
        result: JobAggregationResult,
    ) -> None:
        """Emit per-site failure-rate AggregatedJobFeatureNodes for sites with enough data.

        Only sites that account for at least ``_min_fraction`` of all jobs are
        considered.  Sites whose failure rate is low are skipped (they are not
        the problem) — only sites above 20% failure rate produce a node.

        Also emits a ``computingSite`` dominant-value node for the most common
        site overall (regardless of failure rate).
        """
        site_total: Counter[str] = Counter()
        site_failed: Counter[str] = Counter()
        for job in jobs_data:
            site = str(job.get("computingSite", "")).strip()
            if not site:
                continue
            site_total[site] += 1
            if str(job.get("jobStatus", "")) != "finished":
                site_failed[site] += 1

        # Dominant overall site
        if site_total:
            top_site, top_count = site_total.most_common(1)[0]
            fraction = top_count / total
            if fraction >= self._min_fraction:
                result.feature_items.append(("computingSite", top_site, top_count))

        # Per-site failure rates
        for site, s_total in site_total.items():
            if s_total / total < self._min_fraction:
                continue
            rate = site_failed.get(site, 0) / s_total
            if rate < 0.20:  # low-failure sites are not the problem
                continue
            label = f"{site}:{_bucket_failure_rate(rate)}"
            result.feature_items.append(("site_failure_rate", label, s_total))

    def _aggregate_error_signals(
        self,
        jobs_data: list[dict[str, Any]],
        result: JobAggregationResult,
    ) -> None:
        """Collect error signals from pilot and payload error channels.

        PanDA jobs carry three distinct error channels:

        * **Pilot channel** — ``pilotErrorCode`` / ``pilotErrorDiag``:
          failures detected by the pilot (e.g. lost heartbeat, stage-in
          failure, CPU limit exceeded).
        * **Payload channel** — ``transExitCode``:
          non-zero exit codes from the transformation / payload execution
          (e.g. Athena crash, user script failure).
        * **DDM channel** — ``ddmErrorCode`` / ``ddmErrorDiag``:
          data-management failures reported by Rucio/DDM (e.g. replication
          errors, missing replicas, quota exceeded).

        For each channel, the dominant error code is added to
        ``error_signals`` for the caller to classify via
        :class:`~bamboo.agents.extractors.panda_knowledge_extractor.ErrorCategoryClassifier`.
        One representative diagnostic string per dominant code is added to
        ``context_texts`` (vector DB only, up to ``MAX_CONTEXT_TEXTS``).

        Error code ``"0"`` is always skipped (success / no error).
        """
        channels: list[tuple[str, str, str | None]] = [
            # (source_label, code_key, diag_key)
            ("pilot", "pilotErrorCode", "pilotErrorDiag"),
            ("payload", "transExitCode", None),
            ("ddm", "ddmErrorCode", "ddmErrorDiag"),
        ]

        seen_diags: set[str] = set()

        for source, code_key, diag_key in channels:
            pairs: list[tuple[str, str]] = []  # (code, diag)
            for job in jobs_data:
                if str(job.get("jobStatus", "")) == "finished":
                    continue
                code = job.get(code_key)
                if code is None or not str(code).strip() or str(code) == "0":
                    continue
                diag = str(job.get(diag_key, "")).strip() if diag_key else ""
                pairs.append((str(code), diag))

            if not pairs:
                continue

            code_counter: Counter[str] = Counter(code for code, _ in pairs)

            # Dominant error codes → error_signals (prefixed with source)
            for code, _ in code_counter.most_common():
                signal = f"{source}:{code}"
                if signal not in result.error_signals:
                    result.error_signals.append(signal)

            # One representative diag per dominant code → context_texts
            for code, _ in code_counter.most_common():
                for c, diag in pairs:
                    if c == code and diag and diag not in seen_diags:
                        seen_diags.add(diag)
                        result.context_texts.append(diag)
                        break
                if len(result.context_texts) >= self._max_context:
                    break
