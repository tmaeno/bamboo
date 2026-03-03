"""Job-data aggregation for the Bamboo knowledge pipeline.

:class:`JobDataAggregator` converts a list of raw PanDA job attribute dicts
into a small number of stable, reusable graph nodes plus a set of raw signals
for further processing.

Design rationale
----------------
Storing one graph node per job would produce millions of ephemeral,
non-reusable nodes — a graph anti-pattern.  Instead, aggregation extracts the
*patterns* that are meaningful across many incidents:

* **Dominant discrete values** (site, transformation, queue) become
  :class:`~bamboo.models.graph_element.JobFeatureNode` with the value holding
  the dominant item and its fraction, e.g. ``"computingSite=AGLT2(73%)"``.

* **Per-site failure rates** highlight whether failures are site-specific,
  e.g. ``"site_failure_rate=AGLT2:high(>50%)"``.

* **Continuous numeric values** (CPU time, actual memory) are bucketed using
  the same ``_BUCKETS`` thresholds as ``task_data`` continuous keys.

* **Error signals** (``errorCode``, ``batchErrorCode``, ``errorDiag``) are
  collected as raw strings for the caller to pass through
  :class:`~bamboo.extractors.panda_knowledge_extractor.ErrorCategoryClassifier`,
  which maps them to canonical :class:`~bamboo.models.graph_element.SymptomNode`
  names.

* **Component signals** (pilot version, worker type) become
  :class:`~bamboo.models.graph_element.ComponentNode` values for the caller.

* **Representative error diag texts** (top-N distinct ``errorDiag`` strings
  from the most-failing sites) are collected as ``context_texts`` for
  :class:`~bamboo.models.graph_element.TaskContextNode` (vector DB only),
  preserving the semantic richness of raw messages for similarity search.

Output
------
:meth:`JobDataAggregator.aggregate` returns a :class:`JobAggregationResult`
dataclass.  The caller (``PandaKnowledgeExtractor._extract_from_jobs``) is
responsible for:

1. Instantiating :class:`JobFeatureNode` from ``feature_nodes``.
2. Classifying each string in ``error_signals`` through the error classifier
   to produce :class:`SymptomNode` instances.
3. Creating :class:`ComponentNode` instances from ``component_signals``.
4. Creating :class:`TaskContextNode` instances from ``context_texts`` for
   vector DB indexing.

The aggregator is intentionally stateless and has no database dependency —
it is a pure Python data transformation step.

Constants
---------
``JOB_DISCRETE_KEYS``
    Job-attribute keys whose dominant value is worth a ``JobFeatureNode``.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys whose dominant value becomes a JobFeatureNode.
# Excludes per-job GUIDs (PandaID, jobsetID, jediTaskID) and timestamps.
# ---------------------------------------------------------------------------
JOB_DISCRETE_KEYS: frozenset[str] = frozenset(
    {
        "computingSite",
        "cloud",
        "queue",
        "transformation",
        "processingType",
        "prodSourceLabel",
        "jobStatus",
        "workQueue",
        "resourceType",
        "coreCount",
        "pilotVersion",
        "workerType",
        "batchErrorCode",
        "errorCode",
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
        "actualRamCount",
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
        (9, "2-8-cores"),
        (float("inf"), ">8-cores"),
    ],
    # Memory: MB
    "actualRamCount": [
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
    """Output of :meth:`JobDataAggregator.aggregate`.

    Attributes:
        feature_items:     List of ``(attribute, value, job_count)`` tuples
                           ready to be turned into
                           :class:`~bamboo.models.graph_element.JobFeatureNode`.
        error_signals:     Raw error strings (error codes + ``errorDiag``
                           snippets) for the caller to classify into
                           :class:`~bamboo.models.graph_element.SymptomNode`.
        component_signals: ``(name, system)`` pairs for the caller to create
                           :class:`~bamboo.models.graph_element.ComponentNode`.
        context_texts:     Representative raw ``errorDiag`` strings for
                           :class:`~bamboo.models.graph_element.TaskContextNode`
                           (vector DB only).
        total_jobs:        Total number of job records processed.
        failed_jobs:       Number of jobs with ``jobStatus != "finished"``.
    """

    feature_items: list[tuple[str, str, int]] = field(default_factory=list)
    error_signals: list[str] = field(default_factory=list)
    component_signals: list[tuple[str, str]] = field(default_factory=list)
    context_texts: list[str] = field(default_factory=list)
    total_jobs: int = 0
    failed_jobs: int = 0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class JobDataAggregator:
    """Aggregates a list of raw job dicts into stable, reusable knowledge signals.

    The aggregator is stateless and has no database dependency.  Instantiate
    once and call :meth:`aggregate` for each batch of jobs.

    Args:
        max_context_texts:  Maximum number of representative ``errorDiag``
                            strings forwarded to the vector DB.
        min_dominant_fraction: Minimum fraction a value must represent to be
                            emitted as a dominant ``JobFeatureNode``.  Values
                            below this are too scattered to be meaningful.
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
            component signals, and representative context texts.
        """
        if not jobs_data:
            return JobAggregationResult()

        total = len(jobs_data)
        failed = sum(
            1 for j in jobs_data if str(j.get("jobStatus", "")) != "finished"
        )

        result = JobAggregationResult(total_jobs=total, failed_jobs=failed)

        self._aggregate_discrete(jobs_data, total, result)
        self._aggregate_continuous(jobs_data, total, result)
        self._aggregate_site_failure_rates(jobs_data, total, result)
        self._aggregate_error_signals(jobs_data, result)
        self._aggregate_component_signals(jobs_data, result)

        logger.debug(
            "JobDataAggregator: %d jobs → %d feature items, %d error signals, "
            "%d component signals, %d context texts",
            total,
            len(result.feature_items),
            len(result.error_signals),
            len(result.component_signals),
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
        """Emit a JobFeatureNode for the dominant value of each discrete key.

        Keys that are better handled by dedicated logic (``errorCode``,
        ``batchErrorCode``, ``pilotVersion``, ``computingSite``,
        ``jobStatus``) are excluded here — they are processed in other steps.
        """
        skip_here = {
            "errorCode", "batchErrorCode", "pilotVersion",
            "computingSite", "jobStatus",
        }
        for key in JOB_DISCRETE_KEYS - skip_here:
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
            label = f"{dominant}({int(fraction * 100)}%)"
            result.feature_items.append((key, label, count))

    def _aggregate_continuous(
        self,
        jobs_data: list[dict[str, Any]],
        total: int,
        result: JobAggregationResult,
    ) -> None:
        """Emit a JobFeatureNode for the most common bucketed range per continuous key."""
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
        """Emit per-site failure-rate JobFeatureNodes for sites with enough data.

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
                result.feature_items.append(
                    ("computingSite", f"{top_site}({int(fraction * 100)}%)", top_count)
                )

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
        """Collect distinct error codes and representative errorDiag strings.

        Error codes (``errorCode``, ``batchErrorCode``) are emitted as raw
        strings in ``error_signals`` for the caller to classify via
        :class:`~bamboo.extractors.panda_knowledge_extractor.ErrorCategoryClassifier`.

        Representative ``errorDiag`` texts — one per dominant error code —
        are added to ``context_texts`` (vector DB only, up to
        ``MAX_CONTEXT_TEXTS``).
        """
        # Collect (error_code, error_diag) pairs from failing jobs
        error_pairs: list[tuple[str, str]] = []
        for job in jobs_data:
            if str(job.get("jobStatus", "")) == "finished":
                continue
            for code_key in ("errorCode", "batchErrorCode"):
                code = job.get(code_key)
                if code is not None and str(code).strip() and str(code) != "0":
                    diag = str(job.get("errorDiag", "")).strip()
                    error_pairs.append((str(code), diag))

        # Dominant error codes
        code_counter: Counter[str] = Counter(code for code, _ in error_pairs)
        for code, _ in code_counter.most_common():
            if code not in result.error_signals:
                result.error_signals.append(code)

        # Representative errorDiag texts — one per dominant code, deduped
        seen_diags: set[str] = set()
        for code, _ in code_counter.most_common():
            for c, diag in error_pairs:
                if c == code and diag and diag not in seen_diags:
                    seen_diags.add(diag)
                    result.context_texts.append(diag)
                    break
            if len(result.context_texts) >= self._max_context:
                break

    def _aggregate_component_signals(
        self,
        jobs_data: list[dict[str, Any]],
        result: JobAggregationResult,
    ) -> None:
        """Emit component signals for dominant pilot version and worker type.

        ``pilotVersion`` → ComponentNode(system="PanDA pilot")
        ``workerType``   → ComponentNode(system="computing grid")
        """
        component_sources = {
            "pilotVersion": "PanDA pilot",
            "workerType": "computing grid",
        }
        for key, system in component_sources.items():
            counter: Counter[str] = Counter()
            for job in jobs_data:
                val = str(job.get(key, "")).strip()
                if val:
                    counter[val] += 1
            if not counter:
                continue
            dominant, _ = counter.most_common(1)[0]
            result.component_signals.append((dominant, system))

