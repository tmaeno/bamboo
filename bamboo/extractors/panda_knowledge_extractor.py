"""PanDA knowledge extraction strategy for structured task/external data.

Design
------
* ``external_data``  – treated as a flat key→value dictionary; every pair
  becomes a :class:`~bamboo.models.graph_element.TaskFeatureNode`.

* ``task_data``      – key→value dict whose fields are classified as either:
    - **discrete / categorical** → :class:`TaskFeatureNode`
    - **unstructured / free-form** → :class:`TaskContextNode`

  The special ``errorDialog`` field produces a :class:`SymptomNode` whose
  ``name`` is the canonical error category (clean, stable, reusable across
  incidents) and whose ``description`` is the verbatim raw message (preserved
  for traceability and vector search).  No ``TaskContextNode`` is created —
  ``TaskContextNode`` is not stored in the graph DB so it cannot participate
  in graph relationships, and storing the raw noisy text in the vector DB
  would pollute the embedding space.  Instead, ``SymptomNode.description``
  is indexed in the vector DB by the knowledge accumulator, giving semantic
  search over raw messages while the graph uses the stable canonical name.

* ``email_text``     – passed to the LLM to extract :class:`CauseNode`,
  :class:`ResolutionNode`, and :class:`TaskContextNode` nodes.  The LLM is
  instructed to emit *only* those three types so they complement (rather than
  overlap with) the structured nodes derived from ``task_data`` and
  ``external_data``.  Cause and Resolution names are canonicalised using the
  same VectorDB + LLM pattern as error categories, so the same concept always
  maps to the same canonical name regardless of how it is worded.

Persistence
-----------
All canonical names (error categories, causes, resolutions) are stored in the
same Qdrant collection, partitioned by ``section`` in the payload:
  - ``"canonical_node::ErrorCategory"``
  - ``"canonical_node::Cause"``
  - ``"canonical_node::Resolution"``

The store starts empty and grows organically as incidents are processed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Optional

from bamboo.extractors.base import ExtractionStrategy
from bamboo.llm import EMAIL_EXTRACTION_PROMPT, get_embeddings, get_llm
from bamboo.llm.prompts import (
    CAUSE_RESOLUTION_CANONICALIZE_PROMPT,
    ERROR_CATEGORY_LABEL_PROMPT,
)
from bamboo.models.graph_element import (
    CauseNode,
    ComponentNode,
    GraphRelationship,
    RelationType,
    ResolutionNode,
    SymptomNode,
    TaskContextNode,
    TaskFeatureNode,
)
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys in task_data that carry free-form prose (→ TaskContextNode).
# ---------------------------------------------------------------------------
UNSTRUCTURED_TASK_KEYS: frozenset[str] = frozenset(
    {
        "taskName",
        # job_execution_params is a free-form, incident-specific string with no
        # reliable structure.  Parsing it into graph nodes would produce noise;
        # storing it whole as a TaskContextNode lets the vector DB surface
        # semantically similar incidents.
        "job_execution_params",
        # Note: "errorDialog" is handled separately — it produces a
        # SymptomNode (canonical category name, raw message as description)
        # rather than a TaskContextNode.
        # Note: "task_creation_arguments" is handled separately — its
        # CLI-argument string is parsed into individual TaskFeatureNodes /
        # TaskContextNodes by the CLI_ARGUMENT_KEYS branch.
    }
)

# ---------------------------------------------------------------------------
# Keys in task_data that carry discrete / categorical values (→ TaskFeatureNode).
# Any key that is not in DISCRETE_TASK_KEYS, UNSTRUCTURED_TASK_KEYS, or the
# special reserved keys will be logged as unknown and skipped, preventing
# accidental indexing of unexpected blobs or nested structures.
#
# Notes:
# - "taskID" must NOT appear here; it is the unique incident identifier used
#   as graph_id by the knowledge accumulator and carries no comparative value
#   as a task attribute.
# - "taskName" must NOT appear here; it belongs only in UNSTRUCTURED_TASK_KEYS.
# - "status" must NOT appear here; it is routed to the SymptomNode branch
#   because it describes the task's failure state (e.g. "failed", "broken"),
#   not a task configuration attribute.
# - "splitRule" is handled by a dedicated branch that parses its pipe-separated
#   "key=value|key=value" format into one TaskFeatureNode per sub-rule.
# - Continuous numeric fields (ramCount, walltime, diskIO, etc.) must NOT
#   appear here; they belong in CONTINUOUS_TASK_KEYS and are bucketed into
#   range labels before becoming TaskFeatureNodes.
# ---------------------------------------------------------------------------
DISCRETE_TASK_KEYS: frozenset[str] = frozenset(
    {
        "userName",
        "prodSourceLabel",
        "workingGroup",
        "vo",
        "coreCount",
        "taskType",
        "processingType",
        "taskPriority",
        "architecture",
        "transUses",
        "transHome",
        "transPath",
        "walltimeUnit",
        "outDiskUnit",
        "workDiskUnit",
        "ramUnit",
        "ioIntensityUnit",
        "reqID",
        "site",
        "countryGroup",
        "campaign",
        "goal",
        "cpuTimeUnit",
        "nucleus",
        "requestType",
        "gshare",
        "resource_type",
        "diskIOUnit",
        "container_name",
        "framework",
    }
)

# ---------------------------------------------------------------------------
# Keys whose value is a pipe-separated "key=value|key=value" string.
# Each sub-rule is unpacked into its own TaskFeatureNode.
# ---------------------------------------------------------------------------
SPLIT_RULE_KEYS: frozenset[str] = frozenset({"splitRule"})

# ---------------------------------------------------------------------------
# Keys whose value is a CLI-argument string to be parsed into individual
# flag/value nodes.  Only keys whose structure is reliable enough to yield
# meaningful graph nodes belong here.
#
# Current members:
#   "task_creation_arguments"   – the full client-tool command used to
#                                  create the task, e.g.
#                                  'prun --exec "python a.py" --outDS user.x -v'
#                                  The first positional token (e.g. "prun") is
#                                  the submission tool name and becomes a
#                                  ComponentNode(system="submission_tool") so it
#                                  can participate in Component→Cause edges.
#                                  Any further positional tokens go to a
#                                  TaskContextNode(attribute="task_creation_arguments:positional_args").
#
# Note: "job_execution_params" is NOT here — its value is a free-form,
# incident-specific string with no reliable structure.  It lives in
# UNSTRUCTURED_TASK_KEYS and is stored whole as a TaskContextNode.
# ---------------------------------------------------------------------------
CLI_ARGUMENT_KEYS: frozenset[str] = frozenset({"task_creation_arguments"})

# ---------------------------------------------------------------------------
# Flags within task_creation_arguments split by how incident-specific their
# values are.  (job_execution_params is not parsed at all — it lives in
# UNSTRUCTURED_TASK_KEYS and is stored whole as a TaskContextNode.)
#
# TASK_CREATION_SKIP_ARGS   — pure identifiers / GUIDs / numeric ranges whose
#                             values are unique per incident and carry no
#                             semantic signal worth indexing.  Dropped entirely
#                             (neither graph nor vector DB).
#
# TASK_CREATION_CONTEXT_ARGS — free-form human-readable strings (script names,
#                             exec payloads, file patterns) that are unique per
#                             incident but carry semantic meaning.  Stored as
#                             TaskContextNode for vector DB similarity search.
#
# Flags in neither set (e.g. --nFilesPerJob, --nEvents, --maxCpuCount) carry
# repeatable configuration choices and become TaskFeatureNodes in the graph DB.
# ---------------------------------------------------------------------------
TASK_CREATION_SKIP_ARGS: frozenset[str] = frozenset(
    {
        # Dataset identifiers — always GUIDs / unique scope tokens.
        "outDS",
        "inDS",
        "minDS",
        "cavDS",
        "secondaryDSs",
        "extFile",
    }
)

TASK_CREATION_CONTEXT_ARGS: frozenset[str] = frozenset(
    {
        # Free-form execution payload — arbitrary user script invocation;
        # semantically useful for finding incidents with similar workloads.
        "exec",
        "trf",
        # Output file name patterns — may encode meaningful workflow names.
        "outputs",
        "extOutFile",
    }
)

# ---------------------------------------------------------------------------
# Keys whose values are continuous numerics (→ TaskFeatureNode with bucketed
# value).  Storing raw numbers like ramCount=123 / ramCount=456 would create
# a unique node per incident and prevent any graph merging.  Bucketing maps
# many incidents to the same node, making graph patterns meaningful.
#
# Bucket boundaries are chosen to reflect operationally significant thresholds
# in the PanDA system.  Units follow PanDA conventions (MB for memory/disk,
# seconds for time, integer for pure counts).
# ---------------------------------------------------------------------------
CONTINUOUS_TASK_KEYS: frozenset[str] = frozenset(
    {
        "ramCount",
        "outDiskCount",
        "workDiskCount",
        "diskIO",
        "walltime",
        "baseWalltime",
        "cpuTime",
        "baseRamCount",
        "ioIntensity",
    }
)

# Bucket definitions per key: sorted list of (upper_bound_exclusive, label).
# The last entry's upper_bound should be math.inf.
_BUCKETS: dict[str, list[tuple[float, str]]] = {
    # Memory / disk: MB
    "ramCount": [
        (512, "<512MB"),
        (2048, "512MB-2GB"),
        (8192, "2-8GB"),
        (32768, "8-32GB"),
        (float("inf"), ">32GB"),
    ],
    "baseRamCount": [
        (512, "<512MB"),
        (2048, "512MB-2GB"),
        (8192, "2-8GB"),
        (32768, "8-32GB"),
        (float("inf"), ">32GB"),
    ],
    "outDiskCount": [
        (1024, "<1GB"),
        (10240, "1-10GB"),
        (102400, "10-100GB"),
        (float("inf"), ">100GB"),
    ],
    "workDiskCount": [
        (1024, "<1GB"),
        (10240, "1-10GB"),
        (102400, "10-100GB"),
        (float("inf"), ">100GB"),
    ],
    # I/O: MB/s
    "diskIO": [(10, "<10MB/s"), (100, "10-100MB/s"), (float("inf"), ">100MB/s")],
    # Time: seconds
    "walltime": [
        (3600, "<1h"),
        (21600, "1-6h"),
        (86400, "6-24h"),
        (float("inf"), ">24h"),
    ],
    "baseWalltime": [
        (3600, "<1h"),
        (21600, "1-6h"),
        (86400, "6-24h"),
        (float("inf"), ">24h"),
    ],
    "cpuTime": [
        (3600, "<1h"),
        (21600, "1-6h"),
        (86400, "6-24h"),
        (float("inf"), ">24h"),
    ],
    # I/O intensity: dimensionless score
    "ioIntensity": [
        (100, "low(<100)"),
        (1000, "medium(100-1000)"),
        (float("inf"), "high(>1000)"),
    ],
}


def _bucket_value(key: str, raw: str) -> str:
    """Map a raw numeric string to a bucketed range label.

    Args:
        key: The task_data field name (must be in ``CONTINUOUS_TASK_KEYS``).
        raw: The raw string value from task_data.

    Returns:
        A bucketed label string, e.g. ``"512MB-2GB"``.  If *raw* cannot be
        parsed as a number the original string is returned unchanged so the
        node is still created (with a warning logged by the caller).

    Raises:
        ValueError: If *key* has no bucket definition.
    """
    buckets = _BUCKETS.get(key)
    if buckets is None:
        raise ValueError(f"No bucket definition for continuous key '{key}'")
    try:
        numeric = float(raw)
    except (ValueError, TypeError):
        return raw  # caller will log a warning
    for upper, label in buckets:
        if numeric < upper:
            return label
    return buckets[-1][1]  # should never reach here due to inf sentinel


def _parse_cli_arguments(raw: str) -> list[tuple[str, str]]:
    """Parse a CLI-argument string into a list of *(attribute, value)* pairs.

    Handles the following forms:

    Long options (``--``)::

        --key=value           → ``("key", "value")``
        --key="val ue"        → ``("key", "val ue")``   (quoted value with spaces)
        --key 'val ue'        → ``("key", "val ue")``   (space-separated quoted value)
        --key value           → ``("key", "value")``    (next token is value if it
                                does not start with ``-``)
        --flag                → ``("flag", "true")``    (boolean flag, no value)

    Short options (``-``)::

        -k=value              → ``("k", "value")``
        -k value              → ``("k", "value")``      (next token is value if it
                                does not start with ``-``)
        -v                    → ``("v", "true")``       (boolean flag)
        -abc                  → ``("a", "true"), ("b", "true"), ("c", "true")``
                                (combined single-char flags, expanded individually)

    Positional::

        positional            → collected and returned as
                                ``("positional_args", "<space-joined tokens>")``

    Quoted strings (single or double quotes) are handled via :mod:`shlex`
    so that ``--args="blah blah"`` and ``--args 'blah blah'`` both yield
    ``("args", "blah blah")``.

    If the same key appears more than once the last value wins (consistent with
    most CLI parsers).

    Returns a sorted list so that identical parameter sets produce identical
    node names regardless of the order they appear in the original string.
    """
    import shlex

    try:
        tokens = shlex.split(raw)
    except ValueError:
        # Fallback to naive split if shlex fails (e.g. unmatched quotes).
        logger.warning(
            "_parse_cli_arguments: shlex failed to parse %r — falling back to whitespace split",
            raw,
        )
        tokens = raw.split()

    pairs: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            # Long option: --key, --key=value, --key value
            body = token[2:]  # strip leading "--"
            if "=" in body:
                key, value = body.split("=", 1)
                pairs[key.strip()] = value
            elif body and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                pairs[body.strip()] = tokens[i + 1]
                i += 1
            elif body:
                pairs[body.strip()] = "true"
            # bare "--" (end-of-options marker) falls through with no action
        elif token.startswith("-") and len(token) > 1:
            # Short option: -k, -k=value, -k value, -abc (combined flags)
            body = token[1:]  # strip leading "-"
            if "=" in body:
                # -k=value
                key, value = body.split("=", 1)
                pairs[key.strip()] = value
            elif len(body) == 1:
                # Single short flag: -v or -v value
                if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    pairs[body] = tokens[i + 1]
                    i += 1
                else:
                    pairs[body] = "true"
            else:
                # Combined flags: -abc → -a -b -c (all boolean)
                for ch in body:
                    pairs[ch] = "true"
        else:
            positional.append(token)
        i += 1

    # Sort so identical parameter sets always produce identical node names.
    result = sorted(pairs.items())
    if positional:
        result.append(("positional_args", " ".join(positional)))
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_vector_id(node_type: str, label: str) -> str:
    """Stable, URL-safe vector ID for a canonical node of any type."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    digest = hashlib.md5(f"canonical::{node_type}::{label}".encode()).hexdigest()
    return f"canonical-{node_type.lower()}-{slug[:24]}-{digest[:8]}"


# ---------------------------------------------------------------------------
# CanonicalNodeStore  (VectorDB + LLM — works for any node type)
# ---------------------------------------------------------------------------


class CanonicalNodeStore:
    """Persistent store for canonical names of a single node type,
    backed by the vector database.

    This is the single mechanism used for all three canonicalisable node
    types: ``ErrorCategory``, ``Cause``, and ``Resolution``.

    Each entry is stored as a vector point with::

        section  = "canonical_node::<node_type>"
        content  = <canonical label text>
        metadata = {"label": <label>, "auto_generated": <bool>}

    Flow for :meth:`find_or_create`:
        1. Call *label_fn(raw_text)* — an LLM function that strips
           incident-specific tokens and returns a short canonical label.
        2. Embed the label and run ``search_similar`` against the section.
        3a. score ≥ ``match_threshold`` → return the stored label.
        3b. no match → store the new label, return it with ``is_new=True``.

    Args:
        node_type:       Node type name, e.g. ``"Cause"`` or ``"ErrorCategory"``.
        label_fn:        Async callable ``(raw_text: str) -> str`` that
                         normalises raw text into a clean canonical label.
        vector_client:   Pre-connected VectorDatabaseClient or *None* (lazy).
        embeddings_client: LangChain Embeddings or *None* (uses get_embeddings).
        match_threshold: Minimum cosine similarity to reuse an existing label.
    """

    def __init__(
        self,
        node_type: str,
        label_fn,
        vector_client=None,
        embeddings_client=None,
        match_threshold: float = 0.82,
    ) -> None:
        self._node_type = node_type
        self._section = f"canonical_node::{node_type}"
        self._label_fn = label_fn
        self._vector_client = vector_client
        self._embeddings = embeddings_client or get_embeddings()
        self.match_threshold = match_threshold

    async def _get_client(self):
        if self._vector_client is None:
            from bamboo.database.vector_database_client import VectorDatabaseClient

            self._vector_client = VectorDatabaseClient()
            await self._vector_client.connect()
        return self._vector_client

    async def _store(self, label: str, auto_generated: bool = True) -> str:
        client = await self._get_client()
        vector_id = _canonical_vector_id(self._node_type, label)
        embedding = self._embeddings.embed_query(label)
        await client.upsert_section_vector(
            vector_id=vector_id,
            embedding=embedding,
            content=label,
            section=self._section,
            metadata={"label": label, "auto_generated": auto_generated},
        )
        logger.debug("CanonicalNodeStore[%s]: stored '%s'", self._node_type, label)
        return vector_id

    async def find_or_create(self, raw_text: str) -> tuple[str, float, bool]:
        """Return *(canonical_label, score, is_new)*.

        1. LLM normalises *raw_text* → candidate label.
        2. Embed candidate → search VectorDB.
        3. Match found  → return stored label verbatim (guaranteed exact match
           for ``get_or_create_canonical_node``).
           No match → store candidate, return it.
        """
        client = await self._get_client()

        # Step 1: LLM normalisation
        candidate = await self._label_fn(raw_text)

        # Step 2: vector search using the clean candidate
        query_vec = self._embeddings.embed_query(candidate)
        results = await client.search_similar(
            query_embedding=query_vec,
            limit=1,
            score_threshold=self.match_threshold,
            filter_conditions={"section": self._section},
        )

        if results:
            label = results[0]["metadata"].get("label", candidate)
            score = float(results[0]["score"])
            logger.debug(
                "CanonicalNodeStore[%s]: matched '%s' (score=%.3f)",
                self._node_type,
                label,
                score,
            )
            return label, score, False

        # Step 3b: store candidate as a new entry
        logger.info(
            "CanonicalNodeStore[%s]: no match (threshold=%.2f) for '%s'; storing new",
            self._node_type,
            self.match_threshold,
            candidate,
        )
        await self._store(candidate, auto_generated=True)
        return candidate, 0.0, True

    async def add(self, label: str) -> str:
        """Manually add or update a canonical label (auto_generated=False)."""
        return await self._store(label, auto_generated=False)

    async def list_all(self) -> list[dict[str, Any]]:
        """Return all stored entries for this node type."""
        client = await self._get_client()
        sentinel_vec = self._embeddings.embed_query(self._node_type)
        results = await client.search_similar(
            query_embedding=sentinel_vec,
            limit=500,
            score_threshold=0.0,
            filter_conditions={"section": self._section},
        )
        return [
            {
                "label": r["metadata"].get("label", ""),
                "auto_generated": r["metadata"].get("auto_generated", True),
                "vector_id": r["id"],
            }
            for r in results
        ]


# ---------------------------------------------------------------------------
# LLM label functions — one per canonicalisable node type
# ---------------------------------------------------------------------------


async def _generate_category_label(error_message: str) -> str:
    """LLM: raw error message → CamelCase error-category label.

    Raises on any LLM failure so nothing is stored when classification fails.
    """
    llm = get_llm()
    response = await llm.ainvoke(
        ERROR_CATEGORY_LABEL_PROMPT.format(error_message=error_message)
    )
    label = re.sub(r"[^A-Za-z]", "", response.content.strip())
    if not label:
        raise ValueError(
            f"LLM returned an empty or non-alphabetic label for message: {error_message!r}"
        )
    return label


def _make_cause_resolution_label_fn(node_type: str):
    """Return an async label function for Cause or Resolution nodes."""

    async def _fn(raw_name: str) -> str:
        llm = get_llm()
        # Pass an empty existing-names block — the VectorDB handles matching;
        # the LLM's only job here is stripping incident-specific tokens.
        prompt = CAUSE_RESOLUTION_CANONICALIZE_PROMPT.format(
            node_type=node_type,
            existing_names="(handled by vector search — normalise wording only)",
            raw_name=raw_name,
        )
        response = await llm.ainvoke(prompt)
        canonical = response.content.strip().strip('"').strip("'")
        if not canonical:
            raise ValueError(
                f"LLM returned an empty canonical name for {node_type} {raw_name!r}"
            )
        return canonical

    return _fn


# ---------------------------------------------------------------------------
# ErrorCategoryStore  — specialization of CanonicalNodeStore
# ---------------------------------------------------------------------------


class ErrorCategoryStore(CanonicalNodeStore):
    """Persistent store for error categories.

    Thin specialisation of :class:`CanonicalNodeStore` with
    ``node_type="ErrorCategory"`` and the error-label LLM function.
    Kept as a named class for backward compatibility and to expose
    :meth:`add_category` / :meth:`list_categories` aliases.
    """

    def __init__(
        self,
        vector_client=None,
        embeddings_client=None,
        new_category_threshold: float = 0.70,
    ) -> None:
        super().__init__(
            node_type="ErrorCategory",
            label_fn=_generate_category_label,
            vector_client=vector_client,
            embeddings_client=embeddings_client,
            match_threshold=new_category_threshold,
        )

    # Keep old threshold attribute name for any callers that used it directly.
    @property
    def new_category_threshold(self) -> float:
        return self.match_threshold

    @new_category_threshold.setter
    def new_category_threshold(self, value: float) -> None:
        self.match_threshold = value

    async def add_category(self, label: str, description: str) -> str:  # noqa: ARG002
        """Manually add a named category (description is ignored — label is content)."""
        return await self.add(label)

    async def list_categories(self) -> list[dict[str, Any]]:
        """Return all stored categories (alias for list_all with 'description' key)."""
        entries = await self.list_all()
        for e in entries:
            e["description"] = e["label"]  # content == label in this store
        return entries


# ---------------------------------------------------------------------------
# ErrorCategoryClassifier
# ---------------------------------------------------------------------------


class ErrorCategoryClassifier:
    """Classifies a raw error message using :class:`ErrorCategoryStore`."""

    def __init__(self, store: Optional[ErrorCategoryStore] = None) -> None:
        self._store = store or ErrorCategoryStore()

    async def classify(self, error_message: str) -> tuple[str, float]:
        """Return *(category_label, confidence)* for *error_message*."""
        if not error_message or not error_message.strip():
            return "Unknown", 0.0
        label, score, _ = await self._store.find_or_create(error_message)
        return label, score

    @property
    def store(self) -> ErrorCategoryStore:
        return self._store


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class PandaKnowledgeExtractor(ExtractionStrategy):
    """Structured extraction strategy for Panda-flavoured task data.

    * ``external_data`` key→value pairs → :class:`TaskFeatureNode`
    * ``task_data`` discrete fields     → :class:`TaskFeatureNode`
    * ``task_data`` free-form fields    → :class:`TaskContextNode`
    * ``task_data["errorDialog"]``     → :class:`SymptomNode` whose ``name``
                                          is the canonical error category and
                                          ``description`` is the raw message text
    * ``email_text``                    → :class:`CauseNode`, :class:`ResolutionNode`,
                                          and :class:`TaskContextNode` via LLM.
                                          Cause/Resolution names are canonicalised
                                          with the same VectorDB+LLM pattern as
                                          error categories, guaranteeing dedup.
    """

    _SUPPORTED_SYSTEMS: frozenset[str] = frozenset({"panda", "bamboo_panda"})

    def __init__(
        self,
        unstructured_keys: Optional[frozenset[str]] = None,
        discrete_keys: Optional[frozenset[str]] = None,
        continuous_keys: Optional[frozenset[str]] = None,
        split_rule_keys: Optional[frozenset[str]] = None,
        cli_argument_keys: Optional[frozenset[str]] = None,
        task_creation_context_args: Optional[frozenset[str]] = None,
        task_creation_skip_args: Optional[frozenset[str]] = None,
        error_classifier: Optional[ErrorCategoryClassifier] = None,
        cause_store: Optional[CanonicalNodeStore] = None,
        resolution_store: Optional[CanonicalNodeStore] = None,
    ) -> None:
        """
        Args:
            unstructured_keys:           Override default prose keys (UNSTRUCTURED_TASK_KEYS).
            discrete_keys:               Override default discrete keys (DISCRETE_TASK_KEYS).
            continuous_keys:             Override default continuous keys (CONTINUOUS_TASK_KEYS).
            split_rule_keys:             Override default pipe-split keys (SPLIT_RULE_KEYS).
            cli_argument_keys:           Override default CLI-arg keys (CLI_ARGUMENT_KEYS).
            task_creation_context_args:  Override the set of task_creation_arguments flags
                                         whose values are free-form but semantically useful
                                         (script names, exec payloads) → stored as
                                         TaskContextNode for vector search
                                         (TASK_CREATION_CONTEXT_ARGS).
            task_creation_skip_args:     Override the set of task_creation_arguments flags
                                         whose values are pure identifiers / GUIDs with no
                                         semantic signal → dropped entirely
                                         (TASK_CREATION_SKIP_ARGS).
            error_classifier:            Custom ErrorCategoryClassifier (for testing).
            cause_store:                 Custom CanonicalNodeStore for Cause nodes (for testing).
            resolution_store:            Custom CanonicalNodeStore for Resolution nodes (for testing).
        """
        self._unstructured_keys: frozenset[str] = (
            unstructured_keys
            if unstructured_keys is not None
            else UNSTRUCTURED_TASK_KEYS
        )
        self._discrete_keys: frozenset[str] = (
            discrete_keys if discrete_keys is not None else DISCRETE_TASK_KEYS
        )
        self._continuous_keys: frozenset[str] = (
            continuous_keys if continuous_keys is not None else CONTINUOUS_TASK_KEYS
        )
        self._split_rule_keys: frozenset[str] = (
            split_rule_keys if split_rule_keys is not None else SPLIT_RULE_KEYS
        )
        self._cli_argument_keys: frozenset[str] = (
            cli_argument_keys if cli_argument_keys is not None else CLI_ARGUMENT_KEYS
        )
        self._task_creation_context_args: frozenset[str] = (
            task_creation_context_args
            if task_creation_context_args is not None
            else TASK_CREATION_CONTEXT_ARGS
        )
        self._task_creation_skip_args: frozenset[str] = (
            task_creation_skip_args
            if task_creation_skip_args is not None
            else TASK_CREATION_SKIP_ARGS
        )
        self._error_classifier: ErrorCategoryClassifier = (
            error_classifier or ErrorCategoryClassifier()
        )
        self._cause_store: CanonicalNodeStore = cause_store or CanonicalNodeStore(
            node_type="Cause",
            label_fn=_make_cause_resolution_label_fn("Cause"),
        )
        self._resolution_store: CanonicalNodeStore = (
            resolution_store
            or CanonicalNodeStore(
                node_type="Resolution",
                label_fn=_make_cause_resolution_label_fn("Resolution"),
            )
        )

    # ------------------------------------------------------------------
    # ExtractionStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "panda"

    @property
    def description(self) -> str:
        return (
            "Structured extraction for Panda task data: "
            "discrete fields → TaskFeatureNode, "
            "free-form fields → TaskContextNode, "
            "errorDialog → VectorDB+LLM error-category classification, "
            "email_text → LLM-extracted Cause / Resolution / Task_Context "
            "(all canonicalised via VectorDB+LLM)."
        )

    def supports_system(self, system_type: str) -> bool:
        return system_type.lower() in self._SUPPORTED_SYSTEMS

    async def extract(
        self,
        email_text: str = "",
        task_data: Optional[dict[str, Any]] = None,
        external_data: Optional[dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        nodes: list = []
        relationships: list[GraphRelationship] = []

        for key, value in (external_data or {}).items():
            nodes.append(
                self._make_feature_node(
                    attribute=str(key),
                    value=str(value),
                    description=f"External data field: {key}",
                    source="external_data",
                )
            )

        for key, value in (task_data or {}).items():
            if key == "errorDialog":
                # Dedicated handling: produce a SymptomNode whose name is the
                # canonical error category and whose description is the raw
                # message text (preserved for traceability and vector search).
                if value:
                    category, confidence = await self._classify_error(str(value))
                    nodes.append(
                        SymptomNode(
                            name=category,
                            description=str(value),
                            metadata={
                                "source": "error_classifier",
                                "classifier_confidence": confidence,
                            },
                        )
                    )
            elif key == "status":
                # Task status (e.g. "failed", "broken") describes the failure
                # state of the task, not a configuration attribute.  It is stored
                # as a SymptomNode so it participates in Symptom→Cause edges.
                if value:
                    category, confidence = await self._classify_error(str(value))
                    nodes.append(
                        SymptomNode(
                            name=category,
                            description=str(value),
                            metadata={
                                "source": "task_status",
                                "classifier_confidence": confidence,
                            },
                        )
                    )
            elif key in self._split_rule_keys:
                # splitRule value is a comma-separated "attr=val,attr=val" string.
                # Each sub-rule becomes its own TaskFeatureNode.
                # The attribute is namespaced as "splitRule:<attr>" so that a
                # sub-rule like "site=CERN" cannot collide with the top-level
                # task_data key "site=CERN" in the graph DB.
                for sub_rule in str(value).split(","):
                    sub_rule = sub_rule.strip()
                    if "=" in sub_rule:
                        attr, val = sub_rule.split("=", 1)
                        namespaced_attr = f"splitRule:{attr.strip()}"
                        nodes.append(
                            self._make_feature_node(
                                attribute=namespaced_attr,
                                value=val.strip(),
                                description=f"splitRule sub-rule: {sub_rule}",
                                source="task_data/splitRule",
                            )
                        )
                    elif sub_rule:
                        logger.warning(
                            "PandaKnowledgeExtractor: splitRule sub-rule '%s' "
                            "has no '=' separator — skipped",
                            sub_rule,
                        )
            elif key in self._cli_argument_keys:
                # CLI-argument string: each flag/value pair becomes its own
                # TaskFeatureNode, namespaced as "<key>:<attr>" to avoid
                # collisions with identically-named top-level task_data keys
                # or splitRule sub-keys.
                # For task_creation_arguments the first positional token (the
                # command name, e.g. "prun") is the submission tool — a system
                # component — and becomes a ComponentNode so it can participate
                # in Component -[originated_from]-> Cause edges.  Any further
                # positional tokens become a TaskContextNode.
                pairs = _parse_cli_arguments(str(value))
                for attr, val in pairs:
                    if attr == "positional_args":
                        positional_tokens = val.split()
                        if key == "task_creation_arguments" and positional_tokens:
                            # First token is the submission tool name — a Component.
                            nodes.append(
                                ComponentNode(
                                    name=positional_tokens[0],
                                    description=f"Submission tool used to create the task",
                                    system="submission_tool",
                                    metadata={"source": f"task_data/{key}"},
                                )
                            )
                            # Remaining positional tokens (rare) stay as context.
                            if len(positional_tokens) > 1:
                                nodes.append(
                                    self._make_context_node(
                                        attribute=f"{key}:positional_args",
                                        prose=" ".join(positional_tokens[1:]),
                                    )
                                )
                        else:
                            nodes.append(
                                self._make_context_node(
                                    attribute=f"{key}:positional_args",
                                    prose=val,
                                )
                            )
                    else:
                        namespaced_attr = f"{key}:{attr}"
                        # Classify each flag into three buckets:
                        #   skip args    — pure GUIDs/identifiers, no signal → dropped entirely
                        #   context args — free-form prose, semantic meaning → TaskContextNode (vector DB)
                        #   everything else — repeatable config choices → TaskFeatureNode (graph DB)
                        if attr in self._task_creation_skip_args:
                            continue
                        elif attr in self._task_creation_context_args:
                            nodes.append(
                                self._make_context_node(
                                    attribute=namespaced_attr,
                                    prose=val,
                                )
                            )
                        else:
                            nodes.append(
                                self._make_feature_node(
                                    attribute=namespaced_attr,
                                    value=val,
                                    description=f"{key} arg: --{attr}",
                                    source=f"task_data/{key}",
                                )
                            )
            elif key in self._unstructured_keys:
                nodes.append(
                    self._make_context_node(
                        attribute=str(key),
                        prose=str(value) if value is not None else "",
                    )
                )
            elif key in self._discrete_keys:
                nodes.append(
                    self._make_feature_node(
                        attribute=str(key),
                        value=str(value) if value is not None else "",
                        description=f"Task data field: {key}",
                        source="task_data",
                    )
                )
            elif key in self._continuous_keys:
                # Bucket the raw numeric into a range label so incidents with
                # different but similar values share the same node.
                raw = str(value) if value is not None else ""
                bucketed = _bucket_value(key, raw)
                if bucketed == raw:
                    logger.warning(
                        "PandaKnowledgeExtractor: continuous key '%s' has "
                        "non-numeric value '%s' — stored as-is",
                        key,
                        raw,
                    )
                nodes.append(
                    self._make_feature_node(
                        attribute=str(key),
                        value=bucketed,
                        description=f"Task data field: {key} (bucketed from {raw!r})",
                        source="task_data",
                        extra_metadata={"raw_value": raw},
                    )
                )
            else:
                # ignore unrecognised keys.
                pass

        if email_text and email_text.strip():
            email_nodes, email_rels = await self._extract_from_email(email_text)
            nodes.extend(email_nodes)
            relationships.extend(email_rels)

        graph = KnowledgeGraph(nodes=nodes, relationships=relationships)
        logger.info(
            "PandaKnowledgeExtractor: extracted %d nodes, %d relationships",
            len(nodes),
            len(relationships),
        )
        return graph

    # ------------------------------------------------------------------
    # Email extraction
    # ------------------------------------------------------------------

    async def _extract_from_email(
        self, email_text: str
    ) -> tuple[list, list[GraphRelationship]]:
        """LLM extracts Cause/Resolution/Task_Context from email.

        Cause and Resolution names are canonicalised via their respective
        :class:`CanonicalNodeStore` instances (VectorDB + LLM), using the same
        approach as ErrorCategory.  This guarantees that the same concept from
        two different emails produces the same node name and is merged correctly
        by ``get_or_create_canonical_node`` in the graph database.
        """
        llm = get_llm()
        response = await llm.ainvoke(
            EMAIL_EXTRACTION_PROMPT.format(email_text=email_text)
        )
        raw = self._parse_email_response(response.content)

        _store_for_type = {
            "Cause": self._cause_store,
            "Resolution": self._resolution_store,
        }

        nodes = []
        raw_name_to_node: dict[str, Any] = {}

        for node_data in raw.get("nodes", []):
            raw_name = node_data["name"]
            node_type_str = node_data.get("node_type", "")

            if node_type_str in _store_for_type:
                node_data = dict(node_data)
                canonical, _, _ = await _store_for_type[node_type_str].find_or_create(
                    raw_name
                )
                node_data["name"] = canonical

            node = self._create_email_node(node_data)
            if node is not None:
                nodes.append(node)
                raw_name_to_node[raw_name] = node

        relationships: list[GraphRelationship] = []
        for rel_data in raw.get("relationships", []):
            src = rel_data.get("source_name")
            tgt = rel_data.get("target_name")
            if src not in raw_name_to_node or tgt not in raw_name_to_node:
                continue
            try:
                rel_type = RelationType(rel_data["relation_type"])
            except ValueError:
                logger.warning(
                    "PandaKnowledgeExtractor: unknown relation_type '%s' — skipped",
                    rel_data.get("relation_type"),
                )
                continue
            relationships.append(
                GraphRelationship(
                    source_id=raw_name_to_node[src].name,
                    target_id=raw_name_to_node[tgt].name,
                    relation_type=rel_type,
                    confidence=float(rel_data.get("confidence", 1.0)),
                )
            )

        logger.debug(
            "PandaKnowledgeExtractor: email yielded %d nodes, %d relationships",
            len(nodes),
            len(relationships),
        )
        return nodes, relationships

    @staticmethod
    def _parse_email_response(response: str) -> dict[str, Any]:
        text = response.strip()
        if "```json" in text:
            text = text[text.find("```json") + 7 : text.rfind("```")].strip()
        elif "```" in text:
            text = text[text.find("```") + 3 : text.rfind("```")].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "PandaKnowledgeExtractor: failed to parse email LLM response: %s", exc
            )
            return {"nodes": [], "relationships": []}

    @staticmethod
    def _create_email_node(node_data: dict[str, Any]):
        from bamboo.models.graph_element import NodeType

        try:
            node_type = NodeType(node_data.get("node_type", ""))
        except ValueError:
            logger.warning(
                "PandaKnowledgeExtractor: unexpected node_type '%s' in email extraction — skipped",
                node_data.get("node_type"),
            )
            return None

        base = {
            "name": node_data["name"],
            "description": node_data.get("description"),
            "metadata": node_data.get("metadata", {}),
        }
        if node_type == NodeType.CAUSE:
            return CauseNode(**base, confidence=float(node_data.get("confidence", 1.0)))
        if node_type == NodeType.RESOLUTION:
            return ResolutionNode(
                **base,
                steps=node_data.get("steps", []),
                success_rate=node_data.get("success_rate"),
                estimated_duration=node_data.get("estimated_duration"),
            )
        if node_type == NodeType.TASK_CONTEXT:
            return TaskContextNode(**base)

        logger.warning(
            "PandaKnowledgeExtractor: node_type '%s' is not permitted in email extraction — skipped",
            node_type,
        )
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_feature_node(
        self,
        attribute: str,
        value: str,
        description: str,
        source: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> TaskFeatureNode:
        metadata: dict[str, Any] = {
            "attribute": attribute,
            "value": value,
            "source": source,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return TaskFeatureNode(
            name=f"{attribute}={value}",
            attribute=attribute,
            value=value,
            description=description,
            metadata=metadata,
        )

    def _make_context_node(self, attribute: str, prose: str) -> TaskContextNode:
        return TaskContextNode(
            name=attribute,
            description=prose,
            metadata={"source": "task_data", "attribute": attribute},
        )

    async def _classify_error(self, error_message: str) -> tuple[str, float]:
        try:
            return await self._error_classifier.classify(error_message)
        except Exception as exc:
            logger.warning("Error classification failed: %s", exc)
            return "Unknown", 0.0
