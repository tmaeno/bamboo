"""PanDA extraction strategy for structured task/external data.

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
        "jobParameters",
        "taskName",
        # Note: "errorDialog" is handled separately — it produces a
        # SymptomNode (canonical category name, raw message as description)
        # rather than a TaskContextNode.
    }
)

# ---------------------------------------------------------------------------
# Keys in task_data that carry discrete / categorical values (→ TaskFeatureNode).
# Any key that is not in DISCRETE_TASK_KEYS, UNSTRUCTURED_TASK_KEYS, or the
# special "errorDialog" field will be logged as unknown and skipped, preventing
# accidental indexing of unexpected blobs or nested structures.
#
# Notes:
# - "taskName" must NOT appear here; it belongs only in UNSTRUCTURED_TASK_KEYS.
# - "status" must NOT appear here; it is routed to the SymptomNode branch
#   because it describes the task's failure state (e.g. "failed", "broken"),
#   not a task configuration attribute.
# - "splitRule" is handled by a dedicated branch that parses its pipe-separated
#   "key=value|key=value" format into one TaskFeatureNode per sub-rule.
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
        "walltime",
        "walltimeUnit",
        "outDiskCount",
        "outDiskUnit",
        "workDiskCount",
        "workDiskUnit",
        "ramCount",
        "ramUnit",
        "ioIntensity",
        "ioIntensityUnit",
        "reqID",
        "site",
        "countryGroup",
        "campaign",
        "goal",
        "cpuTime",
        "cpuTimeUnit",
        "baseWalltime",
        "nucleus",
        "baseRamCount",
        "requestType",
        "gshare",
        "resource_type",
        "diskIO",
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
                self._node_type, label, score,
            )
            return label, score, False

        # Step 3b: store candidate as a new entry
        logger.info(
            "CanonicalNodeStore[%s]: no match (threshold=%.2f) for '%s'; storing new",
            self._node_type, self.match_threshold, candidate,
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
# ErrorCategoryStore  — specialisation of CanonicalNodeStore
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
        split_rule_keys: Optional[frozenset[str]] = None,
        error_classifier: Optional[ErrorCategoryClassifier] = None,
        cause_store: Optional[CanonicalNodeStore] = None,
        resolution_store: Optional[CanonicalNodeStore] = None,
    ) -> None:
        """
        Args:
            unstructured_keys: Override default prose keys (UNSTRUCTURED_TASK_KEYS).
            discrete_keys:     Override default discrete keys (DISCRETE_TASK_KEYS).
            split_rule_keys:   Override default pipe-split keys (SPLIT_RULE_KEYS).
            error_classifier:  Custom ErrorCategoryClassifier (for testing).
            cause_store:       Custom CanonicalNodeStore for Cause nodes (for testing).
            resolution_store:  Custom CanonicalNodeStore for Resolution nodes (for testing).
        """
        self._unstructured_keys: frozenset[str] = (
            unstructured_keys if unstructured_keys is not None else UNSTRUCTURED_TASK_KEYS
        )
        self._discrete_keys: frozenset[str] = (
            discrete_keys if discrete_keys is not None else DISCRETE_TASK_KEYS
        )
        self._split_rule_keys: frozenset[str] = (
            split_rule_keys if split_rule_keys is not None else SPLIT_RULE_KEYS
        )
        self._error_classifier: ErrorCategoryClassifier = (
            error_classifier or ErrorCategoryClassifier()
        )
        self._cause_store: CanonicalNodeStore = cause_store or CanonicalNodeStore(
            node_type="Cause",
            label_fn=_make_cause_resolution_label_fn("Cause"),
        )
        self._resolution_store: CanonicalNodeStore = resolution_store or CanonicalNodeStore(
            node_type="Resolution",
            label_fn=_make_cause_resolution_label_fn("Resolution"),
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
            nodes.append(self._make_feature_node(
                attribute=str(key), value=str(value),
                description=f"External data field: {key}", source="external_data",
            ))

        for key, value in (task_data or {}).items():
            if key == "errorDialog":
                # Dedicated handling: produce a SymptomNode whose name is the
                # canonical error category and whose description is the raw
                # message text (preserved for traceability and vector search).
                if value:
                    category, confidence = await self._classify_error(str(value))
                    nodes.append(SymptomNode(
                        name=category,
                        description=str(value),
                        metadata={
                            "source": "error_classifier",
                            "classifier_confidence": confidence,
                        },
                    ))
            elif key == "status":
                # Task status (e.g. "failed", "broken") describes the failure
                # state of the task, not a configuration attribute.  It is stored
                # as a SymptomNode so it participates in Symptom→Cause edges.
                if value:
                    category, confidence = await self._classify_error(str(value))
                    nodes.append(SymptomNode(
                        name=category,
                        description=str(value),
                        metadata={
                            "source": "task_status",
                            "classifier_confidence": confidence,
                        },
                    ))
            elif key in self._split_rule_keys:
                # splitRule value is a comma-separated "attr=val,attr=val" string.
                # Each sub-rule becomes its own TaskFeatureNode.
                for sub_rule in str(value).split(","):
                    sub_rule = sub_rule.strip()
                    if "=" in sub_rule:
                        attr, val = sub_rule.split("=", 1)
                        nodes.append(self._make_feature_node(
                            attribute=attr.strip(),
                            value=val.strip(),
                            description=f"splitRule sub-rule: {sub_rule}",
                            source="task_data/splitRule",
                        ))
                    elif sub_rule:
                        logger.warning(
                            "PandaKnowledgeExtractor: splitRule sub-rule '%s' "
                            "has no '=' separator — skipped", sub_rule,
                        )
            elif key in self._unstructured_keys:
                nodes.append(self._make_context_node(
                    attribute=str(key),
                    prose=str(value) if value is not None else "",
                ))
            elif key in self._discrete_keys:
                nodes.append(self._make_feature_node(
                    attribute=str(key),
                    value=str(value) if value is not None else "",
                    description=f"Task data field: {key}", source="task_data",
                ))
            else:
                logger.warning(
                    "PandaKnowledgeExtractor: unknown task_data key '%s' — skipped "
                    "(add to DISCRETE_TASK_KEYS or UNSTRUCTURED_TASK_KEYS to index it)",
                    key,
                )

        if email_text and email_text.strip():
            email_nodes, email_rels = await self._extract_from_email(email_text)
            nodes.extend(email_nodes)
            relationships.extend(email_rels)

        graph = KnowledgeGraph(nodes=nodes, relationships=relationships)
        logger.info(
            "PandaKnowledgeExtractor: extracted %d nodes, %d relationships",
            len(nodes), len(relationships),
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
                canonical, _, _ = await _store_for_type[node_type_str].find_or_create(raw_name)
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
            relationships.append(GraphRelationship(
                source_id=raw_name_to_node[src].name,
                target_id=raw_name_to_node[tgt].name,
                relation_type=rel_type,
                confidence=float(rel_data.get("confidence", 1.0)),
            ))

        logger.debug(
            "PandaKnowledgeExtractor: email yielded %d nodes, %d relationships",
            len(nodes), len(relationships),
        )
        return nodes, relationships

    @staticmethod
    def _parse_email_response(response: str) -> dict[str, Any]:
        text = response.strip()
        if "```json" in text:
            text = text[text.find("```json") + 7: text.rfind("```")].strip()
        elif "```" in text:
            text = text[text.find("```") + 3: text.rfind("```")].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("PandaKnowledgeExtractor: failed to parse email LLM response: %s", exc)
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
        self, attribute: str, value: str, description: str, source: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> TaskFeatureNode:
        metadata: dict[str, Any] = {"attribute": attribute, "value": value, "source": source}
        if extra_metadata:
            metadata.update(extra_metadata)
        return TaskFeatureNode(
            name=f"{attribute}={value}", attribute=attribute, value=value,
            description=description, metadata=metadata,
        )

    def _make_context_node(self, attribute: str, prose: str) -> TaskContextNode:
        return TaskContextNode(
            name=attribute, description=prose,
            metadata={"source": "task_data", "attribute": attribute},
        )

    async def _classify_error(self, error_message: str) -> tuple[str, float]:
        try:
            return await self._error_classifier.classify(error_message)
        except Exception as exc:
            logger.warning("Error classification failed: %s", exc)
            return "Unknown", 0.0
