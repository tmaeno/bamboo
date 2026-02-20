"""Panda extraction strategy for structured task/external data.

Design
------
* ``external_data``  – treated as a flat key→value dictionary; every pair
  becomes a :class:`~bamboo.models.graph_element.TaskFeatureNode`.

* ``task_data``      – key→value dict whose fields are classified as either:
    - **discrete / categorical** → :class:`TaskFeatureNode`
    - **unstructured / free-form** → :class:`TaskContextNode`

  The special ``ErrorMessage`` field is additionally run through the
  :class:`ErrorCategoryClassifier`, which queries :class:`ErrorCategoryStore`
  to find the closest persisted error category via vector similarity.  When
  no existing category scores above ``new_category_threshold`` a new category
  is minted from the message, stored persistently, and returned.

* ``email_text``     – passed to the LLM to extract :class:`CauseNode`,
  :class:`ResolutionNode`, and :class:`TaskContextNode` nodes.  The LLM is
  instructed to emit *only* those three types so they complement (rather than
  overlap with) the structured nodes derived from ``task_data`` and
  ``external_data``.

Persistence
-----------
Error categories are stored in the same Qdrant collection as all other
knowledge, distinguished by ``section="error_category"`` in the payload.
The store starts empty and grows organically as incidents are processed.
Categories can also be added manually via :meth:`ErrorCategoryStore.add_category`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Optional

from bamboo.extractors.base import ExtractionStrategy
from bamboo.llm import EMAIL_EXTRACTION_PROMPT, get_embeddings, get_llm
from bamboo.llm.prompts import ERROR_CATEGORY_LABEL_PROMPT
from bamboo.models.graph_element import (
    CauseNode,
    GraphRelationship,
    RelationType,
    ResolutionNode,
    TaskContextNode,
    TaskFeatureNode,
)
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keys in task_data that carry free-form prose (→ TaskContextNode).
# Everything else is treated as a discrete / comparable value (→ TaskFeatureNode).
# ---------------------------------------------------------------------------
UNSTRUCTURED_TASK_KEYS: frozenset[str] = frozenset(
    {
        "description",
        "steps_to_reproduce",
        "user_report",
        "comment",
        "notes",
        "additional_info",
        "reproduction_steps",
        "observed_behavior",
        "expected_behavior",
        "workaround",
        "ErrorMessage",  # raw text → TaskContextNode; classified category →
        #                  separate TaskFeatureNode via ErrorCategoryClassifier
    }
)

# The vector payload field used to partition error-category vectors from all
# other knowledge vectors stored in the same collection.
_SECTION = "error_category"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _category_vector_id(label: str) -> str:
    """Stable, URL-safe vector ID derived from the category label."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    digest = hashlib.md5(f"error_category::{label}".encode()).hexdigest()
    return f"errcategory-{slug}-{digest[:8]}"


async def _generate_category_label(error_message: str) -> str:
    """Ask the LLM to produce a short canonical CamelCase category label.

    The LLM strips all incident-specific tokens (dataset names, paths,
    usernames, version strings, numeric IDs) and returns only the structural
    pattern of the error, so two messages that differ only in those tokens
    receive the same label.

    Raises:
        Exception: Any LLM or API error is propagated to the caller so that
            no data is stored in the database when classification fails.
    """
    llm = get_llm()
    prompt = ERROR_CATEGORY_LABEL_PROMPT.format(error_message=error_message)
    response = await llm.ainvoke(prompt)
    # Strip whitespace/quotes and keep only word characters
    label = re.sub(r"[^A-Za-z]", "", response.content.strip())
    if not label:
        raise ValueError(
            f"LLM returned an empty or non-alphabetic label for message: {error_message!r}"
        )
    return label


# ---------------------------------------------------------------------------
# ErrorCategoryStore
# ---------------------------------------------------------------------------


class ErrorCategoryStore:
    """Persistent store for error categories backed by the vector database.

    Each category is stored as a single vector point with::

        section  = "error_category"
        content  = <description text used for similarity matching>
        metadata = {"label": <category label>, "auto_generated": <bool>}

    The store starts empty.  Categories grow automatically as new error
    messages are processed, and can be seeded manually via
    :meth:`add_category`.

    Classification
    ~~~~~~~~~~~~~~
    :meth:`find_or_create` embeds the incoming error message and runs
    ``search_similar`` filtered to ``section="error_category"``.

    * score ≥ ``new_category_threshold`` → existing category returned
    * score <  ``new_category_threshold`` → new category minted, stored, returned
    """

    def __init__(
        self,
        vector_client=None,
        embeddings_client=None,
        new_category_threshold: float = 0.70,
    ) -> None:
        """
        Args:
            vector_client: A connected :class:`VectorDatabaseClient` (or any
                object with the same async interface).  When *None* the store
                creates one lazily.
            embeddings_client: LangChain ``Embeddings`` instance.  When *None*
                :func:`~bamboo.llm.get_embeddings` is used.
            new_category_threshold: Minimum cosine similarity score to accept
                an existing category as a match.  Scores below this threshold
                trigger auto-creation of a new category.
        """
        self._vector_client = vector_client
        self._embeddings = embeddings_client or get_embeddings()
        self.new_category_threshold = new_category_threshold

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    async def _get_client(self):
        if self._vector_client is None:
            from bamboo.database.vector_database_client import VectorDatabaseClient

            self._vector_client = VectorDatabaseClient()
            await self._vector_client.connect()
        return self._vector_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _store_category(
        self,
        label: str,
        description: str,
        auto_generated: bool = True,
    ) -> str:
        """Embed *description* and upsert it into the vector store."""
        client = await self._get_client()
        vector_id = _category_vector_id(label)
        embedding = self._embeddings.embed_query(description)
        await client.upsert_section_vector(
            vector_id=vector_id,
            embedding=embedding,
            content=description,
            section=_SECTION,
            metadata={
                "label": label,
                "auto_generated": auto_generated,
            },
        )
        logger.debug(
            "ErrorCategoryStore: stored category '%s' (id=%s)", label, vector_id
        )
        return vector_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def find_or_create(self, error_message: str) -> tuple[str, float, bool]:
        """Find the closest stored category or mint a new one.

        The label is generated first by the LLM (stripping all
        incident-specific tokens), and that clean label is used for the
        vector similarity search.  This means the vector space contains only
        generalised, canonical labels rather than noisy raw messages, making
        similarity matching far more reliable.

        Flow:
            1. LLM produces a canonical CamelCase label from *error_message*.
            2. Embed the label and search the store.
            3a. Match found  → return the stored label + score.
            3b. No match     → store the new label (content = label) and return it.

        Args:
            error_message: Raw error message text.

        Returns:
            ``(label, score, is_new)`` where *is_new* is ``True`` when a new
            category was created during this call.
        """
        client = await self._get_client()

        # Step 1: normalise the raw message into a clean canonical label.
        candidate_label = await _generate_category_label(error_message)

        # Step 2: search the store using the label embedding.
        query_vec = self._embeddings.embed_query(candidate_label)
        results = await client.search_similar(
            query_embedding=query_vec,
            limit=1,
            score_threshold=self.new_category_threshold,
            filter_conditions={"section": _SECTION},
        )

        if results:
            best = results[0]
            label = best["metadata"].get("label", "Unknown")
            score = float(best["score"])
            logger.debug("ErrorCategoryStore: matched '%s' (score=%.3f)", label, score)
            return label, score, False

        # Step 3b: no match — store the candidate label as a new category.
        logger.info(
            "ErrorCategoryStore: no match (threshold=%.2f) for candidate '%s'; "
            "creating new category",
            self.new_category_threshold,
            candidate_label,
        )
        # Store the label itself as content so the vector space stays clean.
        await self._store_category(candidate_label, candidate_label, auto_generated=True)
        return candidate_label, 0.0, True

    async def add_category(self, label: str, description: str) -> str:
        """Manually add or update a named category.

        Useful for back-filling domain-specific categories from external
        knowledge bases.

        Returns:
            The vector store ID for the upserted entry.
        """
        return await self._store_category(label, description, auto_generated=False)

    async def list_categories(self) -> list[dict[str, Any]]:
        """Return all stored categories as a list of dicts.

        Each dict has keys: ``label``, ``description``, ``auto_generated``,
        ``vector_id``.
        """
        client = await self._get_client()
        # Fetch using a broad similarity search — use a near-zero threshold
        # to retrieve all entries; limit is set high enough for practical use.
        sentinel_vec = self._embeddings.embed_query("error")
        results = await client.search_similar(
            query_embedding=sentinel_vec,
            limit=500,
            score_threshold=0.0,
            filter_conditions={"section": _SECTION},
        )
        return [
            {
                "label": r["metadata"].get("label", ""),
                "description": r["content"],
                "auto_generated": r["metadata"].get("auto_generated", True),
                "vector_id": r["id"],
            }
            for r in results
        ]


# ---------------------------------------------------------------------------
# ErrorCategoryClassifier  (thin async wrapper around ErrorCategoryStore)
# ---------------------------------------------------------------------------


class ErrorCategoryClassifier:
    """Classifies a raw error message using :class:`ErrorCategoryStore`.

    The store is the single source of truth; category vectors are persisted
    across service restarts and grow automatically when novel error patterns
    are encountered.

    Usage::

        clf = ErrorCategoryClassifier()
        label, confidence = await clf.classify("Connection timed out after 30s")
    """

    def __init__(
        self,
        store: Optional[ErrorCategoryStore] = None,
    ) -> None:
        """
        Args:
            store: Provide a custom :class:`ErrorCategoryStore` (useful for
                testing).  When *None* a default instance is created.
        """
        self._store = store or ErrorCategoryStore()

    async def classify(self, error_message: str) -> tuple[str, float]:
        """Return *(category_label, confidence)* for *error_message*.

        Args:
            error_message: Raw error message text.

        Returns:
            A tuple of the matched (or newly created) category label and the
            cosine similarity score (0–1; 0.0 for newly created categories).
        """
        if not error_message or not error_message.strip():
            return "Unknown", 0.0
        label, score, _ = await self._store.find_or_create(error_message)
        return label, score

    @property
    def store(self) -> ErrorCategoryStore:
        """Expose the underlying store for management operations."""
        return self._store


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class PandaKnowledgeExtractor(ExtractionStrategy):
    """Structured extraction strategy for Panda-flavoured task data.

    * ``external_data`` key→value pairs → :class:`TaskFeatureNode`
    * ``task_data`` discrete fields     → :class:`TaskFeatureNode`
    * ``task_data`` free-form fields    → :class:`TaskContextNode`
    * ``task_data["ErrorMessage"]``     → :class:`TaskContextNode` (raw text)
                                          + :class:`TaskFeatureNode` ``ErrorCategory``
                                          via :class:`ErrorCategoryClassifier`
    * ``email_text``                    → :class:`CauseNode`, :class:`ResolutionNode`,
                                          and :class:`TaskContextNode` via LLM
    """

    _SUPPORTED_SYSTEMS: frozenset[str] = frozenset({"panda", "bamboo_panda"})

    def __init__(
        self,
        unstructured_keys: Optional[frozenset[str]] = None,
        error_classifier: Optional[ErrorCategoryClassifier] = None,
    ) -> None:
        self._unstructured_keys: frozenset[str] = (
            unstructured_keys if unstructured_keys is not None else UNSTRUCTURED_TASK_KEYS
        )
        self._error_classifier: ErrorCategoryClassifier = (
            error_classifier or ErrorCategoryClassifier()
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
            "ErrorMessage → persistent semantic error-category classification, "
            "email_text → LLM-extracted Cause / Resolution / Task_Context."
        )

    def supports_system(self, system_type: str) -> bool:
        return system_type.lower() in self._SUPPORTED_SYSTEMS

    async def extract(
        self,
        email_text: str = "",
        task_data: Optional[dict[str, Any]] = None,
        external_data: Optional[dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        """Extract a :class:`KnowledgeGraph` from all three input sources.

        * ``external_data`` and ``task_data`` are processed deterministically
          (no LLM call) into :class:`TaskFeatureNode` / :class:`TaskContextNode`.
        * ``email_text`` is passed to the LLM to extract :class:`CauseNode`,
          :class:`ResolutionNode`, and :class:`TaskContextNode` nodes.
        All results are merged into a single :class:`KnowledgeGraph`.
        """
        nodes: list = []
        relationships: list[GraphRelationship] = []

        # 1. external_data: every key→value → TaskFeatureNode
        for key, value in (external_data or {}).items():
            nodes.append(self._make_feature_node(
                attribute=str(key),
                value=str(value),
                description=f"External data field: {key}",
                source="external_data",
            ))

        # 2. task_data: classify each field
        for key, value in (task_data or {}).items():
            if key in self._unstructured_keys:
                ctx_node = self._make_context_node(
                    attribute=str(key),
                    prose=str(value) if value is not None else "",
                )
                nodes.append(ctx_node)

                if key == "ErrorMessage" and value:
                    category, confidence = await self._classify_error(str(value))
                    feat_node = self._make_feature_node(
                        attribute="ErrorCategory",
                        value=category,
                        description=(
                            f"Semantic classification of the ErrorMessage. "
                            f"Confidence: {confidence:.2f}"
                        ),
                        source="error_classifier",
                        extra_metadata={"classifier_confidence": confidence},
                    )
                    nodes.append(feat_node)
                    relationships.append(GraphRelationship(
                        source_id=ctx_node.name,
                        target_id=feat_node.name,
                        relation_type=RelationType.ASSOCIATED_WITH,
                        confidence=confidence,
                    ))
            else:
                nodes.append(self._make_feature_node(
                    attribute=str(key),
                    value=str(value) if value is not None else "",
                    description=f"Task data field: {key}",
                    source="task_data",
                ))

        # 3. email_text: LLM extracts Cause / Resolution / Task_Context
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
        """Call the LLM to extract Cause, Resolution, and Task_Context nodes
        from *email_text*.

        Returns:
            A ``(nodes, relationships)`` tuple ready to be merged into the
            main graph.
        """
        llm = get_llm()
        prompt = EMAIL_EXTRACTION_PROMPT.format(email_text=email_text)
        response = await llm.ainvoke(prompt)
        raw = self._parse_email_response(response.content)

        nodes = []
        node_name_map: dict[str, Any] = {}
        for node_data in raw.get("nodes", []):
            node = self._create_email_node(node_data)
            if node is not None:
                nodes.append(node)
                node_name_map[node_data["name"]] = node

        relationships: list[GraphRelationship] = []
        for rel_data in raw.get("relationships", []):
            src_name = rel_data.get("source_name")
            tgt_name = rel_data.get("target_name")
            if src_name not in node_name_map or tgt_name not in node_name_map:
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
                source_id=node_name_map[src_name].name,
                target_id=node_name_map[tgt_name].name,
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
        """Parse the LLM JSON response, tolerating markdown code fences."""
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
        """Instantiate a Cause, Resolution, or TaskContext node from LLM output.

        Any other node_type is silently ignored (the prompt restricts the LLM
        to these three, but we guard defensively).
        """
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
            "PandaKnowledgeExtractor: node_type '%s' is not permitted in email "
            "extraction — skipped",
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
        """Delegate to the :class:`ErrorCategoryClassifier`."""
        try:
            return await self._error_classifier.classify(error_message)
        except Exception as exc:  # pragma: no cover – network / API failures
            logger.warning("Error classification failed: %s", exc)
            return "Unknown", 0.0
