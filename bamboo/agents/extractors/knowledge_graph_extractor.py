"""Knowledge graph extractor: assigns IDs and delegates to the active strategy.

This module provides :class:`KnowledgeGraphExtractor`, the single entry point
used by both the knowledge accumulator and the reasoning agent to produce a
:class:`KnowledgeGraph` from raw incident data.  The actual extraction logic
lives in the :class:`~bamboo.agents.extractors.base.ExtractionStrategy` implementation
selected at runtime via :func:`~bamboo.agents.extractors.factory.get_extraction_strategy`.
"""

import logging
import uuid
from typing import Any

from bamboo.agents.extractors import get_extraction_strategy
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """Thin orchestrator that delegates extraction to a pluggable strategy.

    Responsibilities:
    - Select and hold the active :class:`~bamboo.agents.extractors.base.ExtractionStrategy`.
    - Call :meth:`~bamboo.agents.extractors.base.ExtractionStrategy.extract` with the
      raw input data.
    - Assign a stable UUID to every node that does not already have one.

    The strategy is selected once at construction time.  To use a different
    strategy create a new ``KnowledgeGraphExtractor`` instance.

    Args:
        strategy: Strategy name (e.g. ``"panda"``, ``"llm"``).  When ``None``
                  the value of the ``EXTRACTION_STRATEGY`` configuration key
                  is used.
    """

    def __init__(self, strategy: str = None):
        self.strategy = get_extraction_strategy(strategy)
        self._desc_cache: dict[str, str] = {}
        logger.info("KnowledgeGraphExtractor: using strategy '%s'", self.strategy.name)

    async def extract_from_sources(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
        task_logs: dict[str, str] = None,
        doc_hints: dict[str, str] = None,
    ) -> KnowledgeGraph:
        """Extract a knowledge graph and assign stable node IDs.

        Delegates extraction to the configured strategy, then ensures every
        returned node has a non-empty ``id`` field (UUIDs are assigned lazily
        so strategies do not need to manage IDs themselves).

        Args:
            email_text:    Email thread or communication text.
            task_data:     Structured task/issue data as a flat dict.
            external_data: External metadata as a flat dict.
            task_logs:     *Task-level* log output keyed by source name
                           (e.g. ``{"jedi": "...", "harvester": "..."}``).
                           Extracted nodes are tagged ``log_level="task"``.

        Returns:
            :class:`KnowledgeGraph` with all nodes carrying stable UUIDs.
        """
        graph = await self.strategy.extract(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
            task_logs=task_logs,
            doc_hints=doc_hints,
        )

        # Deduplicate nodes by (type, name) — the LLM occasionally emits the
        # same logical node twice with identical or near-identical names.
        # `name_remap` only contains entries for actual duplicates so that
        # relationship endpoints remain as human-readable names throughout
        # (surviving nodes are NOT remapped to their UUID).
        seen: dict[tuple, str] = {}  # (type, name) -> canonical node.name
        deduped_nodes = []
        name_remap: dict[str, str] = {}  # duplicate name -> canonical name

        for node in graph.nodes:
            if not node.id:
                node.id = str(uuid.uuid4())
            key = (node.node_type.value, node.name)
            if key in seen:
                name_remap[node.name] = seen[key]
            else:
                seen[key] = node.name
                deduped_nodes.append(node)

        graph.nodes = deduped_nodes

        # Remap relationship endpoints for merged duplicates and drop
        # self-loops that deduplication may have introduced.
        deduped_rels = []
        seen_rels: set[tuple] = set()
        for rel in graph.relationships:
            src = name_remap.get(rel.source_id, rel.source_id)
            tgt = name_remap.get(rel.target_id, rel.target_id)
            rel_key = (src, tgt, rel.relation_type)
            if src == tgt or rel_key in seen_rels:
                continue
            rel.source_id = src
            rel.target_id = tgt
            seen_rels.add(rel_key)
            deduped_rels.append(rel)

        graph.relationships = deduped_rels

        from bamboo.utils.canonicalize import canonicalize_descriptions  # noqa: PLC0415

        # Canonicalize free-form prose descriptions.
        # Task_Feature (attribute=value) and Environment nodes have short
        # structured names with no prose to strip.  Component names are
        # structured system labels.  Symptom *names* are already canonical via
        # ErrorCategoryClassifier; their *descriptions* (raw error messages /
        # LLM summaries from logs) still need incident-specific tokens stripped.
        _DESC_CANONICALIZE_TYPES = frozenset({"Cause", "Resolution", "Procedure", "Task_Context", "Symptom"})
        prose_nodes = [n for n in graph.nodes if n.node_type.value in _DESC_CANONICALIZE_TYPES]
        await canonicalize_descriptions(prose_nodes, cache=self._desc_cache)

        return graph
