"""Knowledge graph extractor: assigns IDs and delegates to the active strategy.

This module provides :class:`KnowledgeGraphExtractor`, the single entry point
used by both the knowledge accumulator and the reasoning agent to produce a
:class:`KnowledgeGraph` from raw incident data.  The actual extraction logic
lives in the :class:`~bamboo.extractors.base.ExtractionStrategy` implementation
selected at runtime via :func:`~bamboo.extractors.factory.get_extraction_strategy`.
"""

import logging
import uuid
from typing import Any

from bamboo.extractors import get_extraction_strategy
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """Thin orchestrator that delegates extraction to a pluggable strategy.

    Responsibilities:
    - Select and hold the active :class:`~bamboo.extractors.base.ExtractionStrategy`.
    - Call :meth:`~bamboo.extractors.base.ExtractionStrategy.extract` with the
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
        logger.info("KnowledgeGraphExtractor: using strategy '%s'", self.strategy.name)

    async def extract_from_sources(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract a knowledge graph and assign stable node IDs.

        Delegates extraction to the configured strategy, then ensures every
        returned node has a non-empty ``id`` field (UUIDs are assigned lazily
        so strategies do not need to manage IDs themselves).

        Args:
            email_text:    Email thread or communication text.
            task_data:     Structured task/issue data as a flat dict.
            external_data: External metadata as a flat dict.

        Returns:
            :class:`KnowledgeGraph` with all nodes carrying stable UUIDs.
        """
        graph = await self.strategy.extract(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
        )

        for node in graph.nodes:
            if not node.id:
                node.id = str(uuid.uuid4())

        return graph
