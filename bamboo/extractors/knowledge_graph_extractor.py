"""Extract knowledge graphs from unstructured data using pluggable strategies."""

import logging
import uuid
from typing import Any

from bamboo.extractors import get_extraction_strategy
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """Extracts knowledge graphs from text and structured data.

    Uses pluggable extraction strategies selected via EXTRACTION_STRATEGY:
    - LLM-based extraction for unstructured/semi-structured data
    - Rule-based extraction for structured task management systems
    - System-specific strategies (Jira, GitHub, etc.)

    Each strategy is responsible for producing normalised node names directly.
    LLM-based strategies embed canonicalization rules in their extraction prompt.
    Rule-based strategies work from structured source data that is already clean.
    """

    def __init__(self, strategy: str = None):
        """Initialize graph extractor with optional strategy name."""
        self.strategy = get_extraction_strategy(strategy)
        logger.info(f"Initialized GraphExtractor with strategy: {self.strategy.name}")

    async def extract_from_sources(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract knowledge graph from multiple sources and assign stable node IDs.

        Args:
            email_text: Email thread or communication text
            task_data: Structured task/issue data
            external_data: External information (logs, metrics, etc.)

        Returns:
            KnowledgeGraph with extracted nodes (each with a stable ID) and relationships
        """
        graph = await self.strategy.extract(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
        )

        # Assign stable IDs to any nodes that don't have one yet
        for node in graph.nodes:
            if not node.id:
                node.id = str(uuid.uuid4())

        return graph
