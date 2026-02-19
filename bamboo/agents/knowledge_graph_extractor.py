"""Extract knowledge graphs from unstructured data using pluggable strategies."""

import logging
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
    """

    def __init__(self, strategy: str = None):
        """Initialize graph extractor with optional strategy name.

        Args:
            strategy: Extraction strategy name (llm, rule_based, jira, github, etc.)
                     If None, uses EXTRACTION_STRATEGY from configuration.
        """
        self.strategy = get_extraction_strategy(strategy)
        logger.info(f"Initialized GraphExtractor with strategy: {self.strategy.name}")

    async def extract_from_sources(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract knowledge graph from multiple sources.

        Args:
            email_text: Email thread or communication text
            task_data: Structured task/issue data
            external_data: External information (logs, metrics, etc.)

        Returns:
            KnowledgeGraph with extracted nodes and relationships
        """
        return await self.strategy.extract(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
        )
