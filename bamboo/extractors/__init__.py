"""Extraction strategies for knowledge graph generation."""

from bamboo.extractors.factory import (
    get_extraction_strategy,
    list_extraction_strategies,
    register_extraction_strategy,
)
from bamboo.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor

__all__ = [
    "get_extraction_strategy",
    "list_extraction_strategies",
    "register_extraction_strategy",
    "KnowledgeGraphExtractor",
]
