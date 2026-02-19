"""Extract knowledge graphs from unstructured data using pluggable strategies."""

import json
import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.extractors import get_extraction_strategy
from bamboo.llm import CANONICALIZATION_PROMPT, get_llm
from bamboo.models.graph_element import BaseNode
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
        self.llm = get_llm()
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

    async def canonicalize(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Canonicalize node names in the graph using LLM.

        Merges nodes that refer to the same concept and assigns stable IDs,
        so the resulting graph can be matched against the canonical names
        already stored in the graph database.
        """
        logger.info("Canonicalizing graph nodes")

        canonical_nodes = []
        node_id_map: dict[str, str] = {}

        nodes_by_type: dict = {}
        for node in graph.nodes:
            nodes_by_type.setdefault(node.node_type, []).append(node)

        for node_type, nodes in nodes_by_type.items():
            type_canonical_nodes: dict[str, BaseNode] = {}

            for node in nodes:
                canonical_name = await self._get_canonical_name(
                    node, list(type_canonical_nodes.values())
                )

                if canonical_name in type_canonical_nodes:
                    existing = type_canonical_nodes[canonical_name]
                    node_id_map[node.name] = existing.id or existing.name
                else:
                    node.name = canonical_name
                    if not node.id:
                        node.id = str(uuid.uuid4())
                    type_canonical_nodes[canonical_name] = node
                    canonical_nodes.append(node)
                    node_id_map[node.name] = node.id

        canonical_relationships = []
        for rel in graph.relationships:
            if rel.source_id in node_id_map and rel.target_id in node_id_map:
                rel.source_id = node_id_map[rel.source_id]
                rel.target_id = node_id_map[rel.target_id]
                canonical_relationships.append(rel)

        return KnowledgeGraph(
            nodes=canonical_nodes,
            relationships=canonical_relationships,
            metadata=graph.metadata,
        )

    async def _get_canonical_name(
        self, node: BaseNode, existing_nodes: list[BaseNode]
    ) -> str:
        """Get canonical name for a node using LLM."""
        if not existing_nodes:
            return node.name

        existing_nodes_str = "\n".join(
            [f"- {n.name}: {n.description}" for n in existing_nodes[:10]]
        )

        prompt = CANONICALIZATION_PROMPT.format(
            node_type=node.node_type.value,
            existing_nodes=existing_nodes_str,
            new_node_name=node.name,
            new_node_description=node.description,
        )

        messages = [
            SystemMessage(content="You are an expert at canonicalizing knowledge."),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            result = json.loads(response.content)
            return result["canonical_name"]
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Failed to canonicalize node {node.name}, using original")
            return node.name

