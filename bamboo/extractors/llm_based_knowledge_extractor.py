"""LLM-based extraction strategy."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.extractors.base import ExtractionStrategy
from bamboo.llm import EXTRACTION_PROMPT, get_llm
from bamboo.models.graph_element import (
    CauseNode,
    ComponentNode,
    EnvironmentNode,
    ErrorNode,
    GraphRelationship,
    NodeType,
    RelationType,
    ResolutionNode,
    TaskFeatureNode,
)
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)


class LLMBasedKnowledgeExtractor(ExtractionStrategy):
    """LLM-based extraction strategy using prompts.

    This strategy uses an LLM to understand unstructured text and extract
    knowledge graphs with natural language understanding.
    """

    def __init__(self):
        """Initialize LLM extraction strategy."""
        self.llm = get_llm()

    async def extract(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract using LLM-based approach."""
        # Combine all input sources
        input_data = self._prepare_input(email_text, task_data, external_data)

        # Use LLM to extract structured graph
        prompt = EXTRACTION_PROMPT.format(input_data=input_data)

        messages = [
            SystemMessage(
                content="You are an expert at extracting structured knowledge graphs from unstructured data."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        extracted_data = self._parse_llm_response(response.content)

        # Convert to our models
        graph = self._build_graph(extracted_data)

        logger.info(
            f"Extracted graph with {len(graph.nodes)} nodes "
            f"and {len(graph.relationships)} relationships using LLM strategy"
        )

        return graph

    def supports_system(self, system_type: str) -> bool:
        """LLM strategy supports generic/unstructured systems."""
        # LLM can handle any system type
        return True

    @property
    def name(self) -> str:
        """Strategy name."""
        return "llm"

    @property
    def description(self) -> str:
        """Strategy description."""
        return "LLM-based extraction using natural language understanding"

    def _prepare_input(
        self,
        email_text: str,
        task_data: dict[str, Any],
        external_data: dict[str, Any],
    ) -> str:
        """Prepare combined input for LLM."""
        sections = []

        if email_text:
            sections.append(f"EMAIL THREAD:\n{email_text}")

        if task_data:
            sections.append(f"TASK DATA:\n{json.dumps(task_data, indent=2)}")

        if external_data:
            sections.append(
                f"EXTERNAL INFORMATION:\n{json.dumps(external_data, indent=2)}"
            )

        return "\n\n".join(sections)

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response: {response}")
            return {"nodes": [], "relationships": []}

    def _build_graph(self, extracted_data: dict[str, Any]) -> KnowledgeGraph:
        """Build KnowledgeGraph from extracted data."""
        nodes = []
        node_map = {}  # Map node names to IDs for relationship building

        # Create nodes
        for node_data in extracted_data.get("nodes", []):
            node = self._create_node(node_data)
            if node:
                nodes.append(node)
                node_map[node_data["name"]] = len(nodes) - 1

        # Create relationships
        relationships = []
        for rel_data in extracted_data.get("relationships", []):
            source_name = rel_data.get("source_name")
            target_name = rel_data.get("target_name")

            if source_name in node_map and target_name in node_map:
                source_node = nodes[node_map[source_name]]
                target_node = nodes[node_map[target_name]]

                relationship = GraphRelationship(
                    source_id=source_node.id or source_node.name,
                    target_id=target_node.id or target_node.name,
                    relation_type=RelationType(rel_data["relation_type"]),
                    confidence=rel_data.get("confidence", 1.0),
                    properties=rel_data.get("properties", {}),
                )
                relationships.append(relationship)

        return KnowledgeGraph(nodes=nodes, relationships=relationships)

    def _create_node(self, node_data: dict[str, Any]):
        """Create appropriate node type from data."""
        node_type = NodeType(node_data["node_type"])

        base_fields = {
            "name": node_data["name"],
            "description": node_data["description"],
            "metadata": node_data.get("metadata", {}),
        }

        if node_type == NodeType.ERROR:
            return ErrorNode(
                **base_fields,
                error_code=node_data.get("error_code"),
                severity=node_data.get("severity"),
            )
        elif node_type == NodeType.ENVIRONMENT:
            return EnvironmentNode(
                **base_fields,
                category=node_data.get("category"),
            )
        elif node_type == NodeType.TASK_FEATURE:
            return TaskFeatureNode(
                **base_fields,
                feature_type=node_data.get("feature_type"),
            )
        elif node_type == NodeType.COMPONENT:
            return ComponentNode(
                **base_fields,
                system=node_data.get("system"),
                version=node_data.get("version"),
            )
        elif node_type == NodeType.CAUSE:
            return CauseNode(
                **base_fields,
                confidence=node_data.get("confidence", 1.0),
                frequency=node_data.get("frequency", 1),
            )
        elif node_type == NodeType.RESOLUTION:
            return ResolutionNode(
                **base_fields,
                steps=node_data.get("steps", []),
                success_rate=node_data.get("success_rate"),
                estimated_duration=node_data.get("estimated_duration"),
            )

        return None
