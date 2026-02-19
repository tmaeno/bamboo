"""Knowledge accumulation agent for populating databases."""
import copy
import json
import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.agents.knowledge_graph_extractor import KnowledgeGraphExtractor
from bamboo.llm import (
    CANONICALIZATION_PROMPT,
    SUMMARIZATION_PROMPT,
    get_embeddings,
    get_llm,
)
from bamboo.models.graph_element import BaseNode, NodeType
from bamboo.models.knowledge_entity import ExtractedKnowledge, KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeAccumulator:
    """Agent for extracting knowledge and populating databases."""

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        vector_db: VectorDatabaseClient,
    ):
        """Initialize knowledge extraction agent."""
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.extractor = KnowledgeGraphExtractor()

    async def process_knowledge(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> ExtractedKnowledge:
        """Extract knowledge from sources and store in databases."""
        logger.info("Starting knowledge extraction process")

        # Step 1: Extract knowledge graph
        graph = await self.extractor.extract_from_sources(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
        )

        # Step 2: Canonicalize nodes
        canonical_graph = await self._canonicalize_graph(graph)

        # Step 3: Store in Graph Database
        await self._store_graph(canonical_graph)

        # Step 4: Generate summary
        summary = await self._generate_summary(graph)

        # Step 5: Store in Vector Database
        key_insights = await self._extract_key_insights(graph)
        await self._store_in_vector_db(
            graph,
            summary,
            key_insights,
        )

        extracted_knowledge = ExtractedKnowledge(
            graph=canonical_graph,
            summary=summary,
            key_insights=key_insights,
            source_references=[],
            extraction_metadata={
                "has_email": bool(email_text),
                "has_task_data": bool(task_data),
                "has_external_data": bool(external_data),
            },
        )

        logger.info("Knowledge extraction completed successfully")
        return extracted_knowledge

    async def _canonicalize_graph(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Canonicalize node names using LLM."""
        logger.info("Canonicalizing graph nodes")

        canonical_nodes = []
        node_id_map = {}  # Map old IDs to new canonical IDs

        # Group nodes by type for efficient canonicalization
        nodes_by_type = {}
        for node in graph.nodes:
            node_type = node.node_type
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)

        # Process each node type
        for node_type, nodes in nodes_by_type.items():
            type_canonical_nodes = {}  # Track canonical nodes of this type

            for node in nodes:
                # Get existing nodes of this type from database
                canonical_name = await self._get_canonical_name(
                    node, list(type_canonical_nodes.values())
                )

                # Check if we already have this canonical node
                if canonical_name in type_canonical_nodes:
                    # Map to existing canonical node
                    existing_node = type_canonical_nodes[canonical_name]
                    node_id_map[node.name] = existing_node.id or existing_node.name
                else:
                    # Create new canonical node
                    node.name = canonical_name
                    if not node.id:
                        node.id = str(uuid.uuid4())
                    type_canonical_nodes[canonical_name] = node
                    canonical_nodes.append(node)
                    node_id_map[node.name] = node.id

        # Update relationships with canonical IDs
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
            return node.name  # No existing nodes to compare

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

    async def _store_graph(self, graph: KnowledgeGraph):
        """Store knowledge graph in Neo4j."""
        logger.info(f"Storing {len(graph.nodes)} nodes in Neo4j")

        # Store nodes
        node_ids = {}
        for node in graph.nodes:
            node_id = await self.graph_db.get_or_create_canonical_node(node, node.name)
            node_ids[node.id or node.name] = node_id

        # Store relationships
        for rel in graph.relationships:
            rel.source_id = node_ids.get(rel.source_id, rel.source_id)
            rel.target_id = node_ids.get(rel.target_id, rel.target_id)
            await self.graph_db.create_relationship(rel)

        logger.info("Graph stored successfully in Graph Database")

    async def _generate_summary(self, graph: KnowledgeGraph) -> str:
        """Generate entry of knowledge graph using LLM."""
        logger.info("Generating knowledge graph entry")

        graph_data = {
            "nodes": [
                {
                    "type": node.node_type.value,
                    "name": node.name,
                    "description": node.description,
                }
                for node in graph.nodes
            ],
            "relationships": [
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relation_type.value,
                }
                for rel in graph.relationships
            ],
        }

        prompt = SUMMARIZATION_PROMPT.format(
            graph_data=json.dumps(graph_data, indent=2)
        )

        messages = [
            SystemMessage(content="You are an expert at creating technical summaries."),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _extract_key_insights(self, graph: KnowledgeGraph) -> dict[str, list[dict[str, str]]]:
        """Extract key insights from the graph."""
        insights = {}

        # Extract main causes
        causes = [node for node in graph.nodes if node.node_type == NodeType.CAUSE]
        for cause in causes[:3]:  # Top 3 causes
            insights.setdefault(cause.node_type, [])
            insights[cause.node_type].append({"id": cause.id,
                                              "insight": f"Cause: {cause.name}. Description: {cause.description}"})

        # Extract resolutions
        resolutions = [
            node for node in graph.nodes if node.node_type == NodeType.RESOLUTION
        ]
        for resolution in resolutions[:3]:  # Top 3 resolutions
            insights.setdefault(resolution.node_type, [])
            insights[resolution.node_type].append({"id": resolution.id,
                                                   "insight": f"Resolution: {cause.name}. Description: {cause.description}"})

        return insights

    async def _store_in_vector_db(
        self,
        graph: KnowledgeGraph,
        summary: str,
        key_insights: dict[str, list[dict[str, str]]],
    ):
        """Store knowledge in vector database."""
        logger.info("Storing knowledge in Vector Database")

        local_key_insights = copy.deepcopy(key_insights)
        local_key_insights.update({"Summary": [{"insight": summary}]})

        # Store key insights and summary in vector database with section metadata,
        # allowing structured retrieval later based on sections
        for section, insights in local_key_insights.items():
            for item in insights:
                insight = item["insight"]
                node_id = item.get("id")
                # Generate embedding
                embedding = await self.embeddings.aembed_query(insight)

                # Store in vector database
                vector_id = str(uuid.uuid4())
                await self.vector_db.upsert_section_vector(
                    vector_id=vector_id,
                    embedding=embedding,
                    content=insight,
                    section=section,
                    metadata={
                        "graph_node_id": node_id,
                    }
                )

        logger.info("Knowledge stored successfully in vector database")
