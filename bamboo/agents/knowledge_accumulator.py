"""Knowledge accumulation agent for populating databases."""

import json
import logging
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.llm import (
    SUMMARIZATION_PROMPT,
    get_embeddings,
    get_llm,
)
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

        # Derive a stable graph_id from the task ID so that re-processing
        # the same task overwrites existing vectors rather than duplicating them.
        task_id = (task_data or {}).get("task_id")
        graph_id = (
            self._deterministic_id("graph", task_id) if task_id else str(uuid.uuid4())
        )

        # Step 1: Extract knowledge graph (IDs assigned inside extractor)
        graph = await self.extractor.extract_from_sources(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
        )
        graph.metadata["graph_id"] = graph_id

        # Step 2: Store in Graph Database
        await self._store_graph(graph)

        # Step 3: Generate summary
        summary = await self._generate_summary(graph)

        # Step 4: Store in Vector Database
        key_insights = await self._extract_key_insights(graph)
        await self._store_in_vector_db(graph, summary, key_insights)

        extracted_knowledge = ExtractedKnowledge(
            graph=graph,
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

    async def _extract_key_insights(
        self, graph: KnowledgeGraph
    ) -> list[dict[str, Any]]:
        """Collect nodes that carry unstructured prose worth semantic indexing.

        Indexed node types:
        - ``Task_Context``: free-form prose fields (steps, user reports, etc.)
        - ``Symptom``: description holds the raw error message text, which is
          worth semantic search even though the node's canonical name is the
          clean error category.

        Cause and Resolution nodes are intentionally excluded: their canonical
        names are already precisely indexed in the graph database.
        """
        _INDEXABLE = {"Task_Context", "Symptom"}
        insights = []
        for node in graph.nodes:
            if node.description and node.node_type.value in _INDEXABLE:
                insights.append(
                    {
                        "node_id": node.id,
                        "section": node.node_type.value,
                        "content": node.description,
                    }
                )
        return insights

    async def _store_in_vector_db(
        self,
        graph: KnowledgeGraph,
        summary: str,
        key_insights: list[dict[str, Any]],
    ):
        """Store unstructured node descriptions and the summary in the vector database.

        Every entry is tagged with a ``graph_id`` so that, during retrieval,
        a two-step query can be used:
          1. Search unstructured descriptions to find semantically similar graphs.
          2. Fetch the Summary entries for those graph_ids to get the full picture.
        """
        logger.info("Storing knowledge in Vector Database")

        graph_id = graph.metadata["graph_id"]

        # Store unstructured node descriptions
        for item in key_insights:
            # Deterministic ID: re-processing the same graph overwrites the
            # existing point rather than inserting a duplicate.
            vector_id = self._deterministic_id(
                graph_id, item["section"], item["node_id"]
            )
            embedding = await self.embeddings.aembed_query(item["content"])
            await self.vector_db.upsert_section_vector(
                vector_id=vector_id,
                embedding=embedding,
                content=item["content"],
                section=item["section"],
                metadata={
                    "graph_id": graph_id,
                    "graph_node_id": item["node_id"],
                },
            )

        # Store the summary — always present so every graph is retrievable
        summary_vector_id = self._deterministic_id(graph_id, "Summary")
        summary_embedding = await self.embeddings.aembed_query(summary)
        await self.vector_db.upsert_section_vector(
            vector_id=summary_vector_id,
            embedding=summary_embedding,
            content=summary,
            section="Summary",
            metadata={"graph_id": graph_id},
        )

        logger.info(
            f"Stored {len(key_insights)} unstructured descriptions and 1 summary "
            f"for graph {graph_id}"
        )

    @staticmethod
    def _deterministic_id(*parts: str) -> str:
        """Generate a deterministic UUID from the given parts.

        Using UUID5 (SHA-1 namespace hash) so that upserting the same
        graph_id + section + node_id combination always resolves to the
        same point ID in the vector database.
        """
        key = ":".join(str(p) for p in parts)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
