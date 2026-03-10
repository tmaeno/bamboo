"""Knowledge accumulation agent: extracts, stores, and indexes incident graphs.

:class:`KnowledgeAccumulator` is the first half of the Bamboo pipeline.  Given
the raw data for one resolved incident (email thread, task fields, external
metadata) it:

1. Extracts a :class:`~bamboo.models.knowledge_entity.KnowledgeGraph` using
   the configured :class:`~bamboo.extractors.base.ExtractionStrategy`.
2. Stores nodes and relationships in **Neo4j** (graph database), merging on
   canonical names to avoid duplicate nodes.
3. Generates a narrative summary with the LLM.
4. Indexes unstructured node descriptions and the summary in **Qdrant** (vector
   database) so they are retrievable by semantic similarity.
"""

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
    """Extracts knowledge from a resolved incident and persists it to both DBs.

    Args:
        graph_db:  Connected :class:`GraphDatabaseClient`.
        vector_db: Connected :class:`VectorDatabaseClient`.
    """

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        vector_db: VectorDatabaseClient,
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self._llm = None        # lazy — initialised on first use
        self._embeddings = None # lazy — initialised on first use (not needed for dry-run)
        self.extractor = KnowledgeGraphExtractor()

    # ------------------------------------------------------------------
    # Lazy accessors — heavy imports deferred until actually needed
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    async def process_knowledge(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
        task_logs: dict[str, str] = None,
        job_logs: dict[str, str] = None,
        jobs_data: list[dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> ExtractedKnowledge:
        """Process one resolved incident and persist the extracted knowledge.

        A deterministic ``graph_id`` is derived from the composite key
        ``(task_data["taskID"], task_data["status"])`` so that the same task in
        different failure states is stored as a separate incident, while
        re-processing the exact same ``(taskID, status)`` pair overwrites
        existing vectors rather than creating duplicates.

        If only ``taskID`` is present (no ``status``) the id falls back to the
        ``taskID`` alone for backward compatibility.

        Args:
            email_text:    Email thread for the incident.
            task_data:     Structured task fields.  ``taskID`` and ``status``
                           together form the composite unique identifier used
                           to derive ``graph_id``.
            external_data: External environmental factors.
            task_logs:     *Task-level* log output keyed by source name
                           (e.g. ``{"jedi": "...", "harvester": "..."}``).
                           Extracted nodes are tagged ``log_level="task"``.
            job_logs:      *Job-level* log output keyed by a stable source name
                           (e.g. ``{"pilot": "...", "payload": "..."}``, NOT a
                           raw PanDA job ID).
                           Extracted nodes are tagged ``log_level="job"``.
            jobs_data:     List of raw job attribute dicts used for aggregated
                           :class:`~bamboo.models.graph_element.JobFeatureNode`
                           extraction.
            dry_run:       When ``True``, extraction and summarisation are
                           performed normally but **no data is written** to
                           either the graph database or the vector database.
                           Useful for previewing what would be stored before
                           committing.

        Returns:
            :class:`~bamboo.models.knowledge_entity.ExtractedKnowledge` with
            the graph, summary, and key insights.
        """
        logger.info("KnowledgeAccumulator: starting knowledge extraction")

        task_id = (task_data or {}).get("taskID")
        task_status = (task_data or {}).get("status")
        if task_id and task_status:
            graph_id = self._deterministic_id("graph", task_id, task_status)
        elif task_id:
            graph_id = self._deterministic_id("graph", task_id)
        else:
            graph_id = str(uuid.uuid4())

        graph = await self.extractor.extract_from_sources(
            email_text=email_text,
            task_data=task_data,
            external_data=external_data,
            task_logs=task_logs,
            job_logs=job_logs,
            jobs_data=jobs_data,
        )
        graph.metadata["graph_id"] = graph_id

        if dry_run:
            logger.info(
                "KnowledgeAccumulator: dry-run mode — skipping all database writes"
            )
        else:
            await self._store_graph(graph)

        summary = await self._generate_summary(graph)
        key_insights = await self._extract_key_insights(graph)

        if not dry_run:
            await self._store_in_vector_db(graph, summary, key_insights)

        logger.info(
            "KnowledgeAccumulator: extraction completed for graph '%s' "
            "(task_id=%s, task_status=%s)",
            graph_id,
            task_id,
            task_status,
        )
        return ExtractedKnowledge(
            graph=graph,
            summary=summary,
            key_insights=key_insights,
            source_references=[],
            extraction_metadata={
                "has_email": bool(email_text),
                "has_task_data": bool(task_data),
                "has_external_data": bool(external_data),
                "has_task_logs": bool(task_logs),
                "has_job_logs": bool(job_logs),
                "has_jobs_data": bool(jobs_data),
                "jobs_count": len(jobs_data) if jobs_data else 0,
            },
        )

    async def _store_graph(self, graph: KnowledgeGraph):
        """Persist graph nodes and relationships to Graph database.

        Nodes are merged by canonical name via
        :meth:`GraphDatabaseClient.get_or_create_canonical_node`.
        Relationship source/target IDs are remapped to the actual stored IDs
        before insertion.
        """
        logger.info(
            "KnowledgeAccumulator: storing %d nodes in graph DB", len(graph.nodes)
        )

        node_ids: dict[str, str] = {}
        for node in graph.nodes:
            node_id = await self.graph_db.get_or_create_canonical_node(node, node.name)
            node_ids[node.id or node.name] = node_id

        for rel in graph.relationships:
            rel.source_id = node_ids.get(rel.source_id, rel.source_id)
            rel.target_id = node_ids.get(rel.target_id, rel.target_id)
            await self.graph_db.create_relationship(rel)

        logger.info("KnowledgeAccumulator: graph stored successfully")

    async def _generate_summary(self, graph: KnowledgeGraph) -> str:
        """Ask the LLM to produce a narrative summary of the knowledge graph.

        The summary is stored in Qdrant so that users can retrieve full
        incident narratives via semantic search.
        """
        logger.info("KnowledgeAccumulator: generating graph summary")

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
        _INDEXABLE = {"Task_Context", "Symptom", "Job_Feature"}
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
        """Store node descriptions and the graph summary in Vector database.

        Every entry is tagged with ``graph_id`` so that the two-step retrieval
        pattern in :meth:`ReasoningNavigator._query_vector_database` can fetch the
        full summary once matching node descriptions are found.

        Deterministic point IDs are used so that re-processing the same graph
        overwrites existing points rather than inserting duplicates.
        """
        logger.info("KnowledgeAccumulator: storing vectors in vector DB")

        graph_id = graph.metadata["graph_id"]

        for item in key_insights:
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
            "KnowledgeAccumulator: stored %d node descriptions + 1 summary for graph '%s'",
            len(key_insights),
            graph_id,
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
