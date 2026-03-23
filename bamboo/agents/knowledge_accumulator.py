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
    get_summary_llm,
)
from bamboo.models.knowledge_entity import ExtractedKnowledge, KnowledgeGraph
from bamboo.utils.narrator import say, thinking

logger = logging.getLogger(__name__)


_MAX_REVIEW_RETRIES = 2


class KnowledgeAccumulator:
    """Extracts knowledge from a resolved incident and persists it to both DBs.

    Args:
        graph_db:  Connected :class:`GraphDatabaseClient`.
        vector_db: Connected :class:`VectorDatabaseClient`.
        reviewer:  Optional :class:`~bamboo.agents.knowledge_reviewer.KnowledgeReviewer`.
                   When provided, the extracted graph is evaluated before being
                   stored and re-extracted (up to :data:`_MAX_REVIEW_RETRIES`
                   times) if the reviewer finds issues.  Pass ``None`` to
                   skip the review gate entirely (default behaviour).
        explorer:  Optional :class:`~bamboo.agents.extra_source_explorer.ExtraSourceExplorer`.
                   When provided alongside *reviewer*, fires once on the first
                   reviewer rejection to fetch additional source data (log
                   files, retry chain context) before the re-extraction
                   attempt.  Has no effect when *reviewer* is ``None``.
    """

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        vector_db: VectorDatabaseClient,
        reviewer=None,
        explorer=None,
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self._reviewer = reviewer
        self._explorer = explorer
        self._llm = None  # lazy — initialised on first use
        self._embeddings = (
            None  # lazy — initialised on first use (not needed for dry-run)
        )
        self.extractor = KnowledgeGraphExtractor()

    # ------------------------------------------------------------------
    # Lazy accessors — heavy imports deferred until actually needed
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_summary_llm()
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

        from bamboo.agents.knowledge_reviewer import build_sources_summary

        # Local mutable copies — may be augmented by the explorer on first rejection.
        # Original caller dicts are never mutated.
        _task_logs = dict(task_logs or {})
        _external_data = dict(external_data or {})

        sources_summary = build_sources_summary(
            email_text=email_text,
            task_logs=_task_logs,
            job_logs=job_logs,
        )
        review_feedback = ""
        for attempt in range(_MAX_REVIEW_RETRIES + 1):
            graph = await self.extractor.extract_from_sources(
                email_text=email_text,
                task_data=task_data,
                external_data=_external_data,
                task_logs=_task_logs,
                job_logs=job_logs,
                jobs_data=jobs_data,
                review_feedback=review_feedback,
            )
            graph.metadata["graph_id"] = graph_id
            self._reconcile_cross_extractor_links(graph)

            if self._reviewer is None:
                break
            review_result = await self._reviewer.review(graph, sources_summary)
            if review_result.approved or attempt >= _MAX_REVIEW_RETRIES:
                if not review_result.approved:
                    logger.warning(
                        "KnowledgeAccumulator: reviewer not satisfied after %d attempt(s) — "
                        "storing best result (confidence=%.2f)",
                        attempt + 1,
                        review_result.confidence,
                    )
                break
            logger.info(
                "KnowledgeAccumulator: review pass %d/%d found %d issue(s) — retrying extraction",
                attempt + 1,
                _MAX_REVIEW_RETRIES,
                len(review_result.issues),
            )
            say(
                f"Review pass {attempt + 1}: {len(review_result.issues)} issue(s) found — "
                "retrying extraction with reviewer feedback."
            )

            # One-shot source exploration on the first rejection.
            if attempt == 0 and self._explorer is not None:
                say("Launching source explorer to fetch additional data...")
                exploration = await self._explorer.explore(
                    task_data=task_data or {},
                    review_issues=review_result.issues,
                )
                if exploration.task_logs:
                    _task_logs.update(exploration.task_logs)
                    say(
                        f"Explorer fetched {len(exploration.task_logs)} "
                        "additional log source(s)."
                    )
                if exploration.external_data:
                    _external_data.update(exploration.external_data)
                    say(
                        f"Explorer added {len(exploration.external_data)} "
                        "external data entry(ies)."
                    )
                if exploration.task_logs or exploration.external_data:
                    # Rebuild so reviewer sees the new sources on next pass.
                    sources_summary = build_sources_summary(
                        email_text=email_text,
                        task_logs=_task_logs,
                        job_logs=job_logs,
                    )
                logger.info(
                    "KnowledgeAccumulator: source explorer ran %d tool call(s)",
                    len(exploration.tool_calls),
                )

            review_feedback = review_result.feedback

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
                "explorer_ran": self._explorer is not None and self._reviewer is not None,
                "explorer_logs_added": len(_task_logs) - len(task_logs or {}),
                "explorer_external_added": len(_external_data) - len(external_data or {}),
            },
        )

    def _reconcile_cross_extractor_links(self, graph: KnowledgeGraph) -> None:
        """Create schema-defined edges that span extractor boundaries.

        Each extractor (email, log, task-data) runs independently and never
        knows about nodes produced by the others.  After merging, this step
        adds the edges that were structurally impossible to emit during
        extraction:

            Symptom      -[indicate]->        Cause
            Task_Feature -[contribute_to]->   Cause
            Job_Feature  -[contribute_to]->   Cause
            Component    -[originated_from]-> Cause
            Environment  -[associated_with]-> Cause

        Edges that already exist (emitted by an LLM pass) are not duplicated.
        Inferred edges carry ``confidence=0.7`` to distinguish them from
        directly extracted ones.
        """
        from bamboo.models.graph_element import GraphRelationship, RelationType

        by_type: dict[str, list] = {}
        for node in graph.nodes:
            by_type.setdefault(node.node_type.value, []).append(node)

        causes = by_type.get("Cause", [])
        if not causes:
            return

        existing = {
            (r.source_id, r.target_id, r.relation_type) for r in graph.relationships
        }

        pairs = [
            ("Symptom", RelationType.INDICATE),
            ("Task_Feature", RelationType.CONTRIBUTE_TO),
            ("Job_Feature", RelationType.CONTRIBUTE_TO),
            ("Component", RelationType.ORIGINATED_FROM),
            ("Environment", RelationType.ASSOCIATED_WITH),
        ]

        new_rels = []
        for src_type, rel_type in pairs:
            for src_node in by_type.get(src_type, []):
                for cause in causes:
                    key = (src_node.name, cause.name, rel_type)
                    if key not in existing:
                        new_rels.append(
                            GraphRelationship(
                                source_id=src_node.name,
                                target_id=cause.name,
                                relation_type=rel_type,
                                confidence=0.7,
                            )
                        )
                        existing.add(key)

        graph.relationships.extend(new_rels)
        if new_rels:
            say(
                f"Reconciled {len(new_rels)} cross-extractor link(s) between nodes from different sources."
            )
            logger.info(
                "KnowledgeAccumulator: reconciled %d cross-extractor link(s)",
                len(new_rels),
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
        say("Generating a narrative summary of the extracted graph...")

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

        with thinking("Working"):
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
