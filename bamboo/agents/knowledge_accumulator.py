"""Knowledge accumulation agent: extracts, stores, and indexes incident graphs.

:class:`KnowledgeAccumulator` is the first half of the Bamboo pipeline.  Given
the raw data for one resolved incident (email thread, task fields, external
metadata) it:

1. Extracts a :class:`~bamboo.models.knowledge_entity.KnowledgeGraph` using
   the configured :class:`~bamboo.agents.extractors.base.ExtractionStrategy`.
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

from bamboo.agents.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.llm import (
    SUMMARIZATION_PROMPT,
    get_embeddings,
    get_summary_llm,
)
from bamboo.models.knowledge_entity import ExtractedKnowledge, KnowledgeGraph
from bamboo.utils.narrator import say, show_block, thinking

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
        max_review_retries: int = _MAX_REVIEW_RETRIES,
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self._reviewer = reviewer
        self._explorer = explorer
        self._max_review_retries = max_review_retries
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
                           :class:`~bamboo.models.graph_element.AggregatedJobFeatureNode`
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

        task_id = (task_data or {}).get("jediTaskID")
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

        # Pre-fetch PanDA documentation context before the first extraction.
        # Stored separately from task_logs — docs are domain hints, not execution logs.
        _doc_hints: dict[str, str] = {}
        if self._explorer is not None:
            _doc_hints = await self._prefetch_panda_docs(task_data or {})

        sources_summary = build_sources_summary(
            email_text=email_text,
            task_logs=_task_logs,
            job_logs=job_logs,
            doc_hints=_doc_hints,
        )

        review_feedback = ""
        review_result = None
        # Accumulate failure_dimension across all review passes so that
        # dimensions identified in early passes are not lost when a later pass
        # (e.g. the one that sees job data for the first time) produces a
        # different list.  Python selects feature nodes by concept tag, so the
        # LLM never needs to see or name individual feature nodes.
        _all_failure_dimensions: set[str] = set()
        for attempt in range(self._max_review_retries + 1):
            graph = await self.extractor.extract_from_sources(
                email_text=email_text,
                task_data=task_data,
                external_data=_external_data,
                task_logs=_task_logs,
                job_logs=job_logs,
                jobs_data=jobs_data,
                review_feedback=review_feedback,
                doc_hints=_doc_hints,
            )
            graph.metadata["graph_id"] = graph_id
            self._reconcile_cross_extractor_links(graph)

            if self._reviewer is None:
                break
            say(f"Running knowledge reviewer (attempt {attempt + 1}/{self._max_review_retries + 1})...")
            _available_tools = self._explorer._client.list_tools() if self._explorer else None
            review_result = await self._reviewer.review(
                graph, sources_summary, task_data=task_data,
                available_tools=_available_tools, doc_hints=_doc_hints,
            )
            # Accumulate failure dimensions from every review pass,
            # including the final (approved) one — must happen before any break.
            _all_failure_dimensions.update(review_result.failure_dimension)

            if review_result.approved or attempt >= self._max_review_retries:
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
                self._max_review_retries,
                len(review_result.issues),
            )
            say(
                f"Review pass {attempt + 1}: {len(review_result.issues)} issue(s) found — "
                "retrying extraction with reviewer feedback."
            )

            # On first rejection: fetch job data if reviewer requested it and
            # it hasn't been provided yet.  Jobs are fetched directly from PanDA
            # so that successful-but-resource-abusive scouts are included (unlike
            # get_failed_job_details which only returns unsuccessful jobs).
            if attempt == 0 and review_result.needs_job_data and jobs_data is None:
                jobs_data = await self._fetch_jobs_from_panda(task_data)

            # One-shot source exploration on the first rejection.
            if attempt == 0 and self._explorer is not None:
                say("Launching source explorer to fetch additional data...")
                exploration = await self._explorer.explore(
                    task_data=task_data or {},
                    review_issues=review_result.issues,
                    doc_hints=_doc_hints,
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
                        doc_hints=_doc_hints,
                    )
                logger.info(
                    "KnowledgeAccumulator: source explorer ran %d tool call(s)",
                    len(exploration.tool_calls),
                )

            review_feedback = review_result.feedback

        # Create explicit contribute_to edges for all Task_Feature nodes whose
        # concept tag matches a failure dimension declared by the reviewer.
        # Python selects by concept mechanically — the LLM only identifies the
        # dimension string, never individual node names.
        if _all_failure_dimensions:
            self._add_dimension_feature_edges(graph, _all_failure_dimensions)

        if dry_run:
            logger.info(
                "KnowledgeAccumulator: dry-run mode — skipping all database writes"
            )
        else:
            # Pre-check: if a Summary vector already exists for this graph_id the
            # task was previously processed.  Clean up its stale Neo4j contribution
            # before re-storing so old relationships don't accumulate.
            if self.vector_db is not None:
                existing = await self.vector_db.get_summaries_by_graph_ids([graph_id])
                if existing:
                    say(
                        f"Task previously stored (graph_id={graph_id}) — "
                        "removing stale Neo4j data before re-processing..."
                    )
                    counts = await self.graph_db.remove_graph_id(graph_id)
                    say(
                        f"Cleanup: {counts['rels_affected']} relationship(s) "
                        f"updated/removed, {counts['nodes_removed']} isolated "
                        "node(s) removed."
                    )
                    logger.info(
                        "KnowledgeAccumulator: re-processing cleanup for "
                        "graph_id=%s: %s",
                        graph_id,
                        counts,
                    )
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

    async def _prefetch_panda_docs(
        self, task_data: dict[str, Any]
    ) -> dict[str, str]:
        """Fetch PanDA documentation hints before the first extraction pass.

        Derives search queries from key ``task_data`` fields (status,
        errorDialog, splitRule sub-rule keys) and calls ``search_panda_docs``
        via PandaMcpClient.

        Returns a ``doc_hints`` dict: keys are query strings, values are
        plain rendered text (``"[Title] snippet\\n\\n[Title2] snippet2"``).
        Returns an empty dict on any error so the caller is unaffected.
        """
        import re

        queries: list[str] = []

        status = task_data.get("status", "")
        if status:
            queries.append(f"task status {status}")

        error_dialog = task_data.get("errorDialog", "") or ""
        if error_dialog:
            # Strip HTML tags and condense whitespace, then take the first
            # ~120 chars as a search phrase (avoids sending a wall of text).
            plain = re.sub(r"<[^>]+>", " ", error_dialog)
            plain = " ".join(plain.split())[:120]
            if plain:
                queries.append(plain)

        if not queries:
            return {}

        # Use PandaMcpClient directly — search_panda_docs is a built-in PanDA
        # tool and does not need connect().  Going through the composite client
        # would fail here because _tool_owner is only populated after connect().
        from bamboo.mcp.panda_mcp_client import PandaMcpClient  # noqa: PLC0415

        panda_client = PandaMcpClient()
        doc_hints: dict[str, str] = {}
        for query in queries:
            try:
                results = await panda_client.execute(
                    "search_panda_docs", query=query
                )
                if isinstance(results, list) and results:
                    rendered = "\n\n".join(
                        f"[{e.get('title', '')}] {e.get('snippet', '')}".strip()
                        for e in results if e.get("snippet")
                    )
                    if rendered:
                        doc_hints[query] = rendered
                        say(
                            f"Pre-fetched {len(results)} PanDA doc section(s) "
                            f"for query: {query!r}"
                        )
            except Exception as exc:
                logger.warning(
                    "KnowledgeAccumulator._prefetch_panda_docs: "
                    "search_panda_docs failed for query=%r: %s",
                    query,
                    exc,
                )

        # Look up splitRule sub-rule descriptions directly from task_params.rst.
        split_rule_str = task_data.get("splitRule", "") or ""
        if split_rule_str:
            from bamboo.mcp.panda_mcp_client import _fetch_task_params_table  # noqa: PLC0415

            table = await _fetch_task_params_table()
            if table:
                found: dict[str, str] = {}
                for sub_rule in split_rule_str.split(","):
                    sub_rule = sub_rule.strip()
                    if "=" in sub_rule:
                        key = sub_rule.split("=", 1)[0].strip()
                        if key and key not in found and key in table:
                            found[key] = table[key]
                if found:
                    doc_hints["splitRule params"] = "\n".join(
                        f"- {k}: {v}" for k, v in found.items()
                    )
                    say(
                        f"Looked up {len(found)} splitRule description(s) "
                        "from task_params.rst"
                    )

        return doc_hints

    async def _fetch_jobs_from_panda(
        self, task_data: dict[str, Any] | None
    ) -> list[dict[str, Any]] | None:
        """Fetch all job descriptions for the task from PanDA.

        Called when the reviewer sets ``needs_job_data=True`` and no
        ``jobs_data`` was supplied by the caller.  Returns ``None`` on any
        error so the caller can proceed without job data rather than crashing.
        """
        if not task_data:
            return None
        try:
            import asyncio as _asyncio

            from pandaclient import Client as _Client  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "KnowledgeAccumulator: pandaclient not installed — cannot fetch job data"
            )
            return None
        try:
            task_id_int = int(task_data["jediTaskID"])
            status, raw_jobs = await _asyncio.to_thread(
                _Client.get_job_descriptions, task_id_int
            )
            if status == 0 and isinstance(raw_jobs, list):
                say(f"Fetched {len(raw_jobs):,} job(s) from PanDA for job aggregation.")
                logger.info(
                    "KnowledgeAccumulator: fetched %d job(s) for task_id=%s",
                    len(raw_jobs),
                    task_id_int,
                )
                return raw_jobs
            logger.warning(
                "KnowledgeAccumulator: get_job_descriptions returned status=%s — skipping",
                status,
            )
            return None
        except Exception as exc:
            logger.warning(
                "KnowledgeAccumulator: failed to fetch job data: %s", exc
            )
            return None

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
            ("Symptom",     RelationType.INDICATE),
            ("Component",   RelationType.ORIGINATED_FROM),
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

        # Cause -[solved_by]-> Resolution: link every cause to every resolution as a
        # fallback.  The email extractor creates these when it can match the LLM's
        # relationship source/target names; this step covers cases where that matching
        # fails (e.g. name capitalisation mismatch after canonicalisation).
        resolutions = by_type.get("Resolution", [])
        for cause in causes:
            for res in resolutions:
                key = (cause.name, res.name, RelationType.SOLVED_BY)
                if key not in existing:
                    new_rels.append(
                        GraphRelationship(
                            source_id=cause.name,
                            target_id=res.name,
                            relation_type=RelationType.SOLVED_BY,
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

    def _add_dimension_feature_edges(
        self, graph: KnowledgeGraph, dimensions: set[str]
    ) -> None:
        """Create ``contribute_to`` edges for all feature nodes matching failure dimensions.

        The reviewer LLM declares which resource dimension(s) caused the failure
        (e.g. ``{"cpu"}``).  Python then selects every Task_Feature / Job_Feature
        node whose ``metadata["concept"]`` is in that set and connects it to all
        Cause nodes.  This removes individual node selection from the LLM entirely,
        eliminating the over-inclusion problem caused by indirect domain reasoning.

        Args:
            graph:      The current :class:`KnowledgeGraph`.
            dimensions: Concept strings returned in ``ReviewResult.failure_dimension``.
        """
        from bamboo.models.graph_element import GraphRelationship, RelationType

        feature_nodes = [
            n for n in graph.nodes
            if n.node_type.value in ("Task_Feature", "Job_Feature")
            and n.metadata.get("concept") in dimensions
        ]
        cause_nodes = [n for n in graph.nodes if n.node_type.value == "Cause"]

        if not feature_nodes or not cause_nodes:
            return

        existing = {
            (r.source_id, r.target_id, r.relation_type) for r in graph.relationships
        }
        new_rels = []
        for feat in feature_nodes:
            for cause in cause_nodes:
                key = (feat.name, cause.name, RelationType.CONTRIBUTE_TO)
                if key not in existing:
                    new_rels.append(
                        GraphRelationship(
                            source_id=feat.name,
                            target_id=cause.name,
                            relation_type=RelationType.CONTRIBUTE_TO,
                            confidence=0.9,
                        )
                    )
                    existing.add(key)

        graph.relationships.extend(new_rels)
        if new_rels:
            show_block(
                f"dimension-matched features ({', '.join(sorted(dimensions))})",
                "\n".join(f"• {n.node_type.value} '{n.name}'" for n in feature_nodes),
            )
            logger.info(
                "KnowledgeAccumulator: added %d dimension-matched feature edge(s) for %s",
                len(new_rels),
                sorted(dimensions),
            )

    # Node types that must NOT be stored in the graph database (Neo4j).
    # These types hold unstructured prose indexed only in the vector database.
    _GRAPH_DB_SKIP_TYPES = {"Task_Context", "Job_Instance_Context"}

    async def _store_graph(self, graph: KnowledgeGraph):
        """Persist graph nodes and relationships to Graph database.

        Nodes are merged by canonical name via
        :meth:`GraphDatabaseClient.get_or_create_canonical_node`.
        Nodes whose type is in :attr:`_GRAPH_DB_SKIP_TYPES` (e.g.
        ``Task_Context``, ``Job_Context``) are vector-DB-only and are silently
        skipped here.
        Relationship source/target IDs are remapped to the actual stored IDs
        before insertion.
        """
        # Auto-connect context nodes (concept="context") to every Symptom with
        # an ``associated_with`` edge so they survive the isolated-node filter.
        # These are background classification fields (prodSourceLabel, taskType,
        # etc.) that are never direct causes but are important for cross-task
        # pattern queries.
        from bamboo.models.graph_element import GraphRelationship, RelationType

        symptom_names = [
            n.name for n in graph.nodes if n.node_type.value == "Symptom"
        ]
        existing_rels = {
            (r.source_id, r.target_id, r.relation_type) for r in graph.relationships
        }
        for node in graph.nodes:
            if node.metadata.get("concept") != "context":
                continue
            for symptom_name in symptom_names:
                key = (node.name, symptom_name, RelationType.ASSOCIATED_WITH)
                if key not in existing_rels:
                    graph.relationships.append(
                        GraphRelationship(
                            source_id=node.name,
                            target_id=symptom_name,
                            relation_type=RelationType.ASSOCIATED_WITH,
                            confidence=1.0,
                        )
                    )
                    existing_rels.add(key)

        # Collect endpoint names from all relationships so isolated nodes can be
        # identified and dropped.  Relationship source/target IDs are still node
        # names at this point (UUID remapping happens after node creation below).
        endpoint_names: set[str] = set()
        for rel in graph.relationships:
            endpoint_names.add(rel.source_id)
            endpoint_names.add(rel.target_id)

        all_eligible = [
            n for n in graph.nodes
            if n.node_type.value not in self._GRAPH_DB_SKIP_TYPES
        ]
        graph_nodes = [
            n for n in all_eligible
            if n.name in endpoint_names or (n.id and n.id in endpoint_names)
        ]
        n_vector_only = len(graph.nodes) - len(all_eligible)
        n_isolated = len(all_eligible) - len(graph_nodes)
        logger.info(
            "KnowledgeAccumulator: storing %d/%d nodes in graph DB "
            "(%d vector-only skipped, %d isolated skipped)",
            len(graph_nodes),
            len(graph.nodes),
            n_vector_only,
            n_isolated,
        )

        node_ids: dict[str, str] = {}
        for node in graph_nodes:
            node_id = await self.graph_db.get_or_create_canonical_node(node, node.name)
            node_ids[node.id or node.name] = node_id

        graph_id = graph.metadata.get("graph_id", "")
        for rel in graph.relationships:
            rel.source_id = node_ids.get(rel.source_id, rel.source_id)
            rel.target_id = node_ids.get(rel.target_id, rel.target_id)
            # Skip relationships whose endpoints were not stored (vector-only nodes)
            if rel.source_id in node_ids.values() and rel.target_id in node_ids.values():
                # Stamp the graph_id so Neo4j can track per-edge provenance and
                # compute frequency across tasks via find_common_pattern.
                rel.properties["graph_id"] = graph_id
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
        - ``Job_Context``: full diagnostic text from representative failed jobs.
        - ``Symptom``: description holds the raw error message text, which is
          worth semantic search even though the node's canonical name is the
          clean error category.
        - ``Aggregated_Job_Feature``: description holds aggregated job pattern
          details useful for semantic similarity search.
        - ``Job_Instance``: description holds per-job diagnostic text for
          cross-incident similarity search.

        Cause and Resolution nodes are intentionally excluded: their canonical
        names are already precisely indexed in the graph database.
        """
        _INDEXABLE = {
            "Task_Context",
            "Job_Instance_Context",
            "Symptom",
            "Aggregated_Job_Feature",
            "Job_Instance",
        }
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
