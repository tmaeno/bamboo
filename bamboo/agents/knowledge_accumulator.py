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
from bamboo.agents.extractors.panda_knowledge_extractor import _node_concepts
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.llm import (
    PROCEDURE_DESC_MERGE_PROMPT,
    SUMMARIZATION_PROMPT,
    get_embeddings,
    get_llm,
    get_summary_llm,
)
from bamboo.models.knowledge_entity import ExtractedKnowledge, KnowledgeGraph
from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)


_MAX_REVIEW_RETRIES = 2


async def prefetch_panda_docs(
    task_data: dict[str, Any], email_text: str = ""
) -> dict[str, str]:
    """Fetch PanDA documentation hints before the first extraction pass.

    Derives search queries from key ``task_data`` fields (status,
    errorDialog, splitRule sub-rule keys) and an optional ``email_text``,
    then calls ``search_panda_docs`` via PandaMcpClient.

    For ``errorDialog`` and ``email_text`` an LLM call extracts 2-5 focused
    search terms.  Falls back to the raw 120-char errorDialog string if the
    LLM call fails.

    Returns a ``doc_hints`` dict: keys are query strings, values are plain
    rendered text (``"[Title] snippet\\n\\n[Title2] snippet2"``).
    Returns an empty dict on any error so the caller is unaffected.
    """
    import re

    error_dialog = task_data.get("errorDialog", "") or ""
    plain_error = ""
    if error_dialog:
        plain_error = re.sub(r"<[^>]+>", " ", error_dialog)
        plain_error = re.sub("#[^ ]+", " ", plain_error)
        plain_error = " ".join(plain_error.split())

    doc_query = ""
    if plain_error or email_text:
        try:
            from bamboo.llm import DOC_SEARCH_KEYWORDS_PROMPT, get_extraction_llm  # noqa: PLC0415
            from langchain_core.messages import HumanMessage  # noqa: PLC0415

            llm = get_extraction_llm()
            prompt = DOC_SEARCH_KEYWORDS_PROMPT.format(
                error_dialog=plain_error[:500].rsplit(None, 1)[0] if plain_error else "(none)",
                email_text=email_text[:500].rsplit(None, 1)[0] if email_text else "(none)",
            )
            with thinking("Extracting doc search keywords"):
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.startswith("```")
                ).strip()
            keywords: list[str] = json.loads(raw)
            clean = [k for k in keywords if k and isinstance(k, str)]
            if clean:
                doc_query = " ".join(clean)
                say(f"Doc search keywords: {clean}")
        except Exception as exc:
            logger.warning(
                "prefetch_panda_docs: keyword extraction failed (%s) — falling back to raw errorDialog",
                exc,
            )

        if not doc_query and plain_error:
            doc_query = plain_error[:120]

    status = task_data.get("status", "")
    parts: list[str] = []
    if status:
        parts.append(f"task status {status}")
    if doc_query:
        parts.append(doc_query)

    if not parts:
        return {}

    combined_query = " ".join(parts)

    from bamboo.mcp.panda_mcp_client import PandaMcpClient, _fetch_task_params_table  # noqa: PLC0415

    panda_client = PandaMcpClient()
    doc_hints: dict[str, str] = {}
    try:
        results = await panda_client.execute("search_panda_docs", query=combined_query)
        if isinstance(results, list) and results:
            rendered = "\n\n".join(
                f"[{e.get('title', '')}] {e.get('snippet', '')}".strip()
                for e in results if e.get("snippet")
            )
            if rendered:
                doc_hints[combined_query] = rendered
                say(
                    f"Pre-fetched {len(results)} PanDA doc section(s) "
                    f"for query: {combined_query!r}"
                )
                show_block(f"doc_hints: {combined_query}", rendered, max_lines=120)
    except Exception as exc:
        logger.warning(
            "prefetch_panda_docs: search_panda_docs failed for query=%r: %s",
            combined_query,
            exc,
        )

    split_rule_str = task_data.get("splitRule", "") or ""
    if split_rule_str:
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
                show_block("doc_hints: splitRule params", doc_hints["splitRule params"], max_lines=120)

    # Scan all fetched doc_hints text for gdpconfig UPPERCASE key names and
    # append their descriptions — same targeted-lookup approach as splitRule.
    if doc_hints:
        from bamboo.mcp.panda_mcp_client import _fetch_gdpconfig_table  # noqa: PLC0415

        gdp_table = await _fetch_gdpconfig_table()
        if gdp_table:
            all_text = " ".join(doc_hints.values())
            # Match UPPERCASE identifiers (must end on A-Z/0-9, not _)
            candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]*[A-Z0-9]\b", all_text))
            found_gdp: dict[str, str] = {}
            for word in candidates:
                if word in gdp_table:
                    found_gdp[word] = gdp_table[word]
                else:
                    # Strip trailing _<wildcard> e.g. SCOUT_X_<activity> → SCOUT_X
                    base = re.sub(r"_<[^>]+>$", "", word)
                    if base != word and base in gdp_table:
                        found_gdp[base] = gdp_table[base]
            if found_gdp:
                doc_hints["gdpconfig params"] = "\n".join(
                    f"- {k}: {v}" for k, v in sorted(found_gdp.items())
                )
                say(f"Looked up {len(found_gdp)} gdpconfig param(s) from gdpconfig.rst")
                show_block("doc_hints: gdpconfig params", doc_hints["gdpconfig params"], max_lines=120)

    return doc_hints


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
        dry_run: bool = False,
        require_procedures: bool = False,
        debug_trace: dict[str, Any] | None = None,
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
            dry_run:       When ``True``, extraction and summarisation are
                           performed normally but **no data is written** to
                           either the graph database or the vector database.
                           Useful for previewing what would be stored before
                           committing.
            require_procedures: When ``True``, the graph is not written to
                           any database if it contains no Procedure nodes.
                           :attr:`ExtractedKnowledge.stored` is set to
                           ``False`` in that case; the caller should exit
                           with a non-zero status code.
            debug_trace:   Optional dict populated with intermediate state
                           at each pipeline phase (for ``--debug-report``).

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

        if debug_trace is not None:
            debug_trace["graph_id"] = graph_id
            debug_trace["dry_run"] = dry_run

        from bamboo.agents.knowledge_reviewer import build_sources_summary, summarise_email_investigations

        # Local mutable copies — may be augmented by the explorer on first rejection.
        # Original caller dicts are never mutated.
        _task_logs = dict(task_logs or {})
        _external_data = dict(external_data or {})

        # Pre-fetch PanDA documentation context before the first extraction.
        # Stored separately from task_logs — docs are domain hints, not execution logs.
        _doc_hints: dict[str, str] = {}
        if self._explorer is not None:
            _doc_hints = await self._prefetch_panda_docs(task_data or {}, email_text=email_text or "")

        email_investigation = ""
        if email_text and email_text.strip():
            email_investigation = await summarise_email_investigations(email_text)

        sources_summary = build_sources_summary(
            email_text=email_text,
            task_logs=_task_logs,
            doc_hints=_doc_hints,
            email_investigation=email_investigation,
        )

        review_result = None
        # Accumulate failure_dimension across both extraction passes so that
        # dimensions identified in the first pass are not lost if the second
        # pass produces a different list.
        _all_failure_dimensions: set[str] = set()

        # Expose available tools to the reviewer so it can annotate issues with
        # "→ resolvable with <tool_name>".
        _available_tools = self._explorer.available_tools() if self._explorer else []

        async def _extract_and_review(pass_label: str):
            """Run one extraction + review cycle. Returns (graph, review_result)."""
            _graph = await self.extractor.extract_from_sources(
                email_text=email_text,
                task_data=task_data,
                external_data=_external_data,
                task_logs=_task_logs,
                doc_hints=_doc_hints,
            )
            _graph.metadata["graph_id"] = graph_id
            self._reconcile_cross_extractor_links(_graph)
            if self._reviewer is None:
                return _graph, None
            say(f"Running knowledge reviewer ({pass_label})...")
            _result = await self._reviewer.review(
                _graph, sources_summary,
                task_data=task_data,
                doc_hints=_doc_hints,
                available_tools=_available_tools,
            )
            _all_failure_dimensions.update(_result.failure_dimension)
            return _graph, _result

        # Pass 1: extract from current sources → review.
        graph, review_result = await _extract_and_review("pass 1")

        if review_result is not None and not review_result.approved and self._explorer:
            # On rejection: ask the explorer to fetch real additional data based
            # on the reviewer's issues.  Never re-extract from the same sources
            # with a feedback prompt — that risks the LLM fabricating information.
            logger.info(
                "KnowledgeAccumulator: pass 1 rejected (%d issue(s)) — "
                "invoking explorer for additional data",
                len(review_result.issues),
            )
            say(
                f"Pass 1: {len(review_result.issues)} gap(s) found — "
                "fetching additional data sources..."
            )
            exploration = await self._explorer.explore(
                task_data or {}, review_result.issues, doc_hints=_doc_hints,
            )
            if exploration.task_logs or exploration.external_data:
                _task_logs.update(exploration.task_logs)
                _external_data.update(exploration.external_data)
                # Rebuild sources_summary to include newly fetched data.
                sources_summary = build_sources_summary(
                    email_text=email_text,
                    task_logs=_task_logs,
                    doc_hints=_doc_hints,
                    email_investigation=email_investigation,
                )

            # Pass 2: re-extract from enriched sources → final review.
            graph, review_result = await _extract_and_review("pass 2")
            if review_result is not None and not review_result.approved:
                logger.warning(
                    "KnowledgeAccumulator: reviewer not satisfied after explorer pass — "
                    "storing best result (confidence=%.2f)",
                    review_result.confidence,
                )

        # Create explicit contribute_to edges for all Task_Feature nodes whose
        # concept tag matches a failure dimension declared by the reviewer.
        # Python selects by concept mechanically — the LLM only identifies the
        # dimension string, never individual node names.
        if _all_failure_dimensions:
            self._add_dimension_feature_edges(graph, _all_failure_dimensions)

        # Auto-connect context nodes to Symptom nodes so they appear connected
        # in the graph regardless of whether we are in dry-run mode or not.
        self._add_context_edges(graph)

        if debug_trace is not None:
            debug_trace["extracted_nodes"] = [
                {"type": n.node_type.value, "name": n.name, "description": n.description}
                for n in graph.nodes
            ]
            debug_trace["extracted_relationships"] = [
                {"src": r.source_id, "rel": r.relation_type.value, "tgt": r.target_id,
                 "confidence": r.confidence}
                for r in graph.relationships
            ]
            if review_result is not None:
                debug_trace["reviewer_outcome"] = {
                    "approved": review_result.approved,
                    "feedback": getattr(review_result, "feedback", None),
                    "issues": [str(i) for i in getattr(review_result, "issues", [])],
                    "confidence": getattr(review_result, "confidence", None),
                }

        _has_procedures = any(n.node_type.value == "Procedure" for n in graph.nodes)
        if require_procedures and not _has_procedures:
            say(
                "No Procedure nodes found — graph not stored (--require-procedures). "
                "Provide an email thread with investigation steps, or run interactively."
            )
            logger.warning(
                "KnowledgeAccumulator: require_procedures=True but no Procedure nodes — "
                "skipping storage for graph_id=%s",
                graph_id,
            )
            summary = await self._generate_summary(graph, doc_hints=_doc_hints, email_text=email_text)
            key_insights = await self._extract_key_insights(graph)
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
                    "explorer_ran": self._explorer is not None and self._reviewer is not None,
                    "explorer_logs_added": len(_task_logs) - len(task_logs or {}),
                    "explorer_external_added": len(_external_data) - len(external_data or {}),
                },
                stored=False,
            )

        if dry_run:
            logger.info(
                "KnowledgeAccumulator: dry-run mode — skipping all database writes"
            )
            if debug_trace is not None:
                debug_trace["previously_processed"] = None  # unknown in dry-run
        else:
            # Pre-check: if a Summary vector already exists for this graph_id the
            # task was previously processed.  Clean up its stale Neo4j contribution
            # before re-storing so old relationships don't accumulate.
            _previously_processed = False
            if self.vector_db is not None:
                existing = await self.vector_db.get_summaries_by_graph_ids([graph_id])
                _previously_processed = bool(existing)
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
            if debug_trace is not None:
                debug_trace["previously_processed"] = _previously_processed
            await self._store_graph(graph)

        summary = await self._generate_summary(graph, doc_hints=_doc_hints, email_text=email_text)
        key_insights = await self._extract_key_insights(graph)

        if debug_trace is not None:
            debug_trace["generated_summary"] = summary

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
                "explorer_ran": self._explorer is not None and self._reviewer is not None,
                "explorer_logs_added": len(_task_logs) - len(task_logs or {}),
                "explorer_external_added": len(_external_data) - len(external_data or {}),
            },
        )

    async def _prefetch_panda_docs(
        self, task_data: dict[str, Any], email_text: str = ""
    ) -> dict[str, str]:
        return await prefetch_panda_docs(task_data, email_text)

    def _reconcile_cross_extractor_links(self, graph: KnowledgeGraph) -> None:
        """Create schema-defined edges that span extractor boundaries.

        Each extractor (email, log, task-data) runs independently and never
        knows about nodes produced by the others.  After merging, this step
        adds the edges that were structurally impossible to emit during
        extraction:

            Symptom      -[indicate]->        Cause
            Task_Feature -[contribute_to]->   Cause
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

    def _add_context_edges(self, graph: KnowledgeGraph) -> None:
        """Connect concept=context nodes to every Symptom with ``associated_with``.

        Context nodes (prodSourceLabel, taskType, etc.) are never direct causes
        but carry signals for cross-task pattern queries.  Without an edge they
        are isolated and dropped by the graph-DB isolated-node filter.  This
        method is called before the dry-run check so the edges are visible in
        preview output too.
        """
        from bamboo.models.graph_element import GraphRelationship, RelationType

        symptom_names = [n.name for n in graph.nodes if n.node_type.value == "Symptom"]
        if not symptom_names:
            return
        existing = {(r.source_id, r.target_id, r.relation_type) for r in graph.relationships}
        for node in graph.nodes:
            if "context" not in _node_concepts(node):
                continue
            for symptom_name in symptom_names:
                key = (node.name, symptom_name, RelationType.ASSOCIATED_WITH)
                if key not in existing:
                    graph.relationships.append(
                        GraphRelationship(
                            source_id=node.name,
                            target_id=symptom_name,
                            relation_type=RelationType.ASSOCIATED_WITH,
                            confidence=1.0,
                        )
                    )
                    existing.add(key)

    def _add_dimension_feature_edges(
        self, graph: KnowledgeGraph, dimensions: set[str]
    ) -> None:
        """Create ``contribute_to`` edges for all feature nodes matching failure dimensions.

        The reviewer LLM declares which resource dimension(s) caused the failure
        (e.g. ``{"cpu"}``).  Python then selects every Task_Feature node whose
        ``metadata["concept"]`` is in that set and connects it to all
        Cause nodes.  This removes individual node selection from the LLM entirely,
        eliminating the over-inclusion problem caused by indirect domain reasoning.

        Args:
            graph:      The current :class:`KnowledgeGraph`.
            dimensions: Concept strings returned in ``ReviewResult.failure_dimension``.
        """
        from bamboo.models.graph_element import GraphRelationship, RelationType

        feature_nodes = [
            n for n in graph.nodes
            if n.node_type.value == "Task_Feature"
            and any(c in dimensions for c in _node_concepts(n))
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
                "\n".join(
                    f"• {n.node_type.value} '{n.name}' [{', '.join(_node_concepts(n))}]"
                    for n in feature_nodes
                ),
            )
            logger.info(
                "KnowledgeAccumulator: added %d dimension-matched feature edge(s) for %s",
                len(new_rels),
                sorted(dimensions),
            )

    # Node types that must NOT be stored in the graph database (Neo4j).
    # These types hold unstructured prose indexed only in the vector database.
    _GRAPH_DB_SKIP_TYPES = {"Task_Context"}

    async def _merge_procedure_description(
        self, name: str, existing: str, new: str
    ) -> str:
        """Ask the LLM to merge two Procedure descriptions into one."""
        prompt = PROCEDURE_DESC_MERGE_PROMPT.format(
            name=name, desc_a=existing, desc_b=new
        )
        llm = get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        merged = response.content.strip()
        logger.info(
            "KnowledgeAccumulator: merged description for procedure '%s'", name
        )
        return merged

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

        # For Procedure nodes: LLM-merge descriptions when a same-named node exists.
        for node in graph_nodes:
            if node.node_type.value != "Procedure" or not node.description:
                continue
            existing_desc = await self.graph_db.get_node_description(
                "Procedure", node.name
            )
            if existing_desc is not None and existing_desc.strip() != node.description.strip():
                merged = await self._merge_procedure_description(
                    node.name, existing_desc, node.description
                )
                await self.graph_db.update_node_description(
                    "Procedure", node.name, merged
                )
                node.description = merged

        node_ids: dict[str, str] = {}
        for node in graph_nodes:
            node_id = await self.graph_db.get_or_create_canonical_node(node, node.name)
            node_ids[node.name] = node_id

        # Build a name→parameters lookup for procedure edge wiring below.
        proc_params_by_name: dict[str, Any] = {
            n.name: n.metadata.get("parameters")
            for n in graph.nodes
            if n.node_type.value == "Procedure" and n.metadata.get("parameters")
        }

        graph_id = graph.metadata.get("graph_id", "")
        for rel in graph.relationships:
            rel.source_id = node_ids.get(rel.source_id, rel.source_id)
            rel.target_id = node_ids.get(rel.target_id, rel.target_id)
            # Skip relationships whose endpoints were not stored (vector-only nodes)
            if rel.source_id in node_ids.values() and rel.target_id in node_ids.values():
                # Stamp the graph_id so Neo4j can track per-edge provenance and
                # compute frequency across tasks via find_common_pattern.
                rel.properties["graph_id"] = graph_id
                # For investigated_by edges, carry the procedure's parameters on
                # the edge so they accumulate as a list per incident in Neo4j.
                if rel.relation_type.value == "investigated_by":
                    params = rel.properties.pop("parameters", None)
                    if params is None:
                        for proc_name, proc_params in proc_params_by_name.items():
                            if node_ids.get(proc_name) == rel.target_id:
                                params = proc_params
                                break
                    if params is not None:
                        rel.properties["parameters"] = params
                await self.graph_db.create_relationship(rel)

        logger.info("KnowledgeAccumulator: graph stored successfully")

    async def _generate_summary(
        self,
        graph: KnowledgeGraph,
        doc_hints: dict[str, str] | None = None,
        email_text: str = "",
    ) -> str:
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

        hints_text = (
            "\n".join(f"{k}:\n{v}" for k, v in (doc_hints or {}).items())
            or "(none)"
        )
        prompt = SUMMARIZATION_PROMPT.format(
            graph_data=json.dumps(graph_data, indent=2),
            doc_hints=hints_text,
            email_text=email_text or "(none)",
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
        _INDEXABLE = {
            "Task_Context",
            "Symptom",
        }
        insights = []
        for node in graph.nodes:
            if node.description and node.node_type.value in _INDEXABLE:
                # For Symptom nodes the raw error message is preserved in
                # metadata["raw_description"] (before description canonicalization).
                # Use it for richer semantic search; fall back to description.
                content = node.metadata.get("raw_description") or node.description
                insights.append(
                    {
                        "node_id": node.id,
                        "section": node.node_type.value,
                        "content": content,
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
