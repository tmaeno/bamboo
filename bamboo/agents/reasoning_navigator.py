"""Reasoning agent: diagnoses incidents and generates resolution emails.

:class:`ReasoningNavigator` is the second half of the Bamboo pipeline.  Given a
new problematic task it:

1. Extracts a :class:`~bamboo.models.knowledge_entity.KnowledgeGraph` from the
   task's structured fields using the configured
   :class:`~bamboo.agents.extractors.base.ExtractionStrategy`.
2. Queries the **graph database** for candidate causes ranked by how many
   extracted clue types (symptoms, task features, environment, components) point
   to them.
3. Queries the **vector database** using a two-step retrieval pattern:
   a. Searches unstructured node descriptions to find similar past cases
      (returns ``graph_id`` values).
   b. Fetches the narrative summaries for those graphs.
4. Feeds all evidence to an LLM to identify the most likely root cause and
   recommend a resolution  (Phase 1).
5. Queries the **graph database** for :class:`~bamboo.models.graph_element.ProcedureNode`
   entries linked to the identified causes  (Phase 2).  If found, the LLM selects
   the appropriate MCP tool from the available tool catalogue and runs the
   investigation.  If no procedure is found, the navigator notes that human input
   is recommended.
6. Drafts a professional email for the task submitter.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.agents.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.agents.knowledge_reviewer import _join_doc_hints
from bamboo.llm import (
    CAUSE_IDENTIFICATION_PROMPT,
    EMAIL_GENERATION_PROMPT,
    get_embeddings,
    get_llm,
)
from bamboo.models.knowledge_entity import AnalysisResult
from bamboo.utils.sanitize import sanitize_for_llm
from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)


class ReasoningNavigator:
    """Diagnoses a problematic task and drafts a resolution email.

    The agent is stateless between calls: every :meth:`analyze_task`
    invocation is independent.

    Args:
        graph_db:  Connected :class:`GraphDatabaseClient`.
        vector_db: Connected :class:`VectorDatabaseClient`.
        explorer:  Optional :class:`~bamboo.agents.extra_source_explorer.ExtraSourceExplorer`.
                   When provided, Phase 2 (procedure-driven investigation) delegates
                   MCP tool selection and execution to the explorer rather than
                   remaining a no-op stub.
    """

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        vector_db: VectorDatabaseClient,
        explorer=None,
    ):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self._explorer = explorer
        self._llm = None
        self._embeddings = None
        self.extractor = KnowledgeGraphExtractor()

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_clues_from_graph(self, graph) -> dict[str, Any]:
        """Partition graph nodes into typed clue lists for database queries.

        ``TaskContextNode`` values are kept separately (as ``task_contexts``)
        because they are only available via vector search — they are not
        stored in the graph database.

        Args:
            graph: A :class:`~bamboo.models.knowledge_entity.KnowledgeGraph`.

        Returns:
            Dict with keys ``symptoms``, ``task_features``, ``task_contexts``,
            ``environment_factors``, ``components``, ``context``.
        """
        symptoms = []
        task_features = []
        task_contexts = []
        environment_factors = []
        components = []

        for node in graph.nodes:
            node_type_str = str(node.node_type)
            if "SYMPTOM" in node_type_str:
                symptoms.append(node.name)
            elif "TASK_CONTEXT" in node_type_str or "JOB_CONTEXT" in node_type_str:
                if node.description:
                    task_contexts.append(node.description)
            elif "TASK_FEATURE" in node_type_str or "FEATURE" in node_type_str:
                task_features.append(node.name)
            elif "ENVIRONMENT" in node_type_str:
                environment_factors.append(node.name)
            elif "COMPONENT" in node_type_str:
                components.append(node.name)

        return {
            "symptoms": symptoms,
            "task_features": task_features,
            "task_contexts": task_contexts,
            "environment_factors": environment_factors,
            "components": components,
            "context": {},
        }

    async def analyze_task(
        self,
        task_data: dict[str, Any],
        external_data: dict[str, Any] = None,
        task_logs: dict[str, str] = None,
        doc_hints: dict[str, str] | None = None,
    ) -> AnalysisResult:
        """Analyse a problematic task and return a root-cause + resolution result.

        Pipeline:
            1. Extract knowledge graph from structured task fields, logs, job
               logs, and aggregated job data.
            2. Query graph DB for candidate causes (task features + job features).
            3. Query vector DB for similar past cases (two-step retrieval).
            4. Ask LLM to identify the root cause.
            5. Ask LLM to draft a resolution email.

        Args:
            task_data:     Structured task fields (must include ``taskID``; ``status``
                           is used together with ``taskID`` as the composite
                           unique identifier for the incident).
            external_data: Optional supplementary metadata.
            task_logs:     *Task-level* log output keyed by source name
                           (e.g. ``{"jedi": "...", "harvester": "..."}``).
                           Extracted nodes are tagged ``log_level="task"``.
        Returns:
            :class:`~bamboo.models.knowledge_entity.AnalysisResult` with the
            root cause, resolution, explanation, email draft, and evidence.
        """
        raw_task_id = task_data.get("jediTaskID") or "unknown"
        task_status = task_data.get("status")
        task_id = f"{raw_task_id}:{task_status}" if task_status else raw_task_id
        logger.info("ReasoningNavigator: analysing task '%s'", task_id)

        domain_hints = _join_doc_hints(doc_hints)

        extracted_graph = await self.extractor.extract_from_sources(
            email_text="",
            task_data=task_data,
            external_data=external_data,
            task_logs=task_logs,
        )
        extracted_clues = self._extract_clues_from_graph(extracted_graph)

        graph_results = await self._query_graph_database(extracted_clues)
        if not graph_results and extracted_clues.get("symptoms"):
            show_block(
                "root cause analysis",
                "cause:      unknown\n"
                "confidence: 0.0\n"
                "resolution: Manual investigation required\n"
                "reasoning:  No matching causes found in the knowledge base for the "
                f"observed symptoms: {extracted_clues['symptoms']}",
            )
            return AnalysisResult(
                task_id=task_id,
                root_cause="unknown",
                confidence=0.0,
                resolution="Manual investigation required",
                explanation=(
                    "No matching causes found in the knowledge base for the "
                    f"observed symptoms: {extracted_clues['symptoms']}"
                ),
                email_content="",
                metadata={
                    "extracted_clues": extracted_clues,
                    "graph_results_count": 0,
                    "vector_results_count": 0,
                    "procedures_found": 0,
                    "investigation": {},
                },
            )
        vector_results = await self._query_vector_database(extracted_clues, task_data)
        analysis = await self._identify_root_cause(
            task_data, external_data, graph_results, vector_results, domain_hints=domain_hints
        )
        show_block(
            "root cause analysis",
            f"cause:      {analysis.get('root_cause')}\n"
            f"confidence: {analysis.get('confidence')}\n"
            f"resolution: {analysis.get('resolution')}\n"
            f"reasoning:  {analysis.get('reasoning')}",
        )

        # --- Phase 2: procedure-driven investigation ---
        identified_causes = [analysis["root_cause"]] if analysis.get("root_cause") else []
        procedures = await self._query_procedures(identified_causes)

        investigation_result: dict[str, Any] = {}
        if procedures:
            investigation_result = await self._run_investigation(
                task_data, procedures, domain_hints
            )
        else:
            investigation_result["investigation_note"] = (
                "No investigation procedure found for the identified cause. "
                "Manual investigation is recommended."
            )
            logger.info(
                "ReasoningNavigator: no procedures found for cause '%s' — "
                "manual investigation recommended",
                analysis.get("root_cause", "unknown"),
            )

        # --- Merge investigation results and run second analysis pass ---
        inv_external = investigation_result.get("external_data") or {}
        inv_logs = investigation_result.get("task_logs") or {}
        if inv_external or inv_logs:
            merged_external = {**(external_data or {}), **inv_external}
            logger.info("ReasoningNavigator: re-running root cause analysis with investigation data")
            analysis = await self._identify_root_cause(
                task_data, merged_external, graph_results, vector_results, domain_hints=domain_hints
            )
            show_block(
                "root cause analysis (updated)",
                f"cause:      {analysis.get('root_cause')}\n"
                f"confidence: {analysis.get('confidence')}\n"
                f"resolution: {analysis.get('resolution')}\n"
                f"reasoning:  {analysis.get('reasoning')}",
            )
            # Attach raw investigation data so the email LLM sees job IDs and metrics directly
            analysis["investigation_data"] = inv_external

        email_content = await self._generate_email(task_data, analysis, domain_hints=domain_hints)

        return AnalysisResult(
            task_id=task_id,
            root_cause=analysis["root_cause"],
            confidence=analysis["confidence"],
            resolution=analysis["resolution"],
            explanation=analysis["reasoning"],
            supporting_evidence=analysis.get("supporting_evidence", []),
            email_content=email_content,
            metadata={
                "extracted_clues": extracted_clues,
                "graph_results_count": len(graph_results),
                "vector_results_count": len(vector_results),
                "procedures_found": len(procedures),
                "investigation": investigation_result,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _query_graph_database(
        self, extracted_clues: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Query the graph DB for causes that match the extracted clues.

        Args:
            extracted_clues: Output of :meth:`_extract_clues_from_graph`.

        Returns:
            List of cause dicts from :meth:`GraphDatabaseClient.find_causes`.
        """
        logger.info("ReasoningNavigator: querying graph database")
        symptoms = extracted_clues.get("symptoms") or []
        for symptom in symptoms:
            single = await self.graph_db.find_causes(symptoms=[symptom], limit=1)
            if not single:
                logger.info(
                    "ReasoningNavigator: symptom '%s' has no known causes in graph "
                    "— returning no results to avoid spurious match",
                    symptom,
                )
                return []
        results = await self.graph_db.find_causes(
            symptoms=symptoms or None,
            task_features=extracted_clues.get("task_features"),
            environment_factors=extracted_clues.get("environment_factors"),
            components=extracted_clues.get("components"),
            limit=10,
        )
        logger.info("ReasoningNavigator: graph DB returned %d causes", len(results))
        return results

    async def _query_procedures(
        self, cause_names: list[str]
    ) -> list[dict[str, Any]]:
        """Query the graph DB for investigation procedures linked to identified causes.

        Args:
            cause_names: Canonical cause names from Phase 1 analysis.

        Returns:
            List of procedure dicts from
            :meth:`GraphDatabaseClient.find_procedures_for_causes`, ordered by
            frequency descending.
        """
        if not cause_names:
            return []
        logger.info(
            "ReasoningNavigator: querying procedures for %d cause(s)", len(cause_names)
        )
        results = await self.graph_db.find_procedures_for_causes(cause_names)
        logger.info(
            "ReasoningNavigator: found %d procedure(s) for identified causes", len(results)
        )
        return results

    async def _query_vector_database(
        self, extracted_clues: dict[str, Any], task_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find similar past cases using two-step vector retrieval.

        **Step 1** — search each unstructured clue type's section in the vector
        DB to find ``graph_id`` values of similar past cases.

        **Step 2** — fetch the ``Summary`` entries for those graph IDs to
        obtain the full narrative context.

        The result list is sorted by the best hit score across all sections.

        Args:
            extracted_clues: Output of :meth:`_extract_clues_from_graph`.
            task_data:       Raw task data (``description`` field used as an
                             additional query if present).

        Returns:
            List of summary dicts, each augmented with a ``score`` field
            reflecting the strength of the best section match.
        """
        logger.info("ReasoningNavigator: querying vector database (two-step)")

        section_queries: list[tuple[str, str]] = []

        if extracted_clues.get("errors"):
            section_queries.append(
                ("Error", "Error: " + ", ".join(extracted_clues["errors"]))
            )
        if extracted_clues.get("task_features"):
            section_queries.append(
                (
                    "Task_Feature",
                    "Task features: " + ", ".join(extracted_clues["task_features"]),
                )
            )
        for ctx in extracted_clues.get("task_contexts", []):
            section_queries.append(("Task_Context", ctx))
        if extracted_clues.get("environment_factors"):
            section_queries.append(
                (
                    "Environment",
                    "Environment: " + ", ".join(extracted_clues["environment_factors"]),
                )
            )
        if extracted_clues.get("components"):
            section_queries.append(
                (
                    "Component",
                    "Component: " + ", ".join(extracted_clues["components"]),
                )
            )
        if task_data.get("description"):
            section_queries.append(("Error", task_data["description"]))

        # Step 1: collect best score per graph_id across all sections
        graph_id_scores: dict[str, float] = {}
        for section, query_text in section_queries:
            query_embedding = await self.embeddings.aembed_query(query_text)
            hits = await self.vector_db.search_similar(
                query_embedding=query_embedding,
                limit=5,
                score_threshold=0.7,
                filter_conditions={"section": section},
            )
            for hit in hits:
                graph_id = hit.get("metadata", {}).get("graph_id")
                if graph_id:
                    score = hit.get("score", 0.0)
                    if score > graph_id_scores.get(graph_id, 0.0):
                        graph_id_scores[graph_id] = score

        if not graph_id_scores:
            logger.info("ReasoningNavigator: no matching graphs found in vector DB")
            return []

        # Step 2: fetch summaries for all matched graph_ids
        summaries = await self.vector_db.get_summaries_by_graph_ids(
            list(graph_id_scores.keys())
        )
        results = [
            {
                **summary,
                "score": graph_id_scores.get(
                    summary.get("metadata", {}).get("graph_id"), 0.0
                ),
            }
            for summary in summaries
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "ReasoningNavigator: vector DB returned %d past cases via %d graph IDs",
            len(results),
            len(graph_id_scores),
        )
        return results

    async def _identify_root_cause(
        self,
        task_data: dict[str, Any],
        external_data: dict[str, Any],
        graph_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        domain_hints: str = "(none)",
    ) -> dict[str, Any]:
        """Call the LLM to synthesise a root-cause analysis.

        Args:
            task_data:      Raw task fields.
            external_data:  Supplementary metadata (may be ``None``).
            graph_results:  Candidate causes from the graph DB.
            vector_results: Similar past cases from the vector DB.

        Returns:
            Dict with keys ``root_cause``, ``confidence``, ``resolution``,
            ``reasoning``, and ``supporting_evidence``.
        """
        logger.info("ReasoningNavigator: identifying root cause with LLM")

        prompt = CAUSE_IDENTIFICATION_PROMPT.format(
            task_info=json.dumps(sanitize_for_llm(task_data), indent=2, default=str),
            external_info=json.dumps(sanitize_for_llm(external_data) or {}, indent=2, default=str),
            domain_hints=domain_hints,
            graph_results=json.dumps(graph_results, indent=2, default=str),
            vector_results=json.dumps(
                [
                    {
                        "score": r["score"],
                        "entry": r["entry"],
                        "content": r["content"][:500],
                    }
                    for r in vector_results
                ],
                indent=2,
                default=str,
            ),
        )

        messages = [
            SystemMessage(
                content="You are an expert at root cause analysis and problem solving."
            ),
            HumanMessage(content=prompt),
        ]
        with thinking("Working"):
            response = await self.llm.ainvoke(messages)

        text = response.content.strip()
        if "```json" in text:
            text = text[text.find("```json") + 7 : text.rfind("```")].strip()
        elif "```" in text:
            text = text[text.find("```") + 3 : text.rfind("```")].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("ReasoningNavigator: failed to parse LLM response: %s", exc)
            return {
                "root_cause": "Unable to determine root cause",
                "confidence": 0.0,
                "resolution": "Manual investigation required",
                "reasoning": "LLM response could not be parsed.",
            }

    async def _generate_email(
        self, task_data: dict[str, Any], analysis: dict[str, Any], domain_hints: str = "(none)"
    ) -> str:
        """Draft a professional resolution email for the task submitter.

        Args:
            task_data: Raw task fields (``taskID`` and ``description`` used).
            analysis:  Root-cause analysis dict from :meth:`_identify_root_cause`.

        Returns:
            Email body as a plain-text string.
        """
        logger.info("ReasoningNavigator: generating resolution email")

        raw_task_id = task_data.get("jediTaskID") or "unknown"
        task_status = task_data.get("status")
        composite_task_id = (
            f"{raw_task_id}:{task_status}" if task_status else raw_task_id
        )
        prompt = EMAIL_GENERATION_PROMPT.format(
            task_id=composite_task_id,
            task_description=task_data.get("description", ""),
            domain_hints=domain_hints,
            analysis=json.dumps(analysis, indent=2, default=str),
        )
        messages = [
            SystemMessage(
                content="You are an expert at technical communication and customer support."
            ),
            HumanMessage(content=prompt),
        ]

        with thinking("Working"):
            response = await self.llm.ainvoke(messages)
        return response.content

    async def _run_investigation(
        self,
        task_data: dict[str, Any],
        procedures: list[dict[str, Any]],
        domain_hints: str = "(none)",
    ) -> dict[str, Any]:
        """Phase 2: run a procedure-driven investigation for the current task.

        The LLM is given the list of procedures (each with a ``strategy_type``
        and accumulated ``parameters``) and the available MCP tools.  It selects
        which tool to call and with what arguments, then synthesises the findings
        into a structured result.

        This method is a scaffold — full MCP tool invocation will be wired in
        when the navigator gains access to an MCP client.  Currently it returns
        the procedure list and strategy descriptions so callers can see what
        investigation was recommended.

        Args:
            task_data:    Raw task fields for the current task.
            procedures:   Procedure dicts from :meth:`_query_procedures`.
            domain_hints: Domain documentation string.

        Returns:
            Dict with ``procedures`` (the raw procedure list) and
            ``investigation_note`` describing the recommended next steps.
        """
        logger.info(
            "ReasoningNavigator: Phase 2 — %d procedure(s) available",
            len(procedures),
        )

        # Stub fallback when no explorer is wired.
        if self._explorer is None:
            strategies = [
                f"[cause: {p['cause_name']}] {p['strategy_type']} "
                f"(confirmed by {p['frequency']} incident(s))"
                for p in procedures
            ]
            note = (
                "Investigation procedure(s) found but no MCP explorer configured. "
                "Recommended strategies:\n"
                + "\n".join(f"  • {s}" for s in strategies)
            )
            logger.info("ReasoningNavigator: investigation note: %s", note)
            return {"procedures": procedures, "investigation_note": note}

        # Delegate to ExtraSourceExplorer — reuses its LLM tool selection,
        # asyncio.gather execution, and _merge_tool_result routing.
        issues = []
        for p in procedures:
            params_str = json.dumps(p.get("parameters") or [], default=str)
            issues.append(
                f"Investigate '{p['cause_name']}': {p['strategy_type']}. "
                f"Historical parameters: {params_str}"
            )

        say(f"Phase 2: running investigation via MCP ({len(issues)} procedure(s))...")
        exploration = await self._explorer.explore(task_data, issues, skip_gap_analysis=True)

        note = (
            f"Investigation ran {len(exploration.tool_calls)} tool call(s)."
            if exploration.tool_calls
            else "Explorer selected no tools for the investigation procedure."
        )
        logger.info("ReasoningNavigator: %s", note)

        return {
            "procedures": procedures,
            "tool_calls": exploration.tool_calls,
            "external_data": exploration.external_data,
            "task_logs": {k: v[:2000] for k, v in exploration.task_logs.items()},
            "investigation_note": note,
        }
