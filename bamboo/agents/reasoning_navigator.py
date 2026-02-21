"""Reasoning agent: diagnoses incidents and generates resolution emails.

:class:`ReasoningNavigator` is the second half of the Bamboo pipeline.  Given a
new problematic task it:

1. Extracts a :class:`~bamboo.models.knowledge_entity.KnowledgeGraph` from the
   task's structured fields using the configured
   :class:`~bamboo.extractors.base.ExtractionStrategy`.
2. Queries the **graph database** for candidate causes ranked by how many
   extracted clue types (symptoms, features, environment, components) point to
   them.
3. Queries the **vector database** using a two-step retrieval pattern:
   a. Searches unstructured node descriptions to find similar past cases
      (returns ``graph_id`` values).
   b. Fetches the narrative summaries for those graphs.
4. Feeds all evidence to an LLM to identify the most likely root cause and
   recommend a resolution.
5. Drafts a professional email for the task submitter.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.extractors.knowledge_graph_extractor import KnowledgeGraphExtractor
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.llm import (
    CAUSE_IDENTIFICATION_PROMPT,
    EMAIL_GENERATION_PROMPT,
    get_embeddings,
    get_llm,
)
from bamboo.models.knowledge_entity import AnalysisResult

logger = logging.getLogger(__name__)


class ReasoningNavigator:
    """Diagnoses a problematic task and drafts a resolution email.

    The agent is stateless between calls: every :meth:`analyze_task`
    invocation is independent.

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
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.extractor = KnowledgeGraphExtractor()

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
            elif "TASK_CONTEXT" in node_type_str:
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
    ) -> AnalysisResult:
        """Analyse a problematic task and return a root-cause + resolution result.

        Pipeline:
            1. Extract knowledge graph from structured task fields.
            2. Query graph DB for candidate causes.
            3. Query vector DB for similar past cases (two-step retrieval).
            4. Ask LLM to identify the root cause.
            5. Ask LLM to draft a resolution email.

        Args:
            task_data:     Structured task fields (must include ``taskID``).
            external_data: Optional supplementary metadata.

        Returns:
            :class:`~bamboo.models.knowledge_entity.AnalysisResult` with the
            root cause, resolution, explanation, email draft, and evidence.
        """
        task_id = task_data.get("taskID") or "unknown"
        logger.info("ReasoningNavigator: analysing task '%s'", task_id)

        extracted_graph = await self.extractor.extract_from_sources(
            email_text="",
            task_data=task_data,
            external_data=external_data,
        )
        extracted_clues = self._extract_clues_from_graph(extracted_graph)

        graph_results = await self._query_graph_database(extracted_clues)
        vector_results = await self._query_vector_database(extracted_clues, task_data)
        analysis = await self._identify_root_cause(
            task_data, external_data, graph_results, vector_results
        )
        email_content = await self._generate_email(task_data, analysis)

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
        results = await self.graph_db.find_causes(
            symptoms=extracted_clues.get("symptoms"),
            task_features=extracted_clues.get("task_features"),
            environment_factors=extracted_clues.get("environment_factors"),
            components=extracted_clues.get("components"),
            limit=10,
        )
        logger.info("ReasoningNavigator: graph DB returned %d causes", len(results))
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
            task_info=json.dumps(task_data, indent=2),
            external_info=json.dumps(external_data or {}, indent=2),
            graph_results=json.dumps(graph_results, indent=2),
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
            ),
        )

        messages = [
            SystemMessage(
                content="You are an expert at root cause analysis and problem solving."
            ),
            HumanMessage(content=prompt),
        ]
        response = await self.llm.ainvoke(messages)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as exc:
            logger.error("ReasoningNavigator: failed to parse LLM response: %s", exc)
            return {
                "root_cause": "Unable to determine root cause",
                "confidence": 0.0,
                "resolution": "Manual investigation required",
                "reasoning": "LLM response could not be parsed.",
            }

    async def _generate_email(
        self, task_data: dict[str, Any], analysis: dict[str, Any]
    ) -> str:
        """Draft a professional resolution email for the task submitter.

        Args:
            task_data: Raw task fields (``taskID`` and ``description`` used).
            analysis:  Root-cause analysis dict from :meth:`_identify_root_cause`.

        Returns:
            Email body as a plain-text string.
        """
        logger.info("ReasoningNavigator: generating resolution email")

        prompt = EMAIL_GENERATION_PROMPT.format(
            task_id=task_data.get("taskID") or "unknown",
            task_description=task_data.get("description", ""),
            analysis=json.dumps(analysis, indent=2),
        )
        messages = [
            SystemMessage(
                content="You are an expert at technical communication and customer support."
            ),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content
