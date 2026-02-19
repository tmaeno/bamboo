"""Reasoning navigation for analyzing problematic tasks."""

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


class ReasoningAgent:
    """Agent for diagnosing system issues, analyzing failures, and generating resolutions."""

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        vector_db: VectorDatabaseClient,
    ):
        """Initialize diagnostic agent."""
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.extractor = KnowledgeGraphExtractor()

    def _extract_clues_from_graph(self, graph) -> dict[str, Any]:
        """Extract clues and key information from knowledge graph to deduce possible causes and resolutions."""
        errors = []
        task_features = []
        task_contexts = []  # unstructured prose — vector DB only, not graph DB
        environment_factors = []
        components = []

        for node in graph.nodes:
            node_type_str = str(node.node_type)

            if "ERROR" in node_type_str:
                errors.append(node.name)
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
            "errors": errors,
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
        """Analyze exhausted task and generate resolution."""
        logger.info(f"Analyzing task: {task_data.get('task_id', 'unknown')}")

        # Step 1: Extract knowledge graph (IDs assigned, names normalised by strategy)
        extracted_graph = await self.extractor.extract_from_sources(
            email_text="",
            task_data=task_data,
            external_data=external_data,
        )

        # Extract clues from the graph for database queries
        extracted_clues = self._extract_clues_from_graph(extracted_graph)

        # Step 2: Query graph database
        graph_results = await self._query_graph_database(extracted_clues)

        # Step 3: Query vector database
        vector_results = await self._query_vector_database(extracted_clues, task_data)

        # Step 4: Identify root cause using LLM
        analysis = await self._identify_root_cause(
            task_data, external_data, graph_results, vector_results
        )

        # Step 5: Generate email
        email_content = await self._generate_email(task_data, analysis)

        # Step 6: Create analysis result
        result = AnalysisResult(
            task_id=task_data.get("task_id", "unknown"),
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

        logger.info("Task analysis completed successfully")
        return result

    async def _query_graph_database(
        self, extracted_clues: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Query graph database for relevant causes and resolutions."""
        logger.info("Querying graph database")

        results = await self.graph_db.find_causes(
            errors=extracted_clues.get("errors"),
            task_features=extracted_clues.get("task_features"),
            environment_factors=extracted_clues.get("environment_factors"),
            components=extracted_clues.get("components"),
            limit=10,
        )

        logger.info(f"Found {len(results)} causes from graph database")
        return results

    async def _query_vector_database(
        self, extracted_clues: dict[str, Any], task_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find similar past cases using a two-step vector retrieval pattern.

        Step 1 — Search unstructured node descriptions:
            Query each clue type's section (Error, Task_Feature, Environment,
            Component) using the raw description text from the canonical graph.
            These are the only entries stored in the vector DB from those node
            types, because their names are already canonical and live in the
            graph DB.  Hits carry a ``graph_id`` in their metadata.

        Step 2 — Fetch summaries for matched graphs:
            Use the collected ``graph_id`` values to retrieve the corresponding
            Summary entries directly (no embedding needed — it is a metadata
            filter).  This gives the full narrative context of each matched
            past case.

        Returns a list of summary dicts ordered by the hit score of the
        description match that triggered them.
        """
        logger.info("Querying vector database (two-step: descriptions → summaries)")

        # --- Step 1: build per-section queries from canonical clues ---
        section_queries: list[tuple[str, str]] = []  # (section, query_text)

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
        if extracted_clues.get("task_contexts"):
            # Each prose context is its own query — don't concatenate them
            for ctx in extracted_clues["task_contexts"]:
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
                ("Component", "Component: " + ", ".join(extracted_clues["components"]))
            )
        if task_data.get("description"):
            section_queries.append(("Error", task_data["description"]))

        # --- Search each section, collect (graph_id, best_score) ---
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
                    # Keep the best score across all sections for this graph
                    if score > graph_id_scores.get(graph_id, 0.0):
                        graph_id_scores[graph_id] = score

        if not graph_id_scores:
            logger.info("No matching graphs found in vector database")
            return []

        # --- Step 2: fetch summaries for all matched graph_ids ---
        matched_graph_ids = list(graph_id_scores.keys())
        summaries = await self.vector_db.get_summaries_by_graph_ids(matched_graph_ids)

        # Attach the trigger score so the caller can rank results
        results = []
        for summary in summaries:
            graph_id = summary.get("metadata", {}).get("graph_id")
            results.append(
                {
                    **summary,
                    "score": graph_id_scores.get(graph_id, 0.0),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Found {len(results)} matching past cases via "
            f"{len(graph_id_scores)} unique graph IDs"
        )

        return results

    async def _identify_root_cause(
        self,
        task_data: dict[str, Any],
        external_data: dict[str, Any],
        graph_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Use LLM to identify root cause from database results."""
        logger.info("Identifying root cause with LLM")

        prompt = CAUSE_IDENTIFICATION_PROMPT.format(
            task_info=json.dumps(task_data, indent=2),
            external_info=json.dumps(external_data or {}, indent=2),
            graph_results=json.dumps(graph_results, indent=2),
            vector_results=json.dumps(
                [
                    {
                        "score": r["score"],
                        "entry": r["entry"],
                        "content": r["content"][:500],  # Truncate content
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
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "root_cause": "Unable to determine root cause",
                "confidence": 0.0,
                "resolution": "Manual investigation required",
                "reasoning": "Failed to analyze the issue",
            }

    async def _generate_email(
        self, task_data: dict[str, Any], analysis: dict[str, Any]
    ) -> str:
        """Generate email explaining the issue and resolution."""
        logger.info("Generating email content")

        prompt = EMAIL_GENERATION_PROMPT.format(
            task_id=task_data.get("task_id", "unknown"),
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
