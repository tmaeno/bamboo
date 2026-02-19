"""Reasoning navigation for analyzing problematic tasks."""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.agents.knowledge_graph_extractor import KnowledgeGraphExtractor
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

    def _extract_features_from_graph(self, graph) -> dict[str, Any]:
        """Extract features and key information from knowledge graph."""
        errors = []
        features = []
        environment_factors = []
        components = []

        # Extract different node types from the graph
        for node in graph.nodes:
            node_type_str = str(node.node_type)

            if "ERROR" in node_type_str:
                errors.append(node.name)
            elif "TASK_FEATURE" in node_type_str or "FEATURE" in node_type_str:
                features.append(node.name)
            elif "ENVIRONMENT" in node_type_str:
                environment_factors.append(node.name)
            elif "COMPONENT" in node_type_str:
                components.append(node.name)

        return {
            "errors": errors,
            "features": features,
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

        # Step 1: Extract knowledge graph from task data
        extracted_graph = await self.extractor.extract_from_sources(
            email_text="",
            task_data=task_data,
            external_data=external_data,
        )

        # Extract features from the graph for database queries
        extracted_features = self._extract_features_from_graph(extracted_graph)

        # Step 2: Query graph database
        graph_results = await self._query_graph_database(extracted_features)

        # Step 3: Query vector database
        vector_results = await self._query_vector_database(
            extracted_features, task_data
        )

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
                "extracted_features": extracted_features,
                "graph_results_count": len(graph_results),
                "vector_results_count": len(vector_results),
            },
        )

        logger.info("Task analysis completed successfully")
        return result

    async def _query_graph_database(
        self, extracted_features: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Query graph database for relevant causes and resolutions."""
        logger.info("Querying graph database")

        results = []

        # Query by errors
        errors = extracted_features.get("errors", [])
        for error in errors:
            error_results = await self.graph_db.find_causes_by_error(error, limit=5)
            results.extend(error_results)

        # Query by features
        features = extracted_features.get("features", [])
        if features:
            feature_results = await self.graph_db.find_causes_by_features(
                features, limit=5
            )
            results.extend(feature_results)

        # Remove duplicates and rank
        seen = set()
        unique_results = []
        for result in results:
            cause_id = result.get("cause_id")
            if cause_id not in seen:
                seen.add(cause_id)
                unique_results.append(result)

        # Sort by frequency and confidence
        unique_results.sort(
            key=lambda x: (x.get("frequency", 0), x.get("confidence", 0)),
            reverse=True,
        )

        logger.info(f"Found {len(unique_results)} unique causes from graph database")
        return unique_results[:10]  # Top 10

    async def _query_vector_database(
        self, extracted_features: dict[str, Any], task_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Query vector database for similar cases by section type.

        Queries each section type (Summary, Cause, Resolution) stored by KnowledgeAgent
        to get comprehensive and structured results.
        """
        logger.info("Querying vector database by sections")

        all_results = []

        # Initialize result lists
        summary_results = []
        cause_results = []
        resolution_results = []

        # Create base query from extracted features and task data
        query_parts = []
        if extracted_features.get("errors"):
            query_parts.append("Errors: " + ", ".join(extracted_features["errors"]))
        if extracted_features.get("features"):
            query_parts.append("Features: " + ", ".join(extracted_features["features"]))
        if task_data.get("description"):
            query_parts.append(f"Description: {task_data['description']}")

        base_query_text = "\n".join(query_parts)

        # Query Summary section - for general context and overview
        summary_query = f"Overview and summary of issue:\n{base_query_text}"
        summary_results = await self._query_section(
            query_text=summary_query,
            section="Summary",
            limit=3,
            score_threshold=0.7,
        )
        all_results.extend([{**r, "section": "Summary"} for r in summary_results])

        # Query Cause section - for root cause analysis
        if extracted_features.get("errors"):
            cause_query = f"Root causes for:\n{base_query_text}"
            cause_results = await self._query_section(
                query_text=cause_query,
                section="Cause",
                limit=5,
                score_threshold=0.7,
            )
            all_results.extend([{**r, "section": "Cause"} for r in cause_results])

        # Query Resolution section - for solutions
        resolution_query = f"Solutions and resolutions for:\n{base_query_text}"
        resolution_results = await self._query_section(
            query_text=resolution_query,
            section="Resolution",
            limit=5,
            score_threshold=0.7,
        )
        all_results.extend([{**r, "section": "Resolution"} for r in resolution_results])

        # Sort by score (highest first)
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(
            f"Found {len(all_results)} results from vector database "
            f"(Summary: {len(summary_results)}, Cause: {len(cause_results)}, "
            f"Resolution: {len(resolution_results)})"
        )
        return all_results

    async def _query_section(
        self,
        query_text: str,
        section: str,
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Query a specific section in the vector database.

        Args:
            query_text: Text to search for
            section: Section type (Summary, Cause, Resolution, etc.)
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of matching documents from the specified section
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.aembed_query(query_text)

        # Search vector database for this specific section
        # TODO: Implement search_by_section in VectorDatabaseClient
        # results = await self.vector_db.search_by_section(
        #     query_embedding=query_embedding,
        #     section=section,
        #     limit=limit,
        #     score_threshold=score_threshold,
        # )
        results = []  # Placeholder until search_by_section is implemented

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
