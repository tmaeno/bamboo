"""LangGraph workflow for knowledge extraction."""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient


class KnowledgeState(TypedDict):
    """State for knowledge extraction workflow."""

    email_text: str
    task_data: Optional[dict[str, Any]]
    external_data: Optional[dict[str, Any]]
    extracted_graph: Optional[dict[str, Any]]
    summary: Optional[str]
    status: str
    error: Optional[str]

    # Derived composite identifier: "<taskID>:<task_status>" (or just "<taskID>"
    # when status is absent).  Set after extraction so downstream nodes can
    # reference the graph without re-computing it.
    graph_id: Optional[str]


async def extract_knowledge_node(state: KnowledgeState) -> KnowledgeState:
    """Extract knowledge from sources."""
    try:
        graph_db = GraphDatabaseClient()
        vector_db = VectorDatabaseClient()

        await graph_db.connect()
        await vector_db.connect()

        agent = KnowledgeAccumulator(graph_db, vector_db)

        result = await agent.process_knowledge(
            email_text=state["email_text"],
            task_data=state["task_data"],
            external_data=state["external_data"],
        )

        await graph_db.close()
        await vector_db.close()

        # Build the composite identifier for downstream reference.
        task_data = state.get("task_data") or {}
        raw_task_id = task_data.get("taskID")
        task_status = task_data.get("status")
        if raw_task_id and task_status:
            graph_id = f"{raw_task_id}:{task_status}"
        elif raw_task_id:
            graph_id = raw_task_id
        else:
            graph_id = result.graph.metadata.get("graph_id")

        return {
            **state,
            "extracted_graph": result.graph.model_dump(),
            "summary": result.summary,
            "status": "completed",
            "graph_id": graph_id,
        }
    except Exception as e:
        return {
            **state,
            "status": "error",
            "error": str(e),
            "graph_id": None,
        }


async def validate_extraction_node(state: KnowledgeState) -> KnowledgeState:
    """Validate extracted knowledge."""
    if state["status"] == "error":
        return state

    # Basic validation
    if not state.get("extracted_graph"):
        return {
            **state,
            "status": "error",
            "error": "No knowledge extracted",
        }

    return {
        **state,
        "status": "validated",
    }


def create_knowledge_workflow() -> StateGraph:
    """Create knowledge extraction workflow."""
    workflow = StateGraph(KnowledgeState)

    # Add nodes
    workflow.add_node("extract", extract_knowledge_node)
    workflow.add_node("validate", validate_extraction_node)

    # Add edges
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", END)

    return workflow.compile()
