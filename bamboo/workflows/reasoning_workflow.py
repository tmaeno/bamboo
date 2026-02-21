"""LangGraph workflow for task reasoning."""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from bamboo.agents.reasoning_navigator import ReasoningNavigator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient


class ReasoningState(TypedDict):
    """State for reasoning workflow."""

    task_data: dict[str, Any]
    external_data: Optional[dict[str, Any]]
    extracted_features: Optional[dict[str, Any]]
    graph_results: Optional[list[dict[str, Any]]]
    vector_results: Optional[list[dict[str, Any]]]
    analysis: Optional[dict[str, Any]]
    email_content: Optional[str]
    status: str
    error: Optional[str]
    human_feedback: Optional[str]


async def analyze_task_node(state: ReasoningState) -> ReasoningState:
    """Analyze task and determine root cause."""
    try:
        graph_db = GraphDatabaseClient()
        vector_db = VectorDatabaseClient()

        await graph_db.connect()
        await vector_db.connect()

        agent = ReasoningNavigator(graph_db, vector_db)

        result = await agent.analyze_task(
            task_data=state["task_data"],
            external_data=state["external_data"],
        )

        await graph_db.close()
        await vector_db.close()

        return {
            **state,
            "analysis": {
                "root_cause": result.root_cause,
                "confidence": result.confidence,
                "resolution": result.resolution,
                "explanation": result.explanation,
            },
            "email_content": result.email_content,
            "status": "analyzed",
        }
    except Exception as e:
        return {
            **state,
            "status": "error",
            "error": str(e),
        }


async def human_review_node(state: ReasoningState) -> ReasoningState:
    """Wait for human review and feedback."""
    # In a real implementation, this would pause for human input
    # For now, we'll just mark it as ready for review
    return {
        **state,
        "status": "awaiting_review",
    }


def should_send_email(state: ReasoningState) -> str:
    """Determine if email should be sent or needs revision."""
    if state.get("human_feedback") == "approve":
        return "send"
    elif state.get("human_feedback") == "reject":
        return "revise"
    else:
        return "review"


def create_reasoning_workflow() -> StateGraph:
    """Create reasoning workflow with human-in-the-loop."""
    workflow = StateGraph(ReasoningState)

    # Add nodes
    workflow.add_node("analyze", analyze_task_node)
    workflow.add_node("review", human_review_node)

    # Add edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "review")

    # Conditional edge based on human feedback
    workflow.add_conditional_edges(
        "review",
        should_send_email,
        {
            "send": END,
            "revise": "analyze",  # Re-analyze with feedback
            "review": "review",  # Stay in review state
        },
    )

    return workflow.compile()
