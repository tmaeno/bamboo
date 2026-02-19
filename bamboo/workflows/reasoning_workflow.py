"""LangGraph workflow for task reasoning."""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from bamboo.agents.reasoning_navigator import ReasoningAgent
from bamboo.database.neo4j_client import Neo4jClient
from bamboo.database.qdrant_client import QdrantClient


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
        neo4j = Neo4jClient()
        qdrant = QdrantClient()

        await neo4j.connect()
        await qdrant.connect()

        agent = ReasoningAgent(neo4j, qdrant)

        result = await agent.analyze_task(
            task_data=state["task_data"],
            external_data=state["external_data"],
        )

        await neo4j.close()
        await qdrant.close()

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
