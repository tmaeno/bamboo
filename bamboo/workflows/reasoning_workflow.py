"""LangGraph workflow for task reasoning."""

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from bamboo.agents.reasoning_navigator import ReasoningNavigator
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient


class ReasoningState(TypedDict):
    """State for the reasoning workflow.

    Attributes:
        task_data:          Structured task fields.
        external_data:      Optional supplementary metadata.
        task_logs:          *Task-level* log output keyed by source name
                            (e.g. ``{"jedi": "...", "harvester": "..."}``).
        job_logs:           *Job-level* log output keyed by a stable source name
                            (e.g. ``{"pilot": "...", "payload": "..."}``).
        jobs_data:          List of raw job attribute dicts for aggregated
                            :class:`~bamboo.models.graph_element.AggregatedJobFeatureNode`
                            extraction.
        extracted_features: Clue dict produced by
                            :meth:`~bamboo.agents.reasoning_navigator.ReasoningNavigator._extract_clues_from_graph`.
        graph_results:      Candidate causes from the graph DB.
        vector_results:     Similar past cases from the vector DB.
        analysis:           Root-cause analysis dict.
        email_content:      Draft resolution email.
        status:             Workflow status string.
        error:              Error message if ``status == "error"``.
        human_feedback:     ``"approve"`` or ``"reject"`` from human review step.
    """

    task_data: dict[str, Any]
    external_data: Optional[dict[str, Any]]
    task_logs: Optional[dict[str, str]]
    job_logs: Optional[dict[str, str]]
    jobs_data: Optional[list[dict[str, Any]]]
    extracted_features: Optional[dict[str, Any]]
    graph_results: Optional[list[dict[str, Any]]]
    vector_results: Optional[list[dict[str, Any]]]
    analysis: Optional[dict[str, Any]]
    email_content: Optional[str]
    status: str
    error: Optional[str]
    human_feedback: Optional[str]


async def analyze_task_node(state: ReasoningState) -> ReasoningState:
    """Analyse a problematic task and determine its root cause."""
    try:
        graph_db = GraphDatabaseClient()
        vector_db = VectorDatabaseClient()

        await graph_db.connect()
        await vector_db.connect()

        agent = ReasoningNavigator(graph_db, vector_db)

        result = await agent.analyze_task(
            task_data=state["task_data"],
            external_data=state["external_data"],
            task_logs=state.get("task_logs"),
            job_logs=state.get("job_logs"),
            jobs_data=state.get("jobs_data"),
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
    """Wait for human review and feedback.

    In a real deployment this node would pause the workflow and surface the
    draft email to an operator.  The operator sets ``human_feedback`` to
    ``"approve"`` or ``"reject"`` before resuming.
    """
    return {
        **state,
        "status": "awaiting_review",
    }


def should_send_email(state: ReasoningState) -> str:
    """Routing function: decide next step based on human feedback."""
    if state.get("human_feedback") == "approve":
        return "send"
    elif state.get("human_feedback") == "reject":
        return "revise"
    else:
        return "review"


def create_reasoning_workflow() -> StateGraph:
    """Create and compile the reasoning LangGraph workflow with human-in-the-loop."""
    workflow = StateGraph(ReasoningState)

    workflow.add_node("analyze", analyze_task_node)
    workflow.add_node("review", human_review_node)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "review")

    workflow.add_conditional_edges(
        "review",
        should_send_email,
        {
            "send": END,
            "revise": "analyze",
            "review": "review",
        },
    )

    return workflow.compile()
