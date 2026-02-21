"""Knowledge extraction result and query models.

These Pydantic models represent the data transfer objects that flow between
the extraction layer (:mod:`bamboo.extractors`), the storage layer
(:mod:`bamboo.database`), and the reasoning layer (:mod:`bamboo.agents`).
"""

from typing import Any, Union

from pydantic import BaseModel, Field

from .graph_element import (
    AnomalyNode,
    CauseNode,
    ComponentNode,
    EnvironmentNode,
    IssueNode,
    MetricNode,
    OptimizationNode,
    PatternNode,
    ResolutionNode,
    SystemNode,
    SymptomNode,
    GraphRelationship,
    TaskContextNode,
    TaskFeatureNode,
)

#: Union type accepted by :class:`KnowledgeGraph`.  Extend this when adding
#: new node types so that Pydantic can deserialise them correctly.
NodeUnion = Union[
    SymptomNode,
    EnvironmentNode,
    TaskFeatureNode,
    TaskContextNode,
    ComponentNode,
    CauseNode,
    ResolutionNode,
    MetricNode,
    AnomalyNode,
    IssueNode,
    SystemNode,
    PatternNode,
    OptimizationNode,
]


class KnowledgeGraph(BaseModel):
    """An extracted knowledge graph produced by an :class:`ExtractionStrategy`.

    Attributes:
        nodes:         All extracted nodes (any concrete :class:`BaseNode`
                       subclass).
        relationships: Directed edges between nodes.
        metadata:      Arbitrary graph-level metadata (e.g. ``graph_id``
                       assigned by the knowledge accumulator).
    """

    nodes: list[NodeUnion] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedKnowledge(BaseModel):
    """The full output of one knowledge-accumulation run.

    Attributes:
        graph:               The extracted :class:`KnowledgeGraph`.
        summary:             LLM-generated narrative summary of the graph,
                             stored as a searchable vector in Qdrant.
        key_insights:        Unstructured node descriptions selected for
                             vector indexing (``Task_Context`` and ``Symptom``
                             nodes with a non-empty ``description``).
        source_references:   Optional list of source document identifiers.
        extraction_metadata: Flags such as ``has_email``, ``has_task_data``.
    """

    graph: KnowledgeGraph
    summary: str = Field(
        ..., description="LLM-generated narrative summary of the graph."
    )
    key_insights: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Unstructured node descriptions selected for vector indexing. "
            "Each entry has keys: node_id, section, content."
        ),
    )
    source_references: list[str] = Field(default_factory=list)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEntry(BaseModel):
    """A single vector-database entry (used for typed retrieval results).

    Attributes:
        id:        Vector point ID.
        content:   The text that was embedded (node description or summary).
        entry:     Human-readable label or title for the entry.
        metadata:  Payload metadata stored alongside the vector.
        embedding: The embedding vector (populated when retrieved with vectors).
    """

    id: str
    content: str
    entry: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Combined result from a graph + vector database query.

    Attributes:
        graph_results:   Raw results from the graph database query.
        vector_results:  Raw results from the vector database query.
        combined_score:  Aggregated relevance score (set by the caller).
        metadata:        Query metadata.
    """

    graph_results: list[dict[str, Any]] = Field(default_factory=list)
    vector_results: list[dict[str, Any]] = Field(default_factory=list)
    combined_score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """The final output of a :class:`ReasoningNavigator` analysis run.

    Attributes:
        task_id:             Identifier of the analysed task.
        root_cause:          Human-readable root-cause statement.
        confidence:          LLM-reported confidence in [0, 1].
        resolution:          Recommended resolution.
        explanation:         Full LLM reasoning narrative.
        supporting_evidence: Evidence items from graph and vector results.
        email_content:       Draft email generated for the task submitter.
        metadata:            Additional context (clue counts, etc.).
    """

    task_id: str
    root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    resolution: str
    explanation: str
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    email_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
