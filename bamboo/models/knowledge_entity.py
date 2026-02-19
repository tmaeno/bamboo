"""Knowledge extraction and storage models."""

from typing import Any, Union

from pydantic import BaseModel, Field

from .graph_element import (
    CauseNode,
    ComponentNode,
    EnvironmentNode,
    SymptomNode,
    GraphRelationship,
    ResolutionNode,
    TaskContextNode,
    TaskFeatureNode,
)

NodeUnion = Union[
    SymptomNode,
    EnvironmentNode,
    TaskFeatureNode,
    TaskContextNode,
    ComponentNode,
    CauseNode,
    ResolutionNode,
]


class KnowledgeGraph(BaseModel):
    """Represents an extracted knowledge graph."""

    nodes: list[NodeUnion] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedKnowledge(BaseModel):
    """Represents extracted and canonicalized knowledge."""

    graph: KnowledgeGraph
    summary: str = Field(..., description="LLM-generated entry of the graph")
    key_insights: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Unstructured node descriptions selected for vector indexing.",
    )
    source_references: list[str] = Field(default_factory=list)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEntry(BaseModel):
    """Represents a semantic entry for vector storage."""

    id: str
    content: str
    entry: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Represents a query result from databases."""

    graph_results: list[dict[str, Any]] = Field(default_factory=list)
    vector_results: list[dict[str, Any]] = Field(default_factory=list)
    combined_score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Represents the final analysis result."""

    task_id: str
    root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    resolution: str
    explanation: str
    supporting_evidence: list[dict[str, Any]] = Field(default_factory=list)
    email_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
