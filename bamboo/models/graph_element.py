"""Graph database models."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    ERROR = "Error"
    ENVIRONMENT = "Environment"
    TASK_FEATURE = "Task_Feature"
    TASK_CONTEXT = "Task_Context"
    COMPONENT = "Component"
    CAUSE = "Cause"
    RESOLUTION = "Resolution"
    METRIC = "Metric"
    ANOMALY = "Anomaly"
    ISSUE = "Issue"
    SYSTEM = "System"
    PATTERN = "Pattern"
    OPTIMIZATION = "Optimization"
    EVENT = "Event"
    ACTION = "Action"
    DEPENDENCY = "Dependency"
    USER = "User"


class RelationType(str, Enum):
    """Types of relationships in the knowledge graph."""

    INDICATE = "indicate"
    ASSOCIATED_WITH = "associated_with"
    CONTRIBUTE_TO = "contribute_to"
    ORIGINATED_FROM = "originated_from"
    SOLVED_BY = "solved_by"
    SIGNALS = "signals"
    LEADS_TO = "leads_to"
    HAS_COMPONENT = "has_component"
    DEPENDS_ON = "depends_on"
    SUGGESTS = "suggests"
    IMPROVES = "improves"
    TRIGGERS = "triggers"
    AFFECTS = "affects"
    PERFORMED_BY = "performed_by"
    REPORTED_BY = "reported_by"
    ASSIGNED_TO = "assigned_to"
    APPROVED_BY = "approved_by"


class BaseNode(BaseModel):
    """Base class for all graph nodes."""

    id: Optional[str] = None
    name: str = Field(..., description="Canonical name of the node")
    description: Optional[str] = Field(
        default=None,
        description=("Additional detail about the node. "),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorNode(BaseNode):
    """Represents a main error message."""

    node_type: NodeType = NodeType.ERROR
    error_code: Optional[str] = None
    severity: Optional[str] = None


class EnvironmentNode(BaseNode):
    """Represents an external factor."""

    node_type: NodeType = NodeType.ENVIRONMENT
    properties: dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None  # e.g., "system", "network", "resource"


class TaskFeatureNode(BaseNode):
    """Represents a task feature as a key-value pair.

    ``attribute`` and ``value`` are the primary fields.  The canonical
    ``name`` is derived as ``"{attribute}={value}"`` (e.g. ``"RAM=1GB"``)
    so that two tasks with different RAM values produce distinct nodes that
    are never merged during canonicalization, while still being grouped by
    attribute in queries.

    Examples:
        TaskFeatureNode(attribute="RAM",     value="1GB",  name="RAM=1GB")
        TaskFeatureNode(attribute="RAM",     value="4GB",  name="RAM=4GB")
        TaskFeatureNode(attribute="OS",      value="Ubuntu 22.04", name="OS=Ubuntu 22.04")
        TaskFeatureNode(attribute="timeout", value="30s",  name="timeout=30s")
    """

    node_type: NodeType = NodeType.TASK_FEATURE
    attribute: str = Field(..., description="Feature key, e.g. 'RAM', 'OS', 'timeout'")
    value: str = Field(
        ..., description="Feature value, e.g. '1GB', 'Ubuntu 22.04', '30s'"
    )
    properties: dict[str, Any] = Field(default_factory=dict)


class TaskContextNode(BaseNode):
    """Represents an unstructured task context as free-form prose.

    Used for task characteristics whose value cannot be meaningfully
    canonicalized — e.g. reproduction steps, user-reported descriptions,
    free-text comments.

    - ``name`` is the attribute key (e.g. "steps_to_reproduce")
    - ``description`` is the unstructured prose value, indexed in the
      vector database for semantic search
    - This node is NOT stored in the graph database

    Examples:
        TaskContextNode(name="steps_to_reproduce",
                        description="Click submit, wait 5s, observe 500 error")
        TaskContextNode(name="user_report",
                        description="Intermittent failures observed after deploy")
    """

    node_type: NodeType = NodeType.TASK_CONTEXT


class ComponentNode(BaseNode):
    """Represents the origin of a cause."""

    node_type: NodeType = NodeType.COMPONENT
    system: Optional[str] = None
    version: Optional[str] = None


class CauseNode(BaseNode):
    """Represents a root cause."""

    node_type: NodeType = NodeType.CAUSE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency: int = Field(default=1, ge=1)


class ResolutionNode(BaseNode):
    """Represents an action to resolve an issue."""

    node_type: NodeType = NodeType.RESOLUTION
    steps: list[str] = Field(default_factory=list)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    estimated_duration: Optional[str] = None


class MetricNode(BaseNode):
    """Represents a system metric or KPI."""

    node_type: NodeType = NodeType.METRIC
    metric_type: Optional[str] = None  # e.g., "performance", "resource", "business"
    unit: Optional[str] = None
    threshold: Optional[dict[str, Any]] = Field(default_factory=dict)


class AnomalyNode(BaseNode):
    """Represents a detected anomaly."""

    node_type: NodeType = NodeType.ANOMALY
    severity: str = "medium"  # low, medium, high, critical
    detection_method: Optional[str] = None
    detected_at: Optional[str] = None


class IssueNode(BaseNode):
    """Represents a system issue or incident."""

    node_type: NodeType = NodeType.ISSUE
    status: str = "open"  # open, investigating, resolved, closed
    priority: str = "medium"
    impact_scope: Optional[list[str]] = Field(default_factory=list)


class SystemNode(BaseNode):
    """Represents a system or service."""

    node_type: NodeType = NodeType.SYSTEM
    system_type: Optional[str] = None
    version: Optional[str] = None
    status: str = "active"


class PatternNode(BaseNode):
    """Represents an operational pattern."""

    node_type: NodeType = NodeType.PATTERN
    pattern_type: Optional[str] = None  # e.g., "usage", "failure", "performance"
    frequency: int = Field(default=1, ge=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OptimizationNode(BaseNode):
    """Represents an optimization opportunity or implementation."""

    node_type: NodeType = NodeType.OPTIMIZATION
    optimization_type: Optional[str] = None  # performance, cost, reliability
    expected_impact: Optional[dict[str, Any]] = Field(default_factory=dict)
    implementation_status: str = "proposed"


class EventNode(BaseNode):
    """Represents a system event."""

    node_type: NodeType = NodeType.EVENT
    event_type: Optional[str] = None
    timestamp: Optional[str] = None
    severity: Optional[str] = None


class ActionNode(BaseNode):
    """Represents an automated or manual action."""

    node_type: NodeType = NodeType.ACTION
    action_type: Optional[str] = None
    automated: bool = False
    risk_level: str = "low"
    execution_steps: list[str] = Field(default_factory=list)


class DependencyNode(BaseNode):
    """Represents a system dependency."""

    node_type: NodeType = NodeType.DEPENDENCY
    dependency_type: Optional[str] = None  # service, library, infrastructure
    version_requirement: Optional[str] = None
    criticality: str = "medium"


class UserNode(BaseNode):
    """Represents a user in the system."""

    node_type: NodeType = NodeType.USER
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None  # e.g., "operator", "engineer", "admin"
    permissions: list[str] = Field(default_factory=list)
    active: bool = True
    last_activity: Optional[str] = None


class GraphRelationship(BaseModel):
    """Represents a relationship between nodes."""

    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
