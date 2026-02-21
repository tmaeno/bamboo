"""Graph node and relationship models for the Bamboo knowledge graph.

The graph schema is::

    Symptom        -[indicate]->        Cause
    Environment    -[associated_with]-> Cause
    Task_Feature   -[contribute_to]->   Cause
    Component      -[originated_from]-> Cause
    Cause          -[solved_by]->       Resolution

``TaskContextNode`` is a special case: it is only stored in the vector
database (for semantic search) and is never persisted in the graph database.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Discriminator enum used as the ``node_type`` field on every node.

    Values match the Neo4j label (and Qdrant section name) for each type.
    Only the types actively used by the Panda extraction pipeline are listed
    first; the remainder are available for future strategies.
    """

    # Core incident-analysis types
    SYMPTOM = "Symptom"
    CAUSE = "Cause"
    RESOLUTION = "Resolution"
    TASK_FEATURE = "Task_Feature"
    TASK_CONTEXT = "Task_Context"
    ENVIRONMENT = "Environment"
    COMPONENT = "Component"

    # Extended types (available for future strategies)
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
    """Edge type enum.  Values are used verbatim as Neo4j relationship types.

    Core relationships (used by the Panda pipeline):
        - ``indicate``      : Symptom → Cause
        - ``solved_by``     : Cause   → Resolution
        - ``contribute_to`` : Task_Feature / Environment → Cause
        - ``originated_from``: Component → Cause

    Extended relationships (available for future strategies):
        - ``associated_with``, ``signals``, ``leads_to``, ``has_component``,
          ``depends_on``, ``suggests``, ``improves``, ``triggers``,
          ``affects``, ``performed_by``, ``reported_by``, ``assigned_to``,
          ``approved_by``
    """

    INDICATE = "indicate"
    SOLVED_BY = "solved_by"
    CONTRIBUTE_TO = "contribute_to"
    ORIGINATED_FROM = "originated_from"
    ASSOCIATED_WITH = "associated_with"
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
    """Common fields shared by every graph node.

    Attributes:
        id:          Stable UUID assigned by the knowledge accumulator before
                     the node is persisted.  ``None`` until assigned.
        name:        Canonical node identity used as the merge key in
                     ``get_or_create_canonical_node``.  Must be deterministic
                     across incidents for the same real-world concept.
        description: Optional prose detail.  For ``SymptomNode`` this holds the
                     raw error message text; for ``TaskContextNode`` it holds
                     the full unstructured value.  Indexed in the vector DB for
                     semantic search when present.
        metadata:    Arbitrary extra fields (source, confidence scores, etc.)
                     stored as JSON in both databases.
    """

    id: Optional[str] = None
    name: str = Field(..., description="Canonical name used as the graph merge key.")
    description: Optional[str] = Field(
        default=None,
        description=(
            "Prose detail about the node.  Indexed in the vector DB for "
            "semantic search."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    """A directed, typed edge between two nodes.

    Attributes:
        source_id:     ``name`` (or ``id`` after storage) of the source node.
        target_id:     ``name`` (or ``id`` after storage) of the target node.
        relation_type: Edge type; must be a :class:`RelationType` value.
        confidence:    Extraction confidence in [0, 1].
        properties:    Arbitrary extra properties stored on the edge.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Node classes
# ---------------------------------------------------------------------------


class SymptomNode(BaseNode):
    """A canonical, reusable class of failure observed in an incident.

    In the Panda pipeline a ``SymptomNode`` is produced from the
    ``ErrorMessage`` field of ``task_data``.  The LLM classifies the raw
    message into a short, stable category name (e.g. ``"TooManyFilesInDataset"``
    stored as ``name``) while the verbatim raw message is preserved in
    ``description`` for traceability and vector search.

    Because ``name`` is canonical and incident-agnostic, the same symptom from
    many different tasks all merge into a single node in the graph, building up
    frequency evidence that connects the symptom to its root causes.

    Attributes:
        error_code: Optional machine-readable error code (e.g. ``"ERR-4042"``).
        severity:   Optional severity label (e.g. ``"high"``, ``"critical"``).
    """

    node_type: NodeType = NodeType.SYMPTOM
    error_code: Optional[str] = None
    severity: Optional[str] = None


class EnvironmentNode(BaseNode):
    """An external environmental factor that contributes to a cause.

    Examples: ``"kubernetes-1.28"``, ``"python-3.11"``, ``"us-east-1"``.

    Attributes:
        category:   Broad category, e.g. ``"system"``, ``"network"``,
                    ``"resource"``.
        properties: Additional key-value properties.
    """

    node_type: NodeType = NodeType.ENVIRONMENT
    properties: dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None


class TaskFeatureNode(BaseNode):
    """A discrete, comparable task attribute stored as ``attribute=value``.

    The canonical ``name`` is always ``"{attribute}={value}"`` (e.g.
    ``"RAM=4GB"``, ``"OS=Ubuntu 22.04"``).  This format ensures that two tasks
    with different values for the same attribute produce distinct nodes that are
    never merged, while still being queryable by attribute.

    Use ``TaskFeatureNode`` when the value is categorical or unit-based — i.e.
    it could appear in a dropdown or be meaningfully compared.  For free-form
    prose values use :class:`TaskContextNode` instead.

    Attributes:
        attribute:   Feature key, e.g. ``"RAM"``, ``"OS"``, ``"timeout"``.
        value:       Feature value, e.g. ``"4GB"``, ``"Ubuntu 22.04"``, ``"30s"``.
        properties:  Additional key-value properties.
    """

    node_type: NodeType = NodeType.TASK_FEATURE
    attribute: str = Field(..., description="Feature key, e.g. 'RAM', 'OS', 'timeout'.")
    value: str = Field(..., description="Feature value, e.g. '4GB', 'Ubuntu 22.04'.")
    properties: dict[str, Any] = Field(default_factory=dict)


class TaskContextNode(BaseNode):
    """An unstructured task characteristic stored only in the vector database.

    Use this for prose values that cannot be meaningfully canonicalised —
    reproduction steps, user-reported descriptions, free-text comments, etc.

    - ``name`` is the attribute key (e.g. ``"steps_to_reproduce"``).
    - ``description`` is the full unstructured prose value, embedded and
      indexed in Qdrant for semantic search.

    .. important::
        This node is **NOT** stored in the graph database (Neo4j).  It must
        not be referenced by graph relationships.
    """

    node_type: NodeType = NodeType.TASK_CONTEXT


class ComponentNode(BaseNode):
    """A system component or service identified as the origin of a cause.

    Attributes:
        system:  Parent system name, e.g. ``"auth-service"``.
        version: Component version string, e.g. ``"2.3.1"``.
    """

    node_type: NodeType = NodeType.COMPONENT
    system: Optional[str] = None
    version: Optional[str] = None


class CauseNode(BaseNode):
    """A root cause or contributing cause of an incident.

    ``name`` is a canonical, incident-agnostic phrase (e.g.
    ``"input dataset exceeds file limit"``).  The :class:`CanonicalNodeStore`
    in the Panda extractor ensures this is stable across incidents.

    Attributes:
        confidence: Extraction confidence in [0, 1].
        frequency:  Number of incidents in which this cause has been observed
                    (incremented by the graph DB on each re-encounter).
    """

    node_type: NodeType = NodeType.CAUSE
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency: int = Field(default=1, ge=1)


class ResolutionNode(BaseNode):
    """An action or set of actions that resolve a cause.

    ``name`` is a canonical, incident-agnostic phrase (e.g.
    ``"split dataset into smaller subsets"``), normalised by the
    :class:`CanonicalNodeStore`.

    Attributes:
        steps:              Ordered list of discrete action steps.
        success_rate:       Fraction of applications that resolved the issue,
                            updated by the graph DB over time.
        estimated_duration: Human-readable time estimate, e.g. ``"30 minutes"``.
    """

    node_type: NodeType = NodeType.RESOLUTION
    steps: list[str] = Field(default_factory=list)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    estimated_duration: Optional[str] = None


# ---------------------------------------------------------------------------
# Extended node types (available for future strategies)
# ---------------------------------------------------------------------------


class MetricNode(BaseNode):
    """A system metric or KPI."""

    node_type: NodeType = NodeType.METRIC
    metric_type: Optional[str] = None
    unit: Optional[str] = None
    threshold: Optional[dict[str, Any]] = Field(default_factory=dict)


class AnomalyNode(BaseNode):
    """A detected anomaly."""

    node_type: NodeType = NodeType.ANOMALY
    severity: str = "medium"
    detection_method: Optional[str] = None
    detected_at: Optional[str] = None


class IssueNode(BaseNode):
    """A system issue or incident ticket."""

    node_type: NodeType = NodeType.ISSUE
    status: str = "open"
    priority: str = "medium"
    impact_scope: Optional[list[str]] = Field(default_factory=list)


class SystemNode(BaseNode):
    """A system or service."""

    node_type: NodeType = NodeType.SYSTEM
    system_type: Optional[str] = None
    version: Optional[str] = None
    status: str = "active"


class PatternNode(BaseNode):
    """A recurring operational pattern."""

    node_type: NodeType = NodeType.PATTERN
    pattern_type: Optional[str] = None
    frequency: int = Field(default=1, ge=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OptimizationNode(BaseNode):
    """An optimization opportunity or implementation."""

    node_type: NodeType = NodeType.OPTIMIZATION
