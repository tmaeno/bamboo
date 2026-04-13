"""Graph node and relationship models for the Bamboo knowledge graph.

The graph schema is::

    Symptom        -[indicate]->          Cause
    Environment    -[associated_with]->   Cause
    Task_Feature   -[contribute_to]->     Cause
    Component      -[originated_from]->   Cause
    Cause          -[solved_by]->         Resolution
    Cause          -[investigated_by]->   Procedure

``TaskContextNode`` and ``JobInstanceContextNode`` are special cases: they are only
stored in the vector database (for semantic search) and are never persisted in
the graph database.

``ProcedureNode`` encodes how to investigate a given cause type.  It is extracted
from email threads alongside causes and resolutions, and linked from the cause via
an ``investigated_by`` edge.  The edge carries per-incident investigation parameters
(job filter conditions, metrics) as a list appended on each re-encounter, so no
information is lost when multiple incidents map to the same procedure node.

The reasoning navigator queries ``Cause -[investigated_by]-> Procedure`` after
Phase 1 to decide whether and how to run a Phase 2 investigation.
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
    PROCEDURE = "Procedure"
    TASK_FEATURE = "Task_Feature"
    AGGREGATED_JOB_FEATURE = "Aggregated_Job_Feature"  # deprecated — kept for existing graph data
    JOB_INSTANCE = "Job_Instance"                       # deprecated — kept for existing graph data
    TASK_CONTEXT = "Task_Context"
    JOB_INSTANCE_CONTEXT = "Job_Instance_Context"
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
    INVESTIGATED_BY = "investigated_by"
    ASSOCIATED_WITH = "associated_with"
    HAS_JOB_PATTERN = "has_job_pattern"    # deprecated — kept for existing graph data
    HAS_JOB_INSTANCE = "has_job_instance"  # deprecated — kept for existing graph data
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


class AggregatedJobFeatureNode(BaseNode):
    """An aggregated, reusable job-execution pattern derived from one or more jobs.

    Where :class:`TaskFeatureNode` captures how a *task was configured*,
    ``AggregatedJobFeatureNode`` captures how the jobs *actually executed* —
    site failure rates, dominant error codes, CPU consumption ranges,
    transformation versions, etc.  Individual job IDs are never stored; only
    the aggregated pattern is.

    The canonical ``name`` follows the same ``"{attribute}={value}"`` convention
    as :class:`TaskFeatureNode` (e.g. ``"site_failure_rate=AGLT2:high(>50%)"``,
    ``"transformation=Athena-25.0.1"``).

    ``AggregatedJobFeatureNode`` participates in the same ``contribute_to →
    Cause`` edge as ``TaskFeatureNode``, plus the ``has_job_pattern`` edge from
    a :class:`SymptomNode`::

        Symptom -[has_job_pattern]-> Aggregated_Job_Feature
        Aggregated_Job_Feature -[contribute_to]-> Cause

    Attributes:
        attribute:   Feature key, e.g. ``"site_failure_rate"``,
                     ``"transformation"``, ``"cpu_time"``.
        value:       Aggregated label, e.g. ``"AGLT2:high(>50%)"``,
                     ``"Athena-25.0.1"``, ``"1-6h"``.
        job_count:   Number of jobs this aggregate was derived from.
        properties:  Additional key-value properties.
    """

    node_type: NodeType = NodeType.AGGREGATED_JOB_FEATURE
    attribute: str = Field(..., description="Aggregated job feature key.")
    value: str = Field(..., description="Aggregated job feature value / label.")
    job_count: int = Field(default=0, ge=0, description="Number of jobs aggregated.")
    properties: dict[str, Any] = Field(default_factory=dict)


class JobInstanceNode(BaseNode):
    """A canonical job failure pattern node, keyed by site + error code.

    Follows the same incident-agnostic naming convention as all other node
    types: the ``name`` encodes the *failure pattern*, not the job identity.
    This lets the same node be reused across tasks that share the same
    site+error combination, enabling direct multi-hop graph traversal between
    related incidents.

    Naming convention:  ``"job_failure:{site}:{error_channel}:{error_code}"``
      e.g.              ``"job_failure:AGLT2:pilot:1099"``
    If site is unknown: ``"job_failure:unknown:{error_channel}:{error_code}"``

    ``description`` holds representative diagnostic text (``pilotErrorDiag`` /
    ``transExitCode`` diag) from one of the failing jobs; it is embedded and
    indexed in Qdrant for semantic similarity search across incidents.

    ``JobInstanceNode`` can optionally carry a ``HAS_CAUSE`` edge when the
    job-level diagnostic text implies a cause that is **more specific** than the
    task-level :class:`CauseNode`.  Do not duplicate the task-level cause.

    .. note::
        ``job_id`` is NOT a field — it is incident-specific and would prevent
        cross-task node merging.

    Attributes:
        site:          Compute site where the job ran, e.g. ``"AGLT2"``.
        error_code:    Pilot or payload error code, e.g. ``"1099"``.
        error_channel: Error source: ``"pilot"``, ``"payload"``, or ``"ddm"``.
        exit_code:     Raw numeric exit code (``transExitCode``).
    """

    node_type: NodeType = NodeType.JOB_INSTANCE
    site: Optional[str] = None
    error_code: Optional[str] = None
    error_channel: Optional[str] = None
    exit_code: Optional[int] = None


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


class JobInstanceContextNode(BaseNode):
    """Full diagnostic text from representative failing jobs — vector DB only.

    Stores the raw ``pilotErrorDiag`` / ``transExitCode`` diagnostic text that
    cannot be reduced to a canonical site+error pattern but is valuable for
    semantic similarity search (e.g. two tasks with slightly different wording
    but the same underlying failure).

    Complements :class:`JobInstanceNode`: ``JobInstanceNode`` is the graph node
    (merged across tasks by site+error pattern), while ``JobInstanceContextNode``
    carries the full diagnostic text for Qdrant embedding.

    - ``name`` follows the same key as the corresponding ``JobInstanceNode``,
      e.g. ``"job_instance_context:AGLT2:pilot:1099"``.
    - ``description`` is the full diagnostic text — embedded and indexed in
      Qdrant for semantic search.

    .. important::
        This node is **NOT** stored in the graph database (Neo4j).  It must
        not be referenced by graph relationships.
    """

    node_type: NodeType = NodeType.JOB_INSTANCE_CONTEXT


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


class ProcedureNode(BaseNode):
    """An investigation procedure for a cause type, extracted from email threads.

    Encodes *how to investigate* a given cause — which jobs to look at, which
    metrics to examine, or what related tasks to inspect.  Extracted from email
    threads alongside causes and resolutions, then linked from the cause via an
    ``investigated_by`` edge.

    Merge key: ``"{strategy_type}:{cause_name}"`` — one node per
    strategy+cause combination.  Per-incident investigation parameters (job
    filter conditions, metrics) are stored on the ``investigated_by`` edge as a
    list appended on each re-encounter, so no information is lost when multiple
    incidents map to the same node.

    The reasoning navigator queries ``Cause -[investigated_by]-> Procedure``
    after Phase 1 cause identification.  It feeds ``strategy_type`` to an LLM
    that selects the appropriate MCP tool from the available tool catalogue.
    If no procedure is found for a cause, the navigator requests human input.

    Attributes:
        strategy_type: Concise natural-language description of the investigation,
                       e.g. ``"investigate finished normal jobs with high
                       cpuConsumptionTime and wallTime"``.  The LLM uses this to
                       select MCP tools at navigation time.
    """

    node_type: NodeType = NodeType.PROCEDURE
    strategy_type: str = Field(
        default="",
        description=(
            "Concise natural-language description of the investigation strategy, "
            "e.g. 'investigate finished normal jobs with high cpuConsumptionTime "
            "and wallTime'.  Used by the navigator LLM to select MCP tools."
        ),
    )


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
