"""Code Map node models.

The Code Map is a machine-derived view of a target system's source: where the
code *determines* the value of a subject, under what conditions, and where it
hands off to another system.  It is stored in the same Neo4j database as the
incident graph but under its own labels, because the two have opposite
lifecycles -- the Code Map is regenerated from source at will, the incident
graph is human-validated and irreplaceable.

The node kinds:

``SubjectNode``
    An attribute whose value is worth asking "why is it this?" about --
    promoted from the declared spec attributes by the census criteria.
``JunctionNode``
    A place where the code settles a subject's value.  Its branches are the
    possible outcomes, each with the path condition that selects it.  A
    junction does not *judge*; the branch taken follows deterministically from
    the conditions, which is why it is not called a decision point (that term
    is reserved for the constrained points where an LLM or a human chooses).
``FilterStageNode``
    One reason a candidate was dropped on the way to a selection.  Separate
    from a junction because every stage of a chain runs and each removes some
    candidates, where a junction's branches are alternatives and one wins.
``BoundaryNode``
    Where causation crosses into a system this map does not cover.  Modelled
    explicitly rather than left as an absence so that adding the other
    system's map later is a *binding* operation instead of a re-derivation.
``ValueEnumNode``
    One ``NAME = value`` constant, so an observed code can be decoded.

Identity is a semantic signature, never a file position.  Positions move --
between ``panda-server-source`` 0.8.1 and 1.0.2 the pilot boundary moved file
entirely (``jobdispatcher/JobDispatcher.py`` disappeared; the entry is now
``api/v1/pilot_api.py::update_job``) while remaining the same boundary.  The
anchor records where the evidence was found; the signature says what it *is*.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from bamboo.models.graph_element import BaseNode, NodeType


class SourceModule(BaseModel):
    """One parsed source file, shared by every recognizer in a build.

    Parsing and hashing happen once per module rather than once per
    recognizer: with several slices reading the same tree, re-reading it per
    slice would multiply the build's cost for nothing.
    """

    model_config = {"arbitrary_types_allowed": True}

    package: str
    rel_path: str  # package-prefixed, e.g. "pandajedi/jediorder/JobGenerator.py"
    tree: Any  # ast.Module
    source: str
    blob_sha: str


class Anchor(BaseModel):
    """Where in the source a Code Map fact was found.

    Provenance for a human following the map back to the code -- deliberately
    *not* an identity.  ``blob_sha`` lets a rebuild tell whether the enclosing
    file changed at all; ``line_span`` is only ever as good as the version the
    map was derived from.
    """

    package: str
    file: str
    line_start: int
    line_end: Optional[int] = None
    blob_sha: Optional[str] = None

    def as_ref(self) -> str:
        """Return a ``file:line`` reference for display."""
        return f"{self.file}:{self.line_start}"


class SubjectNode(BaseNode):
    """An attribute the reasoning may start from, with its promotion evidence.

    ``name`` is qualified as ``"<spec_class>.<attribute>"`` because the same
    attribute name means different things in different classes -- ``FileSpec``
    and ``JediFileSpec`` both declare ``status``.  The same shape recurs for
    error codes (``(namespace, code)``) and for maps (``map_id``), so every
    Code Map identifier is qualified as a rule.
    """

    node_type: NodeType = NodeType.SUBJECT
    map_id: str
    derived_from: str
    spec_class: str = Field(
        ...,
        description=(
            "What qualifies the attribute: a spec class, or a table where no "
            "spec holds that row.  Named for the common case; ``qualifier_kind`` "
            "says which it is."
        ),
    )
    qualifier_kind: str = Field(
        default="spec",
        description=(
            "spec | table.  Recorded so a reader can tell ``JEDI_Events.status`` "
            "from ``JediTaskSpec.status`` -- one names a table, the other a "
            "class, and they are different kinds of claim."
        ),
    )
    attribute: str
    criteria: list[str] = Field(
        default_factory=list,
        description="Promotion criteria satisfied, e.g. ['1:state-gate-in-where'].",
    )
    vocabulary: list[str] = Field(
        default_factory=list,
        description="Declared or observed value set, when the code states one.",
    )
    selected_values: list[str] = Field(
        default_factory=list,
        description=(
            "Values some query selects rows on.  The counterpart of the "
            "outcomes the junctions write: together they make a state machine "
            "out of a pile of writes, since a value nothing selects on is one "
            "nothing ever moves away from."
        ),
    )
    selected_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "``{value: the functions whose query selects rows on it}``.  Kept "
            "per value rather than per subject because the two answer different "
            "questions: ``JediTaskSpec.status`` is selected on thirty-one values "
            "by dozens of functions, and pooling them says only that the subject "
            "is read.  Per value it says who to ask -- ``finishing`` is selected "
            "by exactly one query in the whole corpus."
        ),
    )
    selection_gates: list[str] = Field(
        default_factory=list,
        description=(
            "Tables that bound which rows the queries selecting this subject "
            "can see at all, and that nothing in this map writes.  A different "
            "reason for a row not to be picked up than any value in "
            "``selected_values``: ``JEDI_AUX_Status_MinTaskID`` is joined by "
            "thirty-one functions on ``jediTaskID >= min_jediTaskID``, so when "
            "it goes stale a task is invisible to all of them whatever its "
            "status is.  Named here because a status the map says is selected "
            "is only half an answer without what bounds the selecting."
        ),
    )

    @staticmethod
    def make_name(spec_class: str, attribute: str) -> str:
        return f"{spec_class}.{attribute}"


#: What a line is about.  A junction leaves two kinds and they are not
#: interchangeable: one carries the value, so a probe can be built from it by
#: substituting the value observed; the other carries the row count the write
#: returned, which is the only trace a compare-and-set leaves when it loses.
#: Reading the second as the first would build a pattern for a value that
#: never appears in it.
REPORTS_DECISION = "decision"
REPORTS_ROWS_CHANGED = "rows_changed"


class Emit(BaseModel):
    """A log line the code leaves when it settles a value, and where it lands.

    The file belongs to the emit rather than to the node, because one junction
    leaves two kinds of line in two different places.  ``updateTask_JEDI``
    writes ``updated N rows`` through the proxy's own inherited logger, while
    the ``set task_status=`` line about that same junction is written by the
    knight that called it.  ``JunctionNode.owns_logger`` answers that with one
    bit for the whole node, which was right for the question it was added for
    and is the wrong shape for this one.

    ``log_files`` empty means the module writes through a wrapper its caller
    supplied and the map cannot say where that goes.  Left empty rather than
    filled in: a probe against a guessed file comes back silent, and silence is
    what the eliminator reads.
    """

    template: str = Field(
        ...,
        description=(
            "The line as the source frames it, with run-time parts left as "
            "``{}`` -- the literal frame is what a production log is matched on."
        ),
    )
    log_level: Optional[str] = Field(
        default=None,
        description=(
            "Level it is written at, or None where one hop could not settle it. "
            "A DEBUG line the map offers as an observable does not exist if the "
            "deployment runs at INFO, which only production can say."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list, description="Files this particular line lands in."
    )
    reports: str = Field(
        default=REPORTS_DECISION,
        description=(
            f"{REPORTS_DECISION} | {REPORTS_ROWS_CHANGED} -- whether the line "
            "carries the value the code settled, or the number of rows the "
            "write actually changed.  Both are evidence and they answer "
            "different questions: the first says the code decided, the second "
            "says the row took it."
        ),
    )


class Branch(BaseModel):
    """One possible outcome of a junction and the condition that selects it.

    ``outcome`` is per-branch, not a property of the junction: a single writer
    reaches several values, and "no transition at all" is itself an outcome
    that matters (a task sitting in ``pending`` because a lock was held is not
    the same as one that took a different branch).

    ``path_condition`` is the conjunction of the ``if``/``elif``/``else`` tests
    dominating the write.  It is not solved symbolically -- the values come
    from observation and are substituted in, which answers "which branch
    actually fired" rather than "which branch could fire".

    ``row_precondition`` is the other kind of condition, and the two are not
    interchangeable: the path condition is evaluated by the code, before the
    write, and decides whether it is attempted; the row precondition is
    evaluated by the database, during the write, against the row as it stands,
    and decides whether it lands.  Only the first is visible in a log.
    """

    outcome: str = Field(
        ...,
        description=(
            "Resulting value, ``passthrough(<attr>)`` when carried from "
            "elsewhere, or ``NO_TRANSITION``."
        ),
    )
    path_condition: list[str] = Field(default_factory=list)
    row_precondition: list[str] = Field(
        default_factory=list,
        description=(
            "What the row already had to say for this write to change it, when "
            "the statement tests a column it also writes.  Losing that race is "
            "silent -- zero rows changed, no exception, a count the caller "
            "usually discards -- so a branch carrying one is a branch whose "
            "outcome the code can announce without the row ever taking it."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Machine-readable tags the code names when this branch fires, e.g. "
            "``action=set_exhausted`` and ``reason=low_efficiency``.  A list "
            "because the signature is a *set*: PanDA writes an action and a "
            "reason as separate tokens and puts words between them (``action="
            "set_exhausted since reason=many_shorter_jobs``), so a single "
            "joined string would match no message production ever writes.  "
            "Read where the write is read, from the block that records the "
            "line, and the same tokens land in the record itself when the "
            "block also calls a message setter -- which is what lets an "
            "observed ``errorDialog`` name one branch out of six."
        ),
    )
    messages: list[str] = Field(
        default_factory=list,
        description=(
            "The frames the block records when this branch fires, holes left "
            "open.  The untagged half of the same signature, and the one "
            "production mostly writes: of thirty tasks found in ``exhausted`` "
            "with a message on the record, none carried a tag -- they were "
            "retry refusals and the goal check, which write prose.  Weaker "
            "evidence than a tag, since wording drifts between releases and "
            "two arms can share a frame, so a match names an arm only where it "
            "is the only one."
        ),
    )
    line: Optional[int] = Field(
        default=None,
        description=(
            "Where this branch's write is.  The junction's anchor is the first "
            "of them, and six branches of one function sharing it answers "
            "*which branch* with a line that belongs to another; an index that "
            "names a branch has to be able to say where to read.  Not part of "
            "the content hash: a branch's identity is its outcome, and a line "
            "moves whenever anything above it is edited."
        ),
    )
    emits: list[Emit] = Field(
        default_factory=list,
        description=(
            "Lines the code writes when this branch settles the value, each "
            "with the file it lands in.  Withdrawn once, when the only "
            "templates that survived promotion were identifier frames rather "
            "than diagnostics; raised again now that a consumer needs them -- "
            "``strategy.line_shape`` reads them to build the probe, and while "
            "they were empty it fell back to one hard-coded shape covering one "
            "subject."
        ),
    )
    log_level: Optional[str] = Field(
        default=None,
        description=(
            "Level of the diagnostic emit.  Recorded because a DEBUG line the "
            "map promises as an observable does not exist if production runs "
            "at INFO -- only comparing against real logs settles that."
        ),
    )
    order: int = 0
    tier: int = Field(
        default=1,
        description=(
            "1 when the outcome is statically resolved; 2 when the writer is "
            "known but the value is only determined at run time.  Tier 2 still "
            "supports localize/prune, which read observed output rather than "
            "the static outcome."
        ),
    )


# How work arrives at a junction.  Declared with the field that carries it
# rather than in the recognizer that assigns it, because both halves of the map
# read them: the extraction to label an entry, and an investigation to decide
# whether a stalled value will come back on its own.
POLLED = "polled"
COMMAND = "command"
MESSAGE = "message"
REQUEST = "request"

#: Triggers that re-evaluate.  A subject only these can reach recovers by
#: itself, so waiting is a plan; a subject only the others can reach is
#: consumed once, and a missed one stays missed.
SELF_REPAIRING_TRIGGERS = frozenset({POLLED})


class EntryPoint(BaseModel):
    """One way control reaches a junction, and what it hands over on the way.

    Plural on purpose.  ``getTasksToBeProcessed_JEDI`` is called by
    ``JobGenerator`` with ``minPriority`` and ``maxNumJobs`` and by the message
    processor without them, so the throttle guard cannot fire on the message
    path at all -- the set of reasons a task was not picked up differs by entry,
    which makes the entry part of the structure rather than context.
    """

    trigger: str = Field(
        ...,
        description=(
            "polled | command | message.  How work arrives, which decides "
            "whether missing it repairs itself: a loop re-evaluates, a command "
            "row and a broker message are each consumed once."
        ),
    )
    entry: str = Field(..., description="Module that carries the trigger.")
    via: Optional[str] = Field(
        default=None,
        description="Method through which the entry reaches this junction, if not its own.",
    )
    arg_binding: dict[str, str] = Field(
        default_factory=dict,
        description="Keyword arguments this entry supplies at that call.",
    )


class JunctionNode(BaseNode):
    """A place where the code settles a subject's value.

    ``name`` is the semantic signature, which must survive refactoring: the
    same junction keeps its identity when its file, line, or enclosing module
    changes.  Positions live in ``anchor``.
    """

    node_type: NodeType = NodeType.JUNCTION_POINT
    map_id: str
    derived_from: str
    subject: str = Field(..., description="Qualified subject name this junction writes.")
    owner: str = Field(..., description="module::qualname that contains the write.")
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files the module holding this junction writes to, e.g. "
            "'panda-ContentsFeeder.log'.  More than one when the code runs in "
            "more than one process: the proxy mixins land in "
            "'panda-DBProxy.log' under the server and 'panda-JediDBProxy.log' "
            "under a knight, and which one is a runtime fact, so both are named."
            "  This is where the code *lives*, which is not always where a line "
            "about it firing appears -- see ``caller_log_files``."
        ),
    )
    caller_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files belonging to the code that reaches this junction.  A "
            "separate field rather than more entries in ``log_files`` because "
            "it answers a different question, and merging them would repeat "
            "the mistake it exists to correct: a ``db_proxy_mods`` method "
            "declares no logger of its own, so ``log_files`` names the mixin's "
            "two proxy files while the diagnostic -- 'set task_status=' -- is "
            "written by the knight that called it.  Production confirms the "
            "split: of 33 files asked, that line is in exactly five -- "
            "ContentsFeeder, JobGenerator, PostProcessor, TaskCommando and "
            "TaskRefiner -- and in neither proxy file."
        ),
    )
    owns_logger: bool = Field(
        default=True,
        description=(
            "Whether the module holding this junction declares its own logger. "
            "False means ``log_files`` are the files of the classes that mix it "
            "in -- true for where its own output lands, since the SQL comment "
            "trace does appear there, but *not* a place a line about this "
            "junction firing can appear, because the code that writes such a "
            "line is the caller.  The difference decides whether an empty grep "
            "may be read as 'it did not fire': asking ``panda-DBProxy.log`` for "
            "``set task_status=`` returns nothing however often the junction "
            "runs, so an eliminator that did not know this would rule out every "
            "proxy candidate at once and confidently keep the wrong one."
        ),
    )
    attribution: str = Field(
        default="certain",
        description=(
            "How the subject's class was settled: ``certain`` (the code states "
            "the type), ``structural`` (only one class declares every "
            "attribute the code touches on the object), ``heuristic`` (the "
            "variable's name, narrowed by the module's imports), or "
            "``unresolved``.  A separate axis from ``Branch.tier`` on purpose "
            "-- tier is about whether the *outcome* is statically known, and a "
            "write can state its outcome as a literal while leaving the class "
            "it wrote to open.  Recorded so a guessed attribution is visible "
            "rather than indistinguishable from a stated one."
        ),
    )
    structural_subject: Optional[str] = Field(
        default=None,
        description=(
            "What the attributes touched on the object imply, derived "
            "independently of how ``subject`` was settled.  Kept even when it "
            "merely agrees, because agreement is the point: the declaration "
            "and the usage are two expressions of one fact, and comparing them "
            "is a gate that needs no production data."
        ),
    )
    branches: list[Branch] = Field(default_factory=list)
    entry_points: list[EntryPoint] = Field(
        default_factory=list,
        description=(
            "How this junction is reached.  Empty means nothing the map "
            "recognises starts it -- reported rather than defaulted, since "
            "assuming a loop would make the self-repair property unusable."
        ),
    )
    anchor: Optional[Anchor] = None

    def observable_log_files(self) -> list[str]:
        """Files where a line *about this junction firing* can appear.

        The caller's, plus its own only where its module declares a logger.  An
        inherited file is deliberately left out: it is a true statement about
        where the code's own output lands -- the SQL comment trace really is in
        ``panda-DBProxy.log`` -- and a false one about where a line saying the
        junction fired appears, because the code writing that line is the
        caller.  Production settles it: of thirty-three files asked,
        ``set task_status=`` is in exactly five, all of them callers, and in
        neither proxy file.

        A method on the node rather than a rule in one reader, because two
        readers need it and they would drift.  Deriving an investigation asks
        it to know where to grep; the gate that reports dead code paths asks it
        to know whether a junction has any surviving log at all, and answering
        that from ``log_files`` alone credits a proxy method with the liveness
        of the mixin that inherits it.
        """
        files = set(self.caller_log_files)
        if self.owns_logger:
            files.update(self.log_files)
        return sorted(files)

    @staticmethod
    def make_name(map_id: str, subject: str, signature: str) -> str:
        """Build the semantic signature used as the merge key."""
        return f"{map_id}:{subject}:{signature}"

    def content_hash(self) -> str:
        """Hash of the junction's *meaning*, used to key derived artefacts.

        Deliberately excludes the anchor.  A gloss cached against a file
        position would be regenerated every time an unrelated edit shifts the
        line numbers; keyed on content it survives until the logic itself
        changes.
        """
        payload = {
            "subject": self.subject,
            "branches": [
                {
                    "outcome": b.outcome,
                    "path_condition": b.path_condition,
                    "tags": b.tags,
                    "emits": [e.model_dump() for e in b.emits],
                }
                for b in self.branches
            ],
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class BoundaryNode(BaseNode):
    """Where causation crosses into a system this map does not cover.

    An unbound boundary is a complete answer in itself ("the pilot reported
    1099") *and* a concrete work item ("bind this boundary"), the same shape as
    a recorded capability gap.

    ``kind`` splits the two diagnostic modes.  A system that *reports state*
    (the pilot posting an error code) leaves its value in the record, so
    non-arrival is visible.  A system that *transports causation* (a message
    broker) leaves nothing on the consumer side when a message is lost, so the
    question flips from "which condition blocked it" to "was it published at
    all", which the consumer alone cannot answer.
    """

    node_type: NodeType = NodeType.BOUNDARY
    map_id: str
    derived_from: str
    system: str
    kind: str = Field(default="reports_state", description="reports_state | transports_causation")
    transport: str = Field(
        default="http",
        description=(
            "http | shared_table.  A separate axis from ``kind``: that one says "
            "whether evidence survives a failure to cross, this one says where "
            "to go looking.  An endpoint is investigated through the arrival "
            "log; a shared table is investigated by querying it, and the row "
            "either exists or it does not."
        ),
    )
    interface: str = Field(
        ...,
        description="Receiving-side function, or ``schema.table`` for a shared table.",
    )
    carried_values: list[str] = Field(
        default_factory=list,
        description="Names the far side supplies, in the receiving side's spelling.",
    )
    handed_over: list[str] = Field(
        default_factory=list,
        description=(
            "Names this map's code writes for the far side to read.  Empty for "
            "an endpoint, which is inbound only; a shared table is a channel in "
            "both directions, and which direction is broken is the first "
            "question to ask about one."
        ),
    )
    operations: list[str] = Field(
        default_factory=list,
        description=(
            "SQL verbs used against a shared table.  Recorded because the set "
            "itself is a finding: a ``DELETE`` alongside an ``INSERT`` on a "
            "command table means a second command silently replaces one that "
            "was never picked up."
        ),
    )
    access_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Preconditions the boundary itself enforces before any junction "
            "sees the request -- transport security, caller role, HTTP method, "
            "ownership.  These are the *first* place a request can be rejected, "
            "so a command that appears to have vanished may never have been "
            "accepted here."
        ),
    )
    observable_values: list[str] = Field(
        default_factory=list,
        description=(
            "Carried values the receiving code logs on arrival.  What is not "
            "logged cannot be recovered afterwards, so this is the difference "
            "between a boundary that can be investigated and one that can only "
            "be guessed at."
        ),
    )
    accepts_arbitrary: bool = Field(
        default=False,
        description=(
            "The endpoint takes ``**kwargs``, so what crosses cannot be "
            "enumerated from the signature.  Recorded rather than ignored: "
            "listing no carried values for such a boundary would claim nothing "
            "crosses it, which is the opposite of the truth."
        ),
    )
    version_binding: list[str] = Field(
        default_factory=list,
        description=(
            "Fields that pin the far side's version at run time.  Needed "
            "because a system whose version moves independently cannot share "
            "this map's version stamp."
        ),
    )
    resolution: str = Field(default="unresolved", description="unresolved | bound")
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, system: str, interface: str) -> str:
        """Identity is the interface, not one value crossing it.

        An endpoint carries many values at once, so keying per value would
        split one boundary into dozens that all move together.
        """
        return f"{map_id}:{system}:{interface}"


class ValueEnumNode(BaseNode):
    """One ``NAME = value`` constant from a declared enumeration.

    Kept separate from a *declared value set* (``FINAL_TASK_STATUSES = [...]``)
    even though both are module-level constants: an enumeration entry is an
    index entry -- code ``100`` in namespace ``taskbuffer`` means ``EC_Kill`` --
    whereas a value set is a vocabulary oracle used to check extraction.
    Counting them together made an ErrorCode module and a config module look
    like the same kind of file.

    ``name`` is ``"<namespace>.<constant>"`` and the reverse index is keyed on
    ``(namespace, value)``.  Numbers are reused across namespaces --
    ``taskbuffer.EC_Kill``, ``dataservice.EC_Setupper`` and
    ``jobdispatcher.EC_Watcher`` are all ``100`` -- so decoding an observed
    code requires knowing which field it came from.
    """

    node_type: NodeType = NodeType.VALUE_ENUM
    map_id: str
    derived_from: str
    namespace: str
    constant: str
    value: Any
    comment: Optional[str] = None
    anchor: Optional[Anchor] = None
    references: int = Field(
        default=0,
        description=(
            "Number of sites referencing this constant.  Zero means it is dead "
            "or was mis-classified, so it does not belong in a decoding index."
        ),
    )

    @staticmethod
    def make_name(namespace: str, constant: str) -> str:
        return f"{namespace}.{constant}"


class CoverageStat(BaseModel):
    """Per-file extraction coverage for one slice.

    Reported per file rather than as a slice average: the filter-chain idiom
    ranges from 26 occurrences to none across sibling broker files, and a mean
    would bury that variance under the slices whose recognizers depend only on
    Python syntax and so match near-perfectly everywhere.
    """

    slice_name: str
    file: str
    candidates: int = 0
    explained: int = 0

    @property
    def ratio(self) -> float:
        return self.explained / self.candidates if self.candidates else 1.0


class MapFragment(BaseModel):
    """What one recognizer contributes to a build.

    The plugin contract is shaped around this rather than around how the
    extraction is done, so a recognizer implemented with a different tool
    still composes -- fragments merge on the semantic signature.
    """

    map_id: str
    derived_from: str
    subjects: list[SubjectNode] = Field(default_factory=list)
    junctions: list[JunctionNode] = Field(default_factory=list)
    boundaries: list[BoundaryNode] = Field(default_factory=list)
    value_enums: list[ValueEnumNode] = Field(default_factory=list)
    filter_stages: list["FilterStageNode"] = Field(default_factory=list)
    diagnostics: list["DiagnosticTemplate"] = Field(default_factory=list)
    enumeration_writes: list["EnumerationWrite"] = Field(default_factory=list)
    coverage: list[CoverageStat] = Field(default_factory=list)
    declaration_yields: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "``Class (file)`` -> how many names the extraction read from the "
            "column declaration there.  Every declaration is listed, including "
            "the ones that yielded nothing, which is the point: a form nobody "
            "reads looks exactly like a class with nothing to declare once the "
            "names have been merged away.  Carried on the fragment rather than "
            "logged so that ``spec-declarations-are-read`` can fail on it."
        ),
    )
    annotation_readings: list["AnnotationAudit"] = Field(
        default_factory=list,
        description=(
            "One row per container element annotation the map depends on -- what "
            "it states, what the code actually stores there, and whether "
            "removing it would change any write.  A bare ``Dict`` says nothing, "
            "so these element types are facts the map takes on trust; these rows "
            "are what let two gates check them instead."
        ),
    )
    spec_annotation_forms: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "``file:line`` -> the spec class the extraction read from the "
            "annotation there, empty when it read none.  Listed only where the "
            "annotation's text names a class the corpus declares, so an empty "
            "value means the two readings disagree and the form is one the "
            "extraction does not understand.  Same shape as "
            "``declaration_yields`` and for the same reason: counting what came "
            "back is the only way an unreadable form differs from an absent one."
        ),
    )

    def extend(self, other: "MapFragment") -> None:
        """Merge *other* into this fragment in place."""
        self.subjects.extend(other.subjects)
        self.junctions.extend(other.junctions)
        self.boundaries.extend(other.boundaries)
        self.value_enums.extend(other.value_enums)
        self.filter_stages.extend(other.filter_stages)
        self.diagnostics.extend(other.diagnostics)
        self.enumeration_writes.extend(other.enumeration_writes)
        self.coverage.extend(other.coverage)
        self.declaration_yields.update(other.declaration_yields)
        self.annotation_readings.extend(other.annotation_readings)
        self.spec_annotation_forms.update(other.spec_annotation_forms)


class AnnotationAudit(BaseModel):
    """Two readings of one annotation the map has to take on trust, side by side.

    Not a node.  This records how much the map trusts a statement PanDA makes
    about a class it cannot otherwise read, which is a fact about the extraction
    rather than about the system, so it belongs in the report and the gates and
    not in the graph.

    Two shapes qualify, and both are trusted for the same reason -- the class
    they name is stated nowhere else in the expression that reaches the write.
    A container's element type (``Dict[str, DatasetSpec]``) has a second
    reading in what the code stores; a context manager's yield type
    (``Iterator[WorkflowSpec | None]``) has one only when the object's usage
    happens to be distinctive, which for two of PanDA's three workflow locks it
    is not.
    """

    where: str = Field(description="``file:line`` of the annotation")
    kind: str = Field(
        default="container",
        description=(
            "``container`` for an element type, ``yield`` for a context "
            "manager's.  Only the first has a stored second reading, so the "
            "agreement gate is about that one and the ablation gate is about "
            "both."
        ),
    )
    container: str = Field(description="the annotated name, for the report")
    stated: str = Field(description="the spec class the annotation names")
    put_in: list[str] = Field(
        default_factory=list,
        description=(
            "Spec classes the code stores in that container, where they resolve "
            "on their own.  Empty means no independent reading was available, "
            "which is not evidence against the annotation -- it is the case the "
            "annotation was asked for."
        ),
    )
    read: bool = Field(
        description=(
            "Whether removing the annotation would change any write in its "
            "scope, in class or in basis.  False means the map does not depend "
            "on it: either it is redundant or a read form is missing."
        )
    )
    audited: bool = Field(
        default=True,
        description=(
            "Whether the ablation had anything to compare.  False where no "
            "spec attribute is written in the annotation's scope, which makes "
            "``read`` vacuous -- the row is kept because its other reading, "
            "what the code puts in the container, is unaffected by that."
        ),
    )
    unsettled: list[str] = Field(
        default_factory=list,
        description=(
            "``line:attribute`` for each write in scope of an attribute the "
            "stated class declares that still resolves by inference or not at "
            "all.  This is what separates the two readings of ``read=False``: "
            "with nothing unsettled the map simply reaches the class another "
            "way, which an annotated tree makes the normal case."
        ),
    )


class DiagnosticTemplate(BaseModel):
    """One place the code assembles text, and the frame it assembles.

    **Deliberately not a node, and deliberately outside promotion.**  Promotion
    answers "is this field worth asking why about?", which for a free-text field
    is the wrong question -- ``ddmErrorDiag`` has no value set to enumerate, and
    a subject node for it would invite a reader to expect a branch table.  The
    question an investigation actually asks of it is the other way round: *this
    message was seen, who wrote it?*  That is an index from a template to an
    anchor, and an index makes no claim about the field, so it needs none of the
    machinery -- no promotion criterion, no threshold, and no rule for telling a
    message from an identifier.

    That last point is what settled the design.  Both structural readings that
    look like they could separate the two were measured and both misclassify:
    the template reading as prose calls ``panda.pp.in.{}.{}`` a message and
    misses ``errorDialog`` (whose other writes go through ``setErrDiag``), and
    "nothing ever compares this field" also matches ``lfn`` and ``jobName``.
    Indexing all of them costs nothing and misleads nobody.

    Only *assembled* text is indexed.  A bare literal is not a template, and an
    exact message can be found by searching the source for itself.
    """

    map_id: str
    derived_from: str
    template: str = Field(
        ...,
        description=(
            "The literal frame, run-time parts as ``{}`` -- what a message "
            "observed in production is matched against."
        ),
    )
    field: str = Field(
        ...,
        description=(
            "Qualified name of the field the text lands in.  Often not a "
            "subject: that is the point."
        ),
    )
    form: str = Field(..., description="attribute | bind")
    anchor: Anchor


class EnumerationWrite(BaseModel):
    """One place the code puts a named enumeration constant into a field.

    The binding between a field and the enumeration that decodes it, which is
    what a reverse index needs and cannot get from the constants alone.  Numbers
    are reused across enumerations by design -- ``taskbuffer.ErrorCode.EC_Kill``
    and ``jobdispatcher.ErrorCode.EC_Watcher`` are both ``100`` -- so the index
    is keyed on ``(namespace, value)``, and decoding a code seen in a record is
    impossible without knowing which field it came from.

    ``errorcode`` calls that binding "separate and explicit, not recoverable
    from the constant's location", and it is right about the location.  It is
    recoverable from the *write*: ``jobSpec.taskBufferErrorCode =
    ErrorCode.EC_Kill`` states the field and the constant in one statement, and
    the corpus does that 79 times over five fields -- three of which are the
    three ``ErrorCode`` modules, one field each.

    Not restricted to error codes, and deliberately: the rule is "a declared
    field assigned a constant the value-enum slice extracted", which also
    catches ``JediTaskSpec.eventService`` and ``JobSpec.job_label``.  Telling
    an error code from any other enumeration would need a classifier, and an
    index makes no claim that would justify one -- the same reason
    :class:`DiagnosticTemplate` indexes every assembled string.

    **Outside promotion**, also for that class's reason: no criterion fires on
    these fields -- every write is tier 2, the right-hand side being a module
    constant rather than a literal -- so promotion drops them, and rightly.
    "Why is this field 100?" is not the question anyone asks of it; "what does
    100 mean here?" is, and that is an index.
    """

    map_id: str
    derived_from: str
    field: str = Field(
        ..., description="Qualified name of the field written, e.g. JobSpec.taskBufferErrorCode."
    )
    constant: str = Field(..., description="The constant's bare name, e.g. EC_Kill.")
    namespace: str = Field(
        ...,
        description=(
            "The enumeration the constant belongs to, as the value-enum index "
            "keys it.  This is the half that makes an observed value decodable."
        ),
    )
    anchor: Anchor


class FilterStageNode(BaseNode):
    """One reason a candidate was dropped on the way to a selection.

    Brokerage does not pick a site; it narrows a list, twenty-odd times in a
    row, and "the distribution is wrong" means asking *which step* threw the
    candidates away.  That is not a branch table -- every step runs, and each
    removes some -- so it is its own node rather than a junction whose
    branches happen to be cumulative.

    ``criteria_tag`` is the identity, because it is the semantic signature and
    the log line at once: the code emits ``criteria=-diskIO`` per rejected site,
    so a stage keyed on it can be counted directly from production logs and
    survives any refactoring that keeps the tag.  ``funnel_label`` is the
    coarser step the summary counter reports (``diskIO check``), which several
    tags can share -- ``-lowmemory`` and ``-highmemory`` are both
    ``memory check``.  Both appear in logs and they are read differently, so
    both are kept.
    """

    node_type: NodeType = NodeType.FILTER_STAGE
    map_id: str
    derived_from: str
    owner: str = Field(..., description="module::function holding the chain.")
    criteria_tag: str = Field(
        default="",
        description="The tag the code emits, e.g. '-diskIO'.  Empty when the step is unnamed.",
    )
    funnel_label: str = Field(
        default="",
        description="Step name the candidate counter reports, e.g. 'diskIO check'.",
    )
    order: int = Field(default=0, description="Position in the chain, by source order.")
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Guards under which a candidate is dropped.  Not solved: the values "
            "come from observation and are substituted in, which is what turns "
            "'120 sites became 3' into a reason."
        ),
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Identifiers the conditions read -- where the backward walk continues.",
    )
    emits: list[str] = Field(
        default_factory=list, description="Message templates the stage logs when it drops one."
    )
    log_level: Optional[str] = Field(default=None, description="Level of those messages.")
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Log files this stage's messages can land in, e.g. "
            "'panda-AtlasProdJobBroker.log'.  Part of the answer rather than a "
            "fetching detail: the map exists to say which component's log to "
            "read.  Empty when the module declares no logger and is mixed into "
            "nothing, so its output goes to a caller's file that the source "
            "does not name."
        ),
    )
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, owner: str, signature: str) -> str:
        return f"{map_id}:{owner}:{signature}"


# ---------------------------------------------------------------------------
# What one investigation makes of the map
# ---------------------------------------------------------------------------
#
# None of these are nodes.  A strategy is the product of asking the map one
# question about one incident; storing it would put an answer into the
# vocabulary the answers are drawn from, and the next build would either
# overwrite it or leave it behind pointing at junctions that have moved.  They
# carry ``derived_from`` for the same reason every node does -- a strategy is a
# statement about one version of the source, and reading one against another
# deployment is exactly the skew the version stamp exists to make visible.


#: What an observation settled about a candidate.
SEEN = "seen"
ELIMINATED = "eliminated"
UNSETTLED = "unsettled"
UNASKABLE = "unaskable"

#: What production answered for one log file.
ANSWER_SEEN = "seen"
ANSWER_ABSENT = "absent"
ANSWER_NO_FILE = "no_file"
ANSWER_INCONCLUSIVE = "inconclusive"
ANSWER_NOT_ASKED = "not_asked"


class Symptom(BaseModel):
    """What was observed, in the map's own vocabulary.

    Deliberately not free text.  Turning "the task is stuck in pending" into
    grep terms and ranking thirty files is the retrieval problem the map exists
    to remove; naming a subject and a value instead makes the lookup exact, and
    the quality of the answer stops depending on how well the question was
    phrased.
    """

    subject: str = Field(..., description="Qualified subject, e.g. JediTaskSpec.status.")
    observed: str = Field(..., description="The value the record actually holds.")
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "The entity this is about, when there is one.  Without it the "
            "evidence can still say which writers are live, but not which one "
            "wrote *this* row -- a different and much weaker claim, so it is "
            "recorded rather than defaulted."
        ),
    )
    observed_diag: Optional[str] = Field(
        default=None,
        description=(
            "The message the record carries, when it carries one.  The cheapest "
            "evidence in the system and the only kind with no window: it is a "
            "column, so one API call returns it whole, where a log line has to "
            "be found in a rotation under a byte cap.  What it buys is the arm "
            "rather than the junction, because PanDA writes the reason into the "
            "same message it logs."
        ),
    )


class CandidateBranch(BaseModel):
    """One way a candidate reaches the observed value.

    Junctions were the unit until a symptom came along whose whole question is
    *which arm*: ``setScoutJobData_JEDI`` sends a task to ``exhausted`` from six
    arms, one per reason, and pooling their conditions into one list per
    candidate answers "why" with the union of six answers.  Flattening was
    right for ``pending``, where fourteen of eighteen candidates have no
    condition at all -- and it hid the question here entirely.
    """

    outcome: str
    tier: int = 1
    line: Optional[int] = Field(
        default=None, description="Where the write is, so a reader can go and look."
    )
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "What the code calls this decision -- ``action=set_exhausted`` and "
            "``reason=low_efficiency``.  Both a search key in production's logs "
            "and, where the same message is persisted, the part of the record "
            "that names this arm."
        ),
    )
    messages: list[str] = Field(
        default_factory=list,
        description=(
            "The frames the block records, holes left open.  What names the arm "
            "where no tag does, which in production is most of the time."
        ),
    )
    conditions: list[str] = Field(default_factory=list)
    row_precondition: list[str] = Field(default_factory=list)
    matched: bool = Field(
        default=False,
        description=(
            "Whether the message the record carries names this arm.  One "
            "direction only: a match proves this arm decided, and a record that "
            "names none proves nothing, because the field holds the last message "
            "written to it and a later junction may have overwritten it."
        ),
    )


class Candidate(BaseModel):
    """One junction that could have produced the observed value.

    The set is the system's real fan-out, not an artefact of how the map was
    built: an expert asked why a task is ``pending`` faces the same eighteen.
    What the map adds is that the enumeration is complete, precomputed, and
    carries the log file to read for each one.
    """

    owner: str
    tier: int = Field(
        default=1,
        description=(
            "1 when a branch states this value outright, 2 when the writer is "
            "known and the value is only settled at run time.  A tier-2 "
            "candidate is never eliminated by the value, because 'this one "
            "could have' is the honest answer for it."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where a line about this junction firing can appear -- the caller's "
            "files where it has callers, its own only where it declares a "
            "logger.  Empty means no file can be asked about it at all, which "
            "is a finding rather than a reason to guess one."
        ),
    )
    branches: list[CandidateBranch] = Field(
        default_factory=list,
        description="The arms that reach this outcome, each with its own reason.",
    )
    triggers: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    verdict: str = Field(
        default=UNSETTLED,
        description=f"{SEEN} | {ELIMINATED} | {UNSETTLED} | {UNASKABLE}",
    )
    because: str = Field(default="", description="Why the verdict, in one clause.")

    @property
    def conditions(self) -> list[str]:
        """Every condition any of the arms is under, in order, deduplicated.

        The summary the arms compose to.  Derived rather than stored so that the
        two cannot say different things -- which is the failure the arms were
        introduced to fix, one level up.
        """
        seen: list[str] = []
        for branch in self.branches:
            for condition in branch.conditions:
                if condition not in seen:
                    seen.append(condition)
        return seen

    @property
    def row_precondition(self) -> list[str]:
        """What the row had to already say for any of these arms to land.

        Kept apart from the conditions because the two fail differently: an
        unmet path condition means the code never got here and the log is
        silent; an unmet row precondition means the code got here, said so, and
        changed nothing.
        """
        seen: list[str] = []
        for branch in self.branches:
            for guard in branch.row_precondition:
                if guard not in seen:
                    seen.append(guard)
        return seen

    @property
    def named(self) -> list[CandidateBranch]:
        """The arms the record's message names, if it names any."""
        return [branch for branch in self.branches if branch.matched]


class Observation(BaseModel):
    """One question put to production, and what its answer would settle.

    Asked of both machine groups, not of the one the package suggests.  JEDI
    opens its own TaskBuffer, so ``pandaserver`` code called by a knight runs in
    the JEDI process and logs there; the package is wrong precisely where most
    junctions are.  A service that does not have the file answers "No such file
    or directory", which is cheap and is itself distinguishable from an empty
    match -- so the union costs one extra query and removes a guess.
    """

    log_file: str
    pattern: str
    role: str = Field(
        default="probe",
        description=(
            "``probe`` asks whether this row was moved here; ``control`` asks "
            "whether the file carries that kind of line at all.  Without the "
            "second, a file that never speaks the sentence is indistinguishable "
            "from one whose writer did not fire, and the map's largest group of "
            "junctions is reached through files of exactly that kind."
        ),
    )
    services: list[str] = Field(default_factory=list)
    settles: list[str] = Field(
        default_factory=list, description="Owners of the candidates this can settle."
    )
    control_for: Optional[str] = Field(
        default=None,
        description=(
            "For a control, the probe pattern it controls.  Paired by pattern "
            "rather than by file because one file can carry two probes that are "
            "different sentences -- the value line every writer shares, and the "
            "tagged line one branch names itself with -- and a control answers "
            "for exactly one of them."
        ),
    )
    verdict: str = Field(
        default=ANSWER_NOT_ASKED,
        description=(
            f"{ANSWER_SEEN} | {ANSWER_ABSENT} | {ANSWER_NO_FILE} | "
            f"{ANSWER_INCONCLUSIVE} | {ANSWER_NOT_ASKED}.  Four ways of not "
            "matching and only two of them are answers, which is the whole "
            "reason this is not a count."
        ),
    )
    matched: int = 0
    sample: list[str] = Field(
        default_factory=list, description="A few matched lines, for the report."
    )


class FollowUp(BaseModel):
    """Whether anything will move the value on, and what to ask if not.

    The other half of the question.  Which junction wrote ``pending`` says how
    the row got where it is; this says whether it is going to leave, and a task
    can be stuck for a reason that has nothing to do with the writer.
    """

    selected: bool = Field(
        ...,
        description="Whether any query in the map selects rows on this value.",
    )
    selection_gates: list[str] = Field(
        default_factory=list,
        description=(
            "Tables bounding what those queries can see, none of which anything "
            "in the map writes.  The reason a row can fail to be picked up while "
            "passing every condition on its own status."
        ),
    )
    selected_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose query selects rows on the observed value.  "
            "Where the row has to be picked up, so where to ask why it was not."
        ),
    )
    reader_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where those readers' diagnostics land, for the ones the map holds "
            "a log for -- which is the readers that also settle something, "
            "since a function that only selects is not a junction and has no "
            "node to hang a file on.  Empty is a finding, not a licence to "
            "guess a file."
        ),
    )
    triggers: list[str] = Field(
        default_factory=list,
        description=(
            "Triggers of the query that selects the observed value, where the "
            "map names a reader that is also a junction; otherwise pooled over "
            "the subject's writers, which is the older approximation and stays "
            "as the fallback rather than leaving the field empty."
        ),
    )
    self_repairing: bool = False
    carried_from: list[str] = Field(
        default_factory=list,
        description=(
            "Fields this subject's value is copied from.  Where to take the "
            "question when nothing selects the observed value: the row is not "
            "going anywhere, so the useful question is who wrote the step before."
        ),
    )
    question: str = Field(default="", description="What to ask next, in one sentence.")


class Strategy(BaseModel):
    """What the map has to say about one symptom.

    Two phases in one object: everything above ``observations`` comes from the
    map alone and needs no network, and the verdicts are filled in afterwards
    from an evidence file.  Separating them is what lets the derivation be
    tested against a fixture, re-evaluated offline, and inspected before a
    single query is put to production -- the same split ``check-map`` makes, and
    for the same reason: first contact turns up the errors in the model.
    """

    symptom: Symptom
    map_id: str
    derived_from: str
    candidates: list[Candidate] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    follow_up: Optional[FollowUp] = None
    findings: list[str] = Field(
        default_factory=list,
        description="Facts about the map itself that this question turned up.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "What the map would have to record for this to go further.  A "
            "capability gap is an answer of its own -- it names the next thing "
            "to build instead of being absorbed as a weaker conclusion."
        ),
    )
