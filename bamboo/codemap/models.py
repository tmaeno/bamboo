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
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

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
    line_end: Optional[int] = Field(
        default=None,
        description=(
            "End of the span, and today nothing in production reads it -- "
            "``reading.region_in`` derives the region from the containing "
            "function instead, and ``diff.py`` leaves the whole anchor out of "
            "its comparison on purpose.  So a wrong one is silent, which is "
            "how seventeen junctions came to carry a span that ends above its "
            "own start.  An analyser reading the span is the natural first use "
            "of it, and that is when a wrong one would be believed."
        ),
    )
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
            "Values something acts on rows by -- a query's ``WHERE`` or an "
            "update's.  The counterpart of the outcomes the junctions write: "
            "together they make a state machine out of a pile of writes, since "
            "a value nothing acts on is one nothing ever moves away from.  Both "
            "verbs count here, because an ``UPDATE ... WHERE status=:old`` "
            "moves a task out of that status as squarely as a query does."
        ),
    )
    selected_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "``{value: the functions whose *query* selects rows on it}``.  Kept "
            "per value rather than per subject because the two answer different "
            "questions: ``JediTaskSpec.status`` is selected on thirty-one values "
            "by dozens of functions, and pooling them says only that the subject "
            "is read.  Per value it says who to ask -- ``finishing`` is selected "
            "by exactly one query in the whole corpus.  Queries only; see "
            "``updated_by`` for the other verb and why the two are apart."
        ),
    )
    updated_by: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "``{value: the functions whose UPDATE or DELETE acts on rows "
            "already holding it}``.  A different claim from ``selected_by``, "
            "and folding them said an update was a query for a fifth of the "
            "corpus.  The distinction is what the reading is for: asked why a "
            "row was not picked up, a query is somewhere that could have "
            "missed it, while an update is the picking up itself -- so the "
            "first names a place to look and the second names what already "
            "happened."
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


class EntityNode(BaseNode):
    """A kind of row, and which functions read it and which write it.

    The map's vocabulary was pairs -- ``(spec class, attribute)`` -- and a pair
    can only say something about a *column*.  ``getPandaIDsWithTask_JEDI``
    selects a task's jobs on nothing but the join key, so it named no promoted
    subject and the map read that as observing nothing.  It observes jobs.  The
    fact had nowhere to go: ``class_for_table`` was already computed beside the
    predicate loop and thrown away when no predicate landed on a promoted
    attribute.

    **Not folded into** :class:`SubjectNode`.  That model is load-bearing for
    promotion and for three gates -- ``declared-status-is-written``, the
    written-but-never-selected report and ``structural-attribution-agrees`` --
    all of which read ``attribute``.  Rows with no attribute would flow into
    every one of them, which is the "one field answering two questions" shape
    this map has already paid for twice (``log_files`` and ``opened_by``).

    **Stored, not merely reported.**  ``DiagnosticTemplate`` and
    ``EnumerationWrite`` are indexes kept on the fragment with no label, and
    nothing in an investigation can reach them -- the database holds none of
    their rows.  An entity is read by ``derive-strategy``, so it is a node.

    **All four verbs stay apart.**  Selecting a task's jobs and updating them
    are different claims, and pooling them is exactly the conflation that let a
    function's ``UPDATE ... WHERE`` be reported as a query that selects on a
    value.  The same is true one level down: an ``INSERT`` is an object
    entering the system, an ``UPDATE`` is one already here moving, and a
    ``DELETE`` is one leaving.  Reported as one "written" they could not be
    asked apart, so the map had no way to answer where a kind of row comes into
    existence -- while telling a reader to go and ask whether a command
    arrived, which is exactly a row appearing in a table.

    **Keyed by the table, not by the spec class.**  Most of this corpus's
    tables have no spec: forty-one of the forty-eight it inserts into, and they
    are the ones where arrival means something -- ``HARVESTER_COMMANDS``,
    ``PRODSYS_COMM``, ``SQL_QUEUE``, ``async_requests``, ``Job_Output_Report``,
    ``users``, ``SiteData``, ``T_TASK``, ``jobs_StatusLog``.  Keying on the
    class answered for seven.

    The class would be the wrong key even where there is one.  ``SubjectNode``
    keys on it because ``jobsActive4``, ``jobsDefined4`` and ``jobsArchived4``
    are one ``JobSpec.jobStatus``, and splitting that would make three subjects
    out of one fact.  A row's *creation* is the opposite: which of those tables
    it was created in is the fact, and folding them loses it.  So the class
    rides along as an attribute instead of being the signature.

    The name is the table folded to lower case, because SQL identifiers are
    case-insensitive and this corpus spells one table several ways.
    """

    node_type: NodeType = NodeType.ENTITY
    map_id: str
    derived_from: str
    spec_class: Optional[str] = Field(
        default=None,
        description=(
            "The spec class this table holds, where one was learned.  An "
            "attribute rather than the key -- see the class docstring -- and "
            "``None`` for the majority of tables, which declare none."
        ),
    )
    tables: list[str] = Field(
        default_factory=list,
        description=(
            "The spellings the corpus uses for this table.  More than one is "
            "ordinary: ``reassignShare`` loops over ``jobsactive4`` where "
            "everything else writes ``jobsActive4``."
        ),
    )
    read_by: list[str] = Field(
        default_factory=list,
        description="``module::method`` of every function whose SELECT names this table.",
    )
    created_by: list[str] = Field(
        default_factory=list,
        description=(
            "The same for INSERT: where a row of this kind comes into "
            "existence.  The question a value's branch table cannot answer, "
            "and the one a stalled command needs -- a command arriving is a "
            "row appearing here."
        ),
    )
    updated_by: list[str] = Field(
        default_factory=list,
        description="The same for UPDATE: where a row already here is moved.",
    )
    deleted_by: list[str] = Field(
        default_factory=list,
        description=(
            "The same for DELETE.  Worth its own list because of what it means "
            "on a command table: a DELETE followed by an INSERT is a second "
            "command silently replacing one that was never picked up."
        ),
    )


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


#: ``passthrough(JediTaskSpec.oldStatus)`` -> ``JediTaskSpec.oldStatus``.  Kept
#: beside the field whose spelling it reads, so the grammar of an outcome has
#: one definition: the readers of it are in two modules and a second copy of the
#: pattern is a second thing free to drift from what the extractor writes.
PASSTHROUGH_OUTCOME = re.compile(r"^passthrough\((?P<field>[^)]+)\)$")


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
            "the walk renders them into the skeleton, one row per line the "
            "code prints, and while they were empty the report fell back to "
            "one hard-coded shape covering one subject."
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


#: A run-time outcome rendered as an f-string -- ``runtime(f'merge_{s}')``.
#: The literal chunks around the holes are the *frame*, and a frame is a
#: constraint on the value, not decoration.
_FRAMED_OUTCOME = re.compile(r"^runtime\(f(['\"])(.*)\1\)$", re.DOTALL)

#: One substitution hole inside a frame.  Nested braces are not matched on
#: purpose: an outcome carrying them is left alone rather than split wrongly.
_OUTCOME_HOLE = re.compile(r"\{[^{}]*\}")


def outcome_excludes(outcome: Optional[str], value: str) -> bool:
    """Whether a branch's own text proves it cannot have produced *value*.

    Tier 2 says the writer is known and the value is settled at run time, and
    everywhere else that is read as "this one could have" -- the honest answer
    for ``runtime(newStatus)``, where nothing bounds what the name holds.  It
    is not the honest answer for ``runtime(f'merge_{s}')``: whatever ``s``
    turns out to be, the result begins with ``merge_``, so a row reading
    ``es_inaction`` was written somewhere else.  Two of the four candidates the
    map offered for that value were of exactly this kind, and a reader
    following them opens a hundred and sixty lines that cannot produce it.

    **Syntactic proof only.**  The frame has to be spelled out in the branch's
    own text, which is why nothing here consults another node: an elimination
    resting on some *other* fact being complete is the shape that turns an
    incomplete candidate set into a confident wrong one.  Copying a value from
    elsewhere -- ``passthrough(JobSpec.jobStatus)`` -- looks like it bounds the
    value too, and does not: it would need that subject's value set to be
    closed, and two of this corpus's hundred and eight subjects have one.

    ``{{`` and ``}}`` are escaped braces rather than holes, so an outcome
    carrying either is left alone instead of being read with the wrong frame.
    """
    match = _FRAMED_OUTCOME.match(outcome or "")
    if not match:
        return False
    frame = match.group(2)
    if "{{" in frame or "}}" in frame:
        return False
    literals = _OUTCOME_HOLE.split(frame)
    if not any(literals):
        return False
    if literals[0] and not value.startswith(literals[0]):
        return True
    if literals[-1] and not value.endswith(literals[-1]):
        return True
    return any(chunk and chunk not in value for chunk in literals[1:-1])

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


#: How control got in.  Three shapes, found three ways: a call is resolved by
#: method name, a dispatch is a constructed worker being started, and a door is
#: a facade method forwarding to a borrowed proxy.  Their ``arg_binding`` keys
#: come from different namespaces -- and a door's are not read at all -- so the
#: set is what makes each comparable only with its own kind.
ARRIVES_BY_CALL = "call"
ARRIVES_BY_DISPATCH = "dispatch"

#: Through a facade.  ``api/v1/job_api`` calls ``TaskBuffer.storeJobs``, which
#: calls ``proxy.insertNewJob`` -- the edge is real and the arguments are not
#: this entry's, since the door rewrites them.  Kept apart from ``call`` so
#: that "nothing was read here" is never compared against "nothing was passed".
ARRIVES_THROUGH_DOOR = "door"


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
        description=(
            "What this entry hands over, keyed by the name the callee's body "
            "reads it under -- the keyword for a call, the field for a "
            "constructed worker.  Comparable only within one ``reached_by``: "
            "the two are different namespaces and an empty set means "
            "'positional, not bound' on one side and 'handed nothing' on the "
            "other."
        ),
    )
    reached_by: str = Field(
        default=ARRIVES_BY_CALL,
        description=(
            "``call``, ``dispatch`` or ``door``.  A call is resolved by "
            "method name; a dispatch is a knight constructing a worker and "
            "starting it, which no name match reaches; a door is a facade "
            "method forwarding to a borrowed proxy, whose arguments are the "
            "door's rather than this entry's and are left unread.  Recorded "
            "because the three bind their arguments differently, and comparing "
            "one against another reports a difference in spelling -- or an "
            "unread list -- as a difference in what the path can do."
        ),
    )


class DispatchFanout(BaseModel):
    """The classes a run-time dispatch could have chosen, and how to tell.

    Not an attribution and never one.  P1-11 refused to say *which* spec class
    a run-time receiver is, because a single wrong answer invents a junction;
    this is the other shape -- "one of these three" -- which the map already
    trades in everywhere else.  The set is a candidate set and the run is
    settled by the log line beside it.

    Read from what the code declares rather than from a naming convention.
    The dispatch asks a config for a class and falls back to a concrete one it
    imports; that concrete class names the interface through its base, and the
    base's subclasses in the corpus are the alternatives.  Measured: the
    structural shape alone -- assign a call, then ``if x is None: x = y`` --
    matches 29 sites and most of them are ordinary defaults
    (``maxHS06sec``, ``coreCount``, ``newScanList``).  Requiring the fallback
    to be a *class the corpus declares* leaves 3, which are the three
    ``getPlugin`` sites.
    """

    at: Optional[int] = Field(default=None, description="Line of the dispatch.")
    selector: str = Field(
        default="", description="The call that asks for a class, as written."
    )
    default: str = Field(
        default="", description="The concrete class the source falls back to."
    )
    base: str = Field(
        default="",
        description=(
            "Its base, which is what names the interface.  Empty when the "
            "default declares none -- the fan-out is then that one class, "
            "which is a true answer and a narrow one."
        ),
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="The base's subclasses in this corpus, the default included.",
    )
    announced_by: str = Field(
        default="",
        description=(
            "The literal text of the line that prints the chosen class, when "
            "the code prints one.  This is what turns a candidate set into an "
            "answer at use time, and it is the third leg the plan asked for: "
            "candidates at build, run-time attribute at use, log for proof."
        ),
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
    calls: list[str] = Field(
        default_factory=list,
        description=(
            "What the owner consults, as ``module::method`` targets.  The one "
            "edge out of a junction that reaches the rows an arm decided on: "
            "``setScoutJobData_JEDI`` writes ``exhausted`` and "
            "``getScoutJobData_JEDI`` selects on finished jobs, with nothing "
            "but the call between them.  Qualified because a bare name is not "
            "an identity here and the reader joins on this -- which module a "
            "call lands in is settled at build time, where a name is followed "
            "only if it means one thing or the caller imports the module it "
            "names.  Recorded rather than joined at read time through the "
            "shared owner, because ``selected_by`` is per function and a "
            "method handling several commands reads one thing in one arm and "
            "writes another in another; treating that as a relation is the "
            "mistake this corpus has already charged for twice."
        ),
    )
    joined_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Kinds of row one of this owner's queries reads *in the same "
            "statement* as the rows it decides about.  The sound form of the "
            "join that is unsound at owner granularity: sharing a function "
            "proves nothing, but a single ``FROM`` list is the corpus stating "
            "the relation itself, so ``prepareTasksToBeFinished_JEDI`` "
            "selecting tasks against their datasets really does say which "
            "datasets that task waits on.  The difference is the whole point "
            "-- 113 junctions would gain an entity from co-residence in the "
            "function and 31 do from a shared statement."
        ),
    )
    dispatch: list[DispatchFanout] = Field(
        default_factory=list,
        description=(
            "Run-time class choices made inside this junction's class.  A "
            "property of the class rather than of the arm, the same scope "
            "``self.<field>`` has: ``AdderGen`` picks its plugin in "
            "``get_plugin_class`` and runs it from ``process_job_report``, "
            "which is where the arm is."
        ),
    )
    gloss_key: str = Field(
        default="",
        description=(
            "Hash of the enclosing function's text -- the key of the reading "
            "that explains it.  Computed at build time so the input a reader "
            "would be given is chosen by the map and not by whoever asks; the "
            "reading itself is filled in lazily.  Empty when the function "
            "could not be located in the snapshot."
        ),
    )

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


class LogSiteNode(BaseNode):
    """Where one function's diagnostics land, for a function that decides nothing.

    The map already answers this for a junction, and a junction is not who is
    asked.  Two thirds of the owners the map names as reading a value own no
    junction at all -- ``getPandaIDsWithTask_JEDI`` selects a task's jobs and
    settles nothing -- so ``_follow_up`` looks the reader up among the writers,
    misses, and says nothing about which log will show the query running.
    Measured on the installed corpus: 279 reader and writer owners, 94 of them
    junctions, and all 185 of the rest resolve to a file.

    **Only for owners that are not junctions.**  One fact in two places is how
    ``log_files`` came to answer both "where does this code live" and "whose
    log mentions it"; a reader checks junctions first and these second, so
    nothing is stored twice.

    **Stored, not merely reported.**  ``DiagnosticTemplate`` and
    ``EnumerationWrite`` are built on every run, carry no label and reach
    nobody -- the database holds none of their rows.  An investigation reads
    this to decide where to grep, so it is a node.

    Scoped to the owners the map names.  Every function in a module with a
    resolvable logger could have a row, but a row for a function nothing in
    the map points at is vocabulary added before a reader asked for it.
    """

    node_type: NodeType = NodeType.LOG_SITE
    map_id: str
    derived_from: str
    owner: str = Field(
        ...,
        description=(
            "``module::function``, the same spelling a junction's ``owner`` "
            "and a subject's ``selected_by`` use, because that is what the "
            "reader joins on.  Also the node's ``name``: here the owner is the "
            "identity, there being nothing else to be."
        ),
    )
    triggers: list[str] = Field(
        default_factory=list,
        description=(
            "How this reader's module is started, from the same classification "
            "a junction's entry points carry.  Without it ``_follow_up`` falls "
            "back to the *writers'* triggers to answer whether anything will "
            "re-evaluate the row -- and for a value whose only re-evaluating "
            "reader is a daemon script that settles nothing, that turns "
            "``polled`` into ``command, request`` and sends the reader to ask "
            "whether a command arrived.  A cadence, not an entry point: a log "
            "site has no arm for an argument to be bound at."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files this module's own output can reach, declared first and "
            "otherwise inherited through the classes that mix it in.  A list "
            "because the answer is genuinely two for a proxy method: the "
            "server's log when the server calls it and JEDI's when a knight "
            "does."
        ),
    )
    caller_log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files belonging to the modules that reach this function.  For a "
            "proxy method this is where a line about the call actually appears "
            "-- the mixin declares no logger, and the knight that called it "
            "does."
        ),
    )
    owns_logger: bool = Field(
        default=False,
        description=(
            "Whether ``log_files`` is declared by this module or inherited.  A "
            "reader that cannot tell the two apart has no way to know that "
            "asking the proxy files for a caller's line always returns "
            "nothing, and would read that nothing as 'it never ran'."
        ),
    )

    def observable_log_files(self) -> list[str]:
        """Files where a line about this function running can appear.

        **Its own file counts here, and on a junction it does not.**  The two
        rules differ because the two nodes are asked about different lines.  A
        junction is asked where the line saying it *settled a value* appears,
        and for a proxy method that line is the knight's -- production settles
        it, ``set task_status=`` is in five files and neither proxy file is one
        of them.  A reader is asked where the line saying its *query ran*
        appears, and that one it writes itself: of the 53 owners here whose
        only file is inherited, 51 emit into it, up to ten calls apiece
        (``getScoutJobData_JEDI`` six, ``toEnableJumbo_JEDI`` eight).  Applying
        the junction's rule said "nothing about this can be seen anywhere"
        about code that logs on every line of its body.

        Both inherited candidates are named.  Which of the two a line landed in
        depends on which process ran the code, which is a run-time fact, and
        naming both is the true answer rather than an ambiguity to resolve.
        """
        return sorted(set(self.log_files) | set(self.caller_log_files))


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
    entities: list[EntityNode] = Field(default_factory=list)
    junctions: list[JunctionNode] = Field(default_factory=list)
    boundaries: list[BoundaryNode] = Field(default_factory=list)
    value_enums: list[ValueEnumNode] = Field(default_factory=list)
    filter_stages: list["FilterStageNode"] = Field(default_factory=list)
    loop_cuts: list["LoopCutNode"] = Field(default_factory=list)
    log_sites: list[LogSiteNode] = Field(default_factory=list)
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
        self.entities.extend(other.entities)
        self.junctions.extend(other.junctions)
        self.boundaries.extend(other.boundaries)
        self.value_enums.extend(other.value_enums)
        self.filter_stages.extend(other.filter_stages)
        self.loop_cuts.extend(other.loop_cuts)
        self.log_sites.extend(other.log_sites)
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
    gloss_key: str = Field(
        default="",
        description=(
            "Hash of the enclosing function's text -- the key of the reading "
            "that explains it.  Computed at build time so the input a reader "
            "would be given is chosen by the map and not by whoever asks; the "
            "reading itself is filled in lazily.  Empty when the function "
            "could not be located in the snapshot."
        ),
    )

    @staticmethod
    def make_name(map_id: str, owner: str, signature: str) -> str:
        return f"{map_id}:{owner}:{signature}"


class LoopCutNode(BaseNode):
    """A guard inside a loop that drops the candidate and says why in prose.

    The same cut as :class:`FilterStageNode` and a different kind of node,
    because the two are evidenced differently and every consumer of a stage
    reads the difference.  Brokerage announces its cuts twice in machine-
    readable form -- ``criteria=-diskIO`` per rejected site and
    ``N candidates passed disk check`` per step -- and the whole of ``localize``
    is built on those two.  Everywhere else in the corpus a loop drops its
    candidate with ``continue`` under a guard and a sentence::

        for workQueue in workQueueList:
            for resource_type in resource_types:
                ...
                if not flagLocked:
                    tmpLog_inner.debug("skip since locked by another process")
                    continue

    No tag, no counter, and the sentence is the only name the step has.  Poured
    into ``FilterStageNode`` these arrive with ``criteria_tag`` and
    ``funnel_label`` both empty, which is two fields answering a question they
    were not asked: ``vocabulary`` would offer 75 new chains described in
    brokerage words, ``_leading``'s fallback would open one of them, and
    ``check-map`` would silently add 36 files to what it asks production --
    a footprint decision that was deliberately made once and by hand.

    ``message`` is the identity, for the reason ``criteria_tag`` is one over
    there: it is the semantic signature and the log line at once, so a cut keyed
    on it can be counted from production and survives any edit that keeps the
    wording.  Position is not identity -- the loop moves when anything above it
    does.

    Measured over ``panda-server-source 1.0.2``: 797 of 834 ``continue``
    statements sit inside a loop under a guard, 200 have a logged line in the
    same block, five of those already carry a ``criteria=`` tag and 37 more are
    inside the five brokerage functions the stage recognizer already reads.
    The remaining 153 are this node, over 75 functions in 36 files.
    """

    node_type: NodeType = NodeType.LOOP_CUT
    map_id: str
    derived_from: str
    owner: str = Field(..., description="module::function holding the loop.")
    message: str = Field(
        ...,
        description=(
            "The line logged before the candidate is dropped, with "
            "interpolations rendered as '{}'.  The step's only name."
        ),
    )
    search_key: str = Field(
        default="",
        description=(
            "The longest fixed run of the message -- what to grep production "
            "for.  Not guaranteed to select only this cut: 'skip {}' leaves "
            "'skip'.  ``build-map`` reports the keys that also match another "
            "cut in the same file, which is the threshold-free form of that "
            "worry, and a capped answer says the rest."
        ),
    )
    scope_prefix: str = Field(
        default="",
        description=(
            "The prefix the logger this cut writes through puts on every line, "
            "when the enclosing function builds one -- 'vo={} queue={} cloud={} "
            "pid={} {}'.  Which candidate a line is about is in the prefix, not "
            "the sentence, so this is the key that narrows the question to one "
            "queue or one task.  Empty when the module logs through a bare "
            "logger."
        ),
    )
    order: int = Field(default=0, description="Position within the owner, by source order.")
    loop_line: int = Field(
        default=0, description="Line of the loop header whose iteration this drops."
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Guards under which the candidate is dropped.",
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="Identifiers the conditions read -- where the backward walk continues.",
    )
    log_level: Optional[str] = Field(default=None, description="Level the message is logged at.")
    log_files: list[str] = Field(
        default_factory=list,
        description="Files the message can land in.  Empty when the module declares no logger.",
    )
    anchor: Optional[Anchor] = None

    @staticmethod
    def make_name(map_id: str, owner: str, signature: str) -> str:
        return f"{map_id}:{owner}:{signature}"


class GlossNode(BaseNode):
    """A reading of one function, and the exact text that reading was given.

    The unit is a function because that is what a reader needs, measured
    rather than assumed.  An arm's own guard is a median of eight lines, but
    only 35 of 927 arms refer to nothing outside it: 96% load names they do not
    bind, a median of four each, and 52% sit in a loop whose header is outside
    their guard.  Following those references -- the backward slice the map
    already names as its method -- reaches a median of 37% of the enclosing
    function and, at the ninetieth percentile, 96% of it.  Slicing would build
    a mechanism to arrive at very nearly the function.

    Sharing is why it is a node at all rather than a field on each junction:
    497 junctions sit in 213 functions, 376 of them share one with another
    junction, and the 109 filter stages sit in five.  Read once per function
    and every arm in it is explained together -- 213 readings rather than 1046.

    ``shown`` is the text that was actually handed over, kept for the same
    reason grep output is written to an evidence file: when a reading is wrong,
    it separates *the input was insufficient* from *the reader misread it*, and
    those want opposite fixes.  It is evidence, never the record -- the source
    tree stays canonical.  Storing the mapped functions instead would not even
    save space: the whole corpus is 5.6 MB of Python, the owner functions alone
    are 1.65 MB of it, and adding the callers a reader might ask for next takes
    that to 63% and then 90%.  The closure does not converge on a subset.  It
    would also cost the oracle, since ``check-map`` rebuilds the fragment from
    source on every run and the offline gates work by comparing two statements
    the *code* makes.

    Identity is ``gloss_key``, the hash of the text.  A reading is only about
    the text it was given, so a changed function is a different node rather
    than a stale field -- which is also why this does not share
    :meth:`JunctionNode.content_hash`: that one hashes a junction's *meaning*
    and deliberately ignores the surrounding source, which is the right key for
    an artefact derived from the branch table and the wrong one for an artefact
    derived from the file.
    """

    node_type: NodeType = NodeType.GLOSS
    map_id: str
    derived_from: str
    gloss_key: str = Field(..., description="Hash of `shown`; the node's identity.")
    owner: str = Field(..., description="module::qualname of the function read.")
    anchor: Optional[Anchor] = None
    shown: str = Field(
        default="",
        description="The exact text the reader was given.  Evidence, not the record.",
    )
    explanation: str = Field(
        default="",
        description=(
            "What the reader made of it.  Empty until something reads it: the "
            "map is built eagerly and this is filled in lazily, when an "
            "investigation actually touches this function."
        ),
    )

    @staticmethod
    def make_name(map_id: str, gloss_key: str) -> str:
        return f"{map_id}:gloss:{gloss_key}"


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


#: What kind of question a symptom is, which decides which derivation answers it.
#: Not a flag anyone types -- a symptom is resolved out of the map's own
#: vocabulary, and the vocabulary entry carries the kind.  See :class:`MapTerm`.
SYMPTOM_VALUE = "value"
SYMPTOM_DISTRIBUTION = "distribution"


class Symptom(BaseModel):
    """What was observed, in the map's own vocabulary.

    Deliberately not free text.  Turning "the task is stuck in pending" into
    grep terms and ranking thirty files is the retrieval problem the map exists
    to remove; naming a subject and a value instead makes the lookup exact, and
    the quality of the answer stops depending on how well the question was
    phrased.

    **Two kinds, because the system settles values two different ways.**  A
    value is settled by one branch of one junction, so the question is *which
    arm*.  A site list is not settled at all -- brokerage starts with every
    candidate and narrows about twenty-five times, every step runs, and the
    question is *which step threw them away*.  One shape cannot carry both
    without one of them lying, and the kind is what the derivation dispatches
    on.  It is deliberately not a command-line flag: adding an option per
    symptom would put the retrieval problem back in the caller's hands, one
    layer up.
    """

    kind: str = Field(
        default=SYMPTOM_VALUE,
        description=f"{SYMPTOM_VALUE} | {SYMPTOM_DISTRIBUTION}.",
    )
    subject: str = Field(default="", description="Qualified subject, e.g. JediTaskSpec.status.")
    observed: str = Field(default="", description="The value the record actually holds.")
    focus: str = Field(
        default="",
        description=(
            "For a distribution, what the description landed on -- a chain, a "
            "``criteria=-`` tag or a funnel step.  It decides what the answer "
            "*leads* with, never what it contains: the whole chain is read "
            "either way, because a description naming one cut is a guess about "
            "which cut mattered and the evidence is what settles that."
        ),
    )
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

    @model_validator(mode="after")
    def _the_kind_has_what_it_needs(self) -> "Symptom":
        """Refuse a symptom its own kind cannot be asked.

        The fields are optional per-field because the two kinds use different
        ones, and leaving it there would let a distribution with no focus reach
        the derivation and come back empty -- an answer shaped like "nothing is
        wrong".  Checked here so the failure is at the point the question is
        formed, where the caller still knows what it meant.
        """
        if self.kind == SYMPTOM_VALUE and not (self.subject and self.observed):
            raise ValueError("a value symptom needs both a subject and an observed value")
        if self.kind == SYMPTOM_DISTRIBUTION and not self.focus:
            raise ValueError("a distribution symptom needs a focus -- a chain, a tag or a step")
        if self.kind not in (SYMPTOM_VALUE, SYMPTOM_DISTRIBUTION):
            raise ValueError(f"{self.kind!r} is not a symptom kind")
        return self


#: What part of the map a vocabulary entry names.  Distinct from
#: ``Symptom.kind``, which says which derivation answers it: several term kinds
#: resolve to one derivation, because a chain, one of its cuts and one of its
#: steps are three ways of pointing at the same brokerage question.
TERM_VALUE = "value"
TERM_CUT = "cut"
TERM_STEP = "step"
TERM_CHAIN = "chain"


class MapTerm(BaseModel):
    """One thing the map can be asked about, and the symptom it resolves to.

    The entry that makes free text tractable.  Asking a source navigator means
    turning a description into grep terms and ranking thirty files; asking the
    map means picking from a set that is *closed and enumerable* -- measured at
    420 entries for PanDA: 296 ``(subject, value)`` pairs, 70 cut tags, 49
    funnel steps and 5 chains.  That number is what "the retrieval problem is
    removed" means quantitatively, and it is small enough to match against
    deterministically and small enough to hand an LLM whole.

    Each entry carries the :class:`Symptom` it resolves to, so resolution has
    nothing to decide beyond which entry the words meant.  Building the symptom
    at the far end instead would put the map's vocabulary into the resolver,
    where it would be a second copy free to drift.
    """

    kind: str = Field(..., description=f"{TERM_VALUE} | {TERM_CUT} | {TERM_STEP} | {TERM_CHAIN}")
    key: str = Field(..., description="How the map spells it, e.g. 'JediTaskSpec.status=exhausted'.")
    words: list[str] = Field(
        default_factory=list,
        description=(
            "The text a description is matched against -- the key split into "
            "words, plus whatever else names the same thing.  A cut carries its "
            "step's words too, because production spells it ``-lowmemory`` and "
            "people say 'memory'."
        ),
    )
    symptom: Symptom


class Match(BaseModel):
    """One vocabulary entry a description could have meant, and how well.

    Ranked rather than chosen.  A description that names two things is a fact
    about the description, and picking one of them silently is the failure mode
    the map was built to remove -- ``too_many_candidates`` answered as though it
    were one candidate.
    """

    term: MapTerm
    score: float = Field(
        default=0.0,
        description=(
            "Share of the entry's own weight the description accounted for. "
            "Words are weighted by how rare they are across the vocabulary, "
            "computed from the vocabulary itself rather than tuned: 'check' "
            "appears in most steps and settles nothing, 'lowmemory' appears in "
            "one and settles it."
        ),
    )
    words: list[str] = Field(
        default_factory=list, description="Which of the entry's words the description said."
    )
    exact: bool = Field(
        default=False,
        description=(
            "The description was this entry's key, spelled the same way.  Kept "
            "as a fact rather than folded into the score: the word weighting "
            "cannot express it -- an entry that carries its step's words too "
            "has a larger denominator and loses to a near-homograph that "
            "carries fewer, which is how ``-t1_weight`` came third to "
            "``-t1weight`` when ``-t1_weight`` was what was typed."
        ),
    )
    tied_with: list[str] = Field(
        default_factory=list,
        description=(
            "Entries scoring exactly the same.  A tie broken by sorting is a "
            "tie decided by how the two keys happen to be spelled -- '-' sorts "
            "below 'T', and nothing about the question said so.  Reported so "
            "the reader breaks it, since the resolver has nothing to break it "
            "with."
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


class Handover(BaseModel):
    """What one entry passed in on its way to this candidate.

    Carried on the candidate because the arm's own body is where a trace
    starts and the body reads ``self.taskList``, not ``taskList``.  Naming the
    function that filled it, and with what, is the one step forward from the
    anchor that cannot be taken by reading the arm's own module: the knight
    hands its rows to a worker through a constructor and a thread, so there is
    no call to follow.
    """

    entry: str = Field(..., description="Module the control came from.")
    via: str = Field(..., description="Function there that made the handover.")
    reached_by: str = Field(
        ...,
        description=(
            "``call`` or ``dispatch`` -- which tells a reader what the keys "
            "of ``fields`` are, a parameter or a worker's own attribute."
        ),
    )
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Name the arm's body reads it under -> the expression supplied.",
    )


class Candidate(BaseModel):
    """One junction that could have produced the observed value.

    The set is the system's real fan-out, not an artefact of how the map was
    built: an expert asked why a task is ``pending`` faces the same eighteen.
    What the map adds is that the enumeration is complete, precomputed, and
    carries the log file to read for each one.
    """

    owner: str
    file: str = Field(default="", description="Where the junction is, for a reader.")
    blob_sha: str = Field(
        default="",
        description=(
            "Hash of the file the map read.  Carried so that reading the code "
            "against a different snapshot is noticed: line drift is one of the "
            "two skew symptoms no gate catches."
        ),
    )
    gloss_key: str = Field(
        default="",
        description=(
            "Key of the reading of its enclosing function.  Carried on the "
            "candidate because eliminating a candidate has to take its reading "
            "with it: the point of the enumeration is that what the evidence "
            "rules out stops being offered."
        ),
    )
    tier: int = Field(
        default=1,
        description=(
            "1 when a branch states this value outright, 2 when the writer is "
            "known and the value is only settled at run time.  A tier-2 "
            "candidate is not eliminated by the value, because 'this one could "
            "have' is the honest answer for it -- unless the branch's own text "
            "is a frame the value does not fit, which is the one case where "
            "the arm rules itself out.  See ``outcome_excludes``."
        ),
    )
    dispatch: list[DispatchFanout] = Field(
        default_factory=list,
        description=(
            "Classes a run-time choice inside this junction's class could "
            "have picked.  Carried to the report because a candidate set "
            "nobody can check is three guesses wearing a bracket -- the line "
            "that prints the chosen class travels with it."
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
    handovers: list[Handover] = Field(
        default_factory=list,
        description=(
            "Entries that passed something in, with what.  Only the ones that "
            "did: an entry handing over nothing says nothing, and listing it "
            "would make 'this path omits the argument' and 'this reader does "
            "not bind positionals' look the same."
        ),
    )
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
    keep_lines: Optional[int] = Field(
        default=None,
        description=(
            "How many matched lines this question needs kept, where the role's "
            "default is wrong for it.  A probe asking *whether* a writer fired "
            "wants a handful for the report; a probe whose answer **is** the "
            "lines -- every candidate a chain dropped -- wants all of them, and "
            "a trimmed answer there is not a smaller sample but a different one."
        ),
    )
    matched: int = 0
    sample: list[str] = Field(
        default_factory=list, description="A few matched lines, for the report."
    )


#: What the reader a value's follow-up names can do to the row.  Three states,
#: because "a query selects this value" and "something will move this row on"
#: are not the same claim and the map was answering the second with the first:
#: ``analy_pmerge_jobs_wait_time`` selects ``cancelled`` jobs to average a wait
#: time and changes nothing, and the verdict told a reader to go and ask a
#: metrics daemon why it had not picked their row up.
#:
#: The split is by whether the reader is started by something of its own, not
#: by where it lives.  A getter settles nothing and writes nothing either, and
#: saying the row will not move would be just as wrong in the other direction
#: -- what acts is its caller, which the line does not name.
ACTS = "acts"
READS_ONLY_AT_TOP = "reads only, and nothing starts it but its own trigger"
READS_ONLY_FOR_A_CALLER = "reads only, on behalf of whoever called it"


class FollowUp(BaseModel):
    """Whether anything will move the value on, and what to ask if not.

    The other half of the question.  Which junction wrote ``pending`` says how
    the row got where it is; this says whether it is going to leave, and a task
    can be stuck for a reason that has nothing to do with the writer.
    """

    selected: bool = Field(
        ...,
        description=(
            "Whether anything in the map acts on rows by this value -- a "
            "query's predicate or an update's.  Both answer 'will the row be "
            "moved on at all'; ``selected_by`` and ``updated_by`` say which."
        ),
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
    reader_acts: str = Field(
        default=ACTS,
        description=(
            "What those readers can do to the row: ``ACTS`` where one settles a "
            "value or writes a row, and one of the two reads-only states where "
            "none does.  Separate from ``selected``, which stays true either "
            "way -- a query really does select on the value; the question this "
            "answers is whether being selected leads anywhere."
        ),
    )
    updated_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose update or delete acts on rows already holding "
            "the observed value.  Not a place the row could have been missed --"
            " an update is the picking up -- so it is named separately and only "
            "answers 'what will move this row' where no query does."
        ),
    )
    creates_rows: bool = Field(
        default=False,
        description=(
            "Whether anything in this map creates rows of the subject's kind.  "
            "Separate from ``created_by`` being empty, which would otherwise "
            "read as 'the map did not look': three kinds of row here are "
            "changed by this corpus and created outside it."
        ),
    )
    created_by: list[str] = Field(
        default_factory=list,
        description=(
            "The functions whose INSERT brings a row of this kind into "
            "existence.  The question a branch table cannot answer, because a "
            "branch is about a value a row already has.  Where the verdict is "
            "'ask whether the command arrived', this is where arriving happens."
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


class StageCut(BaseModel):
    """One filter stage, and what it actually removed for this entity.

    The unit of the brokerage answer.  A stage is not a candidate explanation
    the way a junction is -- every stage runs, on every pass -- so what has to
    be measured is not *whether* it fired but *how much of the list it took*.

    Counted in distinct sites rather than in log lines.  Brokerage re-runs for
    a task many times over the window, so a line count is a count of passes
    multiplied by an effect, and the ranking it produces is about how often the
    task was brokered.  The same weight-not-count reading that made one branch
    the answer for ``'cleanup'``.
    """

    tag: str = Field(default="", description="The ``criteria=-`` token, e.g. '-lowmemory'.")
    funnel_label: str = Field(default="", description="The coarser step, e.g. 'memory check'.")
    owner: str = Field(default="", description="The chain this stage belongs to.")
    order: int = Field(default=0, description="Its position in that chain.")
    line: Optional[int] = Field(default=None, description="Where to go and read.")
    conditions: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(
        default_factory=list, description="What those conditions read -- where a backward walk goes."
    )
    reads: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Attribute name -> the subject the map says that read names.  "
            "Resolved while the map is open, because the rejection line that "
            "supplies the *value* arrives later and by then there is nothing to "
            "resolve it against.  Names several subjects declare and the map "
            "cannot separate are absent here and reported as leads instead."
        ),
    )
    log_files: list[str] = Field(
        default_factory=list,
        description=(
            "Where this stage's own module writes.  What attributes an observed "
            "tag to a chain: two chains emit ``-status`` and the file the line "
            "landed in is the only thing that separates them.  Empty for a "
            "helper that declares no logger, whose lines surface in whichever "
            "broker called it."
        ),
    )
    sites: list[str] = Field(
        default_factory=list, description="The distinct candidates this cut removed."
    )
    reasons: list[str] = Field(
        default_factory=list,
        description=(
            "What the log line said as it dropped each one, most frequent first "
            "-- ``status=offline (31)``.  These carry the measured values that "
            "made the condition true, which is what turns a condition into a "
            "reason and is why the line is better evidence than re-reading the "
            "value now."
        ),
    )
    verdict: str = Field(default=UNSETTLED, description=f"{SEEN} | {ELIMINATED} | {UNSETTLED}")
    because: str = Field(default="")


class FunnelStep(BaseModel):
    """How many candidates were left after one step, as production counted them.

    Reported in the *map's* order, and aggregated rather than split into
    passes.  Splitting was measured and refused: over 51,128 adjacent pairs
    within one ``(file, machine, run key)``, 11.0% of them show the count going
    *up*, and the largest single offender is inside a pass rather than at its
    boundary.  So "the count only falls, therefore a rise is a new pass" is a
    property the code looks like it has and does not, and every way of guessing
    a boundary this check has tried cost it a false finding.
    """

    label: str
    order: Optional[int] = Field(
        default=None, description="Position in the chain, or None where the map has no step for it."
    )
    most: int = Field(default=0, description="The largest count seen after this step.")
    fewest: int = Field(default=0, description="The smallest.")
    seen: int = Field(default=0, description="How many times the step reported at all.")


class Localization(BaseModel):
    """Which step of a chain threw the candidates away.

    The answer to the second symptom class.  ``follow_up`` has no counterpart
    here: a filter stage carries no entry points, and pooling the chain's
    triggers over the module's junctions is the same over-approximation that
    was reverted once already, so what would have gone in it is reported as a
    gap instead.
    """

    chain: str = Field(default="", description="module::function holding the chain.")
    chains_sharing_the_file: list[str] = Field(
        default_factory=list,
        description=(
            "Other chains whose lines land in the same file.  A task broker calls "
            "a job broker and passes its own log slot, so one file holds two "
            "chains and a tag can belong to either.  Named rather than chosen "
            "between: the map records where a stage's own module writes, not who "
            "delegated to it."
        ),
    )
    log_files: list[str] = Field(default_factory=list)
    cuts: list[StageCut] = Field(
        default_factory=list, description="Ranked by how much of the list each took."
    )
    describes: str = Field(
        default="",
        description=(
            "What the description resolved to, when the map holds a stage for "
            "it: the ``criteria=-`` tag production writes per rejected "
            "candidate, or the funnel label it prints per count.  Empty for a "
            "chain, whose key is an owner and names no single stage."
        ),
    )
    emitted_by: list[StageCut] = Field(
        default_factory=list,
        description=(
            "The stages the map says write that tag.  Beside ``cuts`` rather "
            "than inside it because the two are known at different times and a "
            "reader must not confuse them: this is what the map can say before "
            "anything is asked, while a cut's rank is what production measured. "
            "Without it a described cut was never named at all -- the chain was "
            "printed and the stage inside it was left to be found among a "
            "hundred and nine, as deep as forty-seventh."
        ),
    )
    funnel: list[FunnelStep] = Field(default_factory=list)
    entered: Optional[int] = Field(
        default=None, description="Candidates the first step of the chain counted."
    )
    left: Optional[int] = Field(default=None, description="Candidates the last step counted.")
    passes: int = Field(
        default=0, description="How often the chain reported -- not used to split, only stated."
    )
    sample: str = Field(
        default="partial",
        description=(
            "complete | partial.  Only a complete sample licenses saying a stage "
            "removed nothing: an answer cut off at a byte or match bound is "
            "indistinguishable from a stage that never fired."
        ),
    )


#: Why a backward walk stops instead of continuing.  Each is a terminal
#: category -- a place the value came from that this map does not explain --
#: and naming which one is the answer, not the absence of one.
#:
#: The first five are reached by a :class:`Lead`, hop by hop through the map.
#: The last four are reached by the use-time trace, which walks the source of
#: one arm and gets further down into an expression than the map's own edges
#: do.  One vocabulary for both, because it is one question and two spellings
#: of it drift.  Measured on a prototype walk over the map's 983 locatable
#: arms: the configuration terminal fires 545 times and the external ones 34,
#: neither of which had anywhere to be recorded before this.
STOP_NO_WRITER = "nothing in the map writes that field"
STOP_SHARED_TABLE = "a table the map only ever reads"
STOP_NEEDS_VALUE = "read by the condition, but its value was not observed"
STOP_AMBIGUOUS = "several subjects declare that name and the map cannot say which"
STOP_DESCENT = "another entity's rows, which this derivation does not census"
STOP_CONFIG = "a configuration value, set outside the code"
STOP_EXTERNAL = "a value another system supplies"
STOP_UPSTREAM = "a value the database supplies, whose writers the map names"
STOP_PARAMETER = "chosen by whoever called or built this"


#: Where a lead came from, kept because the three are not equally trustworthy
#: and a trace that cannot tell them apart makes the map's coverage read better
#: than it is.  ``map`` is an edge the extraction recorded; ``callee`` is one
#: hop along a ``self.<method>()`` call, deterministic but assembled here;
#: ``gloss`` is a reading's proposal, a hypothesis the next hop's evidence
#: checks.  Only the first two exist today.
LEAD_MAP = "map"
LEAD_CALLEE = "callee"
LEAD_GLOSS = "gloss"


class Lead(BaseModel):
    """Where the answer continues after this one, or the reason it stops.

    **The map is asked once per hop, not once per investigation.**  A symptom is
    not one question but a path: "the task sits in ``finishing``" opens the
    read, the modify and the write, and learning that the modify stalled opens a
    different question -- which job it waits on -- that nobody typed and that
    the evidence, not the description, chose.  The vocabulary being closed is
    what makes each hop a selection out of a known set; the set of *paths* is
    not closed and does not need to be.

    So this deliberately does not recurse.  A hop needs evidence before the next
    one is worth taking, and evidence collection is a separate cadence -- a
    ``derive`` that walked would have to reach production from inside the
    derivation, which is the split the two phases exist to keep.  The loop
    belongs to whoever holds the history: enumeration is the map's job and
    selection is the reasoning's, and that seam is where an LLM can sit while
    still being checkable.

    A lead that continues carries a whole :class:`Symptom`, so the caller can
    put it straight back in.  That is the contract, and it is why a field read
    without a value *stops*: naming something to go and observe is a true and
    useful answer, but it is not yet a question the map can be asked.
    """

    field: str = Field(..., description="What was carried or read, qualified where known.")
    symptom: Optional[Symptom] = Field(
        default=None,
        description="The next question, ready to ask.  None when the walk stops here.",
    )
    stop: str = Field(
        default="",
        description="Why it stops, when it does -- one of the STOP_* categories.",
    )
    why: str = Field(default="", description="What opened this, in one clause.")
    opened_by: str = Field(
        default="",
        description=(
            "The candidate or cut that opened it.  Kept so that eliminating a "
            "candidate takes its leads with it: a walk that keeps descending "
            "from a branch the evidence ruled out is following a path the "
            "system did not take."
        ),
    )
    source: str = Field(
        default=LEAD_MAP,
        description=(
            "Which of the LEAD_* suppliers proposed it.  Separate from "
            "``opened_by``, which names *whose* survival it depends on and is "
            "what the eliminator filters by; this names *how sure* it is."
        ),
    )


#: What a :class:`TraceStep` is a step of.  ``write`` is the arm itself;
#: ``binding`` and ``loop-target`` are reaching definitions inside one
#: function; ``handover`` is the one crossing the map supplies, from a
#: worker's own attribute to the expression the knight built it with; and
#: ``unbound`` is a name this function does not bind, which is where the walk
#: either crosses or stops.
TRACE_WRITE = "write"
TRACE_BINDING = "binding"
TRACE_LOOP = "loop-target"
TRACE_HANDOVER = "handover"
TRACE_UNBOUND = "unbound"


#: What a skeleton row is.  ``branch`` is a compound statement's header and
#: carries no pattern; ``print`` is a line production writes; ``arm`` is one of
#: the writes the map sent the reader here for.
SKELETON_BRANCH = "branch"
SKELETON_PRINT = "print"
SKELETON_ARM = "arm"


class SkeletonLine(BaseModel):
    """One row of what a function prints, in source order, with the arms in place.

    The artefact a reader aligns a grepped log region against.  The map's
    ``log_pattern`` is one sentence for a whole function, settled at build time
    by majority over every writer of the subject; matched against a region it
    says the function ran and nothing more.  A skeleton says *which branch*,
    because the lines either side of an arm are in the region too and the
    nesting says which of them can have been printed together.

    Deliberately not one row per arm.  A line that reports nothing about the
    value -- ``log.debug(blah1)`` before a write and ``log.debug(blah2)`` after
    it -- still proves the write ran when both turn up, and a shape that
    insists every row name the value cannot express that at all.  ``value`` is
    the stronger reading kept beside the weaker one: where the observed value
    does land in a hole, the pattern with it filled in is narrower than the
    pattern without, so both are rendered and the row carries each.

    Nothing here is written into the map.  Like the rest of the walk it is a
    reading of the tree the map was built from, and is refused outright when
    that is not the tree at hand.
    """

    kind: str = Field(description=f"{SKELETON_BRANCH} | {SKELETON_PRINT} | {SKELETON_ARM}")
    line: int = Field(description="Where in the file this row is.")
    depth: int = Field(default=0, description="How deep the nesting is at this row.")
    text: str = Field(
        default="",
        description="The branch's header or the arm's statement, as the source spells it.",
    )
    pattern: str = Field(
        default="",
        description="The line as a pattern, with every hole left open.  Empty when refused.",
    )
    refused: str = Field(
        default="",
        description="Why no pattern was rendered, when none was -- shown in its place.",
    )
    arms: list[int] = Field(
        default_factory=list,
        description=(
            "The arms this row can have been printed alongside.  One of them "
            "means the row names which arm ran; several mean it does not."
        ),
    )
    value: str = Field(
        default="",
        description="The same line with the observed value in its hole, where one lands.",
    )
    hole: str = Field(
        default="", description="The expression whose place the observed value takes."
    )
    because: str = Field(
        default="", description="Why that hole is the one the value fills."
    )


class TraceStep(BaseModel):
    """One thing the source says about why an arm ran with the value it did.

    Computed at use time from the tree, not stored in the map.  The analysis
    is the same one the build already does -- reaching definitions and
    dominating guards -- and the cadence is the whole difference: storing it
    forces one condition string per branch, and that is what made the eager
    version need a loop rule, a length cap and a recursion rule before it
    could be written down at all.  A use-time answer may be a *set*, so none
    of those constraints arise; four sites setting one flag are four steps
    because four of them are the answer.

    ``guards`` is what the path condition says and ``unseen`` is what it
    cannot: only ``ast.If`` contributes to a path condition, so a step inside
    a handler, a loop or a ``try`` body carries an empty ``guards`` and is not
    unconditional.  86% of the map's arms have at least one of those on the
    way to them, which is why the two are separate fields rather than one
    list -- a reader that cannot tell them apart reads silence as certainty.
    """

    kind: str = Field(..., description="One of the TRACE_* kinds.")
    name: str = Field(default="", description="The name this step explains.")
    owner: str = Field(default="", description="module::function the site is in.")
    file: str = Field(default="")
    line: int = Field(default=0)
    value: str = Field(default="", description="The expression, as source.")
    guards: list[str] = Field(
        default_factory=list, description="The path condition of this site."
    )
    unseen: list[str] = Field(
        default_factory=list,
        description="What constrains reaching it that the path condition omits.",
    )
    reads: list[str] = Field(
        default_factory=list, description="Names this step's value depends on."
    )
    terminal: str = Field(
        default="",
        description="Empty while the walk continues, else one of the STOP_* categories.",
    )
    detail: str = Field(default="", description="Which interface, key or caller.")
    depth: int = Field(default=0, description="Hops from the arm.")


class Reading(BaseModel):
    """One function to read, the arms it covers, and the line to read it against.

    Per function rather than per arm because that is where the sharing is: the
    arms that survive a question routinely sit together -- one function holds
    up to eleven junctions -- and asking about each separately reads the same
    text over again and invites two answers about one piece of code.

    ``log_pattern`` is empty for an arm the map records no line for.  That is
    not a defect to be papered over: of 1046 branches, 988 can be pointed at in
    the source and 484 carry a line that production would print, so 43% can be
    put side by side and the rest are honestly *read this, there is nothing to
    match it against*.  Filling the gap with a guessed pattern would turn a
    known silence into an empty query, and an empty query is what this design
    reads as evidence.
    """

    owner: str
    file: str = Field(default="", description="Where the function is.")
    blob_sha: str = Field(
        default="", description="Hash of the file the map read it from."
    )
    gloss_key: str = Field(
        default="",
        description="Key of the reading of this function, from the map.",
    )
    lines: list[int] = Field(
        default_factory=list,
        description="The arms' own lines -- what to mark in the text handed over.",
    )
    outcomes: list[str] = Field(
        default_factory=list, description="The values those arms write."
    )
    dispatch: list[DispatchFanout] = Field(default_factory=list)
    log_files: list[str] = Field(default_factory=list)
    log_pattern: str = Field(
        default="",
        description="What production would print for these arms.  Empty means silent.",
    )

    trace: list[TraceStep] = Field(
        default_factory=list,
        description=(
            "Why these arms ran with the value they did, walked from the "
            "source at use time.  Empty without a source tree, and empty on "
            "purpose when the tree is not the one the map was built from."
        ),
    )
    trace_note: str = Field(
        default="",
        description="Why the trace is empty or short, when it is.",
    )
    skeleton: list[SkeletonLine] = Field(
        default_factory=list,
        description=(
            "What this function prints, in source order, with the arms in "
            "place -- computed from the tree at use time.  Beside "
            "``log_pattern`` rather than instead of it: that one is the map's "
            "shared sentence and is what the questions already asked are "
            "built from, and swapping them would make every answer already "
            "collected read as unasked."
        ),
    )

    @property
    def silent(self) -> bool:
        """True when the map records no production line for any of these arms."""
        return not self.log_pattern


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
    localization: Optional[Localization] = Field(
        default=None,
        description=(
            "Filled in for a distribution symptom instead of ``candidates``.  "
            "Both live on one model because everything around them is shared -- "
            "the observations, the evidence file, the two-phase split -- and "
            "only what the map says in the middle differs."
        ),
    )
    leads: list[Lead] = Field(
        default_factory=list,
        description=(
            "Where the answer goes next, and where it stops.  An investigation "
            "is a path through the map rather than one lookup, and this is the "
            "one hop this derivation can license -- filtered to the surviving "
            "candidates once the evidence is in."
        ),
    )
    readings: list[Reading] = Field(
        default_factory=list,
        description=(
            "The code to read, one entry per function, filtered to the "
            "surviving candidates once the evidence is in.  The map chooses "
            "the input; what reads it is not this module's business."
        ),
    )
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
#: Why a walk stopped.  Each is a fact about the investigation rather than an
#: error: a walk that ran out of map, one that came back to a question it had
#: already asked, and one that spent its budget are three different answers and
#: the reader has to be able to tell them apart.
WALK_TERMINAL = "every lead is a terminal -- the map has nothing further to open"
WALK_ASKED = "every question left had already been asked"
WALK_BUDGET = "the hop budget was spent"


class Hop(BaseModel):
    """One question asked of the map, and what opened it.

    The unit of a path.  Keeping the whole strategy rather than a summary is
    deliberate: a trace exists so the reasoning can be checked afterwards, and
    checking it means seeing which candidates were live at each step, not only
    which field the walk moved to.
    """

    number: int = Field(..., description="0 for the question that was asked, then 1, 2 …")
    symptom: Symptom
    opened: str = Field(
        default="",
        description="The lead's field.  Empty at hop 0, which nothing opened.",
    )
    source: str = Field(
        default=LEAD_MAP,
        description=(
            "Which supplier proposed the lead that opened this hop.  What makes "
            "the trace auditable: a path that went three hops on hypotheses is "
            "not the same claim as one that went three hops on recorded edges."
        ),
    )
    strategy: Strategy


class Investigation(BaseModel):
    """A path through the map, and why it ended.

    The product of an investigation is not the last hop but the path, the
    terminal it reached and which category that terminal is in.  A single
    :class:`Strategy` cannot hold that: it is one hop's worth by construction,
    because evidence has to be collected between hops and collecting it is a
    separate cadence.  So the history lives here and the loop lives in whoever
    owns this object -- the map enumerates, the evidence selects, and neither of
    those is a thing to be done from inside a derivation.

    Not a node.  It is the product of one investigation, not part of the
    vocabulary the map is written in, and storing it would make a question
    somebody asked once look like a fact about the code.
    """

    hops: list[Hop] = Field(default_factory=list)
    visited: list[str] = Field(
        default_factory=list,
        description=(
            "Questions already asked, as ``subject=value``.  ``status`` and "
            "``oldStatus`` copy from each other and ``jobStatus`` copies from "
            "itself, so a walk without this goes round for ever -- and one that "
            "dropped the repeat silently would report a loop as a dead end."
        ),
    )
    cycles: list[str] = Field(
        default_factory=list,
        description="Questions that came round again.  A property of the system, so it is reported.",
    )
    budget: int = Field(default=0, description="Hops allowed.")
    stopped: str = Field(default="", description="One of the WALK_* reasons.")
