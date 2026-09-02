"""Self-verification gates for a Code Map fragment.

The gates split in two, and the split is what lets a build verify itself
offline, before any production data is available:

**(i) code-internal consistency** needs only the source.  It catches extraction
bugs and misses by cross-checking two independent expressions of the same fact
in the code -- PanDA states most things twice, so a recognizer can be graded
against a second, weaker recognizer without any production data.

**(ii) production comparison** needs logs or the database.  It catches version
skew and things the code cannot reveal at all, such as whether a DEBUG-level
diagnostic the map promises as an observable actually survives production's
log level.  Those live in ``check-map``, not here.

A gate reports rather than raises.  A failing gate means one of the two
expressions is wrong -- not necessarily the extraction -- so it is a signal to
look, and silently dropping the affected slice would hide exactly what needs
attention.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from bamboo.codemap import evidence
from bamboo.codemap.models import MapFragment
from bamboo.codemap.panda import values

# ``passthrough(JediTaskSpec.oldStatus)`` -- the subject a branch copies from.
_PASSTHROUGH = re.compile(r"^passthrough\((.+)\)$")

# What a failing gate is a statement *about*.  Not decoration: these three ask
# for three different things, and printing them all as "FAIL" is what made the
# report unreadable.  A mapped log file that no machine has means the map is
# right and this deployment differs -- there is nothing to fix -- while a tag
# production emits and the map has no stage for is an extraction miss.  Telling
# them apart is the difference between a work item and a fact.
MAP_DEFECT = "map_defect"
DEPLOYMENT_FACT = "deployment_fact"
CHECK_BROKEN = "check_broken"

# Severity, worst first.  A check that did not run makes every other verdict
# unsafe to read, so it outranks a real defect; a deployment difference asks
# for no change at all, so it comes last.
KIND_ORDER = (CHECK_BROKEN, MAP_DEFECT, DEPLOYMENT_FACT)

_VERDICT = {MAP_DEFECT: "FAIL", DEPLOYMENT_FACT: "DIFFERS", CHECK_BROKEN: "BROKEN"}

# How much of what a gate read was actually read.  Carried on the result
# because the gates already establish it and until now kept it to themselves:
# a reader cannot tell which conclusions are load-bearing without it.
COMPLETE = "complete"
PARTIAL = "partial"


class GateResult(BaseModel):
    """Outcome of one gate."""

    gate: str
    passed: bool
    checked: int = 0
    unit: str = Field(
        default="checked",
        description=(
            "What ``checked`` counts.  Named per gate because the units are not "
            "comparable -- query answers, log files, filter stages, observed "
            "tags and step pairs all printed as 'checked' invited exactly the "
            "comparison that means nothing."
        ),
    )
    kind: str = MAP_DEFECT
    question: Optional[str] = Field(
        default=None, description="The gate's question in plain words, for the report."
    )
    finding: Optional[str] = Field(
        default=None, description="What a failure of this gate is, as a statement."
    )
    sample: Optional[str] = Field(
        default=None,
        description=(
            "``complete`` or ``partial`` for a gate that read a sample of "
            "production, None where the question is not answered from a sample."
        ),
    )
    failures: list[str] = Field(default_factory=list)
    inconclusive: list[str] = Field(default_factory=list)
    note: Optional[str] = None

    @property
    def verdict(self) -> str:
        """PASS, or the word for what kind of failure this is."""
        return "PASS" if self.passed else _VERDICT.get(self.kind, "FAIL")

    @property
    def actionable(self) -> bool:
        """Whether this failure asks for a change to the map or the query.

        A deployment difference does not: the map and the source agree, and the
        deployment simply does not exercise that code.  Which is why it must
        not fail a pipeline either.
        """
        return not self.passed and self.kind in (MAP_DEFECT, CHECK_BROKEN)

    def summary(self) -> str:
        parts = []
        if self.failures:
            parts.append(f"{len(self.failures)} issue(s)")
        if self.inconclusive:
            parts.append(f"{len(self.inconclusive)} inconclusive")
        detail = f" ({', '.join(parts)})" if parts else ""
        return f"[{self.verdict}] {self.gate}: {self.checked} {self.unit}{detail}"


def value_enum_referenced(fragment: MapFragment) -> GateResult:
    """(i) Every extracted enumeration constant is referenced somewhere.

    A constant nothing reads is either dead or was never an enumeration member
    to begin with -- a threshold or a config default that the name-shape
    heuristic swept up.  Either way it does not belong in a reverse index whose
    whole job is decoding values that actually appear in records.

    This is the code-internal half.  The production half asks the opposite
    question -- whether every code seen in a real record is in the index --
    and only real records can answer it.
    """
    unreferenced = [
        f"{e.name} = {e.value!r} ({e.anchor.as_ref() if e.anchor else 'unknown'})"
        for e in fragment.value_enums
        if e.references == 0
    ]
    return GateResult(
        gate="value-enum-referenced",
        passed=not unreferenced,
        checked=len(fragment.value_enums),
        unit="constants",
        question="is every constant in the index read by something?",
        finding="the index holds constants nothing reads",
        failures=unreferenced,
        note="Unreferenced constants are dead or mis-classified, not index entries.",
    )


def namespace_disambiguates(fragment: MapFragment) -> GateResult:
    """(i) A value is unique *within* its namespace.

    Reused numbers are expected and are the reason the index is keyed on
    ``(namespace, value)`` -- ``EC_Kill``, ``EC_Setupper`` and ``EC_Watcher``
    are all ``100`` in different subsystems.  What must not happen is a
    collision *inside* one namespace, because then the key does not identify
    anything and decoding an observed code becomes ambiguous.
    """
    seen: dict[tuple[str, object], list[str]] = {}
    for enum in fragment.value_enums:
        seen.setdefault((enum.namespace, enum.value), []).append(enum.constant)
    collisions = [
        f"{ns}:{value!r} -> {sorted(names)}"
        for (ns, value), names in seen.items()
        if len(names) > 1
    ]
    return GateResult(
        gate="namespace-disambiguates",
        passed=not collisions,
        checked=len(seen),
        unit="namespaced values",
        question="does one value identify one constant inside its namespace?",
        finding="a namespace reuses a value, so the index key identifies nothing",
        failures=collisions,
        note="Cross-namespace reuse is expected; within one namespace it breaks the key.",
    )


def boundary_ownership_param_declared(fragment: MapFragment) -> GateResult:
    """(i) An ownership check names a parameter the endpoint actually declares.

    ``@request_validation(..., task_owner=True, task_id_param="task_id")``
    states which parameter carries the task whose owner is checked.  The
    signature states which parameters exist.  If they disagree the check reads
    a parameter that is not there, and the endpoint's access control does not
    do what its declaration says -- a disagreement between two independent
    expressions of the same fact, which is exactly what a code-internal gate
    is for.
    """
    failures = [
        f"{b.interface} declares task_owner on "
        f"{b.access_conditions.get('task_id_param', 'task_id')!r}, "
        f"but its parameters are {b.carried_values}"
        for b in fragment.boundaries
        if b.access_conditions.get("task_owner")
        and b.access_conditions.get("task_id_param", "task_id") not in b.carried_values
    ]
    checked = sum(1 for b in fragment.boundaries if b.access_conditions.get("task_owner"))
    return GateResult(
        gate="boundary-ownership-param",
        passed=not failures,
        checked=checked,
        unit="owner checks",
        question="does an ownership check name a parameter the endpoint declares?",
        finding="an endpoint's access check reads a parameter that is not there",
        failures=failures,
        note="The decorator and the signature must agree on which parameter carries the task.",
    )


def structural_attribution_agrees(fragment: MapFragment) -> GateResult:
    """(i) The declared type and the code's usage name the same spec class.

    Two independent expressions of one fact.  A junction's subject is settled
    from what the code *states* -- a constructor call, an annotation, ``self``
    in a spec's own method, or an attribute only one class declares.  Its
    ``structural_subject`` is settled from what the code *does*: the set of
    attributes touched on that object, matched against the classes declaring a
    superset of them.  Neither reads the other, so agreement is evidence and
    disagreement means one of them is wrong.

    This gate replaces the vocabulary comparison, which was designed for this
    role and did not survive contact with the source (see
    :func:`outcomes_outside_declared_subsets`).  Where that one could check 27
    branches and failed on two correct ones, this checks every attributed
    junction and, on PanDA, disagrees nowhere.

    Junctions with no structural answer are skipped: too few attributes were
    touched to imply anything, which is not a disagreement.  Only the attribute
    slice produces one, so the count is well below the junction total -- the
    SQL and alias slices settle their class from the table and from the method's
    defining class, neither of which the object's usage can corroborate.
    """
    failures: list[str] = []
    checked = 0
    for junction in fragment.junctions:
        if junction.structural_subject is None or junction.attribution == "unresolved":
            continue
        checked += 1
        if junction.structural_subject != junction.subject:
            where = junction.anchor.as_ref() if junction.anchor else junction.owner
            failures.append(
                f"{junction.subject} ({junction.attribution}) vs "
                f"{junction.structural_subject} (usage) at {where}"
            )
    return GateResult(
        gate="structural-attribution-agrees",
        passed=not failures,
        checked=checked,
        unit="attributed junctions",
        question="do the declared type and the object's usage name one class?",
        finding="what the code declares and what it does with the object disagree",
        failures=sorted(set(failures)),
        note="What the code declares and what it does with the object must name one class.",
    )


def outcomes_outside_declared_subsets(fragment: MapFragment) -> list[tuple[str, str, str]]:
    """Outcomes no declared status list mentions.  **Reported, not gated.**

    This was designed as a gate -- the spec classes state their status sets in
    ``statusToReassign()`` and friends, so an extracted outcome missing from
    them would mean the declaration or the extraction was wrong.  Running it
    against PanDA showed the premise is false.  The methods declare *subsets
    selected for a purpose*, not vocabularies::

        # return list of status to update contents
        def statusToUpdateContents(cls):
            return ["defined"]

    Their union is a sample of the status space, so an outcome outside it is
    the normal case, not a disagreement.  Both failures the gate produced were
    correct code and correct attribution: ``JediTaskSpec.status = 'finishing'``
    sits next to ``= 'tobroken'`` in the same ``if``/``else``, and
    ``JediDatasetSpec.status = 'ready'`` is two lines from
    ``self.inMasterDatasetSpec.append(datasetSpec)``.

    The sound direction is the opposite one -- every status a subset names
    should be written by *some* junction, since nothing could otherwise reach
    it.  That is a lower bound, and checking it needs every write form
    extracted; with only literal attribute writes recognised it would fail on
    values written through SQL binds.  It belongs with the graph invariants,
    once the writes are complete.

    So this stays a report: it still marks where the declarations and the code
    have drifted apart, which is worth a look, but it cannot decide who is
    wrong and must not fail a build.
    """
    vocabularies = {s.name: set(s.vocabulary) for s in fragment.subjects if s.vocabulary}
    rows: set[tuple[str, str, str]] = set()
    for junction in fragment.junctions:
        vocabulary = vocabularies.get(junction.subject)
        if not vocabulary:
            continue
        where = junction.anchor.as_ref() if junction.anchor else junction.owner
        for branch in junction.branches:
            if branch.outcome not in vocabulary:
                rows.add((junction.subject, branch.outcome, where))
    return sorted(rows)


def unresolved_attributes(fragment: MapFragment) -> list[tuple[str, int]]:
    """Attributes with writes whose spec class could not be settled, worst first.

    Not a gate: the code does not state the type, and reading it harder does
    not change that.  It is reported per attribute rather than as a total
    because the gap is not spread evenly -- it concentrates on the attributes
    several classes declare, which are also the ones the reasoning starts from
    most often, so a single number would hide where the map is thin.

    Each row is a candidate for a type annotation upstream.  That is the
    intended remedy: an unresolved shape is a request for one line of standard
    Python in the target system, not for another inference rule here.
    """
    counts: Counter = Counter()
    for junction in fragment.junctions:
        if junction.attribution == "unresolved":
            counts[junction.subject.split(".", 1)[-1]] += len(junction.branches) or 1
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))


def unobservable_boundaries(
    fragment: MapFragment, threshold: float = 0.5
) -> list[tuple[str, int, int]]:
    """Boundaries that log little of what they receive, worst first.

    Not a gate: logging less is sometimes correct -- an endpoint taking user
    secrets *should* log none of them.  It is reported because it bounds what
    can be investigated afterwards.  A value that crossed a boundary and was
    never written down cannot be recovered from the record, so an incident that
    turns on it can only be guessed at, and knowing that in advance is better
    than discovering it mid-investigation.
    """
    rows = [
        (b.interface, len(b.observable_values), len(b.carried_values))
        for b in fragment.boundaries
        # A shared table *is* the record, so asking whether it was also logged
        # inverts the question: the values are queryable afterwards precisely
        # because nobody had to write them down a second time.
        if b.transport == "http"
        and b.carried_values
        and len(b.observable_values) / len(b.carried_values) < threshold
    ]
    rows.sort(key=lambda r: (r[1] / r[2], -r[2]))
    return rows


def coverage_matrix(fragment: MapFragment) -> list[tuple[str, str, int, int, float]]:
    """Return per-slice, per-file coverage rows sorted worst first.

    Reported as a matrix rather than an average on purpose: recognizer
    reliability varies enormously between files that look alike, and a single
    mean lets a well-behaved slice mask one that matches nothing.  A file with
    candidates and no matches is the signal that it uses a different idiom.
    """
    rows = [
        (c.slice_name, c.file, c.candidates, c.explained, c.ratio)
        for c in fragment.coverage
    ]
    rows.sort(key=lambda r: (r[4], -r[2]))
    return rows


def _tier_one_outcomes(fragment: MapFragment) -> dict[str, set[str]]:
    """Return ``{subject: statically resolved outcomes}``."""
    written: dict[str, set[str]] = {}
    for junction in fragment.junctions:
        for branch in junction.branches:
            if branch.tier == 1:
                written.setdefault(junction.subject, set()).add(branch.outcome)
    return written


def declared_status_is_written(fragment: MapFragment) -> GateResult:
    """(i) A value the code declares must be one some writer produces.

    The sound direction of the comparison that was tried the other way round
    and refuted.  Checking that every extracted outcome sits inside a declared
    list fails on correct code, because ``statusToUpdateContents()`` returns
    ``["defined"]`` -- a purpose-built subset, not a vocabulary.  Checking that
    every *declared* value is written somewhere is a genuine lower bound: a
    status the code names in one of those lists and no writer produces is
    unreachable, so either a write form is still missing or the declaration is
    stale.

    It could only be run once every write form was in: with the SQL bind, the
    inline literal and the copied column all extracted, a miss now means
    something.  Where the subject has tier-2 writers the value may simply be
    computed -- ``commandStatusMap()[cmd]["doing"]`` produces ``aborting``
    without the literal appearing anywhere -- so the count is reported with the
    failure rather than left for the reader to guess at.
    """
    written = _tier_one_outcomes(fragment)
    runtime: dict[str, int] = {}
    for junction in fragment.junctions:
        for branch in junction.branches:
            if branch.tier != 1:
                runtime[junction.subject] = runtime.get(junction.subject, 0) + 1

    failures: list[str] = []
    checked = 0
    for subject in fragment.subjects:
        if not subject.vocabulary:
            continue
        for value in subject.vocabulary:
            checked += 1
            if value in written.get(subject.name, set()):
                continue
            computed = runtime.get(subject.name, 0)
            failures.append(
                f"{subject.name} declares {value!r} and no writer produces it"
                + (f" ({computed} run-time writer(s) could)" if computed else "")
            )
    return GateResult(
        gate="declared-status-is-written",
        passed=not failures,
        checked=checked,
        unit="declared values",
        question="does some writer produce every value the code declares?",
        finding="a declared status no writer produces -- a missing write form or a stale declaration",
        failures=sorted(failures),
        note="A declared status nothing writes is unreachable: a missing write form, or a stale declaration.",
    )


def map_references_resolve(fragment: MapFragment) -> GateResult:
    """(i) Every edge in the map lands on a node the map contains.

    Cheap, and the only check that covers the *assembly* rather than any one
    slice.  Two edges must land: a junction names the subject it writes, and a
    subject is worth keeping only if something writes it.  Both are maintained
    by construction, which is exactly why they are worth asserting -- a
    promotion rule changed in isolation breaks them silently, and a backward
    walk that steps onto a missing node has no way to say so.  Writing this
    caught the promotion closure putting a criterion on a name that had no
    node behind it.

    A passthrough target that is *not* a subject is not a failure: see
    :func:`carried_from_outside`.
    """
    subjects = {subject.name for subject in fragment.subjects}
    failures: list[str] = []
    checked = 0
    for junction in fragment.junctions:
        checked += 1
        if junction.attribution != "unresolved" and junction.subject not in subjects:
            failures.append(f"{junction.owner} writes {junction.subject}, which is not a subject")
    written = {junction.subject for junction in fragment.junctions}
    for subject in fragment.subjects:
        checked += 1
        if subject.name not in written:
            failures.append(f"{subject.name} is a subject nothing writes")
    return GateResult(
        gate="map-references-resolve",
        passed=not failures,
        checked=checked,
        unit="edges",
        question="does every edge land on a node the map contains?",
        finding="an edge points at a node the map does not have",
        failures=sorted(set(failures)),
        note="A backward walk that steps onto a missing node cannot report that it did.",
    )


def spec_declarations_are_read(fragment: MapFragment) -> GateResult:
    """(i) Every column declaration the target system makes yields something.

    The target system states which columns each of its classes holds, and the
    extraction reads those statements.  Two expressions of one fact, so they can
    be compared -- and the comparison needs no threshold, because the failure is
    total: a declaration the reader does not understand yields *zero* names.

    This has now been the shape of two blind spots.  The first pass looked for
    ``attributes`` and found three classes, because most of them say
    ``_attributes``.  Later, four classes moved to a form that carries types --
    ``attributes_with_types = (AttributeWithType("status", str), ...)`` -- and
    derive the old name from it, so the reader matched the assignment and took
    nothing.  Twenty-seven writes went unattributed, twenty-four of them in one
    module, and ``unresolved`` absorbed them all without a gate looking.

    The wider lesson is that a naming convention drifting is invisible to any
    tool that was told the convention: it is exactly as silent as having nothing
    to read.  Only counting what came back says otherwise.
    """
    unread = sorted(where for where, names in fragment.declaration_yields.items() if not names)
    return GateResult(
        gate="spec-declarations-are-read",
        passed=not unread,
        checked=len(fragment.declaration_yields),
        unit="declarations",
        question="does every declaration the extraction reads yield anything?",
        finding="a class declares its columns in a form the extraction does not read",
        failures=[f"{where} declares columns and the extraction read none" for where in unread],
        note="A form nobody reads looks exactly like a class with nothing to declare.",
    )


def container_annotations_agree(fragment: MapFragment) -> GateResult:
    """(i) A container's stated element type matches what the code puts in it.

    ``Dict[str, DatasetSpec]`` is asked of PanDA precisely because a bare
    ``Dict`` says nothing, which makes the element type a fact the map takes on
    trust.  One went in wrong::

        row_id_spec_map: Dict[int, JediFileSpec] = {}
        for fileSpec in job_spec.Files:              # JobSpec.Files: FileSpec
            row_id_spec_map[fileSpec.row_ID] = fileSpec

    ``FileSpec`` declares ``row_ID``, ``JediFileSpec`` does not, so the line
    below the annotation contradicts it.  Nothing caught it:
    ``structural_attribution_agrees`` compares two readings of *one expression*
    and here they are an expression apart -- the annotation is on the mapping,
    the attribute that separates the classes is touched on the loop variable.
    The writes the annotation resolves, ``.fileID`` and ``.attemptNr``, are
    declared by both classes, so checking the write settles nothing either.

    The second reading is what the code stores, and it is used **only** here.
    Inferring element types from assignment resolves nothing the annotation
    does not already resolve while putting 301 containers in scope, which is
    the harvest bar that deleted naming inference.  As a check it costs one
    pass over seventeen sites and catches a wrong subject.
    """
    containers = [row for row in fragment.annotation_readings if row.kind == "container"]
    conflicts = [row for row in containers if row.put_in and row.stated not in row.put_in]
    return GateResult(
        gate="container-annotations-agree",
        passed=not conflicts,
        checked=len(containers),
        unit="container annotations",
        question="does a container hold what its annotation says?",
        finding="an annotation names a class the code contradicts",
        failures=[
            f"{row.where} says {row.container} holds {row.stated}, "
            f"but the code puts in {', '.join(sorted(row.put_in))}"
            for row in conflicts
        ],
        note=(
            "The annotation is trusted above structural inference, so a wrong "
            "one is a confident wrong subject rather than an open question."
        ),
    )


def annotations_are_read(fragment: MapFragment) -> GateResult:
    """(i) Every annotation the map takes on trust changes a write.

    The mirror of ``spec-declarations-are-read``, and it exists for the same
    reason: an annotation nobody reads looks exactly like a container with
    nothing to state.  Two went into PanDA that resolved nothing at all -- one
    naming the wrong class, one naming the right class through a read form the
    extraction did not have (``d.get(key)`` where only ``d[key]`` was read).
    Both were reported as closing writes they never closed.

    Ablation rather than a proxy: the annotation is removed, the writes in its
    scope are resolved again, and the two answers are compared including the
    basis.  Asking instead "is some write attributed to the class this names"
    would be wrong twice over -- an annotation is often read *transitively*
    (``jobs: List[JobSpec]`` types ``job``, which types ``file`` through
    ``JobSpec.Files``, and the write that lands is ``FileSpec.status``), and an
    annotation that only lifts a write from ``structural`` to ``certain``
    changes no class while still being read.

    Scope reaches into subclasses, because PanDA annotates in the base and
    consumes in the derived class.  A verdict of unread means the audit looked
    where the field is actually used and the map still does not depend on it:
    either the annotation is redundant, or a read form is missing here.

    Context manager yield types are audited beside container element types, and
    for them this is the *only* mechanical check.  ``Iterator[WorkflowSpec |
    None]`` on ``workflow_lock`` has no second reading: the attributes touched
    on the locked spec (``status``, ``end_time``, ``workflow_id``) are declared
    by every workflow spec, so ``structural_attribution_agrees`` corroborates
    the step lock's annotation and says nothing about the other two.
    """
    unread = [row for row in fragment.annotation_readings if not row.read]
    return GateResult(
        gate="annotations-are-read",
        passed=not unread,
        checked=len(fragment.annotation_readings),
        unit="spec annotations",
        question="does the extraction read every annotation it asked for?",
        finding="an annotation the map asked for changes nothing",
        failures=[
            f"{row.where} states {row.stated} for {row.container} "
            "and no write in scope resolves differently without it"
            for row in unread
        ],
        note=(
            "Either the annotation is redundant or the read form is missing; "
            "the two look identical from here, so both are worth a look."
        ),
    )


def map_identities_are_distinct(fragment: MapFragment) -> GateResult:
    """(i) No two nodes of one kind share a semantic signature.

    A signature is the merge key, so two nodes sharing one are not two nodes:
    storing the map keeps whichever was written last and the other is gone.
    That loss is invisible from either side -- the build reports what it built
    and the database reports what it holds, and nobody had put the two numbers
    next to each other.  It cost a real arm of the map: ``AtlasProdJobBroker``
    runs "temporary problem check" at two points that test different things,
    both were built, and only the second survived being stored, so the map
    said that cut was unconditional when it is not.

    Cheap and general, in the way ``map-references-resolve`` is: it makes no
    claim about any slice, only that the identity scheme is total.  A slice
    that starts producing indistinguishable nodes says so here rather than in
    a diagnosis six months later.
    """
    failures: list[str] = []
    checked = 0
    for kind in ("subjects", "junctions", "boundaries", "value_enums", "filter_stages"):
        nodes = getattr(fragment, kind)
        checked += len(nodes)
        for name, count in Counter(node.name for node in nodes).items():
            if count > 1:
                where = sorted(
                    f"{n.anchor.file}:{n.anchor.line_start}"
                    for n in nodes
                    if n.name == name and getattr(n, "anchor", None)
                )
                failures.append(
                    f"{count} {kind[:-1]}(s) share the signature {name}"
                    + (f" -- {', '.join(where)}" if where else "")
                )
    return GateResult(
        gate="map-identities-are-distinct",
        passed=not failures,
        checked=checked,
        unit="nodes",
        question="does every node have a signature no other node shares?",
        finding="two nodes share one merge key, so storing the map keeps only one",
        failures=sorted(failures),
        note="The build counts what it made; the store keeps one per signature. Nothing else compares them.",
    )


def carried_from_outside(fragment: MapFragment) -> list[tuple[str, str]]:
    """Subjects whose value is copied from a field the map does not explain.

    ``JediTaskSpec.currentPriority`` is set from ``taskPriority``, and nothing
    in this map writes ``taskPriority`` -- it arrives with the task.  That is a
    terminal, in the same sense as a boundary: the backward walk stops there,
    and the answer "it was whatever the submitter asked for" is complete rather
    than missing.  Reported so the stop is visible instead of looking like a
    gap in extraction.
    """
    subjects = {subject.name for subject in fragment.subjects}
    rows = {
        (junction.subject, match.group(1))
        for junction in fragment.junctions
        for branch in junction.branches
        if (match := _PASSTHROUGH.match(branch.outcome)) and match.group(1) not in subjects
    }
    return sorted(rows)


def unreachable_values(fragment: MapFragment) -> list[tuple[str, list[str]]]:
    """Values something writes that nothing selects rows on.  **Reported.**

    The plan's "in-edges but no out-edge" invariant, and it cannot be a gate:
    a terminal status is *supposed* to be a sink, and nothing in the source
    declares which ones those are.  What it can do is keep the list short
    enough to read -- ``JediTaskSpec.status`` comes out as ``broken``, ``lost``,
    ``finishing``, ``staging`` and ``waiting``, of which the first two are
    plainly terminal and the last three are transitional names that no query in
    this map ever acts on.

    Subjects nothing selects on at all are skipped.  There the map has no read
    side to compare against, so every value would appear to be a sink -- which
    is not an answer of "nothing moves away from these" but an absence of the
    question.
    """
    written = _tier_one_outcomes(fragment)
    rows = [
        (subject.name, sorted(written.get(subject.name, set()) - set(subject.selected_values)))
        for subject in fragment.subjects
        if subject.selected_values
    ]
    return sorted((name, sinks) for name, sinks in rows if sinks)


def unexplainable_rejections(fragment: MapFragment) -> list[tuple[str, list[str], str]]:
    """Filter stages whose message carries none of what their condition tested.

    Two expressions of one fact again, and this time they are the condition and
    the message beside it.  ``criteria=-max_io_intensity`` is logged as ``skip
    site={} since ioIntensity={} is larger than site max_io_intensity={}``,
    which carries both sides of its own comparison -- so the log alone settles
    why that site went, with nothing re-fetched.

    Where they disagree the map is promising an observation it cannot deliver.
    ``criteria=-diskIO`` tests three values and logs only the site name, and
    the numbers are on a separate line that a different code path emits: the
    stage is still correct, but "which branch fired" cannot be answered from
    the rejection alone.

    Not a gate, for the same reason :func:`unobservable_boundaries` is not:
    logging less is often deliberate.  The point is to know, before an incident
    turns on it, which rejections can be explained from the record and which
    can only be guessed at.
    """
    rows: list[tuple[str, list[str], str]] = []
    for stage in fragment.filter_stages:
        if not stage.emits or not stage.inputs:
            continue
        logged = " ".join(stage.emits)
        if any(name in logged for name in stage.inputs):
            continue
        where = stage.anchor.as_ref() if stage.anchor else stage.owner
        rows.append((stage.criteria_tag or stage.funnel_label, stage.inputs[:4], where))
    rows.sort()
    return rows


# ---------------------------------------------------------------------------
# (ii) Production comparison
# ---------------------------------------------------------------------------


def _sample_word(whole: int, asked: int) -> Optional[str]:
    """``complete``, ``partial``, or None when nothing was asked."""
    if not asked:
        return None
    return COMPLETE if whole == asked else PARTIAL


def _present(ev: "evidence.Evidence") -> list[str]:
    """The log files production actually has, sorted."""
    return sorted(f for f in ev.log_filenames() if ev.file_status(f) == "present")


def log_format_recognised(ev: "evidence.Evidence") -> GateResult:
    """(ii) The sampled log lines parse as log lines.

    First and least interesting of the production gates, and the one that has
    to run before any of the others mean anything.  Every later gate reads
    production by matching a pattern against log text; if the format is not
    what the map assumed -- a different formatter, a wrapper, the wrong file
    -- those patterns match nothing, and a gate that matches nothing looks
    exactly like a gate that passed.  So the format is checked directly, once,
    against the one thing every line has.
    """
    failures: list[str] = []
    checked = 0
    for filename in sorted(ev.log_filenames()):
        results = ev.matching(evidence.ANY_LINE_PATTERN, log_filename=filename)
        checked += len(results)
        # A file that is not there is reported by ``code_paths_are_live`` as
        # the finding it is; only some other error means the query itself
        # failed, and that has to be visible before anything is concluded.
        broken = [
            r for r in results if r.error and not evidence._MISSING_FILE.search(r.error)
        ]
        for result in broken:
            failures.append(f"{result.query.service}/{result.machine}: {result.error}")
        # Counted from what matched, not from what was kept: the level query
        # keeps no lines, so len(lines) would read as "nothing matched".
        sampled = sum(r.matched for r in results)
        if sampled and not evidence.level_histogram(ev, log_filename=filename):
            failures.append(
                f"{filename}: sampled {sampled} line(s), none of which carry a "
                f"level -- the log format is not {evidence.ANY_LINE_PATTERN!r}"
            )
    whole, asked = evidence.sample_state(
        ev, evidence.ANY_LINE_PATTERN, _present(ev), needs_lines=False
    )
    return GateResult(
        gate="log-format-recognised",
        passed=not failures,
        checked=checked,
        unit="query answers",
        # A failure here says nothing about PanDA: the query did not run, or ran
        # against something that is not a PandaLogger file.  Naming it as such
        # keeps it from reading as production disagreeing with the map.
        kind=CHECK_BROKEN,
        question="is the log in the format the map assumed?",
        finding="the sampled log does not parse, so no other production gate can conclude",
        sample=_sample_word(whole, asked),
        failures=failures,
        note="Later production gates read this format; unrecognised means they cannot conclude.",
    )


def observables_are_emitted(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) What the map offers as an observable survives production's log level.

    The map records the level of every diagnostic it points an investigation
    at.  A line written at DEBUG does not exist in a service running at INFO,
    so an observable below the threshold is a promise the map cannot keep --
    worse than having no observable at all, because a strategy will spend a
    step fetching it.

    The threshold is measured per service: JEDI and the server are separate
    machine groups under separate operation, so one level does not imply the
    other.  A service that yielded no sample gets no verdict rather than an
    optimistic one; those land in ``inconclusive``.
    """
    failures: list[str] = []
    unknown: list[str] = []
    checked = 0
    for stage in fragment.filter_stages:
        if not stage.log_level:
            continue
        checked += 1
        where = stage.anchor.as_ref() if stage.anchor else stage.owner
        # Reason first, anchor second.  These rows are summarised by their first
        # line elsewhere, and with the anchor in front the summary cut away the
        # half that says what happened.
        label = f"{stage.criteria_tag or stage.funnel_label} at {where}"
        if not stage.log_files:
            unknown.append(f"no log file is named in the source for {label}")
            continue
        # Any candidate that carries the line is enough: the proxy mixins run
        # under two processes and emitting in either one makes the observable
        # real.  Absent everywhere is reported by ``code_paths_are_live``, not
        # here -- a path that never ran is not a broken promise about logging.
        emitted = False
        suppressed: list[str] = []
        unproven: list[str] = []
        absent: list[str] = []
        for filename in stage.log_files:
            # Only an *absent* file is a reason to skip.  "unknown" merely means
            # this particular question was never put to it, and treating that as
            # a reason to say nothing would silence the gate whenever the level
            # query was not among the ones issued.
            if ev.file_status(filename) == "absent":
                absent.append(filename)
                continue
            threshold = evidence.effective_level(ev, log_filename=filename)
            if threshold is None:
                continue
            if not evidence.below_threshold(stage.log_level, threshold):
                # A line at or under this level was seen.  Positive evidence,
                # so how much of the log was read does not matter.
                emitted = True
                break
            # The opposite direction is not symmetric.  "No line this low in
            # the sample" only means production suppresses it if the sample was
            # everything the query asked for; a capped one may simply have
            # stopped before reaching one.
            if ev.conclusive(evidence.ANY_LINE_PATTERN, log_filename=filename):
                suppressed.append(f"{filename} at {threshold}")
            else:
                unproven.append(f"{filename} (sample incomplete)")
        if emitted:
            continue
        if suppressed:
            failures.append(
                f"{label} emits at {stage.log_level.upper()} but "
                + ", ".join(suppressed)
            )
        elif unproven:
            unknown.append(f"the sample is incomplete for {label} ({', '.join(unproven)})")
        elif absent:
            # Say why rather than "no sample": an absent file is a finding of
            # its own, reported by ``code_paths_are_live``, and describing it
            # here as missing evidence makes one fact look like two problems.
            unknown.append(
                f"{', '.join(absent)} is on no machine, so {label} never ran here"
            )
        else:
            unknown.append(f"no sample from {', '.join(stage.log_files)} for {label}")
    whole, asked = evidence.sample_state(
        ev, evidence.ANY_LINE_PATTERN, _present(ev), needs_lines=False
    )
    return GateResult(
        gate="observables-are-emitted",
        passed=not failures,
        checked=checked,
        unit="filter stages",
        # The map and the source agree; the deployment runs at a level that
        # drops the line.  Nothing to fix in the extraction -- the stage comes
        # out of strategies instead.
        kind=DEPLOYMENT_FACT,
        question="do the promised lines survive the log level?",
        finding="the deployment's log level drops an observable the map offers",
        sample=_sample_word(whole, asked),
        failures=failures,
        inconclusive=unknown,
        note="An observable below the threshold must be dropped from strategies, not fetched.",
    )


def code_paths_are_live(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Every mapped log file exists in the deployment.

    PandaLogger opens ``panda-<logger>.log`` the first time that logger
    emits, so a file none of the machines has is not a gap in the evidence --
    it says the code writing to it has never run there.  That is the strongest
    statement production makes about the map, and it outranks the level check:
    a stage nobody executes is not an observability problem, it is a part of
    the map that does not apply to this deployment.

    Reported rather than silently dropped.  The map is built from the source,
    and the source is right about the code existing; what this adds is that
    the deployment does not exercise it, which an investigation needs to know
    before it starts looking for lines that will never be there.
    """
    counts: dict[str, Counter] = {}
    for kind, nodes in (("stage", fragment.filter_stages), ("junction", fragment.junctions)):
        for node in nodes:
            for filename in node.log_files:
                counts.setdefault(filename, Counter())[kind] += 1
    # Reported as how much of the map goes with the file, because that is the
    # consequence: these are the nodes a strategy must not send an investigation
    # to.  The module is not named -- the filename comes from the logger, which
    # comes from the module, so it would be the same word twice.
    verdicts = {filename: ev.file_status(filename) for filename in sorted(counts)}
    failures = [
        f"{filename} is on no machine: {counts[filename]['stage']} stage(s), "
        f"{counts[filename]['junction']} junction(s) of the map never run here"
        for filename, status in verdicts.items()
        if status == "absent"
    ]
    # Only the files the evidence answered.  Counting every file the map names
    # would put the ones nobody asked about in the denominator, which makes the
    # corroboration below read weaker than the evidence actually is -- and the
    # set the map names grows whenever a slice widens, without a query behind it.
    answered = [status for status in verdicts.values() if status != "unknown"]
    return GateResult(
        gate="code-paths-are-live",
        passed=not failures,
        checked=len(answered),
        unit="log files",
        # The map is right and the deployment differs.  Nothing to fix.
        kind=DEPLOYMENT_FACT,
        question="are the map's code paths running here?",
        finding="the map describes code this deployment never runs",
        # Not a sample: the file either exists on a machine or it does not, and
        # every machine in the service answered.  This is also the only negative
        # claim production supports at all -- see the note.
        failures=failures,
        note=(
            "A log file is created on first emit, so its absence means the path never "
            "ran here -- unless the map named the logger wrongly, which is unlikely "
            f"when {sum(1 for status in answered if status == 'present')} of "
            f"{len(answered)} files it named and asked about do exist.  It is also the "
            "only absence production can prove: writing any other line needs a branch to fire."
        ),
    )


def _stage_files(fragment: MapFragment) -> dict[str, list]:
    """Filter stages grouped by the log file they write to."""
    grouped: dict[str, list] = {}
    for stage in fragment.filter_stages:
        for filename in stage.log_files:
            grouped.setdefault(filename, []).append(stage)
    return grouped


def tags_are_known(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Every rejection tag production emits is a tag the map extracted.

    The strongest direction of the (a) slice's check, and the reason it is
    worth running against production at all: a tag in the log that the map has
    no stage for is a cut the map cannot explain -- a blind spot, stated by the
    system itself rather than inferred.

    It is also the direction that survives an incomplete sample.  Seeing a tag
    proves it exists no matter how much of the log was read; *not* seeing one
    proves nothing unless every matched line came back and was kept.

    **Compared across the whole map, not per log file.**  Per file it looked
    like fourteen blind spots and thirteen were the gate's own fault: shared
    helpers such as ``AtlasBrokerUtils`` declare no logger, so their stages
    carry no log file, and the tags they emit surface in whichever broker's log
    called them.  Grouping by file therefore accuses the map of missing stages
    it has.  The mismatch is still worth knowing -- it points at exactly those
    stages whose log file is unresolved -- so it is counted and reported, not
    turned into a finding about coverage.
    """
    extracted = {s.criteria_tag for s in fragment.filter_stages if s.criteria_tag}
    observed = evidence.observed_tags(ev)
    failures = [
        f"production logs {tag} ({observed[tag]}x) and the map has no stage for it"
        for tag in sorted(set(observed) - extracted)
    ]

    unknown: list[str] = []
    misfiled = 0
    for filename, stages in sorted(_stage_files(fragment).items()):
        if ev.file_status(filename) == "absent":
            continue
        here = evidence.observed_tags(ev, filename)
        mine = {s.criteria_tag for s in stages if s.criteria_tag}
        misfiled += len(set(here) & extracted - mine)
        # The completeness guard is necessary and *not* sufficient, so what it
        # licenses stays in ``inconclusive``.  A mapped tag missing from the log
        # has two causes -- the map's tag is wrong, or that cut simply did not
        # happen in the window -- and no sample size separates them, because
        # writing the line requires the branch to fire.  Narrowing the window
        # until every match fits would not upgrade this direction; it would
        # manufacture findings, and it would cost the positive one, which is
        # where the value is (``-link_unusable`` appeared six times in a wide
        # window and would be missed in a small one).
        if ev.complete(evidence.TAG_PATTERN, log_filename=filename):
            for tag in sorted(mine - set(here)):
                unknown.append(
                    f"{filename}: the map has {tag} and this window shows no cut using it"
                )
    if misfiled:
        unknown.append(
            f"{misfiled} tag/file pair(s) appear in a log no stage of theirs claims "
            "-- those stages' log file is unresolved, not their tag"
        )
    # Absent files are excluded from the sample figure: they answered, and the
    # answer was "this never ran", which is not a partial reading of anything.
    whole, asked = evidence.sample_state(
        ev,
        evidence.TAG_PATTERN,
        [f for f in sorted(_stage_files(fragment)) if ev.file_status(f) != "absent"],
    )
    return GateResult(
        gate="tags-are-known",
        passed=not failures,
        checked=len(observed),
        unit="observed tags",
        question="is every cut production makes in the map?",
        finding="the map is missing a cut production makes",
        sample=_sample_word(whole, asked),
        failures=failures,
        inconclusive=unknown,
        note="A tag with no stage is a cut the map cannot explain -- the system naming its own blind spot.",
    )


def _decodes(fragment: MapFragment) -> dict[str, set[str]]:
    """``{job field: the enumerations that decode it}``, read from the writes.

    A field can have more than one: ``job_label`` takes constants from both
    ``JobUtils.PROD`` and ``JobUtils.ANALY``, and either is a legitimate answer,
    so the index for that field is the union.
    """
    bound: dict[str, set[str]] = {}
    for write in fragment.enumeration_writes:
        bound.setdefault(write.field.split(".")[-1], set()).add(write.namespace)
    return bound


def error_codes_are_known(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Every coded value production puts in a record is one the index decodes.

    The (c) slice's only production check, and the last of the map's indexes to
    meet a real record.  A code sitting in a job field that the index cannot
    name is either a constant the extraction missed or one whose enumeration the
    map bound to the wrong field, and both make the index answer a question
    wrongly rather than not at all -- which is worse, because P2 means to use it
    as a lookup.

    Positive only, like every production check here: seeing ``100`` in
    ``taskBufferErrorCode`` proves the entry is live whatever fraction of
    production was sampled, while a constant the sample lacks may simply not
    have fired.  ``value-enum-referenced`` already asks the other direction of
    the source, where it can be answered.

    **Only the fields the map binds to an enumeration are checked**, and that
    binding is read from the writes rather than declared here.  Three fields
    have one -- the three ``ErrorCode`` modules, one field each.  Of the rest,
    ``pilotErrorCode`` belongs to the pilot, which is a boundary the map
    deliberately does not index; ``exeErrorCode`` carries a transform's exit
    code; ``supErrorCode`` is decoded by a table rather than by constants; and
    ``brokerageErrorCode`` has no declared constants at all.  Checking those
    against this index would report every value they hold as unknown, which
    says nothing about the index.
    """
    decodes = _decodes(fragment)
    index: dict[str, set] = {}
    for enum in fragment.value_enums:
        index.setdefault(enum.namespace, set()).add(enum.value)

    observed = evidence.observed_codes(ev)
    failures: list[str] = []
    checked = 0
    for field, counts in sorted(observed.items()):
        spaces = decodes.get(field)
        if not spaces:
            continue
        known = set().union(*(index.get(space, set()) for space in spaces))
        for value, times in sorted(counts.items(), key=lambda kv: str(kv[0])):
            checked += 1
            if value in known:
                continue
            failures.append(
                f"production sets {field}={value} ({times}x) and "
                f"{', '.join(sorted(spaces))} has no constant with that value"
            )
    sampled = len(ev.records)
    return GateResult(
        gate="error-codes-are-known",
        passed=not failures,
        checked=checked,
        unit="coded values",
        question="is every code production records in the index?",
        finding="the index cannot decode a code production records",
        sample=_sample_word(sampled, max(sampled, ev.tasks_available)),
        failures=failures,
        note="A code the index cannot name makes a lookup answer wrongly, not not at all.",
    )


def _precedence(runs: list[list[int]]) -> dict[tuple[int, int], list[int]]:
    """For each pair of steps, how often each order was observed.

    Returns ``{(lower rank, higher rank): [in map order, reversed]}``.

    Counting orders rather than cutting the log into traversals is what makes
    this answerable at all.  Brokerage walks the chain many times under one
    ``<jediTaskID=... datasetID=...>`` -- thirteen times in one observed run --
    and the log marks no boundary between them, so every attempt to segment it
    was a guess that cost false findings: reading the run as one sequence gave
    22, splitting on a return to the start gave 8, splitting on the size of the
    backward jump gave 2, and all of them were wraps rather than disagreements.

    A majority needs no boundary.  Each wrap contributes one reversed pair
    against many in order, so noise stays a minority, while a step production
    really does run out of place is reversed every time.
    """
    counts: dict[tuple[int, int], list[int]] = {}
    for run in runs:
        seen: Counter = Counter()
        for rank in run:
            for earlier, times in seen.items():
                if earlier == rank:
                    continue
                key = (min(earlier, rank), max(earlier, rank))
                tally = counts.setdefault(key, [0, 0])
                tally[0 if earlier < rank else 1] += times
            seen[rank] += 1
    return counts


def _chains(fragment: MapFragment) -> dict[str, list]:
    """Stages carrying a funnel label, grouped by the chain that owns them."""
    chains: dict[str, list] = {}
    for stage in fragment.filter_stages:
        if stage.funnel_label:
            chains.setdefault(stage.owner, []).append(stage)
    return chains


def _label_owners(chains: dict[str, list]) -> dict[str, set[str]]:
    """``{funnel label: the chains that use it}``.

    Sibling brokers name their steps alike -- ``status check`` is used by three
    of them -- so a label does not identify a chain on its own.
    """
    owners: dict[str, set[str]] = {}
    for owner, stages in chains.items():
        for stage in stages:
            owners.setdefault(stage.funnel_label, set()).add(owner)
    return owners


def _step_names(fragment: MapFragment) -> set[str]:
    """Every step name the map holds, templates included."""
    return {s.funnel_label for s in fragment.filter_stages if s.funnel_label}


def _as_mapped(observed: str, names) -> Optional[str]:
    """The map's name for a step production called *observed*, if it has one.

    Exact first, then the templates: ``AtlasProdTaskBroker`` names one of its
    steps after a threshold it reads from configuration, so production writes
    ``endpoint check with DISK_THRESHOLD=10 TB`` and ``... =1000 TB`` for the
    step the map holds as ``... ={} TB``.  Exact wins so a template can never
    take a name another step already owns.
    """
    if observed in names:
        return observed
    for name in sorted(names):
        if values.has_literal_text(name) and values.template_matches(name, observed):
            return name
    return None


def _positionally_ambiguous(stages: list) -> set[str]:
    """Labels the map places at more than one point in one chain.

    Not the same as a step with several reasons, which is common and fine --
    those stages sit together.  This is the same step written twice on paths
    that exclude each other: ``AtlasProdJobBroker`` runs "temporary problem
    check" early and returns when it was called for a task-brokerage hint, and
    otherwise runs it last.  The map holds both, order takes the first, and
    production mostly runs the other -- 110 observations "out of order" for a
    chain doing exactly what the map says.  A label with two positions cannot
    testify about position, so it is dropped for the same reason a label two
    chains share is.
    """
    sequence = [s.funnel_label for s in sorted(stages, key=lambda s: s.order)]
    runs: Counter = Counter()
    for index, label in enumerate(sequence):
        if index == 0 or sequence[index - 1] != label:
            runs[label] += 1
    return {label for label, times in runs.items() if times > 1}


def _steps_in_order(stages: list, comparable) -> list[str]:
    """One entry per step, in map order, keeping only comparable labels.

    Deduplicated: several stages can share a step -- ``AtlasAnalJobBroker``
    rejects for two reasons under "disk check" -- and the funnel counts steps,
    not stages, so a position has to mean the step.
    """
    ordered: list[str] = []
    for stage in sorted(stages, key=lambda s: s.order):
        if stage.funnel_label in comparable and stage.funnel_label not in ordered:
            ordered.append(stage.funnel_label)
    return ordered


def funnel_steps_are_known(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Every step production counts a cut at is a step the map extracted.

    The funnel counter's positive direction, and the same shape as
    :func:`tags_are_known`: a ``candidates passed <name>`` line production
    writes for a step the map does not hold is a place candidates demonstrably
    disappear and the map has nothing to say about where they went.  Stated by
    the system about itself, and true whatever fraction of the log came back --
    seeing the line proves the step.

    The other direction is not reported, and not because the sample is short.
    A step logs its count only when the chain reaches it, so a name the map has
    and the window lacks may be a stale name or may be a chain that returned
    early; nothing in the log separates those, and neither would a complete
    read.  ``GenJobBroker``'s eight steps are the standing example -- absent
    because that broker does not run in this deployment at all.

    **Across the whole map rather than per file**, for the reason
    :func:`tags_are_known` is: which chains write into which log is something
    the evidence knows and the map does not, so a per-file comparison accuses
    the map of missing the forty-six steps ``AtlasProdJobBroker`` writes
    through the log slot ``AtlasProdTaskBroker`` handed it.
    """
    names = _step_names(fragment)
    files = sorted({f for s in fragment.filter_stages if s.funnel_label for f in s.log_files})
    observed: Counter = Counter()
    for filename in files:
        for run in evidence.observed_runs(ev, filename):
            observed.update(run)
    failures = [
        f"production counts a cut at {label!r} ({observed[label]}x) "
        "and the map has no step for it"
        for label in sorted(observed)
        if _as_mapped(label, names) is None
    ]
    whole, asked = evidence.sample_state(
        ev, evidence.FUNNEL_PATTERN, [f for f in files if ev.file_status(f) != "absent"]
    )
    return GateResult(
        gate="funnel-steps-are-known",
        passed=not failures,
        checked=len(observed),
        unit="observed steps",
        question="is every step production counts a cut at in the map?",
        finding="the map is missing a step production counts a cut at",
        sample=_sample_word(whole, asked),
        failures=failures,
        note="A counted cut with no step is candidates going somewhere the map cannot name.",
    )


def funnel_order_matches(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Production runs each chain in the order the map extracted.

    "Which step cut the candidates" is a question about position, so an order
    that disagrees makes every answer off by one step.  Compared as a
    subsequence rather than an equality: the sample spans many tasks and a
    chain can exit early, so production shows a prefix or a gapped run of the
    map's order, and only a genuine transposition is a finding.

    **Per chain, and which chains a file holds is read off the evidence.**  One
    log file is not one chain: ``AtlasProdTaskBroker`` runs its own three steps
    and then calls the job broker, whose forty-six steps are written through the
    log slot it was handed, so both chains land in
    ``panda-AtlasProdTaskBroker.log``.  Both begin with a step named ``status
    check``, so read as one chain every traversal contributed one pair in order
    and one reversed -- 4836 against 4833, a majority decided by nothing.

    The map cannot say which chains share a file: it records where a stage's own
    module logs, and this delegation is invisible there.  The evidence can, and
    without segmenting anything: a label only one chain uses names that chain,
    so the chains present in a file are the ones whose unique labels appear in
    it, and a label shared by two of *those* is dropped as unattributable.  In
    ``AtlasProdTaskBroker``'s log that costs exactly ``status check``; in the
    single-chain files it costs nothing, because a label shared with a sibling
    broker that does not write there is not ambiguous here.

    **A lone reversed observation is one traversal's worth of noise.**  Wrapping
    is not a rare artefact here -- pairs the map orders correctly still come
    back 4440 reversed against 6119 in order -- and every traversal that wraps
    contributes exactly one reversed observation.  So a pair whose whole
    evidence is a single reversal is indistinguishable from one wrap, and there
    is no majority in it to speak of.  The corpus makes the line easy: of 624
    pairs, 604 are seen ten times or more and the rest are seen once or twice.
    """
    chains = _chains(fragment)
    label_owners = _label_owners(chains)
    names = _step_names(fragment)
    failures: list[str] = []
    checked = 0
    files = sorted({f for stages in chains.values() for s in stages for f in s.log_files})
    for filename in files:
        # Production writes the step's name with its run-time detail filled in;
        # the map holds the frame.  Resolving here keeps everything downstream
        # comparing one vocabulary.
        runs = [
            [name for label in run if (name := _as_mapped(label, names)) is not None]
            for run in evidence.observed_runs(ev, filename)
        ]
        observed = {label for run in runs for label in run}
        present = {
            owner
            for label in observed
            if len(label_owners.get(label, ())) == 1
            for owner in label_owners[label]
        }
        for owner in sorted(present):
            ambiguous = _positionally_ambiguous(chains[owner])
            comparable = {
                label
                for label in label_owners
                if len(label_owners[label] & present) == 1
                and owner in label_owners[label]
                and label not in ambiguous
            }
            expected = _steps_in_order(chains[owner], comparable)
            rank = {label: index for index, label in enumerate(expected)}
            ranked = [[rank[label] for label in run if label in rank] for run in runs]
            for (lower, higher), (in_order, reversed_) in sorted(_precedence(ranked).items()):
                checked += 1
                if reversed_ > in_order and reversed_ > 1:
                    failures.append(
                        f"{filename} ({owner.split('::')[-1]}): production logs "
                        f"{expected[lower]!r} after {expected[higher]!r} in {reversed_} "
                        f"of {in_order + reversed_} observations, the map has it before"
                    )
    whole, asked = evidence.sample_state(
        ev,
        evidence.FUNNEL_PATTERN,
        [f for f in files if ev.file_status(f) != "absent"],
    )
    return GateResult(
        gate="funnel-order-matches",
        passed=not failures,
        checked=checked,
        unit="step pairs",
        question="does production run the chain in map order?",
        finding="production runs a chain step out of the order the map has",
        sample=_sample_word(whole, asked),
        failures=failures,
        note="Which step cut the candidates is a question about position.",
    )


def transitions_are_explained(fragment: MapFragment, ev: "evidence.Evidence") -> GateResult:
    """(ii) Every task status production sets is one the map can produce.

    Conformance checking, with the map as the model and the knights' own log as
    the trace.  The strongest of these checks in principle, because it depends
    on nothing about how the code is written: a status the system was observed
    to enter and the branch tables cannot produce is a blind spot stated by the
    system itself.

    The database cannot supply this -- ``JEDI_Tasks`` keeps the current status
    and the one before it, so a *sequence* only exists in the log, and the
    knights write one: ``set task_status=<value>`` at twelve sites in eight
    files, all of them places the map already holds as junctions.

    **What a bounded sample supports here, and what it does not.**  Seeing a
    value proves the system produced it, so the positive direction stands
    however little was read.  The pairs do not: a step the sample missed leaves
    its neighbours adjacent, and ``a -> c`` then looks like a transition the
    code makes, with nothing in the log marking the gap.  So the gate reads
    values and the pairs are reported.

    Two independent readings decide how hard a miss is.  A value the code
    *declares* and production *sets*, with no writer the map resolved, is a
    missing write form: both sources agree the value is real, which is what
    ``declared-status-is-written`` could not settle on its own -- it can only
    say "a missing write form, or a stale declaration".  An undeclared value
    could be one a run-time writer computes, which the map records as tier 2 by
    design, so that stays inconclusive with the count of writers that could
    account for it.
    """
    histories = evidence.observed_task_status(ev)
    seen = Counter(status for rows in histories.values() for _, status, _ in rows)
    subject = evidence.TRANSITION_SUBJECT
    junctions = [j for j in fragment.junctions if j.subject == subject]
    resolved = {b.outcome for j in junctions for b in j.branches if b.tier == 1}
    at_runtime = sum(1 for j in junctions for b in j.branches if b.tier != 1)
    declared: set[str] = set()
    selected: set[str] = set()
    for node in fragment.subjects:
        if node.name == subject:
            declared = set(node.vocabulary)
            selected = set(node.selected_values)

    failures: list[str] = []
    unknown: list[str] = []
    for status, times in sorted(seen.items()):
        if status in resolved:
            continue
        if status in declared:
            failures.append(
                f"production sets {status} ({times}x), the code declares it, and no "
                f"branch in the map produces it ({at_runtime} writer(s) of this "
                "subject decide the value at run time)"
            )
        else:
            unknown.append(
                f"production sets {status} ({times}x) and no branch in the map produces "
                f"it, but nothing declares it either -- it may be a value one of the "
                f"{at_runtime} run-time writer(s) computes"
            )
    # The read side, and it cannot be a failure: a task can be moved on by a
    # junction that selects it by id or on a command, with no status predicate
    # for the map to have missed.
    for status in sorted(evidence.observed_departures(histories)):
        if status not in selected:
            unknown.append(
                f"production moved tasks out of {status} and no query in the map "
                "selects on it"
            )
    whole, asked = evidence.sample_state(
        ev,
        evidence.TRANSITION_PATTERN,
        sorted(f for f in ev.log_filenames() if ev.file_status(f) == "present"),
    )
    return GateResult(
        gate="transitions-are-explained",
        passed=not failures,
        checked=len(seen),
        unit="statuses seen",
        question="can the map produce every status production set?",
        finding="production puts tasks in a status the map cannot produce",
        sample=_sample_word(whole, asked),
        failures=failures,
        inconclusive=unknown,
        note=(
            "Independent of how the code is written, which is what makes it the last "
            "line against a blind spot both the recognizer and the writer census miss. "
            "A failure here narrows declared-status-is-written, which can only say "
            "'a missing write form, or a stale declaration': production setting the "
            "value rules the second out."
        ),
    )


def _stage_label(stage, filename: str) -> str:
    """A stage's identity as the map holds it: the step's name and its tag.

    The tag alone will not do.  ``AtlasAnalJobBroker`` emits ``-disk`` from both
    its "scratch disk check" and its "storage space check", so a report keyed on
    the tag prints one row twice and identifies neither.
    """
    parts = [part for part in (stage.funnel_label, stage.criteria_tag) if part]
    return f"{' '.join(parts)} ({filename})"


def templates_confirmed(fragment: MapFragment, ev: "evidence.Evidence") -> list[str]:
    """The map's diagnostic lines production was seen to write, verbatim.

    One-sided, and that is not a limitation of the sample -- it is the only
    direction that exists.  A template missing from the log has two causes that
    nothing in the log distinguishes: the wording has moved on
    (``AtlasProdTaskBroker``'s space check went from "free ... reserved" to
    "usable ... projected demand"), or that branch simply did not fire in the
    window.  Writing the line *requires* the branch to fire, so no amount of
    reading separates them; a complete sample would license a claim just as
    false as a cut one.

    Which is why the unconfirmed half is not returned at all.  It was reported
    once, as "40 of 92 confirmed" with the other 52 listed, and that is the
    useless half: a reader cannot act on it, and the ratio invites reading a
    rare rejection as a stale template.

    What the confirmed half buys is narrower and real: each line found verbatim
    is one piece of evidence that the deployed code and the map agree at that
    point, which is the only production check on version skew there is.  It
    also marks the templates an investigation can rely on.  Where drift can be
    detected properly is between two source trees, offline, which is
    ``diff-map``'s job -- and it has already caught some.
    """
    confirmed: list[str] = []
    for filename, stages in sorted(_stage_files(fragment).items()):
        haystack = "\n".join(ev.lines(evidence.TAG_PATTERN, log_filename=filename))
        if not haystack:
            continue
        for stage in stages:
            for template in stage.emits:
                stem = _template_stem(template)
                if stem and stem in haystack:
                    confirmed.append(_stage_label(stage, filename))
    return confirmed


# A template's longest run of fixed words.  Taking the prefix instead was the
# obvious choice and the wrong one: ``"  skip site={} due to disk shortage
# criteria=-disk"`` begins with ten characters shared by half the messages in
# the file, so a prefix rule either matched everything or, with a floor high
# enough to be distinctive, skipped nearly every template there is.  What
# identifies the message is on the other side of the interpolation.
_MIN_STEM = 12
_FIELD = re.compile(r"\{[^{}]*\}")


def _template_stem(template: str) -> Optional[str]:
    """The longest literal fragment, or None when none is distinctive enough."""
    fragments = [part.strip() for part in _FIELD.split(template)]
    longest = max(fragments, key=len, default="")
    return longest if len(longest) >= _MIN_STEM else None


def run_production(fragment: MapFragment, ev: "evidence.Evidence") -> list[GateResult]:
    """Run the gates that need production evidence.

    Separate from ``run_all`` because the two have different preconditions and
    different failure meanings: a code-internal gate fails when the extraction
    or the source disagrees with itself, a production gate fails when the map
    and the running system have drifted apart.
    """
    results = [log_format_recognised(ev)]
    # Liveness first: it decides how the level verdicts should be read.
    results.append(code_paths_are_live(fragment, ev))
    if fragment.filter_stages:
        results.append(observables_are_emitted(fragment, ev))
        results.append(tags_are_known(fragment, ev))
        results.append(funnel_steps_are_known(fragment, ev))
        results.append(funnel_order_matches(fragment, ev))
    if ev.matching(evidence.TRANSITION_PATTERN):
        results.append(transitions_are_explained(fragment, ev))
    if ev.records and fragment.enumeration_writes:
        results.append(error_codes_are_known(fragment, ev))
    return results


def run_all(fragment: MapFragment) -> list[GateResult]:
    """Run every code-internal gate that applies to *fragment*."""
    results: list[GateResult] = []
    if fragment.value_enums:
        results.append(value_enum_referenced(fragment))
        results.append(namespace_disambiguates(fragment))
    if fragment.boundaries:
        results.append(boundary_ownership_param_declared(fragment))
    if fragment.junctions:
        results.append(structural_attribution_agrees(fragment))
        results.append(declared_status_is_written(fragment))
        results.append(map_references_resolve(fragment))
    if fragment.declaration_yields:
        results.append(spec_declarations_are_read(fragment))
    if fragment.annotation_readings:
        results.append(container_annotations_agree(fragment))
        results.append(annotations_are_read(fragment))
    results.append(map_identities_are_distinct(fragment))
    return results


def slice_totals(fragment: MapFragment) -> dict[str, Counter]:
    """Aggregate candidates/explained per slice, for the build summary."""
    totals: dict[str, Counter] = {}
    for stat in fragment.coverage:
        bucket = totals.setdefault(stat.slice_name, Counter())
        bucket["candidates"] += stat.candidates
        bucket["explained"] += stat.explained
        bucket["files"] += 1
    return totals
