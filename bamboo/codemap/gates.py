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

# ``passthrough(JediTaskSpec.oldStatus)`` -- the subject a branch copies from.
_PASSTHROUGH = re.compile(r"^passthrough\((.+)\)$")


class GateResult(BaseModel):
    """Outcome of one gate."""

    gate: str
    passed: bool
    checked: int = 0
    failures: list[str] = Field(default_factory=list)
    inconclusive: list[str] = Field(default_factory=list)
    note: Optional[str] = None

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        parts = []
        if self.failures:
            parts.append(f"{len(self.failures)} issue(s)")
        if self.inconclusive:
            parts.append(f"{len(self.inconclusive)} inconclusive")
        detail = f" ({', '.join(parts)})" if parts else ""
        return f"[{status}] {self.gate}: {self.checked} checked{detail}"


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
        failures=sorted(set(failures)),
        note="A backward walk that steps onto a missing node cannot report that it did.",
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
        sampled = sum(len(r.lines) for r in results)
        if sampled and not evidence.level_histogram(ev, log_filename=filename):
            failures.append(
                f"{filename}: sampled {sampled} line(s), none of which carry a "
                f"level -- the log format is not {evidence.ANY_LINE_PATTERN!r}"
            )
    return GateResult(
        gate="log-format-recognised",
        passed=not failures,
        checked=checked,
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
        label = f"{stage.criteria_tag or stage.funnel_label} at {where}"
        if not stage.log_files:
            unknown.append(f"{label}: the source does not name a log file")
            continue
        # Any candidate that carries the line is enough: the proxy mixins run
        # under two processes and emitting in either one makes the observable
        # real.  Absent everywhere is reported by ``code_paths_are_live``, not
        # here -- a path that never ran is not a broken promise about logging.
        verdicts = []
        for filename in stage.log_files:
            if ev.file_status(filename) != "present":
                continue
            threshold = evidence.effective_level(ev, log_filename=filename)
            if threshold is None:
                continue
            verdicts.append(
                (filename, threshold, evidence.below_threshold(stage.log_level, threshold))
            )
        if not verdicts:
            unknown.append(f"{label}: no sample from {', '.join(stage.log_files)}")
        elif all(suppressed for _, _, suppressed in verdicts):
            detail = ", ".join(f"{name} at {level}" for name, level, _ in verdicts)
            failures.append(f"{label} emits at {stage.log_level.upper()} but {detail}")
    return GateResult(
        gate="observables-are-emitted",
        passed=not failures,
        checked=checked,
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
    owners: dict[str, set[str]] = {}
    for node in list(fragment.filter_stages) + list(fragment.junctions):
        for filename in node.log_files:
            owners.setdefault(filename, set()).add(node.owner.split("::")[0])
    failures = [
        f"{filename} is on no machine: " + ", ".join(sorted(modules)[:3]) + " never ran"
        for filename, modules in sorted(owners.items())
        if ev.file_status(filename) == "absent"
    ]
    return GateResult(
        gate="code-paths-are-live",
        passed=not failures,
        checked=len(owners),
        failures=failures,
        note="A log file is created on first emit, so its absence means the path never ran here.",
    )


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
