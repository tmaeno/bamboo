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

from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from bamboo.codemap.models import MapFragment


class GateResult(BaseModel):
    """Outcome of one gate."""

    gate: str
    passed: bool
    checked: int = 0
    failures: list[str] = Field(default_factory=list)
    note: Optional[str] = None

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = f" ({len(self.failures)} issue(s))" if self.failures else ""
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
