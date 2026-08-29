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


def outcome_in_declared_vocabulary(fragment: MapFragment) -> GateResult:
    """(i) Every extracted outcome appears in its subject's declared vocabulary.

    The spec classes state their status sets outright (``statusToReassign()``
    and friends), and the junctions state what the code actually writes.  Both
    describe the same vocabulary, so a value in one and not the other means the
    declaration has fallen behind the code or the extraction read the wrong
    write -- and which of those it is cannot be settled from here, only pointed
    at.

    Subjects with no declared vocabulary are skipped rather than failed: having
    nothing to compare against is not a disagreement.
    """
    vocabularies = {s.name: set(s.vocabulary) for s in fragment.subjects if s.vocabulary}
    failures: list[str] = []
    checked = 0
    for junction in fragment.junctions:
        vocabulary = vocabularies.get(junction.subject)
        if not vocabulary:
            continue
        for branch in junction.branches:
            checked += 1
            if branch.outcome not in vocabulary:
                where = junction.anchor.as_ref() if junction.anchor else junction.owner
                failures.append(f"{junction.subject} = {branch.outcome!r} ({where})")
    return GateResult(
        gate="outcome-in-vocabulary",
        passed=not failures,
        checked=checked,
        failures=sorted(set(failures)),
        note="The declared status sets and the writes should describe one vocabulary.",
    )


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
        if b.carried_values
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
        results.append(outcome_in_declared_vocabulary(fragment))
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
