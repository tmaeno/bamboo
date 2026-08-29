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
