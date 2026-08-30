"""Which attributes are worth asking "why is it this?" about.

The spec classes declare 421 attributes and the tables add more, but most are
identifiers, timestamps and counters.  A map that made every one of them a
subject would answer "where is this written" for things nobody investigates,
and bury the dozen that matter.

Three criteria promote, and at least one must fire.  They were derived by
census over the whole corpus, and each was narrowed by watching something
absurd come out on top:

**1. Gated by a SQL selection predicate.**  ``WHERE t.status IN ('ready',
'running')`` means the field decides whether another component proceeds, which
is what makes "why is it this?" a question worth asking.  The narrowing was
threefold -- ``WHERE PandaID=:PandaID`` is a lookup key, ``WHERE
t.jediTaskID=f.jediTaskID`` is a join, and a quote inside ``IN (SELECT ...)``
belongs to the subquery -- and each was found because ``lfn``, then
``jediTaskID``, ranked first.

**2. A declared vocabulary.**  The rarest and strongest: the class states the
value set outright.

**3. A closed set of literals dominates the writes.**  Existence of two
literals is not enough -- ``jediTaskID`` has 528 writes of which a few are
literal -- so the literal writes must be a majority.

Two further signals (the field is mentioned in logs; it is written under a
guard) fire on 69% and 79% of everything and so corroborate rather than
promote.

A fourth criterion promotes nothing on its own: it closes the set under the
passthrough edge, so a subject a promoted one copies its value from comes along.
Without it the backward walk's first hop out of ``JediTaskSpec.status`` lands on
a subject the map does not contain.

A fifth restates the first where the first cannot see.  Criterion 1 reads SQL
predicates, and brokerage gates in Python: ``nucleus``, ``minRamCount``,
``ioIntensity`` and ``maxwdir`` decide whether a site survives the filter chain
without appearing in any ``WHERE``.  Taking them from the selection slice keeps
the evidence as strong -- the code tagged the rejection and counted the cut.

Applied to junctions as well as subjects: a junction writing an attribute
nobody investigates is the same noise one level down.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

from bamboo.codemap.models import MapFragment, SourceModule
from bamboo.codemap.panda.attribution import UNRESOLVED

WHERE_GATE = "1:state-gate-in-where"
DECLARED_VOCABULARY = "2:declared-vocabulary"
CLOSED_LITERAL_SET = "3:closed-literal-set"
PASSTHROUGH_SOURCE = "4:carried-into-a-promoted-subject"
FILTER_GATE = "5:gates-a-filter-stage"

# ``passthrough(JediTaskSpec.oldStatus)`` -- the subject a branch carries its
# value from.
_PASSTHROUGH = re.compile(r"^passthrough\((.+)\)$")

# Share of a subject's outcomes that must be literals for the set to count as
# closed.  A majority keeps identifiers out while tolerating the passthrough
# and computed writes real status fields also have.
CLOSED_SET_SHARE = 0.5
# Two values are a set; forty are a free-form field wearing one.
_CLOSED_SET_RANGE = range(2, 41)

# ``WHERE status='ready'`` / ``AND type IN ('input','pseudo_input')``.  The
# literal is what distinguishes a state gate from a lookup: a predicate against
# a bind variable selects a row, a predicate against literals selects a state.
_PREDICATE = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s*"
    r"(?:(?:=|<>|!=|<|>)\s*'[^']*'|\bIN\b\s*\((?P<inlist>[^)]*)\))",
    re.IGNORECASE,
)


def gated_fields(modules: list[SourceModule]) -> Counter:
    """Count fields a SQL predicate tests against literal values."""
    counts: Counter = Counter()
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if "WHERE" not in node.value.upper():
                continue
            for match in _PREDICATE.finditer(node.value):
                inlist = match.group("inlist")
                if inlist is not None:
                    if "'" not in inlist:
                        continue  # ``IN (:a,:b)`` -- a lookup, not a state gate.
                    if "SELECT" in inlist.upper():
                        continue  # The quote belongs to the subquery.
                counts[match.group(1)] += 1
    return counts


def filter_gated_fields(fragment: MapFragment) -> set[str]:
    """Return the attributes a filter stage's exclusion condition reads.

    Criterion 1 asks whether a field decides that another component proceeds,
    and answers it from SQL: ``WHERE t.status IN ('ready','running')``.  That
    misses every field brokerage gates, because brokerage gates in Python --
    ``nucleus``, ``coreCount``, ``minRamCount``, ``ioIntensity``, ``walltime``,
    ``maxwdir``, ``pledgedCPU`` decide whether a site survives the chain and
    none of them appears in a predicate.

    Reading them from the selection slice rather than from Python conditions at
    large is what makes this evidence instead of a guess: a census of names
    compared against literals in an ``if`` matched 72 known column names over
    540 sites, led by ``key``, ``name`` and ``value``.  A filter stage's
    condition is different in kind -- the code tagged the rejection and counted
    the cut, so it has stated that this test excludes candidates.
    """
    return {name for stage in fragment.filter_stages for name in stage.inputs}


def criteria_for(
    fragment: MapFragment,
    gated: Counter,
    vocabularies: dict[tuple[str, str], set[str]],
) -> dict[str, list[str]]:
    """Return ``{subject name: criteria satisfied}`` for every candidate.

    Criterion 3 reads the extracted junctions rather than rescanning the
    source: they already hold every outcome of every write site, which is
    exactly the evidence -- and it means a subject reached through SQL is
    judged by the same rule as one reached through an attribute.
    """
    literal = Counter()
    total = Counter()
    values: dict[str, set[str]] = {}
    for junction in fragment.junctions:
        for branch in junction.branches:
            total[junction.subject] += 1
            if branch.tier == 1:
                literal[junction.subject] += 1
                values.setdefault(junction.subject, set()).add(branch.outcome)

    filtered = filter_gated_fields(fragment)
    found: dict[str, list[str]] = {}
    for subject in fragment.subjects:
        criteria: list[str] = []
        if gated.get(subject.attribute):
            criteria.append(WHERE_GATE)
        if subject.attribute in filtered:
            criteria.append(FILTER_GATE)
        if vocabularies.get((subject.spec_class, subject.attribute)):
            criteria.append(DECLARED_VOCABULARY)
        seen = values.get(subject.name, set())
        written = total.get(subject.name, 0)
        if (
            len(seen) in _CLOSED_SET_RANGE
            and written
            and literal[subject.name] / written >= CLOSED_SET_SHARE
        ):
            criteria.append(CLOSED_LITERAL_SET)
        if criteria:
            found[subject.name] = criteria
    return found


def close_over_passthrough(
    fragment: MapFragment, criteria: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Promote the subjects a promoted subject carries its value from.

    ``UPDATE JEDI_Tasks SET status=oldStatus`` is how a task leaves ``pending``,
    so the branch reads ``passthrough(JediTaskSpec.oldStatus)``.  On its own
    ``oldStatus`` satisfies no criterion -- no predicate compares it to a
    literal, and every write to it is a copy or a ``NULL`` -- so it would be
    dropped, and the branch would point at a subject the map does not contain.

    That is not a tidy gap.  The backward walk exists to answer "why is it this
    value", and its very first hop out of the flagship symptom would land on
    nothing.  A subject that decides a promoted one is worth asking about by the
    same argument that promoted the first, so the set is closed under the edge.
    """
    promoted = dict(criteria)
    while True:
        added = False
        for junction in fragment.junctions:
            if junction.subject not in promoted:
                continue
            for branch in junction.branches:
                match = _PASSTHROUGH.match(branch.outcome)
                if match is None or match.group(1) in promoted:
                    continue
                promoted[match.group(1)] = [PASSTHROUGH_SOURCE]
                added = True
        if not added:
            return promoted


def apply(fragment: MapFragment, criteria: dict[str, list[str]]) -> tuple[int, int]:
    """Keep only promoted subjects and the junctions that write them.

    Returns ``(subjects dropped, junctions dropped)``.  Dropping is the point:
    an unpromoted subject is not a gap in extraction, it is a field nobody
    investigates, and reporting it as coverage would make the map look larger
    and less useful at once.
    """
    before = (len(fragment.subjects), len(fragment.junctions))
    for subject in fragment.subjects:
        subject.criteria = criteria.get(subject.name, subject.criteria)
    fragment.subjects = [s for s in fragment.subjects if s.name in criteria]
    fragment.junctions = [
        junction
        for junction in fragment.junctions
        # A junction whose subject could not be settled is kept regardless: it
        # has no subject to judge, and dropping it would turn a known and
        # reported gap into a silent one.
        if junction.subject in criteria or junction.attribution == UNRESOLVED
    ]
    return before[0] - len(fragment.subjects), before[1] - len(fragment.junctions)
