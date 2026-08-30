"""SQL write recognizer -- where the code settles a subject's value in the database.

The dominant write form in PanDA, and the one the attribute slice does not see.
A knight decides and a ``db_proxy_mods`` method writes, so most task and job
state changes reach the database as a bind rather than as an attribute
assignment::

    sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
    sqlU += "SET status=:status,modificationTime=:updateTime "
    sqlU += " WHERE jediTaskID=:jediTaskID "
    varMap[":status"] = "tobroken"
    self.cur.execute(sqlU + comment, varMap)

Two things had to be settled before this could be read at all, and both are in
``sql`` and ``attribution`` rather than here:

* which statement a bind belongs to -- the ``execute`` call pairs them, and
  the ``SET`` clause is what distinguishes a write from a ``WHERE`` predicate
  using the identically named bind;
* which spec class a table holds -- nothing declares it, so it is inferred
  from the column names and pooled per table.

The junction is anchored at the **bind assignment**, not at the ``execute``:
the bind is where the value is decided and where the surrounding ``if`` says
why, which is what a reader following the map back needs to see.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    path_condition,
)

SLICE_NAME = "sql-write"


def _declared_spelling(
    attributor: SpecAttributor, spec_class: str, column: str
) -> Optional[str]:
    """Return the attribute as the spec declares it, matching case-insensitively.

    SQL and Python disagree on case for the same field -- ``modificationTime``
    is written ``modificationtime`` in statements -- and a subject keyed on the
    SQL spelling would never join with one keyed on the attribute spelling,
    quietly splitting one subject in two.
    """
    declared = attributor.declared_attributes(spec_class)
    lowered = column.lower()
    for attribute in declared:
        if attribute.lower() == lowered:
            return attribute
    return None


def _subject_of(
    attributor: SpecAttributor,
    spec_class: Optional[str],
    table: str,
    column: str,
) -> tuple[str, Optional[str], str]:
    """Return ``(qualifier, attribute, qualifier kind)`` for a written column.

    A subject's key needs a qualifier that disambiguates -- ``status`` means
    eight different things unqualified -- and a *table* qualifies as well as a
    class does.  For a SQL write it is arguably the better name: the class is
    inferred from the columns, while the table is what the statement says.

    **The class stays canonical wherever one exists**, though, because
    ``jobsActive4``, ``jobsDefined4`` and ``jobsArchived4`` are one
    ``JobSpec.jobStatus`` split across a job's lifetime.  Keying on the table
    would make three subjects out of one, and would file an attribute write and
    a SQL write to the same field under different names.

    Where no class exists the table qualifies instead, which is how
    ``ddm_endpoint.blacklisted`` -- the writer behind a blacklisted RSE, and a
    root cause the map is supposed to reach -- becomes reachable at all.  What
    keeps the bookkeeping out is promotion, not the absence of a spec class.
    """
    if spec_class is not None:
        attribute = _declared_spelling(attributor, spec_class, column)
        if attribute is not None:
            return spec_class, attribute, "spec"
        # The table holds this spec but the column is not a declared attribute
        # -- a join key or a housekeeping column.  It is still a column of a
        # real table, so it is qualified by the table like any other.
    return table, column, "table"


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    attributor: SpecAttributor,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat], set[str]]:
    """Extract junctions for SQL bind writes.

    Returns the tables holding no spec alongside the usual three, so the
    build can say what the map does not cover instead of burying it in a
    coverage ratio.

    *attributor* is passed in already taught: the table map is learned from the
    whole corpus, so it cannot be built from the module in hand.
    """
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str, str]] = set()
    uncovered: set[str] = set()

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            for text, varmap, _call in sql.executions(func):
                if varmap is None:
                    continue
                for statement in sql.writes(text):
                    spec_class = attributor.class_for_table(statement.table)
                    if spec_class is None:
                        uncovered.add(statement.table)
                    for column, key in statement.columns.items():
                        if key is None:
                            # ``stateChangeTime=CURRENT_DATE``: a write, but the
                            # value is in the statement and carries no branch.
                            continue
                        candidates += 1
                        qualifier, attribute, kind = _subject_of(
                            attributor, spec_class, statement.table, column
                        )
                        if attribute is None:
                            continue
                        binds = sql.bound_values(func, varmap, key)
                        if not binds:
                            continue
                        explained += 1
                        attributed.add((qualifier, attribute, kind))
                        _record(
                            junctions,
                            map_id=map_id,
                            derived_from=derived_from,
                            module=module,
                            func=func,
                            spec_class=qualifier,
                            attribute=attribute,
                            binds=binds,
                        )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )

    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(qualifier, attribute),
            spec_class=qualifier,
            qualifier_kind=kind,
            attribute=attribute,
            criteria=["sql-write"],
        )
        for qualifier, attribute, kind in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage, uncovered


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec_class: str,
    attribute: str,
    binds: list[ast.Assign],
) -> None:
    """Add one branch per bind assignment to this write site's junction."""
    subject = SubjectNode.make_name(spec_class, attribute)
    owner = f"{module.rel_path}::{func.name}"
    name = JunctionNode.make_name(map_id, subject, owner)

    junction = junctions.get(name)
    if junction is None:
        junction = JunctionNode(
            map_id=map_id,
            derived_from=derived_from,
            name=name,
            subject=subject,
            owner=owner,
            # The table settled the class, and a table holds one kind of row.
            attribution="certain",
            anchor=Anchor(
                package=module.package,
                file=module.rel_path,
                line_start=binds[0].lineno,
                line_end=binds[-1].end_lineno,
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction

    known = {(branch.outcome, tuple(branch.path_condition)) for branch in junction.branches}
    for bind in binds:
        value = bind.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            outcome, tier = value.value, 1
        else:
            # The writer is known, the value is not until run time.  Recorded
            # rather than dropped: localize and prune read observed values, so
            # they work from the writer alone.
            outcome, tier = f"runtime({ast.unparse(value)})", 2
        condition = path_condition(bind)
        if (outcome, tuple(condition)) in known:
            continue
        known.add((outcome, tuple(condition)))
        junction.branches.append(
            Branch(
                outcome=outcome,
                path_condition=condition,
                order=len(junction.branches),
                tier=tier,
            )
        )

