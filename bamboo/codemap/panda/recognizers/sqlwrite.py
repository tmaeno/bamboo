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
    attributed: set[tuple[str, str]] = set()
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
                        # Not every table holds a spec.  ``worker_node_gpus``,
                        # ``ddm_endpoint`` and DEFT's ``T_TASK`` are real tables
                        # with no spec class, so writes to them are out of the
                        # slice rather than missed by it -- counting them as
                        # candidates would report a permanent 60% gap that no
                        # amount of work could close.  They are listed by
                        # ``gates.tables_without_a_spec`` instead.
                        uncovered.add(statement.table)
                        continue
                    for column, key in statement.columns.items():
                        if key is None:
                            # ``stateChangeTime=CURRENT_DATE``: a write, but the
                            # value is in the statement and carries no branch.
                            continue
                        candidates += 1
                        attribute = _declared_spelling(attributor, spec_class, column)
                        if attribute is None:
                            # The table holds this spec but the column is not a
                            # declared attribute -- a join key or a housekeeping
                            # column.  Not a subject.
                            continue
                        binds = sql.bound_values(func, varmap, key)
                        if not binds:
                            continue
                        explained += 1
                        attributed.add((spec_class, attribute))
                        _record(
                            junctions,
                            map_id=map_id,
                            derived_from=derived_from,
                            module=module,
                            func=func,
                            spec_class=spec_class,
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
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=["sql-write"],
        )
        for spec_class, attribute in sorted(attributed)
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

