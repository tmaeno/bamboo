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

The junction is anchored where the value is *decided*, not at the ``execute``:
that is where the surrounding ``if`` says why, which is what a reader following
the map back needs to see.  Which statement that is depends on the form -- a
bind is decided at the Python assignment filling it, a value written inline at
the fragment that appended it to the statement.

Three forms reach a subject, and the third was the plan's named blind spot::

    SET status=:status      # decided in Python
    SET status='ready'      # decided in the statement
    SET status=oldStatus    # carried from another column

The last one is how a task comes back out of ``pending``.  Missing it did not
merely lose a write: it removed "the release path never fired" from the
candidate causes of a task stuck in ``pending``, which is one of the two
symptoms this map exists to explain.

``selected_values`` reads the same statements the other way, for the graph
invariants: a ``WHERE`` clause says which rows were asked for, and a status
nothing selects on is a status nothing moves a task out of.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    DiagnosticTemplate,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import sql, values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    literal_values,
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
) -> tuple[str, str, str]:
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
) -> tuple[
    list[SubjectNode],
    list[JunctionNode],
    list[CoverageStat],
    set[str],
    list[DiagnosticTemplate],
]:
    """Extract junctions for SQL bind writes.

    Returns the tables holding no spec alongside the usual three, so the
    build can say what the map does not cover instead of burying it in a
    coverage ratio, and the diagnostic templates bound into statements -- which
    belong here rather than in a scan of their own because the column a bind
    lands in is the join this slice has already made.

    *attributor* is passed in already taught: the table map is learned from the
    whole corpus, so it cannot be built from the module in hand.
    """
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str, str]] = set()
    uncovered: set[str] = set()
    diagnostics: list[DiagnosticTemplate] = []
    settle = values.resolver(values.declared_mappings(modules))

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            # A method that runs one statement from two places yields the same
            # run twice.  Counting both would inflate the coverage denominator
            # with duplicates and make the slice look worse than it reads.
            seen: set[tuple[str, str, Optional[str]]] = set()
            for run in sql.executions(func):
                if (run.variable, run.sql, run.varmap) in seen:
                    continue
                seen.add((run.variable, run.sql, run.varmap))
                for statement in sql.writes(run.sql):
                    spec_class = attributor.class_for_table(statement.table)
                    if spec_class is None:
                        uncovered.add(statement.table)
                    for column, supplied in statement.columns.items():
                        candidates += 1
                        qualifier, attribute, kind = _subject_of(
                            attributor, spec_class, statement.table, column
                        )
                        templates: list[tuple[str, ast.stmt]] = []
                        outcomes = _outcomes(
                            attributor,
                            func=func,
                            run=run,
                            statement=statement,
                            column=column,
                            supplied=supplied,
                            spec_class=spec_class,
                            settle=settle,
                            templates=templates,
                        )
                        diagnostics.extend(
                            DiagnosticTemplate(
                                map_id=map_id,
                                derived_from=derived_from,
                                template=template,
                                field=SubjectNode.make_name(qualifier, attribute),
                                form="bind",
                                anchor=Anchor(
                                    package=module.package,
                                    file=module.rel_path,
                                    line_start=node.lineno,
                                    line_end=node.end_lineno,
                                    blob_sha=module.blob_sha,
                                ),
                            )
                            for template, node in templates
                        )
                        if not outcomes:
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
                            outcomes=outcomes,
                            # Every predicate the statement makes about a column
                            # it also writes, not only the one for this column:
                            # any of them failing leaves the row untouched, so
                            # they bound this write as much as its own does.
                            row_precondition=sorted(statement.preconditions.values()),
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
    return subjects, list(junctions.values()), coverage, uncovered, diagnostics


def _deciding_fragment(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    variable: str,
    column: str,
    supplied: sql.ColumnValue,
) -> Optional[ast.stmt]:
    """Return the statement fragment that put ``<column>=<value>`` into the SQL.

    For a value written inline there is no bind assignment to anchor at, and
    anchoring at the ``execute`` instead would throw away the reason: the two
    forms of ``getTasksToExecCommand_JEDI``'s update are appended in the two
    arms of one ``if``, so the fragment carries the condition and the statement
    does not.
    """
    for node, text in sql.fragments(func, variable):
        for found, value in sql.assigns_in(text):
            if found == column and value == supplied:
                return node
    return None


def _outcomes(
    attributor: SpecAttributor,
    *,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    statement: sql.SqlWrite,
    column: str,
    supplied: sql.ColumnValue,
    spec_class: Optional[str],
    settle,
    templates: list[tuple[str, ast.stmt]],
) -> list[tuple[str, int, ast.stmt, list[str]]]:
    """Return ``(outcome, tier, node, extra conditions)`` for one written column.

    The node is where the value was decided, which differs by form: a bind is
    decided at the Python assignment filling it, and a literal or a copied
    column at the fragment that put it in the statement.

    The extra conditions are for the one form where the deciding node is not the
    whole story.  A bind filled from a local is decided twice over -- the guards
    that reached ``varMap[":status"] = newTaskStatus`` say the write happened,
    and the guards on the assignment that gave the local its value say which
    value.  Both are needed, and neither is derivable from the other's node.

    Diagnostic templates are appended to *templates* rather than returned: they
    are not outcomes, and the caller files them against the column rather than
    the subject.  Collected here because *which column the text lands in* is the
    join this slice has already made -- on its own the bind is called
    ``:errDiag`` and could belong to any statement in the method.
    """
    if supplied.kind == "bind":
        if run.varmap is None:
            return []
        found = []
        for bind in sql.bound_values(func, run.varmap, supplied.text):
            value = bind.value
            settled = settle(value, func)
            if settled:
                found.extend((outcome, 1, bind, []) for outcome in settled)
                continue
            template = values.diagnostic_template(value)
            if template:
                templates.append((template, bind))
            if isinstance(value, ast.Name):
                # Reaching definitions, the same reading the attribute slice
                # makes of a local: 19% of the corpus's writes fill the bind
                # from a variable a guarded chain assigned above it.
                reached = literal_values(func, value.id, settle)
                if reached:
                    dominating = path_condition(bind)
                    found.extend(
                        (outcome, 1, bind, [c for c in conditions if c not in dominating])
                        for outcome, conditions, _line in reached
                    )
                    continue
            # The writer is known, the value is not until run time.
            # Recorded rather than dropped: localize and prune read
            # observed values, so they work from the writer alone.
            found.append((f"runtime({ast.unparse(value)})", 2, bind, []))
        return found

    node = _deciding_fragment(func, run.variable, column, supplied)
    if node is None:
        return []
    if supplied.kind == "literal":
        return [(supplied.text, 1, node, [])]
    if supplied.kind == "expression":
        # ``stateChangeTime=CURRENT_DATE``, ``nFiles=nFiles+:iFiles``: the
        # database decides, from the clock or from the row's own prior
        # contents.  Tier 2 for the reason a bind filled at run time is --
        # the writer is known and that is what localize and prune read -- and
        # dropping it instead was not neutral.  Promotion weighs a subject's
        # literal writes against all of them, so a column the source only ever
        # increments looked, from its one ``= 0``, like a field with a closed
        # vocabulary of one.
        return [(f"runtime({supplied.text})", 2, node, [])]

    # A copied column.  The source is named as a subject rather than as a bare
    # column so the edge joins: ``passthrough(JediTaskSpec.oldStatus)`` points
    # at a node the backward walk can continue from, ``passthrough(oldStatus)``
    # at a string.
    source, attribute, _kind = _subject_of(
        attributor, spec_class, statement.table, supplied.text
    )
    outcome = f"passthrough({SubjectNode.make_name(source, attribute)})"
    return [(outcome, 2, node, [])]


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec_class: str,
    attribute: str,
    outcomes: list[tuple[str, int, ast.stmt, list[str]]],
    row_precondition: list[str],
) -> None:
    """Add one branch per decided value to this write site's junction."""
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
                line_start=outcomes[0][2].lineno,
                line_end=outcomes[-1][2].end_lineno,
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction

    known = {(branch.outcome, tuple(branch.path_condition)) for branch in junction.branches}
    for outcome, tier, node, extra in outcomes:
        condition = path_condition(node) + extra
        if (outcome, tuple(condition)) in known:
            continue
        known.add((outcome, tuple(condition)))
        junction.branches.append(
            Branch(
                outcome=outcome,
                path_condition=condition,
                row_precondition=row_precondition,
                order=len(junction.branches),
                tier=tier,
            )
        )



def selected_values(
    modules: list[SourceModule], attributor: SpecAttributor
) -> dict[str, dict[str, set[str]]]:
    """Return ``{subject: {value: the functions whose query selects on it}}``.

    The same statements read the other way.  A write says what a value becomes;
    a predicate says which rows were asked for, and only both together make a
    state machine out of a pile of writes -- a status nothing ever selects on
    is a status nothing ever moves a task out of, which is the shape of "stuck"
    that no branch table can show.

    **The asking function is kept, not only the value.**  It is in hand here --
    the walk is over functions -- and dropping it left "something selects
    ``finishing``" as the whole answer when exactly one query does, which is
    the difference between a fact and a direction to look in.  Measured over
    the corpus: 134 ``(subject, value)`` pairs have a named reader and 69 of
    them have exactly one.  Downstream, ``FollowUp`` had been pooling triggers
    over the subject's *writers* as a stand-in for the reader's -- an
    approximation its own docstring had to name.

    Attributed exactly like a write, so ``JediTaskSpec.status`` means the same
    thing on both sides and the two can be compared at all.

    **Bound values are settled the same way the write side settles them.**  Both
    sides read the same declared mappings, and taking only a literal here left
    the read side blind to every status a command passes through:
    ``varMap[":status"] = taskStatusMap["doing"]`` is the orphan-rescue query in
    ``getTasksToExecCommand_JEDI``, and without it ``aborting``, ``finishing``,
    ``toretry``, ``toincexec`` and ``toreassign`` all looked like values no
    query ever asks for -- that is, like states a task can enter and never
    leave.  The write side already resolved the same mapping through the other
    spelling, so the two slices were disagreeing about one fact.

    That mapping's ``dummy`` sentinel comes along with them, and the loop
    skips it (``if varMap[":status"] in ["dummy", "paused"]: continue``) before
    the statement runs.  Left in rather than modelled: the guard is a
    ``continue`` above the execution, which is not the dominating-``if`` shape
    the path conditions read, and building that for one site would be a
    mechanism for one site.  Over-crediting is the direction this function
    already takes -- see :func:`_tables_of` -- and it costs nothing measurable
    here: ``dummy`` was not among the values reported as written-but-unselected
    for this subject, so nothing is hidden by it.
    """
    settle = values.resolver(values.declared_mappings(modules))
    found: dict[str, dict[str, set[str]]] = {}

    def record(subject: str, value: str, owner: str) -> None:
        found.setdefault(subject, {}).setdefault(value, set()).add(owner)

    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            seen: set[str] = set()
            for run in sql.executions(func):
                if run.sql in seen:
                    continue
                seen.add(run.sql)
                for table in _tables_of(run.sql):
                    spec_class = attributor.class_for_table(table)
                    for column, key in sql.predicates(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        subject = SubjectNode.make_name(qualifier, attribute)
                        for bind in sql.bound_values(func, run.varmap or "", key):
                            for value in settle(bind.value, func):
                                record(subject, value, owner)
                    for column, value in sql.selected_literals(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        record(SubjectNode.make_name(qualifier, attribute), value, owner)
    return found


def selection_gates(
    modules: list[SourceModule], attributor: SpecAttributor, never_written: set[str]
) -> dict[str, set[str]]:
    """Return ``{subject: tables bounding the queries that select on it}``.

    Read from the same statements as :func:`selected_values` and kept beside it
    because the two are halves of one answer.  That one says a query asks for
    this value; this one says what limits which rows the query can see, and a
    task can be invisible for the second reason while the first is satisfied.
    That is not hypothetical: a task sat in ``finishing`` while the query that
    rescues it ran every cycle, because ``JEDI_AUX_Status_MinTaskID`` had
    stopped being updated and its watermark was above the task's id.

    Only tables *nothing in this map writes* count.  A join to a table the
    corpus maintains is a step in a query; a join to one it only ever reads is
    a dependency on something outside, and only the second can go stale in a
    way the map cannot account for.
    """
    found: dict[str, set[str]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            seen: set[str] = set()
            for run in sql.executions(func):
                if run.sql in seen:
                    continue
                seen.add(run.sql)
                gates = {t for t in sql.joins(run.sql) if t in never_written}
                if not gates:
                    continue
                for table in _tables_of(run.sql):
                    spec_class = attributor.class_for_table(table)
                    for column, _key in sql.predicates(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        found.setdefault(
                            SubjectNode.make_name(qualifier, attribute), set()
                        ).update(gates)
    return found


def _tables_of(statement: str) -> list[str]:
    """Return the tables a statement names, so a predicate can be qualified.

    All four verbs, not just ``SELECT``: most of the interesting predicates are
    on an ``UPDATE`` -- ``SET status=:status WHERE status=:oldStatus`` is one
    statement that both writes a status and says which one it is willing to
    move away from, and reading only the queries loses every one of them.

    A statement joining two tables qualifies its predicates against both, which
    over-reads: ``status`` in a task/dataset join is attributed to each.  That
    is the safe direction here -- the values answer whether *anything* selects
    on a status, so one credited too widely weakens a report while a missing
    one would invent a dead end.
    """
    tables = {table for table, _columns in sql.reads(statement)}
    tables.update(write.table for write in sql.writes(statement))
    tables.update(sql.deletes(statement))
    return sorted(tables)
