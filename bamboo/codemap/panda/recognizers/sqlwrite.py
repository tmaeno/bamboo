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

``read_side`` reads the same statements the other way, at two resolutions.  A
``WHERE`` clause says which rows were asked for, and a status nothing selects
on is a status nothing moves a task out of; the table it names says *whose*
rows, which is the only thing a query predicated on nothing but a join key has
to say -- and the thing that makes "the finish is waiting on jobs" a fact the
map holds rather than one it has no room for.
"""

from __future__ import annotations

import ast
import re
from typing import NamedTuple, Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    DiagnosticTemplate,
    EntityNode,
    JunctionNode,
    ReadSiteNode,
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
    targets_of,
)

# The block a decision records about itself reads the same way whichever slice
# found the write, so it is read in one place and imported rather than copied.
# Both halves of the map disagreeing about a fact is the failure this codebase
# has spent the most effort undoing.
from bamboo.codemap.panda.recognizers import progress

SLICE_NAME = "sql-write"

#: Why a written column produced no outcome, named at the branch that decided
#: it rather than inferred afterwards.  Counting is not enough here: the slice
#: went five rounds with a ratio that said 89% and nothing that said which
#: tenth, and a shape that should be read reads exactly like a value this code
#: does not decide.
#:
#: ``ACCOUNTED_FOR`` holds the reasons that are **not** gaps in the extraction
#: -- the same distinction ``uncovered_tables`` already draws for a table that
#: holds no spec.  Kept as a set beside the strings so a reader adding a reason
#: has to decide which side it is on.
NO_VARMAP = "the execute call names no varmap"
VARMAP_FROM_OUTSIDE = "the varmap is not built in this function"
BOUND_OUTSIDE_WINDOW = "bound outside the window this execution owns"
PLACEHOLDER_COMPUTED = "the placeholder is spelled by an expression"
VARMAP_IS_A_LIST = "the varmap is one of a list handed to executemany"
PLACEHOLDER_MISSING_COLON = "the placeholder is bound without its leading colon"
NOTHING_BINDS_IT = "nothing in this function binds this placeholder"
NO_DECIDING_FRAGMENT = "the fragment that wrote this value could not be located"

#: The read side counts statements, not columns, so its reasons are about how
#: a query is run rather than where a value came from.  ``sql.executions``
#: accepts ``execute``/``executemany``/``querySQL``/``querySQLS`` with at least
#: two arguments; each reason below is one of the ways a statement this slice
#: assembled fails to reach it.
RUN_WITHOUT_VARMAP = "run through an accepted call with no varmap argument"
RUN_BUT_TEXT_DIFFERS = "run through an accepted call, but the two reassemblies disagree"
RUN_THROUGH_OTHER_FORM = "run through a call form the reader does not accept"
NOT_RUN_HERE = "assembled here and not run here"

#: Reasons that are not gaps in the extraction.  ``NOT_RUN_HERE`` is a statement
#: handed to a caller to run, so the function that assembled it is not where it
#: could have been read.  ``BOUND_OUTSIDE_WINDOW`` is the window doing its job;
#: see :func:`sql.bound_values`.  ``VARMAP_FROM_OUTSIDE`` is here for the shape
#: where the map really does arrive as a parameter, and on this corpus it fires
#: **nowhere** -- which is a correction worth keeping: 51 columns were first read
#: that way, and the varmap turned out to be a list handed to ``executemany``
#: with every value filled in the same function.  A reason that sounds like "not
#: our business" is the one to check against the call form before believing.
#:
#: What is deliberately *not* here: ``hs_scrapers`` writes five of seven columns
#: from a row of a parsed HTML table, and those land on the gap side under
#: ``NOTHING_BINDS_IT``.  The walk has a terminal for exactly that ("a value
#: another system supplies"), but nothing here can yet tell such a row from a
#: helper filling the same map, and guessing would put a judgement where a
#: measurement belongs.
#:
#: **Being on this side does not take a candidate out of the denominator.**  The
#: ratio still counts it, because changing what counts as a candidate would
#: move ``slice coverage`` and hide the judgement inside a number that looks
#: like a measurement.
ACCOUNTED_FOR = frozenset(
    {VARMAP_FROM_OUTSIDE, BOUND_OUTSIDE_WINDOW, NOT_RUN_HERE}
)


class Unexplained(NamedTuple):
    """One counted column the slice could not account for, and why.

    Carries the junction name it *would* have joined so that a reader can tell
    the two severities apart: a column whose junction exists is a way of
    setting a value missing from a junction that looks complete, and the map's
    enumeration is the thing being relied on.  A column whose junction does not
    exist is a value the map cannot offer at all.
    """

    reason: str
    file: str
    owner: str
    table: str
    column: str
    junction: str
    detail: str = ""
    """What the failing branch had in hand, for a reader who has to go look.

    The reason names a class; this names the instance.  Without it "nothing in
    this function binds this placeholder" sends a reader to a function that
    plainly does bind it, and the disagreement is in which varmap or which
    spelling -- neither of which the class can carry.
    """


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
    list[Unexplained],
]:
    """Extract junctions for SQL bind writes.

    Returns the tables holding no spec alongside the usual three, so the
    build can say what the map does not cover instead of burying it in a
    coverage ratio, and the diagnostic templates bound into statements -- which
    belong here rather than in a scan of their own because the column a bind
    lands in is the join this slice has already made.

    And the columns it counted and could not account for, each with the reason
    the branch that gave up recorded.  For the same argument as ``uncovered``,
    one step finer: a ratio at 89% says a tenth is missing and not which tenth,
    and this slice held 140 of those for five rounds while every number beside
    it stayed still.  Scanning for them afterwards was tried and is what this
    replaces -- a second reader of the same source disagreed with this one by 77
    columns.

    *attributor* is passed in already taught: the table map is learned from the
    whole corpus, so it cannot be built from the module in hand.
    """
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str, str]] = set()
    uncovered: set[str] = set()
    diagnostics: list[DiagnosticTemplate] = []
    unexplained: list[Unexplained] = []
    settle = values.resolver(values.declared_mappings(modules))
    # The same census the attribute slice makes, for the same reading: which
    # call persists a message into a declared column.  Asked of the attributor
    # rather than recomputed from the source, so the two halves of the map
    # cannot disagree about what a setter is.
    setters = progress.spec_setters(
        modules,
        {cls: attributor.declared_attributes(cls) for cls in attributor.declared_classes()},
    )

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            # A method that runs one statement from two places yields the same
            # run twice.  Counting both would inflate the coverage denominator
            # with duplicates and make the slice look worse than it reads.
            seen: set[tuple[str, str, Optional[str], Optional[tuple[int, int]]]] = set()
            for run in sql.executions(func):
                key = (run.variable, run.sql, run.varmap, run.window)
                if key in seen:
                    continue
                seen.add(key)
                for statement in sql.writes(run.sql):
                    spec_class = attributor.class_for_table(statement.table)
                    # A name nobody could read is not a table that holds no
                    # spec.  Listed together they read as one finding, and only
                    # one of the two is a gap in the extraction -- see
                    # ``ReadSide.unreadable``, which counts these instead.
                    if spec_class is None and not _UNREAD_TABLE.fullmatch(statement.table):
                        uncovered.add(statement.table)
                    for column, supplied in statement.columns.items():
                        candidates += 1
                        qualifier, attribute, kind = _subject_of(
                            attributor, spec_class, statement.table, column
                        )
                        templates: list[tuple[str, ast.stmt]] = []
                        why: list[str] = []
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
                            why=why,
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
                            unexplained.append(
                                Unexplained(
                                    reason=why[0] if why else "no reason given",
                                    detail=why[1] if len(why) > 1 else "",
                                    file=module.rel_path,
                                    owner=f"{module.rel_path}::{func.name}",
                                    table=statement.table,
                                    column=column,
                                    # The name ``_record`` would have used, so
                                    # that resolving it against the built
                                    # junctions asks the recogniser's own
                                    # question rather than a second one.
                                    junction=JunctionNode.make_name(
                                        map_id,
                                        SubjectNode.make_name(qualifier, attribute),
                                        f"{module.rel_path}::{func.name}",
                                    ),
                                )
                            )
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
                            setters=setters,
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
    # ``unexplained`` last: it is the only one of the five that is about what
    # this pass could *not* do, and a caller ignoring it still builds a map.
    return (
        subjects,
        list(junctions.values()),
        coverage,
        uncovered,
        diagnostics,
        unexplained,
    )


def _runs_many(run: sql.Execution) -> bool:
    """Is this execution ``executemany``, whose varmap argument is a sequence?

    Asked of the call the reader already paired with the statement, so the
    answer cannot disagree with the pairing.
    """
    call = getattr(run, "call", None)
    return bool(
        call is not None
        and isinstance(getattr(call, "func", None), ast.Attribute)
        and call.func.attr == "executemany"
    )


def _deciding_fragment(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    run: sql.Execution,
    column: str,
    supplied: sql.ColumnValue,
    kind: str,
) -> Optional[ast.stmt]:
    """Return the statement fragment that put ``<column>=<value>`` into the SQL.

    For a value written inline there is no bind assignment to anchor at, and
    anchoring at the ``execute`` instead would throw away the reason: the two
    forms of ``getTasksToExecCommand_JEDI``'s update are appended in the two
    arms of one ``if``, so the fragment carries the condition and the statement
    does not.

    **The match is made on the assembled variant and mapped back.**  Reading a
    fragment at a time is what ``assigns_in`` did, and it answers only for
    ``SET col=value``, where the pair is adjacent.  An ``INSERT`` names its
    columns in one parenthesised run and its values in another, so the pair
    straddles whatever line breaks the author chose -- sixty-eight columns went
    unanchored for nothing but that, and the same reading lost three ``UPDATE``
    values written as multi-line ``CASE`` expressions.  Two statements spelling
    one thing two ways must not get two answers.

    Grouping fragments by the guard they share does not work either.  Given a
    head, a second fragment, and a third appended under an ``if``, a reader can
    meet the statement with the third or without it; the group holding the
    third alone is neither, and nothing ever runs it.  :func:`sql.variant_spans`
    enumerates the statements that exist and says which fragment contributed
    which stretch, so the guard falls out of the mapping rather than being the
    thing split on.

    **A statement with no local name is anchored at its call.**  The daemons
    write the text into ``querySQLS`` directly, so there are no fragments and
    the call is where the text is -- the opposite of the case above rather than
    an exception to it.
    """
    if not run.variable:
        return run.call
    for statement, spans in sql.variant_spans(func, run.variable):
        at = sql.supplied_at(statement, kind, column, supplied)
        if at is not None:
            return sql.fragment_holding(spans, at)
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
    why: list[str],
) -> list[tuple[str, int, ast.stmt, list[str], ast.stmt]]:
    """Return ``(outcome, tier, node, extra conditions, decided at)`` for a column.

    The node is where the value was decided, which differs by form: a bind is
    decided at the Python assignment filling it, and a literal or a copied
    column at the fragment that put it in the statement.

    The extra conditions are for the one form where the deciding node is not the
    whole story.  A bind filled from a local is decided twice over -- the guards
    that reached ``varMap[":status"] = newTaskStatus`` say the write happened,
    and the guards on the assignment that gave the local its value say which
    value.  Both are needed, and neither is derivable from the other's node.

    *why* is appended to, and only when the return is empty: the reason a
    column produced no outcome is decided at the branch that gave up, and
    nowhere else can tell the four bind cases apart.  An out-parameter for the
    same reason *templates* is one -- it is not an outcome.

    Diagnostic templates are appended to *templates* rather than returned: they
    are not outcomes, and the caller files them against the column rather than
    the subject.  Collected here because *which column the text lands in* is the
    join this slice has already made -- on its own the bind is called
    ``:errDiag`` and could belong to any statement in the method.
    """
    if supplied.kind == "bind":
        if run.varmap is None:
            why.append(NO_VARMAP)
            return []
        scan = sql.scan_binds(func, run.varmap, supplied.text, run.window)
        found = []
        for bind in scan.binds:
            value, site = bind.value, bind.site
            settled = settle(value, func)
            if settled:
                found.extend((outcome, 1, site, [], site) for outcome in settled)
                continue
            template = values.diagnostic_template(value)
            if template:
                templates.append((template, site))
            if isinstance(value, ast.Name):
                # Reaching definitions, the same reading the attribute slice
                # makes of a local: 19% of the corpus's writes fill the bind
                # from a variable a guarded chain assigned above it.
                reached = literal_values(func, value.id, settle)
                if reached:
                    dominating = path_condition(site)
                    found.extend(
                        (
                            outcome,
                            1,
                            site,
                            [c for c in conditions if c not in dominating],
                            # Where the value was chosen, which is not where it
                            # was bound: ``newTaskStatus = "exhausted"`` sits in
                            # the block that also records why, and the bind is
                            # further down, after the chain has closed.
                            _statement_at(func, line) or site,
                        )
                        for outcome, conditions, line in reached
                    )
                    continue
            # The writer is known, the value is not until run time.
            # Recorded rather than dropped: localize and prune read
            # observed values, so they work from the writer alone.
            found.append((f"runtime({ast.unparse(value)})", 2, site, [], site))
        if not found:
            # Which of the several reasons, from the walk that failed rather
            # than from a second walk asking why -- see ``sql.BindScan``.
            if _runs_many(run):
                # ``executemany(sql, varMaps)`` is handed a *sequence* of maps,
                # each appended after being filled, so the name in the call
                # never carries a placeholder as a key.  Read off the call form
                # rather than off the list, which is what makes it exact.
                why.append(VARMAP_IS_A_LIST)
            elif not scan.varmap_assigned:
                why.append(VARMAP_FROM_OUTSIDE)
            elif scan.outside_window:
                why.append(BOUND_OUTSIDE_WINDOW)
            elif supplied.text.lstrip(":") in scan.constant_keys:
                # ``hs_scrapers`` writes ``r["source"] = url`` against
                # ``VALUES (:source)``.  The key is there and the colon is not,
                # so a reader matching the spelled placeholder misses a value
                # that was decided right here.
                why.append(PLACEHOLDER_MISSING_COLON)
            elif scan.computed_keys and supplied.text not in scan.constant_keys:
                why.append(PLACEHOLDER_COMPUTED)
            else:
                why.append(NOTHING_BINDS_IT)
            why.append(
                f"varmap={run.varmap} window={run.window} "
                f"keys={len(scan.constant_keys)} computed={scan.computed_keys}"
            )
        return found

    node = _deciding_fragment(func, run, column, supplied, statement.kind)
    if node is None:
        why.append(NO_DECIDING_FRAGMENT)
        return []
    if supplied.kind == "literal":
        return [(supplied.text, 1, node, [], node)]
    if supplied.kind == "expression":
        # ``stateChangeTime=CURRENT_DATE``, ``nFiles=nFiles+:iFiles``: the
        # database decides, from the clock or from the row's own prior
        # contents.  Tier 2 for the reason a bind filled at run time is --
        # the writer is known and that is what localize and prune read -- and
        # dropping it instead was not neutral.  Promotion weighs a subject's
        # literal writes against all of them, so a column the source only ever
        # increments looked, from its one ``= 0``, like a field with a closed
        # vocabulary of one.
        return [(f"runtime({supplied.text})", 2, node, [], node)]

    # A copied column.  The source is named as a subject rather than as a bare
    # column so the edge joins: ``passthrough(JediTaskSpec.oldStatus)`` points
    # at a node the backward walk can continue from, ``passthrough(oldStatus)``
    # at a string.
    source, attribute, _kind = _subject_of(
        attributor, spec_class, statement.table, supplied.text
    )
    outcome = f"passthrough({SubjectNode.make_name(source, attribute)})"
    return [(outcome, 2, node, [], node)]


def _statement_at(
    func: ast.FunctionDef | ast.AsyncFunctionDef, line: int
) -> Optional[ast.stmt]:
    """The statement of *func* that starts on *line*.

    ``literal_values`` answers with a line rather than a node, and the block
    around that line is what says why the value was chosen.  Looked up rather
    than threaded through because the resolver is shared with the attribute
    slice, which has the node in hand and needs no line at all.
    """
    for node in ast.walk(func):
        if isinstance(node, ast.stmt) and node.lineno == line:
            return node
    return None


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    spec_class: str,
    attribute: str,
    outcomes: list[tuple[str, int, ast.stmt, list[str], ast.stmt]],
    row_precondition: list[str],
    setters: dict[str, object],
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
                # Over all the outcomes, not the first and last the scan
                # happened to visit: ``outcomes`` is in discovery order, so an
                # arm above the first one gave seventeen junctions a span that
                # ends before it starts.  Nothing complained, because no
                # production reader looks at ``line_end`` (see ``Anchor``).
                line_start=min(node.lineno for _o, _t, node, _e, _d in outcomes),
                line_end=max(
                    node.end_lineno or node.lineno for _o, _t, node, _e, _d in outcomes
                ),
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction

    known = {(branch.outcome, tuple(branch.path_condition)) for branch in junction.branches}
    for outcome, tier, node, extra, decided_at in outcomes:
        condition = path_condition(node) + extra
        if (outcome, tuple(condition)) in known:
            continue
        known.add((outcome, tuple(condition)))
        # Read at the deciding statement rather than at the bind.  The two are
        # the same for a literal and far apart for a local: ``retryTask_JEDI``
        # picks the status and writes the reason in one block and binds it
        # after the whole chain has closed, so the bind's block says nothing
        # about which of five refusals this is.
        tags, messages = progress.recorded_signature(decided_at, func, setters)
        junction.branches.append(
            Branch(
                outcome=outcome,
                path_condition=condition,
                row_precondition=row_precondition,
                order=len(junction.branches),
                tier=tier,
                tags=tags,
                messages=messages,
                line=decided_at.lineno,
            )
        )



class EntityUse(NamedTuple):
    """Where one kind of row is read, made, moved and removed."""

    tables: set[str]
    read_by: set[str]
    created_by: set[str]
    updated_by: set[str]
    deleted_by: set[str]


class ValueUse(NamedTuple):
    """Who asks for a value, and who acts on rows that already hold it.

    One ``WHERE`` clause, two questions, and they part company here.  *Does
    anything move a row out of this state?* is answered by ``UPDATE ... SET
    status=:new WHERE status=:old`` as squarely as by a query, so both verbs
    count towards it.  *Who has to pick this row up, so who to ask why they did
    not?* is not: an update is the picking up, not a chance to have missed it.

    Pooling them said an update was a query for a fifth of the corpus -- 99 of
    524 ``(function, subject, value)`` claims came from nothing but a write's
    predicate.  The cost was not only wording: one hop out of a junction opens
    a descent on whatever the called helper selects, so a writer counted as a
    reader sent the walk forward while it said it was stepping down.
    """

    selected_by: set[str]
    updated_by: set[str]


class ReadSide(NamedTuple):
    """What the predicates and the row sources of the corpus's statements say.

    Two questions about the same statements, answered in one walk because the
    walk is the cost: which *column values* something asks for, and whose
    *rows* it asks for at all.  Keeping them apart as two passes would also
    have written the table-to-class join twice.
    """

    #: ``{subject: {value: ValueUse}}`` -- the values a predicate names, with
    #: the verb that brought the table in kept beside each function.
    values: dict[str, dict[str, ValueUse]]
    #: ``{spec class: EntityUse}``
    entities: dict[str, EntityUse]
    #: ``{owner: sets of kinds of row one of its queries reads together}``.
    #: The sound form of a join between two entities: co-residence in a
    #: function is not a relation -- 112 functions read more than one kind
    #: of row somewhere, and 51 read two in one statement -- but a single
    #: ``FROM`` list is the corpus stating the relation itself.
    joins: dict[str, set[frozenset[str]]]
    #: ``{owner: statements whose table reference is still a placeholder}``.
    #: Counted rather than left out: a statement whose table nobody could read
    #: is one whose rows belong to nothing, and an unreadable name looks
    #: exactly like a table that holds no spec once both end up in one list.
    unreadable: dict[str, int]
    #: ``{(owner, line): ReadStatement}`` -- the same records as ``values``,
    #: kept per statement instead of per function.  The walk already tells the
    #: statements apart, because ``(sql, window)`` is what it deduplicates on;
    #: until now ``record`` was handed the owner and the position was dropped,
    #: which made three queries in ``copyArchive.main`` one answer.
    statements: dict[tuple[str, int], "ReadStatement"]


class ReadStatement(NamedTuple):
    """One run of one statement, and every value its predicates ask rows by.

    Per run, not per text: the forwarded form pairs one statement with several
    varmaps, and ``window`` is what tells those runs apart -- so two runs of the
    same text in one function are two of these, which is the distinction the
    whole record exists to keep.
    """

    owner: str
    package: str
    file: str
    blob_sha: str
    line: int
    #: The lines the forwarded form filled this run's binds within, or ``None``
    #: for ``execute``, where the binds are the function's own.
    window: Optional[tuple[int, int]]
    #: ``{(subject, value)}`` the predicates name, for the read verb only.
    selects: set[tuple[str, str]]
    tables: set[str]


READ_SLICE_NAME = "sql-read"


def read_coverage(modules: list[SourceModule]) -> list[CoverageStat]:
    """How many of the corpus's queries the read side actually reads.

    The denominator that did not exist.  ``sql-write`` has one, so a write the
    slice cannot read shows up as a ratio falling; the read side had none, and
    508 statements could stop being read without a single number moving.  That
    is P1-25's rule 6 -- a slice with no denominator passes silently even when
    it is missing entirely.

    **Counted by a different reading from the one it measures.**  The
    numerator comes from :func:`sql.executions`, which recognises the call
    forms; the denominator must not, or the measure would define its
    population as whatever the extractor already sees and could never report
    a form it does not know.  So candidates are found from the *statements* --
    every local a function assembles SQL into, whatever executes it -- and a
    query reachable only through a call form ``executions`` does not accept
    lands in the denominator and not the numerator, which is the whole point.

    Per file, like every other slice, so the build's lowest-coverage listing
    can name where the gap is.
    """
    found: list[CoverageStat] = []
    unexplained: list[Unexplained] = []
    for module in modules:
        candidates = explained = 0
        for func, _owner in functions_with_owner(module.tree):
            assembled = _assembled_queries(func)
            candidates += len(assembled)
            run = {
                statement
                for execution in sql.executions(func)
                for statement in (execution.sql,)
            }
            explained += sum(1 for text in assembled if text in run)
            for text, name in sorted(assembled.items()):
                if text in run:
                    continue
                unexplained.append(
                    Unexplained(
                        reason=_why_not_run(func, name),
                        file=module.rel_path,
                        owner=f"{module.rel_path}::{func.name}",
                        # The variable, not a table: this slice counts
                        # statements, and which tables one names is what it
                        # could not get to.
                        table=name,
                        column="",
                        junction="",
                    )
                )
        if candidates:
            found.append(
                CoverageStat(
                    slice_name=READ_SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )
    return found, unexplained


def _why_not_run(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> str:
    """Why a statement this function assembled is not one the reader saw run.

    Asked of the calls that take *name* as their first argument, which is the
    same place :func:`sql.executions` looks and declines.  Naming the form it
    declined is the point: the denominator exists so that a query form the
    slice does not recognise cannot go missing silently, and a count alone says
    a form is missing without saying which one to add.
    """
    others: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        mentions = isinstance(first, ast.Name) and first.id == name
        if not mentions:
            # ``execute(sql + comment, varMap)`` -- the reader fills and
            # concatenates before matching, so a statement reaching a call
            # inside an expression still counts as reaching it.
            mentions = any(
                isinstance(inner, ast.Name) and inner.id == name
                for inner in ast.walk(first)
            )
        if not mentions:
            continue
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else "?"
        )
        if attr in sql.RUNNERS:
            if len(node.args) < 2:
                return RUN_WITHOUT_VARMAP
            # Accepted form, yet the text did not match: the reassembly this
            # slice did and the one ``executions`` did disagree, which is a
            # different finding from an unrecognised call.
            return RUN_BUT_TEXT_DIFFERS
        others.add(attr)
    if others:
        return f"{RUN_THROUGH_OTHER_FORM}: {', '.join(sorted(others))}"
    return NOT_RUN_HERE


def _assembled_queries(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Every statement *func* builds that asks for rows, however it is run.

    Read off the assignments rather than off the executions.  A name is a
    candidate when what it holds names a table under a read verb; the call
    that runs it is not consulted, which is what keeps this independent of the
    reading it is the denominator for.

    Maps each statement to a local that holds it, so that one left out of the
    numerator can be traced back to the variable -- which is what a reader needs
    in order to see how it is run.  **Keyed by the text**, exactly as the set it
    replaces was: two locals assembling the same statement were one candidate
    and stay one, because the denominator counts statements.  Keying by the pair
    instead would have moved ``sql-read``'s candidate count.
    """
    names = {
        target.id
        for node in ast.walk(func)
        for target in (
            [node.target] if isinstance(node, ast.AugAssign) else targets_of(node)
        )
        if isinstance(target, ast.Name)
    }
    return {
        text: name
        for name in sorted(names, reverse=True)
        for text in sql.variants(func, name)
        if any(verb == "read" for verb, _table in _table_verbs(text))
    }


def read_side(modules: list[SourceModule], attributor: SpecAttributor) -> ReadSide:
    """Return what the corpus's statements ask for: which values, and whose rows.

    **The rows are half of it, and the half the map had nowhere to put.**  A
    vocabulary of ``(spec class, attribute)`` pairs can only say something
    about a column, so a query selecting a task's jobs on the join key alone
    -- ``SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:jediTaskID`` --
    named no promoted subject and read as observing nothing.  It observes jobs.
    The class was already being computed here and dropped whenever no predicate
    landed on a promoted attribute.

    Verbs are kept apart.  A function that selects a task's jobs and one that
    updates them make different claims, and pooling them is the same
    conflation that let an ``UPDATE ... WHERE`` be reported as a query that
    selects on a value.

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
    found: dict[str, dict[str, ValueUse]] = {}
    rows: dict[str, EntityUse] = {}
    unreadable: dict[str, int] = {}
    joins: dict[str, set[frozenset[str]]] = {}

    statements: dict[tuple[str, int], ReadStatement] = {}

    def record(
        subject: str,
        value: str,
        owner: str,
        verb: str,
        at: Optional[ReadStatement] = None,
    ) -> None:
        use = found.setdefault(subject, {}).setdefault(value, ValueUse(set(), set()))
        (use.selected_by if verb == "read" else use.updated_by).add(owner)
        if verb != "read" or at is None:
            return
        # Filed against the statement as well as the function, and only kept
        # when a pair actually lands: a run whose predicates name nothing the
        # map promoted is not somewhere to send a reader, so it gets no row.
        statements.setdefault((at.owner, at.line), at).selects.add((subject, value))

    def touch(table: str, owner: str, verb: str) -> None:
        use = rows.setdefault(table.lower(), EntityUse(set(), set(), set(), set(), set()))
        use.tables.add(table)
        getattr(use, f"{verb}_by").add(owner)

    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            # Keyed by the window as well as the text.  A statement is emitted
            # once per call site, so deduplicating on the text alone keeps the
            # first pairing and throws the rest away -- which is exactly the
            # pairing the forwarded form exists to make.  ``copyArchive.main``
            # runs one ``jobsActive4`` query from three places with
            # ``:jobStatus`` bound to ``holding`` at each, and the text-only
            # key reported none of them.
            seen: set[tuple[str, Optional[tuple[int, int]]]] = set()
            for run in sql.executions(func):
                if (run.sql, run.window) in seen:
                    continue
                seen.add((run.sql, run.window))
                together: set[str] = set()
                for verb, table in _rows_touched(run.sql):
                    if _UNREAD_TABLE.fullmatch(table):
                        unreadable[owner] = unreadable.get(owner, 0) + 1
                        continue
                    touch(table, owner, verb)
                    if verb == "read":
                        # Joined entities stay keyed by the spec class: that
                        # reading asks which *other kind of thing* a decision
                        # looked at in the same statement, and there the three
                        # job tables really are one kind of thing.
                        spec_class = attributor.class_for_table(table)
                        if spec_class is not None:
                            together.add(spec_class)
                if len(together) > 1:
                    joins.setdefault(owner, set()).add(frozenset(together))
                # Built before the verbs so ``record`` has somewhere to file a
                # pair, and thrown away again if none lands.  The line is the
                # one the statement *runs* on: the text is assembled further up
                # and the binds are not in scope there.
                at = ReadStatement(
                    owner=owner,
                    package=module.package,
                    file=module.rel_path,
                    blob_sha=module.blob_sha,
                    line=run.call.lineno,
                    window=run.window,
                    selects=set(),
                    tables=set(),
                )
                read_tables: set[str] = set()
                for verb, table in _table_verbs(run.sql):
                    spec_class = attributor.class_for_table(table)
                    if verb == "read":
                        read_tables.add(table)
                    for column, key in sql.predicates(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        subject = SubjectNode.make_name(qualifier, attribute)
                        for bind in sql.bound_values(
                            func, run.varmap or "", key, run.window
                        ):
                            for value in settle(bind.value, func):
                                record(subject, value, owner, verb, at)
                    for column, value in sql.selected_literals(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        record(
                            SubjectNode.make_name(qualifier, attribute),
                            value,
                            owner,
                            verb,
                            at,
                        )
                # After the verbs, and onto whichever record was kept: a join
                # reads two tables in one statement, and the second one would
                # otherwise land on the copy ``setdefault`` discarded.
                kept = statements.get((owner, run.call.lineno))
                if kept is not None:
                    kept.tables.update(read_tables)
    return ReadSide(
        values=found,
        entities=rows,
        unreadable=unreadable,
        joins=joins,
        statements=statements,
    )


def read_site_nodes(
    statements: dict[tuple[str, int], "ReadStatement"],
    map_id: str,
    derived_from: str,
) -> list[ReadSiteNode]:
    """Turn the statements :func:`read_side` saw into one node each.

    No promotion and no criterion.  A statement that asks for rows by a value
    the map already holds a subject for is somewhere an investigation can be
    sent, and there is nothing further to judge -- which is the difference
    between this and :func:`entity_nodes`, where "is this worth asking about"
    is a question about a column.

    ``selects`` is rendered rather than kept as pairs because it is read, not
    matched: the pair is already in ``SubjectNode.selected_by``, and what this
    row adds is which statement, in what company.
    """
    nodes: list[ReadSiteNode] = []
    for (owner, line), at in sorted(statements.items()):
        nodes.append(
            ReadSiteNode(
                map_id=map_id,
                derived_from=derived_from,
                name=ReadSiteNode.make_name(map_id, owner, line),
                owner=owner,
                selects=sorted(f"{subject}={value}" for subject, value in at.selects),
                tables=sorted(at.tables),
                # Kept where it is a fact and left empty where it is not:
                # ``execute`` fills its binds anywhere in the function, so
                # claiming a region there would be claiming a scope the code
                # does not declare.
                bind_window=list(at.window) if at.window else [],
                anchor=Anchor(
                    package=at.package,
                    file=at.file,
                    line_start=line,
                    line_end=line,
                    blob_sha=at.blob_sha,
                ),
            )
        )
    return nodes


def entity_nodes(
    uses: dict[str, EntityUse],
    map_id: str,
    derived_from: str,
    attributor: Optional[SpecAttributor] = None,
) -> list[EntityNode]:
    """Turn what :func:`read_side` saw into nodes, one per kind of row.

    No promotion.  The criteria decide whether asking "why is this attribute
    this value?" is a question worth having, which is a question about a
    column; an entity is not a candidate for it and would be judged by rules
    that read an attribute it does not have.

    **A table this corpus only reads is left out.**  It already has a node, and
    a better-fitting one: ``tables_never_written`` finds it and
    ``extract_read_only_tables`` makes it an unbound boundary -- a dependency on
    something outside, which is what a table nothing here maintains is.
    Recording it here as well would put one fact in two node types, the shape
    that ``log_files`` and ``opened_by`` have already cost this map twice.  The
    two readings agree on the installed corpus: the tables with no writing verb
    are exactly ``tables_never_written``'s thirty-nine.

    What is left is sixty-seven kinds of row where keying on the spec class
    gave thirteen, and the fifty-four that appear are the ones a stalled
    command is about.
    """
    return [
        EntityNode(
            name=table,
            map_id=map_id,
            derived_from=derived_from,
            spec_class=attributor.class_for_table(table) if attributor else None,
            tables=sorted(use.tables),
            read_by=sorted(use.read_by),
            created_by=sorted(use.created_by),
            updated_by=sorted(use.updated_by),
            deleted_by=sorted(use.deleted_by),
        )
        for table, use in sorted(uses.items())
        if use.created_by or use.updated_by or use.deleted_by
    ]


def selection_gates(
    modules: list[SourceModule], attributor: SpecAttributor, never_written: dict[str, str]
) -> dict[str, set[str]]:
    """Return ``{subject: tables bounding the queries that select on it}``.

    Read from the same statements as :func:`read_side` and kept beside it
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

    The membership test folds case, because :func:`tables_never_written` keys
    on the folded name: the corpus reads ``jobs_statuslog`` and writes
    ``jobs_StatusLog``, and comparing spellings made twenty subjects -- among
    them ``JobSpec.jobStatus`` -- carry a gate on a table PanDA maintains.
    """
    found: dict[str, set[str]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            seen: set[str] = set()
            for run in sql.executions(func):
                if run.sql in seen:
                    continue
                seen.add(run.sql)
                # Folded for the lookup and shown as ``never_written`` spells
                # it, so one table does not reach a reader under two names.
                gates = {
                    never_written[t.lower()]
                    for t in sql.joins(run.sql)
                    if t.lower() in never_written
                }
                if not gates:
                    continue
                for table in _tables_of(run.sql):
                    # A table does not bound itself.  The statement's gates are
                    # read once and offered to every table in it, so a subject
                    # sitting on the gating table got its own table back as the
                    # thing limiting which of its rows can be seen --
                    # ``JEDI_AUX_Status_MinTaskID.status`` bounded by
                    # ``JEDI_AUX_Status_MinTaskID``.  Seventeen of the
                    # fifty-eight gated subjects were that, and the verdict
                    # they produce tells a reader to go and find out who
                    # maintains a table they are already looking at.
                    bounds = {gate for gate in gates if gate.lower() != table.lower()}
                    if not bounds:
                        continue
                    spec_class = attributor.class_for_table(table)
                    for column, _key in sql.predicates(run.sql):
                        qualifier, attribute, _kind = _subject_of(
                            attributor, spec_class, table, column
                        )
                        found.setdefault(
                            SubjectNode.make_name(qualifier, attribute), set()
                        ).update(bounds)
    return found


#: A table reference that is nothing but holes -- ``{}`` or ``{0}``.  What is
#: left when a name the source supplies at run time could not be resolved.
_UNREAD_TABLE = re.compile(r"(?:\{\d*\})+")


def _rows_touched(statement: str) -> list[tuple[str, str]]:
    """``[("read" | "created" | "updated" | "deleted", table)]`` for *statement*.

    The same three readings :func:`_tables_of` folds together, kept apart.  A
    predicate does not care which verb brought the table in -- an ``UPDATE``'s
    ``WHERE`` says which rows it was willing to act on as much as a query's
    does -- but "who reads a task's jobs" cares about nothing else.  Folding
    the verbs is how an ``UPDATE ... WHERE`` came to be reported as a query
    that selects on a value.
    """
    touched: list[tuple[str, str]] = [
        ("read", table) for table, _columns in sql.reads(statement)
    ]
    # The rest of the ``FROM`` list too.  :func:`sql.reads` keeps naming the
    # leading table because it answers *where the row came from*; the question
    # here is which kinds of row the statement touches at all, and a join
    # partner's rows are read as surely as the first table's.  Sixty-one
    # (function, entity) readings were missing for the difference, sixteen of
    # them on datasets and sixteen on files -- the two a task waits for.
    touched.extend(("read", table) for table in sql.joins(statement))
    # The three writing verbs stay apart.  Making a row, changing one and
    # removing one are three different claims about a kind of row, and pooling
    # them is why nothing downstream could ask where an object enters the
    # system.  ``sql.deletes``'s own docstring already said what that costs: a
    # DELETE followed by an INSERT on a command table means a second command
    # silently replaced one that was never picked up, and the map had nowhere
    # to put it.
    touched.extend(("created", table) for table in sql.creates(statement))
    touched.extend(("updated", table) for table in sql.updates(statement))
    touched.extend(("deleted", table) for table in sql.deletes(statement))
    return touched


def _table_verbs(statement: str) -> list[tuple[str, str]]:
    """``(verb, table)`` for each way *statement* touches a table, deduplicated.

    :func:`_tables_of` with the verb kept, for the reading that needs it.  A
    statement naming one table under both verbs -- ``INSERT INTO a SELECT FROM
    a`` -- yields both, which is what it does.
    """
    return sorted(set(_rows_touched(statement)))


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
    return sorted({table for _verb, table in _rows_touched(statement)})
