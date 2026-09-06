"""Reading SQL out of Python.

PanDA writes most of its state through SQL, not attribute assignment, so the
progress slice is mostly here.  The shape is regular enough to read but not
regular enough to read naively::

    sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
    sqlU += "SET status=:status,modificationTime=:updateTime,"
    sqlU += "lockedBy=NULL,frozenTime=:frozenTime "
    sqlU += " WHERE jediTaskID=:jediTaskID "
    ...
    varMap[":status"] = taskStatus
    self.cur.execute(sqlU + comment, varMap)

Four things follow, each of which cost a wrong assumption to learn:

**The statement has to be reassembled.**  It is built by ``=`` then a run of
``+=``, often with an f-string holding the schema, and interleaved with other
statements being built in the same function.  Reassembly is per variable name.

**Reassembly is the wrong unit for reading a branch.**  The fragments are
appended conditionally -- ``getTasksToExecCommand_JEDI`` appends either ``SET
status=:status`` or ``SET status=oldStatus`` -- so the ``if`` that explains the
value is on the fragment, not on the statement.  ``fragments`` keeps them.

**The bind key does not say whether it is a write.**  ``:status`` appears in
``SET status=:status`` *and* in another statement's ``WHERE status=:status``
in the same function -- the flagship ``updateTaskStatusByContFeeder_JEDI`` does
exactly this.  Only the ``SET`` clause distinguishes a write from a predicate,
so the statement is required; the bind key alone is not enough.

**Which spec a table holds is not declared anywhere**, so it is inferred from
the column names -- see ``SpecAttributor.learn_table_classes``.

Not attempted: parsing SQL properly.  These are regexes over reassembled
strings, which is enough for ``UPDATE ... SET`` and ``INSERT INTO ... VALUES``
and nothing more.  Anything they cannot read is reported as unexplained rather
than guessed at, which is what the coverage matrix is for.
"""

from __future__ import annotations

import ast
import itertools
import logging
import math
import re
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field

from bamboo.codemap.panda.pathcond import exclusive, path_condition
from bamboo.codemap.panda.values import rendered_text

logger = logging.getLogger(__name__)

# A run with this many branch combinations is folded rather than split.  Set
# well above the corpus's worst case (12, in ``propagateResultToJEDI``) so it is
# a guard against a pathological method rather than a working limit.
_MAX_BRANCH_VARIANTS = 24

# ``UPDATE [/*+ hint */] <schema>.<table> [alias] SET <assignments> [WHERE ...]``.
# The schema is usually an f-string placeholder, which reassembly leaves as
# ``{}``.  The optimizer hint and the table alias are both optional and both
# occur: ``UPDATE /*+ index(tab ...) */ ATLAS_PANDA.filesTable4 tab SET
# status='ready'`` is one statement, and without either allowance it reads as no
# statement at all.
_UPDATE = re.compile(
    r"\bUPDATE\s+(?:/\*.*?\*/\s*)?([\w{}.]+)(?:\s+(?!SET\b)\w+)?\s+SET\s+(.*?)(?:\bWHERE\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT = re.compile(
    r"\bINSERT\s+INTO\s+([\w{}.]+)\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*=\s*(:?[A-Za-z_]\w*|[^,]+?)(?=\s*,|\s*$)")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)

_QUOTED = re.compile(r"^'([^']*)'$")
_BARE = re.compile(r"^[A-Za-z_]\w*$")
_NUMBER = re.compile(r"^[+-]?\d+(\.\d+)?$")
# Bare words that are SQL, not columns.  This has to stay a list of *keywords*
# rather than of names that look like plumbing: ``T_TASK`` really does have a
# column called ``timeStamp``, and ``JEDI_Events`` one called ``event_offset``,
# both of which are copied into other columns.  ``NULL`` is settled earlier as
# a literal and never reaches here; it stays listed because what the set means
# is "not a column name", which is true of it however it is read.
_KEYWORDS = frozenset(
    {"NULL", "CURRENT_DATE", "CURRENT_TIMESTAMP", "SYSDATE", "SYSTIMESTAMP", "DEFAULT", "TRUE", "FALSE"}
)


class ColumnValue(BaseModel):
    """Where one column's new value comes from.

    The four forms are four different kinds of branch, which is why the
    distinction is drawn here rather than left to the caller:

    ``bind``
        ``SET status=:status`` -- the value is decided in Python, at the
        assignment filling the bind, and that is where the ``if`` explaining it
        is too.
    ``literal``
        ``SET status='ready'``, ``SET oldStatus=NULL``, ``SET coreCount=0`` --
        decided in the statement itself.  Quoting is a property of the type,
        not of how settled the value is, and reading only the quoted ones was
        the same mistake the attribute slice made in only reading string
        right-hand sides.
    ``column``
        ``SET status=oldStatus`` -- carried from another column of the same
        row.  This is the form the recognizers were blind to, and it is not a
        curiosity: it is how a task returns from ``pending``, so without it the
        map cannot offer "the release did not fire" as a candidate at all.
    ``expression``
        ``CURRENT_DATE``, ``nFiles+1``, a subquery.  A write whose value the
        statement does not settle: the clock and the row's own prior contents
        are read when it runs.
    """

    kind: str = Field(..., description="bind | literal | column | expression")
    text: str = Field(
        ...,
        description="Bind key, literal value, source column, or the raw SQL, per ``kind``.",
    )


def classify(value: str) -> ColumnValue:
    """Classify the right-hand side of one SQL column assignment.

    ``NULL`` is reported as Python's ``None`` because the two spell one fact,
    and the column it clears is one subject: ``JediTaskSpec.oldStatus`` is
    written ``= None`` through the attribute slice and ``=NULL`` through this
    one, and a branch table offering both spellings would read as two outcomes
    where the source has one.  Python's is the spelling that wins for the same
    reason ``runtime(...)`` holds unparsed Python -- everything downstream that
    compares an outcome to an observed value is reading PanDA through Python.
    """
    value = value.strip()
    if value.startswith(":"):
        return ColumnValue(kind="bind", text=value)
    quoted = _QUOTED.match(value)
    if quoted is not None:
        return ColumnValue(kind="literal", text=quoted.group(1))
    if value.upper() == "NULL":
        return ColumnValue(kind="literal", text="None")
    if _NUMBER.match(value):
        return ColumnValue(kind="literal", text=value)
    if _BARE.match(value) and value.upper() not in _KEYWORDS:
        return ColumnValue(kind="column", text=value)
    return ColumnValue(kind="expression", text=value)


def assigns_in(fragment: str) -> list[tuple[str, ColumnValue]]:
    """Return the ``column = value`` pairs a SQL *fragment* sets.

    Everything from the first ``WHERE`` on is dropped: a predicate uses the
    same ``column = value`` spelling as an assignment, so a fragment carrying
    both would report its selection criteria as writes.
    """
    head = _WHERE.split(fragment, maxsplit=1)[0]
    return [(column, classify(value)) for column, value in _ASSIGNMENT.findall(head)]


class SqlWrite(BaseModel):
    """One ``UPDATE ... SET`` or ``INSERT``, as far as it could be read."""

    kind: str = Field(..., description="update | insert")
    table: str = Field(..., description="Table name, schema stripped.")
    columns: dict[str, ColumnValue] = Field(
        default_factory=dict,
        description=(
            "Column -> where its new value comes from.  Columns written to an "
            "expression are kept even though they carry nothing to trace: they "
            "are part of what identifies the table."
        ),
    )
    preconditions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column -> the predicate testing it, for columns this statement "
            "also writes.  That combination is a compare-and-set: whoever moved "
            "the row first wins and this write changes no rows, returning a "
            "count the caller usually discards.  Predicates on columns the "
            "statement does not write are row identity, not a race."
        ),
    )


def _parts(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    seen: frozenset[str] = frozenset(),
) -> list[tuple[int, str, str, ast.stmt]]:
    """Return ``(line, operator, text, node)`` for each statement building *name*."""
    if name in seen:
        return []
    seen = seen | {name}
    parts: list[tuple[int, str, str, ast.stmt]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not (isinstance(target, ast.Name) and target.id == name):
                    continue
                text = rendered_text(node.value)
                if text is None and isinstance(node.value, ast.Name):
                    # ``sql = sqlTU`` -- the statement was built under another
                    # name and *chosen* here.  The choosing assignment is the
                    # better anchor of the two: in ``reactivatePendingTasks_JEDI``
                    # the release statement is appended unconditionally and
                    # picked in the ``else`` of the timeout test, so the reason
                    # is on the alias and nowhere else.
                    text = _fold(_parts(func, node.value.id, seen))
                if text:
                    parts.append((node.lineno, "=", text, node))
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and isinstance(node.op, ast.Add)
        ):
            text = rendered_text(node.value)
            if text is not None:
                parts.append((node.lineno, "+=", text, node))
    parts.sort(key=lambda part: part[0])
    return parts


def _fold(parts: list[tuple[int, str, str, ast.stmt]]) -> str:
    """Concatenate one ``=``-started run of fragments."""
    assembled = ""
    for _line, operator, text, _node in parts:
        assembled = text if operator == "=" else assembled + text
    return assembled


def _exclusive_groups(
    parts: list[tuple[int, str, str, ast.stmt]], start: int
) -> list[list[int]]:
    """Return the index groups whose members cannot all be in one statement.

    Connected components rather than pairs, because an ``if``/``elif``/``else``
    appending a fragment from each arm makes three alternatives, not three
    pairs.  The test is :func:`pathcond.exclusive`, reading the negations the
    path condition already carries.
    """
    conditions = {index: path_condition(parts[index][3]) for index in range(start, len(parts))}
    parent = {index: index for index in conditions}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    indices = sorted(conditions)
    for position, one in enumerate(indices):
        for other in indices[position + 1 :]:
            if exclusive(conditions[one], conditions[other]):
                parent[root(one)] = root(other)
    grouped: dict[int, list[int]] = {}
    for index in indices:
        grouped.setdefault(root(index), []).append(index)
    return [members for members in grouped.values() if len(members) > 1]


def _run_variants(parts: list[tuple[int, str, str, ast.stmt]], name: str) -> list[str]:
    """Return the statements one ``=``-started run of fragments can produce.

    Fragments appended under mutually exclusive branches are *alternatives*, not
    parts of one statement, and folding them together builds a statement that
    cannot exist::

        UPDATE {}.JEDI_Tasks SET status=:status,SET status=oldStatus,...

    which is not merely unreadable -- the second ``SET`` overwrote the first in
    the column map, so the map lost the bind arm of the write entirely and, in
    another method, gained a column no statement ever assigns.

    Only mutual exclusion splits.  A fragment appended under an ``if`` with no
    ``else`` is left in every variant: splitting on optional fragments too would
    double the count for each one, and the resulting over-read is the safe
    direction -- an extra ``AND`` clause is read, not a value invented.

    The head of the run is never dropped.  It is what the statement *is*; the
    alternatives are among the fragments appended to it.
    """
    start = 1 if parts and parts[0][1] == "=" else 0
    groups = _exclusive_groups(parts, start)
    if not groups:
        return [_fold(parts)]
    total = math.prod(len(members) for members in groups)
    if total > _MAX_BRANCH_VARIANTS:
        # Folded as before rather than split into an arbitrary subset: a
        # truncated list of statements reads as the complete one.  Said out
        # loud, because a silent cap is indistinguishable from full coverage.
        logger.warning(
            "%s at line %d is built across %d branch combinations, over the cap "
            "of %d: read as one folded statement, so a conditionally assembled "
            "column may be misread here",
            name,
            parts[0][0],
            total,
            _MAX_BRANCH_VARIANTS,
        )
        return [_fold(parts)]

    alternatives = {index for members in groups for index in members}
    texts: list[str] = []
    for combination in itertools.product(*groups):
        keep = set(combination)
        text = _fold(
            [part for index, part in enumerate(parts) if index not in alternatives or index in keep]
        )
        if text and text not in texts:
            texts.append(text)
    return texts


def variants(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """Return every statement local *name* can hold.

    Two things make one name hold several statements, and both were learned by
    getting them wrong.

    **Reassignment.**  A name given a new ``=`` mid-function holds a different
    statement from then on, and reading only the last quietly drops the others:
    ``reactivatePendingTasks_JEDI`` picks between a timeout statement and a
    release statement through one variable, so keeping one of the two loses half
    of what the method does.

    **Mutually exclusive fragments.**  A statement assembled across an
    ``if``/``else`` is as much two statements as a reassigned one is -- see
    :func:`_run_variants`.

    Either way they are returned separately rather than concatenated, because
    concatenating runs one statement's ``SET`` clause into the next one's, which
    the column regexes cannot tell from a wider ``SET``.
    """
    runs: list[list[tuple[int, str, str, ast.stmt]]] = []
    for part in _parts(func, name):
        if part[1] == "=" or not runs:
            runs.append([])
        runs[-1].append(part)
    return [text for run in runs for text in _run_variants(run, name) if text]


def reconstruct(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> str:
    """Reassemble the SQL string held by local *name*, as last assigned.

    Fragments are ordered by line, with ``=`` restarting the string and ``+=``
    extending it.  This is not flow-sensitive: a statement built differently in
    two branches comes back as one concatenation.  That over-reads rather than
    under-reads, and the table and column names -- the parts used here -- are
    the same in both branches when it happens.

    Where the branches disagree about the *value*, ``fragments`` recovers what
    this cannot: ``getTasksToExecCommand_JEDI`` appends ``SET status=:status``
    or ``SET status=oldStatus`` depending on the command, and both appear here.
    """
    return _fold(_parts(func, name))


def fragments(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[ast.stmt, str]]:
    """Return the statements building *name*, each with the text it contributes.

    Reassembly deliberately flattens a conditionally built statement, which is
    right for reading the table and the columns and wrong for reading *why*.
    A fragment appended under an ``else`` carries that ``else`` in its own path
    condition, so the fragment is the anchor for any value written inline --
    a literal or a copied column, neither of which has a bind assignment to
    point at instead.
    """
    return [(node, text) for _line, _operator, text, node in _parts(func, name)]


class Execution(NamedTuple):
    """One ``cursor.execute(<statement>, <binds>)``."""

    sql: str
    varmap: Optional[str]
    variable: str
    call: ast.Call


def interpolations(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """Return the expressions interpolated into the fragments building *name*.

    Reassembly blanks them, which is right for reading columns and wrong for
    reading *whose* table this is: the schema is interpolated into the head of
    the statement (``f"UPDATE {panda_config.schemaDEFT}.T_TASK "``) while the
    columns arrive in later ``+=`` fragments, so neither half can be read
    without the other.
    """
    found: list[str] = []
    for _line, _operator, _text, node in _parts(func, name):
        value = node.value if isinstance(node, (ast.Assign, ast.AugAssign)) else None
        if not isinstance(value, ast.JoinedStr):
            continue
        found.extend(
            ast.unparse(part.value)
            for part in value.values
            if isinstance(part, ast.FormattedValue)
        )
    return found


def _concatenated(node: ast.expr) -> list[ast.expr]:
    """Flatten a chain of ``+`` into its operands, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concatenated(node.left) + _concatenated(node.right)
    return [node]


#: ``UPDATE %s SET`` and ``UPDATE ATLAS_PANDA.{0} SET`` -- a table name the call
#: supplies rather than the statement.  Both markers are unambiguous: rendered
#: text spells an f-string interpolation ``{}``, never either of these.
_PERCENT_S = "%s"
_BARE_FIELD = "{}"


def _interpolates(node: ast.expr) -> bool:
    """Whether any brace this expression renders could be a hole, not a brace."""
    return any(
        isinstance(child, ast.JoinedStr)
        or (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "format"
        )
        for child in ast.walk(node)
    )


def _braces_are_literal(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Whether every ``{}`` in *name*'s text is a placeholder the source wrote.

    Rendered text spells an f-string interpolation ``{}`` as well, so a bare
    brace is either the hole a schema went into or a field a ``.format`` at the
    call fills.  The fragments settle it without a guess: if none of them
    interpolates, every brace came from a literal.
    """
    return not any(_interpolates(node.value) for _line, _op, _text, node in _parts(func, name))


def _literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, expression: ast.expr
) -> list[str]:
    """Return the strings *expression* can hold, where the source writes them out.

    Two forms, both of which put the names in plain sight: the argument is a
    literal, or it is the target of a ``for`` over a literal sequence --
    ``for table in ("ATLAS_PANDA.jobsDefined4", "ATLAS_PANDA.jobsActive4")``.
    Anything else returns nothing, and the placeholder is left as written: a
    table name invented here would be attributed to a spec class with the same
    confidence as one the code states.
    """
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    if not isinstance(expression, ast.Name):
        return []
    found: list[str] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == expression.id
            and isinstance(node.iter, (ast.Tuple, ast.List))
        ):
            values = [
                element.value
                for element in node.iter.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if len(values) == len(node.iter.elts):
                found.extend(values)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and any(
                isinstance(target, ast.Name) and target.id == expression.id
                for target in node.targets
            ):
                found.append(node.value.value)
    return found


def _substituted(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    text: str,
    name: str,
    fillers: list[list[str]],
    percent: bool,
) -> list[str]:
    """Return *text* with the call's arguments filled in, or as it stands."""
    if not fillers:
        return [text]
    if percent:
        return [text.replace(_PERCENT_S, value) for value in fillers[0]] or [text]
    filled = [text]
    for index, values in enumerate(fillers):
        marker = f"{{{index}}}"
        if not values:
            continue
        if marker in filled[0]:
            filled = [one.replace(marker, value) for one in filled for value in values]
        elif index == 0 and _BARE_FIELD in filled[0] and _braces_are_literal(func, name):
            filled = [
                one.replace(_BARE_FIELD, value) for one in filled for value in values
            ]
    return filled


def _call_site_fill(
    func: ast.FunctionDef | ast.AsyncFunctionDef, expression: ast.expr
) -> tuple[ast.expr, list[list[str]], bool]:
    """Split ``execute(...)``'s first argument into statement and substitution.

    Three spellings put the table name at the call rather than in the
    statement, and until now all three were read as though the name were part
    of the text -- which for ``%s`` meant no statement at all, because the
    table pattern does not accept a ``%``.
    """
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mod):
        return expression.left, [_literal_values(func, expression.right)], True
    inner = (
        expression.left
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
        else expression
    )
    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "format"
    ):
        return (
            inner.func.value,
            [_literal_values(func, argument) for argument in inner.args],
            False,
        )
    # ``execute(sqlU + comment, varMap)`` -- the comment is a tracing tag
    # appended at the call, so the statement is everything to its left.
    if isinstance(expression, ast.BinOp):
        return expression.left, [], False
    return expression, [], False


def executions(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Execution]:
    """Return one :class:`Execution` per cursor execution in *func*.

    The call is what pairs a statement with its binds.  Reading them separately
    -- every statement in the function against every bind in the function --
    would attach a task's status write to a dataset's statement whenever both
    appear in one method, which in ``db_proxy_mods`` is most of them.

    **The statement can arrive in more than two pieces.**  Reading only the
    operand to the left of the comment is right for ``execute(sqlU + comment)``
    and wrong for ``execute(sqlU + sql + comment)``, where the second name holds
    the ``WHERE`` clause -- and it did not merely truncate those statements, it
    dropped them, because the left operand was itself an addition rather than a
    name.  That cost the ``jobStatus`` write in ``updateJobStatus`` and the
    whole of ``updateTask_JEDI``, silently: a statement that is never read
    leaves nothing behind for a graph invariant to catch.

    An operand that says nothing about its own text becomes ``{}``, the same
    hole an f-string interpolation leaves, rather than sinking the statement.
    The head is still required to be a readable name: it is what the statement
    *is*, and without it there is no anchor to hang the execution on.
    """
    found: list[Execution] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"execute", "executemany"} or len(node.args) < 2:
            continue
        expression = node.args[0]
        statement, fillers, percent = _call_site_fill(func, expression)
        operands = _concatenated(statement)
        if not isinstance(operands[0], ast.Name):
            continue
        base = operands[0]
        head = variants(func, base.id)
        if not head:
            continue
        pieces: list[list[str]] = [head]
        for operand in operands[1:]:
            if isinstance(operand, ast.Name):
                pieces.append(variants(func, operand.id) or ["{}"])
                continue
            rendered = rendered_text(operand)
            pieces.append([rendered if rendered is not None else "{}"])
        total = math.prod(len(choices) for choices in pieces)
        if total > _MAX_BRANCH_VARIANTS:
            # Said out loud, for the reason in ``_run_variants``: a silent cap
            # reads exactly like full coverage.  Falling back to the head alone
            # is what this function did for every statement until now.
            logger.warning(
                "the statement executed at line %d joins %d variants of %d name(s), "
                "over the cap of %d: read as %s alone, so its trailing clauses are "
                "not read here",
                node.lineno,
                total,
                len(pieces),
                _MAX_BRANCH_VARIANTS,
                base.id,
            )
            pieces = [head]
        varmap = node.args[1].id if isinstance(node.args[1], ast.Name) else None
        for combination in itertools.product(*pieces):
            for text in _substituted(func, "".join(combination), base.id, fillers, percent):
                found.append(
                    Execution(sql=text, varmap=varmap, variable=base.id, call=node)
                )
    return found


#: A single ``WHERE``/``AND``/``OR`` term, kept whole so the guard reads as the
#: source wrote it.  ``NOT`` is part of the term rather than a separate one:
#: ``updateJobStatus`` guards with ``AND NOT jobStatus=:ngStatus``, and dropping
#: the negation would report the opposite condition.
_TERM = re.compile(
    r"(?:WHERE|AND|OR)\s+((?:NOT\s+)?(?:\w+\.)?([A-Za-z_]\w*)\s*"
    r"(?:(?:=|<>|!=|<=|>=|<|>)\s*[^\s)]+|(?:NOT\s+)?IN\s*\([^)]*\)|IS\s+(?:NOT\s+)?NULL))",
    re.IGNORECASE,
)


def _preconditions(clause: str, columns: set[str]) -> dict[str, str]:
    """Return the terms of *clause* testing a column in *columns*.

    A statement that tests a column it also assigns is doing a compare-and-set,
    and losing that race is silent: no exception, no log, just a row count the
    caller discards.  The map has no other way to say that seeing a value
    *decided* is not the same as seeing the row take it.
    """
    found: dict[str, str] = {}
    lowered = {column.lower(): column for column in columns}
    for term, column in _TERM.findall(clause):
        owner = lowered.get(column.lower())
        if owner is not None and owner not in found:
            found[owner] = re.sub(r"\s+", " ", term).strip()
    return found


def writes(sql: str) -> list[SqlWrite]:
    """Return the write statements in *sql*, as far as they can be read."""
    found: list[SqlWrite] = []
    for match in _UPDATE.finditer(sql):
        columns = {
            column: classify(value) for column, value in _ASSIGNMENT.findall(match.group(2))
        }
        if columns:
            found.append(
                SqlWrite(
                    kind="update",
                    table=_table_of(match.group(1)),
                    columns=columns,
                    preconditions=_preconditions(sql[match.end(2) :], set(columns)),
                )
            )
    for match in _INSERT.finditer(sql):
        names = [c.strip().split(".")[-1] for c in match.group(2).split(",")]
        values = [v.strip() for v in match.group(3).split(",")]
        columns = {}
        for name, value in zip(names, values, strict=False):
            if not _IDENTIFIER.fullmatch(name):
                continue
            supplied = classify(value)
            if supplied.kind == "column":
                # An INSERT has no prior row to copy from, so a bare word in
                # its VALUES list is a sequence or a function, not a source
                # column.  Calling it one would invent a passthrough edge.
                supplied = ColumnValue(kind="expression", text=supplied.text)
            columns[name] = supplied
        if columns:
            found.append(
                SqlWrite(kind="insert", table=_table_of(match.group(1)), columns=columns)
            )
    return found


def _table_of(reference: str) -> str:
    """Strip the schema: ``{}.JEDI_Tasks`` and ``ATLAS_PANDA.jobsActive4``."""
    return reference.split(".")[-1]


# ``f"INSERT INTO ATLAS_PANDA.filesTable4 ({FileSpec.columnNames()}) "`` -- the
# statement names the spec class that owns the row, in the interpolation.
_COLUMN_NAMES = re.compile(r"^(\w+)\.column_?[Nn]ames\s*\(")
_TABLE_REFERENCE = re.compile(r"\b(?:INSERT\s+INTO|UPDATE|FROM)\s+([\w.{}]+)", re.IGNORECASE)


def declared_row_classes(
    func: ast.FunctionDef | ast.AsyncFunctionDef, spec_classes: set[str]
) -> dict[str, str]:
    """Return ``{table: spec class}`` the statements in *func* state outright.

    Stronger than inferring a table's class from its column names, and it
    settles the case inference cannot: ``filesTable4``'s widest ``UPDATE`` sets
    four columns that ``FileSpec`` and ``JediFileSpec`` both declare, so no
    column set ever separates them -- while its ``INSERT`` names ``FileSpec``
    in the f-string.

    Read here rather than during reassembly because reassembly deliberately
    blanks interpolations: they are usually the schema, and keeping them would
    make every statement unique.  This looks at the one interpolation that
    carries meaning instead of at all of them.
    """
    found: dict[str, str] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = ""
        classes: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text += value.value
                continue
            text += "{}"
            if isinstance(value, ast.FormattedValue):
                match = _COLUMN_NAMES.match(ast.unparse(value.value))
                if match and match.group(1) in spec_classes:
                    classes.append(match.group(1))
        if len(classes) != 1:
            continue
        reference = _TABLE_REFERENCE.search(text)
        if reference is None:
            continue
        table = _table_of(reference.group(1))
        # ``{}`` means the table name was itself interpolated, so the statement
        # does not say which table this is.
        if table and table != "{}":
            found[table] = classes[0]
    return found


_SELECT = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s+([\w{}.]+)", re.IGNORECASE | re.DOTALL)
_DELETE = re.compile(r"\bDELETE\s+FROM\s+([\w{}.]+)", re.IGNORECASE)


def reads(sql: str) -> list[tuple[str, list[str]]]:
    """Return ``(table, selected columns)`` for each ``SELECT`` in *sql*.

    Only the columns that are plain names are kept.  A projection built from
    expressions (``COUNT(1)``, ``CASE WHEN ...``) says what the query computes
    rather than what the row carries, and a boundary is about the latter.
    """
    found: list[tuple[str, list[str]]] = []
    for match in _SELECT.finditer(sql):
        columns = [
            part.strip().split(".")[-1]
            for part in match.group(1).split(",")
            if _IDENTIFIER.fullmatch(part.strip().split(".")[-1] or "_")
        ]
        found.append((_table_of(match.group(2)), columns))
    return found


# The whole FROM list, aliases and all: ``FROM {0}.JEDI_Tasks tabT,
# {0}.JEDI_AUX_Status_MinTaskID tabA``.
_FROM_LIST = re.compile(
    r"\bFROM\s+((?:[\w{}.]+(?:\s+\w+)?\s*,\s*)*[\w{}.]+(?:\s+\w+)?)", re.IGNORECASE
)

# ``WITH tmpTab AS (SELECT ...)`` -- a name the statement defines for itself.
_CTE = re.compile(r"\b(?:WITH|,)\s+(\w+)\s+AS\s*\(", re.IGNORECASE)


def joins(sql: str) -> list[str]:
    """Return the tables *sql* names in a ``FROM`` beyond the first.

    Separate from :func:`reads`, which answers *where the row came from* and so
    keeps naming the leading table.  This answers a different question -- *what
    bounds which rows can be seen at all* -- and a join partner is the usual
    way that bound is written::

        FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID

    A task below ``min_jediTaskID`` is invisible to that query no matter what
    its status is, so a stale ``JEDI_AUX_Status_MinTaskID`` silently narrows
    thirty-one functions at once -- which is a real stall this map could not
    explain, because ``reads`` stopped at the first table and the auxiliary one
    had never been seen.

    A name the statement defines for itself is not one of these.
    ``checkDuplication_JEDI`` builds ``WITH tmpTab AS (...)`` and then joins
    ``tmpTab`` to itself, which is a step in the query rather than a dependency
    on anything outside it.
    """
    defined = {match.group(1).lower() for match in _CTE.finditer(sql)}
    found: list[str] = []
    for match in _FROM_LIST.finditer(sql):
        parts = [part.strip() for part in match.group(1).split(",")]
        for part in parts[1:]:
            table = _table_of(part.split()[0]) if part.split() else ""
            # ``{}`` means the name was interpolated, so the statement does not
            # say which table this is.
            if not table or table == "{}" or table.lower() in defined:
                continue
            if table not in found:
                found.append(table)
    return found


def joined_columns(sql: str, table: str) -> list[str]:
    """Columns of *table* the statement names, found through its alias.

    The alias is how a join predicate refers to a table --
    ``tabA.min_jediTaskID`` -- so without resolving it the most useful thing
    about a join, *which column bounds the rows*, cannot be read.
    """
    found: dict[str, str] = {}
    for match in _FROM_LIST.finditer(sql):
        for part in match.group(1).split(","):
            words = part.strip().split()
            if len(words) != 2 or _table_of(words[0]) != table:
                continue
            for column in re.finditer(rf"\b{re.escape(words[1])}\.(\w+)", sql):
                found.setdefault(column.group(1).lower(), column.group(1))
    return sorted(found.values())


def deletes(sql: str) -> list[str]:
    """Return the tables *sql* deletes rows from.

    Worth reading separately because a ``DELETE`` immediately followed by an
    ``INSERT`` on a command table is not bookkeeping: it means a second command
    silently replaces one that was never picked up.
    """
    return [_table_of(match.group(1)) for match in _DELETE.finditer(sql)]


_PREDICATE_BIND = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:\w+\.)?([A-Za-z_]\w*)\s*(?:=|<>|!=)\s*(:\w+)", re.IGNORECASE
)
_PREDICATE_IN = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:\w+\.)?([A-Za-z_]\w*)\s+IN\s*\(([^)]*)\)", re.IGNORECASE
)
_PREDICATE_LITERAL = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:\w+\.)?([A-Za-z_]\w*)\s*(?:=|<>|!=)\s*'([^']*)'", re.IGNORECASE
)
_BIND_KEY = re.compile(r":\w+")
_QUOTED = re.compile(r"'([^']*)'")


def predicates(sql: str) -> list[tuple[str, str]]:
    """Return ``(column, bind key)`` for each predicate testing a bind.

    The other half of the same statements.  ``writes`` reads the ``SET`` clause
    to learn what a value becomes; this reads the ``WHERE`` clause to learn
    which rows were asked for, and the two together are what makes a state
    machine out of a pile of writes: a status nothing selects on is one nothing
    ever moves a task out of.

    Bind keys only; :func:`selected_literals` returns the values written into
    the statement itself, which need no lookup in Python.
    """
    found = [(column, key) for column, key in _PREDICATE_BIND.findall(sql)]
    for column, inner in _PREDICATE_IN.findall(sql):
        found.extend((column, key) for key in _BIND_KEY.findall(inner))
    return found


def selected_literals(sql: str) -> list[tuple[str, str]]:
    """Return ``(column, value)`` for each predicate testing a literal.

    Both forms occur on the same fields and the pair is the whole answer:
    ``WHERE status=:oldStatus`` puts the value in Python, ``AND status IN
    ('running','scouting')`` puts it in the statement.  Reading only one of
    them reports states nothing selects on that something plainly does.
    """
    found = [(column, value) for column, value in _PREDICATE_LITERAL.findall(sql)]
    for column, inner in _PREDICATE_IN.findall(sql):
        if "SELECT" in inner.upper():
            continue  # The quotes belong to the subquery, not to this column.
        found.extend((column, value) for value in _QUOTED.findall(inner))
    return found


def bound_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, varmap: str, key: str
) -> list[ast.Assign]:
    """Return the assignments filling ``<varmap>[<key>]`` in *func*.

    Several are normal and correct: a method that runs the same statement twice
    with different values has two, and both are outcomes of that write.  No
    attempt is made to pair a bind with one particular execution -- the binds
    reachable by key are exactly the values that write site can produce.
    """
    found: list[ast.Assign] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == varmap
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key
            ):
                found.append(node)
    return found
