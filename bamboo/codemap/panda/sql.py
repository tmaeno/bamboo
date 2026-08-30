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
import re
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field

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
# Bare words that are SQL, not columns.  This has to stay a list of *keywords*
# rather than of names that look like plumbing: ``T_TASK`` really does have a
# column called ``timeStamp``, and ``JEDI_Events`` one called ``event_offset``,
# both of which are copied into other columns.
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
        ``SET status='ready'`` -- decided in the statement itself.
    ``column``
        ``SET status=oldStatus`` -- carried from another column of the same
        row.  This is the form the recognizers were blind to, and it is not a
        curiosity: it is how a task returns from ``pending``, so without it the
        map cannot offer "the release did not fire" as a candidate at all.
    ``expression``
        ``NULL``, ``CURRENT_DATE``, ``nFiles+1``, a subquery.  A write, but not
        one that settles a subject to a traceable value.
    """

    kind: str = Field(..., description="bind | literal | column | expression")
    text: str = Field(
        ...,
        description="Bind key, literal value, source column, or the raw SQL, per ``kind``.",
    )


def classify(value: str) -> ColumnValue:
    """Classify the right-hand side of one SQL column assignment."""
    value = value.strip()
    if value.startswith(":"):
        return ColumnValue(kind="bind", text=value)
    quoted = _QUOTED.match(value)
    if quoted is not None:
        return ColumnValue(kind="literal", text=quoted.group(1))
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


def _literal(node: ast.expr) -> Optional[str]:
    """Render a string expression, marking unreadable parts as ``{}``.

    An f-string's interpolations are almost always the schema name, which does
    not change what the statement does; blanking them keeps the rest readable
    rather than discarding the whole statement.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{}"
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal(node.left), _literal(node.right)
        if left is None and right is None:
            return None
        return (left or "") + (right or "")
    return None


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
                text = _literal(node.value)
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
            text = _literal(node.value)
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


def variants(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> list[str]:
    """Return every statement local *name* can hold, one per ``=`` it is given.

    A name reassigned mid-function holds a different statement each time, and
    reading only the last quietly drops the others: ``reactivatePendingTasks_JEDI``
    picks between a timeout statement and a release statement through one
    variable, so keeping one of the two loses half of what the method does.

    They are returned separately rather than concatenated because concatenating
    would run one statement's ``SET`` clause into the next one's, which the
    column regexes cannot tell from a wider ``SET``.
    """
    runs: list[list[tuple[int, str, str, ast.stmt]]] = []
    for part in _parts(func, name):
        if part[1] == "=" or not runs:
            runs.append([])
        runs[-1].append(part)
    return [text for text in (_fold(run) for run in runs) if text]


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


def executions(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Execution]:
    """Return one :class:`Execution` per cursor execution in *func*.

    The call is what pairs a statement with its binds.  Reading them separately
    -- every statement in the function against every bind in the function --
    would attach a task's status write to a dataset's statement whenever both
    appear in one method, which in ``db_proxy_mods`` is most of them.
    """
    found: list[Execution] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"execute", "executemany"} or len(node.args) < 2:
            continue
        # ``execute(sqlU + comment, varMap)`` -- the comment is a tracing tag
        # appended at the call, so the statement is the left operand.
        expression = node.args[0]
        base = expression.left if isinstance(expression, ast.BinOp) else expression
        if not isinstance(base, ast.Name):
            continue
        varmap = node.args[1].id if isinstance(node.args[1], ast.Name) else None
        for text in variants(func, base.id):
            found.append(Execution(sql=text, varmap=varmap, variable=base.id, call=node))
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
                SqlWrite(kind="update", table=_table_of(match.group(1)), columns=columns)
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
