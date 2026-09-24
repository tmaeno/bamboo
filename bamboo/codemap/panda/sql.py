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

from bamboo.codemap.panda.pathcond import (
    attach_parents,
    can_reach,
    exclusive,
    path_condition,
)
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


class Span(NamedTuple):
    """Where one fragment landed in the statement it helped build.

    The unit an inline value is anchored at.  A statement is assembled from
    fragments and each fragment carries its own path condition, so the fragment
    holding the value is the one that says *why* it was written -- but which
    fragment that is cannot be decided fragment by fragment, because an
    ``INSERT``'s column list and value list are two parenthesised runs and the
    correspondence between them is positional.  Reading the assembled variant
    and mapping the match back here answers both at once.
    """

    node: ast.stmt
    start: int
    end: int


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


def _compatible_sets(
    parts: list[tuple[int, str, str, ast.stmt]], start: int
) -> list[frozenset[int]]:
    """Return the fragment sets a single run of the code can append, largest first.

    One set per statement a reader can meet.  The test between two fragments is
    :func:`pathcond.exclusive`, reading the negations the path condition already
    carries; a statement is then a set no two of whose members contradict, and
    which nothing else can be added to -- a maximal clique of the compatibility
    graph, enumerated with Bron-Kerbosch.

    **Connected components were the wrong shape, and the corpus says where.**
    Grouping the exclusive pairs and taking one member per group is right for
    an ``if``/``elif``/``else``, whose arms are pairwise exclusive, and wrong
    the moment two unrelated ``if``s touch.  ``insertTaskParams_JEDI`` builds
    one INSERT across a backend test and a ``parent_tid`` test, and the inner
    ``if`` repeats the outer one -- so one arm of each is exclusive with one arm
    of the other and all five fragments collapse into a single alternative.
    Exactly one of the five then survives per statement, and every statement
    comes out with an opening parenthesis and no closing one, or the reverse.
    ``writes`` reads none of them: the map reported six statements it could name
    no column of, and ``ATLAS_DEFT.T_TASK`` lost ``priority``.

    A fragment under no test is compatible with everything, so it lands in every
    set without being special-cased.
    """
    indices = list(range(start, len(parts)))
    conditions = {index: path_condition(parts[index][3]) for index in indices}
    agrees = {
        index: {
            other
            for other in indices
            if other != index and not exclusive(conditions[index], conditions[other])
        }
        for index in indices
    }
    if all(len(agrees[index]) == len(indices) - 1 for index in indices):
        # Nothing contradicts anything: one statement, which is most runs.
        return [frozenset(indices)]

    found: list[frozenset[int]] = []

    def expand(chosen: set[int], candidates: set[int], refused: set[int]) -> None:
        if len(found) > _MAX_BRANCH_VARIANTS:
            # The caller folds past the cap; stopping here keeps a pathological
            # run from costing more than the answer it would be given.
            return
        if not candidates and not refused:
            found.append(frozenset(chosen))
            return
        pivot = max(candidates | refused, key=lambda one: len(agrees[one] & candidates))
        for one in sorted(candidates - agrees[pivot]):
            expand(chosen | {one}, candidates & agrees[one], refused & agrees[one])
            candidates = candidates - {one}
            refused = refused | {one}

    expand(set(), set(indices), set())
    return sorted(found, key=lambda members: sorted(members))


#: A hole standing where a whole table reference goes.  The keyword settles it
#: from the left and the character after the hole settles it from the right:
#: ``FROM {schemaJEDI}.JEDI_Tasks`` interpolates the *schema* and is followed by
#: a dot, which is why reassembly blanks interpolations in the first place.
#:
#: The schema may itself be a hole, so the keyword need not be the hole's
#: immediate neighbour: ``INSERT INTO {schema}.{table}`` puts a filled-in
#: ``{}.`` between them.  Requiring adjacency lost the two statements that
#: create a job row -- ``INSERT INTO {}.{} (...) RETURNING PandaID`` -- and so
#: the map could not say where the system's most-read row comes into existence.
_BEFORE_TABLE = re.compile(
    r"\b(?:FROM|UPDATE|INTO|JOIN)\s+(?:(?:\{\}|[A-Za-z_][\w$]*)\.)?$", re.IGNORECASE
)


def _table_holes(text: str) -> set[int]:
    """Which of *text*'s ``{}`` holes stand for a whole table name.

    Position, not resolvability, is what this decides, and it decides it
    narrowly on purpose.  :func:`rendered_text` serves two readers that want
    opposite things from the same ``{}``: this one wants the hole *filled*,
    because a statement needs its table named, while ``DiagnosticTemplate``
    wants it *kept*, because the unfilled frame is the part a line observed in
    production can be matched against.  Filling every hole a value could be
    recovered for fills ninety-nine more across the corpus, nearly all of them
    in log messages and dataset names -- and a template expanded over its own
    values no longer matches the line that carried a different one.
    """
    found: set[int] = set()
    position = index = 0
    while (hole := text.find(_BARE_FIELD, position)) >= 0:
        if text[hole + 2 : hole + 3] != "." and _BEFORE_TABLE.search(text[:hole]):
            found.add(index)
        position, index = hole + 2, index + 1
    return found


def _hole_expressions(node: ast.expr) -> Optional[list[ast.expr]]:
    """The expressions behind each ``{}`` :func:`rendered_text` left, in order.

    ``None`` when the alignment is not assured -- a ``.format`` call renders
    its own field markers as holes too, and a hole filled with the wrong
    expression's value would put an invented table name in the statement with
    the same confidence as one the code wrote.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return []
    if isinstance(node, ast.JoinedStr):
        found: list[ast.expr] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
            if not isinstance(value, ast.FormattedValue):
                return None
            found.append(value.value)
        return found
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _hole_expressions(node.left), _hole_expressions(node.right)
        return None if left is None or right is None else left + right
    return None


def _fill(text: str, values: dict[int, str]) -> str:
    """Replace the numbered holes of *text*, leaving the rest as they are."""
    out: list[str] = []
    position = index = 0
    while (hole := text.find(_BARE_FIELD, position)) >= 0:
        out.append(text[position:hole])
        # A filled name carries its own schema hole -- ``{}.jobsActive4`` -- so
        # the scan continues past the original hole rather than over what
        # replaced it.
        out.append(values.get(index, _BARE_FIELD))
        position, index = hole + 2, index + 1
    out.append(text[position:])
    return "".join(out)


def _table_choices(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
) -> tuple[list[tuple[list[ast.expr], set[int]]], dict[str, list[str]]]:
    """Return each fragment's table holes, and the names each one can hold.

    Keyed by the expression rather than by the hole, because one loop variable
    interpolated into two fragments of a statement takes *one* value per pass;
    treating the holes independently would cross them and read a union of two
    tables as a single statement over both.
    """
    per_part: list[tuple[list[ast.expr], set[int]]] = []
    choices: dict[str, list[str]] = {}
    for _line, _operator, text, node in parts:
        value = node.value if isinstance(node, (ast.Assign, ast.AugAssign)) else None
        expressions = _hole_expressions(value) if value is not None else None
        if expressions is None or len(expressions) != text.count(_BARE_FIELD):
            per_part.append(([], set()))
            continue
        holes = {
            hole for hole in _table_holes(text) if _literal_values(func, expressions[hole])
        }
        per_part.append((expressions, holes))
        for hole in holes:
            choices.setdefault(
                ast.unparse(expressions[hole]), _literal_values(func, expressions[hole])
            )
    return per_part, choices


def _assignments(choices: dict[str, list[str]]) -> list[dict[str, str]]:
    """One mapping of expression to table name per combination the code allows."""
    if not choices:
        return [{}]
    names = sorted(choices)
    return [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*(choices[name] for name in names))
    ]


def _fold_filled(
    parts: list[tuple[int, str, str, ast.stmt]],
    per_part: list[tuple[list[ast.expr], set[int]]],
    assignment: dict[str, str],
    dropped: set[int],
) -> tuple[str, list[Span]]:
    """:func:`_fold`, with each fragment's table holes filled from *assignment*.

    Returns the statement and where in it each fragment landed.  The spans are
    taken here rather than recomputed because they have to describe the text
    *after* filling: ``_fill`` changes the length of any fragment carrying a
    hole, so offsets measured on the unfilled text point at the wrong column as
    soon as a table name is interpolated.
    """
    assembled = ""
    spans: list[Span] = []
    for index, (_line, operator, text, node) in enumerate(parts):
        if index in dropped:
            continue
        expressions, holes = per_part[index]
        if holes and assignment:
            text = _fill(
                text,
                {
                    hole: assignment[ast.unparse(expressions[hole])]
                    for hole in holes
                    if ast.unparse(expressions[hole]) in assignment
                },
            )
        if operator == "=":
            assembled = ""
            spans = []
        spans.append(Span(node, len(assembled), len(assembled) + len(text)))
        assembled += text
    return assembled, spans


def _run_variants(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    parts: list[tuple[int, str, str, ast.stmt]],
    name: str,
) -> list[tuple[str, list[Span]]]:
    """Return the statements one ``=``-started run of fragments can produce.

    Each comes with the spans saying which fragment contributed which stretch
    of it -- see :func:`variant_spans` for why that is the interesting part.

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

    **A table name the loop supplies splits the run the same way.**  A hole
    standing for a whole table reference is not the schema -- ``FROM
    {tableName}`` where ``tableName`` comes from a local list of names the
    source writes out is four statements, one per table, and read as one it is
    a statement about a table called ``{}``.  That is what hid the helper
    fetching a task's jobs: the rows it selects belong to a table the map had
    no name for.
    """
    start = 1 if parts and parts[0][1] == "=" else 0
    statements = _compatible_sets(parts, start)
    per_part, choices = _table_choices(func, parts)
    assignments = _assignments(choices)
    total = len(statements) * len(assignments)
    if total > _MAX_BRANCH_VARIANTS:
        # Folded as before rather than split into an arbitrary subset: a
        # truncated list of statements reads as the complete one.  Said out
        # loud, because a silent cap is indistinguishable from full coverage.
        logger.warning(
            "%s at line %d is built across %d combinations of branch and table, over "
            "the cap of %d: read as one folded statement, so a conditionally assembled "
            "column or an interpolated table name may be misread here",
            name,
            parts[0][0],
            total,
            _MAX_BRANCH_VARIANTS,
        )
        return [_fold_filled(parts, per_part, {}, set())]

    every = set(range(start, len(parts)))
    found: list[tuple[str, list[Span]]] = []
    seen: set[str] = set()
    for members in statements:
        dropped = every - members
        for assignment in assignments:
            text, spans = _fold_filled(parts, per_part, assignment, dropped)
            if text and text not in seen:
                seen.add(text)
                found.append((text, spans))
    return found


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
    return [text for run in runs for text, _spans in _run_variants(func, run, name) if text]


def variant_spans(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[tuple[str, list[Span]]]:
    """:func:`variants`, with each statement's fragments and where they landed.

    **The variant is the unit, not the fragment and not the whole function.**
    A fragment is where a path condition lives, so it is tempting to read one
    at a time; that is what ``assigns_in`` was doing, and it cannot see an
    ``INSERT`` whose columns and values were appended on different lines.  It
    is equally tempting to group fragments that share a guard, but::

        sql  = "SELECT ... "          # no test above it
        sql += "WHERE a=:a "          # no test above it
        if wanted:
            sql += "AND b=:b "        # under ``wanted``

    has two statements a reader can meet -- with and without the third
    fragment -- and the group ``["AND b=:b "]`` is neither of them.  It is not
    a statement at all; nothing ever runs it alone.

    :func:`_run_variants` already enumerates the statements properly, splitting
    on mutual exclusion and keeping an ``if`` with no ``else`` in every variant.
    All that was missing is which fragment contributed which stretch, so this
    returns that rather than inventing a second decomposition.
    """
    runs: list[list[tuple[int, str, str, ast.stmt]]] = []
    for part in _parts(func, name):
        if part[1] == "=" or not runs:
            runs.append([])
        runs[-1].append(part)
    return [
        (text, spans)
        for run in runs
        for text, spans in _run_variants(func, run, name)
        if text
    ]


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
    """One statement run against the database, in either of the two forms.

    ``cursor.execute(<statement>, <binds>)`` is the proxy's spelling.  The
    daemon layer speaks through the task buffer instead --
    ``taskBuffer.querySQLS(sql, var_map)`` -- and ``window`` is what the
    second form needs that the first does not.
    """

    sql: str
    varmap: Optional[str]
    variable: str
    call: ast.Call
    window: Optional[tuple[int, int]] = None
    """Lines within which this run's binds were filled, for the forwarded form.

    ``None`` for ``execute``, and that is not an omission.  There the binds
    reachable by key are exactly the values the write site can produce, so
    reading them all over-reads and an over-read is safe.  Pairing a statement
    with a varmap makes an over-read a *mis*-read: ``copyArchive.main`` binds
    ``sql`` 23 times and ``:jobStatus`` to 11 values, and the unwindowed
    reading would claim all 253.  The window is not guessed -- the same
    function writes ``var_map = {}`` 31 times, so the code declares its own
    block boundaries and this only reads them.
    """


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


def _is_literal(node: ast.expr) -> bool:
    """Whether *node* is a plain string constant."""
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _gluing_fstring(node: ast.expr) -> Optional[list[ast.expr]]:
    """The operands of an f-string that only joins a statement to its tag.

    ``execute(f"{sql} {comment}")`` is ``execute(sql + " " + comment)`` written
    another way.  Read as one opaque expression it has no head that names a
    statement, so the execution was dropped whole rather than truncated -- and
    a statement that is never read leaves nothing behind for a graph invariant
    to catch.  Five went that way, one of them the only place
    ``JOBSDEFINED_SHARE_STATS`` is named.

    Recognised by its literal parts being nothing but whitespace.  An f-string
    that carries text of its own *is* the statement, and splitting that one
    would keep only the piece before the first hole -- a truncation no reader
    could tell from a short statement.
    """
    if not isinstance(node, ast.JoinedStr):
        return None
    if not any(isinstance(value, ast.FormattedValue) for value in node.values):
        return None
    if any(
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.strip()
        for value in node.values
    ):
        return None
    return [
        value.value if isinstance(value, ast.FormattedValue) else value
        for value in node.values
    ]


def _concatenated(node: ast.expr) -> list[ast.expr]:
    """Flatten a chain of ``+`` into its operands, left to right."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _concatenated(node.left) + _concatenated(node.right)
    glued = _gluing_fstring(node)
    if glued is not None:
        return [operand for value in glued for operand in _concatenated(value)]
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


def _braces_are_literal(sources: list[ast.stmt]) -> bool:
    """Whether every ``{}`` the fragments in *sources* wrote is a placeholder.

    Rendered text spells an f-string interpolation ``{}`` as well, so a bare
    brace is either the hole a schema went into or a field a ``.format`` at the
    call fills.  The fragments settle it without a guess: if none of them
    interpolates, every brace came from a literal.

    **The fragments of this statement, not every assignment to the name.**
    Asked across the whole function it is a different question, and a bigger
    one than the answer can bear: ``copyArchive.main`` assigns ``sql``
    forty-four times and exactly one of those is an f-string, which disqualified
    the other forty-three.  Among them is the query whose table the loop
    supplies, so ``JOBS_SHARE_STATS`` and ``JOBSDEFINED_SHARE_STATS`` were read
    as ``FROM ATLAS_PANDA.{}`` -- naming no table at all.
    """
    return not any(
        isinstance(node, (ast.Assign, ast.AugAssign)) and _interpolates(node.value)
        for node in sources
    )


def _elements(node: ast.expr) -> Optional[list[str]]:
    """The strings a literal sequence holds, or ``None`` if one is not written out.

    Rendered rather than required to be constant: the corpus spells a table
    name ``f"{panda_config.schemaPANDA}.jobsActive4"``, and the schema half is
    exactly the part no reader here needs -- :func:`_table_of` drops it.  One
    element nobody can read sinks the whole sequence, because half a list of
    tables read as the list is a statement about tables the code never names.
    """
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    rendered = [rendered_text(element) for element in node.elts]
    return None if any(text is None for text in rendered) else [str(t) for t in rendered]


def _sequence_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> Optional[list[str]]:
    """The strings local list *name* holds, across ``=`` and ``+=``.

    ``None`` rather than a partial answer wherever the list is built some other
    way -- an ``append``, or a sequence that is not a literal.  A missing
    element is not a smaller answer here: it is a statement the map says the
    code runs over a set of tables it does not.
    """
    found: list[str] = []
    assigned = False
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            elements = _elements(node.value)
            if elements is None:
                return None
            found.extend(elements)
            assigned = True
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            elements = _elements(node.value) if isinstance(node.op, ast.Add) else None
            if elements is None:
                return None
            found.extend(elements)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "extend", "insert"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            return None
    return found if assigned else None


def _rewritten_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.expr,
    seen: frozenset[str],
) -> list[str]:
    """The strings ``re.sub(<pattern>, <replacement>, <name>)`` can produce.

    Computed, not inferred.  ``peekJobLog`` derives three archive table names
    from one -- ``re.sub("jobsArchived", "metaTable_ARCH", table)`` -- and all
    three parts are written down: the pattern, the replacement, and the values
    the third argument already resolves to.  Running the substitution on those
    values is the same reading the rest of this function does, not a new guess.

    All three have to be readable.  A pattern or replacement the map cannot see
    would make the result invented, and an invented table name is attributed
    with exactly the confidence of one the code states.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sub"
        and len(node.args) >= 3
    ):
        return []
    pattern, replacement = node.args[0], node.args[1]
    if not (_is_literal(pattern) and _is_literal(replacement)):
        return []
    try:
        return [
            re.sub(pattern.value, replacement.value, value)
            for value in _literal_values(func, node.args[2], seen)
        ]
    except re.error:
        # A pattern that will not compile says nothing about the name, and
        # raising here would sink every statement in the function.
        return []


def _default_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> list[str]:
    """The string a parameter falls back to, where the signature states it.

    ``insertDataset(self, dataset, tablename="ATLAS_PANDA.Datasets")`` names
    the table it writes, and names it in the function's own signature -- the
    only caller passes nothing.  Reading a parameter with no default would
    need the call graph, which this does not have; reading the default needs
    only the function already in hand.
    """
    arguments = func.args.posonlyargs + func.args.args
    paired = list(
        zip(
            arguments[len(arguments) - len(func.args.defaults) :],
            func.args.defaults,
            strict=True,
        )
    )
    paired += [
        (argument, default)
        for argument, default in zip(
            func.args.kwonlyargs, func.args.kw_defaults, strict=True
        )
        if default is not None
    ]
    return [
        default.value
        for argument, default in paired
        if argument.arg == name and _is_literal(default)
    ]


def _literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    expression: ast.expr,
    seen: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the strings *expression* can hold, where the source writes them out.

    Five forms, all of which put the names in plain sight: the argument is a
    literal; it is the target of a ``for`` over a literal sequence --
    ``for table in ("ATLAS_PANDA.jobsDefined4", "ATLAS_PANDA.jobsActive4")``;
    or over a local list assembled from literals, which is how the same loop is
    written when a flag decides whether the archive tables are in it; it is
    rewritten from such a value by :func:`_rewritten_values`; or it is a
    parameter whose default the signature states.
    Anything else returns nothing, and the placeholder is left as written: a
    table name invented here would be attributed to a spec class with the same
    confidence as one the code states.

    *seen* stops a name that is defined in terms of itself -- ``x = re.sub(p,
    r, x)`` -- from recurring forever.
    """
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return [expression.value]
    rewritten = _rewritten_values(func, expression, seen)
    if rewritten:
        return rewritten
    if not isinstance(expression, ast.Name) or expression.id in seen:
        return []
    deeper = seen | {expression.id}
    found: list[str] = _default_values(func, expression.id)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == expression.id
        ):
            elements = (
                _sequence_values(func, node.iter.id)
                if isinstance(node.iter, ast.Name)
                else _elements(node.iter)
            )
            if elements is not None:
                found.extend(elements)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == expression.id
            for target in node.targets
        ):
            if _is_literal(node.value):
                found.append(node.value.value)
            else:
                found.extend(_rewritten_values(func, node.value, deeper))
    return found


def _substituted(
    text: str,
    sources: list[ast.stmt],
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
        elif index == 0 and _BARE_FIELD in filled[0] and _braces_are_literal(sources):
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
    glued = _gluing_fstring(expression)
    if glued is not None and sum(1 for value in glued if not _is_literal(value)) > 1:
        # Same rule as the ``+`` spelling below: the last operand is the
        # tracing tag the call appends, so the statement is everything before
        # it.  Only when there is something to drop -- ``f"{sql}"`` carries no
        # tag, and taking one off would leave nothing.
        last = max(index for index, value in enumerate(glued) if not _is_literal(value))
        expression = ast.JoinedStr(
            values=[
                value if _is_literal(value) else ast.FormattedValue(
                    value=value, conversion=-1, format_spec=None
                )
                for value in glued[:last]
            ]
        )
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

    **A statement is not run by every call in the function.**  The variants of
    a name are read across the whole function, and pairing each with every
    execution in it is a cross product, not a reading.  ``copyArchive.main``
    assigns ``sql`` forty-four times and runs it twenty-three: 21 of the 31
    statements this build reported as having a table name supplied at run time
    were *one* statement claimed at 21 call sites that never run it, and its
    table is resolved at the call that does.  A fragment below the call, or
    under an arm the call is not in, cannot be part of what the call runs --
    :func:`pathcond.can_reach` is the predicate the walk already uses to decide
    whether a binding can be what an arm read, and this is the same question
    asked of a different pair.
    """
    # ``_compatible_sets`` reads the arms a fragment sits under off the ancestor
    # chain, and two of this function's callers -- ``boundary`` and ``trigger``
    # -- never attached one.  A tree without parents does not raise: every
    # fragment reads as unconditional, so the two arms of one ``if`` are
    # concatenated into a statement nobody runs and the two the code does run
    # are never seen.  Idempotent, so attaching again costs a walk and nothing
    # else.
    attach_parents(func)
    found: list[Execution] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        forwarded = node.func.attr in _FORWARDED
        if not forwarded and node.func.attr not in {"execute", "executemany"}:
            continue
        if len(node.args) < 2:
            continue
        expression = node.args[0]
        statement, fillers, percent = _call_site_fill(func, expression)
        operands = _concatenated(statement)
        sources: dict[str, list[ast.stmt]] = {}
        reaching: dict[str, list[ast.stmt]] = {}
        if isinstance(operands[0], ast.Name):
            base = operands[0]
            head = []
            for text, spans in variant_spans(func, base.id):
                if not text:
                    continue
                head.append(text)
                # Which fragments built this variant, so that filling its holes
                # asks about them and not about every other statement the name
                # has held.
                sources.setdefault(text, [span.node for span in spans])
                # The same fragments, pooled across variants instead of kept
                # for the first.  Two arms of one ``if`` often build the same
                # statement, and asking the first arm's copy whether it reaches
                # a call in the second answers no: ``activateJob`` loses
                # ``sqlF`` and ``sqlJob`` that way, four statements over the
                # corpus.  Filling cannot pool them -- ``_braces_are_literal``
                # is answered per statement, for the reason its docstring
                # gives -- so this is a second reading and not a wider one.
                reaching.setdefault(text, []).extend(span.node for span in spans)
            held = base.id
        elif forwarded:
            # The daemons write the statement into the call.  ``execute`` is
            # left requiring a name on purpose: widening it would move the
            # stored map and make ``diff-map`` report condition drift where
            # there is none.  Measured, that costs 36 statements on the
            # ``execute`` side and is a deliberate debt, not an oversight.
            rendered = rendered_text(operands[0])
            head = [rendered] if rendered is not None else []
            held = ""
        else:
            continue
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
        window = (
            _binding_window(func, varmap, node.lineno)
            if forwarded and varmap is not None
            else None
        )
        for combination in itertools.product(*pieces):
            built_by = reaching.get(combination[0], [])
            if built_by and not any(can_reach(fragment, node) for fragment in built_by):
                # Nothing that built this statement can be part of what this
                # call runs.  Guarded on having a fragment at all, because the
                # daemon layer writes the statement into the call and has none:
                # "no fragment reaches here" and "there is no fragment" are
                # different answers, and reading the second as the first would
                # drop 30 pairs.
                continue
            assembled = "".join(combination)
            for text in _substituted(
                assembled, sources.get(combination[0], []), fillers, percent
            ):
                found.append(
                    Execution(
                        sql=text,
                        varmap=varmap,
                        variable=held,
                        call=node,
                        window=window,
                    )
                )
    return found


#: How the daemon layer runs a statement.  A second execution form rather than
#: a second reading: the statement, the binds and the pairing between them are
#: the same three things, reached through the task buffer instead of a cursor.
_FORWARDED = frozenset({"querySQL", "querySQLS"})


def _binding_window(
    func: ast.FunctionDef | ast.AsyncFunctionDef, varmap: str, line: int
) -> tuple[int, int]:
    """Lines whose ``<varmap>[...]`` assignments belong to the run at *line*.

    Read off the code rather than guessed.  ``var_map = {}`` restarts the map,
    so the nearest one above the call is the block boundary the author wrote;
    ``copyArchive.main`` writes it 31 times for 23 statements.  With no such
    assignment the window opens at the top of the function, which is the same
    answer the unwindowed reading gives.
    """
    start = func.lineno
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or node.lineno >= line:
            continue
        if not isinstance(node.value, (ast.Dict, ast.DictComp)) and not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "dict"
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == varmap:
                start = max(start, node.lineno)
    return start, line


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
        columns = {name: supplied for name, supplied, _at in _insert_supplies(match)}
        if columns:
            found.append(
                SqlWrite(kind="insert", table=_table_of(match.group(1)), columns=columns)
            )
    return found


def _insert_supplies(match: re.Match) -> list[tuple[str, ColumnValue, tuple[int, int]]]:
    """``(column, value, where the value sits)`` for one ``INSERT`` match.

    One reading for the two questions asked of an ``INSERT``: which columns it
    writes, and where in the statement each value was written.  Kept together
    because pairing a column with a value here is positional -- the names are
    in one parenthesised run and the values in another -- and a second reading
    that paired them its own way would be free to pair them differently.

    The offsets are into the same string the match was taken from, so a caller
    that matched on an assembled variant can map a value back to the fragment
    that contributed it.
    """
    names = [c.strip().split(".")[-1] for c in match.group(2).split(",")]
    supplies: list[tuple[str, ColumnValue, tuple[int, int]]] = []
    at = match.start(3)
    for name, value in zip(names, match.group(3).split(","), strict=False):
        start, end = at, at + len(value)
        at = end + 1  # the comma the split consumed
        name = name.strip()
        if not _IDENTIFIER.fullmatch(name):
            continue
        supplied = classify(value)
        if supplied.kind == "column":
            # An INSERT has no prior row to copy from, so a bare word in its
            # VALUES list is a sequence or a function, not a source column.
            # Calling it one would invent a passthrough edge.
            supplied = ColumnValue(kind="expression", text=supplied.text)
        supplies.append((name, supplied, (start, end)))
    return supplies


def supplied_at(
    statement: str, kind: str, column: str, supplied: ColumnValue
) -> Optional[int]:
    """Where *statement* writes *supplied* into *column*, as an end offset.

    ``None`` when this statement does not write that value into that column.

    **The reading is chosen by kind rather than tried in turn.**  An UPSERT
    builds its ``INSERT`` and its ``UPDATE`` through one local -- seven
    functions here do -- and both spell the same column with the same value, so
    a reading that accepted either would answer for the ``INSERT`` when asked
    about the ``UPDATE``.  That is the same mis-answer this whole path exists
    to remove, arriving from the other side.
    """
    if kind == "insert":
        for match in _INSERT.finditer(statement):
            for name, value, (_start, end) in _insert_supplies(match):
                if name == column and value == supplied:
                    return end
        return None
    # Everything from the first ``WHERE`` on is a predicate, which uses the
    # same ``column = value`` spelling as an assignment -- see ``assigns_in``.
    where = _WHERE.search(statement)
    head = statement[: where.start()] if where else statement
    for match in _ASSIGNMENT.finditer(head):
        if match.group(1) == column and classify(match.group(2)) == supplied:
            return match.end()
    return None


def fragment_holding(spans: list[Span], offset: int) -> Optional[ast.stmt]:
    """The fragment whose stretch of the statement contains *offset*."""
    for span in spans:
        if span.start < offset <= span.end:
            return span.node
    return None


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


# The whole FROM list, aliases and all: ``FROM {0}.JEDI_Tasks tabT,
# {0}.JEDI_AUX_Status_MinTaskID tabA``.
_FROM_LIST = re.compile(
    r"\bFROM\s+((?:[\w{}.]+(?:\s+\w+)?\s*,\s*)*[\w{}.]+(?:\s+\w+)?)", re.IGNORECASE
)

# ``WITH tmpTab AS (SELECT ...)`` -- a name the statement defines for itself,
# and the second and later ones after a comma.  The word boundary goes before
# ``WITH`` only: a comma is not a word character, so ``\b,`` demands one to its
# left and ``), wq_results AS (`` -- the spelling every multi-clause WITH in
# this corpus uses -- did not match.
_CTE = re.compile(r"(?:\bWITH|,)\s+(\w+)\s+AS\s*\(", re.IGNORECASE)

# ``JOIN {0}.JEDI_WORK_QUEUE jwq ON ...`` -- the other spelling of a join
# partner, which ``_FROM_LIST`` cannot see because it is not in the comma list.
_JOIN = re.compile(r"\bJOIN\s+([\w{}.]+)", re.IGNORECASE)

#: Oracle's one-row pseudo-table.  ``SELECT <seq>.nextval FROM dual`` asks a
#: sequence for a number; ``dual`` sits where a table goes and holds none of
#: PanDA's rows.  ``WrappedCursor`` deletes ``FROM dual`` outright when the
#: backend is not Oracle, which settles what it is.
_PSEUDO_TABLES = frozenset({"dual"})


def _names_a_table(name: str, defined: set[str]) -> bool:
    """Whether *name*, written where a table goes, names one that stores rows.

    Two ways it can fail to, and both belong wherever a table position is
    read: the statement may have introduced the name itself in a ``WITH``
    clause, and it may be Oracle's ``dual``.  Either one counted as a table is
    read by the corpus and written by none of it, which is the exact shape
    :func:`~bamboo.codemap.panda.recognizers.boundary.tables_never_written`
    turns into a claim that some other system maintains it.

    An interpolated name -- a bare ``{}`` -- is *not* decided here, because the
    two callers want opposite things from it.  :func:`joins` drops it, since a
    partner nobody can name cannot be said to bound anything; :func:`reads`
    keeps it, since the reader downstream counts those and the build report
    says how many statements and which functions.  Dropping it there would
    turn a counted gap into silence.
    """
    folded = name.lower()
    return bool(name) and folded not in defined and folded not in _PSEUDO_TABLES


_SELECT = re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\s+([\w{}.]+)", re.IGNORECASE | re.DOTALL)
_DELETE = re.compile(r"\bDELETE\s+FROM\s+([\w{}.]+)", re.IGNORECASE)


def reads(sql: str) -> list[tuple[str, list[str]]]:
    """Return ``(table, selected columns)`` for each ``SELECT`` in *sql*.

    Only the columns that are plain names are kept.  A projection built from
    expressions (``COUNT(1)``, ``CASE WHEN ...``) says what the query computes
    rather than what the row carries, and a boundary is about the latter.
    """
    defined = {match.group(1).lower() for match in _CTE.finditer(sql)}
    found: list[tuple[str, list[str]]] = []
    for match in _SELECT.finditer(sql):
        table = _table_of(match.group(2))
        if not _names_a_table(table, defined):
            continue
        columns = [
            part.strip().split(".")[-1]
            for part in match.group(1).split(",")
            if _IDENTIFIER.fullmatch(part.strip().split(".")[-1] or "_")
        ]
        found.append((table, columns))
    return found


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

    def offer(reference: str) -> None:
        table = _table_of(reference)
        # ``{}`` means the name was interpolated, so the statement does not say
        # which table this is, and a partner nobody can name bounds nothing.
        if table == "{}" or not _names_a_table(table, defined):
            return
        if table not in found:
            found.append(table)

    for match in _FROM_LIST.finditer(sql):
        parts = [part.strip() for part in match.group(1).split(",")]
        for part in parts[1:]:
            offer(part.split()[0] if part.split() else "")
    for match in _JOIN.finditer(sql):
        offer(match.group(1))
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


#: ``INSERT INTO <table>`` with nothing required after it.  ``_INSERT`` wants a
#: column list and a ``VALUES`` list because it is answering which columns get
#: which values; this answers only which kind of row is being made, and the
#: corpus has five statements where the first question has no answer and the
#: second does -- two ``INSERT ... SELECT``, one positional insert with no
#: column list, and two whose column list is interpolated.
_INSERT_HEAD = re.compile(r"\bINSERT\s+INTO\s+([\w{}.]+)", re.IGNORECASE)


def creates(sql: str) -> list[str]:
    """Return the tables *sql* inserts rows into.

    Separate from :func:`writes` on purpose.  *Which columns get which values*
    and *which kind of row is being made* are two questions, and deriving the
    second from the first made a table look externally maintained whenever the
    first could not be answered: ``TASK_ATTEMPTS`` is filled by an
    ``INSERT ... SELECT`` in ``log_task_attempt_start``, and the map said
    something outside PanDA keeps it current.
    """
    return [_table_of(match.group(1)) for match in _INSERT_HEAD.finditer(sql)]


def updates(sql: str) -> list[str]:
    """Return the tables *sql* changes rows in.

    Matched with :data:`_UPDATE`, which requires the ``SET``, rather than with
    a bare ``UPDATE <name>``: ``SELECT ... FOR UPDATE NOWAIT`` otherwise reads
    as a statement updating a table called ``nowait``.
    """
    return [_table_of(match.group(1)) for match in _UPDATE.finditer(sql)]


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


class Bind(NamedTuple):
    """One value filling one placeholder, and the statement that put it there.

    ``value`` is what the placeholder receives; ``site`` is the statement to
    ask for the guards that reached it.  They are the same node for
    ``varMap[":x"] = v`` and different ones for a dict literal, which is the
    whole reason this is a pair rather than the assignment.
    """

    value: ast.expr
    site: ast.stmt


def bound_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    varmap: str,
    key: str,
    window: Optional[tuple[int, int]] = None,
) -> list[Bind]:
    """Return the assignments filling ``<varmap>[<key>]`` in *func*.

    Several are normal and correct: a method that runs the same statement twice
    with different values has two, and both are outcomes of that write.  With
    no *window* no attempt is made to pair a bind with one particular
    execution -- the binds reachable by key are exactly the values that write
    site can produce, and that over-read is safe because the statement is not
    being paired with them.

    *window* is for the forwarded form, where it is.  ``querySQLS(sql,
    var_map)`` names both halves in one call, so reading every bind in the
    function would have ``copyArchive.main`` claim its 23 statements each
    select all 11 values of ``:jobStatus``.  The bounds come from the code:
    see :func:`_binding_window`.

    **Two spellings, one meaning.**  ``varMap[":status"] = v`` and ``var_map =
    {":status": v}`` bind the same placeholder to the same value, and reading
    only the first left 176 of the corpus's 1261 written columns unexplained
    -- 56% of everything this slice could not account for, and the whole of
    the harvester, worker and data-carousel side, which writes its binds as
    dict literals throughout.  A :class:`Bind` rather than the assignment
    itself because the two forms do not share a node: what the caller needs is
    the value, and the statement it sits in for the guards that reached it.
    """
    found: list[Bind] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if window is not None and not window[0] <= node.lineno <= window[1]:
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == varmap
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key
            ):
                found.append(Bind(node.value, node))
            elif isinstance(target, ast.Name) and target.id == varmap:
                # The dict literal form.  Only constant keys: a computed one
                # (``var_map[f":{column}"] = val``, which the worker modules
                # also use) names a placeholder the statement itself does not
                # spell either, so there is nothing to pair it with.
                if not isinstance(node.value, ast.Dict):
                    continue
                found.extend(
                    Bind(value, node)
                    for spelled, value in zip(node.value.keys, node.value.values, strict=True)
                    if isinstance(spelled, ast.Constant) and spelled.value == key
                )
    return found
