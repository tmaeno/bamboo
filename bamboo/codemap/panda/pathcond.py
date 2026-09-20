"""Where a write sits in the code: its enclosing scope and its path condition.

Shared by the recognizers rather than owned by one of them.  Both the attribute
slice and the SQL slice need the same two answers -- which function contains
this write, and which conditions had to hold to reach it -- and a recognizer
importing another recognizer to get them is the kind of dependency that turns
into a cycle the first time either grows.

**A path condition records what decides, not merely that something decided.**
An ``else`` contributes the negation of its ``if``, and a condition written as
a bare local name carries the expression that produced it -- otherwise a branch
guarded by ``if not allowed:`` reads as "some condition held", which cannot be
checked against anything.  PanDA moved a command's acceptance test into exactly
such a helper between two releases.
"""

from __future__ import annotations

import ast
from typing import Callable, Iterator, NamedTuple, Optional


def functions_with_owner(
    node: ast.AST, owner: Optional[str] = None
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Optional[str]]]:
    """Yield every function in *node* paired with the class enclosing it.

    Carried down the walk rather than read back from a ``parent`` link, so
    this works on a bare tree -- the element-type pass runs before the
    recognizer attaches parents.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from functions_with_owner(child, child.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, owner
            yield from functions_with_owner(child, owner)


def attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def ancestors(node: ast.AST) -> Iterator[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None:
        yield current
        current = getattr(current, "parent", None)


def enclosing_function(node: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for ancestor in ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ancestor
    return None


def single_definition(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> Optional[ast.expr]:
    """Return the expression assigned to *name*, when exactly one assigns it.

    A condition written as a bare local name says nothing on its own: ``if not
    allowed:`` names no predicate.  Substituting the single expression that
    produced it recovers which check decides -- typically a call such as
    ``self._check_command_allowed(...)``, whose own branches are a deeper
    expansion than this slice attempts.

    Returns ``None`` when several statements assign the name.  That is the
    fan-out case (a flag set from many places), where one expression would
    misrepresent the branch rather than explain it.

    The node rather than its text, because the other caller resolves what the
    expression *evaluates to* -- a local holding a declared mapping, which needs
    the tree.
    """
    found: list[ast.expr] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found.append(node.value)
                elif isinstance(target, ast.Tuple):
                    for index, element in enumerate(target.elts):
                        if isinstance(element, ast.Name) and element.id == name:
                            found.append(node.value)
                            del index
    return found[0] if len(found) == 1 else None


def assigned_expressions(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, list[ast.expr]]:
    """Every expression bound to each local name in *func*, ``+=`` included.

    The unrestricted sibling of :func:`single_definition`, for the readers that
    want *all* the values a name can hold rather than the one that explains a
    condition.  Both of them are the same hop: a message is written into a
    local and logged or persisted a few lines later, and following it is what
    the emit pass needs to find ``set task_status=`` at all and what the tag
    reading needs to find the reason a branch names for itself.

    An augmented assignment contributes its own right-hand side rather than the
    concatenation.  ``errMsg += ...`` under a condition may not run, so the
    pieces are what the reading establishes and the whole is not.
    """
    found: dict[str, list[ast.expr]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        for target in targets:
            found.setdefault(target.id, []).append(node.value)
    return found


def path_condition(node: ast.AST) -> list[str]:
    """Return the conjunction of tests dominating *node*, outermost first.

    Walks the ancestor chain rather than the call stack: an ``else`` branch
    contributes ``not <test>``, because "the condition did not hold" is as much
    a reason for the outcome as the condition holding.
    """
    conditions: list[str] = []
    func = enclosing_function(node)
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.If):
            try:
                test = ast.unparse(ancestor.test)
            except Exception:  # noqa: BLE001
                previous = ancestor
                continue
            if previous in ancestor.orelse:
                test = f"not ({test})"
            elif previous not in ancestor.body:
                previous = ancestor
                continue
            if func is not None:
                test = _substitute_bare_name(test, ancestor.test, func)
            conditions.append(test)
        previous = ancestor
    conditions.reverse()
    return conditions


#: What :func:`enclosing_guards` found around a site.  Each kind is a way the
#: code constrains reaching a line that ``path_condition`` is structurally
#: blind to, because only ``ast.If`` contributes to a path condition.
UNSEEN_TRY = "try"
UNSEEN_EXCEPT = "except"
UNSEEN_LOOP = "loop"
UNSEEN_WITH = "with"
UNSEEN_EARLY_EXIT = "early-exit"


class Unseen(NamedTuple):
    """One constraint on reaching a line that its path condition omits."""

    kind: str
    detail: str
    line: int


def _text(node: Optional[ast.AST]) -> str:
    if node is None:
        return "anything"
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _jumps(body: list[ast.stmt]) -> bool:
    """Whether entering *body* means leaving the block it is in.

    The last statement, not any statement: control reaching the end of a block
    has run it, so a trailing ``continue`` is unconditional for anyone who
    entered.  Read from the tree rather than from a flow graph, which is the
    whole reason this is cheap enough to be sound.
    """
    return bool(body) and isinstance(
        body[-1], (ast.Continue, ast.Break, ast.Return, ast.Raise)
    )


def _implied_by(statement: ast.stmt) -> Optional[str]:
    """The condition a preceding ``if ...: continue`` puts on what follows it."""
    if not isinstance(statement, ast.If):
        return None
    test = _text(statement.test)
    if test == "...":
        return None
    if _jumps(statement.body):
        return f"not ({test})"
    if statement.orelse and _jumps(statement.orelse):
        return test
    return None


def _early_exits(ancestor: ast.AST, previous: ast.AST) -> list[Unseen]:
    for field in ("body", "orelse", "finalbody"):
        block = getattr(ancestor, field, None)
        if not isinstance(block, list) or previous not in block:
            continue
        return [
            Unseen(UNSEEN_EARLY_EXIT, implied, statement.lineno)
            for statement in block[: block.index(previous)]
            if (implied := _implied_by(statement))
        ]
    return []


def _structure(ancestor: ast.AST, previous: ast.AST) -> list[Unseen]:
    if isinstance(ancestor, ast.Try) and previous in ancestor.body and ancestor.handlers:
        caught = ", ".join(_text(handler.type) for handler in ancestor.handlers)
        return [Unseen(UNSEEN_TRY, f"nothing above raised {caught}", ancestor.lineno)]
    if isinstance(ancestor, ast.ExceptHandler):
        return [
            Unseen(
                UNSEEN_EXCEPT,
                f"something above raised {_text(ancestor.type)}",
                ancestor.lineno,
            )
        ]
    if isinstance(ancestor, (ast.For, ast.AsyncFor)) and previous in ancestor.body:
        target, over = _text(ancestor.target), _text(ancestor.iter)
        return [Unseen(UNSEEN_LOOP, f"for {target} in {over}", ancestor.lineno)]
    if isinstance(ancestor, ast.While) and previous in ancestor.body:
        return [Unseen(UNSEEN_LOOP, f"while {_text(ancestor.test)}", ancestor.lineno)]
    if isinstance(ancestor, (ast.With, ast.AsyncWith)):
        held = ", ".join(_text(item.context_expr) for item in ancestor.items)
        return [Unseen(UNSEEN_WITH, f"with {held}", ancestor.lineno)]
    return []


def enclosing_guards(node: ast.AST) -> list[Unseen]:
    """What had to hold to reach *node* that its path condition cannot say.

    **A path condition is a necessary condition, not a sufficient one**, and
    this is the part that makes the difference nameable.  Only ``ast.If``
    contributes to :func:`path_condition`, so a site inside a ``try`` body, a
    handler, a loop or a ``with`` carries whatever tests are above it and
    nothing about the structure itself.  Measured on the installed corpus, of
    18018 reaching-definition sites 10619 are in a loop, 6515 in a ``try`` body
    and 433 in a handler -- and 3058, 2180 and 111 of those have an *empty*
    path condition, which reads as "unconditional" and is not.  On the map's
    own arms the exposure is higher still: 86% have at least one of these on
    the way to them.

    Four of the five kinds come from the ancestor chain.  The fifth is the
    early exit -- ``if X: continue`` above the site in the same block -- which
    is sound without any dataflow, because statements in a block run in order
    and a trailing jump is unconditional for whoever entered it.  14% of the
    map's arms have one on their walk, which is not thin enough to leave to a
    report.

    **A ``try`` body names its handlers, not the statements before it.**  Both
    are true constraints, but the handler set is bounded and usually logs,
    where the preceding statements have a median of twelve and a maximum of
    135 -- a list that long is noise, and it is the handler that says which
    external call could have diverted control here.

    Nothing above the enclosing function is read: module-scope statements run
    at import, so counting them as preceding would report a reason that never
    applied to this call.
    """
    found: list[Unseen] = []
    previous = node
    for ancestor in ancestors(node):
        found.extend(_early_exits(ancestor, previous))
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        found.extend(_structure(ancestor, previous))
        previous = ancestor
    found.sort(key=lambda entry: entry.line)
    return found


def own_test(node: ast.AST) -> Optional[str]:
    """The test of the innermost ``if``/``elif`` whose body contains *node*.

    Read from the tree rather than taken as the last entry of
    :func:`path_condition`, whose entries may carry the ``[name := expr]``
    annotation -- wrapping that in ``not (...)`` produces text nothing can read
    back.
    """
    previous = node
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.If) and previous in ancestor.body:
            try:
                return ast.unparse(ancestor.test)
            except Exception:  # noqa: BLE001
                return None
        previous = ancestor
    return None


def exclusive(one: list[str], other: list[str]) -> bool:
    """Whether two path conditions cannot both hold.

    Detected from the negations :func:`path_condition` already writes down: an
    ``elif`` branch carries ``not (<the test before it>)``, so two branches of
    one chain each hold a negation of something the other asserts.

    Deliberately conservative -- an annotated test will not match textually, so
    some exclusive pairs read as compatible.  That is the safe direction for
    both callers.  Deciding whether a later write overwrites an earlier one, a
    missed exclusion adds a condition that is true but redundant where a missed
    *overlap* would drop a required one; splitting a SQL statement built across
    branches, a missed exclusion leaves the statement folded as it was before,
    where a missed overlap would split a statement that is really one.
    """
    return any(f"not ({test})" in other for test in one) or any(
        f"not ({test})" in one for test in other
    )


def literal_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    resolve: Optional[Callable[[ast.expr, ast.AST], list[str]]] = None,
) -> list[tuple[str, list[str], int]]:
    """``(value, conditions, line)`` for every settled ``name = ...`` in *func*.

    The line is the assignment's, because that is where the value is decided --
    an anchor pointing at the use would send a reader to the place that merely
    passes it on.

    Reaching definitions for one local, which is what a value assigned to a
    variable before it is used needs -- the tag a broker interpolates into its
    rejection message, or the status a post-processor's helper returns.

    **Reassignment is the part dominating-guard analysis cannot see.**  The
    conditions on an assignment are necessary and, on their own, not sufficient:
    a later write reaching the same name replaces it.  So each assignment also
    carries the negation of every later one that is not mutually exclusive with
    it -- ``criteria = "-link_unusable"`` sits above the ``elif`` chain that
    replaces it, and ``status = "aborted"`` sits above two rechecks at the end
    of the function that can replace it whatever the chain decided.

    Exclusive siblings are skipped, or every branch of a chain would carry the
    negation of every other: ``-dest_blacklisted`` would come out requiring
    ``not (totalQueued >= limit)``, a condition with nothing to do with it.

    What counts as settled is the caller's to widen.  By default a string
    literal, and nothing here knows anything else; *resolve* lets a caller that
    does -- one holding the corpus's declared mappings -- settle
    ``newTaskStatus = commandStatusMap[commandStr]["doing"]`` to the six statuses
    it can hold.  One assignment may then contribute several values, all under
    the same guards, since the guards are what reached the assignment and the
    mapping is what chose among its entries.
    """

    def literal_only(expression: ast.expr, _func: ast.AST) -> list[str]:
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return [expression.value]
        return []

    settle = resolve or literal_only
    found: list[tuple[ast.Assign, list[str]]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        if enclosing_function(node) is not func:
            continue
        settled = settle(node.value, func)
        if settled:
            found.append((node, settled))
    found.sort(key=lambda pair: pair[0].lineno)
    conditions = {node: path_condition(node) for node, _ in found}
    values: list[tuple[str, list[str], int]] = []
    for assignment, settled in found:
        guards = list(conditions[assignment])
        for other, _ in found:
            if other.lineno <= assignment.lineno:
                continue
            if exclusive(conditions[assignment], conditions[other]):
                continue
            test = own_test(other)
            if test and f"not ({test})" not in guards:
                guards.append(f"not ({test})")
        # A copy per value: two branches sharing one condition list is a
        # mutation away from one of them rewriting the other's reason.
        values.extend((value, list(guards), assignment.lineno) for value in settled)
    return values


def _substitute_bare_name(
    rendered: str, test: ast.expr, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    """Annotate a test that is a bare local name with the expression behind it."""
    target = test.operand if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) else test
    if not isinstance(target, ast.Name):
        return rendered
    node = single_definition(func, target.id)
    if node is None:
        return rendered
    try:
        definition = ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return rendered
    if definition == target.id:
        return rendered
    return f"{rendered}  [{target.id} := {definition}]"


def enclosing_class(node: ast.AST) -> Optional[str]:
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.ClassDef):
            return ancestor.name
    return None
