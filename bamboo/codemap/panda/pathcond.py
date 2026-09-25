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
from typing import Callable, Iterator, NamedTuple, Optional, Sequence


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


def targets_of(node: ast.AST) -> list[ast.expr]:
    """What an ``=`` statement writes to, or ``[]`` if *node* is not one.

    ``x: dict[str, Any] = {...}`` is the same write as ``x = {...}`` and the
    corpus spells it that way 914 times in a function body, against five in the
    release the map was last built from.  Reading only :class:`ast.Assign` cost
    the SQL slice eleven facts -- three tables their reader, three values their
    selector, a criterion, two junctions and the whole of
    ``harvester_workers.pilotStatus`` -- and left one statement's table
    unresolved.  None of that was a change in PanDA: both versions of
    ``getDispatchDatasetsPerUser`` are identical but for the annotation.

    Here rather than in one recognizer because the same reading was found and
    fixed three separate times, each time at the single site that had just been
    caught.  A reader asking "is this an assignment" should get one answer.

    An ``AnnAssign`` with no value -- ``found: list[str]`` -- states a type and
    writes nothing, so it has no targets here: reading one as a write would have
    the name hold whatever the *next* statement assigns.

    ``AugAssign`` is deliberately absent.  Every caller that wants ``+=`` treats
    it as a different operator, and folding it in here would make ``sql +=
    " AND x=1"`` look like the statement rather than a fragment of it.
    """
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [node.target]
    return []


def written_value(node: ast.AST) -> Optional[ast.expr]:
    """What an assignment writes, across all three spellings of one.

    Unlike :func:`targets_of` this does fold in ``AugAssign``, because its
    callers are reassembling a statement from the ``=`` and ``+=`` fragments
    that build it and need the right-hand side of both.
    """
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return node.value
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

    Only names, still: an attribute target belongs to :func:`attribute_expressions`
    and putting one here would attach a spec field's history to a local's.  But
    a name is a name however it is declared, and ``error_message: str | None =
    f"method {method_name} is forbidden"`` is the write the emit pass has to
    follow to give the 403 any text at all.
    """
    found: dict[str, list[ast.expr]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names = [node.target]
        else:
            names = [t for t in targets_of(node) if isinstance(t, ast.Name)]
        value = written_value(node)
        if value is None:
            continue
        for name in names:
            found.setdefault(name.id, []).append(value)
    return found


def attribute_expressions(
    scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, field: str
) -> list[ast.expr]:
    """Every expression bound to ``self.<field>`` inside *scope*.

    The ``ast.Attribute`` sibling of :func:`assigned_expressions`, and a
    separate function rather than a widening of it.  That one is shared with
    the build, where its targets being bare names is load-bearing: the write
    slice reads it to find the value a bind was filled from, and an attribute
    target there would attach a spec field's history to a local's.  Widening
    it would move the map; this is read at use time only.

    Scoped to one field rather than returning a mapping, because the caller
    arrives with the field in hand and the classes this is asked about carry
    hundreds of attributes between them -- ``JobSpec`` alone declares 126.

    Nested functions are not entered.  A closure assigning ``self.x`` runs
    when it is called and not where it is written, so its guards are not the
    ones that reached the arm; the walk would report a condition that never
    dominated the write.
    """
    found: list[ast.expr] = []
    for node in _own_body(scope):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for bound in _bound_names(target):
                if bound == field and node.value is not None:
                    found.append(node.value)
    return found


def _bound_names(target: ast.AST) -> list[str]:
    """The ``self.<field>`` names an assignment target binds."""
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return [target.attr]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _bound_names(element)]
    return []


def _own_body(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node under *scope* that belongs to it, nested functions excluded."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _own_body(child)


def methods_of(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """The methods *cls* declares, in source order."""
    return [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def class_of(tree: ast.AST, func: ast.AST) -> Optional[ast.ClassDef]:
    """The class *func* is a method of, without needing parent links.

    ``attribution`` answers the same question by walking ``parent`` back, which
    it can because the recognizer attaches them.  The walk parses files of its
    own at use time and has no such guarantee, so this descends instead -- the
    same rule, read from the other end.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            method is func for method in methods_of(node)
        ):
            return node
    return None


def bases_in(tree: ast.AST, cls: ast.ClassDef) -> list[ast.ClassDef]:
    """*cls*'s bases that this module declares, nearest first.

    The same order ``attribution._class_and_ancestors`` produces, cut where
    this reading's evidence stops: it has one parsed module and no corpus
    index, so a base declared elsewhere is not followed.  Measured before the
    cut was made -- of 292 classes in the corpus, 8 have all their bases in
    their own module and 141 have at least one elsewhere, and the 83 steps
    this whole reading exists for need none of them.  The limit is real and
    it is not in the way.
    """
    declared = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    order: list[ast.ClassDef] = []
    seen = {cls.name}
    queue = [_base_name(base) for base in cls.bases]
    while queue:
        name = queue.pop(0)
        if not name or name in seen:
            continue
        seen.add(name)
        node = declared.get(name)
        if node is None:
            continue
        order.append(node)
        queue.extend(_base_name(base) for base in node.bases)
    return order


def _base_name(base: ast.expr) -> Optional[str]:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


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


# ---------------------------------------------------------------------------
# Whether what one site writes can be what another site reads
# ---------------------------------------------------------------------------
#
# Written for the walk, which offers a reader every binding of a name it could
# not value, and wrong there in one direction only: a site below the arm never
# ran before it.  The build asks the same question of a different pair -- can
# the fragments that built a statement reach the call that runs it -- and was
# wrong in the same direction, attributing one statement to every call site in
# the function.
#
# Here rather than in either caller because the build must not import the
# walk: one makes the stored map and one reads it, and the separation is what
# keeps a use-time convenience out of the stored facts.  One spelling, so the
# two cannot drift into disagreeing about which writes a read can see.


def site_of(expression: ast.AST) -> ast.AST:
    """The statement *expression* is part of, or the expression itself.

    :func:`assigned_expressions` hands back the right-hand side, and everything
    positional -- the line, the guards above it, whether a loop holds it -- is
    a property of the statement it sits in.
    """
    node: Optional[ast.AST] = expression
    while node is not None and not isinstance(node, ast.stmt):
        node = getattr(node, "parent", None)
    return node if node is not None else expression


def loops_over(node: ast.AST) -> set[int]:
    """The header line of every loop whose *body* holds *node*."""
    found: set[int] = set()
    previous = node
    for ancestor in ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, (ast.For, ast.AsyncFor, ast.While)) and previous in ancestor.body:
            found.add(ancestor.lineno)
        previous = ancestor
    return found


def not_below(site: ast.AST, target: ast.AST) -> bool:
    """Whether *site* could have run before *target* on one pass.

    A write below the read cannot have run before it -- unless a loop holds
    them both, where the next iteration reaches it.  That second half is not a
    refinement: without it the rule is unsound, and a message assigned at the
    foot of a loop body and logged at its head is dropped although every
    iteration but the first reaches it.
    """
    return getattr(site, "lineno", 0) <= getattr(target, "lineno", 0) or bool(
        loops_over(site) & loops_over(target)
    )


def reaches_one(site: ast.AST, targets: Sequence[ast.AST]) -> bool:
    """Whether *site* is positioned to reach any of *targets*."""
    return any(not_below(site, target) for target in targets)


def reachable_when(node: ast.AST) -> list[str]:
    """The conditions on reaching *node*, including the ones an ``if`` cannot say.

    :func:`path_condition` sees only ``ast.If``, so an ``if X: continue`` above
    a site leaves the site reading as unconditional.  That blindness is not
    harmless here: a log call inside the skipped block and a write after it are
    on paths that cannot both run, and comparing their path conditions alone
    says they are compatible.  The early exit puts ``not (X)`` on the write and
    the block puts ``X`` on the call, which is exactly the shape
    :func:`exclusive` was built to detect.
    """
    return path_condition(node) + [
        unseen.detail
        for unseen in enclosing_guards(node)
        if unseen.kind == UNSEEN_EARLY_EXIT
    ]


def compatible(one: ast.AST, other: ast.AST) -> bool:
    """Whether both nodes can be on one run through the function.

    A path condition is a necessary condition, so this is conservative in the
    direction that is safe: it separates two sites only where one's condition
    contradicts the other's.
    """
    return not exclusive(reachable_when(one), reachable_when(other))


def can_reach(site: ast.AST, target: ast.AST) -> bool:
    """Whether what *site* writes can be what *target* reads."""
    return not_below(site, target) and compatible(site, target)


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
