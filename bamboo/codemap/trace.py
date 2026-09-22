"""Why one arm ran, walked from the source at the moment it is asked.

**What the map cannot answer and this can.**  The map is a reverse index: it
turns a value into the arms that can write it, an arm into the log that would
mention it, and a field into whoever reads it -- facts that need the whole
corpus and cannot be reached by walking forward from anywhere.  Everything
*reachable* from an anchor is a different job, and doing it here rather than
in the build is the point:

  A fact you can reach forward from the anchor belongs to the trace.  Only
  the facts you cannot reach belong to the map.

**Cadence, not analysis, is what went wrong the last time.**  The same walk was
built into ``build-map`` once and reverted: it cost half the build again, it
made one extra name visible in production, and four of its nine rejection
rules only appeared once it was running.  The cause was storage, not the
reading -- a stored branch has to fold its reasons into one condition string,
and that is where the loop rule, the length cap and the recursion rule all came
from.  At use time an answer may be a **set**, so a flag set in four places is
four steps because four of them are the answer, and none of those rules is
needed.

**Three components, because one arm is rarely explained by one of them.**  A
guard (71% of the map's arms have an ``if`` chain), where the value came from
(44% have a right-hand side that is not a literal), and what the enclosing
function was given.  Guards and values are walked together: a guard's names
are followed exactly like a value's, which is how ``taskSpec.status =
"tobroken"`` -- a constant with no provenance of its own -- reaches the DDM
call whose failure set the flag above it.

**Necessary, not sufficient.**  Only ``ast.If`` contributes to a path
condition, so :func:`pathcond.enclosing_guards` runs beside it and every step
carries what it could not say.  86% of the map's arms have a ``try``, a loop,
a handler or an early exit on the way to them; a step with empty ``guards`` is
one the walk found no ``if`` for, not one that is unconditional.

**Demand-driven is not a closure.**  Measured over the 930 arms of the 481
junctions the installed map can locate: a median of one file parsed and a
maximum of four, a median of one function visited and a maximum of four, and
a median of four steps with 24 at the ninetieth percentile.  478 of the 481
walks finish on their own and three reach the step bound, which they say.
The downward closure of the same corpus reaches 1566 functions at one hop and
90% of the tree at two, so the difference is the whole point of asking only
for what a condition names.

Nothing here is written into the map.
"""

from __future__ import annotations

import ast
import builtins
import logging
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from bamboo.codemap.gitsource import blob_sha as git_blob_sha
from bamboo.codemap.models import (
    ARRIVES_BY_CALL,
    ARRIVES_BY_DISPATCH,
    STOP_PARAMETER,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
    TraceStep,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.reading import containing_function

logger = logging.getLogger(__name__)

_BUILTINS = frozenset(dir(builtins))


class Budget(NamedTuple):
    """Bounds on the walk, sized well above what the corpus asks for.

    Not a tuning knob: the measurement says the walk stops on its own, so
    these exist to keep a pathological function from turning an investigation
    into a scan.  When one bites it is said out loud -- a silently truncated
    answer is the failure this whole design is built to avoid.

    ``depth`` is set where the walk converges rather than where it is
    comfortable: at six, two of the 481 walks were still cut short; at
    eight, none are, and the terminals the whole corpus reaches are the
    same as six's but for two steps.  ``steps`` still bites three times.
    """

    files: int = 12
    functions: int = 40
    steps: int = 120
    depth: int = 8


DEFAULT_BUDGET = Budget()

#: What a step's terminal is, when the classifier is not supplied.  Injected
#: rather than imported so that the walk stays about Python and the naming of
#: interfaces stays with the map that has them.
Classifier = Callable[[ast.expr, Optional[Callable[[str], Optional[ast.expr]]]], tuple[str, str]]


def _nothing(_expression: ast.expr, _resolve=None) -> tuple[str, str]:
    return "", ""


#: Where a binding the walk reports actually lives, when that is not the
#: function the arm is in.  Both are notes on ``TraceStep.unseen`` rather than
#: new kinds: the row is still a binding, and what changes is whether the
#: reader may take it as a step on the path to the arm.
MODULE_SCOPE = "module-scope"
NESTED_SCOPE = "nested-def"


class _Frame(NamedTuple):
    file: str
    owner: str
    func: ast.AST
    handovers: tuple[Handover, ...]
    imports: frozenset[str] = frozenset()


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _add(out: list[str], name: str) -> None:
    if name not in out:
        out.append(name)


def _reads(expression: ast.AST) -> list[str]:
    """The names an expression depends on, in the two forms the walk resolves.

    A bare local (which reaching definitions answer) and ``self.<field>``
    (which only the handover the map recorded can answer).  Builtins are
    dropped: resolving ``len`` would produce an unbound step per call and say
    nothing.
    """
    # ``self.helper(...)`` reads no attribute: the value is what the call
    # returns, and the method's body is a hop downwards that this walk does
    # not take.  Counting it as a field put 30 steps on names like
    # ``self.getClobObj`` and said the caller chose them.
    called = {
        node.func
        for node in ast.walk(expression)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    out: list[str] = []
    for node in ast.walk(expression):
        if isinstance(node, ast.Attribute):
            base: ast.AST = node
            nearest = ""
            depth = 0
            while isinstance(base, ast.Attribute):
                nearest = base.attr
                base = base.value
                depth += 1
            if depth == 1 and node in called:
                continue
            if isinstance(base, ast.Name) and base.id == "self":
                _add(out, f"self.{nearest}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id != "self" and node.id not in _BUILTINS:
                _add(out, node.id)
    return out


def _guard_reads(node: ast.AST) -> list[str]:
    """The names the ``if`` tests dominating *node* depend on.

    Walked as well as the value, which is what makes a constant right-hand
    side explainable at all: the reason ``status = "tobroken"`` ran is the
    flag above it, not the string.
    """
    out: list[str] = []
    previous = node
    for ancestor in pathcond.ancestors(node):
        if isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            break
        if isinstance(ancestor, ast.If) and (
            previous in ancestor.body or previous in ancestor.orelse
        ):
            for name in _reads(ancestor.test):
                _add(out, name)
        previous = ancestor
    return out


def _unseen(node: ast.AST) -> list[str]:
    return [
        f"{entry.kind} at {entry.line}: {entry.detail}"
        for entry in pathcond.enclosing_guards(node)
    ]


def _statement_at(func: ast.AST, line: int) -> Optional[ast.stmt]:
    """The statement the map's line addresses, innermost first."""
    exact = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.stmt) and node.lineno == line
    ]
    if exact:
        return min(exact, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)
    holding = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.stmt)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not holding:
        return None
    return min(holding, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)


def _value_of(statement: ast.stmt) -> Optional[ast.expr]:
    if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return statement.value
    if isinstance(statement, ast.Expr):
        return statement.value
    if isinstance(statement, ast.Return):
        return statement.value
    return None


def _loop_bindings(func: ast.AST, name: str) -> list[ast.For | ast.AsyncFor]:
    """Loops whose target binds *name*.

    Reaching definitions do not cover a loop target, and 7.7% of the names
    conditions read are one.  The chain that matters most runs straight
    through one: a command is read out of a map that a ``for`` took out of the
    list the knight handed over.
    """
    found = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        targets = [node.target]
        if isinstance(node.target, (ast.Tuple, ast.List)):
            targets = list(node.target.elts)
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            found.append(node)
    return found


def _tuple_bindings(func: ast.AST, name: str) -> list[ast.Assign]:
    """Assignments that bind *name* as one element of an unpacking.

    ``pathcond.assigned_expressions`` deliberately reads only ``ast.Name``
    targets, and it is shared with the build, so the form is picked up here
    instead of widening it -- the map's output must not move for a change to
    the walk.  The form matters: ``tmpStat, taskSpec = getTaskWithID_JEDI(...)``
    is how a spec arrives, and without it a guard reading ``taskSpec`` looks
    like a name nothing binds.
    """
    found: list[ast.Assign] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, (ast.Tuple, ast.List)):
                continue
            if any(isinstance(e, ast.Name) and e.id == name for e in target.elts):
                found.append(node)
    return found


def _imported_names(tree: ast.Module) -> frozenset[str]:
    """Names that stand for modules, classes and functions rather than values.

    Dropped from the walk rather than reported as unresolved: ``JediTaskSpec``
    and ``Interaction`` appear in half the conditions in this corpus and the
    walk has nothing to say about either, so keeping them produces one
    unexplained step per mention and hides the ones that matter.

    A ``def`` or ``class`` at module scope is the same kind of name as an
    import -- ``_compFunc`` passed as a sort key is a function, not a value --
    so it is dropped here too.  Module-level *assignments* are not: those hold
    values, and :func:`_module_bindings` reads them.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
    return frozenset(found)


def _with_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The context expressions a ``with ... as <name>`` binds.

    Reaching definitions do not cover the form, and it is how every
    ``TaskBuffer`` method reaches the database -- ``with self.proxyPool.get()
    as proxy`` -- so without it the walk said a function does not bind a name
    it plainly binds.  Read here rather than in ``pathcond``: that module is
    shared with the build, and widening it would move the map.
    """
    found: list[ast.expr] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            held = item.optional_vars
            targets = [held]
            if isinstance(held, (ast.Tuple, ast.List)):
                targets = list(held.elts)
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                found.append(item.context_expr)
    return found


def _walrus_bindings(func: ast.AST, name: str) -> list[ast.expr]:
    """The values an assignment expression ``(<name> := ...)`` binds."""
    return [
        node.value
        for node in ast.walk(func)
        if isinstance(node, ast.NamedExpr)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    ]


def _module_bindings(tree: ast.Module, name: str) -> list[ast.expr]:
    """Module-scope assignments to *name*, nothing nested.

    The last place to look before saying a name is not bound here, and the
    trace's own rule says to look: a fact reachable forward from the anchor
    belongs to the trace, and a constant in the same file is as reachable as
    one a line above the arm.  ``skipBrokerageProTypes = ["prod_test"]``
    is the shape -- a value the reader wants, previously reported as a name
    nothing binds.

    Only the module's own body, so a name assigned inside some unrelated
    function of the same file is not offered as this one's.
    """
    found: list[ast.expr] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        for target in targets:
            elements = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            if any(isinstance(e, ast.Name) and e.id == name for e in elements):
                found.append(node.value)
    return found


def _caught_as(func: ast.AST, name: str) -> list[ast.ExceptHandler]:
    """Handlers that bind *name* as the exception they caught."""
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.ExceptHandler) and node.name == name
    ]


def _nested_scope(func: ast.AST, site: ast.AST) -> Optional[ast.AST]:
    """The inner ``def`` or ``class`` between *site* and *func*, if any.

    ``pathcond.assigned_expressions`` walks the whole subtree, so a name
    assigned inside a nested function comes back as though *func* bound it.
    Its ``self.<field>`` sibling refuses to enter nested functions for the
    reason this note carries -- a closure runs when it is called, not where it
    is written -- but that one is read only at use time while this one is
    shared with the build, where the wider reading is load-bearing.  So the
    row stays and says where it came from instead.

    Keeping it matters: a daemon whose work lives in inner functions has its
    only answer there, and dropping the row would turn an incomplete
    explanation into a confident wrong one.
    """
    held = getattr(site, "parent", None)
    while held is not None and held is not func:
        if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            return held
        held = getattr(held, "parent", None)
    return None


def _is_parameter(func: ast.AST, name: str) -> bool:
    args = getattr(func, "args", None)
    if args is None:
        return False
    every = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        every.append(args.vararg)
    if args.kwarg:
        every.append(args.kwarg)
    return any(argument.arg == name for argument in every)


def _raised_in(node: ast.AST, classify: Classifier, resolve) -> tuple[str, str]:
    """The terminal of what the ``try`` body above this handler called.

    A handler ran because something above it raised, so the interface that
    could raise is where the value came from.  This is the only route from an
    arm whose own right-hand side is ``True`` to "DDM said so", and it is
    bounded -- the paired body's calls, not every statement before the site.
    """
    for ancestor in pathcond.ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
        if not isinstance(ancestor, ast.ExceptHandler):
            continue
        block = getattr(ancestor, "parent", None)
        if not isinstance(block, ast.Try):
            break
        for statement in block.body:
            for sub in ast.walk(statement):
                if isinstance(sub, ast.Call):
                    terminal, detail = classify(sub, resolve)
                    if terminal:
                        return terminal, detail
        break
    return "", ""


class _Walk:
    def __init__(
        self,
        roots: dict[str, Path],
        classify: Classifier,
        budget: Budget,
    ) -> None:
        self.roots = roots
        self.classify = classify
        self.budget = budget
        self.parsed: dict[str, tuple[str, ast.Module]] = {}
        self.frames: dict[tuple[str, str], Optional[_Frame]] = {}
        self.steps: list[TraceStep] = []
        self.seen: set[tuple[str, str]] = set()
        self._scopes: dict[tuple[str, str], str] = {}
        self.note = ""
        self.queue: deque[tuple[_Frame, str, int]] = deque()

    # -- source ------------------------------------------------------------

    def module(self, file: str) -> Optional[tuple[str, ast.Module]]:
        if file in self.parsed:
            return self.parsed[file]
        if len(self.parsed) >= self.budget.files:
            self._exhausted("files")
            return None
        package, _, relative = file.partition("/")
        root = self.roots.get(package)
        if root is None or not relative:
            # Outside the corpus the map covers, which is the walk's stop rule
            # rather than a failure: a name that comes from somewhere else is
            # exactly what a boundary is for.
            return None
        try:
            source = (root / relative).read_text(errors="replace")
            tree = ast.parse(source, filename=str(root / relative))
        except (OSError, SyntaxError, ValueError) as exc:
            logger.debug("trace: %s unreadable (%s)", file, exc)
            return None
        pathcond.attach_parents(tree)
        self.parsed[file] = (source, tree)
        return self.parsed[file]

    def frame(
        self,
        file: str,
        owner: str,
        *,
        line: Optional[int] = None,
        handovers: Sequence[Handover] = (),
    ) -> Optional[_Frame]:
        key = (file, owner)
        if key in self.frames:
            return self.frames[key]
        if len(self.frames) >= self.budget.functions:
            self._exhausted("functions")
            return None
        self.frames[key] = None
        parsed = self.module(file)
        if parsed is None:
            return None
        _source, tree = parsed
        wanted = owner.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        if line is not None:
            func = containing_function(tree, line, owner)
        else:
            func = next(
                (
                    node
                    for node, _cls in pathcond.functions_with_owner(tree)
                    if node.name == wanted
                ),
                None,
            )
        if func is None:
            return None
        self.frames[key] = _Frame(
            file, owner, func, tuple(handovers), _imported_names(tree)
        )
        return self.frames[key]

    # -- steps -------------------------------------------------------------

    def _exhausted(self, what: str) -> None:
        if not self.note:
            self.note = (
                f"the walk's {what} budget ran out; what follows is part of the answer"
            )

    def record(self, step: TraceStep) -> bool:
        if len(self.steps) >= self.budget.steps:
            self._exhausted("steps")
            return False
        self.steps.append(step)
        return True

    def resolver(self, frame: _Frame):
        return lambda name: pathcond.single_definition(frame.func, name)

    def scope_key(self, frame: _Frame) -> str:
        """What a name asked in *frame* is unique within.

        A local belongs to its function.  ``self.<field>`` belongs to the
        object, so the class is its scope and not the method that happened to
        read it -- and once stage three answers a field from every method of
        the class, asking again from a sibling frame would replay the same
        assignments under a second heading.  Keyed by the class where there is
        one, so the two kinds of name cannot collide.
        """
        held = self._scopes.get((frame.file, frame.owner))
        if held is not None:
            return held
        parsed = self.module(frame.file)
        cls = None
        if parsed is not None:
            cls = pathcond.class_of(parsed[1], frame.func)
        key = f"{frame.file}::{cls.name}" if cls is not None else f"{frame.file}::{frame.owner}"
        self._scopes[(frame.file, frame.owner)] = key
        return key

    def want(self, frame: _Frame, names: Sequence[str], depth: int) -> None:
        if depth > self.budget.depth:
            self._exhausted("depth")
            return
        for name in names:
            if name in frame.imports:
                continue
            if name.startswith("self."):
                key = (self.scope_key(frame), name)
            else:
                key = (f"{frame.file}::{frame.owner}", name)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.queue.append((frame, name, depth))

    # -- the walk ----------------------------------------------------------

    def arm(self, frame: _Frame, line: int) -> None:
        statement = _statement_at(frame.func, line)
        if statement is None:
            return
        value = _value_of(statement)
        reads = _reads(value) if value is not None else []
        for name in _guard_reads(statement):
            _add(reads, name)
        terminal, detail = (
            self.classify(value, self.resolver(frame)) if value is not None else ("", "")
        )
        step = TraceStep(
            kind=TRACE_WRITE,
            owner=frame.owner,
            file=frame.file,
            line=line,
            value=_text(value) if value is not None else _text(statement),
            guards=pathcond.path_condition(statement),
            unseen=_unseen(statement),
            reads=reads,
            terminal=terminal,
            detail=detail,
        )
        if not self.record(step):
            return
        if not terminal:
            self.want(frame, reads, 1)

    def explain(self, frame: _Frame, name: str, depth: int) -> None:
        if name.startswith("self."):
            self._field(frame, name, depth)
            return
        bindings = list(pathcond.assigned_expressions(frame.func).get(name, []))
        bindings.extend(
            node.value for node in _tuple_bindings(frame.func, name)
        )
        bindings.extend(_with_bindings(frame.func, name))
        bindings.extend(_walrus_bindings(frame.func, name))
        loops = _loop_bindings(frame.func, name)
        caught = _caught_as(frame.func, name)
        if not bindings and not loops and not caught:
            # The file's own body is the last place to look.  Asked only here,
            # after the function has had its say, so a local always wins over a
            # module constant of the same name.
            parsed = self.module(frame.file)
            outer = _module_bindings(parsed[1], name) if parsed is not None else []
            if outer:
                for expression in outer:
                    self._module_binding(frame, name, expression, depth)
                return
            self._unbound(frame, name, depth)
            return
        for expression in bindings:
            self._binding(frame, name, expression, depth)
        for loop in loops:
            self._loop(frame, name, loop, depth)
        for handler in caught:
            self._caught(frame, name, handler, depth)

    def _binding(self, frame: _Frame, name: str, expression: ast.expr, depth: int) -> None:
        statement = expression
        while statement is not None and not isinstance(statement, ast.stmt):
            statement = getattr(statement, "parent", None)
        site: ast.AST = statement if statement is not None else expression
        resolve = self.resolver(frame)
        terminal, detail = self.classify(expression, resolve)
        unseen = _unseen(site)
        inner = _nested_scope(frame.func, site)
        if inner is not None:
            unseen.append(
                f"{NESTED_SCOPE} at {inner.lineno}: inside "
                f"{getattr(inner, 'name', 'a lambda')}, which runs when it is "
                f"called and not on the way here"
            )
        if not terminal and any(
            entry.startswith(pathcond.UNSEEN_EXCEPT) for entry in unseen
        ):
            terminal, detail = _raised_in(site, self.classify, resolve)
        reads = _reads(expression)
        for read in _guard_reads(site):
            _add(reads, read)
        step = TraceStep(
            kind=TRACE_BINDING,
            name=name,
            owner=frame.owner,
            file=frame.file,
            line=getattr(site, "lineno", 0),
            value=_text(expression),
            guards=pathcond.path_condition(site),
            unseen=unseen,
            reads=reads,
            terminal=terminal,
            detail=detail,
            depth=depth,
        )
        if not self.record(step):
            return
        if not terminal:
            self.want(frame, reads, depth + 1)

    def _module_binding(
        self, frame: _Frame, name: str, expression: ast.expr, depth: int
    ) -> None:
        """A value the file binds once, when it is imported.

        Recorded as a binding because that is what it is, and marked because
        the reader must not read it as a step on the path: it ran at import,
        not on the way to this arm, so no guard of this function dominated it.

        The walk stops here rather than following what the expression reads.
        Those names live in the module's scope, and resolving them inside this
        function's frame would answer with whatever local happened to share the
        name -- the mistake the scope note below exists to prevent.
        """
        terminal, detail = self.classify(expression, self.resolver(frame))
        line = getattr(expression, "lineno", 0)
        self.record(
            TraceStep(
                kind=TRACE_BINDING,
                name=name,
                owner=f"{frame.file}::{MODULE_SCOPE}",
                file=frame.file,
                line=line,
                value=_text(expression),
                unseen=[
                    f"{MODULE_SCOPE} at {line}: bound when the file is imported, "
                    f"not on the way here"
                ],
                terminal=terminal,
                detail=detail,
                depth=depth,
            )
        )

    def _loop(self, frame: _Frame, name: str, loop: ast.AST, depth: int) -> None:
        over = _text(loop.iter)
        reads = _reads(loop.iter)
        step = TraceStep(
            kind=TRACE_LOOP,
            name=name,
            owner=frame.owner,
            file=frame.file,
            line=loop.lineno,
            value=f"an element of {over}",
            guards=pathcond.path_condition(loop),
            unseen=_unseen(loop),
            reads=reads,
            depth=depth,
        )
        if self.record(step):
            self.want(frame, reads, depth + 1)

    def _caught(
        self, frame: _Frame, name: str, handler: ast.ExceptHandler, depth: int
    ) -> None:
        """An ``except ... as e``: the value is whatever the body raised."""
        resolve = self.resolver(frame)
        terminal, detail = _raised_in(
            handler.body[0] if handler.body else handler, self.classify, resolve
        )
        self.record(
            TraceStep(
                kind=TRACE_BINDING,
                name=name,
                owner=frame.owner,
                file=frame.file,
                line=handler.lineno,
                value=f"raised under {_text(handler.type)}",
                unseen=_unseen(handler),
                terminal=terminal,
                detail=detail,
                depth=depth,
            )
        )

    def _sibling(self, frame: _Frame, method: ast.AST) -> Optional[_Frame]:
        """A frame for another method of the same class.

        Its own frame, not the caller's, because the guards the walk reports
        come from the statement's ancestors: recording ``__init__``'s
        assignment inside the frame of the method that read it would print a
        condition that never dominated the write.  Handovers travel with it --
        they are a fact about how the object was constructed, which is as true
        in one of its methods as in another.
        """
        owner = f"{frame.owner.rsplit('::', 1)[0]}::{method.name}"
        key = (frame.file, owner)
        held = self.frames.get(key)
        if held is not None:
            return held
        if key in self.frames:
            return None
        if len(self.frames) >= self.budget.functions:
            self._exhausted("functions")
            self.frames[key] = None
            return None
        self.frames[key] = _Frame(
            frame.file, owner, method, frame.handovers, frame.imports
        )
        return self.frames[key]

    def _assigned_field(self, frame: _Frame, name: str, field: str, depth: int) -> bool:
        """Stages two and three: ``self.<field> = ...`` in scope.

        Ordered, and the order is the design.  ``classify()`` has already run
        above this -- it has to, because ``self.cur`` is assigned in an
        ``__init__`` like any other field and is still the database, and this
        reading would otherwise answer a question about a row with a line
        about a connection.

        Two before three because the nearer assignment is the one whose guards
        reached the arm, and both rather than either because the answer is a
        set: ``self.jobs`` is written in three methods of
        ``setupper_atlas_plugin`` and which one ran last is a fact about the
        run, not about the text.  Nothing is folded.
        """
        answered = False
        for expression in pathcond.attribute_expressions(frame.func, field):
            self._binding(frame, name, expression, depth)
            answered = True
        parsed = self.module(frame.file)
        if parsed is None:
            return answered
        _source, tree = parsed
        cls = pathcond.class_of(tree, frame.func)
        if cls is None:
            return answered
        for holder in [cls, *pathcond.bases_in(tree, cls)]:
            for method in pathcond.methods_of(holder):
                if method is frame.func:
                    continue
                other = None
                for expression in pathcond.attribute_expressions(method, field):
                    other = other or self._sibling(frame, method)
                    if other is None:
                        break
                    self._binding(other, name, expression, depth)
                    answered = True
        return answered

    def _field(self, frame: _Frame, name: str, depth: int) -> None:
        field = name.split(".", 1)[1]
        try:
            expression = ast.parse(name, mode="eval").body
        except SyntaxError:
            return
        terminal, detail = self.classify(expression, self.resolver(frame))
        if terminal:
            self.record(
                TraceStep(
                    kind=TRACE_UNBOUND,
                    name=name,
                    owner=frame.owner,
                    file=frame.file,
                    value=name,
                    terminal=terminal,
                    detail=detail,
                    depth=depth,
                )
            )
            return
        answered = self._assigned_field(frame, name, field, depth)
        # Only a dispatch: its keys are the worker's own attributes, which is
        # what ``self.<field>`` is.  A call's keys are parameter names, and
        # letting those answer here would hand ``self.x`` whatever a caller
        # passed for a parameter that happens to be spelled ``x``.
        #
        # Every one of them, not the first.  ``AdderGen`` is built in three
        # places and the pilot's and the daemon's differ in what they pass;
        # stopping at whichever the map listed first would answer "who chose
        # this" with one of the callers and no sign that there were others.
        for handover in frame.handovers:
            if handover.reached_by == ARRIVES_BY_DISPATCH and field in handover.fields:
                self._handover(frame, name, handover, depth)
                answered = True
        if answered:
            # The assignments are the answer.  Saying "no handover names it"
            # underneath them would report an absence as though it were the
            # reason, which is the shape this whole change is about.
            return
        self.record(
            TraceStep(
                kind=TRACE_UNBOUND,
                name=name,
                owner=frame.owner,
                file=frame.file,
                value=name,
                terminal=STOP_PARAMETER,
                detail="an attribute of the worker, and the map names no handover for it",
                depth=depth,
            )
        )

    def _handover(
        self, frame: _Frame, name: str, handover: Handover, depth: int
    ) -> None:
        key = name.split(".", 1)[1] if name.startswith("self.") else name
        supplied = handover.fields[key]
        owner = f"{handover.entry}::{handover.via}"
        other = self.frame(handover.entry, owner)
        try:
            expression = ast.parse(supplied, mode="eval").body
        except SyntaxError:
            expression = None
        # ``reads`` is a promise that each name will get a step of its own, so
        # it is only made when there is a frame to make it in.  Six names in
        # the corpus were promised by a crossing whose far side would not
        # parse, and a promise nothing keeps reads as a hole in the walk.
        reads = _reads(expression) if expression is not None and other else []
        step = TraceStep(
            kind=TRACE_HANDOVER,
            name=name,
            owner=owner,
            file=handover.entry,
            line=getattr(other.func, "lineno", 0) if other else 0,
            value=supplied,
            reads=reads,
            detail=(
                f"handed over by {handover.via}, reached by {handover.reached_by}"
                if other
                else f"handed over by {handover.via}, which could not be read here"
            ),
            depth=depth,
        )
        if not self.record(step):
            return
        if other is not None:
            self.want(other, reads, depth + 1)

    def _unbound(self, frame: _Frame, name: str, depth: int) -> None:
        # A parameter is settled at the call site, and for the entries the map
        # reached by a plain call it recorded what was passed -- so the same
        # crossing that answers a worker's attribute answers this too.  Kept
        # to parameters of *this* function, since a dispatch's keys name the
        # worker's attributes and could collide with an unrelated local.
        parameter = _is_parameter(frame.func, name)
        if parameter:
            for handover in frame.handovers:
                if handover.reached_by == ARRIVES_BY_CALL and name in handover.fields:
                    self._handover(frame, name, handover, depth)
                    return
        self.record(
            TraceStep(
                kind=TRACE_UNBOUND,
                name=name,
                owner=frame.owner,
                file=frame.file,
                value=name,
                terminal=STOP_PARAMETER if parameter else "",
                detail=(
                    "a parameter, so its value is chosen at the call site"
                    if parameter
                    else "not bound in this function -- module scope, an import, or a form the walk does not read"
                ),
                depth=depth,
            )
        )


def walk(
    roots: dict[str, Path],
    *,
    file: str,
    owner: str,
    lines: Sequence[int],
    expected_sha: str = "",
    handovers: Sequence[Handover] = (),
    classify: Optional[Classifier] = None,
    budget: Budget = DEFAULT_BUDGET,
) -> tuple[list[TraceStep], str]:
    """Explain the arms at *lines*, and say why the answer is short when it is.

    *lines* are the arms of one function, walked together so that a flag they
    share is explained once.  Returns the steps and a note -- empty when the
    walk finished on its own, which the measurement says is the usual case.

    **A tree that is not the map's is refused rather than warned about.**  A
    trace is shaped like an answer, and reading line 4971 of the wrong release
    produces plausible code that is not the code -- which already happened
    once, to a report that only printed coordinates.  Computing from the wrong
    tree is worse than printing from it.
    """
    state = _Walk(roots, classify or _nothing, budget)
    parsed = state.module(file)
    if parsed is None:
        return [], "the map's file could not be read from this tree"
    source, _tree = parsed
    if expected_sha and git_blob_sha(source) != expected_sha:
        return [], "the walk was not run: this tree is not the one that was mapped"
    if not lines:
        return [], "the map records no line for these arms"
    frame = state.frame(file, owner, line=lines[0], handovers=handovers)
    if frame is None:
        return [], "the map's function could not be located in this tree"
    for line in lines:
        state.arm(frame, line)
    while state.queue:
        held, name, depth = state.queue.popleft()
        state.explain(held, name, depth)
    return state.steps, state.note
