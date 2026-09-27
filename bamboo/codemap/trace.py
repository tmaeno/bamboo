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
import re
from collections import deque
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

from bamboo.codemap.gitsource import blob_sha as git_blob_sha
from bamboo.codemap.models import (
    ARRIVES_BY_CALL,
    ARRIVES_BY_DISPATCH,
    SKELETON_ARM,
    SKELETON_BRANCH,
    SKELETON_PRINT,
    STOP_PARAMETER,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
    SkeletonLine,
    TraceStep,
)
from bamboo.codemap.panda import pathcond

# ``_logged_arguments`` is the build's own spelling of "what this corpus logs",
# reused rather than restated.  A second spelling here would be a second thing
# to keep current, and the one thing the two cadences must agree on is which
# calls are log calls: the map's question and the trace's prediction have to be
# about the same sentence or they cannot be compared at all.
from bamboo.codemap.panda.recognizers.emit import _logged_arguments, logging_arguments
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
    comfortable: when eight was chosen, six still cut two of 481 walks short
    and eight cut one, and the terminals the whole corpus reaches were the same
    as six's but for two steps.

    Re-measured since, over the derivation for every term in the vocabulary
    rather than that one sample -- 1567 readings, all of them walked.  A bite
    is counted per reading, so one awkward function accounts for many:
    ``steps`` bites 49 times and ``depth`` 15, over **four functions in all**
    -- ``insertFilesForDataset_JEDI`` (43), ``closer.py::run`` (12),
    ``JobGenerator.py::runImpl`` (5) and ``datasetManager.py::run`` (4).  The
    population is not the old one, so read the shape and not the difference:
    four functions out of 1567 readings is the design working.  The numbers are
    not a tuning knob, and "none are" was written when the corpus was smaller.
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


def _written_texts(statement: ast.stmt) -> list[str]:
    """How the arm spells what it writes, and what it writes it to.

    Both, because a log line reports either one.  ``taskSpec.status = newStatus``
    is followed by a line interpolating ``newStatus`` as often as by one
    interpolating ``taskSpec.status``, and the two are the same fact said from
    the two ends of the assignment.
    """
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
        targets = [statement.target]
    texts = [_text(target) for target in targets]
    value = _value_of(statement)
    if value is not None:
        texts.append(_text(value))
    return [text for text in texts if text and text != "..."]


# How much literal text a pattern needs before it is a question rather than a
# sieve.  Set from two measurements, neither of which found a clean answer.
#
# Put against the production lines already in the evidence file, the patterns
# a lower bar would admit match the *wrong* message at 13 characters and below
# -- ``IO intensity `` finds ``candidates passed max IO intensity check``, and
# ``registering`` finds ``registering <dataset> with location=...``.  From 14
# up, none of the admitted patterns matched a line it did not mean.
#
# The second measurement is the one to be honest about.  Asking how often a
# message's literal run sits inside *another* message in the corpus gives 89%
# at 0-4 characters, 65% at 5-9, 41% at 10-14, 37% at 15-19 -- and **11% at 40
# and above**.  It decays and never reaches zero, so no length makes a pattern
# safe and this guard only removes the worst of them.  A reader who takes a
# long pattern for a reliable one is the next version of this bug.
MIN_LITERAL = 14

# What has to *not* follow a value for the pattern to be about that value.
# ``newPrio=100`` matches ``newPrio=1000``: without a literal after the hole
# there is nothing to say the number ended, and a question that cannot miss is
# read by the eliminator as a question that was answered.
NOT_A_WORD = r"(?![\w.-])"


def _neighbouring_literal(message: ast.JoinedStr, hole: int) -> bool:
    """Whether the hole at *hole* has literal text to anchor on beside it.

    The requirement the map's shared sentence cannot make.  A pattern with no
    literal next to the value is ``.*finished.*``, which matches every line in
    the file that mentions the word -- and a question that cannot miss is read
    by the eliminator as a question that was answered.
    """
    for side in (hole - 1, hole + 1):
        if not 0 <= side < len(message.values):
            continue
        part = message.values[side]
        if isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value.strip():
            return True
    return False


def _literal_text(part: ast.expr) -> str:
    """*part*'s text if it is a literal piece of the message, else ``""``."""
    if isinstance(part, ast.Constant) and isinstance(part.value, str):
        return part.value
    return ""


def literal_length(message: ast.JoinedStr) -> int:
    """How many characters of literal text the message carries.

    What is left once the holes are taken out is the whole of what a pattern
    can anchor on, so it is the only honest measure of whether the pattern is a
    sentence.  Stripped, because a hole sitting between two spaces contributes
    two characters that match anything with a space on either side.
    """
    return sum(len(_literal_text(part).strip()) for part in message.values)


def _bound(message: ast.JoinedStr, hole: int) -> str:
    """A pattern for an unfilled hole, closed against the literal after it.

    ``.*`` is unbounded, and unbounded is how ``set\\ .*=None`` came to match
    ``set task_status=pending oldTask=False with (True, 1) ...``: the run
    crossed the field name, its value and two more fields to reach an ``=None``
    that belonged to nothing in this statement.  The literal after the hole
    says where the hole has to stop, and excluding its first character is what
    stops it there.

    Falls back to ``.*`` with nothing to close against -- a trailing hole, or
    one butted straight against another.
    """
    following = _literal_text(message.values[hole + 1]) if hole + 1 < len(message.values) else ""
    if following:
        return f"[^{re.escape(following[0])}]*"
    return ".*"


def _pattern(message: ast.JoinedStr, hole: Optional[int], observed: str) -> str:
    """The message as a regular expression, with *hole* filled by *observed*.

    *hole* is ``None`` for the skeleton's own line, where no value is put in
    and the pattern says only *this line was printed here*.  That is the whole
    difference between the two kinds this module now renders, and both go
    through here so that neither can drift into being the looser one.

    Three things keep the result a question that can come back empty.  Every
    other hole is closed against the literal that follows it rather than left
    as ``.*``; a filled hole with no literal after it gets :data:`NOT_A_WORD`,
    so the value has to end where the pattern says it ends; and a message with
    less than :data:`MIN_LITERAL` characters of literal is refused outright,
    because there is not enough of it left to be a sentence.

    Leading and trailing holes are dropped.  The search is unanchored, so a
    run at either end asks for nothing the rest does not already ask for, and
    printing it invites a reader to think it does.
    """
    if literal_length(message) < MIN_LITERAL:
        return ""
    pieces: list[tuple[str, bool]] = []
    for position, part in enumerate(message.values):
        if position == hole:
            pieces.append((re.escape(observed), False))
            after = (
                _literal_text(message.values[position + 1])
                if position + 1 < len(message.values)
                else ""
            )
            if not after:
                pieces.append((NOT_A_WORD, False))
            continue
        literal = _literal_text(part)
        if literal:
            pieces.append((re.escape(literal), False))
        else:
            pieces.append((_bound(message, position), True))
    while pieces and pieces[0][1]:
        pieces.pop(0)
    while pieces and pieces[-1][1]:
        pieces.pop()
    out: list[str] = []
    for text, is_hole in pieces:
        if is_hole and out and out[-1] == text:
            continue
        out.append(text)
    return "".join(out)


def _rendered(message: ast.JoinedStr, hole: int, observed: str) -> str:
    """The value pattern: the observed value put into the hole it fills."""
    return _pattern(message, hole, observed)


def line_pattern(message: ast.JoinedStr) -> str:
    """The line pattern: every hole left open, so it names the line only."""
    return _pattern(message, None, "")


def _enclosing_statement(node: ast.AST) -> Optional[ast.stmt]:
    """The statement *node* sits in, so a call is placed where the source puts it.

    A call's first argument carries the line number the walk has in hand, and
    for a message split over four lines that is not the line the reader sees
    the call on.  The statement is.
    """
    if isinstance(node, ast.stmt):
        return node
    for ancestor in pathcond.ancestors(node):
        if isinstance(ancestor, ast.stmt):
            return ancestor
    return None


def _blocks_of(statement: ast.stmt) -> list[tuple[str, list[ast.stmt]]]:
    """``(header, body)`` for each block a compound statement opens.

    The nesting is the whole reason a skeleton beats a list of lines: two rows
    under one ``if`` were printed together or not at all, and two rows either
    side of an ``else`` cannot both have been.  Returning the headers in source
    order is what lets the reader see that without being told it.

    ``elif`` comes back as ``else:`` wrapping an ``if``, which is what it is.
    """
    if isinstance(statement, ast.If):
        blocks = [(f"if {_clip(_text(statement.test))}:", statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        head = f"for {_clip(_text(statement.target))} in {_clip(_text(statement.iter))}:"
        blocks = [(head, statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, ast.While):
        blocks = [(f"while {_clip(_text(statement.test))}:", statement.body)]
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        return blocks
    if isinstance(statement, ast.Try):
        blocks = [("try:", statement.body)]
        for handler in statement.handlers:
            caught = _clip(_text(handler.type)) if handler.type else ""
            blocks.append((f"except {caught}:" if caught else "except:", handler.body))
        if statement.orelse:
            blocks.append(("else:", statement.orelse))
        if statement.finalbody:
            blocks.append(("finally:", statement.finalbody))
        return blocks
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        held = ", ".join(_clip(_text(item.context_expr)) for item in statement.items)
        return [(f"with {held}:", statement.body)]
    return []


def _of_interest(
    statement: ast.stmt,
    printed: dict[int, list[tuple[ast.expr, ast.expr]]],
    armed: dict[int, ast.stmt],
) -> bool:
    """Whether anything under *statement* is a line printed or an arm written.

    The rule that keeps a skeleton a skeleton.  Without it a 540-line function
    comes back whole, and burying the rows a reader came for is the same
    failure this round is fixing in the report's other half.
    """
    for node in ast.walk(statement):
        if isinstance(node, ast.stmt):
            if id(node) in printed or getattr(node, "lineno", 0) in armed:
                return True
    return False


def _constant_pattern(message: ast.expr) -> str:
    """A pattern for a message with no holes in it at all.

    Worth rendering: a constant line says nothing about the value but proves
    the branch it sits in ran, and that is what a reader aligning a region
    needs from the rows around an arm.
    """
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        text = message.value
        return re.escape(text) if len(text.strip()) >= MIN_LITERAL else ""
    return ""


def _refusal(message: ast.expr) -> str:
    """Why no pattern was rendered, in the words the skeleton prints in its place.

    Said rather than left blank.  A row that silently loses its pattern reads
    as a line production does not print, and the reader would go looking for
    it in the log.
    """
    if isinstance(message, ast.JoinedStr):
        return f"too little literal ({literal_length(message)} chars)"
    if isinstance(message, ast.Constant) and isinstance(message.value, str):
        return f"too little literal ({len(message.value.strip())} chars)"
    return "the message is not a literal this walk can render"


def _clip(text: str, width: int = 72) -> str:
    """*text* on one line, cut to *width* with an ellipsis when it is longer."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "\u2026"


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


def _tuple_bindings(func: ast.AST, name: str) -> list[tuple[ast.Assign, Optional[int]]]:
    """Assignments that bind *name* as one element of an unpacking, and which.

    ``pathcond.assigned_expressions`` deliberately reads only ``ast.Name``
    targets, and it is shared with the build, so the form is picked up here
    instead of widening it -- the map's output must not move for a change to
    the walk.  The form matters: ``tmpStat, taskSpec = getTaskWithID_JEDI(...)``
    is how a spec arrives, and without it a guard reading ``taskSpec`` looks
    like a name nothing binds.

    The slot comes back with it because the site is only half the answer.
    ``tmpStat, taskSpec = get(...)`` rendered as ``taskSpec = get(...)`` says
    the call returns the spec, when it returns a pair whose second element is
    the spec -- and the first is the status that decides whether the second
    means anything.  38 of 218 binding steps over the 41-case sample are this
    shape.  A ``*rest`` in the target makes the position of everything after it
    depend on the length of the value, so the slot is left unsaid there rather
    than guessed.
    """
    found: list[tuple[ast.Assign, Optional[int]]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, (ast.Tuple, ast.List)):
                continue
            starred = any(isinstance(e, ast.Starred) for e in target.elts)
            for slot, element in enumerate(target.elts):
                if isinstance(element, ast.Name) and element.id == name:
                    found.append((node, None if starred else slot))
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
        observed: str = "",
    ) -> None:
        self.roots = roots
        self.classify = classify
        self.budget = budget
        self.observed = observed
        self.parsed: dict[str, tuple[str, ast.Module]] = {}
        self.frames: dict[tuple[str, str], Optional[_Frame]] = {}
        self.steps: list[TraceStep] = []
        self.armed: dict[int, ast.stmt] = {}
        self.seen: set[tuple[str, str]] = set()
        self._scopes: dict[tuple[str, str], str] = {}
        self._logged: dict[tuple[str, str], list[tuple[ast.expr, ast.expr]]] = {}
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
        # Kept for the skeleton, which is built once for the whole function
        # after the walk rather than once per arm: the lines a function prints
        # do not change with which arm is being explained, and rendering them
        # per arm is what made the old shape repeat itself.
        self.armed[line] = statement
        if not terminal:
            self.want(frame, reads, 1)

    def logged(self, frame: _Frame) -> list[tuple[ast.expr, ast.expr]]:
        """The logging calls in this frame's function, found once per function."""
        key = (frame.file, frame.owner)
        if key not in self._logged:
            self._logged[key] = (
                _logged_arguments(frame.func)
                if isinstance(frame.func, (ast.FunctionDef, ast.AsyncFunctionDef))
                else []
            )
        return self._logged[key]

    # -- skeleton ----------------------------------------------------------

    def reachable_calls(self, frame: _Frame) -> list[tuple[ast.expr, ast.expr]]:
        """The logging calls in this frame, less the messages that cannot reach one.

        ``_logged_arguments`` resolves ``log.debug(msg_str)`` one hop through a
        local, and ``assigned_expressions`` hands back *every* assignment to
        that name in the function.  Five of them for one call is common, and
        four of the five are in branches the call cannot be reached from.  The
        build cannot narrow it -- a stored answer has to hold for every caller
        -- but at use time two rules in this module already do:

        An assignment below the call cannot have run before it.  And an
        assignment the call's own path condition contradicts cannot have run
        on the way to it, which is the same exclusivity test the arms use.

        A narrow fix, measured: 2188 (call, message) pairs become 1843, and
        117 of 1568 call sites shrink at all.  The function this was found in
        held eleven of them.
        """
        out: list[tuple[ast.expr, ast.expr]] = []
        for argument, message in self.logged(frame):
            if message is not argument and not pathcond.can_reach(message, argument):
                continue
            out.append((argument, message))
        return out

    def _value_hole(
        self, frame: _Frame, message: ast.JoinedStr, arms: Sequence[ast.stmt]
    ) -> tuple[str, str, str]:
        """``(pattern, hole, because)`` for the hole the observed value lands in.

        The one thing the map's shared sentence cannot pick.  The arm's own
        statement and the value actually observed are both in hand here, so
        the hole can be chosen by what it *spells* rather than by where it
        sits -- the map anchors on the text before the first hole, and for 80
        of the 182 subjects that get a probe at all that is not the hole the
        value fills.

        First choice is a hole one of *arms* names, which is a fact about a
        write.  Second is a hole holding a name a reaching definition settles
        to the observed value, which is only a fact about the function.  Empty
        when neither, and nothing is guessed to fill the gap: an arm no line
        reports keeps its silence, which is the honest answer.

        *arms* is every arm this row could have been printed alongside, and
        the names they write are taken together.  A row is in the skeleton
        once, so a value pattern that holds for any of them belongs on it --
        and which ones those are is on the row, in ``arms``, rather than
        implied by there being a value at all.
        """
        if not self.observed:
            return "", "", ""
        written: set[str] = set()
        for arm in arms:
            written.update(_written_texts(arm))
        holes = [
            (position, part)
            for position, part in enumerate(message.values)
            if isinstance(part, ast.FormattedValue)
        ]
        fallback = ("", "", "")
        for position, part in holes:
            if not _neighbouring_literal(message, position):
                continue
            spelled = _text(part.value)
            rendered = _rendered(message, position, self.observed)
            if not rendered:
                continue
            if spelled in written:
                return rendered, spelled, "the arm writes it"
            if not fallback[0] and isinstance(part.value, ast.Name):
                if self._holds_observed(frame, part.value.id):
                    fallback = (rendered, spelled, f"{spelled} is set to this value above")
        return fallback

    def skeleton(self, frame: _Frame) -> list[SkeletonLine]:
        """What this function prints, in source order, with the arms in place.

        Built from the whole function rather than from the arms outward, and
        that is the point: a line that says nothing about the value still
        proves the code between two of them ran, so the rows that carry no
        hole are as much of the answer as the rows that do.

        Only the parts of the tree that lead to a printed line or an arm are
        walked into.  A skeleton is not the source again -- a 540-line
        function whose every branch is reproduced buries the handful of rows a
        reader came for, which is the failure this round is elsewhere fixing.
        """
        func = frame.func
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return []
        printed: dict[int, list[tuple[ast.expr, ast.expr]]] = {}
        carried: set[int] = set()
        for argument, message in self.reachable_calls(frame):
            statement = _enclosing_statement(argument)
            if statement is not None:
                printed.setdefault(id(statement), []).append((argument, message))
                carried.add(id(argument))
        # A call left with no message at all -- a bare name nothing in the
        # function binds, or one whose every binding is out of reach -- used to
        # leave no row.  The reader lays this over a grep of the log region, so
        # a line that is there and not here reads as certainty rather than as
        # a gap.  Refused out loud, in the words the other refusals use.
        for argument in logging_arguments(frame.func):
            if id(argument) in carried:
                continue
            statement = _enclosing_statement(argument)
            if statement is not None:
                printed.setdefault(id(statement), []).append((argument, argument))
        rows: list[SkeletonLine] = []
        self._emit(frame, func.body, 0, printed, rows)
        return rows

    def _emit(
        self,
        frame: _Frame,
        body: Sequence[ast.stmt],
        depth: int,
        printed: dict[int, list[tuple[ast.expr, ast.expr]]],
        rows: list[SkeletonLine],
    ) -> None:
        for statement in body:
            if not _of_interest(statement, printed, self.armed):
                continue
            blocks = _blocks_of(statement)
            if blocks:
                # A call in the header itself -- ``if log.warning(x):`` is
                # absurd but ``with open(name(log.debug(x))):`` is not, and
                # ``_enclosing_statement`` hands both back keyed on the
                # compound.  Emitted before the header so the row is not lost
                # to a branch that only ever recurses.
                for row in self._rows_for(frame, statement, depth, printed):
                    if row.kind == SKELETON_PRINT:
                        rows.append(row)
                # Every header of a statement that is worth showing at all,
                # even where that block holds nothing.  Dropping the empty
                # ones produced an ``except Exception:`` with no ``try:`` over
                # it and an ``else:`` whose condition was nowhere on the page
                # -- and the nesting is the thing a reader is here for, so a
                # header that explains a shown block is not decoration.
                for header, inner in blocks:
                    rows.append(
                        SkeletonLine(
                            kind=SKELETON_BRANCH,
                            line=getattr(statement, "lineno", 0),
                            depth=depth,
                            text=header,
                        )
                    )
                    if any(_of_interest(one, printed, self.armed) for one in inner):
                        self._emit(frame, inner, depth + 1, printed, rows)
                continue
            for row in self._rows_for(frame, statement, depth, printed):
                rows.append(row)

    def _rows_for(
        self,
        frame: _Frame,
        statement: ast.stmt,
        depth: int,
        printed: dict[int, list[tuple[ast.expr, ast.expr]]],
    ) -> list[SkeletonLine]:
        line = getattr(statement, "lineno", 0)
        rows: list[SkeletonLine] = []
        for argument, message in printed.get(id(statement), ()):
            # Compatibility only, not :func:`pathcond.can_reach`: a line printed after
            # the arm is printed alongside it just as much as one before, and
            # the position rule would drop exactly the lines that say what the
            # arm went on to do.
            arms = [
                arm for arm, held in self.armed.items() if pathcond.compatible(held, argument)
            ]
            if isinstance(message, ast.JoinedStr):
                pattern = line_pattern(message)
                value, hole, because = self._value_hole(
                    frame, message, [self.armed[arm] for arm in arms]
                )
            else:
                pattern = _constant_pattern(message)
                value, hole, because = "", "", ""
            refused = "" if pattern else _refusal(message)
            # Deduped on what the row would *say*, refusals included: one call
            # can resolve to several messages that render the same, and three
            # identical "too little literal" rows on one line tell a reader
            # nothing three times.
            if any(
                (row.pattern, row.text) == (pattern, _text(message)) for row in rows
            ):
                continue
            rows.append(
                SkeletonLine(
                    kind=SKELETON_PRINT,
                    line=line,
                    depth=depth,
                    text=_text(message),
                    pattern=pattern,
                    refused=refused,
                    arms=sorted(arms),
                    value=value,
                    hole=hole,
                    because=because,
                )
            )
        if line in self.armed:
            rows.append(
                SkeletonLine(
                    kind=SKELETON_ARM,
                    line=line,
                    depth=depth,
                    text=_text(statement),
                )
            )
        return rows

    def _holds_observed(self, frame: _Frame, name: str) -> bool:
        """Whether a reaching definition in this function settles *name* to it."""
        if not isinstance(frame.func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        return any(
            value == self.observed
            for value, _conditions, _line in pathcond.literal_values(frame.func, name)
        )

    def explain(self, frame: _Frame, name: str, depth: int) -> None:
        if name.startswith("self."):
            self._field(frame, name, depth)
            return
        bindings = list(pathcond.assigned_expressions(frame.func).get(name, []))
        unpacked = _tuple_bindings(frame.func, name)
        slots = {id(node.value): slot for node, slot in unpacked}
        bindings.extend(node.value for node, _slot in unpacked)
        bindings.extend(_with_bindings(frame.func, name))
        bindings.extend(_walrus_bindings(frame.func, name))
        loops = _loop_bindings(frame.func, name)
        caught = _caught_as(frame.func, name)
        held = bool(bindings or loops or caught)
        if depth == 1 and self.armed:
            # ``assigned_expressions`` hands back every assignment in the
            # function and the arms are at one point in it.  Pruning is by
            # elimination, so a site that cannot have run on the way to any of
            # them does not merely add noise: it dilutes "one of these three"
            # into "one of these four" and sends a reader down a path the code
            # did not take.  Measured on the 41-case sample, eight of 287
            # steps.
            #
            # Depth one only, and that is a limit rather than a choice.  A
            # name is explained once for every place that reads it, so deeper
            # than this the reading point is whichever want arrived first and
            # filtering against it would drop sites the other readers can
            # reach.  At depth one the readers are the arms, all of them
            # armed before the queue drains, and a site is kept if it reaches
            # any.
            arms = list(self.armed.values())
            bindings = [b for b in bindings if pathcond.reaches_one(pathcond.site_of(b), arms)]
            loops = [loop for loop in loops if pathcond.reaches_one(loop, arms)]
            caught = [one for one in caught if pathcond.reaches_one(one, arms)]
            if held and not (bindings or loops or caught):
                # Bound here, and by nothing that could have run first.
                # "Not bound in this function" would be false and silence
                # would leave the guard unexplained, so it says which.
                self._unbound(frame, name, depth, only_below=True)
                return
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
            self._binding(frame, name, expression, depth, slots.get(id(expression)))
        for loop in loops:
            self._loop(frame, name, loop, depth)
        for handler in caught:
            self._caught(frame, name, handler, depth)

    def _binding(
        self,
        frame: _Frame,
        name: str,
        expression: ast.expr,
        depth: int,
        slot: Optional[int] = None,
    ) -> None:
        site = pathcond.site_of(expression)
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
            value=_text(expression) if slot is None else f"{_text(expression)}[{slot}]",
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

    def _unbound(
        self, frame: _Frame, name: str, depth: int, only_below: bool = False
    ) -> None:
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
                    else "bound in this function only below the arms, and in no loop that "
                    "holds both, so none of those can have set it"
                    if only_below
                    else "not bound in this function -- module scope, an import, or a form the walk does not read"
                ),
                depth=depth,
            )
        )


class Walked(NamedTuple):
    """What one walk found.

    A named tuple rather than a pair because the walk answers two questions
    about the same function -- why the arms ran, and what the function prints
    when they do -- and the second is empty whenever the first is refused.

    The two are siblings rather than one inside the other because they are
    ordered on different axes.  ``steps`` runs along data dependence and
    crosses into other functions; the skeleton runs in source order and stays
    in this one, because the thing it is laid against is a log region and a
    log region is in time order.  Folding the skeleton into ``steps`` would
    put rows that say nothing about the value into a list whose every row is
    about the value.
    """

    steps: list[TraceStep]
    note: str
    skeleton: list[SkeletonLine] = []


def walk(
    roots: dict[str, Path],
    *,
    file: str,
    owner: str,
    lines: Sequence[int],
    observed: str = "",
    expected_sha: str = "",
    handovers: Sequence[Handover] = (),
    classify: Optional[Classifier] = None,
    budget: Budget = DEFAULT_BUDGET,
) -> Walked:
    """Explain the arms at *lines*, and say why the answer is short when it is.

    *lines* are the arms of one function, walked together so that a flag they
    share is explained once.  Returns the steps and a note -- empty when the
    walk finished on its own, which the measurement says is the usual case.

    Also returns the skeleton: what this function prints, in source order,
    with the arms in place -- see
    :class:`~bamboo.codemap.models.SkeletonLine`.  *observed* is the value the
    record actually holds, and given one the rows whose hole it lands in carry
    the narrower pattern with it filled in as well.  Without one the skeleton
    still comes back, because a line that reports nothing about the value
    still says the code around it ran.

    **A tree that is not the map's is refused rather than warned about.**  A
    trace is shaped like an answer, and reading line 4971 of the wrong release
    produces plausible code that is not the code -- which already happened
    once, to a report that only printed coordinates.  Computing from the wrong
    tree is worse than printing from it.
    """
    state = _Walk(roots, classify or _nothing, budget, observed)
    parsed = state.module(file)
    if parsed is None:
        return Walked([], "the map's file could not be read from this tree")
    source, _tree = parsed
    if expected_sha and git_blob_sha(source) != expected_sha:
        return Walked([], "the walk was not run: this tree is not the one that was mapped")
    if not lines:
        return Walked([], "the map records no line for these arms")
    frame = state.frame(file, owner, line=lines[0], handovers=handovers)
    if frame is None:
        return Walked([], "the map's function could not be located in this tree")
    for line in lines:
        state.arm(frame, line)
    while state.queue:
        held, name, depth = state.queue.popleft()
        state.explain(held, name, depth)
    return Walked(state.steps, state.note, state.skeleton(frame))
