"""Loop-cut recognizer -- the fourth filter idiom, the one with no vocabulary.

Three ways of dropping a candidate are already read: a message carrying
``criteria=-diskIO``, a rejection counted by a ``candidates passed`` step, and a
branch that appends the survivor in one arm and not the other.  All three are
brokerage, and all three work because brokerage declares itself.  The fourth is
what the rest of the corpus does::

    for workQueue in workQueueList:
        for resource_type in resource_types:
            ...
            if not flagLocked:
                tmpLog_inner.debug("skip since locked by another process")
                continue

Nothing here is named.  The loop, the guard and the ``continue`` are the whole
structure, and the sentence is the only thing that distinguishes one cut from
the next.  That is why this went unread for so long: there is no token to grep
the source for, so it does not turn up by looking for one.

**The structure is the proof; the sentence is only the name.**  This was the
plan's one substantive error and it took the motivating case with it.  The rule
written down was "the message interpolates the loop variable", which reads
respectably and scores 124 cuts -- and misses all three guards of
``JobGenerator.start``'s ``(workQueue, resource_type)`` loop, the case the whole
round exists for.  Two of the three interpolate nothing at all, and the third
interpolates ``cycleStr``, which is built from the loop variables two hops away
through ``.format()``.  Chasing that would have meant a third hop, then a
fourth, which is how P1-15's flag substitution ended up reverted.  A guarded
``continue`` in a loop drops the iteration -- that is what ``continue`` means --
so no candidate test is needed, and adding one only decides which real cuts to
throw away.

Brokerage is left to the stage recognizer.  Not a scope convenience: 37 of
these sit inside the five functions that already have stages, and reading them
twice would put a second, differently-keyed node on cuts the funnel already
counts.  A function whose chain is read over there is not read here.

Measured over ``panda-server-source 1.0.2``::

    continue                                    834
      inside a loop                             834
      closed by a guard                         797
        with a logged reason in the same block  200
          already carrying a criteria= tag        5
          inside a chain the stage slice reads   37
          new ground                            153   (75 functions, 36 files)
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, Optional

from bamboo.codemap.models import Anchor, LoopCutNode, SourceModule
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    enclosing_function,
    functions_with_owner,
    path_condition,
    targets_of,
)
from bamboo.codemap.panda.recognizers.selection import _TAG, _identifiers, _rendered

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def _innermost_loop(node: ast.AST) -> Optional[ast.AST]:
    """The loop whose iteration *node* would skip, or ``None`` outside one."""
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, _LOOPS):
            return parent
        if isinstance(parent, _FUNCS):
            return None
        parent = getattr(parent, "parent", None)
    return None


def _is_guarded(node: ast.AST) -> bool:
    """Whether an ``if`` decides that *node* runs, rather than the loop itself.

    An unconditional ``continue`` at the end of a loop body is control flow, not
    a cut: every candidate reaches it.  The walk stops at the loop and at the
    function, so an ``if`` outside the loop does not count -- it decides whether
    the loop runs at all, which is a different claim.

    ``except`` handlers are not guards either, and fall out without being
    excluded by hand: a handler is not an ``ast.If``.  That is the right answer
    rather than a lucky one -- a candidate dropped because the code raised was
    not tested and rejected, and the message says so.
    """
    parent = getattr(node, "parent", None)
    while parent is not None:
        if isinstance(parent, ast.If):
            return True
        if isinstance(parent, _LOOPS + _FUNCS):
            return False
        parent = getattr(parent, "parent", None)
    return False


def _preceding(node: ast.stmt) -> list[ast.stmt]:
    """The statements before *node* in its own block.

    Its own block and not the whole guard: the reason is stated on the way out,
    next to the ``continue``, and statements in an enclosing block ran for every
    candidate including the ones that survived.
    """
    parent = getattr(node, "parent", None)
    if parent is None:
        return []
    for field in ("body", "orelse", "finalbody"):
        block = getattr(parent, field, None)
        if isinstance(block, list) and node in block:
            return block[: block.index(node)]
    return []


def _logging_calls(statement: ast.stmt) -> Iterator[ast.Call]:
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LEVELS
        ):
            yield node


def _last_assignment(
    name: str, func: ast.FunctionDef | ast.AsyncFunctionDef, before: int
) -> Optional[ast.stmt]:
    """The last statement above *before* that assigns *name*."""
    found: Optional[ast.stmt] = None
    for node in ast.walk(func):
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            continue
        if node.lineno >= before:
            continue
        targets = [node.target] if isinstance(node, ast.AugAssign) else targets_of(node)
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if found is None or node.lineno > found.lineno:
            found = node
    return found


def _message(
    call: ast.Call, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> Optional[str]:
    """The text a logging call puts on the line, following one local hop.

    The hop is not optional.  A message assembled into a local first --
    ``tmpMsg = f"..."`` then ``tmpLog.info(tmpMsg)`` -- is the same line to
    production and invisible to a reader that only looks at the argument.  It is
    also the shape that hid ``criteria=-link_unusable`` from the tag recognizer
    until production named it from the other side.  One hop and no further: the
    second hop is where the message stops being a sentence and starts being an
    accumulation, and what it renders to then is not what any single line says.
    """
    if not call.args:
        return None
    argument = call.args[0]
    text = _rendered(argument)
    if text is not None:
        return text
    if not isinstance(argument, ast.Name):
        return None
    assignment = _last_assignment(argument.id, func, call.lineno)
    return _rendered(assignment.value) if assignment is not None else None


def _search_key(message: str) -> str:
    """The longest fixed run of *message* -- what to ask production for.

    Longest run rather than the leading one, for the reason the diagnostic
    template index uses the same rule: ``"  skip site={} ..."`` shares its first
    ten characters with half the file, so a key taken from the front confirms
    nothing.

    No length floor, though one was written and measured first.  Keys run from
    four characters to forty-five with no gap anywhere -- 4, 5, 6, 7, 8, 9, 10,
    11, 12 are all occupied -- so any cut-off is a number this corpus does not
    supply, and the repo has paid for those before.  What actually makes a key
    unusable is that it selects more than this cut, and that is answerable:
    ``build-map`` reports the keys that also match another cut in the same log
    file, and the evidence layer's ``truncated`` already says when an answer
    came back capped.  Stating the run and letting those two speak keeps the
    judgement where the information is.
    """
    return max((part.strip() for part in message.split("{}")), key=len, default="")


def _scope_prefix(
    call: ast.Call, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    """The prefix the logger this call writes through stamps on every line.

    Which candidate a line is about is usually not in the sentence.
    ``JobGenerator`` wraps its logger once per iteration with
    ``pid={} vo={} cloud={} queue={} ( id={} ) label={} resource_type={}`` and
    then logs ``"throttled"``, so the queue and the resource type -- the whole
    identity of the candidate -- are in the prefix and the message is a bare
    word.  A question scoped by task id cannot narrow that; a question scoped by
    queue can, and the key to do it with is written on the cut itself.

    Read at most two hops: the wrapper's second argument, and the local it was
    built from.  ``.format()`` is unwrapped to its template because that is what
    the line is stamped with -- the arguments are the run's values, and holes
    are what a production pattern wants there anyway.
    """
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return ""
    built = _last_assignment(call.func.value.id, func, call.lineno)
    if built is None or not isinstance(built.value, ast.Call) or len(built.value.args) < 2:
        return ""
    argument = built.value.args[1]
    text = _rendered(argument)
    if text is None and isinstance(argument, ast.Name):
        inner = _last_assignment(argument.id, func, built.lineno)
        if inner is not None:
            text = _rendered(inner.value)
            if text is None and isinstance(inner.value, ast.Call):
                callee = inner.value.func
                if isinstance(callee, ast.Attribute) and callee.attr == "format":
                    text = _rendered(callee.value)
    # ``.format`` numbers its holes and f-strings do not, and the rest of the
    # map renders every hole as ``{}``.  Two spellings of the same thing would
    # make a consumer build the production pattern two ways.
    return re.sub(r"\{\d+\}", "{}", (text or "")).strip()


class _Cut:
    """One guarded ``continue``, pooled over the sites that word it alike."""

    def __init__(self, message: str, line: int, loop_line: int, level: Optional[str]) -> None:
        self.message = message
        self.line = line
        self.loop_line = loop_line
        self.level = level
        self.conditions: list[str] = []
        self.prefix = ""

    def add(self, conditions: list[str], prefix: str) -> None:
        for condition in conditions:
            if condition not in self.conditions:
                self.conditions.append(condition)
        self.prefix = self.prefix or prefix


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    covered: set[str],
) -> tuple[list[LoopCutNode], int]:
    """Extract loop cuts, and count the guarded ``continue``s that say nothing.

    *covered* names the functions the stage recognizer already reads, by
    ``module::function``; their cuts are its business and are skipped here.

    No coverage stat, and the first version had one.  A ratio needs a
    denominator that means something, and the only candidate here is "every
    guarded ``continue`` inside a loop", which is 685 statements and mostly not
    filtering at all -- ``JediTaskSpec`` and ``InputChunk`` contribute
    seventeen between them from ordinary iteration in pure computation.  Scored
    that way the slice reads 22% and puts 135 files in the low-coverage list,
    which says nothing about the extraction and buries the slices whose
    coverage does.  What separates a filter from iteration control *is* the
    logged reason, so the population and the reading are the same set.  Same
    call as ``tables_never_written``: kept out of the matrix because there is
    no denominator to be a fraction of.

    The silent ones are still counted and reported, because "532 loops drop a
    candidate and do not say why" is a fact about PanDA worth one line.
    """
    cuts: list[LoopCutNode] = []
    silent = 0

    for module in modules:
        attach_parents(module.tree)
        for func, _owner in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            # Keyed by message, so several sites wording a cut alike are one
            # node: production cannot tell them apart either, and a reader who
            # greps the sentence gets all of them.  Their guards pool, which is
            # the honest reading -- any of those conditions drops a candidate
            # onto this line.
            found: dict[str, _Cut] = {}
            for node in ast.walk(func):
                if not isinstance(node, ast.Continue):
                    continue
                if enclosing_function(node) is not func:
                    continue
                loop = _innermost_loop(node)
                if loop is None or not _is_guarded(node):
                    continue
                if owner in covered:
                    continue
                calls = [
                    call
                    for statement in _preceding(node)
                    for call in _logging_calls(statement)
                ]
                if not calls:
                    silent += 1
                    continue
                # The last one: a block that logs twice states the reason on the
                # line nearest the exit, and the earlier one is context.
                call = calls[-1]
                message = _message(call, func)
                if message is None or _TAG.search(message):
                    # A tagged rejection outside a brokerage chain is still the
                    # tag slice's to record, keyed on the tag that production
                    # carries.  Recording it again here would give one cut two
                    # identities.
                    silent += 1
                    continue
                cut = found.get(message)
                if cut is None:
                    cut = _Cut(message, node.lineno, loop.lineno, call.func.attr)
                    found[message] = cut
                cut.add(path_condition(node), _scope_prefix(call, func))

            for order, cut in enumerate(sorted(found.values(), key=lambda c: c.line)):
                cuts.append(_node(map_id, derived_from, module, owner, order, cut))
    return cuts, silent


def _signature(cut: _Cut) -> str:
    """What identifies this cut, independently of where it sits in the file.

    The message, for the reason a tag identifies a filter stage: it is what
    production prints and what a reader looks for, so a cut keyed on it survives
    every edit that keeps the wording and moves when the wording moves.  Line
    numbers are not identity -- the loop shifts whenever anything above it does,
    and a node keyed on position reports a change on every unrelated edit.
    """
    return re.sub(r"\s+", " ", cut.message).strip()


def _node(
    map_id: str,
    derived_from: str,
    module: SourceModule,
    owner: str,
    order: int,
    cut: _Cut,
) -> LoopCutNode:
    inputs: list[str] = []
    for condition in cut.conditions:
        for name in _identifiers(condition):
            if name not in inputs:
                inputs.append(name)
    return LoopCutNode(
        map_id=map_id,
        derived_from=derived_from,
        name=LoopCutNode.make_name(map_id, owner, _signature(cut)),
        owner=owner,
        message=cut.message,
        search_key=_search_key(cut.message),
        scope_prefix=cut.prefix,
        order=order,
        loop_line=cut.loop_line,
        conditions=cut.conditions,
        inputs=inputs,
        log_level=cut.level,
        anchor=Anchor(
            package=module.package,
            file=module.rel_path,
            line_start=cut.line,
            blob_sha=module.blob_sha,
        ),
    )
