"""The write that happens somewhere other than where the value was decided.

JEDI is a read-modify-write system, and most of its writes are spelled the way
that implies::

    taskSpec.status = self.getFinalTaskStatus(taskSpec)   # decided here
    tmpLog.info(f"set task_status={taskSpec.status}")      # announced here
    self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": ...})   # written here

The proxy end of that is a generic flush::

    sqlU = f"UPDATE {schema}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "

whose ``SET`` clause the map cannot read, because there is nothing to read: the
columns are ``_changedAttrs``, which ``JediTaskSpec.__setattr__`` filled from
the caller's assignments.  So the write is not invisible, it is *displaced* --
the attribute slice already holds every decision at full coverage, and what was
missing is that the flush is where they land.

Two things follow, and only the second needs recording.

The decision is already a junction, so no new node is warranted here.  What is
not already recorded is the flush's own ``WHERE``: ``updateTask_JEDI`` appends
``status IN (:old_...)`` when its caller supplies ``oldStatus``, and twelve of
its twenty-two call sites do.  That is the same compare-and-set the SQL slice
records on a statement it can read whole, arriving in a different spelling --
and letting the two spellings disagree about one fact is the failure this map
has had to correct more than once.

Three readings keep it from over-claiming, each answering a way the naive rule
is wrong:

*Which columns the flush writes* is what the caller changed, since that is what
``_changedAttrs`` holds -- so a guard on a column this caller never touched is
row selection, not a race.

*Minus what the flush resets.*  ``updateTask_JEDI`` opens with
``taskSpec.resetChangedAttr("jediTaskID")``, which is the method saying outright
that it does not write the column its ``WHERE`` identifies rows by.

*Minus guards the caller did not ask for.*  The ``status IN`` fragment is
appended under ``if oldStatus is not None``, a path condition on a parameter,
so whether it applies is settled one step up at the call site -- the one hop
the backward-slicing rules allow for a formal argument.  Without it the ten
callers that pass no ``oldStatus`` would be told their write races when it
does not, and a caveat invented is worse here than a caveat missed: it is the
confirmed candidates it weakens.
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, NamedTuple, Optional

from bamboo.codemap.models import Emit, JunctionNode, SourceModule, SubjectNode
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    path_condition,
)
from bamboo.codemap.panda.recognizers import logfile
from bamboo.codemap.panda.recognizers.emit import row_count_lines
from bamboo.codemap.panda.recognizers.trigger import sole_definitions
from bamboo.codemap.panda.values import rendered_text

#: The spec classes declare this to build a ``SET`` clause out of whatever was
#: assigned to them.  Its presence is the method saying it is a flush.
FLUSH_EXPRESSION = "bindUpdateChangesExpression"

#: And this to take a column back out of that clause.
RESET_CHANGED = "resetChangedAttr"


class Flush(NamedTuple):
    """A method that persists whatever its caller changed on a spec."""

    spec_classes: frozenset[str]
    #: column -> (the predicate term, the parameters whose presence appends it)
    guards: dict[str, tuple[str, frozenset[str]]]
    #: columns the method takes back out of the clause before writing
    excluded: frozenset[str]
    parameters: tuple[str, ...]
    #: lines the method writes about how many rows its write changed
    rows_emits: tuple[Emit, ...] = ()


def _string_fragments(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[tuple[ast.stmt, str]]:
    """Every statement in *func* that appends literal text to a local name."""
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        if not targets:
            continue
        text = rendered_text(node.value)
        if text:
            yield node, text


def _parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Positional parameters in order, ``self`` included so indices line up."""
    return tuple(arg.arg for arg in func.args.args) + tuple(
        arg.arg for arg in func.args.kwonlyargs
    )


def _gating_parameters(
    func: ast.FunctionDef | ast.AsyncFunctionDef, term: str, parameters: tuple[str, ...]
) -> frozenset[str]:
    """Parameters whose value decides whether *term* is appended to the clause.

    Matched on the whole term rather than on the column name.  The column name
    is not selective enough: ``updateTask_JEDI`` also builds a ``T_TASK`` update
    mentioning ``status`` under ``if updateDEFT``, and reading that fragment's
    condition made the guard look conditional on a parameter that has nothing
    to do with it -- which then dropped the guard for every caller relying on
    that parameter's default.
    """
    gating: set[str] = set()
    for node, text in _string_fragments(func):
        if term not in re.sub(r"\s+", " ", text):
            continue
        for condition in path_condition(node):
            gating.update(
                name
                for name in parameters
                if name != "self" and re.search(rf"\b{re.escape(name)}\b", condition)
            )
    return frozenset(gating)


def flush_methods(
    modules: list[SourceModule], attributor: SpecAttributor
) -> dict[str, Flush]:
    """Return ``{method name: what it flushes}`` for the generic-update idiom.

    Keyed by name, so restricted to names one module implements -- the same
    limit the trigger slice puts on a cross-module hop, for the same reason.
    """
    only_here = sole_definitions(modules)
    declared = logfile.declared_files(modules)
    inherited = logfile.inherited_files(modules, declared)
    found: dict[str, Flush] = {}
    for module in modules:
        # Whether a fragment is appended under a test is half of what this
        # reader needs, and that is unreadable without parent links.  Attached
        # here rather than relied on from an earlier slice: a recognizer that
        # works only when another ran first fails silently when it does not.
        attach_parents(module.tree)
        for func, owner in functions_with_owner(module.tree):
            if only_here.get(func.name) != module.rel_path:
                continue
            classes: set[str] = set()
            excluded: set[str] = set()
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr == FLUSH_EXPRESSION:
                    spec = attributor.class_of(node.func.value, func, owner)
                    if spec:
                        classes.add(spec)
                elif node.func.attr == RESET_CHANGED and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        excluded.add(argument.value)
            if not classes:
                continue
            parameters = _parameters(func)
            guards: dict[str, tuple[str, frozenset[str]]] = {}
            for run in sql.executions(func):
                for match in sql._UPDATE.finditer(run.sql):
                    if match.group(2).strip().strip(",") not in ("{}", ""):
                        # A readable ``SET`` is the SQL slice's business; this
                        # reader exists for the clause that is a single hole.
                        continue
                    for raw, column in sql._TERM.findall(run.sql[match.end(2) :]):
                        if column in guards:
                            continue
                        term = re.sub(r"\s+", " ", raw).strip()
                        guards[column] = (
                            term,
                            _gating_parameters(func, term, parameters),
                        )
            found[func.name] = Flush(
                spec_classes=frozenset(classes),
                guards=guards,
                excluded=frozenset(excluded),
                parameters=parameters,
                rows_emits=tuple(
                    row_count_lines(
                        func, logfile.files_of(module.rel_path, declared, inherited)
                    )
                ),
            )
    return found


def _supplied(call: ast.Call, parameter: str, parameters: tuple[str, ...]) -> bool:
    """Whether this call passes *parameter* something other than ``None``."""
    for keyword in call.keywords:
        if keyword.arg == parameter:
            return not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is None
            )
        if keyword.arg is None:
            # ``**kwargs``: what it holds is not readable here, so the guard
            # is left in rather than ruled out.
            return True
    if parameter in parameters:
        # ``self`` is in the signature and never in the call's arguments.
        index = parameters.index(parameter) - (1 if parameters and parameters[0] == "self" else 0)
        if 0 <= index < len(call.args):
            argument = call.args[index]
            return not (isinstance(argument, ast.Constant) and argument.value is None)
    return False


def _changed_attributes(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    attributor: SpecAttributor,
    spec_classes: frozenset[str],
) -> dict[str, set[str]]:
    """``{spec class: attributes assigned on it}`` inside *func*."""
    changed: dict[str, set[str]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            spec = attributor.class_of(target.value, func, owner)
            if spec in spec_classes:
                changed.setdefault(spec, set()).add(target.attr)
    return changed


def attach(
    junctions: list[JunctionNode], modules: list[SourceModule], attributor: SpecAttributor
) -> int:
    """Give each caller's branches the row its flush needed.  Returns how many.

    Applied to every branch of a matching junction rather than to the writes
    that provably reach the flush call: a method that changes a spec and then
    flushes it flushes what it changed, and pairing each assignment with the
    call that dominates it would be a dominance analysis for a distinction the
    corpus does not appear to make.  Stated because it over-reads: a function
    that flushes on one path and not another has the caveat on both.
    """
    flushes = flush_methods(modules, attributor)
    if not flushes:
        return 0
    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in junctions:
        by_owner.setdefault(junction.owner, []).append(junction)

    annotated = 0
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            here = by_owner.get(f"{module.rel_path}::{func.name}")
            if not here:
                continue
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                flush = flushes.get(node.func.attr)
                if flush is None:
                    continue
                changed = _changed_attributes(func, owner, attributor, flush.spec_classes)
                for spec, attributes in changed.items():
                    written = attributes - flush.excluded
                    guards = sorted(
                        term
                        for column, (term, gating) in flush.guards.items()
                        if column in written
                        and all(_supplied(node, p, flush.parameters) for p in gating)
                    )
                    if not guards and not flush.rows_emits:
                        continue
                    for junction in here:
                        if junction.subject not in {
                            SubjectNode.make_name(spec, attribute) for attribute in written
                        }:
                            continue
                        for branch in junction.branches:
                            missing = [g for g in guards if g not in branch.row_precondition]
                            if missing:
                                branch.row_precondition = sorted(
                                    branch.row_precondition + missing
                                )
                                annotated += 1
                            # The line saying whether the row actually moved is
                            # written by the flush, in the flush's own log, and
                            # the code that chose the value never sees it.
                            for line in flush.rows_emits:
                                if any(e.template == line.template for e in branch.emits):
                                    continue
                                branch.emits.append(line)
    return annotated
