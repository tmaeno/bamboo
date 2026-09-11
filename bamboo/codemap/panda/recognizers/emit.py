"""The line a junction leaves behind when it settles a value.

Filter stages have carried their message templates from the start -- 94 of 109
have one and 104 have a level -- and junctions never did: 0 of 1066 branches.
The mechanism existed and was simply never pointed at the other node kind, and
what that cost is visible in the consumer.  ``strategy.line_shape`` had to keep
a hard-coded dictionary of one entry, ``set task_status={value}``, so
``derive-strategy`` could observe exactly one subject and had to report every
other as a capability gap.

Written as an attach pass rather than inside the slices that build junctions,
because two of them do -- the attribute slice and the SQL slice -- and reading
the same functions twice is how the two ended up disagreeing about a fact
before.  It runs over the merged junctions, keyed by owner, the way
:mod:`bamboo.codemap.panda.recognizers.flush` does.

Three readings, each answering a way the naive rule is wrong.

**A message is usually built before it is logged.**  ``tmpMsg = f"set
task_status={taskSpec.status}"`` then ``tmpLog.info(tmpMsg)`` accounts for 99
of the 265 matching emits in the corpus, and without following it every one of
the four knights that writes ``set task_status=`` is missed -- which is every
writer the transition gate reads.  The same shape gate 1 found when
``criteria = "-link_unusable"`` was assigned before being interpolated.

**Whether a line is about this junction is not a substring test.**  The first
estimate of this harvest matched the attribute name against the rendered text,
which counted ``status`` inside ``task_status`` and would have counted every
line mentioning a status anywhere.  What ties a line to a junction is that it
interpolates the value: either an attribute of the right spec class, or a local
holding one of the values this junction writes.  ``TaskRefiner`` logs the local
and is only reachable by the second.

**A line under one arm is not evidence for the other.**  ``ContentsFeeder``
writes ``set task_status=`` in three arms of one function.  Attaching all three
to every branch would lose which arm; :func:`pathcond.exclusive` filters them,
and it is conservative in the direction that costs least here -- an emit
attached too widely means a grep that comes back empty, which the surrounding
design already treats as proving nothing, while one attached too narrowly means
no probe can be built at all.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from bamboo.codemap.models import (
    REPORTS_DECISION,
    REPORTS_ROWS_CHANGED,
    Emit,
    JunctionNode,
    SourceModule,
)
from bamboo.codemap.panda import values
from bamboo.codemap.panda.attribution import SpecAttributor
from bamboo.codemap.panda.pathcond import (
    assigned_expressions,
    attach_parents,
    exclusive,
    functions_with_owner,
    literal_values,
    path_condition,
)
from bamboo.codemap.panda.recognizers import logfile
from bamboo.codemap.panda.recognizers.selection import log_level

#: ``runtime(<expr>)`` -- how a branch records an outcome it cannot value.
#: The expression is the connection to a line that interpolates the same one.
_RUNTIME = re.compile(r"^runtime\((.*)\)$", re.DOTALL)


def _logged_arguments(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[ast.expr, ast.expr]]:
    """Every ``(argument, message expression)`` a logging call in *func* writes.

    The message is resolved one hop through a local, which is how most of the
    corpus writes it -- see the module docstring.  Both come back: the argument
    sits at the call, which is where the level and the dominating conditions
    are read from, and the expression carries the text -- and for the local
    spelling those are in two different places.
    """
    assignments = assigned_expressions(func)

    found: list[tuple[ast.expr, ast.expr]] = []
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if not node.args:
            continue
        argument = node.args[0]
        # ``log_level`` reads the level off the call *enclosing* a node, so it
        # is asked about the argument rather than about the call.
        if log_level(argument) is None:
            continue
        if isinstance(argument, ast.Name):
            found.extend((argument, built) for built in assignments.get(argument.id, ()))
        else:
            found.append((argument, argument))
    return found


def _row_counts(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Locals holding what a write returned -- ``nRows = self.cur.rowcount``.

    A line interpolating one of these is about the write landing rather than
    about the value, and that is the whole of what the map can observe of a
    compare-and-set losing: the count comes back, most callers discard it, and
    the one place it is written down is a line like ``updated 0 rows``.
    """
    found: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute)):
            continue
        if node.value.attr != "rowcount":
            continue
        found |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return found


def _reports_rows(message: ast.expr, counts: set[str]) -> bool:
    """Whether *message* interpolates a row count this function took."""
    if not isinstance(message, ast.JoinedStr) or not counts:
        return False
    return any(
        isinstance(part, ast.FormattedValue) and ast.unparse(part.value) in counts
        for part in message.values
    )


def row_count_lines(
    func: ast.FunctionDef | ast.AsyncFunctionDef, files: list[str]
) -> list[Emit]:
    """The lines *func* writes about how many rows its write changed.

    Public because the flush pass needs the same reading: a knight decides in
    memory and a proxy persists it, so the line saying whether the row moved is
    written in a different function from the one that chose the value, and the
    caller's branches are where an investigation looks for it.
    """
    counts = _row_counts(func)
    if not counts:
        return []
    found: list[Emit] = []
    for at_call, message in _logged_arguments(func):
        if not _reports_rows(message, counts):
            continue
        template = values.rendered_text(message)
        if not template or not values.has_literal_text(template):
            continue
        if any(e.template == template for e in found):
            continue
        found.append(
            Emit(
                template=template,
                log_level=log_level(at_call),
                log_files=files,
                reports=REPORTS_ROWS_CHANGED,
            )
        )
    return found


def _names_the_value(
    message: ast.expr,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    owner: Optional[str],
    attributor: SpecAttributor,
    spec_class: str,
    attribute: str,
    outcomes: set[str],
    settle,
) -> bool:
    """Whether *message* interpolates the value this junction settles.

    Three spellings, and the corpus needs all three: the line reads the spec
    back (``{taskSpec.status}``); it logs the local that was just assigned to
    it, tied by the values that local can hold rather than by its name; or the
    branch could not value the local at all and recorded ``runtime(<expr>)``,
    in which case the outcome names the very expression the line interpolates.

    The third is not a weaker version of the second -- it is what is left when
    the second fails, and it is exactly where the line is most worth having,
    since the map has nothing else to say about that branch's value.
    """
    if not isinstance(message, ast.JoinedStr):
        return False
    unvalued = {m.group(1) for o in outcomes if (m := _RUNTIME.match(o))}
    for part in message.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        held = part.value
        if isinstance(held, ast.Attribute) and held.attr == attribute:
            if attributor.class_of(held.value, func, owner) == spec_class:
                return True
        if ast.unparse(held) in unvalued:
            return True
        if isinstance(held, ast.Name):
            reached = {value for value, _conditions, _line in literal_values(func, held.id, settle)}
            if reached & outcomes:
                return True
    return False


def _about(branch, named: set[str], names_value: bool, reports: str) -> bool:
    """Whether a line is about *branch*, given what it names.

    The fourth reading, and the one the others cannot make.  Six arms of
    ``setScoutJobData_JEDI`` write ``exhausted`` and none of their lines
    interpolates the value -- they say ``action=set_exhausted reason=
    scout_cpuTime`` and let the tag carry the meaning -- so ``_names_the_value``
    finds nothing and the deciding junction ends up with no line to ask
    production for at all.

    **The branch's tags decide, not the line's.**  A tagged branch takes only a
    line covering its tags, which is what keeps the six arms apart; an untagged
    branch is judged the way it always was.  Reading it the other way round --
    a tagged *line* belongs only to a tagged branch -- cost three branches of
    ``doActionForReassign`` their line, because ``#ATM #KV label=managed
    action=trigger_new_brokerage by setting task_status={}`` does both at once
    and those branches carry no tag of their own (the block settles two
    subjects, so the tag reading declines to say which).

    A row count is about the write landing rather than about which branch chose
    it, so it is exempt.
    """
    if reports == REPORTS_ROWS_CHANGED:
        return True
    if branch.tags:
        return set(branch.tags) <= named
    return names_value


def attach(
    fragment,
    modules: list[SourceModule],
    attributor: SpecAttributor,
) -> int:
    """Give each branch the line the code writes when it fires.  Returns how many.

    Runs after :func:`logfile.attach`, which is what resolves a module's
    loggers -- and the file an emit lands in is the one of the module that
    writes the line, not the one of the node it is about.
    """
    declared = logfile.declared_files(modules)
    inherited = logfile.inherited_files(modules, declared)
    settle = values.resolver(values.declared_mappings(modules))

    by_owner: dict[str, list[JunctionNode]] = {}
    for junction in fragment.junctions:
        by_owner.setdefault(junction.owner, []).append(junction)

    attached = 0
    for module in modules:
        attach_parents(module.tree)
        files = logfile.files_of(module.rel_path, declared, inherited)
        for func, owner in functions_with_owner(module.tree):
            here = by_owner.get(f"{module.rel_path}::{func.name}")
            if not here:
                continue
            logged = _logged_arguments(func)
            if not logged:
                continue
            counts = _row_counts(func)
            for junction in here:
                spec_class, _, attribute = junction.subject.rpartition(".")
                outcomes = {b.outcome for b in junction.branches}
                for at_call, message in logged:
                    template = values.rendered_text(message)
                    named = values.decision_tags(template) if template else set()
                    names_value = _names_the_value(
                        message, func, owner, attributor, spec_class, attribute,
                        outcomes, settle,
                    )
                    if _reports_rows(message, counts):
                        reports = REPORTS_ROWS_CHANGED
                    elif named or names_value:
                        reports = REPORTS_DECISION
                    else:
                        continue
                    if not template or not values.has_literal_text(template):
                        # A frame with no literal text is not something a
                        # production line can be matched on -- it says only
                        # that something was interpolated.
                        continue
                    emit = Emit(
                        template=template,
                        log_level=log_level(at_call),
                        log_files=files,
                        reports=reports,
                    )
                    where = path_condition(at_call)
                    for branch in junction.branches:
                        if not _about(branch, named, names_value, reports):
                            continue
                        if exclusive(branch.path_condition, where):
                            continue
                        if any(e.template == template for e in branch.emits):
                            continue
                        branch.emits.append(emit)
                        attached += 1
    return attached
