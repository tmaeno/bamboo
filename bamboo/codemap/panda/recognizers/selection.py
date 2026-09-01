"""Selection recognizer -- which step of a filter chain threw the candidates away.

Brokerage never picks a site.  It starts with every site and narrows the list
about twenty-five times, and "the job distribution is wrong" means asking which
of those steps did the cutting.  That question has no branch table behind it:
every step runs, and each removes some candidates, so this slice produces its
own node kind rather than junctions whose branches happen to be cumulative.

The plan expected this to be the fragile slice -- the ``newScanSiteList = []``
idiom appears 26, 21, 9 and 0 times across four sibling broker files, so a
recognizer built on it would work well on two of them and not at all on a
third.  Measuring first turned up two markers that are declarations rather than
habits, and between them they cover all four:

**The reason.**  ``criteria=-diskIO``, interpolated into the message logged for
each rejected site.  102 of these, and every one has a dominating ``if`` -- so
the tag and the condition come out together, which is the whole content of the
node.  The tag is also what production logs carry per site, which is what makes
it the identity.

**The step.**  ``self.add_summary_message(old, new, "diskIO check", ...)`` or
``f"{len(scanSiteList)} candidates passed scratch disk check"``.  Either way
the code names the step in a literal, and the funnel counter it feeds is what
the logs report per step.  ``AtlasProdTaskBroker``, which has no chains at all,
has seven of these; ``GenJobBroker``, which has one tag, has eight.

So the ladder is tags first, then named steps whose reason is untagged (the
exclusions there are ``continue`` statements under a guard).  What is left is
``AtlasBrokerUtils``, which has five tags and no steps -- its exclusions are a
helper the brokers call, so they are recorded under it with no funnel label.
Nothing here needs the chain matcher the plan budgeted for.
"""

from __future__ import annotations

import ast
import re
from typing import Iterator, Optional

from bamboo.codemap.models import Anchor, CoverageStat, FilterStageNode, SourceModule
from bamboo.codemap.panda import values
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    enclosing_function,
    functions_with_owner,
    literal_values,
    path_condition,
)

SLICE_NAME = "selection"

# ``criteria=-diskIO`` -- the reason, per rejected candidate.
_TAG = re.compile(r"\bcriteria=(-?[\w.]+)")
# The same reason assigned to a variable first: ``criteria = "-link_unusable"``.
# Only a signed word, so an unrelated string assigned to the same name is not
# swept up as a tag.
_TAG_LITERAL = re.compile(r"[-+][\w.]+\Z")
# ``f"{len(scanSiteList)} candidates passed scratch disk check"`` -- the step.
_FUNNEL = re.compile(r"candidates passed(?:\s+for)?\s+(.+?)\s*$")
_SUMMARY = "add_summary_message"

_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def _rendered(node: ast.AST) -> Optional[str]:
    """Render a string literal, marking interpolations as ``{}``."""
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else "{}"
            for value in node.values
        )
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _strings(tree: ast.AST) -> Iterator[tuple[ast.AST, str]]:
    """Yield each string literal once, not once per f-string piece."""
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            yield node, _rendered(node) or ""
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if isinstance(getattr(node, "parent", None), ast.JoinedStr):
                continue
            yield node, node.value


def _identifiers(expression: str) -> list[str]:
    """Return the names an already-rendered condition reads.

    Parsed back rather than walked in place: a path condition is text by the
    time it reaches here, and the annotation ``_substitute_bare_name`` adds
    makes it not quite Python.  Anything unparseable contributes nothing, which
    is right -- an input that cannot be named cannot be followed.
    """
    try:
        tree = ast.parse(expression.split("  [")[0], mode="eval")
    except SyntaxError:
        return []
    # The callee of a call is an operation, not a value: ``hasZeroShare`` and
    # ``getDatasets`` say what the guard *does*, and listing them as inputs
    # would send the backward walk after a method name.
    called = {
        node.func
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    names: list[str] = []
    for node in ast.walk(tree):
        if node in called:
            continue
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.Name):
            names.append(node.id)
    return names


def _interpolated_tag_name(node: ast.AST) -> Optional[str]:
    """The variable a ``criteria=`` value is read from, when it is one.

    ``tmpStr += f": criteria={criteria}"`` names no tag.  The reason was decided
    earlier, by whichever branch last assigned that variable, so a recognizer
    reading only the literal form sees a step that cuts candidates and has
    nothing to say about why -- which is what ``tags-are-known`` caught from the
    other side: production logs ``criteria=-link_unusable`` and the map had no
    stage for it.

    Requiring a bare name is what keeps this narrow.  The two other interpolated
    ``criteria=`` messages in the corpus render ``str(criteria)`` and are
    progress logs rather than rejections; a call is not a name, so they stay out
    without needing to be excluded by hand.
    """
    if not isinstance(node, ast.JoinedStr):
        return None
    for index, value in enumerate(node.values[:-1]):
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if not value.value.endswith("criteria="):
            continue
        following = node.values[index + 1]
        if isinstance(following, ast.FormattedValue) and isinstance(following.value, ast.Name):
            return following.value.id
    return None


def _tag_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: Optional[str]
) -> list[tuple[str, list[str], int]]:
    """The tags *name* can hold, with the conditions that leave it holding each.

    Filtered to signed words so that an unrelated string assigned to the same
    variable is not read as a rejection reason.
    """
    if not name:
        return []
    return [
        (literal, conditions, line)
        for literal, conditions, line in literal_values(func, name)
        if _TAG_LITERAL.match(literal)
    ]


def _emitting_call(node: ast.AST) -> Optional[ast.Call]:
    """Return the logging call a string is an argument of, if any."""
    parent = getattr(node, "parent", None)
    while parent is not None and not isinstance(parent, ast.stmt):
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute):
            return parent
        parent = getattr(parent, "parent", None)
    return None


def _log_level(node: ast.AST) -> Optional[str]:
    call = _emitting_call(node)
    if call is None or not isinstance(call.func, ast.Attribute):
        return None
    return call.func.attr if call.func.attr in _LOG_LEVELS else None


def helper_log_levels(modules: list[SourceModule]) -> dict[str, Optional[str]]:
    """Return the level each summary helper logs at, by defining class.

    Most rejection messages are never logged where they are written: the broker
    puts them in a ``msg_map`` and hands it to ``add_summary_message``, which
    logs them.  Recording ``None`` for those would say the map cannot see the
    level, when in fact one hop settles it -- and the level is exactly what
    decides whether the observation the map promises exists in production at
    all.  ``JobBrokerBase`` logs at ``info``; ``AtlasProdTaskBroker``'s own
    version only appends to a list, and stays ``None``.
    """
    levels: dict[str, Optional[str]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if func.name != _SUMMARY or owner is None:
                continue
            found = None
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _LOG_LEVELS
                ):
                    found = node.func.attr
                    break
            levels[owner] = found
    return levels


class _Step:
    """One named step of a chain: where the funnel counter reports a cut."""

    def __init__(
        self, line: int, label: str, level: Optional[str], funnel_line: bool = False
    ) -> None:
        self.line = line
        self.label = label
        self.level = level
        # Whether the name was read off a ``candidates passed`` line here, as
        # opposed to off an ``add_summary_message`` argument.  Only matters
        # where a step reports itself both ways; see :func:`_merged`.
        self.funnel_line = funnel_line


def _merged(steps: list[_Step]) -> list[_Step]:
    """Collapse the two ways one step reports itself.

    ``AtlasProdTaskBroker`` writes its own funnel line and then calls its own
    ``add_summary_message``, which -- unlike ``JobBrokerBase``'s, where the
    funnel line is emitted inside the helper -- only files a summary entry.
    Two statements, one cut, and nothing rejected between them, so no funnel
    count and no rejection can tell them apart.

    The corpus draws the line itself: consecutive steps are one line apart
    seven times and never again nearer than sixteen.  Six of the seven give
    both statements the same name and already read as one step.  The seventh
    calls the funnel line ``endpoint check with DISK_THRESHOLD={} TB`` and the
    summary entry ``storage endpoint check``; read as two steps, the second
    counts a cut whose every reason attached to the line above it.

    The funnel line's name wins, because it is the one production puts on a
    ``candidates passed`` line and that is where anything comparing the map
    against a log reads a step's name.  Where the helper is the one logging,
    the pair names the step alike and the choice does not arise.
    """
    kept: list[_Step] = []
    for step in steps:
        if not kept or step.line - kept[-1].line > 1:
            kept.append(step)
            continue
        previous = kept[-1]
        winner = step if step.funnel_line and not previous.funnel_line else previous
        winner.line = step.line
        kept[-1] = winner
    return kept


def _steps(
    func: ast.FunctionDef | ast.AsyncFunctionDef, helpers: dict[str, Optional[str]]
) -> list[_Step]:
    """Return the named steps of *func*'s chain, in source order."""
    found: list[_Step] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == _SUMMARY
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Constant)
            and isinstance(node.args[2].value, str)
        ):
            # Whichever class defines the helper, they agree often enough that
            # the first answer is the right one; where none does, ``None`` says
            # the level is unknown rather than absent.
            level = next((v for v in helpers.values() if v), None)
            found.append(_Step(node.lineno, node.args[2].value, level))
    for node, text in _strings(func):
        match = _FUNNEL.search(text.strip())
        if match is None:
            continue
        label = match.group(1).strip()
        if not values.has_literal_text(label):
            # ``f"{len(new_list)} candidates passed {message}"`` inside the
            # helper itself: the whole name is a hole, so the code templated
            # the step name rather than naming one and there is no step here.
            continue
        # A hole with words around it is different, and the difference is not
        # cosmetic: ``endpoint check with DISK_THRESHOLD={} TB`` is a step this
        # broker runs 1176 times in a day's logs, and requiring the name to be
        # fixed left the funnel counting cuts the map could not place.  The
        # threshold is a detail of the run; the frame is the step's name.
        found.append(_Step(node.lineno, label, _log_level(node), funnel_line=True))
    found.sort(key=lambda step: step.line)
    return _merged(found)


def _step_at(steps: list[_Step], line: int) -> Optional[_Step]:
    """Return the step a line belongs to.

    The counter is reported *after* the loop that does the cutting, so a
    rejection belongs to the first step at or after it.
    """
    for step in steps:
        if step.line >= line:
            return step
    return None


class _Stage:
    """A tag's exclusions within one chain, pooled across its sites."""

    def __init__(self, tag: str, label: str, line: int) -> None:
        self.tag = tag
        self.label = label
        self.line = line
        self.conditions: list[str] = []
        self.emits: list[str] = []
        self.level: Optional[str] = None

    def add(self, conditions: list[str], template: str, level: Optional[str]) -> None:
        for condition in conditions:
            if condition not in self.conditions:
                self.conditions.append(condition)
        if template and template not in self.emits:
            self.emits.append(template)
        # A message logged where it is written settles the level outright; the
        # helper's level, set when the stage was created, is the fallback.
        self.level = level or self.level


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[FilterStageNode], list[CoverageStat], list[str]]:
    """Extract filter stages, per-file coverage, and the steps left unexplained.

    A candidate is a tagged rejection or a named step; explained means a
    condition came out of it.  A step that cuts candidates for a reason the
    slice cannot read stays in the denominator rather than being skipped, and
    is named in the third return value: the funnel will report candidates
    disappearing there and the map has nothing to say about where they went,
    which is the one way this slice can be quietly wrong.
    """
    stages: list[FilterStageNode] = []
    coverage: list[CoverageStat] = []
    unexplained: list[str] = []
    helpers = helper_log_levels(modules)

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            steps = _steps(func, helpers)
            # Keyed by (step, tag), not by tag: ``AtlasProdJobBroker`` emits
            # ``criteria=-disk`` from both its "disk check" and its "Storage
            # check", and folding them together loses one of the two cuts.  The
            # tag alone is ambiguous within a chain; the funnel counter is what
            # tells a reader which of the two a log line came from.
            found: dict[tuple[str, str], _Stage] = {}
            for node, text in _strings(func):
                if enclosing_function(node) is not func:
                    continue
                match = _TAG.search(text)
                if match:
                    # The tag is in the message: the guards on the message are
                    # the guards on the cut, and the message is the template an
                    # investigation will look for.
                    tagged = [(match.group(1), path_condition(node), text, node.lineno)]
                else:
                    name = _interpolated_tag_name(node)
                    # One message, one stage per value the variable can hold --
                    # the branch that set it is what distinguishes the cuts, so
                    # each assignment is its own stage, anchored where the reason
                    # is decided rather than where the message is built.  No
                    # template: this fragment is the tail of a message assembled
                    # across statements, and ``": criteria={}"`` is shared by
                    # every rejection in the file, so offering it as the line to
                    # look for would confirm nothing.
                    tagged = [
                        (tag, conditions, "", line)
                        for tag, conditions, line in _tag_values(func, name)
                    ]
                step = _step_at(steps, node.lineno)
                label = step.label if step else ""
                for tag, conditions, template, line in tagged:
                    stage = found.get((label, tag))
                    if stage is None:
                        stage = _Stage(tag, label, line)
                        stage.level = step.level if step else None
                        found[(label, tag)] = stage
                    stage.add(conditions, template, _log_level(node))
            named = _untagged_steps(func, steps, found)
            # A tag repeated at several sites is one reason, so the denominator
            # counts reasons and steps rather than lines -- otherwise a stage
            # that rejects in two places reads as a coverage miss.
            labelled = {stage.label for stage in found.values() if stage.label}
            candidates += len(found) + sum(1 for step in steps if step.label not in labelled)
            explained += sum(
                1 for stage in (*found.values(), *named) if stage.conditions
            )

            owner = f"{module.rel_path}::{func.name}"
            explained_labels = {
                stage.label for stage in (*found.values(), *named) if stage.conditions
            }
            unexplained.extend(
                f"{owner} counts a cut at {step.label!r} with no readable reason"
                for step in steps
                if step.label not in explained_labels
            )
            for order, stage in enumerate(
                sorted([*found.values(), *named], key=lambda s: s.line)
            ):
                if not stage.conditions:
                    continue
                stages.append(
                    _node(map_id, derived_from, module, owner, order, stage)
                )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )
    return stages, coverage, sorted(set(unexplained))


def _appends(statements: list[ast.stmt]) -> bool:
    """Whether one arm of a branch puts the candidate on the surviving list."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        for statement in statements
        for node in ast.walk(statement)
    )


def _untagged_steps(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    steps: list[_Step],
    tagged: dict[tuple[str, str], _Stage],
) -> list[_Stage]:
    """Return stages for the named steps whose rejections carry no tag.

    ``GenJobBroker`` names eight steps and tags one of them, so without this
    seven cuts in a production broker would be invisible.  A rejection there is
    a ``continue`` under a guard -- the loop skips to the next candidate rather
    than appending it -- which is the same evidence the tag would have carried,
    minus the name.

    **Not appending is the other way out of the iteration.**  A loop can drop a
    candidate by skipping ahead or by keeping the survivors elsewhere, and the
    two are the same cut written differently::

        for tmpSiteName in scanSiteList:
            if tmpSiteName in siteSkippedTmp:
                msg_map[...] = siteSkippedTmp[tmpSiteName]      # excluded
            else:
                newScanSiteList.append(tmpSiteName)             # kept

    So an ``if`` whose one arm appends and whose other does not has stated its
    exclusion condition, positively, in the arm that does not: no negation to
    compose and no reachability to reason about.  This is the deferred cut in
    ``AtlasProdJobBroker`` -- "temporary problem check", 1111 runs of it in a
    day's logs -- and it is the whole reason the funnel could count candidates
    disappearing where the map had nothing to say.

    Its reason is carried rather than stated: ``siteSkippedTmp`` was filled by
    earlier steps, whose tags the map already holds.  It is deliberately *not*
    recorded as a passthrough -- that names a place a value lives, a spec field
    or a table column, and a local is neither -- and the condition names the
    variable anyway, which is as much as the source says.
    """
    labelled = {stage.label for stage in tagged.values() if stage.label}
    found: list[_Stage] = []
    previous = 0
    for step in steps:
        if step.label in labelled:
            previous = step.line
            continue
        stage = _Stage("", step.label, step.line)
        stage.level = step.level
        for node in ast.walk(func):
            if not isinstance(node, (ast.Continue, ast.If)):
                continue
            if not previous < node.lineno <= step.line:
                continue
            if enclosing_function(node) is not func:
                continue
            if isinstance(node, ast.Continue):
                stage.add(path_condition(node), "", None)
            elif node.orelse and _appends(node.body) != _appends(node.orelse):
                excluded = node.orelse if _appends(node.body) else node.body
                stage.add(path_condition(excluded[0]), "", None)
        previous = step.line
        if stage.conditions:
            found.append(stage)
    return found


def _node(
    map_id: str,
    derived_from: str,
    module: SourceModule,
    owner: str,
    order: int,
    stage: _Stage,
) -> FilterStageNode:
    signature = f"{stage.label}|{stage.tag}" if stage.label and stage.tag else (
        stage.tag or stage.label
    )
    inputs: list[str] = []
    for condition in stage.conditions:
        for name in _identifiers(condition):
            if name not in inputs:
                inputs.append(name)
    return FilterStageNode(
        map_id=map_id,
        derived_from=derived_from,
        name=FilterStageNode.make_name(map_id, owner, signature),
        owner=owner,
        criteria_tag=stage.tag,
        funnel_label=stage.label,
        order=order,
        conditions=stage.conditions,
        inputs=inputs,
        emits=stage.emits,
        log_level=stage.level,
        anchor=Anchor(
            package=module.package,
            file=module.rel_path,
            line_start=stage.line,
            blob_sha=module.blob_sha,
        ),
    )
