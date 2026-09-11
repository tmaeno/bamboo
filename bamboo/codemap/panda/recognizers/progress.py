"""Progress recognizer -- where the code settles a subject's value.

This slice produces the junctions the reasoning actually walks backwards from.

**Every write is recorded; only the outcome may be unresolved.**  The simplest
form, ``spec.status = "tobroken"``, states its outcome outright, and for a long
time it was the only form this slice read.  That was a modelling error rather
than a staged rollout: it left the map with two states for an attribute write,
literal or invisible, when the model has a third that the SQL slice was already
using -- the writer is known and the value is only settled at run time.  So a
right-hand side this slice cannot resolve produces a tier-2 branch reading
``runtime(<expression>)``, never an absence.

The distinction matters because pruning is *elimination*.  A candidate set with
a writer missing does not yield "unknown", it yields a confident wrong answer,
and the writers that were missing included the ones that carry a task out of
``pending`` -- ``taskSpec.status = taskSpec.oldStatus`` -- which is the very
edge the flagship symptom's backward walk needs.

Two further decisions worth stating.

**Which spec class a write belongs to is decided per write site, and the basis
is recorded.**  ``taskSpec.status = ...`` names an attribute eight classes
declare, so the class has to come from somewhere else; that resolution lives in
``bamboo.codemap.panda.attribution`` because selection needs the same answer.
A write whose class cannot be settled is still emitted, under the placeholder
subject: the writer is known even when the subject is not, and that is enough
for the steps that read observed values.

**A path condition records what decides, not merely that something decided.**
An ``else`` contributes the negation of its ``if``, and a condition written as
a bare local name carries the expression that produced it -- otherwise a
branch guarded by ``if not allowed:`` reads as "some condition held", which
cannot be checked against anything.
"""

from __future__ import annotations

import ast
from typing import Iterator, NamedTuple, Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    DiagnosticTemplate,
    EnumerationWrite,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda import values
from bamboo.codemap.panda.attribution import (
    NOT_A_SPEC,
    UNRESOLVED_CLASS,
    SpecAttributor,
    class_bases,
)
from bamboo.codemap.panda.pathcond import (
    assigned_expressions,
    attach_parents,
    enclosing_class,
    enclosing_function,
    exclusive,
    functions_with_owner,
    literal_values,
    path_condition,
)
from bamboo.codemap.panda.recognizers.selection import log_level

SLICE_NAME = "progress"

# Vocabulary-declaring methods return a list or dict of literals and take only
# ``cls``/``self``; a returned collection of >= 2 strings is the shape.
_MIN_VOCABULARY_SIZE = 2


# --------------------------------------------------------------------------- #
# declared vocabularies
# --------------------------------------------------------------------------- #


def _literal_strings(node: ast.AST) -> list[str]:
    """Collect string literals from a list/tuple/set/dict literal expression."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [
            e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    if isinstance(node, ast.Dict):
        out: list[str] = []
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out.append(key.value)
            out.extend(_literal_strings(value))
        return out
    if isinstance(node, ast.BinOp):
        # ``["done"] + cls.statusToRetry()`` -- only the literal half is
        # recoverable here, and that is enough: the other half is declared by
        # the method it calls, which is collected in its own right.
        return _literal_strings(node.left) + _literal_strings(node.right)
    return []


def _described_attribute(method: str, attributes: set[str]) -> Optional[str]:
    """Return the attribute a vocabulary method describes, from its name.

    ``statusToReassign`` and ``statusForJobGenerator`` both declare values for
    ``status`` and say so in the name.  Reading it is what keeps a vocabulary
    attached to the field it belongs to: held per class instead, ``status``\'s
    value set would be compared against every other attribute of that class,
    and a free-form field like ``JediDatasetSpec.attributes`` would be judged
    against statuses it was never meant to hold.
    """
    for end in range(len(method), 0, -1):
        candidate = method[:end]
        if candidate in attributes:
            return candidate
    return None


def declared_vocabularies(
    modules: list[SourceModule], declarations: dict[str, set[str]]
) -> dict[tuple[str, str], set[str]]:
    """Return ``{(spec_class, attribute): {declared value, ...}}``.

    The strongest oracle in the codebase: a method returning a literal list of
    statuses states the vocabulary outright, so extracted outcomes can be
    checked against it without touching production data.  Keyed by attribute
    rather than by class, because a class declares vocabularies for some of its
    fields and not others.
    """
    vocabularies: dict[tuple[str, str], set[str]] = {}
    for module in modules:
        for cls in (n for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)):
            attributes = declarations.get(cls.name, set())
            if not attributes:
                continue
            for func in cls.body:
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                attribute = _described_attribute(func.name, attributes)
                if attribute is None:
                    continue
                for stmt in ast.walk(func):
                    if not isinstance(stmt, ast.Return) or stmt.value is None:
                        continue
                    literals = _literal_strings(stmt.value)
                    if len(literals) >= _MIN_VOCABULARY_SIZE:
                        vocabularies.setdefault((cls.name, attribute), set()).update(literals)
    return vocabularies


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def _attribute_writes(
    tree: ast.Module,
) -> Iterator[tuple[ast.Attribute, ast.expr, ast.Assign]]:
    """Yield ``(target, right-hand side, node)`` for every ``x.attr = <value>``.

    The whole target is yielded, not just its name: attributing the write needs
    the object expression, which is where the class comes from.  The right-hand
    side is yielded unresolved -- what can be made of it is :func:`_resolve`'s
    question, and it needs the enclosing function this does not have.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                yield target, node.value, node


# --------------------------------------------------------------------------- #
# the tag a branch names for itself, and the setter that persists it
# --------------------------------------------------------------------------- #


class _Setter(NamedTuple):
    """A method that writes one declared column from one of its parameters."""

    spec_class: str
    attribute: str
    parameter: str
    position: int


def spec_setters(
    modules: list[SourceModule], declarations: dict[str, set[str]]
) -> dict[str, _Setter]:
    """Methods whose call site writes a declared column, by method name.

    ``taskSpec.setErrDiag(errMsg)`` is how ``errorDialog`` is written 127 times
    in the corpus, and the attribute slice never saw one of them: a call is not
    an assignment.  That field is where an investigation starts -- it is in the
    record, so reading it costs one API call and no log window -- so the index
    that says *who wrote this message* has to cover the call form.

    Narrow on purpose, and the corpus drew both edges.

    **One column and one parameter.**  ``Share.__init__`` writes twelve and
    ``convertFromJobFileSpec`` ten; matching a call's arguments to those needs
    positional binding for a reading that buys nothing, since neither assembles
    a message.

    **One defining class.**  ``setDdmBackEnd`` writes ``JediTaskSpec.splitRule``
    in one class and ``JobSpec.specialHandling`` in another, so the name alone
    cannot say which field the text landed in -- the same restriction the
    trigger slice and ``callers_of`` put on every name-based hop, for the same
    reason.
    """
    found: dict[str, Optional[_Setter]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if owner not in declarations:
                continue
            parameters = [argument.arg for argument in func.args.args[1:]]
            if not parameters:
                continue
            written: dict[str, set[str]] = {}
            for node in ast.walk(func):
                if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                    continue
                target = node.targets[0]
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in declarations[owner]
                ):
                    continue
                flowing = {
                    name.id for name in ast.walk(node.value) if isinstance(name, ast.Name)
                } & set(parameters)
                if flowing:
                    written.setdefault(target.attr, set()).update(flowing)
            if len(written) != 1:
                continue
            attribute, flowing = next(iter(written.items()))
            if len(flowing) != 1:
                continue
            parameter = next(iter(flowing))
            setter = _Setter(owner, attribute, parameter, parameters.index(parameter))
            if found.setdefault(func.name, setter) != setter:
                # Two classes, two fields, one name: nothing here can say which.
                found[func.name] = None
    return {name: setter for name, setter in found.items() if setter is not None}


def _recorded_argument(call: ast.Call, setters: dict[str, _Setter]) -> Optional[ast.expr]:
    """The message *call* records, whether it logs it or persists it.

    Both spellings sit in the block that writes, and they are the same text:
    ``tmpLog.info(errMsg)`` is what a grep can find and
    ``taskSpec.setErrDiag(errMsg)`` is what the record keeps.
    """
    if not isinstance(call.func, ast.Attribute):
        return None
    setter = setters.get(call.func.attr)
    if setter is not None:
        for keyword in call.keywords:
            if keyword.arg == setter.parameter:
                return keyword.value
        if len(call.args) > setter.position:
            return call.args[setter.position]
        return None
    if call.args and log_level(call.args[0]) is not None:
        return call.args[0]
    return None


def _recorded_in(
    statement: ast.stmt, setters: dict[str, _Setter], settling: Optional[str]
) -> Iterator[Optional[ast.expr]]:
    """Every message *statement* records, by any of the three spellings.

    Logged, handed to a setter, or written straight into a message column --
    ``PostProcessorBase.doPreCheck`` uses the third, and it is the one task in
    the production sample whose record says why it was exhausted in the words
    of the code rather than of a retry refusal.

    The column being settled is excluded from the third.  ``taskSpec.status =
    "exhausted"`` is a string assigned to a declared field like any other, and
    reading it as this branch's message would make the value its own
    description -- a frame that matches any message with the word in it.
    """
    for node in ast.walk(statement):
        if isinstance(node, ast.Call):
            yield _recorded_argument(node, setters)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr != settling
            and isinstance(node.value, (ast.Constant, ast.JoinedStr, ast.BinOp))
        ):
            yield node.value


def _one_hop(argument: ast.expr, assignments: dict[str, list[ast.expr]]) -> list[ast.expr]:
    """Every expression *argument* can be, following a local back one step.

    All of them, because the index this feeds asks which texts can land in a
    field.  The reading that asks which text belongs to *one* write needs the
    definitions that reach it instead -- see :func:`_reaching`.
    """
    if isinstance(argument, ast.Name):
        return assignments.get(argument.id, [])
    return [argument]


def _reaching(
    name: str, write: ast.stmt, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[ast.expr]:
    """The definitions of *name* that reach *write*.

    Following every definition instead is what the corpus punished: the six
    arms of ``setScoutJobData_JEDI`` all build their message in a local called
    ``errMsg``, so taking the whole function's definitions gave every arm all
    eight tags -- a branch table that names six reasons and cannot tell them
    apart is worse than one that names none.

    A later assignment shadows an earlier one **unless the two are arms of the
    same decision**, and an augmented one appends rather than replaces.  That
    second clause is not a refinement: ``reason=low_efficiency`` is assigned in
    the ``else`` of an IO-intensity check whose ``if`` assigns an untagged
    message, and dropping one arm because the other is written later would be
    picking by source order between two things that can both hold.
    """
    definitions = sorted(
        (
            node
            for node in ast.walk(func)
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            )
            or (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            )
        ),
        key=lambda node: node.lineno,
    )
    settled = path_condition(write)
    current: list[ast.stmt] = []
    for definition in definitions:
        if definition.lineno > write.lineno:
            break
        if isinstance(definition, ast.AugAssign):
            current.append(definition)
            continue
        conditions = path_condition(definition)
        if exclusive(conditions, settled):
            # An arm the write's own conditions rule out never ran.  Without
            # this, ``retryTask_JEDI`` -- five refusals in one ``elif`` chain,
            # each assigning the same local -- gave its fifth branch all five
            # messages, since the arms are mutually exclusive with each other
            # and that is exactly what the clause below preserves.
            continue
        current = [
            held for held in current if exclusive(path_condition(held), conditions)
        ] + [definition]
    return [definition.value for definition in current]


def _block_containing(node: ast.AST) -> list[ast.stmt]:
    """The statement list *node* is a member of."""
    parent = getattr(node, "parent", None)
    if parent is None:
        return []
    for _field, value in ast.iter_fields(parent):
        if isinstance(value, list) and any(statement is node for statement in value):
            return value
    return []


def recorded_signature(
    node: ast.stmt,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    setters: dict[str, _Setter],
    attribute: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """``(tags, messages)`` the block around *node* records about its decision.

    **The block, not the path condition.**  ``setScoutJobData_JEDI`` reaches
    ``exhausted`` from six arms, and they are not ``if``/``elif`` siblings --
    each is its own ``if taskSpec.status != "exhausted":`` -- so nothing in the
    conditions contradicts anything, ``exclusive`` separates none of them, and
    attaching by non-exclusivity puts all nine tags on all six branches.  The
    block that records the line names one reason.

    **One hop through the local**, because ``reason=low_efficiency`` is assigned
    in the ``else`` arm of an unrelated check and only the local reaches the
    block that writes.

    **Both halves, because production writes the untagged one.**  The tag is the
    contract and survives rewording, but of thirty tasks found in ``exhausted``
    with a message on the record, *none* carried a tag: they are the retry
    refusals and the goal check, which write prose.  So the frame comes back
    alongside, and matching it is how those thirty name an arm -- weaker
    evidence, and the only evidence there is.

    **Refused when the block writes this same attribute twice.**  Which write
    the message is about is then unanswerable.  Two writes to *different*
    attributes are not the same problem: the block ran, so both happened, and a
    message naming it names both.
    """
    block = _block_containing(node)
    if not block:
        return [], []
    if attribute is not None:
        settling = sum(
            1
            for statement in block
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == attribute
                for target in statement.targets
            )
        )
        if settling > 1:
            return [], []
    tags: set[str] = set()
    messages: list[str] = []
    for statement in block:
        for argument in _recorded_in(statement, setters, attribute):
            if argument is None:
                continue
            held = (
                _reaching(argument.id, node, func)
                if isinstance(argument, ast.Name)
                else [argument]
            )
            for expression in held:
                text = values.rendered_text(expression)
                if not text:
                    continue
                tags.update(values.decision_tags(text))
                if values.has_literal_text(text) and text not in messages:
                    messages.append(text)
    return sorted(tags), messages


def _persisted_templates(
    module: SourceModule,
    setters: dict[str, _Setter],
    map_id: str,
    derived_from: str,
) -> list[DiagnosticTemplate]:
    """Index rows for the text a setter call persists into a declared column.

    Anchored where the text is *assembled*, not where the setter is called:
    the index answers "this message was seen, who wrote it", and the frame is
    written in the block that decided, which is the place worth reading.  One
    row per piece for the same reason :func:`assigned_expressions` keeps them
    apart -- an ``errMsg +=`` under a condition may not have run.
    """
    found: list[DiagnosticTemplate] = []
    seen: set[tuple[str, str, int]] = set()
    for func, _owner in functions_with_owner(module.tree):
        assignments = assigned_expressions(func)
        for call in ast.walk(func):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
                continue
            setter = setters.get(call.func.attr)
            if setter is None:
                continue
            argument = _recorded_argument(call, setters)
            if argument is None:
                continue
            field = SubjectNode.make_name(setter.spec_class, setter.attribute)
            for expression in _one_hop(argument, assignments):
                template = values.diagnostic_template(expression)
                if template is None:
                    continue
                key = (template, field, expression.lineno)
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    DiagnosticTemplate(
                        map_id=map_id,
                        derived_from=derived_from,
                        template=template,
                        field=field,
                        form="alias",
                        anchor=Anchor(
                            package=module.package,
                            file=module.rel_path,
                            line_start=expression.lineno,
                            line_end=expression.end_lineno,
                            blob_sha=module.blob_sha,
                        ),
                    )
                )
    return found


def unclaimed_tags(
    modules: list[SourceModule], junctions: list[JunctionNode]
) -> list[tuple[str, list[str]]]:
    """Tags named in a function that settles subjects, that no branch claims.

    The second reading of the same fact, and the only one that can report a
    message whose action never happens.  The corpus already has one: a block
    logs ``action=set_exhausted reason=scout_memory_leak`` with the write on
    the next line commented out, so production can carry a tag saying the task
    was exhausted by a path that no longer exhausts it.

    Restricted to functions that own a junction.  Half the corpus names an
    action for something this slice does not model -- a priority boost, a share
    reassignment -- and reporting those as unclaimed would be reporting that
    the map is a map of subjects.
    """
    claimed: dict[str, set[str]] = {}
    for junction in junctions:
        for branch in junction.branches:
            claimed.setdefault(junction.owner, set()).update(branch.tags)
    found: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{func.name}"
            if owner not in claimed:
                continue
            # A literal piece of an f-string renders as itself and the whole
            # renders as the frame, so counting both reports every tag twice.
            inside = {
                id(part)
                for node in ast.walk(func)
                if isinstance(node, ast.JoinedStr)
                for part in node.values
            }
            for node in ast.walk(func):
                if not isinstance(node, (ast.Constant, ast.JoinedStr)) or id(node) in inside:
                    continue
                text = values.rendered_text(node)
                if not text:
                    continue
                left = tuple(sorted(values.decision_tags(text) - claimed[owner]))
                key = (f"{module.rel_path}:{node.lineno}", left)
                if left and key not in seen:
                    seen.add(key)
                    found.append((key[0], list(left)))
    return found


# --------------------------------------------------------------------------- #
# what the right-hand side settles
# --------------------------------------------------------------------------- #


class Resolved(NamedTuple):
    """One outcome a write's right-hand side can produce.

    ``conditions`` are guards *beyond* the write's own path condition -- what
    left a local holding this value -- and are kept separate so the caller can
    drop the ones it already has.  A tuple, because a shared mutable default on
    a NamedTuple is one edit away from a bug.
    """

    outcome: str
    tier: int
    conditions: tuple[str, ...] = ()


def _runtime(value: ast.expr) -> str:
    """Render an outcome that is only settled when the code runs.

    Spelled as a call so nothing mistakes it for a value the code writes, and
    spelled the same way the SQL slice spells it -- the two slices see the same
    shapes and one map should not describe them two ways.
    """
    try:
        return f"runtime({ast.unparse(value)})"
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "runtime(?)"


def _defers_to_return_alias(value: ast.expr) -> bool:
    """Whether the return-alias slice owns this right-hand side.

    ``taskSpec.status = self.getFinalTaskStatus(...)`` is resolved one hop up,
    into a branch per value the helper can return.  A second reading of the
    same write here would put "the value is decided at run time" beside the
    thirteen branches that say what it is: a weaker claim contradicting a
    stronger one at the same junction, on the very junction that gate found
    ``aborted`` missing from.

    Safe only because that slice records the calls it *cannot* follow as
    run-time outcomes too, so staying out of this shape drops no writer.
    """
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "self"
    )


def _resolve(
    value: ast.expr,
    *,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    dominating: list[str],
    attributor: SpecAttributor,
    owner_class: Optional[str],
    spec_names: set[str],
    settle,
) -> list[Resolved]:
    """Return every outcome *value* can settle to, with the tier of each.

    A ladder, strongest rung first, and it never falls off the bottom: the last
    rung says "settled at run time" rather than declining to record the write.

    The rung that needs stating is the one for a bare name.  A local is resolved
    by reaching definitions and, failing that, becomes ``runtime(<name>)`` --
    **not** ``passthrough(<name>)``.  ``passthrough(X)`` claims the value lives
    in X and the backward walk continues there, which is only true when X is a
    qualified field name: a spec attribute, or a table column.  A local variable
    is a step in a computation, not a place a value lives, so a passthrough onto
    one is a type error in the map's own vocabulary -- an edge whose far end
    cannot exist.  Nothing catches it either: the reference gate deliberately
    excuses a passthrough that lands off the promoted set, because a genuine
    provenance terminal looks exactly like that.

    *settle* is the corpus-level value resolver, which is what lets a subscript
    of a declared mapping come out as the values it can hold.
    """
    if isinstance(value, ast.Constant):
        if isinstance(value.value, str):
            return [Resolved(value.value, 1)]
        # ``taskSpec.oldStatus = None`` clears the field, which decides it as
        # much as any word does; the same goes for a count or a flag.  Spelled
        # as the source spells it, which is also how a log line interpolating
        # the field would read.
        return [Resolved(ast.unparse(value), 1)]

    if isinstance(value, ast.Name) and func is not None:
        resolved = literal_values(func, value.id, settle)
        if resolved:
            return [
                Resolved(
                    literal,
                    1,
                    tuple(test for test in conditions if test not in dominating),
                )
                for literal, conditions, _line in resolved
            ]

    settled = settle(value, func)
    if settled:
        # ``commandStatusMap()[commandStr]["done"]`` -- the key is only known at
        # run time, the values it can select are not.
        return [Resolved(outcome, 1) for outcome in settled]

    if isinstance(value, ast.Attribute) and value.attr in spec_names:
        source, _basis = attributor.attribute_write(
            value, func=func, enclosing_class=owner_class
        )
        if source is not None:
            outcome = f"passthrough({SubjectNode.make_name(source, value.attr)})"
            return [Resolved(outcome, 2)]

    # Assembled text lands here too, and the rendered expression is its own
    # search key: ``runtime(f'no files for {name}')`` carries the literal frame
    # a production diagnostic can be matched against.  It is deliberately not
    # copied into ``emits``, which means "the branch logs this" -- true of a
    # diagnostic, false of a computed ``lfn``, and nothing distinguishes the two
    # from structure alone (measured: the readings that look like they would --
    # the template reading as prose, the field never being compared -- misclassify
    # ``errorDialog`` and ``datasetName`` in opposite directions).
    return [Resolved(_runtime(value), 2)]


#: Class-level names a spec uses to say which database columns it holds.
#: Read together because PanDA uses all three, and which one a class picks is a
#: matter of when it was written rather than of what it means.
SPEC_DECLARATION_NAMES = frozenset({"attributes", "_attributes", "attributes_with_types"})


def _declared_names(value: ast.expr) -> set[str]:
    """The column names one declaration states, whichever form it takes.

    Two forms carry names, and they are read the same way because they say the
    same thing:

    * ``attributes = ("jediTaskID", "status", ...)`` -- a tuple of strings.
    * ``attributes_with_types = (AttributeWithType("status", str), ...)`` --
      the newer form, which adds the column's type.

    A class using the second one also assigns ``attributes``, but derives it
    (``tuple([a.attribute for a in attributes_with_types])``), so reading only
    literals takes nothing from it.  Nothing about that is visible at the
    assignment: the name matches, the statement is there, and the result is
    empty.  ``spec-declarations-are-read`` exists because of it.
    """
    if isinstance(value, (ast.Tuple, ast.List)):
        names = {
            element.value
            for element in value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        if names:
            return names
        # A tuple of AttributeWithType(...) calls: the column name is the first
        # argument, and the type beside it is more than the older form states.
        return {
            call.args[0].value
            for call in value.elts
            if isinstance(call, ast.Call)
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        }
    return set()


def spec_declarations(modules: list[SourceModule]) -> list[tuple[str, str, set[str]]]:
    """Every class-level column declaration in the corpus: ``(class, file, names)``.

    Kept separate from :func:`spec_attributes` so that a declaration yielding
    *nothing* is still on the list.  That is the whole point -- a form the
    reader does not understand is indistinguishable from an absent declaration
    once the names have been merged into a dict.

    An annotated declaration counts, and the promise above is why it has to.
    Accepting only ``ast.Assign`` meant ``_attributes: tuple[str, ...] = (...)``
    was not misread but never looked at, so the class contributed no entry and
    the gate had nothing to count -- a reader able to lose every column of a
    class while the gate passes.  ``pandacommon`` already writes
    ``attributes: tuple[str, ...] = ()``, so the form is arriving rather than
    hypothetical.

    **A declaration assigns the list.**  That is the discriminator, and it is a
    reading rather than a heuristic: ``JediDatasetSpec`` has a column *called*
    ``attributes``, and the type block near the top of the class states its
    type (``attributes: str | None``) without assigning anything.  Requiring a
    value keeps that statement out, whatever the aggregation upstream happens
    to do with an empty result.
    """
    found: list[tuple[str, str, set[str]]] = []
    for module in modules:
        for cls in (n for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)):
            for stmt in cls.body:
                if isinstance(stmt, ast.Assign):
                    targets: list[ast.expr] = list(stmt.targets)
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    targets = [stmt.target]
                else:
                    continue
                names = {t.id for t in targets if isinstance(t, ast.Name)}
                if not names & SPEC_DECLARATION_NAMES:
                    continue
                found.append((cls.name, module.rel_path, _declared_names(stmt.value)))
    return found


def spec_attributes(modules: list[SourceModule]) -> dict[str, set[str]]:
    """Return ``{spec_class: {declared attribute, ...}}``.

    The declaration bounds attribution: a write can only belong to a class that
    says it has the attribute.  Without that bound, overlap alone hands every
    status-like write to whichever class declares the largest vocabulary --
    ``JobSpec.jobStatus`` lands under ``JediTaskSpec`` because task and job
    statuses share words like "running" and "failed".
    """
    declarations: dict[str, set[str]] = {}
    for cls_name, _file, names in spec_declarations(modules):
        if names:
            declarations.setdefault(cls_name, set()).update(names)
    return declarations


def _named_constant(value: ast.expr) -> Optional[str]:
    """The bare name of the module constant a right-hand side reads, if it is one.

    ``ErrorCode.EC_Kill`` and ``pandaserver.taskbuffer.ErrorCode.EC_Transfer``
    both give ``EC_Kill``-shaped answers, and the qualifier is deliberately
    dropped: the constant's own name is what the value-enum index knows it by,
    and matching on the name alone means no import has to be resolved.  Safe
    because the names are effectively unique -- 44 ``EC_`` constants over 43
    distinct names, and the one collision is between two modules that declare
    no error codes at all.
    """
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    enumerations: Optional[dict[str, str]] = None,
) -> tuple[
    list[SubjectNode],
    list[JunctionNode],
    list[CoverageStat],
    list[DiagnosticTemplate],
    list[EnumerationWrite],
]:
    """Extract subjects, junctions for attribute writes, and coverage.

    *enumerations* maps a constant's bare name to the namespace the value-enum
    index keys it under, and is what turns ``jobSpec.taskBufferErrorCode =
    ErrorCode.EC_Kill`` into a binding between a field and an enumeration.  The
    slice cannot derive it -- the constants belong to the (c) slice -- so it is
    handed in, the way the attributor is.  Omitted, no bindings are recorded and
    everything else is unchanged.

    Coverage measures *attribution*, not resolution: candidates are the writes
    that could belong to a spec, explained are those whose spec class could be
    settled.  It kept that meaning when the slice widened past literals, so the
    denominator grew from 278 to the real number of writes -- which is the point
    of widening, and makes the figure incomparable with the earlier one.  A write
    whose class is unresolved is still emitted, under the placeholder subject:
    that is a gap in attribution, and dropping it would lose the fact that the
    attribute is written here at all.

    The diagnostic index comes back alongside rather than as part of the
    junctions, because it survives promotion and they do not: a message field is
    not a subject, and the writes that assemble its text are still the answer to
    "who wrote this line".
    """
    declarations = spec_attributes(modules)
    vocabularies = declared_vocabularies(modules, declarations)
    setters = spec_setters(modules, declarations)
    attributor = SpecAttributor(declarations, class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    attributor.learn_return_types(modules)
    settle = values.resolver(values.declared_mappings(modules))

    # The subject universe is what the spec classes declare.  Without this
    # bound every ``x.attr = "literal"`` in the corpus becomes a junction --
    # ``self.plugin_flavor``, ``self.comp_name``, message-processor bookkeeping
    # -- none of which any spec declares, none of which can ever be attributed,
    # and all of which would sit in the map as permanently unresolved noise.
    spec_attribute_names = set().union(*declarations.values()) if declarations else set()

    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()
    diagnostics: list[DiagnosticTemplate] = []
    bindings: list[EnumerationWrite] = []
    enumerations = enumerations or {}

    for module in modules:
        attach_parents(module.tree)
        diagnostics.extend(_persisted_templates(module, setters, map_id, derived_from))
        candidates = 0
        explained = 0
        for target, value, node in _attribute_writes(module.tree):
            if target.attr not in spec_attribute_names:
                continue
            if _defers_to_return_alias(value):
                continue
            func = enclosing_function(node)
            owner_class = enclosing_class(node)
            spec_class, basis = attributor.attribute_write(
                target,
                # Module scope is a scope.  ``SiteMapper`` builds its default
                # site at import time -- ``DEFAULT_SITE = SiteSpec()`` and then
                # eight writes to it -- and passing ``None`` here meant the
                # constructor one line above was never looked for, so writes
                # the code types outright came out unresolved.
                func=func if func is not None else module.tree,
                enclosing_class=owner_class,
            )
            if basis == NOT_A_SPEC:
                # Not a candidate at all, so it is not counted as one: a
                # WatchDog's own ``vo`` field would otherwise sit in the
                # denominator forever as coverage the slice can never reach.
                continue
            candidates += 1
            if spec_class is not None:
                explained += 1
                attributed.add((spec_class, target.attr))

            qualname = func.name if func is not None else "<module>"
            owner = f"{module.rel_path}::{qualname}"
            subject = SubjectNode.make_name(spec_class or UNRESOLVED_CLASS, target.attr)
            name = JunctionNode.make_name(map_id, subject, owner)

            anchor = Anchor(
                package=module.package,
                file=module.rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno,
                blob_sha=module.blob_sha,
            )

            template = values.diagnostic_template(value)
            if template:
                diagnostics.append(
                    DiagnosticTemplate(
                        map_id=map_id,
                        derived_from=derived_from,
                        template=template,
                        field=subject,
                        form="attribute",
                        anchor=anchor,
                    )
                )

            constant = _named_constant(value)
            if constant in enumerations:
                bindings.append(
                    EnumerationWrite(
                        map_id=map_id,
                        derived_from=derived_from,
                        field=subject,
                        constant=constant,
                        namespace=enumerations[constant],
                        anchor=anchor,
                    )
                )

            structural = attributor.structural_class(target, func, owner_class)
            junction = junctions.get(name)
            if junction is None:
                junction = JunctionNode(
                    map_id=map_id,
                    derived_from=derived_from,
                    name=name,
                    subject=subject,
                    owner=owner,
                    attribution=basis,
                    structural_subject=(
                        SubjectNode.make_name(structural, target.attr) if structural else None
                    ),
                    anchor=anchor,
                )
                junctions[name] = junction

            # The write's own guards, shared by every outcome its right-hand
            # side can produce: reaching a write is necessary for any of them.
            dominating = path_condition(node)
            tags, messages = (
                recorded_signature(node, func, setters, target.attr)
                if func is not None
                else ([], [])
            )
            for resolved in _resolve(
                value,
                func=func,
                dominating=dominating,
                attributor=attributor,
                owner_class=owner_class,
                spec_names=spec_attribute_names,
                settle=settle,
            ):
                junction.branches.append(
                    Branch(
                        outcome=resolved.outcome,
                        path_condition=dominating + list(resolved.conditions),
                        order=len(junction.branches),
                        tier=resolved.tier,
                        tags=tags,
                        messages=messages,
                        line=node.lineno,
                    )
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

    # Subjects are emitted alongside so a gate can check an outcome against the
    # vocabulary without being handed the source again -- the fragment carries
    # both halves of the comparison.
    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=_attribution_evidence(spec_class, attribute, declarations, vocabularies),
            vocabulary=sorted(vocabularies.get((spec_class, attribute), set())),
        )
        for spec_class, attribute in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage, diagnostics, bindings


def _attribution_evidence(
    spec_class: str,
    attribute: str,
    declarations: dict[str, set[str]],
    vocabularies: dict[tuple[str, str], set[str]],
) -> list[str]:
    """Record which evidence attributed the write, so the claim can be audited."""
    evidence = ["declared-attribute"]
    if sum(1 for attrs in declarations.values() if attribute in attrs) > 1:
        # Several classes declare the name, so something else had to decide;
        # which one is recorded per junction in ``JunctionNode.attribution``.
        evidence.append("shared-attribute-name")
    if vocabularies.get((spec_class, attribute)):
        evidence.append("declared-vocabulary")
    return evidence


def unattributed_outcomes(
    modules: list[SourceModule],
) -> dict[str, set[str]]:
    """Return ``{attribute: outcomes}`` that no declared vocabulary covers.

    Either the vocabulary is incomplete or the extraction is wrong; both are
    worth looking at, and neither is served by quietly assigning the write to
    the nearest-looking class.

    Deliberately literal-only, unlike the extraction: this compares outcomes
    against a declared word list, and a value settled at run time cannot be
    compared against anything.  Widening it would report every
    ``runtime(...)`` as a word no list mentions, which is true and useless.
    """
    declarations = spec_attributes(modules)
    vocabularies = declared_vocabularies(modules, declarations)
    declared = set().union(*vocabularies.values()) if vocabularies else set()
    unattributed: dict[str, set[str]] = {}
    for module in modules:
        for target, value, _node in _attribute_writes(module.tree):
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            if value.value not in declared:
                unattributed.setdefault(target.attr, set()).add(value.value)
    return unattributed
