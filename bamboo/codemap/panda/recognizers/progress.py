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
    attach_parents,
    enclosing_class,
    enclosing_function,
    literal_values,
    path_condition,
)

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


def spec_attributes(modules: list[SourceModule]) -> dict[str, set[str]]:
    """Return ``{spec_class: {declared attribute, ...}}`` from ``_attributes``.

    The declaration bounds attribution: a write can only belong to a class that
    says it has the attribute.  Without that bound, overlap alone hands every
    status-like write to whichever class declares the largest vocabulary --
    ``JobSpec.jobStatus`` lands under ``JediTaskSpec`` because task and job
    statuses share words like "running" and "failed".
    """
    declarations: dict[str, set[str]] = {}
    for module in modules:
        for cls in (n for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)):
            for stmt in cls.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                names = {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                if not names & {"attributes", "_attributes"}:
                    continue
                if isinstance(stmt.value, (ast.Tuple, ast.List)):
                    declarations.setdefault(cls.name, set()).update(
                        e.value
                        for e in stmt.value.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    return declarations


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat], list[DiagnosticTemplate]]:
    """Extract subjects, junctions for attribute writes, and coverage.

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
    attributor = SpecAttributor(declarations, class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
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

    for module in modules:
        attach_parents(module.tree)
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
                func=func,
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

            template = values.diagnostic_template(value)
            if template:
                diagnostics.append(
                    DiagnosticTemplate(
                        map_id=map_id,
                        derived_from=derived_from,
                        template=template,
                        field=subject,
                        form="attribute",
                        anchor=Anchor(
                            package=module.package,
                            file=module.rel_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            blob_sha=module.blob_sha,
                        ),
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
                    anchor=Anchor(
                        package=module.package,
                        file=module.rel_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        blob_sha=module.blob_sha,
                    ),
                )
                junctions[name] = junction

            # The write's own guards, shared by every outcome its right-hand
            # side can produce: reaching a write is necessary for any of them.
            dominating = path_condition(node)
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
    return subjects, list(junctions.values()), coverage, diagnostics


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
