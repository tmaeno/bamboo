"""Progress recognizer -- where the code settles a subject's value.

This slice produces the junctions the reasoning actually walks backwards from.
It starts with the simplest write form, ``spec.status = "tobroken"``, because
the outcome is stated right there and nothing has to be resolved: that keeps
the first contact between the node model and real source narrow enough that a
modelling error shows up before the harder write forms depend on it.

Two decisions worth stating.

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
from typing import Iterator, Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda.attribution import (
    NOT_A_SPEC,
    UNRESOLVED_CLASS,
    SpecAttributor,
    class_bases,
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
# path conditions
# --------------------------------------------------------------------------- #


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def _ancestors(node: ast.AST) -> Iterator[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None:
        yield current
        current = getattr(current, "parent", None)


def _enclosing_function(node: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for ancestor in _ancestors(node):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ancestor
    return None


def _single_definition(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> Optional[str]:
    """Return the expression assigned to *name*, when exactly one assigns it.

    A condition written as a bare local name says nothing on its own: ``if not
    allowed:`` names no predicate.  Substituting the single expression that
    produced it recovers which check decides -- typically a call such as
    ``self._check_command_allowed(...)``, whose own branches are a deeper
    expansion than this slice attempts.

    Returns ``None`` when several statements assign the name.  That is the
    fan-out case (a flag set from many places), where one expression would
    misrepresent the branch rather than explain it.
    """
    found: list[ast.AST] = []
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
    if len(found) != 1:
        return None
    try:
        return ast.unparse(found[0])
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return None


def path_condition(node: ast.AST) -> list[str]:
    """Return the conjunction of tests dominating *node*, outermost first.

    Walks the ancestor chain rather than the call stack: an ``else`` branch
    contributes ``not <test>``, because "the condition did not hold" is as much
    a reason for the outcome as the condition holding.
    """
    conditions: list[str] = []
    func = _enclosing_function(node)
    previous = node
    for ancestor in _ancestors(node):
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


def _substitute_bare_name(
    rendered: str, test: ast.expr, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> str:
    """Annotate a test that is a bare local name with the expression behind it."""
    target = test.operand if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) else test
    if not isinstance(target, ast.Name):
        return rendered
    definition = _single_definition(func, target.id)
    if definition is None or definition == target.id:
        return rendered
    return f"{rendered}  [{target.id} := {definition}]"


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def _literal_attribute_writes(
    tree: ast.Module,
) -> Iterator[tuple[ast.Attribute, str, ast.Assign]]:
    """Yield ``(target, literal value, node)`` for ``x.attr = "literal"``.

    The whole target is yielded, not just its name: attributing the write needs
    the object expression, which is where the class comes from.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                yield target, value.value, node


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


def _enclosing_class(node: ast.AST) -> Optional[str]:
    for ancestor in _ancestors(node):
        if isinstance(ancestor, ast.ClassDef):
            return ancestor.name
    return None


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract subjects, junctions for literal attribute writes, and coverage.

    Coverage counts literal attribute writes as candidates and those whose spec
    class could be settled as explained.  Unresolved writes are still emitted as
    junctions under the placeholder subject -- they are a gap in *attribution*,
    not in extraction, and dropping them would lose the fact that the attribute
    is written here at all.
    """
    declarations = spec_attributes(modules)
    vocabularies = declared_vocabularies(modules, declarations)
    attributor = SpecAttributor(declarations, class_bases(modules))

    # The subject universe is what the spec classes declare.  Without this
    # bound every ``x.attr = "literal"`` in the corpus becomes a junction --
    # ``self.plugin_flavor``, ``self.comp_name``, message-processor bookkeeping
    # -- none of which any spec declares, none of which can ever be attributed,
    # and all of which would sit in the map as permanently unresolved noise.
    spec_attribute_names = set().union(*declarations.values()) if declarations else set()

    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()

    for module in modules:
        _attach_parents(module.tree)
        imported = attributor.imported_specs(module)
        candidates = 0
        explained = 0
        for target, literal, node in _literal_attribute_writes(module.tree):
            if target.attr not in spec_attribute_names:
                continue
            func = _enclosing_function(node)
            spec_class, basis = attributor.attribute_write(
                target,
                imported=imported,
                func=func,
                enclosing_class=_enclosing_class(node),
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

            structural = attributor.structural_class(target, func)
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
            junction.branches.append(
                Branch(
                    outcome=literal,
                    path_condition=path_condition(node),
                    order=len(junction.branches),
                    # The outcome is a literal at the write site, so nothing
                    # has to be resolved at run time.  Independent of whether
                    # the subject's class was settled.
                    tier=1,
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
    return subjects, list(junctions.values()), coverage


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
    """
    declarations = spec_attributes(modules)
    vocabularies = declared_vocabularies(modules, declarations)
    declared = set().union(*vocabularies.values()) if vocabularies else set()
    unattributed: dict[str, set[str]] = {}
    for module in modules:
        for target, literal, _node in _literal_attribute_writes(module.tree):
            if literal not in declared:
                unattributed.setdefault(target.attr, set()).add(literal)
    return unattributed
