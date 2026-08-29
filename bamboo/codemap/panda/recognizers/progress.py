"""Progress recognizer -- where the code settles a subject's value.

This slice produces the junctions the reasoning actually walks backwards from.
It starts with the simplest write form, ``spec.status = "tobroken"``, because
the outcome is stated right there and nothing has to be resolved: that keeps
the first contact between the node model and real source narrow enough that a
modelling error shows up before the harder write forms depend on it.

Two decisions worth stating.

**Which attribute belongs to which spec class is settled by evidence, not by
naming.** ``taskSpec.status = ...`` looks like ``JediTaskSpec.status``, and in
PanDA it usually is, but inferring that from the variable name is a guess.
Instead the written literals are matched against the value sets the spec
classes *declare* (``statusToReassign()`` and friends): an attribute whose
outcomes are drawn from a class's declared vocabulary belongs to that class.
Outcomes that match nothing are reported rather than assigned, because that
means either the vocabulary is incomplete or the extraction is wrong, and
guessing would hide both.

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
) -> Iterator[tuple[str, str, ast.Assign]]:
    """Yield ``(attribute, literal value, node)`` for ``x.attr = "literal"``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                yield target.attr, value.value, node


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


def _owning_class(
    attribute: str,
    outcomes: set[str],
    vocabularies: dict[tuple[str, str], set[str]],
    declarations: dict[str, set[str]],
) -> Optional[str]:
    """Return the spec class a write belongs to, or ``None`` if undecidable.

    Settled only by the attribute declaration: where exactly one class says it
    has the attribute, that class owns the write.  ``jobStatus`` and
    ``proc_status`` resolve this way.

    ``status`` does not -- eight classes declare it -- and this deliberately
    leaves it unattributed rather than choosing.  Two tempting tie-breakers
    were tried and rejected:

    * *Largest vocabulary overlap.* Task and job statuses share words like
      "running" and "failed", so the class declaring the most values wins every
      tie regardless of what is being written; it filed ``fileSpec.status =
      'cached'`` under ``JediTaskSpec``.
    * *The literal being written.* This decides correctly, but it decides using
      the vocabulary, which is what the vocabulary gate then checks -- the gate
      becomes a tautology and stops being able to find anything.

    Deciding per write site needs the type of the object being written to, and
    that is not available here: of the literal ``.status`` writes, only about a
    fifth have an object variable whose single definition is a constructor
    call, the rest arriving as parameters or attributes.  Leaving them
    unattributed names the missing capability instead of hiding it behind a
    plausible-looking answer.
    """
    del outcomes, vocabularies
    candidates = [cls for cls, attrs in declarations.items() if attribute in attrs]
    return candidates[0] if len(candidates) == 1 else None


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract subjects, junctions for literal attribute writes, and coverage.

    Coverage counts literal attribute writes as candidates and those attributed
    to a spec class as explained, so a file writing statuses no class declares
    shows up as a gap rather than passing silently.
    """
    declarations = spec_attributes(modules)
    vocabularies = declared_vocabularies(modules, declarations)

    # Outcomes are pooled per attribute across the whole corpus before
    # attribution: one write site rarely shows enough of a vocabulary to
    # identify it, but the union over every site does.
    outcomes_by_attribute: dict[str, set[str]] = {}
    for module in modules:
        for attribute, literal, _node in _literal_attribute_writes(module.tree):
            outcomes_by_attribute.setdefault(attribute, set()).add(literal)

    owner_by_attribute = {
        attribute: _owning_class(attribute, outcomes, vocabularies, declarations)
        for attribute, outcomes in outcomes_by_attribute.items()
    }

    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []

    for module in modules:
        _attach_parents(module.tree)
        candidates = 0
        explained = 0
        for attribute, literal, node in _literal_attribute_writes(module.tree):
            candidates += 1
            spec_class = owner_by_attribute.get(attribute)
            if spec_class is None:
                continue
            explained += 1

            func = _enclosing_function(node)
            qualname = func.name if func is not None else "<module>"
            owner = f"{module.rel_path}::{qualname}"
            subject = f"{spec_class}.{attribute}"
            name = JunctionNode.make_name(map_id, subject, owner)

            junction = junctions.get(name)
            if junction is None:
                junction = JunctionNode(
                    map_id=map_id,
                    derived_from=derived_from,
                    name=name,
                    subject=subject,
                    owner=owner,
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
                    # has to be resolved at run time.
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
        for attribute, spec_class in sorted(owner_by_attribute.items())
        if spec_class is not None
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
        # Several classes declare the name, so the vocabulary is what decided.
        evidence.append("vocabulary-disambiguated")
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
        for attribute, literal, _node in _literal_attribute_writes(module.tree):
            if literal not in declared:
                unattributed.setdefault(attribute, set()).add(literal)
    return unattributed
