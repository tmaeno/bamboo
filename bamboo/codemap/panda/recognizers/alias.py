"""Write-alias recognizer -- a method that settles a subject, seen from its callers.

``JediTaskSpec.setOnHold`` is four lines and one of them writes the most
contended value in the system::

    def setOnHold(self):
        if self.status in ["ready", "running", "merging", ...]:
            self.oldStatus = self.status
            self.status = "pending"

The attribute slice already records that write, and records it exactly once,
under ``JediTaskSpec.py::setOnHold``.  That is a true statement and a useless
one for an investigation: asked why a task is in ``pending``, the map would
answer "because ``setOnHold`` was called" and stop.  The fourteen callers -- ten
in ``JobGenerator``, three in ``TaskRefiner``, one in ``ContentsFeeder`` -- each
call it under a different condition, and *those* are the candidate causes.

So the call sites are junctions too.  A branch's path condition is the caller's
conjunction followed by the alias's own guard, marked with where it came from:
reaching ``pending`` requires both, and a reader who cannot tell them apart
cannot tell a caller that never ran from a guard that rejected the status.

**Only literal writes make a method an alias.**  Widened to any write to a
declared attribute, the same rule matches 36 methods and 600 call sites, led by
``__init__``, ``pack``, ``__setattr__`` and ``setErrDiag`` -- serialization and
parameter passing, which settle nothing.  Narrowed to literals it matches five
methods, and promotion keeps the one that matters.

**The same method, seen the other way round.**  A helper can settle a subject by
*returning* the value instead of writing it, and the caller assigns it::

    taskSpec.status = self.getFinalTaskStatus(taskSpec, update_error_dialog=True)

The attribute slice takes only literal right-hand sides, so a write like this
was not on the map at all -- and production put 104 tasks into ``aborted``,
which nothing in the map produced, because the only place that value is decided
is inside that helper.  ``getFinalTaskStatus`` is a chain of eight guarded
assignments to a local followed by ``return status``, so the values and their
conditions are there to be read; what it needs is one hop, the same hop the path
condition extractor already makes for a predicate that is a same-class helper's
return.  Recorded like an alias's, with the caller's conditions first and the
helper's own marked with where they came from.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.models import (
    Anchor,
    Branch,
    CoverageStat,
    JunctionNode,
    SourceModule,
    SubjectNode,
)
from bamboo.codemap.panda.attribution import CERTAIN, SpecAttributor
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    enclosing_class,
    enclosing_function,
    functions_with_owner,
    literal_values,
    path_condition,
)

SLICE_NAME = "write-alias"
# The same relationship read the other way: a helper that returns the value.
PRODUCER_SLICE = "return-alias"


class Alias:
    """A method whose body settles a subject to a literal."""

    def __init__(self, spec_class: str, method: str) -> None:
        self.spec_class = spec_class
        self.method = method
        # (attribute, literal, the conditions inside the method)
        self.writes: list[tuple[str, str, list[str]]] = []


def _literal_self_writes(
    func: ast.FunctionDef | ast.AsyncFunctionDef, declared: set[str]
) -> list[tuple[str, str, list[str]]]:
    """Return ``(attribute, literal, inner conditions)`` for ``self.attr = "lit"``."""
    found: list[tuple[str, str, list[str]]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in declared
            ):
                found.append((target.attr, value.value, path_condition(node)))
    return found


def find_aliases(
    modules: list[SourceModule], declarations: dict[str, set[str]]
) -> dict[str, list[Alias]]:
    """Return ``{method name: [alias, ...]}`` over the whole corpus.

    Keyed by method name rather than by class because that is what a call site
    offers: ``taskSpec.setOnHold()`` names the method and leaves the class to be
    worked out.  Where exactly one class defines the name, the call site needs
    no further evidence -- the same "only one declaration" argument the
    attribute slice uses, one level up.
    """
    aliases: dict[str, list[Alias]] = {}
    for module in modules:
        attach_parents(module.tree)
        for func, owner in functions_with_owner(module.tree):
            declared = declarations.get(owner or "", set())
            if not declared:
                continue
            writes = _literal_self_writes(func, declared)
            if not writes:
                continue
            alias = Alias(owner, func.name)  # type: ignore[arg-type]
            alias.writes = writes
            aliases.setdefault(func.name, []).append(alias)
    return aliases


class Producer:
    """A method whose return value settles a subject."""

    def __init__(self, defining_class: str, method: str) -> None:
        self.defining_class = defining_class
        self.method = method
        # (literal, the conditions inside the method)
        self.values: list[tuple[str, list[str]]] = []
        # Whether it can also return something this slice could not resolve.
        self.opaque = False


def _returned_values(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[tuple[str, list[str]]], bool]:
    """``([(literal, inner conditions)], whether anything is unresolved)``.

    Two shapes are read: a literal returned outright, and a local returned after
    a guarded chain assigned it, which is the shape that matters here and is
    what :func:`literal_values` answers.  Anything else -- a call, a computed
    expression, ``True`` -- is counted as unresolved so the junction can say the
    value may be something other than what was listed, rather than implying the
    list is closed.
    """
    values: list[tuple[str, list[str]]] = []
    opaque = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        if enclosing_function(node) is not func:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            values.append((node.value.value, path_condition(node)))
        elif isinstance(node.value, ast.Name):
            resolved = literal_values(func, node.value.id)
            if resolved:
                # The return's own guards as well: a chain assigning the local
                # can sit inside a branch of the function.
                returning = path_condition(node)
                values.extend(
                    (literal, returning + [c for c in conditions if c not in returning])
                    for literal, conditions, _line in resolved
                )
            else:
                opaque = True
        else:
            opaque = True
    return values, opaque


def find_producers(modules: list[SourceModule]) -> dict[str, list[Producer]]:
    """Return ``{method name: [producer, ...]}`` over the whole corpus.

    Keyed by name for the same reason the aliases are: a call site offers the
    name, and where exactly one class defines it no further evidence is needed.
    """
    producers: dict[str, list[Producer]] = {}
    for module in modules:
        attach_parents(module.tree)
        for func, owner in functions_with_owner(module.tree):
            if owner is None:
                continue
            values, opaque = _returned_values(func)
            if not values:
                continue
            producer = Producer(owner, func.name)
            producer.values = values
            producer.opaque = opaque
            producers.setdefault(func.name, []).append(producer)
    return producers


def extract_producers(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    declarations: dict[str, set[str]],
    attributor: SpecAttributor,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract a junction per ``spec.attr = self.helper(...)`` write.

    Restricted to a bare ``self`` receiver.  Widened to any call, the shape
    matches forty-two sites of which two touch a subject, and the rest are
    plugin lookups and config reads -- the same ratio that got the naming
    heuristic deleted.  Restricted, it reads the two that decide a status.

    **This slice owns the shape outright**, including the writes whose helper it
    cannot follow: those get a run-time outcome rather than nothing, so the
    attribute slice can stay out of the shape entirely instead of adding a
    second, weaker reading of the two writes resolved here.  ``explained``
    still counts only the resolved ones, so the coverage figure keeps saying how
    much of the shape was actually followed.
    """
    producers = find_producers(modules)
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "self"
            ):
                continue
            targets = [t for t in node.targets if isinstance(t, ast.Attribute)]
            if not targets:
                continue
            func = enclosing_function(node)
            owner_class = enclosing_class(node)
            for target in targets:
                spec_class = attributor.class_of(target.value, func, owner_class)
                if spec_class is None or target.attr not in declarations.get(spec_class, set()):
                    continue
                candidates += 1
                attributed.add((spec_class, target.attr))
                caller = path_condition(node)
                producer = _producer_at(producers, call.func.attr, owner_class, attributor)
                if producer is None:
                    # The helper returns something this slice cannot follow --
                    # a computed value, or another call.  The writer is still
                    # the answer to "who set this", which is what localize and
                    # prune work from, so it is recorded as a run-time outcome.
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=f"runtime({call.func.attr}())",
                        condition=caller,
                        tier=2,
                    )
                    continue
                explained += 1
                for literal, inner in producer.values:
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=literal,
                        condition=caller
                        + [f"{test}  [in {producer.method}()]" for test in inner],
                    )
                if producer.opaque:
                    # Says the list is not closed rather than letting it read as
                    # complete: the helper has a return this slice cannot follow.
                    _record(
                        junctions,
                        map_id=map_id,
                        derived_from=derived_from,
                        module=module,
                        node=call,
                        owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                        spec_class=spec_class,
                        attribute=target.attr,
                        literal=f"runtime({producer.method}())",
                        condition=caller,
                        tier=2,
                    )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=PRODUCER_SLICE,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )

    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=["return-alias"],
        )
        for spec_class, attribute in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage


def _producer_at(
    producers: dict[str, list[Producer]],
    method: str,
    owner_class: Optional[str],
    attributor: SpecAttributor,
) -> Optional[Producer]:
    """Return the producer ``self.<method>()`` resolves to.

    ``self`` names the class the call sits in, so a producer defined there or in
    one of its bases is the answer.  Where the name is defined once in the whole
    corpus that check is unnecessary, which is the common case.
    """
    candidates = producers.get(method)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if owner_class is None:
        return None
    family = attributor.family(owner_class)
    for producer in candidates:
        if producer.defining_class in family:
            return producer
    return None


def _alias_at(
    aliases: dict[str, list[Alias]],
    call: ast.Call,
    attributor: SpecAttributor,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    owner_class: Optional[str],
) -> Optional[Alias]:
    """Return the alias a call resolves to, or ``None`` when the class is open."""
    if not isinstance(call.func, ast.Attribute):
        return None
    candidates = aliases.get(call.func.attr)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    spec_class = attributor.class_of(call.func.value, func, owner_class)
    for alias in candidates:
        if alias.spec_class == spec_class:
            return alias
    return None


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    declarations: dict[str, set[str]],
    attributor: SpecAttributor,
) -> tuple[list[SubjectNode], list[JunctionNode], list[CoverageStat]]:
    """Extract a junction per call site of a write alias."""
    aliases = find_aliases(modules, declarations)
    junctions: dict[str, JunctionNode] = {}
    coverage: list[CoverageStat] = []
    attributed: set[tuple[str, str]] = set()

    for module in modules:
        attach_parents(module.tree)
        candidates = 0
        explained = 0
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in aliases:
                continue
            func = enclosing_function(node)
            candidates += 1
            alias = _alias_at(aliases, node, attributor, func, enclosing_class(node))
            if alias is None:
                continue
            explained += 1
            caller = path_condition(node)
            for attribute, literal, inner in alias.writes:
                attributed.add((alias.spec_class, attribute))
                _record(
                    junctions,
                    map_id=map_id,
                    derived_from=derived_from,
                    module=module,
                    node=node,
                    owner=f"{module.rel_path}::{func.name if func else '<module>'}",
                    spec_class=alias.spec_class,
                    attribute=attribute,
                    literal=literal,
                    # The caller's conditions decide whether the alias runs; the
                    # alias's own decide whether running it changes anything.
                    # Both are required and they fail differently, so the second
                    # set says where it lives.
                    condition=caller
                    + [f"{test}  [in {alias.method}()]" for test in inner],
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

    subjects = [
        SubjectNode(
            map_id=map_id,
            derived_from=derived_from,
            name=SubjectNode.make_name(spec_class, attribute),
            spec_class=spec_class,
            attribute=attribute,
            criteria=["write-alias"],
        )
        for spec_class, attribute in sorted(attributed)
    ]
    return subjects, list(junctions.values()), coverage


def _record(
    junctions: dict[str, JunctionNode],
    *,
    map_id: str,
    derived_from: str,
    module: SourceModule,
    node: ast.Call,
    owner: str,
    spec_class: str,
    attribute: str,
    literal: str,
    condition: list[str],
    tier: int = 1,
) -> None:
    """Add one branch for what this call site settles."""
    subject = SubjectNode.make_name(spec_class, attribute)
    name = JunctionNode.make_name(map_id, subject, owner)
    junction = junctions.get(name)
    if junction is None:
        junction = JunctionNode(
            map_id=map_id,
            derived_from=derived_from,
            name=name,
            subject=subject,
            owner=owner,
            attribution=CERTAIN,
            anchor=Anchor(
                package=module.package,
                file=module.rel_path,
                line_start=node.lineno,
                line_end=node.end_lineno,
                blob_sha=module.blob_sha,
            ),
        )
        junctions[name] = junction
    known = {(b.outcome, tuple(b.path_condition)) for b in junction.branches}
    if (literal, tuple(condition)) in known:
        return
    junction.branches.append(
        Branch(
            outcome=literal,
            path_condition=condition,
            order=len(junction.branches),
            tier=tier,
        )
    )
