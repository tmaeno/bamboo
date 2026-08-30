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
    path_condition,
)

SLICE_NAME = "write-alias"


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
            tier=1,
        )
    )
