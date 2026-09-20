"""Where a value came from, when the answer is outside this map.

The terminal classification the backward walk needs and the map's own edges
cannot give: an expression is a call on an interface, and which interface it
is decides whether the answer is "DDM said so", "a configuration row said
so", or "another junction decided it and the map names which".

**Named interfaces are what makes this static.**  PanDA reaches everything
outside itself through a handful of attributes held on ``self`` --
``ddmIF``, ``siteMapper``, ``taskBufferIF`` -- so the receiver, once resolved,
is the classification.  This is the plan's provenance table, and the reason it
lives in the plugin rather than beside the walk: the walk is about Python, and
these names are about PanDA.

**Resolved, not matched.**  The obvious implementation is a substring test for
``ddmIF`` in the source text, and a prototype that did that put the config
terminal on 545 arms and the external ones on 34.  Those counts are a lower
bound and nothing more: ``ddmIF = self.ddmIF.getInterface(vo, cloud)`` binds a
bare local that no ``self.`` test would match, and the corpus has punished
name matching twice already.  So the receiver chain is walked to its base and,
when the base is a local, one reaching definition is followed -- which is what
turns ``ddmIF.getDatasetMetaData(...)`` into DDM.
"""

from __future__ import annotations

import ast
from typing import Callable, Optional

from bamboo.codemap.models import STOP_CONFIG, STOP_EXTERNAL, STOP_UPSTREAM

#: ``self.<attribute>`` -> the system on the other side of it.
EXTERNAL_INTERFACES = {
    "ddmIF": "DDM / Rucio",
    "siteMapper": "CRIC site declaration",
}

#: Interfaces that land back in PanDA's own database.  A terminal for the
#: trace and a continuation for the map, which is the point of naming it
#: separately: the walk stops here and ``producers_of`` takes over.
#:
#: ``cur`` is the Oracle cursor the proxy modules read rows through, and it is
#: the single commonest place a walk ends -- 120 steps of 4852.  Without it
#: those come back as "chosen by whoever called or built this", which is true
#: of the cursor object and false of the row: the row came from the statement
#: executed a few lines above, and the map has read that statement.
DATABASE_INTERFACES = {"taskBufferIF", "taskBuffer", "cur"}

_CONFIG_CALL = "getConfigValue"
_CONFIG_MODULE = "jedi_config"

Resolver = Callable[[str], Optional[ast.expr]]


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _called(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _config_key(call: ast.Call) -> str:
    """The literal arguments of a config lookup, which are its key.

    Template keys (``SCOUT_NUM_CPU_INEFFICIENT_{label}``) come back as written.
    Rendering them would need the label, which is another walk; the key as the
    code spells it is already what a person changes.
    """
    literals = [
        argument.value
        for argument in call.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    ]
    return ".".join(literals) if literals else _text(call)


def _base(attribute: ast.Attribute) -> tuple[Optional[ast.expr], str]:
    """Descend a dotted chain to its base, with the attribute nearest to it."""
    node: ast.expr = attribute
    nearest = ""
    while isinstance(node, ast.Attribute):
        nearest = node.attr
        node = node.value
    return node, nearest


def _rooted_in(
    attribute: ast.Attribute, resolve: Optional[Resolver], hops: int = 1
) -> Optional[tuple[str, str]]:
    """``(self.<field>, the call on it)`` a dotted chain is rooted in.

    Both halves, because a hop through a local moves the interesting name.
    ``taskSpec.getNumFilesPerJob()`` where ``taskSpec`` came out of
    ``self.taskBufferIF.getTaskWithID_JEDI(...)`` is the database's, and the
    call to name is the one that went to the database -- reporting
    ``getNumFilesPerJob`` instead would be a sentence that is false about
    where the value came from.
    """
    base, nearest = _base(attribute)
    if not isinstance(base, ast.Name):
        return None
    if base.id == "self":
        return nearest, attribute.attr
    if hops <= 0 or resolve is None:
        return None
    bound = resolve(base.id)
    if bound is None:
        return None
    target = bound.func if isinstance(bound, ast.Call) else bound
    if isinstance(target, ast.Attribute):
        return _rooted_in(target, resolve, hops - 1)
    return None


def _terminal_of(
    node: ast.AST, resolve: Optional[Resolver]
) -> Optional[tuple[str, str]]:
    if isinstance(node, ast.Call) and _called(node.func) == _CONFIG_CALL:
        return STOP_CONFIG, _config_key(node)
    if isinstance(node, ast.Attribute):
        base, _ = _base(node)
        if isinstance(base, ast.Name) and base.id == _CONFIG_MODULE:
            return STOP_CONFIG, _text(node)
        rooted = _rooted_in(node, resolve)
        if rooted is None:
            return None
        field, call = rooted
        if field in EXTERNAL_INTERFACES:
            return STOP_EXTERNAL, EXTERNAL_INTERFACES[field]
        if field in DATABASE_INTERFACES:
            return STOP_UPSTREAM, call if call != field else "the task buffer"
    return None


def classify(
    expression: ast.expr, resolve: Optional[Resolver] = None
) -> tuple[str, str]:
    """``(terminal, detail)`` for *expression*, or ``("", "")`` to keep walking.

    Breadth-first, so the outermost construct wins: in
    ``res = self.taskBufferIF.insertFiles(getConfigValue("jedi", "x"))`` the
    value came from the database call and the config key is an argument to it.
    Returning one rather than all of them keeps the step's answer a single
    category, which is what "where did this come from" means.
    """
    for node in ast.walk(expression):
        found = _terminal_of(node, resolve)
        if found is not None:
            return found
    return "", ""
