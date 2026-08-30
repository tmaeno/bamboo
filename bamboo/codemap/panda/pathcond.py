"""Where a write sits in the code: its enclosing scope and its path condition.

Shared by the recognizers rather than owned by one of them.  Both the attribute
slice and the SQL slice need the same two answers -- which function contains
this write, and which conditions had to hold to reach it -- and a recognizer
importing another recognizer to get them is the kind of dependency that turns
into a cycle the first time either grows.

**A path condition records what decides, not merely that something decided.**
An ``else`` contributes the negation of its ``if``, and a condition written as
a bare local name carries the expression that produced it -- otherwise a branch
guarded by ``if not allowed:`` reads as "some condition held", which cannot be
checked against anything.  PanDA moved a command's acceptance test into exactly
such a helper between two releases.
"""

from __future__ import annotations

import ast
from typing import Iterator, Optional


def functions_with_owner(
    node: ast.AST, owner: Optional[str] = None
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Optional[str]]]:
    """Yield every function in *node* paired with the class enclosing it.

    Carried down the walk rather than read back from a ``parent`` link, so
    this works on a bare tree -- the element-type pass runs before the
    recognizer attaches parents.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from functions_with_owner(child, child.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, owner
            yield from functions_with_owner(child, owner)


def attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]


def ancestors(node: ast.AST) -> Iterator[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None:
        yield current
        current = getattr(current, "parent", None)


def enclosing_function(node: ast.AST) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for ancestor in ancestors(node):
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
    func = enclosing_function(node)
    previous = node
    for ancestor in ancestors(node):
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


def enclosing_class(node: ast.AST) -> Optional[str]:
    for ancestor in ancestors(node):
        if isinstance(ancestor, ast.ClassDef):
            return ancestor.name
    return None
