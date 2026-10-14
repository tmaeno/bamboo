"""What a write to a spec can set where the attribute's name is computed.

``job.jobStatus = "failed"`` names its column.  ``setattr(job, attr, None)``
does not, and neither does a spec's own machinery: ``__init__`` sets every
column to its first value, ``pack`` every column to what the row held, and
``__setattr__`` whichever one its caller assigned.  Which columns such a write
can set, and what it means that it sets them, is said here, so that a list of
the assignments between a spec's read and its flush can name the columns of
each.

Each write gets a role:

``mechanism``
    In a spec class's ``__setattr__``: it writes the name its caller assigned,
    and is that caller's assignment, not one of its own.
``construct`` / ``restore``
    In ``__init__`` / ``__setstate__``: the first values, or the values a
    pickle held.
``row_read``
    In a method :data:`ROW_READ` names: every column set to what the row held
    when it was read.  An assignment, but of the row's own value -- which is
    what tells a flush that writes a value back from one that changes it.
``copy``
    The value is read off another object by the same name (``getattr(other,
    attr)``): the names copied.
``names``
    Any other write whose names run over a declaration or a literal: the names
    a test on the name alone lets through.  A test that reads anything else is
    not decided here, and its names are kept.
``any``
    The names come from a run-time value -- ``updateWorkers`` sets whatever
    keys its caller's dict holds -- so every column the class declares.

A write whose object may be a spec, but whose class is not read, has no role
(:data:`NO_ROLE`).  It is listed, not dropped: a silent drop would read as a
write that sets nothing.

``tracked`` says whether the write goes through a ``__setattr__`` the class
defines -- the one that records what changed.  ``object.__setattr__`` and
``__dict__`` go around it, and a class that defines none records nothing, so a
flush of what changed (``bindUpdateChangesExpression``) does not write those;
a flush of the whole row does.
"""

from __future__ import annotations

import ast
import itertools
import operator
from typing import Iterator, NamedTuple, Optional

from bamboo.codemap.calls import BUILTIN, Types, own_nodes
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.recognizers.progress import (
    SPEC_DECLARATION_NAMES,
    _declared_names,
)

#: The spec methods that set every column from a row the database returned.
#: Declared rather than read: that ``pack`` is handed a row is PanDA's
#: convention, and nothing in its body says where its values came from.
ROW_READ = ("pack",)

MECHANISM = "mechanism"
CONSTRUCT = "construct"
RESTORE = "restore"
READS_ROW = "row_read"
COPY = "copy"
NAMES = "names"
ANY = "any"
NO_ROLE = ""

#: The roles Python itself gives a class's own methods.
_BY_METHOD = {"__setattr__": MECHANISM, "__init__": CONSTRUCT, "__setstate__": RESTORE}

_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

#: A value :func:`_holds` cannot read.
_UNKNOWN = object()


class NameWrite(NamedTuple):
    """One write of a computed attribute name on an object that may be a spec."""

    file: str
    line: int
    function: str
    #: the object written, as the code spells it
    target: str
    #: the spec classes it can be; empty where its class is not read
    classes: tuple[str, ...]
    role: str
    #: the names it can set; ``None`` where they are not read
    names: Optional[frozenset[str]]
    tracked: bool
    reason: str = ""


class _Write(NamedTuple):
    node: ast.AST
    target: ast.expr
    #: ``None`` for ``__dict__.update(...)``, whose names are its argument's keys
    name: Optional[ast.expr]
    value: Optional[ast.expr]
    tracked: bool


def name_writes(types: Types) -> list[NameWrite]:
    """Every write of a computed attribute name in the corpus whose object may be a spec."""
    found: list[NameWrite] = []
    for file in types.corpus_files():
        tree = types.tree(file)
        if tree is None:
            continue
        for func, _owner in pathcond.functions_with_owner(tree):
            for node in own_nodes(func):
                write = _write_of(node)
                if write is None:
                    continue
                read = _read(types, file, func, write)
                if read is not None:
                    found.append(read)
    return found


def columns_of(types: Types, file: str, cls: ast.ClassDef) -> frozenset[str]:
    """The columns *cls* declares, through its bases; the nearest declaration of each name wins."""
    found: set[str] = set()
    for value in _declarations(types, file, cls):
        found |= _declared_names(value)
    return frozenset(found)


def is_spec(types: Types, file: str, cls: ast.ClassDef) -> bool:
    """Whether *cls*, or a base of it, declares its columns."""
    return any(True for _ in _declarations(types, file, cls))


def _declarations(types: Types, file: str, cls: ast.ClassDef) -> Iterator[ast.expr]:
    named: set[str] = set()
    for _where, holder in types.lineage(file, cls):
        for statement in holder.body:
            value = getattr(statement, "value", None)
            names = {
                t.id for t in pathcond.targets_of(statement) if isinstance(t, ast.Name)
            }
            names = (names & SPEC_DECLARATION_NAMES) - named
            if names and value is not None:
                named |= names
                yield value


def _is_dict(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "__dict__"


def _write_of(node: ast.AST) -> Optional[_Write]:
    """The write *node* makes of an attribute by a name it is handed, if it is one."""
    if isinstance(node, ast.Call) and not node.keywords:
        called = node.func
        if (
            isinstance(called, ast.Name)
            and called.id == "setattr"
            and len(node.args) == 3
        ):
            return _Write(node, node.args[0], node.args[1], node.args[2], True)
        if isinstance(called, ast.Attribute) and called.attr == "__setattr__":
            if len(node.args) == 3:
                untracked = (
                    isinstance(called.value, ast.Name) and called.value.id == "object"
                )
                return _Write(
                    node, node.args[0], node.args[1], node.args[2], not untracked
                )
            if len(node.args) == 2:
                return _Write(node, called.value, node.args[0], node.args[1], True)
        if (
            isinstance(called, ast.Attribute)
            and called.attr == "update"
            and _is_dict(called.value)
        ):
            return _Write(node, called.value.value, None, None, False)  # type: ignore[attr-defined]
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and _is_dict(target.value):
                return _Write(node, target.value.value, target.slice, node.value, False)  # type: ignore[attr-defined]
    return None


def _read(types: Types, file: str, func: ast.AST, write: _Write) -> Optional[NameWrite]:
    if isinstance(write.name, ast.Constant) and isinstance(write.name.value, str):
        return None  # a name the code spells is an ordinary assignment
    classes, opened = _classes(types, file, func, write.target)
    specs = [(where, cls) for where, cls in classes if is_spec(types, where, cls)]
    if classes and not specs and not opened:
        return None  # declared some other class
    line = getattr(write.node, "lineno", 0)

    def said(role: str, names: Optional[frozenset[str]], reason: str = "") -> NameWrite:
        return NameWrite(
            file,
            line,
            pathcond.qualified_name(func),
            ast.unparse(write.target),
            tuple(sorted(cls.name for _where, cls in specs)),
            role,
            names,
            write.tracked
            and any(types.method(where, cls, "__setattr__") for where, cls in specs),
            reason,
        )

    if not specs:
        return said(NO_ROLE, None, "the object's class is not read")
    declared = frozenset().union(
        *(columns_of(types, where, cls) for where, cls in specs)
    )
    role = _own_role(types, file, func, write.target)
    if role == MECHANISM:
        return said(MECHANISM, None, "the name its caller assigns")
    if write.name is None:
        return said(role or ANY, declared, "the keys of a dict, not read")
    names, reason = _names(types, file, func, write, specs)
    if role:
        return said(role, names if names is not None else declared, reason)
    if names is None:
        return said(ANY, declared, reason)
    return said(COPY if _copied_from(func, write.value) else NAMES, names, reason)


def _classes(
    types: Types, file: str, func: ast.AST, target: ast.expr
) -> tuple[list[tuple[str, ast.ClassDef]], bool]:
    """The classes *target* can be, and whether there is another this cannot name."""
    if isinstance(target, ast.Name) and target.id == "self":
        held = types.method_class(file, func)
        return ([held] if held is not None else []), held is None
    found, opened = types.candidates(file, func, target)
    resolved: list[tuple[str, ast.ClassDef]] = []
    for where, name in sorted(found):
        cls = types.class_def(where, name) if where != BUILTIN else None
        if cls is not None:
            resolved.append((where, cls))
    return resolved, opened


def _own_role(types: Types, file: str, func: ast.AST, target: ast.expr) -> str:
    """The role a spec's own method gives a write on ``self``, by the method's name."""
    if not (isinstance(target, ast.Name) and target.id == "self"):
        return NO_ROLE
    held = types.method_class(file, func)
    if (
        held is None
        or getattr(func, "parent", None) is not held[1]
        or not is_spec(types, *held)
    ):
        return NO_ROLE
    name = getattr(func, "name", "")
    if name in ROW_READ:
        return READS_ROW
    return _BY_METHOD.get(name, NO_ROLE)


def _names(
    types: Types,
    file: str,
    func: ast.AST,
    write: _Write,
    specs: list[tuple[str, ast.ClassDef]],
) -> tuple[Optional[frozenset[str]], str]:
    """The names *write* can set, and why where they are not read."""
    assert write.name is not None
    bound, source, part = _source(func, write.name)
    if source is None:
        return None, f"{ast.unparse(write.name)} is a run-time value"
    candidates = _sequence(types, file, func, source, part)
    if candidates is None:
        return None, f"{ast.unparse(source)} is not read"
    if bound is None:
        return frozenset(candidates), ""
    loop = _loop_binding(write.node, func, bound)
    return frozenset(
        one for one in candidates if _passes(write.node, loop or func, bound, one)
    ), ""


def _source(
    func: ast.AST, name: ast.expr
) -> tuple[Optional[str], Optional[ast.expr], str]:
    """``(the name's variable, what it runs over, which part)``.

    The part is ``elements``, ``keys`` or ``values``, or ``subscript`` for an
    entry looked up in it: a sequence's element or a mapping's value.
    """
    if isinstance(name, ast.Subscript):
        return None, name.value, "elements"  # ``self._attributes[i]``
    if not isinstance(name, ast.Name):
        return None, None, ""
    loop = _loop_binding(name, func, name.id)
    if loop is not None:
        target, over = loop.target, loop.iter
        if isinstance(target, ast.Name):
            return name.id, over, "elements"
        if (
            isinstance(target, ast.Tuple)
            and len(target.elts) == 2
            and isinstance(over, ast.Call)
        ):
            at = next(
                (
                    i
                    for i, one in enumerate(target.elts)
                    if isinstance(one, ast.Name) and one.id == name.id
                ),
                None,
            )
            called = over.func
            if (
                at == 1
                and isinstance(called, ast.Name)
                and called.id == "enumerate"
                and over.args
            ):
                return name.id, over.args[0], "elements"
            if (
                isinstance(called, ast.Attribute)
                and called.attr == "items"
                and at is not None
            ):
                return name.id, called.value, ("keys", "values")[at]
        return None, None, ""
    bound = pathcond.assigned_expressions(func).get(name.id, [])
    if len(bound) == 1 and isinstance(bound[0], ast.Subscript):
        # ``attr = self._attributes[i]``, or ``attr_wq = mapping[attr]``.
        return name.id, bound[0].value, "subscript"
    return None, None, ""


def _loop_binding(node: ast.AST, func: ast.AST, name: str) -> Optional[ast.For]:
    """The innermost ``for`` around *node* in *func* whose target binds *name*."""
    for ancestor in pathcond.ancestors(node):
        if ancestor is func:
            return None
        if isinstance(ancestor, ast.For) and name in {
            one.id for one in ast.walk(ancestor.target) if isinstance(one, ast.Name)
        }:
            return ancestor
    return None


def _sequence(
    types: Types, file: str, func: ast.AST, node: ast.expr, part: str
) -> Optional[set[str]]:
    """The strings *node* runs over, its *part* of each entry; ``None`` where not read."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _sequence(types, file, func, node.left, part)
        right = _sequence(types, file, func, node.right, part)
        return None if left is None or right is None else left | right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and not node.args
    ):
        if node.func.attr in ("keys", "values"):
            return _sequence(types, file, func, node.func.value, node.func.attr)
    if isinstance(node, ast.Attribute) and node.attr in SPEC_DECLARATION_NAMES:
        classes, _opened = _classes(types, file, func, node.value)
        specs = [(where, cls) for where, cls in classes if is_spec(types, where, cls)]
        if not specs:
            return None
        return set().union(*(columns_of(types, where, cls) for where, cls in specs))
    literal = _literal(types, file, func, node)
    if literal is None:
        return None
    if isinstance(literal, ast.Dict):
        entries = [
            one
            for one in (
                literal.values if part in ("values", "subscript") else literal.keys
            )
            if one is not None
        ]
    else:
        entries = list(literal.elts)
    strings = {
        one.value
        for one in entries
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    }
    return strings if len(strings) == len(entries) else None


def _literal(
    types: Types, file: str, func: ast.AST, node: ast.expr
) -> Optional[ast.expr]:
    """The literal tuple, list, set or dict *node* is: written here, or a class attribute of ``self``."""
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return node
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        held = types.method_class(file, func)
        for _where, holder in types.lineage(*held) if held is not None else ():
            for statement in holder.body:
                value = getattr(statement, "value", None)
                if value is not None and any(
                    isinstance(t, ast.Name) and t.id == node.attr
                    for t in pathcond.targets_of(statement)
                ):
                    return (
                        value
                        if isinstance(value, (ast.Tuple, ast.List, ast.Set, ast.Dict))
                        else None
                    )
    if isinstance(node, ast.Name):
        bound = pathcond.assigned_expressions(func).get(node.id, [])
        if len(bound) == 1 and isinstance(
            bound[0], (ast.Tuple, ast.List, ast.Set, ast.Dict)
        ):
            return bound[0]
    return None


def _passes(node: ast.AST, stop: ast.AST, name: str, value: str) -> bool:
    """Whether a test on the way from *stop* to *node* can let *node* run with *name* bound to *value*.

    Loops over literals in between are run through, each value in turn: a
    ``for patt in [...]: if attr.endswith(patt)`` lets a name through when
    one of them does.
    """
    required = _required(node, stop)
    loops = [
        (loop.target.id, [one.value for one in loop.iter.elts])  # type: ignore[attr-defined]
        for loop in _literal_loops(node, stop)
    ]
    for chosen in itertools.product(*(values for _name, values in loops)):
        bound = {
            name: value,
            **{
                loop_name: one
                for (loop_name, _values), one in zip(loops, chosen, strict=True)
            },
        }
        if all(_holds(test, bound) in (wanted, None) for test, wanted in required):
            return True
    return False


def _required(node: ast.AST, stop: ast.AST) -> list[tuple[ast.expr, bool]]:
    """``(test, the value it must have)`` for every test *node* is reached under, below *stop*."""
    found: list[tuple[ast.expr, bool]] = []
    previous = node
    for ancestor in pathcond.ancestors(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(ancestor, field, None)
            if isinstance(block, list) and previous in block:
                for statement in block[: block.index(previous)]:
                    if isinstance(statement, ast.If):
                        if pathcond._jumps(statement.body):
                            found.append((statement.test, False))
                        elif statement.orelse and pathcond._jumps(statement.orelse):
                            found.append((statement.test, True))
                break
        if isinstance(ancestor, ast.If):
            if previous in ancestor.body:
                found.append((ancestor.test, True))
            elif previous in ancestor.orelse:
                found.append((ancestor.test, False))
        if ancestor is stop:
            break
        previous = ancestor
    return found


def _literal_loops(node: ast.AST, stop: ast.AST) -> list[ast.For]:
    found = []
    for ancestor in pathcond.ancestors(node):
        if ancestor is stop:
            break
        if (
            isinstance(ancestor, ast.For)
            and isinstance(ancestor.target, ast.Name)
            and isinstance(ancestor.iter, (ast.Tuple, ast.List))
            and all(isinstance(one, ast.Constant) for one in ancestor.iter.elts)
        ):
            found.append(ancestor)
    return found


def _known(node: ast.expr, bound: dict[str, object]) -> object:
    if isinstance(node, ast.Name):
        return bound.get(node.id, _UNKNOWN)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return _UNKNOWN


def _holds(test: ast.expr, bound: dict[str, object]) -> Optional[bool]:
    """``True`` or ``False`` where the names *bound* decide *test*, else ``None``."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _holds(test.operand, bound)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp):
        results = [_holds(one, bound) for one in test.values]
        stop = isinstance(test.op, ast.Or)
        if any(result is stop for result in results):
            return stop
        return (not stop) if all(result is (not stop) for result in results) else None
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = _known(test.left, bound), _known(test.comparators[0], bound)
        compare = _COMPARE.get(type(test.ops[0]))
        if compare is None or left is _UNKNOWN or right is _UNKNOWN:
            return None
        try:
            return bool(compare(left, right))
        except TypeError:
            return None
    if (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr in ("startswith", "endswith")
        and len(test.args) == 1
        and not test.keywords
    ):
        subject, argument = _known(test.func.value, bound), _known(test.args[0], bound)
        if isinstance(subject, str) and isinstance(argument, (str, tuple)):
            return getattr(subject, test.func.attr)(argument)
    return None


def _copied_from(func: ast.AST, value: Optional[ast.expr]) -> bool:
    """Whether *value* is read off a parameter by name -- ``getattr(other, attr)``.

    Here, or through a local every binding of which is one: ``pack_gs`` binds
    ``val = getattr(gshare, attr)`` in two branches.
    """
    arguments = getattr(func, "args", None)
    if arguments is None:
        return False
    parameters = {
        one.arg
        for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    } - {"self", "cls"}
    values = [value]
    if isinstance(value, ast.Name):
        values = list(pathcond.assigned_expressions(func).get(value.id, []))
    return bool(values) and all(
        isinstance(one, ast.Call)
        and isinstance(one.func, ast.Name)
        and one.func.id == "getattr"
        and one.args
        and isinstance(one.args[0], ast.Name)
        and one.args[0].id in parameters
        for one in values
    )
