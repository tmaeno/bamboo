"""What values an expression can take.

Resolution rather than recognition, and outside the recognizers for the same
reason attribution is: two slices need the same answer.  The attribute slice
asks it of ``taskSpec.status = <expression>`` and the SQL slice of
``varMap[":status"] = <expression>``, and the value is settled by the same rules
either way -- a map whose two halves resolved the same expression differently
would be describing PanDA twice.

**A declared mapping is the strongest value evidence in the corpus after a
declared vocabulary, and unlike a vocabulary it is complete.**
``JediTaskSpec.commandStatusMap()`` is not a sample of the statuses a command
leads to, it *is* the command-to-status relation::

    def commandStatusMap(cls):
        return {
            "kill":   {"doing": "aborting",  "done": "toabort"},
            "finish": {"doing": "finishing", "done": "passed"},
            ...

So ``commandStatusMap()[commandStr]["done"]`` has a resolvable value set even
though ``commandStr`` is only known at run time: fix the keys the code fixes,
enumerate the ones it does not.  Two of the statuses PanDA declares and the map
could not account for -- ``aborting`` and ``passed`` -- exist nowhere else.

Enumeration is only sound when the whole mapping is literal, so a dict with one
computed entry resolves to nothing rather than to a set missing that entry.  The
difference matters downstream: pruning eliminates, so a value set with a member
missing produces a confident wrong answer where no value set at all produces an
honest tier-2 branch.

The result shape is left to the caller.  The two slices anchor differently -- an
attribute assignment *is* the write, while a bind is written inside a SQL string
and decided at a Python assignment somewhere else -- so they carry different
things alongside the value, and forcing one tuple on both would give each a
field it ignores.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional, Union

from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda.pathcond import functions_with_owner, single_definition

# A literal mapping, nested as deeply as the source nests it.
Mapping = dict[str, Union[str, "Mapping"]]

# ``{0}`` / ``{}`` / ``{schema}`` in a ``str.format`` template.
_FIELD = re.compile(r"\{[^{}]*\}")


def rendered_text(node: ast.expr) -> Optional[str]:
    """Render a string expression, marking the parts only run time knows as ``{}``.

    The shape of the text rather than the text: what the code will produce with
    the holes left open.  Two callers want the same answer for opposite reasons
    -- reading a SQL statement, where the interpolation is almost always the
    schema name and blanking it keeps the statement readable; and indexing a
    diagnostic, where the literal frame is precisely the part a message observed
    in production can be matched against.

    ``None`` when nothing literal can be recovered, which is different from an
    empty string: it means the expression says nothing about its own text.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{}"
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = rendered_text(node.left), rendered_text(node.right)
        if left is None and right is None:
            return None
        return (left or "") + (right or "")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        # ``"FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        # ".format(panda_config.schemaJEDI)`` -- the older spelling of the same
        # f-string, and 62 statements still use it.  Missing it did not merely
        # lose coverage: those statements are where JEDI selects tasks by
        # status, so the map read a dozen task states as ones nothing ever
        # selects on.
        text = rendered_text(node.func.value)
        return None if text is None else _FIELD.sub("{}", text)
    return None


def diagnostic_template(node: ast.expr) -> Optional[str]:
    """The template *node* assembles, if it is one worth indexing.

    Two conditions, both properties of the expression rather than claims about
    the field it is written to -- which is what lets the index avoid deciding
    what counts as a message.

    **Assembled, not written out.**  A template is a frame with holes; a bare
    literal is not one, and an exact message can be found by searching the
    source for itself.  That case the index does not have to earn.

    **Some literal text.**  ``setErrDiag`` appends with ``f"{self.errorDialog}
    {diag}"``, whose frame is two holes and a space: true, and matching nothing.
    A row that cannot serve as a search key is noise in an index that exists to
    be searched.
    """
    assembled = isinstance(node, (ast.JoinedStr, ast.BinOp)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    )
    if not assembled:
        return None
    text = rendered_text(node)
    return text if text and _FIELD.sub("", text).strip() else None


def _literal_mapping(node: ast.expr) -> Optional[Mapping]:
    """Return a dict literal as a plain mapping, or ``None`` if not all literal.

    All-or-nothing on purpose: a mapping with one computed entry cannot be
    enumerated, and a set of values that quietly omits that entry is worse than
    no set at all, because elimination treats a short candidate list as complete.
    """
    if not isinstance(node, ast.Dict):
        return None
    mapping: Mapping = {}
    for key, value in zip(node.keys, node.values, strict=False):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            mapping[key.value] = value.value
            continue
        nested = _literal_mapping(value)
        if nested is None:
            return None
        mapping[key.value] = nested
    return mapping or None


def declared_mappings(modules: list[SourceModule]) -> dict[str, Mapping]:
    """Return ``{method name: the literal mapping it returns}``.

    Keyed by method name for the reason the aliases and producers are: a call
    site offers the name, and the class is what is often being resolved.  Where
    two classes return different mappings from the same name the entry is
    dropped -- the call site cannot be told which one it reached, and a wrong
    value set is worse than none.

    Restricted to methods taking nothing but ``cls``/``self``.  A parameter means
    the mapping is built for a caller rather than declared, and the returned dict
    would then be one of several the method can produce.
    """
    found: dict[str, list[Mapping]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if owner is None:
                continue
            arguments = [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs]
            if len(arguments) > 1 or func.args.vararg or func.args.kwarg:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                mapping = _literal_mapping(node.value)
                if mapping is not None:
                    found.setdefault(func.name, []).append(mapping)
    return {
        name: mappings[0]
        for name, mappings in found.items()
        # ``json`` rather than a set: a nested dict is not hashable, and the
        # comparison wanted here is on content.
        if len({json.dumps(m, sort_keys=True) for m in mappings}) == 1
    }


def _mapping_behind(
    node: ast.expr,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    mappings: dict[str, Mapping],
) -> Optional[Mapping]:
    """Return the declared mapping *node* evaluates to.

    Two shapes, both real in the corpus: the call itself, and a local holding
    it -- ``commandStatusMap = JediTaskSpec.commandStatusMap()`` sits at the top
    of the method that subscripts it a hundred lines further down.
    """
    if isinstance(node, ast.Call):
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
        )
        return mappings.get(name) if name else None
    if isinstance(node, ast.Name) and func is not None:
        definition = single_definition(func, node.id)
        if definition is not None and definition is not node:
            return _mapping_behind(definition, func, mappings)
    return None


def mapping_values(
    expression: ast.expr,
    *,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    mappings: dict[str, Mapping],
) -> list[str]:
    """The values a subscript of a declared mapping can produce, if it is one.

    A key the code states narrows to one entry; a key settled at run time
    enumerates the level.  So ``[commandStr]["done"]`` gives the six statuses a
    completed command leaves behind, and ``["incexec"]["done"]`` gives exactly
    ``rerefine``.

    Returns ``[]`` when the expression is not this shape, when the mapping is
    not declared, or when a stated key is not in it -- that last one being a
    disagreement between two parts of the source, which is not this function's
    to settle.
    """
    keys: list[ast.expr] = []
    node: ast.expr = expression
    while isinstance(node, ast.Subscript):
        keys.append(node.slice)
        node = node.value
    if not keys:
        return []
    mapping = _mapping_behind(node, func, mappings)
    if mapping is None:
        return []

    level: list[Union[str, Mapping]] = [mapping]
    for key in reversed(keys):
        stepped: list[Union[str, Mapping]] = []
        for entry in level:
            if not isinstance(entry, dict):
                return []
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value not in entry:
                    return []
                stepped.append(entry[key.value])
            else:
                stepped.extend(entry.values())
        level = stepped
    if not level or any(not isinstance(entry, str) for entry in level):
        return []
    return sorted({entry for entry in level if isinstance(entry, str)})


def resolver(mappings: dict[str, Mapping]):
    """Return a ``(expression, func) -> [value, ...]`` callable for one build.

    Handed to :func:`bamboo.codemap.panda.pathcond.literal_values` so that a
    local assigned from a declared mapping resolves the same way one assigned a
    literal does.  That is where the two statuses this closes actually live:
    ``newTaskStatus = commandStatusMap[commandStr]["doing"]`` reaches the
    database through a bind, not through the subscript directly.
    """

    def resolve(
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[str]:
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            return [expression.value]
        return mapping_values(expression, func=func, mappings=mappings)

    return resolve
