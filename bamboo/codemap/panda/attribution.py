"""Which spec class does a write belong to?

``x.status = "cached"`` names an attribute but not a class, and eight PanDA
spec classes declare ``status``.  Until that is settled the write cannot be
filed under a subject, because a subject is ``(spec_class, attribute)`` -- and
attributing it wrongly is worse than leaving it open, since a junction filed
under the wrong subject is a false lead that the reasoning will follow.

Attribution is kept out of the recognizers because it is *resolution*, not
recognition: the progress recognizer knows a write happened, this decides what
it wrote to.  Selection works on the same spec variables, so folding it into
one recognizer would mean writing it twice.

Three bases, strongest first, and the map records which one was used:

``certain``
    A constructor call, a parameter annotation, ``self`` in a spec class's own
    method (or a subclass of one), or an attribute only one class declares.
    The code states the type.
``container``
    The variable iterates a list whose element type the code states, one hop
    through the adder idiom -- see :meth:`SpecAttributor.learn_element_types`.
``structural``
    Exactly one class declares every attribute the code touches on the object.
    A name is what someone called it; the attributes touched are what the code
    requires it to be.

A write none of them settle is still recorded, under a placeholder subject:
the writer is known even when the subject is not, which is enough for localize
and prune, both of which read observed values rather than the static subject.

Every basis reads evidence rather than guessing.  There used to be a fourth --
matching the variable's name against the spec class names, narrowed by the
module's imports -- and it was removed: it settled two writes out of 278, and
it was the only basis whose answers had to be marked as untrusted.  The rule
this leaves behind is that **when a shape cannot be resolved from evidence,
the answer is a standard type annotation in PanDA (kept honest by its own CI),
not another inference mechanism here**.  Inference is for systems that cannot
be asked -- pilot, harvester, DDM -- and PanDA can be.

**Attribution by declared vocabulary** was tried and does not work: only two
such vocabularies exist in the whole corpus (``JediTaskSpec.status`` and
``JediDatasetSpec.status``), so it cannot separate eight classes, and it would
make the vocabulary gate a tautology by checking what the attribution used.
The gate that does work compares the stated class against the structural one
-- two independent readings, no production data (see ``codemap.gates``).
"""

from __future__ import annotations

import ast
from typing import Iterator, Optional

from bamboo.codemap.models import SourceModule

CERTAIN = "certain"
# One hop through the adder idiom: what the code put into the container.
CONTAINER = "container"
STRUCTURAL = "structural"
UNRESOLVED = "unresolved"

# Not a write to a spec at all -- the caller drops it instead of recording an
# unresolved junction.  ``self.vo = "atlas"`` in a WatchDog writes the
# WatchDog's own field; that ``vo`` is also a spec attribute name is a
# collision, not a relationship.
NOT_A_SPEC = "not-a-spec"

# Subject placeholder for a write whose class could not be settled.  Spelled
# with a character no Python identifier can contain, so an unresolved subject
# can never collide with a real one.
UNRESOLVED_CLASS = "?"


def _functions_with_owner(
    node: ast.AST, owner: Optional[str] = None
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Optional[str]]]:
    """Yield every function in *node* paired with the class enclosing it.

    Carried down the walk rather than read back from a ``parent`` link, so
    this works on a bare tree -- the element-type pass runs before the
    recognizer attaches parents.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _functions_with_owner(child, child.name)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child, owner
            yield from _functions_with_owner(child, owner)
        else:
            yield from _functions_with_owner(child, owner)


def class_bases(modules: list[SourceModule]) -> dict[str, list[str]]:
    """Return ``{class: [base class, ...]}`` for every class in the corpus.

    Needed because a spec class can be subclassed: ``PickleFileSpec(FileSpec)``
    and ``PickleJobSpec(JobSpec)`` both exist, and a ``self.status`` inside one
    of them is a genuine ``FileSpec.status`` write.  Without the hierarchy the
    rule for ``self`` would either miss those or, worse, refuse to drop the
    WatchDog writes it is there to drop.
    """
    bases: dict[str, list[str]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            names = [
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                for base in node.bases
            ]
            bases[node.name] = [n for n in names if n]
    return bases


class SpecAttributor:
    """Resolves the spec class behind an attribute write.

    Built once per build: the declaration table, the class hierarchy and the
    container element types are shared by every write site, and rebuilding them
    per site would make attribution quadratic in a corpus with ~1000 writes.
    """

    def __init__(
        self,
        declarations: dict[str, set[str]],
        bases: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._declarations = declarations
        self._bases = bases or {}
        self._declared_names: set[str] = (
            set().union(*declarations.values()) if declarations else set()
        )
        self._accessed_cache: dict[ast.AST, dict[str, set[str]]] = {}
        self._element_types: dict[tuple[str, str], set[str]] = {}

    # -- container element types ----------------------------------------- #

    def learn_element_types(self, modules: list[SourceModule]) -> None:
        """Record what each spec's list attributes hold, from the adder idiom.

        PanDA states this outright, twice over, and in the one place it matters
        most -- the ``FileSpec``/``JediFileSpec`` split that defeats naming::

            # JobSpec.py            file_spec = FileSpec()
            #                       self.addFile(file_spec)
            # TaskRefinerBase.py    fileSpec = JediFileSpec()
            #                       datasetSpec.addFile(fileSpec)

        so ``JobSpec.Files`` holds ``FileSpec`` and ``JediDatasetSpec.Files``
        holds ``JediFileSpec``.  Reading it turns ``for file in job.Files``
        from unresolvable into settled, with no annotation asked of PanDA.

        The rule is one hop, not interprocedural analysis: a method whose body
        appends a parameter to ``self.<attr>`` passes its caller's argument
        type through to that attribute.  Direct ``x.attr.append(v)`` counts
        too.  Nothing here needs the element types it produces, so a single
        pass suffices -- receivers and arguments are typed from constructor
        calls, annotations, ``self``, and structural inference, none of which
        consult this table.

        The assumption is homogeneity: that nothing else is ever added to the
        same list.  That is what keeps this out of ``certain``.
        """
        adders = self._adder_methods(modules)
        for module in modules:
            for func, owner in _functions_with_owner(module.tree):
                for call in (n for n in ast.walk(func) if isinstance(n, ast.Call)):
                    if not isinstance(call.func, ast.Attribute) or not call.args:
                        continue
                    receiver = call.func.value
                    if call.func.attr == "append" and isinstance(receiver, ast.Attribute):
                        holder, attribute = receiver.value, receiver.attr
                    else:
                        holder, attribute = receiver, adders.get(call.func.attr)
                        if attribute is None:
                            continue
                    # The container attribute is deliberately not required to
                    # be a declared one: ``_attributes`` is the DB column list,
                    # and ``Files`` is an edge in the object graph rather than
                    # a column -- ``__slots__ = _attributes + ("Files", ...)``.
                    holder_class = self.class_of(holder, func, owner)
                    if holder_class is None:
                        continue
                    element = self.class_of(call.args[0], func, owner)
                    if element is not None:
                        self._element_types.setdefault((holder_class, attribute), set()).add(
                            element
                        )

    def _adder_methods(self, modules: list[SourceModule]) -> dict[str, str]:
        """Return ``{method name: attribute}`` for ``self.<attr>.append(<param>)``.

        Keyed by method name rather than by class because the receiver's class
        is often what is being resolved; requiring it first would make the
        table useless exactly where it is needed.  The name is distinctive
        enough in practice -- ``addFile`` means the same thing on ``JobSpec``
        and ``JediDatasetSpec``, differing only in what it holds.
        """
        adders: dict[str, str] = {}
        for module in modules:
            for func, _owner in _functions_with_owner(module.tree):
                parameters = {a.arg for a in (*func.args.posonlyargs, *func.args.args)}
                for call in (n for n in ast.walk(func) if isinstance(n, ast.Call)):
                    if (
                        isinstance(call.func, ast.Attribute)
                        and call.func.attr == "append"
                        and isinstance(call.func.value, ast.Attribute)
                        and isinstance(call.func.value.value, ast.Name)
                        and call.func.value.value.id == "self"
                        and len(call.args) == 1
                        and isinstance(call.args[0], ast.Name)
                        and call.args[0].id in parameters
                    ):
                        adders[func.name] = call.func.value.attr
        return adders

    def class_of(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
    ) -> Optional[str]:
        """Return the spec class an expression evaluates to, if it can be told.

        Attribute-agnostic, unlike :meth:`attribute_write`, because the object
        of an ``addFile`` call has to be typed without knowing which attribute
        is being written.
        """
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            return expression.func.id if expression.func.id in self._declarations else None
        if isinstance(expression, ast.Name):
            if expression.id == "self":
                return enclosing_class if enclosing_class in self._declarations else None
            if func is not None:
                stated = self._stated_local_class(expression.id, func)
                if stated is not None:
                    return stated
        return self._structural_of(expression, func)

    def _stated_local_class(
        self, variable: str, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Optional[str]:
        """Return the class a local is stated to hold, from annotation or constructor."""
        for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs):
            if arg.arg != variable or arg.annotation is None:
                continue
            annotation = arg.annotation
            name = (
                annotation.id
                if isinstance(annotation, ast.Name)
                else getattr(annotation, "attr", None)
            )
            if name in self._declarations:
                return name
        constructors = {
            node.value.func.id
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self._declarations
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == variable
        }
        return next(iter(constructors)) if len(constructors) == 1 else None

    def _iterated_element_class(
        self,
        variable: str,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
    ) -> Optional[str]:
        """Return the class of a variable that iterates a typed container."""
        if func is None:
            return None
        found: set[str] = set()
        for node in ast.walk(func):
            if (
                not isinstance(node, (ast.For, ast.AsyncFor))
                or not isinstance(node.target, ast.Name)
                or node.target.id != variable
                or not isinstance(node.iter, ast.Attribute)
            ):
                continue
            holder = self.class_of(node.iter.value, func, enclosing_class)
            if holder is None:
                continue
            found |= self._element_types.get((holder, node.iter.attr), set())
        return next(iter(found)) if len(found) == 1 else None

    # -- per-site resolution --------------------------------------------- #

    def _declares(self, spec_class: str, attribute: str) -> bool:
        return attribute in self._declarations.get(spec_class, ())

    def _declaring_ancestor(self, cls: str, attribute: str) -> Optional[str]:
        """Return the class in *cls*'s hierarchy that declares *attribute*."""
        seen: set[str] = set()
        pending = [cls]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if self._declares(name, attribute):
                return name
            pending.extend(self._bases.get(name, ()))
        return None

    def _certain(
        self,
        variable: str,
        attribute: str,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
    ) -> Optional[str]:
        """Resolve from what the code states outright, or return ``None``."""
        if variable == "self":
            # Handled before this point: ``self`` is the enclosing class, so
            # the answer is certain either way and never falls through to the
            # heuristic.
            return (
                self._declaring_ancestor(enclosing_class, attribute)
                if enclosing_class
                else None
            )
        if func is None:
            return None

        for arg in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs):
            if arg.arg != variable or arg.annotation is None:
                continue
            annotation = arg.annotation
            name = (
                annotation.id
                if isinstance(annotation, ast.Name)
                else getattr(annotation, "attr", None)
            )
            if name and self._declares(name, attribute):
                return name

        constructors = {
            node.value.func.id
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self._declarations
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == variable
        }
        # Several constructors mean the variable is reused for different
        # classes; that is a genuine ambiguity, not something to pick from.
        if len(constructors) == 1:
            only = next(iter(constructors))
            if self._declares(only, attribute):
                return only
        return None

    # -- structural inference -------------------------------------------- #

    def _accessed_attributes(
        self, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, set[str]]:
        """Return ``{object expression: attributes touched on it}`` for *func*.

        Only names some spec declares are kept.  ``_attributes`` is the class's
        *column* list, not its attribute surface, so methods and computed
        fields -- ``addFile``, ``getTransient``, ``isAllowedNoOutput`` -- are
        never in it.  Leaving them in made the superset test fail for eleven
        writes whose class was in fact unambiguous.

        Memoised per function: the map is read once per write site, and
        recomputing it would make attribution quadratic in a function's size.
        """
        cached = self._accessed_cache.get(func)
        if cached is not None:
            return cached
        found: dict[str, set[str]] = {}
        for node in ast.walk(func):
            if not isinstance(node, ast.Attribute) or node.attr not in self._declared_names:
                continue
            try:
                expression = ast.unparse(node.value)
            except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
                continue
            found.setdefault(expression, set()).add(node.attr)
        self._accessed_cache[func] = found
        return found

    def structural_class(
        self,
        target: ast.Attribute,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[str]:
        """Return the spec class implied by what the code does with the object.

        The strongest evidence available without leaving the file, and the one
        a reader uses without noticing::

            for file in self.job.Files:
                file.status = "merging"     # which status is this?
            ...
            if file.lfn in ...:             # lfn, type, GUID, checksum, fsize,
            file.md5sum                     # md5sum -- only FileSpec has them all

        A variable's name is what someone called it; the attributes touched on
        it are what the code requires it to be.  Where exactly one spec class
        declares a superset of them, that class is the answer.

        Returns ``None`` when several classes fit -- ``{lfn, status, type}``
        belongs to both ``FileSpec`` and ``JediFileSpec`` -- rather than
        picking, since a wrong subject is a false lead.
        """
        return self._structural_of(target.value, func)

    def _structural_of(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[str]:
        """Structural inference over an arbitrary object expression."""
        if func is None:
            return None
        try:
            rendered = ast.unparse(expression)
        except Exception:  # noqa: BLE001
            return None
        touched = self._accessed_attributes(func).get(rendered)
        if not touched:
            return None
        candidates = [
            spec_class
            for spec_class, declared in self._declarations.items()
            if touched <= declared
        ]
        return candidates[0] if len(candidates) == 1 else None

    def attribute_write(
        self,
        target: ast.Attribute,
        *,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
    ) -> tuple[Optional[str], str]:
        """Return ``(spec_class, basis)`` for a write to *target*.

        ``spec_class`` is ``None`` when unresolved; the caller records the
        junction anyway, under the placeholder subject, because knowing where a
        status is written is useful even when it is not yet known whose status
        it is.
        """
        attribute = target.attr

        # ``self.attr`` writes the enclosing class's own attribute, by
        # definition.  So the hierarchy answers it outright, in both
        # directions: a subclass of a spec resolves to the spec, and a class
        # whose hierarchy never declares the attribute is not writing a spec at
        # all -- which is a fact about the code, not a failure to resolve it.
        if isinstance(target.value, ast.Name) and target.value.id == "self":
            if enclosing_class is None:
                return None, UNRESOLVED
            owner = self._declaring_ancestor(enclosing_class, attribute)
            return (owner, CERTAIN) if owner else (None, NOT_A_SPEC)

        # A single declaring class settles it without looking at the object
        # expression at all -- ``jobStatus`` and ``ddmErrorDiag`` resolve here,
        # including in ``self.job.jobStatus = ...`` where the object is itself
        # an attribute rather than a plain name.
        declaring = [c for c, attrs in self._declarations.items() if attribute in attrs]
        if len(declaring) == 1:
            return declaring[0], CERTAIN

        if isinstance(target.value, ast.Name):
            certain = self._certain(target.value.id, attribute, func, enclosing_class)
            if certain is not None:
                return certain, CERTAIN

            # A loop over a container whose element type the code states:
            # ``for file in job.Files`` is settled by the ``FileSpec()`` that
            # was put into that list, not by inference about ``file``.
            element = self._iterated_element_class(target.value.id, func, enclosing_class)
            if element is not None and self._declares(element, attribute):
                return element, CONTAINER

        # Structural inference: what the code does with the object.  Measured
        # against the writes the code states outright, the two never disagreed
        # (205 of 205), which is what makes the gate in ``gates`` worth having.
        # It is the last reading because it is the last one that is evidence --
        # what remains after it would be a guess from the variable's name.
        structural = self.structural_class(target, func)
        if structural is not None and self._declares(structural, attribute):
            return structural, STRUCTURAL

        return None, UNRESOLVED
