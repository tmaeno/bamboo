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

Three bases, and the map records which one was used:

``certain``
    A constructor call, a parameter annotation, or ``self`` in a spec class's
    own method.  The code states the type.
``heuristic``
    The variable's name matched a spec class *and* the module imports that
    class.  Naming alone is too weak -- PanDA variables do not carry the
    ``Jedi`` prefix, so ``file`` fits both ``FileSpec`` and ``JediFileSpec``
    and only 24% of ambiguous writes resolve uniquely.  Intersecting with what
    the module imports raises that to 61%, because a module that never imports
    ``JediFileSpec`` is not holding one.
``unresolved``
    Neither applied.  The write is still recorded -- the writer is known even
    when the subject is not, which is enough for localize and prune, both of
    which read observed values rather than the static subject.

Two alternatives were measured and rejected:

* **Fixpoint type propagation** seeded from constructor calls and pushed
  through assignment, ``append`` and iteration.  It resolved nothing extra
  (40% before and after), because the chains do not bottom out at constructors
  -- they die at unannotated parameters, ``JobSpec.addFile(self, file)`` being
  the one that matters most.
* **Attribution by declared vocabulary**, matching the written literal against
  the value sets classes declare.  Only two such vocabularies exist in the
  whole corpus (``JediTaskSpec.status`` and ``JediDatasetSpec.status``), so it
  cannot separate eight classes.  It would also make the vocabulary gate a
  tautology, since the gate checks what the attribution used.

Because the vocabulary cannot check the heuristic either, a heuristic
attribution is *marked* rather than trusted: the real check is conformance
against observed transitions, which needs production data.
"""

from __future__ import annotations

import ast
import re
from typing import Optional

from bamboo.codemap.models import SourceModule

CERTAIN = "certain"
STRUCTURAL = "structural"
HEURISTIC = "heuristic"
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

# Qualifiers PanDA puts in front of a spec variable's name.  They say something
# about the value's role in the function, never about its type.
_PREFIXES = re.compile(r"^(tmp|new|old|orig|cur)_?", re.IGNORECASE)
_SUFFIXES = re.compile(r"_?(list|spec)$", re.IGNORECASE)


def _normalise(name: str) -> str:
    """Reduce an identifier to the bare noun both spellings share.

    ``tmpFileSpec``, ``tmp_file`` and ``FileSpec`` all reduce to ``file``, which
    is what lets a variable be compared against a class name at all.
    """
    stripped = _SUFFIXES.sub("", _PREFIXES.sub("", name))
    return re.sub(r"[^a-z]", "", stripped.lower())


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

    Built once per build: the declaration table and the per-module import sets
    are shared by every write site, and rebuilding them per site would make
    attribution quadratic in a corpus with ~1000 writes.
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
        self._by_noun: dict[str, set[str]] = {}
        for spec_class in declarations:
            noun = _normalise(spec_class)
            self._by_noun.setdefault(noun, set()).add(spec_class)
            # ``JediFileSpec`` answers to ``file`` as well: the Jedi variants
            # are spelled without the prefix at every call site.
            if noun.startswith("jedi"):
                self._by_noun.setdefault(noun[len("jedi") :], set()).add(spec_class)

    # -- module-level context ------------------------------------------- #

    def imported_specs(self, module: SourceModule) -> set[str]:
        """Return the spec classes *module* imports.

        The disambiguator for the ``FileSpec``/``JediFileSpec`` pair, and the
        reason the heuristic is worth having at all.  A module holds what it
        imports; one that imports neither yields no heuristic answer, which is
        the correct outcome rather than a coin flip.
        """
        found: set[str] = set()
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    tail = alias.name.split(".")[-1]
                    if tail in self._declarations:
                        found.add(tail)
                    if alias.asname in self._declarations:
                        found.add(alias.asname)
        return found

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
        if func is None:
            return None
        try:
            expression = ast.unparse(target.value)
        except Exception:  # noqa: BLE001
            return None
        touched = self._accessed_attributes(func).get(expression)
        if not touched:
            return None
        candidates = [
            spec_class
            for spec_class, declared in self._declarations.items()
            if touched <= declared
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _heuristic(
        self, variable: str, attribute: str, imported: set[str]
    ) -> Optional[str]:
        """Resolve from the variable's name, narrowed by the module's imports."""
        candidates = {
            spec_class
            for spec_class in self._by_noun.get(_normalise(variable), set())
            if self._declares(spec_class, attribute)
        }
        narrowed = candidates & imported
        chosen = narrowed or candidates
        return next(iter(chosen)) if len(chosen) == 1 else None

    def attribute_write(
        self,
        target: ast.Attribute,
        *,
        imported: set[str],
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
            variable: Optional[str] = target.value.id
            certain = self._certain(variable, attribute, func, enclosing_class)
            if certain is not None:
                return certain, CERTAIN
        elif isinstance(target.value, ast.Attribute):
            # ``impl.taskSpec.status``: the chain's last attribute names the
            # value as squarely as a variable would, and treating it as one
            # recovers writes that are otherwise dropped for having no plain
            # name on the left -- including ``TaskRefiner``'s task statuses.
            variable = target.value.attr
        else:
            # A subscript or a call has no name to read, but the attributes
            # touched on it are still evidence, so it falls through to the
            # structural pass rather than stopping here.
            variable = None

        # Structural inference outranks the name: what the code does with the
        # object is stronger evidence than what someone called it.  Measured
        # against the writes the code states outright, the two never disagreed
        # (205 of 205), which is what makes the gate in ``gates`` worth having.
        structural = self.structural_class(target, func)
        if structural is not None and self._declares(structural, attribute):
            return structural, STRUCTURAL

        if variable is not None:
            guess = self._heuristic(variable, attribute, imported)
            if guess is not None:
                return guess, HEURISTIC
        return None, UNRESOLVED
