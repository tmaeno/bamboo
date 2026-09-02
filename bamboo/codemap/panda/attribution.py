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
from typing import Optional

from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.pathcond import functions_with_owner

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


def _rooted_at_self(expression: ast.expr) -> bool:
    """Whether *expression* is ``self.<field>``, however many fields deep.

    A bare ``self`` is excluded: that write is the enclosing class's own
    attribute and resolves outright, with no inference involved.
    """
    if not isinstance(expression, ast.Attribute):
        return False
    node: ast.expr = expression
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "self"


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
        self._table_classes: dict[str, str] = {}
        self._self_fields: dict[tuple[str, str], set[str]] = {}

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
            for func, owner in functions_with_owner(module.tree):
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

    # -- what a class does with its own fields ---------------------------- #

    def learn_self_attributes(self, modules: list[SourceModule]) -> None:
        """Pool the attributes each class touches on ``self.<field>``.

        Structural inference is otherwise per function, because a local name is
        only one thing for as long as the function lasts -- two methods using
        ``tmpFileSpec`` need not mean the same kind of object.  ``self.taskSpec``
        is different: it is one field of one class, so every method that touches
        it is describing the same object, and the attributes they touch belong in
        one set.  Same argument as before, on the scope the language guarantees.

        It matters where a class holds a spec and spreads its use thinly:
        ``TaskRefinerBase`` touches ``self.taskSpec.status`` beside only
        ``jediTaskID`` in that method -- two attributes half the specs declare --
        while across the class it touches twelve, which only ``JediTaskSpec``
        has.  That write is the sole producer of the ``topreprocess`` status, so
        without this the graph invariant reported a declared status as
        unreachable.

        Measured before it was believed: five writes settled that were not
        before, no attribution taken away, and 78 sites where a class was
        already stated another way all agreed.
        """
        for module in modules:
            for node in ast.walk(module.tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for inner in ast.walk(node):
                    if (
                        not isinstance(inner, ast.Attribute)
                        or inner.attr not in self._declared_names
                        or not _rooted_at_self(inner.value)
                    ):
                        continue
                    try:
                        expression = ast.unparse(inner.value)
                    except Exception:  # noqa: BLE001
                        continue
                    self._self_fields.setdefault((node.name, expression), set()).add(inner.attr)

    # -- table -> spec class --------------------------------------------- #

    def learn_table_classes(self, modules: list[SourceModule]) -> dict[str, set[str]]:
        """Work out which spec class each SQL table holds, and return conflicts.

        Nothing in PanDA declares this -- no spec names its table -- but the
        column names give it away: a statement writing ``status``,
        ``modificationTime``, ``lockedBy``, ``frozenTime`` and ``errorDialog``
        can only be about ``JediTaskSpec``, because no other spec declares all
        five.  It is the structural argument again, applied to a table instead
        of a variable.

        Evidence is pooled per table across the corpus before being applied.  A
        single statement is often too narrow to decide -- ``SET gshare=:gshare``
        fits both ``JediTaskSpec`` and ``JobSpec`` -- while a seven-column
        ``SET`` elsewhere on the same table settles it, and then the narrow one
        inherits the answer.  Deciding statement by statement leaves a third of
        them open for no reason.

        ``SELECT`` looks like a richer source and must not be used: its column
        list frequently belongs to a joined table rather than the one named
        after ``FROM``, which produced exactly the contradictions this returns
        (``JEDI_Tasks`` reading as both task and dataset) and dropped coverage
        by half.  ``UPDATE`` and ``INSERT`` name one table and mean it.

        Returns the tables whose evidence disagreed, for the gate to report.
        Tables that match no spec are left out, not forced: ``async_results``
        and DEFT's ``T_TASK`` are real tables that hold no spec.
        """
        stated: dict[str, set[str]] = {}
        inferred: dict[str, set[str]] = {}
        for module in modules:
            for func, _owner in functions_with_owner(module.tree):
                for table, spec_class in sql.declared_row_classes(
                    func, set(self._declarations)
                ).items():
                    stated.setdefault(table, set()).add(spec_class)
                seen: set[str] = set()
                for run in sql.executions(func):
                    if run.sql in seen:
                        continue
                    seen.add(run.sql)
                    for write in sql.writes(run.sql):
                        only = self._only_class_declaring(set(write.columns))
                        if only is not None:
                            inferred.setdefault(write.table, set()).add(only)

        conflicts = {
            table: classes
            for source in (stated, inferred)
            for table, classes in source.items()
            if len(classes) > 1
        }
        # Where both sources answer they must agree -- two independent readings
        # of one fact, the same shape as the attribution gate.
        for table in set(stated) & set(inferred):
            if len(stated[table]) == 1 and len(inferred[table]) == 1 and stated[table] != inferred[table]:
                conflicts[table] = stated[table] | inferred[table]

        # What the code states wins where it speaks; inference fills the rest.
        self._table_classes = {
            table: next(iter(classes))
            for source in (inferred, stated)
            for table, classes in source.items()
            if len(classes) == 1 and table not in conflicts
        }
        return conflicts

    def _only_class_declaring(self, columns: set[str]) -> Optional[str]:
        """Return the sole spec declaring every column, matched case-insensitively.

        SQL is written in the column's own spelling but not reliably in the
        spec's -- ``modificationTime`` appears as ``modificationtime`` -- and a
        case-sensitive comparison silently loses most of the wide statements
        that are the only ones able to decide anything.
        """
        lowered = {column.lower() for column in columns}
        if not lowered:
            return None
        matches = [
            spec_class
            for spec_class, declared in self._declarations.items()
            if lowered <= {attribute.lower() for attribute in declared}
        ]
        return matches[0] if len(matches) == 1 else None

    def class_for_table(self, table: str) -> Optional[str]:
        """Return the spec class *table* holds, if it was learned."""
        return self._table_classes.get(table)

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
            for func, _owner in functions_with_owner(module.tree):
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
        if isinstance(expression, ast.Call):
            return self._constructed_class(expression, func)
        if isinstance(expression, ast.Name):
            if expression.id == "self":
                return enclosing_class if enclosing_class in self._declarations else None
            if func is not None:
                stated = self._stated_local_class(expression.id, func)
                if stated is not None:
                    return stated
        return self._structural_of(expression, func, enclosing_class)

    def _annotated_receiver(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[str]:
        """The class an annotation gives the object being written to.

        Three shapes, all of which the annotation policy produces:

        ``self.dataset``
            An attribute rather than a name, so ``self.dataset: DatasetSpec``
            was written and never read.
        ``self.dataset_map[key]``
            A subscript, whose class is the mapping's value type -- and a
            mapping annotated bare (``Dict``) says nothing, which is what makes
            ``Dict[str, DatasetSpec]`` worth asking for.
        ``dataset`` bound from ``self.dataset_map[key]``
            The same, one assignment removed.
        """
        if isinstance(expression, ast.Subscript):
            annotation = self._annotation_of(expression.value, func)
            return self._annotated_element(annotation) if annotation else None
        annotation = self._annotation_of(expression, func)
        stated = self._annotated_class(annotation) if annotation else None
        if stated is not None:
            return stated
        # A local assigned from an annotated container's element.
        if isinstance(expression, ast.Name) and func is not None:
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign) or not any(
                    isinstance(t, ast.Name) and t.id == expression.id
                    for t in node.targets
                ):
                    continue
                if isinstance(node.value, ast.Subscript):
                    return self._annotated_receiver(node.value, func)
        return None

    def _annotation_of(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[ast.expr]:
        """The annotation written for *expression*, wherever it was written.

        The scope searched depends on what the expression is, and the
        distinction is the whole point:

        ``self.dataset_map``
            One object's field, so the class is the scope -- annotated in
            ``__init__`` while the write happens in another method.
        ``dataset_map`` (a bare name)
            A local or a parameter, so **only the enclosing function**.

        Searching the module for either would be name matching across
        unrelated functions, which is the naming heuristic that was measured
        and deleted.  It showed up immediately when this did search the module:
        ``apply_sub_workflow_outputs`` took its type from a same-named
        parameter of ``_check_all_inputs_of_step`` five hundred lines away.
        The answer happened to be right, which is worse than being wrong --
        it is a guess that looks like a reading.
        """
        if isinstance(expression, ast.Name):
            named, scope = expression.id, func
        elif isinstance(expression, ast.Attribute) and _rooted_at_self(expression):
            named = expression.attr
            scope = func
            while scope is not None and not isinstance(scope, ast.ClassDef):
                scope = getattr(scope, "parent", None)
        else:
            return None
        if scope is None:
            return None
        for node in ast.walk(scope):
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if (
                    isinstance(target, ast.Name) and target.id == named
                ) or (isinstance(target, ast.Attribute) and target.attr == named):
                    return node.annotation
        args = getattr(scope, "args", None)
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
            if arg.arg == named and arg.annotation is not None:
                return arg.annotation
        return None

    def _annotated_class(self, annotation: ast.expr) -> Optional[str]:
        """The spec class an annotation names, through the wrappers it may wear.

        Reading only ``Name`` and ``Attribute`` left annotations PanDA had
        already written unread: ``upsert_workflow_entities`` declares
        ``workflow_spec: WorkflowSpec | None = None`` and three writes under it
        came out unattributed, because a PEP 604 union is a ``BinOp`` and
        neither branch matched.

        ``Optional[X]`` and ``X | None`` are the same statement, and both mean
        "X, or absent" -- absence writes nothing, so for the purpose of naming
        what a write went to they say X.  A union of two spec classes says
        neither, and is left open rather than guessed at.
        """
        if isinstance(annotation, ast.Name):
            return annotation.id if annotation.id in self._declarations else None
        if isinstance(annotation, ast.Attribute):
            return annotation.attr if annotation.attr in self._declarations else None
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            found = {
                cls
                for side in (annotation.left, annotation.right)
                if (cls := self._annotated_class(side)) is not None
            }
            return next(iter(found)) if len(found) == 1 else None
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if name == "Optional":
                return self._annotated_class(annotation.slice)
        return None

    def _annotated_element(self, annotation: ast.expr) -> Optional[str]:
        """The spec class a container annotation says it holds.

        ``jobs: List[JobSpec]`` and ``dataset_map: Dict[str, DatasetSpec]`` state
        the element type outright, which is what the annotation policy asks for
        when a container's contents cannot be learned from an adder call.  The
        value type is taken as the last argument, so a mapping gives its values
        rather than its keys.
        """
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            found = {
                cls
                for side in (annotation.left, annotation.right)
                if (cls := self._annotated_element(side)) is not None
            }
            return next(iter(found)) if len(found) == 1 else None
        if not isinstance(annotation, ast.Subscript):
            return None
        base = annotation.value
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name == "Optional":
            return self._annotated_element(annotation.slice)
        inner = annotation.slice
        parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
        return self._annotated_class(parts[-1]) if parts else None

    def _stated_local_class(
        self,
        variable: str,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        seen: Optional[frozenset[str]] = None,
    ) -> Optional[str]:
        """Return the class a local is stated to hold, from annotation or constructor."""
        seen = seen or frozenset()
        if variable in seen:
            return None
        args = getattr(func, "args", None)  # a module has no parameters
        for arg in (
            (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ()
        ):
            if arg.arg != variable or arg.annotation is None:
                continue
            stated = self._annotated_class(arg.annotation)
            if stated is not None:
                return stated
        # ``x: JediDatasetSpec`` on a local or on ``self``, which is the form the
        # annotation policy asks PanDA for when nothing else states the type.
        for node in ast.walk(func):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            named = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if named != variable:
                continue
            stated = self._annotated_class(node.annotation)
            if stated is not None:
                return stated
        constructors: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets
            ):
                continue
            built = self._constructed_class(node.value, func, seen | {variable})
            if built is not None:
                constructors.add(built)
        return next(iter(constructors)) if len(constructors) == 1 else None

    def _constructed_class(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef] = None,
        seen: Optional[frozenset[str]] = None,
    ) -> Optional[str]:
        """Return the spec class an expression constructs or carries over.

        Two shapes beyond the obvious ``FileSpec()``:

        ``SiteSpec.SiteSpec()``
            PanDA imports the module and calls through it, so the callee is an
            attribute rather than a name.  Reading only names left
            ``entity_module.getSiteInfo`` to inference, when the code was
            stating the type outright one attribute along.
        ``copy.copy(spec)``
            A copy holds what the original held.  ``JobGenerator`` builds most
            of its file specs this way, so treating the result as unknown
            discards a type the code has already established.
        """
        if not isinstance(expression, ast.Call):
            return None
        callee = expression.func
        if isinstance(callee, ast.Name) and callee.id in self._declarations:
            return callee.id
        if isinstance(callee, ast.Attribute):
            if callee.attr in self._declarations:
                return callee.attr
            if callee.attr in {"copy", "deepcopy"} and expression.args:
                source = expression.args[0]
                if isinstance(source, ast.Name) and func is not None:
                    # ``seen`` guards ``a = copy.copy(b); b = copy.copy(a)``,
                    # which is not in PanDA but costs one frozenset to rule out.
                    return self._stated_local_class(source.id, func, seen)
                return self._constructed_class(source, func, seen)
        return None

    def _built_class(
        self,
        variable: str,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[str]:
        """The class name *variable* is constructed from, spec or not.

        Deliberately unfiltered, unlike :meth:`_constructed_class`, which only
        answers for classes that declare columns.  Telling "constructed from
        something that is not a spec" apart from "not constructed here" is what
        lets the caller drop a write instead of recording it as unresolved, and
        a filter that returns ``None`` for both cannot make that distinction.
        """
        if func is None:
            return None
        built: set[str] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets
            ):
                continue
            callee = node.value.func if isinstance(node.value, ast.Call) else None
            name = (
                callee.id
                if isinstance(callee, ast.Name)
                else getattr(callee, "attr", None)
                if isinstance(callee, ast.Attribute)
                else None
            )
            if name and name[:1].isupper():
                built.add(name)
        return next(iter(built)) if len(built) == 1 else None

    def _copied_element_class(
        self,
        variable: str,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
        seen: Optional[frozenset[str]] = None,
    ) -> Optional[str]:
        """Return the element class a chain of copies carries over.

        ``copy.copy`` already counts as stating a type when the original was
        constructed, but the original is often a loop variable instead, and the
        two paths did not meet::

            job = JobSpec()                                    # certain
            for tmpFileSpec in job.Files:                      # container
                tmpInputFileSpec = copy.copy(tmpFileSpec)
                tmpZipInputFileSpec = copy.copy(tmpInputFileSpec)
                tmpZipInputFileSpec.lfn = ...                  # was unresolved

        Every hop of that is readable, and three writes in ``getJobs`` fell out
        of the map because the constructor scan does not look at loops and the
        loop scan does not look through copies.

        Reported as ``container`` rather than ``certain`` on purpose: a copy is
        exactly as trustworthy as what it copied, and the weakest link here is
        the assumption that nothing else was appended to that list.
        """
        if func is None:
            return None
        seen = seen or frozenset()
        if variable in seen:
            return None
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets
            ):
                continue
            value = node.value
            if (
                not isinstance(value, ast.Call)
                or not isinstance(value.func, ast.Attribute)
                or value.func.attr not in {"copy", "deepcopy"}
                or not value.args
                or not isinstance(value.args[0], ast.Name)
            ):
                continue
            source = value.args[0].id
            element = self._iterated_element_class(source, func, enclosing_class)
            if element is None:
                element = self._copied_element_class(
                    source, func, enclosing_class, seen | {variable}
                )
            if element is not None:
                return element
        return None

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
                or not isinstance(node.iter, (ast.Attribute, ast.Name))
            ):
                continue
            if isinstance(node.iter, ast.Name):
                # ``for job in jobs`` -- a parameter rather than a spec's own
                # list, so there is no adder to have taught us anything and the
                # annotation is the only statement of the element type.
                annotation = self._annotation_of(node.iter, func)
                stated = (
                    self._annotated_element(annotation) if annotation is not None else None
                )
                if stated is not None:
                    found.add(stated)
                continue
            holder = self.class_of(node.iter.value, func, enclosing_class)
            if holder is None:
                # No adder taught us this container, but it may say so itself:
                # ``self.jobs: List[JobSpec]`` states the element type where
                # the adder idiom is absent.
                annotation = self._annotation_of(node.iter, func)
                stated = (
                    self._annotated_element(annotation) if annotation is not None else None
                )
                if stated is not None:
                    found.add(stated)
                continue
            found |= self._element_types.get((holder, node.iter.attr), set())
        return next(iter(found)) if len(found) == 1 else None

    # -- per-site resolution --------------------------------------------- #

    def declared_attributes(self, spec_class: str) -> set[str]:
        """Return the attributes *spec_class* declares, in their own spelling."""
        return self._declarations.get(spec_class, set())

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

    def family(self, cls: str) -> set[str]:
        """*cls* and every class it inherits from, transitively.

        What ``self`` covers.  Already walked privately to find which class in a
        hierarchy declares an attribute; exposed because resolving
        ``self.<method>()`` asks the same question of methods --
        ``PickleFileSpec(FileSpec)`` and ``AtlasProdPostProcessor``'s two levels
        of base are both real here.
        """
        seen: set[str] = set()
        pending = [cls]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            pending.extend(self._bases.get(name, ()))
        return seen

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
        # The annotation-or-constructor scan lives in one place: typing the
        # object of an ``addFile`` call needs the same answer, and two copies
        # would drift the moment either grew a case.
        stated = self._stated_local_class(variable, func)
        if stated is not None and self._declares(stated, attribute):
            return stated
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
        enclosing_class: Optional[str] = None,
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
        return self._structural_of(target.value, func, enclosing_class)

    def _structural_of(
        self,
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str] = None,
    ) -> Optional[str]:
        """Structural inference over an arbitrary object expression.

        The function is the scope, except for ``self.<field>``, where the class
        is -- see :meth:`learn_self_attributes`.  Widening can only narrow the
        candidates, never move them, so an answer this gives is one the
        per-function reading would have given or left open.
        """
        if func is None:
            return None
        try:
            rendered = ast.unparse(expression)
        except Exception:  # noqa: BLE001
            return None
        touched = set(self._accessed_attributes(func).get(rendered, ()))
        if enclosing_class is not None and _rooted_at_self(expression):
            touched |= self._self_fields.get((enclosing_class, rendered), set())
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

            # The same reasoning as the ``self`` branch above, for a receiver
            # the code constructs: if its class declares no columns it is not a
            # spec, so this is not a spec write and there is nothing to resolve.
            # ``check_result = WFDataTargetCheckResult()`` then
            # ``check_result.metadata = ...`` was being recorded as a junction
            # with an unknown subject -- a decision point the map invented, and
            # the worst kind for elimination, since it can never be the answer.
            built = self._built_class(target.value.id, func)
            if built is not None and built not in self._declarations:
                return None, NOT_A_SPEC

            # A loop over a container whose element type the code states:
            # ``for file in job.Files`` is settled by the ``FileSpec()`` that
            # was put into that list, not by inference about ``file``.
            element = self._iterated_element_class(target.value.id, func, enclosing_class)
            if element is None:
                element = self._copied_element_class(
                    target.value.id, func, enclosing_class
                )
            if element is not None and self._declares(element, attribute):
                return element, CONTAINER

        # An annotation on the receiver, wherever the receiver's shape.  The
        # branches above only look at a bare name, so the two forms the
        # annotation policy actually produces were both unread:
        # ``self.dataset: DatasetSpec`` is an attribute, not a name, and
        # ``self.dataset_map[key].status`` is a subscript of an annotated
        # mapping.  Both state the type outright, which is why they rank here
        # rather than after structural inference.
        annotated = self._annotated_receiver(target.value, func)
        if annotated is not None and self._declares(annotated, attribute):
            return annotated, CERTAIN

        # Structural inference: what the code does with the object.  Measured
        # against the writes the code states outright, the two never disagreed
        # (205 of 205), which is what makes the gate in ``gates`` worth having.
        # It is the last reading because it is the last one that is evidence --
        # what remains after it would be a guess from the variable's name.
        structural = self.structural_class(target, func, enclosing_class)
        if structural is not None and self._declares(structural, attribute):
            return structural, STRUCTURAL

        return None, UNRESOLVED
