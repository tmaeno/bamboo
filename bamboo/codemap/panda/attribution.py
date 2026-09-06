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
import re
from typing import Iterator, NamedTuple, Optional

from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.pathcond import attach_parents, functions_with_owner

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

# Subscripted forms whose arguments name the members of a record rather than
# the type of an element.  ``Callable`` describes a signature and ``Tuple``
# describes positions, so in neither case is a spec named inside one the class
# of the annotated object or of anything taken out of it.
_RECORD_WRAPPERS = frozenset({"Callable", "Tuple", "tuple"})


def _unquoted(annotation: ast.expr, depth: int = 0) -> ast.expr:
    """*annotation* with a quoted type expression opened up.

    A quote is a runtime concern, not a change of statement.  panda-server
    evaluates annotations at import time -- no module in the corpus uses
    ``from __future__ import annotations`` -- so ``if TYPE_CHECKING`` plus a
    quoted name is the only way to name a type that cannot be imported at
    runtime, and ``base_module`` says why in a comment of its own: importing
    ``WrappedCursor`` there would close an import cycle.  Discarding the quoted
    form makes runtime safety and map visibility exclusive, which is a choice
    the target system should not have to make on our behalf.

    Read by compiling the string as an expression, which is what
    ``typing.ForwardRef`` does with the same text.  Prose in an annotation slot
    is not a type and need not parse, so a failure is "nothing" rather than an
    exception -- one malformed annotation must not take down a build.  The
    depth cap covers a quote inside a quote, which no real code writes.
    """
    if not isinstance(annotation, ast.Constant) or not isinstance(annotation.value, str):
        return annotation
    if depth >= 2:
        return annotation
    try:
        parsed = ast.parse(annotation.value.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return annotation
    return _unquoted(parsed, depth + 1)


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
        self._class_nodes: dict[str, ast.ClassDef] = {}

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
                # Kept for the same reason the attribute pool is: a field
                # annotated in a base class is read in the derived one.
                self._class_nodes.setdefault(node.name, node)
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
        """Return the spec class *table* holds, if it was learned.

        Matched without regard to case, as :meth:`_only_class_declaring` already
        matches columns: SQL identifiers are case-insensitive and the corpus
        uses both spellings of the same table -- ``reassignShare`` loops over
        ``["jobsactive4", "jobsdefined4"]`` where everything else writes
        ``jobsActive4``.  Requiring the spelling to agree made those two look
        like tables holding no spec, which put a ``JobSpec.gshare`` write on a
        table-qualified subject of its own.
        """
        found = self._table_classes.get(table)
        if found is not None:
            return found
        lowered = table.lower()
        return next(
            (
                spec
                for known, spec in self._table_classes.items()
                if known.lower() == lowered
            ),
            None,
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

    @staticmethod
    def _container_read(expression: ast.expr) -> Optional[ast.expr]:
        """The container an expression takes one element out of, if it does.

        ``d[key]`` and ``d.get(key)`` are the same read and PanDA writes both,
        but only the subscript was read.  That left an annotation which was
        already correct doing nothing: ``workflow_core`` states
        ``data_spec_map: Dict[str, WFDataSpec]`` and binds the receiver with
        ``data_spec = data_spec_map.get(output_data_name)``, so three writes
        stayed unattributed with the answer written above them.

        The argument is required.  ``.get`` is a common method on objects that
        are not mappings, and a ``Queue``'s ``get()`` takes none -- claiming an
        element type there would be a confident wrong answer, which is worse
        than the unresolved write it replaces.
        """
        if isinstance(expression, ast.Subscript):
            return expression.value
        if (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "get"
            and expression.args
        ):
            return expression.func.value
        return None

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
        ``self.dataset_map[key]`` / ``self.dataset_map.get(key)``
            A read of one element, whose class is the mapping's value type --
            and a mapping annotated bare (``Dict``) says nothing, which is what
            makes ``Dict[str, DatasetSpec]`` worth asking for.
        ``dataset`` bound from either of those
            The same, one assignment removed.
        """
        container = self._container_read(expression)
        if container is not None:
            annotation = self._annotation_of(container, func)
            return self._annotated_element(annotation) if annotation else None
        annotation = self._annotation_of(expression, func)
        stated = self._annotated_class(annotation) if annotation else None
        if stated is not None:
            return stated
        # A local assigned from something already annotated -- an element of an
        # annotated container, or an annotated field of the object.  ``finisher``
        # guards ``self.dataset`` for None once at the top and works through a
        # local from there, which is the right way to write it and put the sole
        # producer of ``DatasetSpec.status = 'cleanup'`` out of reach: the
        # annotation is on the field, the write is on the local, one hop apart.
        #
        # Every candidate is collected rather than the first one returned,
        # because returning on the first match would make the answer depend on
        # walk order, and a local assigned from two differently annotated
        # places states nothing about either.
        if isinstance(expression, ast.Name) and func is not None:
            sources: set[str] = set()
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign) or not any(
                    isinstance(t, ast.Name) and t.id == expression.id
                    for t in node.targets
                ):
                    continue
                if self._container_read(node.value) is None and not (
                    isinstance(node.value, ast.Attribute) and _rooted_at_self(node.value)
                ):
                    continue
                held = self._annotated_receiver(node.value, func)
                if held is not None:
                    sources.add(held)
            return next(iter(sources)) if len(sources) == 1 else None
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
            ``__init__`` while the write happens in another method -- **and its
            base classes with it**.  PanDA annotates a field where it is
            initialised and consumes it where it is specialised:
            ``self.jobs: List[JobSpec]`` is stated in ``SetupperPluginBase`` and
            iterated in ``setupper_dummy_plugin``, and
            ``self.outDatasetSpecList: List[JediDatasetSpec]`` is stated in
            ``TaskRefinerBase`` and iterated in three refiners.  Stopping at the
            enclosing class left three correct annotations doing nothing, which
            ``annotations_are_read`` reported.
        ``dataset_map`` (a bare name)
            A local or a parameter, so **only the enclosing function**.

        Following the bases is not the deleted naming heuristic: inheritance is
        already how ``self`` writes resolve at all (see
        :meth:`_declaring_ancestor`), so this reads the same relationship the
        language guarantees rather than matching a name across unrelated code.
        Searching the *module* is what was measured and deleted -- it showed up
        immediately: ``apply_sub_workflow_outputs`` took its type from a
        same-named parameter of ``_check_all_inputs_of_step`` five hundred lines
        away.  The answer happened to be right, which is worse than being wrong
        -- it is a guess that looks like a reading.
        """
        if isinstance(expression, ast.Name):
            named, scopes = expression.id, [func]
        elif isinstance(expression, ast.Attribute) and _rooted_at_self(expression):
            named = expression.attr
            owner = self._enclosing_class(func)
            scopes = self._class_and_ancestors(owner) if owner is not None else []
        else:
            return None
        for scope in scopes:
            if scope is None:
                continue
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

    def _stored_element_class(
        self,
        container: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
        enclosing_class: Optional[str],
    ) -> Optional[str]:
        """The element class a container gets from what the code stores in it.

        ``row_id_spec_map[fileSpec.row_ID] = fileSpec`` states the value type as
        plainly as an annotation would, and one line closer to the read.  That
        matters because the alternative was asking PanDA to annotate a *local*,
        and a local's evidence is never anywhere else -- which is how a wrong
        element type got in and sat there resolving nothing.  Of eighteen
        container element annotations in the corpus, seventeen are on parameters
        or fields, where the value really does arrive from another scope.  One
        was on a local, and it is this one.

        **Ranked below the annotation on purpose.**  ``container_annotations_agree``
        compares the stated element type against this reading, so preferring
        this one would leave the gate comparing the attribution against itself
        -- the tautology that sank attribution by declared vocabulary.  Here it
        only fills the annotation's silence.

        Reported as ``container``: the assumption is that the container is
        homogeneous, exactly as for the adder idiom, and a container visibly
        holding two classes settles nothing.

        Measured on 36 sites before being trusted -- 32 locals and 4 fields,
        29 of them one block of ``fileSpecMap`` writes in
        ``task_complex_module`` -- where it agreed with the existing answer 36
        times and disagreed none.
        """
        if func is None:
            return None
        if isinstance(container, ast.Name):
            name = container.id
            scope: Scope = [(func, enclosing_class)]
        elif _rooted_at_self(container):
            name = container.attr
            owner = self._enclosing_class(func)
            if owner is None:
                return None
            # The same scope the annotation lookup uses, for the same reason:
            # one field of one class, wherever its methods happen to touch it.
            scope = [
                pair
                for cls in self._class_and_ancestors(owner)
                for pair in functions_with_owner(cls, cls.name)
            ]
        else:
            return None
        found = _classes_put_in(name, scope, self)
        return next(iter(found)) if len(found) == 1 else None

    @staticmethod
    def _enclosing_class(
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> Optional[ast.ClassDef]:
        """The class *func* is a method of, walked back through parent links.

        Depends on :func:`attach_parents` having run, which the recognizer does
        and the element-type pass does not -- so a reading built on this
        degrades to "cannot tell" in the earlier pass rather than answering
        wrongly.  Three readings ask this same question (a field's annotation,
        a field's stored element type, a method's yield type), and three copies
        of the walk would drift the moment one of them grew a case.
        """
        owner: Optional[ast.AST] = func
        while owner is not None and not isinstance(owner, ast.ClassDef):
            owner = getattr(owner, "parent", None)
        return owner

    def _class_and_ancestors(self, cls: ast.ClassDef) -> list[ast.ClassDef]:
        """*cls* then its base classes, nearest first, as far as the corpus goes."""
        order: list[ast.ClassDef] = [cls]
        seen = {cls.name}
        queue = list(self._bases.get(cls.name, ()))
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            node = self._class_nodes.get(name)
            if node is not None:
                order.append(node)
                queue.extend(self._bases.get(name, ()))
        return order

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
        annotation = _unquoted(annotation)
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

        **Not every subscripted annotation in a field position is a container.**
        PanDA declares its columns through a descriptor generic --
        ``PandaID: SpecColumn[int]`` -- and 160 of the 418 class-scope
        annotations on the declaring classes have that shape.  ``self.PandaID``
        *is* an int rather than a collection of them, so if a column ever held a
        spec class the last argument would name the right class for the wrong
        reason.  It stays harmless for a structural reason rather than a lucky
        one: this reading is consulted only where something takes an element
        *out of* the annotated thing, and a descriptor field holding one spec is
        never subscripted or iterated.  Measured at zero such columns today, so
        there is nothing to special-case -- recorded because the next reader
        would otherwise have to rediscover why it is safe.

        **A tuple is a record, not a container**, so the last-argument rule does
        not apply to it and it is refused rather than descended into.  The
        corpus has four annotations naming a spec inside one, and taking the
        last argument answered two of them wrongly::

            failedRet: tuple[bool, JediDatasetSpec | None, JediFileSpec | None]

        ``failedRet`` is a three-tuple; it is not a ``JediFileSpec`` and it does
        not hold ``JediFileSpec``\\ s.  The spec names describe the *members* of
        a record, which is the same reason ``Callable`` is refused: the class
        named is not the class of the annotated thing.  A homogeneous
        ``tuple[X, ...]`` would be a real container, and there are none, so
        nothing is built for it.
        """
        annotation = _unquoted(annotation)
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
        if name in _RECORD_WRAPPERS:
            return None
        if name == "Optional":
            return self._annotated_element(annotation.slice)
        inner = annotation.slice
        parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
        return self._annotated_class(parts[-1]) if parts else None

    def _yielded_class(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Optional[str]:
        """The spec class a context manager declares it yields.

        ``@contextmanager`` is required, and it is the declaration that makes
        this reading sound rather than a shape being taken for a fact.  ``as``
        binds what ``__enter__()`` returns, which a plain generator has not
        got: an undecorated ``-> Iterator[X]`` entered by a ``with`` describes
        code that cannot run, and answering ``X`` for it would be a confident
        wrong answer about a bug.

        The yield type is the **first** argument, unlike the container reading
        in :meth:`_annotated_element`, which takes the last so that
        ``Dict[K, V]`` gives its values.  Sharing that would read
        ``Generator[WorkflowSpec, None, None]`` as ``None`` -- a silent no-op
        on the fuller spelling of the same annotation.
        """
        decorators = {
            node.id if isinstance(node, ast.Name) else getattr(node, "attr", "")
            for node in method.decorator_list
        }
        if not decorators & {"contextmanager", "asynccontextmanager"}:
            return None
        annotation = method.returns
        if not isinstance(annotation, ast.Subscript):
            return None
        base = annotation.value
        wrapper = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if wrapper not in {"Iterator", "Generator", "AsyncIterator", "AsyncGenerator"}:
            return None
        inner = annotation.slice
        parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
        return self._annotated_class(parts[0]) if parts else None

    def _yielded_local_class(
        self,
        variable: str,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Optional[str]:
        """The class a ``with ... as`` binding takes from the lock it enters.

        ``workflow_core`` cancels a workflow, a step and a data entry the same
        way::

            with self.workflow_lock(workflow_id) as workflow_spec:
                ...
                workflow_spec.status = WorkflowStatus.cancelled

        and nothing between the annotation and the write names a class: the
        attributes touched (``status``, ``end_time``, ``workflow_id``) are
        declared by all three workflow specs, so structural inference separates
        the step -- which also touches ``flavor`` and ``member_id`` -- and
        leaves the other two open.  These were the last unresolved writes in
        the corpus.

        Restricted to ``self.<method>()``, which is the same one-hop-into-a
        -same-class-helper rule that path conditions and return aliases already
        use, and here it is not a restriction that costs anything: of the
        corpus's ``with ... as <name>`` bindings, fourteen call a method on
        ``self`` and every one of those fourteen enters a context manager
        declared in that same class.  The rest are ``open()``, executors and
        cursors, which hold no spec.
        """
        owner = self._enclosing_class(func)
        if owner is None:
            return None
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for cls in self._class_and_ancestors(owner):
            for item in cls.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.setdefault(item.name, item)
        for node in ast.walk(func):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for entry in node.items:
                target = entry.optional_vars
                call = entry.context_expr
                if not (isinstance(target, ast.Name) and target.id == variable):
                    continue
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                ):
                    continue
                method = methods.get(call.func.attr)
                yielded = self._yielded_class(method) if method is not None else None
                if yielded is not None:
                    return yielded
        return None

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
        # Still an annotation, one indirection away: the class is stated on the
        # context manager the ``with`` enters rather than on the name it binds.
        yielded = self._yielded_local_class(variable, func)
        if yielded is not None:
            return yielded
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
            ):
                continue
            copied = value.args[0]
            # ``copy.copy(row_id_spec_map[row])`` -- the original is an element
            # of a typed container rather than a name, and the chain stopped
            # here.  Two writes in ``create_pseudo_files_for_dyn_num_events``
            # were unattributed *with an annotation on the mapping right above
            # them*, because nothing looked past the ``copy``.
            if (opened := self._container_read(copied)) is not None:
                element = self._annotated_receiver(copied, func)
                if element is None:
                    element = self._stored_element_class(opened, func, enclosing_class)
                if element is not None:
                    return element
                continue
            if not isinstance(copied, ast.Name):
                continue
            source = copied.id
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
        seen: Optional[frozenset[str]] = None,
    ) -> Optional[str]:
        """Return the class of a variable that iterates a typed container."""
        if func is None:
            return None
        seen = seen or frozenset()
        if variable in seen:
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
            if holder is None and isinstance(node.iter.value, ast.Name):
                # Nested loops, and the outer one already answered this.
                # ``class_of`` does not consult the loop rule, so a container
                # held by a loop variable was unreadable even with the outer
                # container annotated -- ``for job in jobs`` then
                # ``for file in job.Files`` in ``update_failed_jobs``.  The two
                # entry points disagreed about which readings exist, which is
                # the same divergence this design keeps finding elsewhere.
                holder = self._iterated_element_class(
                    node.iter.value.id, func, enclosing_class, seen | {variable}
                )
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

    def declared_classes(self) -> frozenset[str]:
        """Return every class the corpus declares columns for."""
        return frozenset(self._declarations)

    def class_and_subclasses(self, cls: str) -> frozenset[str]:
        """Return *cls* together with every class in the corpus deriving from it."""
        family = {cls}
        while True:
            grown = family | {
                name
                for name, bases in self._bases.items()
                if family.intersection(bases)
            }
            if grown == family:
                return frozenset(family)
            family = grown

    def element_annotation(self, annotation: ast.expr) -> Optional[str]:
        """Public reading of a container annotation's element type."""
        return self._annotated_element(annotation)

    def stated_annotation(self, annotation: ast.expr) -> Optional[str]:
        """Public reading of an annotation that names a class outright."""
        return self._annotated_class(annotation)

    def yield_annotation(
        self, method: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Optional[str]:
        """Public reading of a context manager's declared yield type."""
        return self._yielded_class(method)

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

        # What the code stores in the container, where nothing annotated it.
        # Strictly after the annotation, so that the gate comparing the two has
        # something independent to compare -- see ``_stored_element_class``.
        if (opened := self._container_read(target.value)) is not None:
            stored = self._stored_element_class(opened, func, enclosing_class)
            if stored is not None and self._declares(stored, attribute):
                return stored, CONTAINER

        # Structural inference: what the code does with the object.  Measured
        # against the writes the code states outright, the two never disagreed
        # (205 of 205), which is what makes the gate in ``gates`` worth having.
        # It is the last reading because it is the last one that is evidence --
        # what remains after it would be a guess from the variable's name.
        structural = self.structural_class(target, func, enclosing_class)
        if structural is not None and self._declares(structural, attribute):
            return structural, STRUCTURAL

        return None, UNRESOLVED


# --------------------------------------------------------------------------- #
# auditing the annotations the map asked for
# --------------------------------------------------------------------------- #
#
# A bare ``Dict`` says nothing about what it holds, which is the whole reason
# the annotation policy asks the target system for ``Dict[str, DatasetSpec]``.
# That makes the element type a fact the map depends on and cannot check the way
# it checks the rest -- and one went in wrong.
#
#     row_id_spec_map: Dict[int, JediFileSpec] = {}
#     for fileSpec in job_spec.Files:              # JobSpec.Files holds FileSpec
#         row_id_spec_map[fileSpec.row_ID] = fileSpec
#
# ``FileSpec`` declares ``row_ID`` and ``JediFileSpec`` does not, so the line
# below the annotation contradicts it.  ``structural_attribution_agrees`` could
# not see it: that gate compares two readings of *one expression*, and here the
# annotation is on the mapping while the attribute that tells the classes apart
# is touched on the loop variable.  The writes the annotation resolves --
# ``.fileID`` and ``.attemptNr`` -- are declared by both classes, so no check of
# the write itself distinguishes them either.
#
# Two readings do exist, one expression apart, and comparing them is the same
# move as everywhere else in this design:
#
#   stated  the element type the annotation declares
#   put in  the class of what the code actually stores in that container
#
# The second reading is deliberately *not* used to resolve writes.  Measured, it
# settles nothing the annotation does not already settle, and inferring element
# types from assignment would put 301 containers in scope for a mechanism whose
# harvest is zero -- the bar that deleted naming inference. It earns its place
# as a check and only as a check.


def _consumed_annotations(
    module: SourceModule,
) -> Iterator[tuple[ast.AST, ast.expr, str, str, bool]]:
    """Every annotation position the reader actually consults.

    Yields ``(node, annotation, kind, name, on_self)``.  ``on_self`` says which
    scope both the audit and the legibility census have to look in: an object's
    own field is annotated where it is initialised and used wherever a method
    needs it, while a bare name is a local or a parameter and belongs to one
    function.

    Defined once because two passes need the same list and a second copy would
    drift.  The exclusion matters as much as the inclusion: a plain ``->``
    return annotation is **not** here, because the reader deliberately does not
    read one (measured: 29 of the corpus's 493 return annotations name a
    declared spec, too few for the mechanism).  Listing them anyway would make
    the census report a reading that was never attempted -- five heterogeneous
    returns like ``tuple[DataCarouselRequestSpec | None, str | None]`` are
    exactly the shape it would misreport.
    """
    for node in ast.walk(module.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {
                d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                for d in node.decorator_list
            }
            if node.returns is not None and decorators & {
                "contextmanager",
                "asynccontextmanager",
            }:
                yield node, node.returns, "yield", f"{node.name}()", True
            continue
        annotation = getattr(node, "annotation", None)
        if annotation is None:
            continue
        if isinstance(node, ast.arg):
            yield node, annotation, "parameter", node.arg, False
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node, annotation, "name", node.target.id, False
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and _rooted_at_self(node.target)
        ):
            yield node, annotation, "field", node.target.attr, True


def spec_annotation_forms(
    modules: list[SourceModule], attributor: SpecAttributor
) -> dict[str, str]:
    """``file:line`` -> the class the reader got, or ``""`` when it got none.

    The same fact expressed twice, for annotations rather than declarations:
    the annotation's *text* names a class the corpus declares, and the reader
    either read it or did not.  Only the disagreement is interesting, and it
    means one thing -- **the reader does not understand this form**.

    The text side is deliberately crude, matching identifier tokens in the
    unparsed source rather than walking the tree, because a reading that shares
    the reader's notion of shape cannot check it.  A quoted name, an unknown
    wrapper and a form nobody has written yet all tokenise the same.

    This is the fourth time one form of one fact has drifted out of reach --
    ``attributes`` to ``_attributes``, then ``attributes_with_types``, then the
    annotated declaration, and quoted annotations alongside.  Each was found by
    reading source, which is not a mechanism.  ``spec_declarations_are_read``
    counts what came back for declarations; this counts what came back for
    annotations, so the fifth form reports itself.

    ``Callable`` and ``Tuple`` are excluded wherever they appear.  Both name
    the members of a record -- a signature's arguments, a tuple's positions --
    so a spec inside one is not the class of the annotated object and the
    reader is right to say nothing; a finding there would be a false one.
    ``Callable`` was excluded on meaning alone, with no instance in the corpus.
    ``Tuple`` was not, and the corpus corrected that: this gate reported
    ``dict[int, list[tuple[JediTaskSpec, str, InputChunk]]]``, and looking at
    the other three showed the reader answering ``failedRet: tuple[bool,
    JediDatasetSpec | None, JediFileSpec | None]`` with ``JediFileSpec`` --
    unread was the honest half, and the two it *had* read were wrong.
    """
    declared = attributor.declared_classes()
    forms: dict[str, str] = {}
    for module in modules:
        for node, annotation, kind, _name, _on_self in _consumed_annotations(module):
            try:
                text = ast.unparse(annotation)
            except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
                continue
            if _RECORD_WRAPPERS & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)):
                continue
            if not declared & set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)):
                continue
            read = (
                attributor.yield_annotation(node)
                if kind == "yield" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else attributor.stated_annotation(annotation)
                or attributor.element_annotation(annotation)
            )
            forms[f"{module.rel_path}:{node.lineno}"] = read or ""
    return forms


class AnnotationReading(NamedTuple):
    """What one trusted annotation states, and what the code does."""

    where: str  # "file:line"
    kind: str  # "container" (an element type) or "yield" (a context manager's)
    container: str  # the annotated name, for the report
    stated: str  # the spec class the annotation names
    put_in: frozenset[str]  # classes the code stores in it, where they resolve
    read: bool  # whether removing it would change any write in scope


Scope = list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Optional[str]]]


def _trusted_annotations(
    module: SourceModule,
    attributor: SpecAttributor,
    specs: frozenset[str],
    corpus: Scope,
) -> list[tuple[ast.AST, str, str, str, Scope]]:
    """Every annotation in *module* naming a spec class nothing else states.

    Yields ``(node, kind, name, stated_class, scope)``.  The scope is where
    both readings have to be taken, and it differs by annotation shape: a bare
    name is a local or a parameter, so its scope is the one function, while
    ``self.<field>`` is the object's own -- annotated in ``__init__`` and used
    in whatever method needs it, the shape ``closer`` and ``adder_atlas_plugin``
    both use.  A context manager's yield type has the same scope as a field:
    the method is entered from wherever the class or a subclass needs the lock.

    For a field the scope reaches into subclasses, and it has to.  PanDA
    annotates in the base and consumes in the derived class:
    ``self.jobs: List[JobSpec]`` is stated in ``SetupperPluginBase`` and
    iterated in ``setupper_dummy_plugin``, and ``self.outDatasetSpecList`` is
    stated in ``TaskRefinerBase`` and iterated in three refiners.  Scoping to
    the annotating class would call those unread without having looked where
    they are used, which is a verdict the audit has not earned.
    """
    found: list[tuple[ast.AST, str, str, str, Scope]] = []
    local: Scope = list(functions_with_owner(module.tree))
    contains = {id(func): set(map(id, ast.walk(func))) for func, _ in local}

    def scope_of(node: ast.AST, on_self: bool) -> Scope:
        holder = next(((f, o) for f, o in local if id(node) in contains[id(f)]), None)
        if holder is None:
            return []
        if not on_self:
            return [holder]
        owner = holder[1]
        if owner is None:
            return [holder]
        family = attributor.class_and_subclasses(owner)
        return [(f, o) for f, o in corpus if o in family]

    for node, annotation, kind, name, on_self in _consumed_annotations(module):
        if kind == "yield":
            yielded = attributor.yield_annotation(node)
            if yielded is not None and yielded in specs:
                found.append((node, kind, name, yielded, scope_of(node, on_self)))
            continue
        stated = attributor.element_annotation(annotation)
        # A plain ``x: JobSpec`` is not what this audits: the class is stated
        # outright and the single-declaring-class rule usually settles those
        # writes without it, so "unread" would be true and mean nothing.
        if stated is None or stated not in specs:
            continue
        if attributor.stated_annotation(annotation) is not None:
            continue
        found.append((node, "container", name, stated, scope_of(node, on_self)))
    return found


def annotation_readings(
    modules: list[SourceModule],
    attributor: SpecAttributor,
    spec_attribute_names: set[str],
) -> list[AnnotationReading]:
    """Read every trusted annotation twice and report both readings."""
    specs = attributor.declared_classes()
    readings: list[AnnotationReading] = []

    for module in modules:
        # Structural inference reads an expression's ancestors, and this pass
        # can run before or after the recognizer that attaches them.  Attaching
        # is idempotent, so doing it here makes the audit independent of order
        # rather than quietly weaker when it runs first.
        attach_parents(module.tree)
    corpus: Scope = [
        pair for module in modules for pair in functions_with_owner(module.tree)
    ]

    for module in modules:
        for node, kind, name, stated, scope in _trusted_annotations(
            module, attributor, specs, corpus
        ):
            used = (
                _manager_is_entered(name.removesuffix("()"), scope)
                if kind == "yield"
                else _container_is_opened(name, scope)
            )
            if not scope or not used:
                continue
            readings.append(
                AnnotationReading(
                    where=f"{module.rel_path}:{node.lineno}",
                    kind=kind,
                    container=name,
                    stated=stated,
                    # Only a container has a second reading here.  A yield type
                    # is corroborated, when it is at all, by the object's usage
                    # -- which ``structural_attribution_agrees`` compares, and
                    # which for two of PanDA's three workflow locks says
                    # nothing, since the attributes touched are declared by
                    # every workflow spec.
                    put_in=(
                        _classes_put_in(name, scope, attributor)
                        if kind == "container"
                        else frozenset()
                    ),
                    read=_annotation_changes_a_write(
                        node, scope, attributor, spec_attribute_names
                    ),
                )
            )
    return readings


def _manager_is_entered(method: str, scope: Scope) -> bool:
    """Whether anything in *scope* enters ``self.<method>()`` and binds it.

    The counterpart of :func:`_container_is_opened`, and there for the same
    reason: a context manager annotated but never entered with an ``as`` target
    in scope states its yield type for whoever enters it elsewhere, so the map
    cannot depend on that annotation here and "unread" would be a true
    statement about nothing.
    """
    for func, _owner in scope:
        for node in ast.walk(func):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for entry in node.items:
                call = entry.context_expr
                if (
                    isinstance(entry.optional_vars, ast.Name)
                    and isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == method
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                ):
                    return True
    return False


def _container_is_opened(name: str, scope: Scope) -> bool:
    """Whether anything in *scope* takes an element out of the container.

    A container that is only passed along states its element type for the
    callee, not for anything here, so the map cannot depend on that annotation
    in this scope and "unread" would be a true statement about nothing.
    ``process_step_pending(self, step_spec, data_spec_map: Dict[str, WFDataSpec])``
    is that shape: PanDA wrote it for the type checker and the parameter is
    handed on without being opened.

    An annotation on a container the code *does* open is a different matter --
    the map either reads it or is missing the read form, and both are worth
    knowing.  Narrowing to those is what keeps the gate about the extraction.
    """
    for func, _owner in scope:
        for node in ast.walk(func):
            if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(
                node.iter, (ast.Name, ast.Attribute)
            ):
                iterated = (
                    node.iter.id
                    if isinstance(node.iter, ast.Name)
                    else node.iter.attr
                )
                if iterated == name:
                    return True
            opened = SpecAttributor._container_read(node) if isinstance(node, ast.expr) else None
            if opened is None:
                continue
            held = (
                opened.id
                if isinstance(opened, ast.Name)
                else getattr(opened, "attr", None)
            )
            if held == name:
                return True
    return False


def _classes_put_in(
    name: str,
    scope: Scope,
    attributor: SpecAttributor,
) -> frozenset[str]:
    """The spec classes the code stores in the container called *name*.

    Only values whose class resolves on their own count.  An unresolvable value
    is not evidence against the annotation -- it is the case the annotation was
    asked for.
    """
    found: set[str] = set()
    for func, owner in scope:
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Subscript):
                continue
            base = target.value
            if isinstance(base, ast.Name):
                holder = base.id
            elif _rooted_at_self(base):
                holder = base.attr
            else:
                continue
            if holder != name:
                continue
            stored = attributor.class_of(node.value, func, owner)
            if stored is not None:
                found.add(stored)
    return frozenset(found)


def _annotation_changes_a_write(
    node: ast.AST,
    scope: Scope,
    attributor: SpecAttributor,
    spec_attribute_names: set[str],
) -> bool:
    """Whether any write in *scope* resolves differently without the annotation.

    Ablation rather than a proxy, because the annotation is often read
    *transitively*: ``jobs: List[JobSpec]`` types ``job``, which types ``file``
    through ``JobSpec.Files``, and the write that lands is ``FileSpec.status``.
    Asking "is some write attributed to the class the annotation names" would
    call that annotation unread while it is doing all the work.
    """
    before = _scope_resolutions(scope, attributor, spec_attribute_names)
    if not before:
        # Nothing in scope for the annotation to affect, so there is nothing to
        # claim either way; treated as read so the gate stays quiet.
        return True
    # A function states the class in ``returns``; everything else in
    # ``annotation``.  The substitute differs with it: a bare container name is
    # exactly the annotation this policy calls uninformative, while for a yield
    # type the uninformative state is having none, which is how all eight of
    # PanDA's context managers were written before it was asked for.
    field = "returns" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "annotation"
    saved = getattr(node, field)
    setattr(
        node,
        field,
        None if field == "returns" else ast.Name(id="dict", ctx=ast.Load()),
    )
    try:
        after = _scope_resolutions(scope, attributor, spec_attribute_names)
    finally:
        setattr(node, field, saved)
    return before != after


def _scope_resolutions(
    scope: Scope,
    attributor: SpecAttributor,
    spec_attribute_names: set[str],
) -> dict[tuple[int, str], tuple[Optional[str], str]]:
    """``(line, attribute) -> (class, basis)`` for every spec write in *scope*.

    The basis is part of the answer, not decoration.  Several of these
    annotations turn out to name a class structural inference reaches anyway,
    so comparing classes alone calls them unread while they are in fact moving
    the write from ``structural`` to ``certain`` -- and every such pair is one
    more thing ``structural_attribution_agrees`` gets to check.
    """
    out: dict[tuple[int, str], tuple[Optional[str], str]] = {}
    for func, owner in scope:
        for target in ast.walk(func):
            if not isinstance(target, ast.Attribute) or not isinstance(
                target.ctx, ast.Store
            ):
                continue
            if target.attr not in spec_attribute_names:
                continue
            cls, basis = attributor.attribute_write(
                target, func=func, enclosing_class=owner
            )
            out[(target.lineno, target.attr)] = (cls, basis)
    return out
