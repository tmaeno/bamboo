"""What makes a junction run, and whether missing it repairs itself.

The map's flagship question -- "this task is stuck, will it come back on its
own?" -- is not answered by the branch table.  It is answered by how the code
that writes the value gets started.  Three ways, and they fail differently:

``polled``
    A ``while True`` loop with a sleep in it, or a module under
    ``daemons/scripts`` that the daemon master runs on a cycle.  Whatever it
    misses it re-evaluates next time round, so a stall here is a delay.
``command``
    Work arrives as a row in a table another system owns.  The row is the
    evidence -- it is either there or it is not -- but it is consumed once, so
    a command that was rejected or overwritten does not come back.
``message``
    A plugin on a broker queue, recognised by its base class.  Consumed once
    *and* leaving nothing behind: the consumer cannot tell a message that was
    never published from one it never received, which is why this is the only
    trigger whose failure the map can point at but not confirm.
``request``
    An HTTP endpoint, recognised by the same ``@request_validation`` decorator
    the boundary slice reads.  Not in the plan's four kinds, but it is how a
    user's kill or a client's submission starts work, it is consumed once, and
    it is the most strongly declared entry in the corpus -- leaving it out put
    every ``db_proxy_mods`` method that only the API reaches at "nothing starts
    this", which is false.

The plan this implements splits ``polled`` into ``state`` (a scope query
re-evaluated each cycle) and ``periodic`` (a timer sweep).  **That split is not
made here, because the source does not draw it**: every JEDI knight is a
``while True`` loop that sleeps on a configured ``loopCycle`` and then runs a
query, ``WatchDog`` included.  The distinction the plan needs it for -- next
cycle versus eventually -- is a property of the query, not of the trigger, and
inventing a marker for it would put a guess where the code says nothing.  The
axis that does the diagnostic work, self-repairing versus consumed-once, is
kept exactly.

Reach is resolved by method name.  ``self.taskBufferIF.updateTaskStatus...``
names the ``db_proxy_mods`` method that owns the junction, and the facade in
between (``JediTaskBuffer``, then ``JediTaskBufferInterface.__getattr__``)
forwards under the same name -- so the name *is* the edge, and no call graph
has to be built to follow it.  One hop is where it stops: that covers a knight
or a message processor reaching a proxy method, which is the shape that
matters, and going further would compound a name match into a claim.
"""

from __future__ import annotations

import ast
from typing import Collection, Iterator, Optional

from bamboo.codemap.models import (
    ARRIVES_BY_CALL,
    ARRIVES_BY_DISPATCH,
    ARRIVES_THROUGH_DOOR,
    COMMAND,
    MESSAGE,
    POLLED,
    REQUEST,
    SELF_REPAIRING_TRIGGERS,
    DispatchFanout,
    EntryPoint,
    JunctionNode,
    SourceModule,
)
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.attribution import annotated_class, class_bases
from bamboo.codemap.panda.pathcond import (
    functions_with_owner,
    targets_of,
    written_value,
)
from bamboo.codemap.panda.recognizers.boundary import endpoint_decorator

# The trigger vocabulary is imported rather than declared here.  It lives with
# the field that carries it because both halves of the map read it: this one to
# label an entry, and an investigation to ask whether a stalled value will come
# back on its own -- the same distinction seen from the other end.

# A message-processing plugin says so in its base class.
_MESSAGE_BASE = "MsgProc"

# Modules the daemon master runs on a cycle.  Placement is the declaration
# here: the schedule itself lives in ``panda_server.cfg``, which is deployment
# configuration and not in the source at all, so the interval cannot be read --
# only the fact that something runs this on a timer.
_DAEMON_PATH = "/daemons/scripts/"


def _is_message_plugin(module: SourceModule) -> bool:
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if _MESSAGE_BASE in (name or ""):
                return True
    return False


def _is_polled(module: SourceModule) -> bool:
    if _DAEMON_PATH in f"/{module.rel_path}":
        return True
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.While):
            continue
        if not (isinstance(node.test, ast.Constant) and node.test.value is True):
            # A conditional loop terminates, so sleeping in it is a retry or a
            # wait, not a cycle.  Accepting any sleeping loop matched 33 modules
            # including ``Interaction`` and ``ddm``, which start nothing.
            continue
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "sleep"
            for call in ast.walk(node)
        ):
            return True
    return False


def foreign_readers(
    modules: list[SourceModule], foreign_tables: set[str]
) -> set[str]:
    """Return the method names that read a table another system writes.

    Their callers are receiving work rather than finding it, which is what
    makes the loss of one permanent.  Derived from the shared-table boundaries
    rather than from a list of table names, so a new channel is picked up by
    adding the schema and nothing else.
    """
    found: set[str] = set()
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            for run in sql.executions(func):
                if any(table in foreign_tables for table, _cols in sql.reads(run.sql)):
                    found.add(func.name)
    return found


def _calls_by_name(module: SourceModule) -> dict[str, ast.Call]:
    """Return ``{method name: first call}`` for the attribute calls in *module*."""
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.setdefault(node.func.attr, node)
    return calls


#: How both facades reach the implementation.  ``TaskBuffer`` and
#: ``JediTaskBuffer`` borrow a connection and call through it, so a name bound
#: from here is the proxy, not a collaborator.
_POOL = "proxyPool"


def _rooted_at_pool(node: ast.expr) -> bool:
    """True for ``self.proxyPool.get()`` and friends, however deep."""
    while isinstance(node, (ast.Call, ast.Attribute)):
        if isinstance(node, ast.Call):
            node = node.func
        else:
            if node.attr == _POOL:
                return True
            node = node.value
    return False


def _pool_bindings(func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Names *func* binds to a borrowed proxy.

    Both spellings are in the corpus: ``with self.proxyPool.get() as proxy``
    (JediTaskBuffer) and ``proxy = self.proxyPool.getProxy()`` (TaskBuffer).
    """
    bound: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name) and _rooted_at_pool(
                    item.context_expr
                ):
                    bound.add(item.optional_vars.id)
        elif (borrowed := written_value(node)) is not None and _rooted_at_pool(borrowed):
            bound.update(t.id for t in targets_of(node) if isinstance(t, ast.Name))
    return frozenset(bound)


def forwarder_classes(modules: list[SourceModule]) -> dict[str, tuple[str, int]]:
    """Return ``{class: (parameter, its position)}`` for the call forwarders.

    A class whose ``__init__`` keeps a parameter on ``self`` and whose other
    methods *call* it is not holding data, it is holding a callee:
    ``TimedMethod(global_task_buffer.updateJobStatus, timeout).run(...)`` runs
    ``updateJobStatus``, and because the method arrives as a value rather than
    a call, every name-based reading walks straight past it.  Thirteen sites
    in ``api/v1`` cross this way, and ``updateJobStatus`` -- the junction the
    pilot's ``holding`` goes through -- reported that no log file named it.

    **Read from the declaration, not from a list of names.**  Naming
    ``TimedMethod`` here would be a constant that stops being true, and the
    weaker rule "a bare attribute passed as an argument" is far too wide: the
    corpus passes one 432 times, and the receivers are ``len``, ``str``,
    ``int``, ``hasattr``, ``getattr`` and ``LogWrapper``.  Requiring the class
    to *call* what it stored separates them without a list -- measured over
    the corpus, exactly two classes qualify, ``TimedMethod.method`` and
    ``CachedObject.updateFunc``, and ``LogWrapper``'s nineteen sites do not.

    A name two modules declare is dropped rather than chosen between, the same
    restriction :func:`worker_classes` puts on a dispatched worker: the
    construction site names the class and nothing else.
    """
    found: dict[str, list[tuple[str, int]]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            held = _held_callables(node)
            if held:
                found.setdefault(node.name, []).append(held)
    return {name: places[0] for name, places in found.items() if len(places) == 1}


def _held_callables(cls: ast.ClassDef) -> Optional[tuple[str, int]]:
    """``(parameter, position)`` for the one callee *cls* stores and calls.

    One, not several: two forwarded fields would make a construction site
    ambiguous about which argument is the callee, and the corpus has none.
    """
    init = _initialiser(cls)
    if init is None:
        return None
    parameters = [argument.arg for argument in init.args.args[1:]]
    kept: dict[str, str] = {}
    for node in ast.walk(init):
        targets = targets_of(node)
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Attribute)
            and isinstance(targets[0].value, ast.Name)
            and targets[0].value.id == "self"
            and isinstance(node.value, ast.Name)
            and node.value.id in parameters
        ):
            kept[targets[0].attr] = node.value.id
    called = {
        node.func.attr
        for method in cls.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method is not init
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in kept
    }
    if len(called) != 1:
        return None
    field = next(iter(called))
    parameter = kept[field]
    return parameter, parameters.index(parameter)


def _forwarded_call(
    call: ast.Call, forwarders: dict[str, tuple[str, int]]
) -> Optional[str]:
    """The method name a construction of a forwarder hands over, if any."""
    if not isinstance(call.func, ast.Name):
        return None
    held = forwarders.get(call.func.id)
    if held is None:
        return None
    parameter, position = held
    supplied: Optional[ast.expr] = None
    for keyword in call.keywords:
        if keyword.arg == parameter:
            supplied = keyword.value
    if supplied is None and position < len(call.args):
        supplied = call.args[position]
    if isinstance(supplied, ast.Attribute):
        return supplied.attr
    return None


def _forwarded_sites(
    module: SourceModule, forwarders: dict[str, tuple[str, int]]
) -> list[tuple[tuple[str, ...], str, ast.Call]]:
    """Call sites a forwarder's construction makes on someone else's behalf.

    The call recorded is the one made *on the constructed object*, not the
    construction: ``TimedMethod(...)`` carries the callee and
    ``timed_method.run(job_id, tmp_status, ...)`` carries its arguments, so
    reading the arguments off the constructor would report the timeout as
    what was passed to ``updateJobStatus``.  Where the construction is not
    bound to a name, or nothing is called on it, the construction stands in --
    the edge is still true and only the binding is missing.
    """
    if not forwarders:
        return []
    sites: list[tuple[tuple[str, ...], str, ast.Call]] = []
    for func, _owner in functions_with_owner(module.tree):
        built: list[tuple[Optional[str], str, ast.Call]] = []
        for node in ast.walk(func):
            built_here = written_value(node)
            if isinstance(built_here, ast.Call) and targets_of(node):
                method = _forwarded_call(built_here, forwarders)
                if method:
                    built.extend(
                        (_bound_name(target), method, built_here)
                        for target in targets_of(node)
                    )
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                method = _forwarded_call(node.value, forwarders)
                if method:
                    built.append((None, method, node.value))
        for bound, method, construction in built:
            sites.append(
                (
                    (func.name,),
                    method,
                    _called_on(func, bound, construction) if bound else construction,
                )
            )
    return sites


def _called_on(
    func: ast.FunctionDef | ast.AsyncFunctionDef, bound: str, fallback: ast.Call
) -> ast.Call:
    """The first call made on *bound* inside *func*, or *fallback*."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == bound
        ):
            return node
    return fallback


def _outward_call_sites(
    module: SourceModule,
    forwarders: Optional[dict[str, tuple[str, int]]] = None,
    doors: Optional[dict[str, set[str]]] = None,
) -> list[tuple[tuple[str, ...], str, ast.Call]]:
    """Return ``[(enclosing functions, method, call)]`` for *module*'s real calls.

    The primitive :func:`_outward_calls` and :func:`attach_calls` share, so the
    facade rule below is read once rather than written twice.  Which of them
    needs the enclosing function is the whole difference between the two
    questions: *what does this module call* is right for the door, because the
    door is where an entry hands over its arguments, and wrong for what a
    junction consults -- a knight's two methods consult different things, and
    pooling them under the file is the shared-owner join by another name.

    A tuple of names, not one, because a call inside a nested function is made
    by the outer function too; :func:`_self_calls` walks whole function bodies
    and says the same.  Module-level calls carry the empty tuple.

    **The receiver is what identifies a facade hop, not the name.**  ``TaskBuffer``
    and ``JediTaskBuffer`` borrow a connection from ``self.proxyPool`` and call
    the implementation through it, so a call on a borrowed proxy is the same
    handoff :func:`_forwards_to_itself` already refuses to count as a
    *definition*, refused here as a *call*.  A facade method may forward under a
    different name -- ``JediTaskBuffer.checkWaitingTaskPrio_JEDI`` calls
    ``proxy.getTasksToBeProcessed_JEDI`` -- and conversely a call matching the
    enclosing function's name is usually not a handoff at all:
    ``datasetManager.run`` constructs a ``Closer`` and calls
    ``closer_process.run()``, which is a genuine caller and the only thing that
    puts ``closer.py`` on a daemon cycle.  Keying on the name lost that edge and
    three others.

    Why this matters for the log question in particular: ``JediTaskBuffer``
    declares a logger and then logs twice in the entire file, both in
    ``__init__``.  Naming ``panda-JediTaskBuffer.log`` as the place to look for
    a proxy method would send every query to a file that says nothing, and an
    empty answer there reads as "this code never ran".  It sat on 71 junctions
    before the receiver rule removed it.
    """
    sites: list[tuple[tuple[str, ...], str, ast.Call]] = []
    # Breadth-first, because which call a name resolves to decides the
    # ``arg_binding`` reported for that entry and that answer must not move.
    queue: list[tuple[ast.AST, tuple[str, ...], frozenset[str]]] = [
        (module.tree, (), frozenset())
    ]
    while queue:
        node, enclosing, pooled = queue.pop(0)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner_names = enclosing + (child.name,)
                inner_pooled = _pool_bindings(child)
            else:
                inner_names, inner_pooled = enclosing, pooled
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                receiver = child.func.value
                if not (isinstance(receiver, ast.Name) and receiver.id in pooled):
                    sites.append((enclosing, child.func.attr, child))
                elif doors is not None and enclosing:
                    # The hop this refuses to call a call is still a door, and
                    # *which* door is the one thing a caller of the facade needs
                    # to reach past it.  Recorded only when asked for, so the
                    # log question keeps reading exactly what it read before.
                    doors.setdefault(enclosing[0], set()).add(child.func.attr)
            queue.append((child, inner_names, inner_pooled))
    sites.extend(_forwarded_sites(module, forwarders or {}))
    return sites


def _outward_calls(
    module: SourceModule,
    forwarders: Optional[dict[str, tuple[str, int]]] = None,
    doors: Optional[dict[str, set[str]]] = None,
) -> dict[str, ast.Call]:
    """Return ``{method name: first call}`` for the calls *module* really makes.

    :func:`_calls_by_name` with the facade hop removed -- see
    :func:`_outward_call_sites`, which does the reading.  What the door needs is
    the module's whole surface and the *first* call under each name, because
    that is the call whose keyword arguments distinguish one entry from another.
    """
    calls: dict[str, ast.Call] = {}
    for _enclosing, method, call in _outward_call_sites(module, forwarders, doors):
        calls.setdefault(method, call)
    return calls


def _forwards_to_itself(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the body just hands the call on under the same name.

    ``TaskBuffer`` and ``JediTaskBuffer`` both define
    ``getTasksToExecCommand_JEDI`` as ``return proxy.getTasksToExecCommand_JEDI(...)``.
    Counting those as definitions would make every proxy method look
    ambiguously defined in three places, which is the opposite of the truth:
    there is one implementation and two doors onto it.

    Any call to the same name on something other than ``self`` counts, not only
    one in a ``return``: several facade methods keep the result in a local and
    return that, and requiring the tighter shape left half of them looking like
    rival implementations.  ``self`` is excluded so that genuine recursion is
    not mistaken for a handoff.
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != func.name:
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Name) and receiver.id == "self":
            continue
        return True
    return False


def definitions(modules: list[SourceModule]) -> dict[str, set[str]]:
    """Return ``{method name: the modules that implement it}``, doors removed.

    The primitive behind :func:`sole_definitions`.  Both restrictions a
    cross-module hop is allowed to rest on need it -- "the name means one thing"
    reads the singletons, "the caller says which module it means" needs the
    rivals as well -- and computing it twice would be two readings of the same
    fact that could drift apart.
    """
    defined: dict[str, set[str]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            if _forwards_to_itself(func):
                continue
            defined.setdefault(func.name, set()).add(module.rel_path)
    return defined


def sole_definitions(modules: list[SourceModule]) -> dict[str, str]:
    """Return ``{method name: the one module that implements it}``.

    The cross-module hop rests on a name match, so it is only allowed where the
    name means one thing.  It does not: ``run`` is defined by every daemon
    script and every message processor, and following it gave one
    ``datasetManager`` junction fourteen entry points, thirteen of them from
    daemons that have never heard of it.  Requiring a single implementation is
    the same "only one declaration" argument the attribute and alias slices
    already turn on, one level up.
    """
    return {
        name: next(iter(where))
        for name, where in definitions(modules).items()
        if len(where) == 1
    }


def imported_modules(module: SourceModule) -> set[str]:
    """Return the ``rel_path``\\ s *module* imports from.

    The second way a cross-module edge can be evidenced, and the one that
    rescues the common doors.  ``add_main`` calls ``.run()``, which twenty
    modules define, but it says ``from pandaserver.dataservice.adder_gen import
    AdderGen`` -- so within that pair the name is unambiguous even though it is
    hopeless across the corpus.
    """
    found: set[str] = set()
    for node in ast.walk(module.tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.replace(".", "/") + ".py")
            found.update(
                f"{node.module.replace('.', '/')}/{alias.name}.py" for alias in node.names
            )
        elif isinstance(node, ast.Import):
            found.update(alias.name.replace(".", "/") + ".py" for alias in node.names)
    return found


def _self_calls(tree: ast.AST) -> dict[str, set[str]]:
    """Return ``{method: the methods it calls on ``self``}`` within *tree*.

    Needed because the door and the write are rarely the same method.
    ``add_main`` starts ``AdderGen.run``; the ``jobStatus`` writes are in
    ``finalize_job_status`` and ``handle_failed_job``, several ``self`` calls
    further in.  Stopping at the door left 16 of ``setupper_atlas_plugin``'s
    junctions and 11 of ``adder_gen``'s looking as though nothing runs them.

    Within one module ``self.<name>()`` is an unambiguous edge -- no resolution
    is involved, which is why this is followed transitively while the hop
    *between* modules, which rests on a name match, is not.

    Takes a tree rather than a module so that a single class can be scoped.
    Two readers need that: ``TaskBroker`` declares two workers and both spell
    their entry ``runImpl``, and ``ThreadUtils`` declares ``ZombieCleaner``
    beside ``WorkerThread`` with a ``run`` of its own, so a file-wide walk
    would pool two classes under one method name.  **Measured, both agree
    today** -- neither pair actually collides in what it calls on ``self`` --
    so this is the shape being read correctly rather than a difference in the
    answer, and the second reader would fail silently if it stopped being.
    """
    edges: dict[str, set[str]] = {}
    for func, _owner in functions_with_owner(tree):
        targets = edges.setdefault(func.name, set())
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                targets.add(node.func.attr)
    return edges


def _ancestors(bases: dict[str, list[str]], cls: str) -> Iterator[str]:
    """Yield every class *cls* derives from, transitively.

    Shared by the two readings that need the hierarchy in this module: which
    file an inherited method lives in, and whether a module declares a class
    the factory's return type names.  One walk, so the two cannot disagree
    about what derives from what.
    """
    seen: set[str] = set()
    queue = list(bases.get(cls, ()))
    while queue:
        base = queue.pop(0)
        if base in seen:
            continue
        seen.add(base)
        yield base
        queue.extend(bases.get(base, ()))


def _inherited_homes(
    modules: list[SourceModule], bases: dict[str, list[str]]
) -> dict[str, dict[str, str]]:
    """Return ``{module: {method: the module a base class defines it in}}``.

    The companion to :func:`_self_calls`, and it rests on the same fact.  That
    function follows ``self.<name>()`` transitively within one module because no
    resolution is involved -- and an *inherited* ``self.<name>()`` is equally
    unresolved: the class states its base, the base states where it lives, and
    the language decides which method runs.  Stopping at the file boundary was
    reading the hierarchy as though it were a name match.

    What it is worth, measured on ``1.0.4-295-gbf2812ba``, as a funnel rather
    than as the one big number -- the shape count says nothing on its own:

    ==========================================================  =====
    ``self.<name>()`` calls naming a method the module lacks      1811
    ``(module, method)`` pairs this index settles                 1372
    pairs :func:`_downstream` actually asks about                  180
    ... of which the index answers                                 154
    **junctions that gain an entry point**                        **4**
    ==========================================================  =====

    Four, and they are the right four: three arms of
    ``base_module.recordStatusChange`` and one of ``setDeftStatus_JEDI``, called
    through ``self`` from the ``job_standalone`` and ``task_complex`` mixins and
    started by ``JobGenerator``, ``TaskCommando``, ``PostProcessor``,
    ``ContentsFeeder`` and two message processors.  They read as "nothing starts
    this", which was false.  A mixin's ``self.foo()`` is the same object at run
    time, so the edge is not an inference about which object -- there is only
    one.  Nothing else moved: no junction gained a second-hand entry, none lost
    one, and no stored node count changed.

    **Unique or nothing**, in two places, so this never degrades into the name
    match the module docstring refuses:

    * a base class declared in more than one file is skipped -- the name would
      be doing the resolving, not the hierarchy;
    * a method two different ancestor modules define is skipped, even though
      Python's MRO would pick one.  Reading the MRO would mean ordering bases
      that ``class_bases`` records unordered, and a confident wrong home is
      worse than the unresolved method it replaces.
    """
    declared_in: dict[str, set[str]] = {}
    defines: dict[str, set[str]] = {}
    declares: dict[str, list[str]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ClassDef):
                declared_in.setdefault(node.name, set()).add(module.rel_path)
                declares.setdefault(module.rel_path, []).append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defines.setdefault(module.rel_path, set()).add(node.name)
    homes: dict[str, dict[str, str]] = {}
    for module in modules:
        here = defines.get(module.rel_path, set())
        found: dict[str, set[str]] = {}
        for cls in declares.get(module.rel_path, ()):
            for ancestor in _ancestors(bases, cls):
                where = declared_in.get(ancestor, set())
                if len(where) != 1:
                    continue
                home = next(iter(where))
                if home == module.rel_path:
                    continue
                for method in defines.get(home, set()) - here:
                    found.setdefault(method, set()).add(home)
        settled = {
            method: next(iter(where))
            for method, where in found.items()
            if len(where) == 1
        }
        if settled:
            homes[module.rel_path] = settled
    return homes


def _declared_family(
    modules: list[SourceModule], bases: dict[str, list[str]]
) -> dict[str, frozenset[str]]:
    """Return ``{module: the classes it declares, and everything those derive from}``.

    The answer to "is this module one of the things that factory hands back?".
    Keyed by module rather than by class because a junction is owned by a file,
    and a plugin file declares the one class it exists to provide.
    """
    family: dict[str, set[str]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            here = family.setdefault(module.rel_path, set())
            here.add(node.name)
            here.update(_ancestors(bases, node.name))
    return {rel: frozenset(names) for rel, names in family.items()}


def _dispatch_families(
    modules: list[SourceModule], bases: dict[str, list[str]]
) -> dict[str, dict[str, str]]:
    """Return ``{module: {method called on a dispatched object: the class declared}}``.

    The third thing a cross-module hop is allowed to rest on, after "the name
    means one thing" and "the caller imports the module it names".  A JEDI
    knight names its plugin in neither: ``WatchDog.start`` calls
    ``impl.doAction()`` on whatever ``self.getImpl`` returned, the plugin is
    loaded from a string in ``jedi_config``, and eight modules define
    ``doAction``.  What the corpus *does* state is the factory's return type,
    and the class hierarchy says which files are in it.

    **Two receiver forms, because that is where the sites are.**
    ``impl = self.getImpl(...)`` resolves against the enclosing class.
    ``impl = self.implFactory.instantiateImpl(...)`` does not: the factory was
    handed to a worker thread and kept as an attribute, so the only statement
    about its type is the annotation on the ``__init__`` parameter it came
    from.  That form carries the larger share of the sites -- the knight's own
    ``getImpl`` accounts for the ``WatchDog`` family alone.

    A name bound twice to different types, or a method the receiver's class and
    its ancestors never annotate, yields nothing: unique or nothing, the same
    rule :func:`_inherited_homes` follows and for the same reason.  An answer
    must also name a class the corpus declares, which is what ``bases`` is
    passed for -- ``FactoryBase.instantiateImpl`` is annotated ``-> Any``, and
    calling ``Any`` the declared family would be stating something untrue even
    where no module declares it and nothing can match.

    **The binding is per function, not per statement order.**  ``ast.walk``
    does not run the function, so a local assigned anywhere in the body is
    taken to be that type throughout it.  Reassigning ``impl`` to a different
    family within one method is a shape the corpus does not have, and the
    alternative -- ordering the walk -- would claim a flow analysis this is
    deliberately not doing.

    What it is worth, measured on ``1.0.4-302-g219ed0af`` as a funnel, because
    the number of sites with the shape says nothing on its own:

    ============================================================  =====
    ``(module, method)`` pairs this index settles as the corpus
    stands today                                                     42
    ... of which move a junction's entry points                       0
    ------------------------------------------------------------  -----
    pairs added by annotating the two families that can be
    annotated (see :func:`reaching_modules`)                          5
    **junctions that gain an entry point**                        **18**
    ============================================================  =====

    The first two lines are the point.  Forty-two receivers already resolve --
    ``ddmIF``, ``siteMapper``, a site spec -- and every one of them was already
    reached by the name or by the import, so this rescue admits nothing new for
    any of them.  Sizing the rule by how often its shape occurs would have
    claimed forty-two and delivered none of them.
    """
    returns: dict[tuple[str, str], str] = {}
    held: dict[tuple[str, str], str] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if owner is None:
                continue
            if func.returns is not None:
                stated_return = annotated_class(func.returns, bases)
                if stated_return is not None:
                    returns[(owner, func.name)] = stated_return
            if func.name != "__init__":
                continue
            arguments = (
                *func.args.posonlyargs,
                *func.args.args,
                *func.args.kwonlyargs,
            )
            stated = {
                arg.arg: named
                for arg in arguments
                if arg.annotation is not None
                and (named := annotated_class(arg.annotation, bases)) is not None
            }
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target, value = node.targets[0], node.value
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(value, ast.Name)
                    and value.id in stated
                ):
                    held[(owner, target.attr)] = stated[value.id]

    def stated_by(
        table: dict[tuple[str, str], str], cls: str, name: str
    ) -> Optional[str]:
        for candidate in (cls, *_ancestors(bases, cls)):
            found = table.get((candidate, name))
            if found is not None:
                return found
        return None

    families: dict[str, dict[str, set[str]]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if owner is None:
                continue
            bound: dict[str, str] = {}
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                target, value = node.targets[0], node.value
                if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)):
                    continue
                if not isinstance(value.func, ast.Attribute):
                    continue
                receiver = value.func.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    factory: Optional[str] = owner
                elif (
                    isinstance(receiver, ast.Attribute)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id == "self"
                ):
                    factory = stated_by(held, owner, receiver.attr)
                else:
                    continue
                if factory is None:
                    continue
                named = stated_by(returns, factory, value.func.attr)
                if named is not None:
                    bound[target.id] = named
            if not bound:
                continue
            for node in ast.walk(func):
                if not (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                ):
                    continue
                receiver = node.func.value
                if isinstance(receiver, ast.Name) and receiver.id in bound:
                    families.setdefault(module.rel_path, {}).setdefault(
                        node.func.attr, set()
                    ).add(bound[receiver.id])
    return {
        rel: {
            method: next(iter(named))
            for method, named in found.items()
            if len(named) == 1
        }
        for rel, found in families.items()
    }


def _downstream(edges: dict[str, set[str]], start: str) -> set[str]:
    """Return every method reachable from *start* through ``self`` calls."""
    seen = {start}
    queue = [start]
    while queue:
        for target in edges.get(queue.pop(), ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def classify(
    modules: list[SourceModule], foreign_tables: set[str]
) -> dict[str, set[str]]:
    """Return ``{module: triggers}`` for the modules that start something."""
    receiving = foreign_readers(modules, foreign_tables)
    triggers: dict[str, set[str]] = {}
    for module in modules:
        kinds: set[str] = set()
        if _is_message_plugin(module):
            kinds.add(MESSAGE)
        if _is_polled(module):
            kinds.add(POLLED)
        if any(endpoint_decorator(func) for func, _o in functions_with_owner(module.tree)):
            kinds.add(REQUEST)
        if kinds and receiving & set(_calls_by_name(module)):
            # It polls, but what it polls for is a row another system wrote.
            # Both are true of ``TaskCommando`` and both matter: it runs again
            # next cycle, and the command it failed to act on does not.
            #
            # Only for a module that already starts something.  Without that,
            # the facade forwarding the read (``TaskBuffer``, ``JediTaskBuffer``,
            # and the ``db_proxy_mods`` file the reader lives in) each counted
            # as a command entry, which made ``command`` the map's most common
            # trigger -- for code that starts nothing at all.
            kinds.add(COMMAND)
        if kinds:
            triggers[module.rel_path] = kinds
    return triggers


def _arg_binding(call: ast.Call) -> dict[str, str]:
    """Return the keyword arguments one entry supplies at a call.

    Keywords only.  The interesting comparison is already there --
    ``JobGenerator`` passes ``minPriority`` and ``maxNumJobs`` to
    ``getTasksToBeProcessed_JEDI`` and the message processor does not, so the
    throttle guard is inert on the message path -- and binding positionals
    would mean trusting that a facade forwards them in order.
    """
    return {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }


def reaching_modules(
    modules: list[SourceModule],
    through_doors: bool = False,
    through_inheritance: bool = False,
    through_dispatch: bool = False,
) -> dict[str, dict[str, list[tuple[str, str, Optional[ast.Call]]]]]:
    """Return ``{module: {method: [(entry module, door, call)]}}`` -- who reaches what.

    Deliberately unfiltered: *every* module is allowed to be an entry here.
    Two different questions read this, and only one of them cares whether the
    caller starts anything:

    * **What makes this run?**  Only a module carrying a trigger answers that,
      so :func:`attach` filters on ``classify`` before building an
      ``EntryPoint``.
    * **Whose log will say it ran?**  Any caller answers that.  The
      ``db_proxy_mods`` mixins declare no logger and the knights that call them
      do, so the file to read belongs to a caller that very often starts
      nothing itself -- ``AtlasProdWatchDog`` is a plugin the WatchDog knight
      drives, ``event_picker`` is driven by a daemon script.

    Folding both behind one filter is what made 11 of the 18 junctions that can
    write ``pending`` look mutually indistinguishable: they all reported
    ``panda-DBProxy.log``, which is where the *code* lives, while production
    writes ``set task_status=`` from the caller.  The filter belongs on the
    trigger question alone, which is why it is not applied here.

    The honesty of the cross-module hop is unchanged and does the work instead:
    a name is followed only where it means one thing (:func:`sole_definitions`)
    or where the entry imports the module it names.

    **A caller in the owner's own module is skipped**, which is why the bucket
    below is named for another module rather than for any caller at all.  The
    question is what starts the owner, and a sibling does not answer it -- it
    moves the question to what starts the sibling, which is the one-hop rule
    again.  It is a deliberate narrowing and not a gap, so the name says so.

    *through_doors* lets a caller of the facade reach what the facade calls.
    ``TaskBuffer.storeJobs`` is the only thing in the corpus that calls
    ``proxy.insertNewJob``, and ``JobGenerator`` and ``api/v1/job_api`` both
    call ``storeJobs`` -- so where a job row is created reported that nothing
    starts it.  A facade method is a *door*, which is what ``EntryPoint.via``
    already means, so crossing one is not the second hop the docstring above
    refuses: the name is still matched once, against the door.

    **Off by default, because only one of the two questions wants it.**  Whose
    log will say it ran is answered by the immediate caller, and ``JobGenerator``
    is not that for a line ``insertNewJob`` writes.  Turning it on for both
    would move 23 junctions' log files on the strength of a change made for the
    trigger question -- the failure the facade rule exists to prevent, in the
    other direction.

    *through_inheritance* carries the same edge into the module a base class
    lives in (:func:`_inherited_homes`).  ``AtlasProdTaskRefiner.doRefine``
    calls ``self.doBasicRefine``, which ``TaskRefinerBase`` defines in another
    file, and the eleven junctions in ``doBasicRefine`` and ``doPreProRefine``
    read as though nothing ran them.  The method set is recorded under the
    *base's* module, because that is where the junction that needs it is owned.

    **Off by default for the same reason as the doors**, and left off for the
    log question deliberately: the immediate caller answers whose log says it
    ran, and a mixin's file is that caller even when the method it calls lives
    one file up.

    *through_dispatch* adds the third thing the hop may rest on
    (:func:`_dispatch_families`): the entry calls the method on an object whose
    declared type names a class this module derives from.  A knight neither
    imports its plugins nor calls a name only they define, so ``doRefine``,
    ``doPostProcess`` and ``doAction`` read as though nothing ran them -- while
    the factory's return type says exactly which family runs.

    **The corpus does not state those return types yet, so this earns nothing
    here today.**  Measured against a copy of ``1.0.4-302-g219ed0af`` with them
    added: ``WatchDog.getImpl -> WatchDogBase | None`` reaches 3 junctions, and
    ``TaskRefiner.instantiateImpl -> TaskRefinerBase | None`` with
    ``implFactory: TaskRefiner`` on its worker reaches 15 -- 18, none lost, and
    nothing changed for a junction that already had an entry.

    ``PostProcessor``'s 6 are not among them and cannot be, by an annotation:
    ``PostProcessorThread`` is built twice, and
    ``jedi_post_processor_msg_processor`` hands it a bare ``FactoryBase``, so no
    single class names that parameter.  ``FactoryBase(Generic[T])`` would state
    it, and reading that needs a type variable bound at the subscript, which
    this does not do.

    **Off by default, and off for the log question**, which is the same
    reasoning once more: ``AtlasProdWatchDog`` writes its own log file, and the
    knight that dispatched it writes another.
    """
    forwarders = forwarder_classes(modules)
    doors: dict[str, set[str]] = {}
    callers: dict[str, list[tuple[str, Optional[ast.Call], Optional[str]]]] = {}
    for module in modules:
        for name, call in _outward_calls(module, forwarders, doors).items():
            callers.setdefault(name, []).append((module.rel_path, call, None))

    if through_doors:
        for door, forwarded in doors.items():
            for entry, _call, via in list(callers.get(door, ())):
                if via is not None:
                    continue
                for target in forwarded:
                    # No call is carried across.  The keywords at the door
                    # describe the door -- ``storeJobs(jobs, user, fqans=...)``
                    # is not ``insertNewJob(job, user, serNum, ...)`` -- and
                    # reporting them as what this entry handed over would put a
                    # false argument list under a true edge.
                    callers.setdefault(target, []).append((entry, None, door))

    implemented = sole_definitions(modules)
    imports = {module.rel_path: imported_modules(module) for module in modules}
    # Read once and handed to both, so the two readings of the hierarchy cannot
    # come to different conclusions about what derives from what.
    bases = class_bases(modules) if through_inheritance or through_dispatch else {}
    homes = _inherited_homes(modules, bases) if through_inheritance else {}
    dispatched = _dispatch_families(modules, bases) if through_dispatch else {}
    kin = _declared_family(modules, bases) if through_dispatch else {}
    self_calls = {module.rel_path: _self_calls(module.tree) for module in modules}
    inward: dict[str, dict[str, list[tuple[str, str, Optional[ast.Call]]]]] = {}
    for module in modules:
        edges = self_calls[module.rel_path]
        inherited = homes.get(module.rel_path, {})
        for door in edges:
            unambiguous = implemented.get(door) == module.rel_path
            for entry, call, via in callers.get(door, ()):
                if entry == module.rel_path:
                    continue
                if not unambiguous and module.rel_path not in imports[entry]:
                    stated = dispatched.get(entry, {}).get(door)
                    if stated is None or stated not in kin.get(module.rel_path, ()):
                        continue
                for method in _downstream(edges, door):
                    inward.setdefault(module.rel_path, {}).setdefault(method, []).append(
                        (entry, via or door, call)
                    )
                    home = inherited.get(method)
                    if home is None:
                        continue
                    # Inside the base's file the within-one-module rule applies
                    # again, so the inherited method's own ``self`` calls are
                    # followed there -- the same reading, not a second hop.
                    for reached in _downstream(self_calls[home], method):
                        inward.setdefault(home, {}).setdefault(reached, []).append(
                            (entry, via or door, call)
                        )
    return inward


#: The base class a JEDI knight's real work runs under.  Named here for the
#: same reason ``_MESSAGE_BASE`` is: it is the declaration the corpus makes
#: about which objects are dispatched rather than called.
_WORKER_BASE = "WorkerThread"

#: How a constructed worker is set going.  ``threading.Thread.start`` spawns it
#: and ``run`` is the inline spelling two message processors use to avoid an
#: extra thread; both reach the same method, so both count.
_DISPATCH = frozenset({"start", "run"})


def subclass_edges(
    modules: list[SourceModule],
) -> dict[str, list[tuple[ast.ClassDef, str]]]:
    """Return ``{base class name: [(subclass node, its module), ...]}``.

    Keyed by the bare name because that is how a base is written at the point
    of inheritance, qualified or not.  Two classes share the name ``DBProxy``
    -- the server's and JEDI's -- and that collision is load-bearing here
    rather than a nuisance: it is the edge from one to the other.

    The node and not only its name, because two readers want different halves
    of one fact -- which log a mixin's output reaches, and which methods a
    dispatched worker runs -- and walking the tree twice for the same reading
    is how two answers drift apart.
    """
    edges: dict[str, list[tuple[ast.ClassDef, str]]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name:
                    edges.setdefault(name, []).append((node, module.rel_path))
    return edges


def worker_classes(modules: list[SourceModule]) -> dict[str, tuple[str, ast.ClassDef]]:
    """Return ``{class name: (its module, its node)}`` for the dispatched workers.

    Transitive, so a worker reached through an intermediate base is included.
    A name two modules declare is dropped rather than picked between: the
    construction site names the class and nothing else, so an ambiguous name
    would resolve a dispatch by guess -- the restriction
    :func:`sole_definitions` puts on methods, applied to classes.
    """
    edges = subclass_edges(modules)
    seen_bases: set[str] = set()
    frontier = [_WORKER_BASE]
    found: dict[str, list[tuple[str, ast.ClassDef]]] = {}
    while frontier:
        base = frontier.pop()
        if base in seen_bases:
            continue
        seen_bases.add(base)
        for node, where in edges.get(base, ()):
            found.setdefault(node.name, []).append((where, node))
            frontier.append(node.name)
    return {name: places[0] for name, places in found.items() if len(places) == 1}


def worker_door(modules: list[SourceModule]) -> Optional[str]:
    """The method a dispatched worker actually runs, read from the base class.

    ``WorkerThread.run`` calls ``self.runImpl()``, so the corpus states which
    method the dispatch enters and a constant here would only restate it --
    and would go on restating it after the base class changed.  Ambiguity is
    refused: if the base's ``run`` calls more than one of its own methods,
    which one carries the work is a guess.
    """
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ClassDef) and node.name == _WORKER_BASE:
                called = _self_calls(node).get("run", set())
                if len(called) == 1:
                    return next(iter(called))
    return None


def _bound_name(node: ast.expr) -> Optional[str]:
    """``thr`` and ``self.worker``, the two ways a construction is kept."""
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return f"self.{node.attr}"
    return None


def _initialiser(node: ast.ClassDef) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for statement in node.body:
        if (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ):
            return statement
    return None


def _handover(
    call: ast.Call, init: Optional[ast.FunctionDef | ast.AsyncFunctionDef]
) -> dict[str, str]:
    """What the construction site hands the worker, keyed as the body reads it.

    Positional as well as keyword, which the call case refuses.  The reason it
    refuses -- that binding by position means trusting a facade to forward in
    order -- does not apply: the class is resolved, so its own signature is in
    hand and the positions are the ones the constructor declares.

    **Keyed by the field, not the parameter.**  A worker's body says
    ``self.taskList``, and PanDA renames on the way in
    (``self.taskBufferIF = taskbufferIF``), so keying by the parameter would
    make every reader re-derive the constructor's own assignment.  The rule is
    the same one the call case follows -- the name the callee's body uses --
    and a parameter the body never stores is left out rather than reported
    under a name nothing mentions.  ``threadPool`` is that case: it goes to the
    base class and no arm ever asks about it.
    """
    if init is None:
        return {}
    parameters = [arg.arg for arg in (init.args.posonlyargs + init.args.args)[1:]]
    # Not strict: a construction may pass more than the signature declares
    # (``*args``), and the surplus has no name the body reads, so it is dropped
    # rather than reported under a position.
    supplied = {
        name: ast.unparse(value)
        for name, value in zip(parameters, call.args, strict=False)
    }
    supplied.update(
        {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg is not None}
    )
    fields: dict[str, str] = {}
    for node in ast.walk(init):
        targets = targets_of(node)
        if len(targets) != 1:
            continue
        target = targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and isinstance(node.value, ast.Name)
        ):
            fields[node.value.id] = target.attr
    return {
        fields[parameter]: expression
        for parameter, expression in supplied.items()
        if parameter in fields
    }


def worker_uplinks(
    modules: list[SourceModule],
) -> dict[tuple[str, str], list[tuple[str, str, dict[str, str], tuple[int, int]]]]:
    """Return ``{(worker module, method): [(entry, via, handover, class span)]}``.

    The edge no name match can find.  A knight reads its rows, builds a worker
    with them and starts a thread; the work happens in another class, reached
    by a constructor argument and a dispatch rather than by a call, so the
    method-name resolution the rest of this slice turns on walks straight past
    Measured on the installed corpus: thirteen construction sites, eleven of
    them dispatched, against 72 entry points on junctions in the worker
    modules of which four had a ``via`` -- which is to say the map could name
    what runs an arm but not what handed it its input.

    **The dispatch is required, not assumed.**  Two message processors keep a
    constructed worker on ``self`` and call one of its methods directly; that
    edge *is* a call and :func:`reaching_modules` already follows it.  Claiming
    the dispatch for them as well would put the worker's whole entry method
    behind a caller that never runs it.

    The span comes back with each entry because ``owner`` is spelled
    ``module::method`` and says nothing about the class.  ``TaskBroker``
    declares two workers and both call their entry ``runImpl``, so the key
    alone would hand each one the other's callers; the arm's anchor is what
    settles which class the line is in.

    Walked with the whole function body, nested definitions included, matching
    what :func:`_self_calls` already does.  A construction inside a nested
    function would then be credited to the outer one as well; the corpus has
    none, so the looser reading costs nothing and the tighter one would be
    machinery for a case that does not exist.
    """
    door = worker_door(modules)
    if door is None:
        # Reported by the build rather than returned quietly: with no door the
        # whole uplink is off, and an empty result looks exactly like a corpus
        # that dispatches nothing.
        return {}
    workers = worker_classes(modules)
    # Per class, not per site: three sites construct ``JobGeneratorThread`` and
    # the methods it enters are a property of the class.
    entered = {
        name: (where, _downstream(_self_calls(node), door),
               (node.lineno, node.end_lineno or node.lineno), _initialiser(node))
        for name, (where, node) in workers.items()
    }
    reach: dict[tuple[str, str], list[tuple[str, str, dict[str, str], tuple[int, int]]]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            built: list[tuple[Optional[str], ast.Call]] = []
            dispatched: set[str] = set()
            for node in ast.walk(func):
                value = written_value(node)
                if (
                    targets_of(node)
                    and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id in workers
                ):
                    built.extend((_bound_name(t), value) for t in targets_of(node))
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _DISPATCH
                ):
                    name = _bound_name(node.func.value)
                    if name:
                        dispatched.add(name)
            for name, call in built:
                if name is None or name not in dispatched:
                    continue
                where, methods, span, init = entered[call.func.id]
                handover = _handover(call, init)
                for method in methods:
                    reach.setdefault((where, method), []).append(
                        (module.rel_path, func.name, handover, span)
                    )
    return reach


def construction_uplinks(
    modules: list[SourceModule],
    junctions: list[JunctionNode],
    spec_classes: Collection[str],
) -> dict[tuple[str, str], list[tuple[str, str, dict[str, str], tuple[int, int]]]]:
    """The same edge as :func:`worker_uplinks`, keyed on the class alone.

    ``AdderGen(taskBuffer, job_id, job_status, attempt_nr)`` is a crossing --
    the arm that decides a job's status reads ``self.job_status``, and what it
    holds was chosen two modules away, at ``pilot_api:634`` and
    ``add_main:135``.  :func:`worker_uplinks` does not reach it: that one asks
    for a ``WorkerThread`` subclass dispatched in the same function, and
    ``AdderGen`` is neither.

    **Both restrictions come off and one takes their place: the class owns a
    junction.**  That is what a reverse index is for -- the edge cannot be
    followed forwards, so it has to be built by scanning, and the question is
    only what to scan for.  Measured on the installed corpus: pairing every
    construction with every call gives **1627** pairs, which is the generic
    reference edge this corpus has punished twice; classes owning a junction
    number 49, of which 44 are not declared specs, of which 17 are constructed
    at all -- **66 sites**, three of them ``AdderGen``.

    Declared specs are left out because ``classify`` and the attribution
    already answer for their fields, and because ``JobSpec.pack`` handing a
    row around is data movement rather than a handover of control.

    Nothing is folded here: three constructions of ``AdderGen`` are three
    entries, and the trace returns the set.  Where two land in the same
    function of the same module, :func:`attach` keys them together -- the key
    is ``(trigger, entry, via)`` and that is a property of the door, not of
    this reading.
    """
    wanted = _classes_owning_a_junction(modules, junctions, spec_classes)
    if not wanted:
        return {}
    reach: dict[tuple[str, str], list[tuple[str, str, dict[str, str], tuple[int, int]]]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            for node in ast.walk(func):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                held = wanted.get(node.func.id)
                if held is None:
                    continue
                where, methods, span, init = held
                handover = _handover(node, init)
                if not handover:
                    continue
                for method in methods:
                    reach.setdefault((where, method), []).append(
                        (module.rel_path, func.name, handover, span)
                    )
    return reach


def _classes_owning_a_junction(
    modules: list[SourceModule],
    junctions: list[JunctionNode],
    spec_classes: Collection[str],
) -> dict[str, tuple[str, list[str], tuple[int, int], Optional[ast.AST]]]:
    """``{class: (module, its methods, its span, its __init__)}``.

    Every method, not the ones a dispatch door reaches: without a door there
    is nothing to walk from, and what the constructor handed over is a fact
    about the object rather than about one entry into it -- the same reason
    the walk scopes ``self.<field>`` to the class.  :func:`attach` still
    checks that the arm's line falls inside the class, which is what keeps two
    classes in one module from answering for each other.

    A name two modules declare is dropped rather than chosen between: a
    construction site names the class and nothing else.
    """
    owners = {junction.owner for junction in junctions}
    found: dict[str, list[tuple[str, list[str], tuple[int, int], Optional[ast.AST]]]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef) or node.name in spec_classes:
                continue
            methods = [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not any(f"{module.rel_path}::{name}" in owners for name in methods):
                continue
            found.setdefault(node.name, []).append(
                (
                    module.rel_path,
                    methods,
                    (node.lineno, node.end_lineno or node.lineno),
                    _initialiser(node),
                )
            )
    return {
        name: places[0]
        for name, places in found.items()
        if len(places) == 1 and places[0][3] is not None
    }


def dispatch_fanouts(modules: list[SourceModule]) -> dict[str, list[DispatchFanout]]:
    """Return ``{module::class: [fan-out]}`` for the run-time class choices.

    ``panda_config.getPlugin("adder_plugins", vo, group)`` returns whatever a
    config names, and the source cannot say which.  What the source *can* say
    is the shape of the answer: the call is followed by a guard on ``None``
    that imports a concrete class and uses it instead, so the corpus declares
    both the interface (that class's base) and the alternatives (the base's
    other subclasses).

    **The class condition is what makes this a reading rather than a pattern.**
    The structural shape on its own -- assign a call, then ``if x is None: x =
    y`` -- matches 29 sites in this corpus, and most are ordinary defaults:
    ``maxHS06sec``, ``coreCount``, ``newScanList``.  Requiring *y* to be a
    class the corpus declares leaves 3, and they are the three ``getPlugin``
    sites.  The guard may be a conjunction -- ``closer.py`` asks
    ``is None and self.job.VO == "atlas"`` -- so a conjunction counts, which
    is what found the third.

    Keyed by class because that is the scope of the fact.  ``AdderGen`` picks
    its plugin in ``get_plugin_class`` and runs it from
    ``process_job_report``, which is where the arm is.

    Not reachable this way: the 18 ``getImpl`` / ``instantiateImpl`` sites.
    Their candidate set is in ``jedi_config.<x>.modConfig`` as a
    ``module:className`` string, so the source does not hold it at all.  What
    the source does hold is the proof: ``FactoryBase.initializeMods`` prints
    ``getting class {className}`` and ``{cls} is ready for ...`` at INFO.
    """
    subclasses: dict[str, list[str]] = {}
    declared: dict[str, list[ast.ClassDef]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            declared.setdefault(node.name, []).append(node)
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name:
                    subclasses.setdefault(name, []).append(node.name)

    found: dict[str, list[DispatchFanout]] = {}
    for module in modules:
        for cls in [n for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)]:
            fanouts = [
                fanout
                for method in cls.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                for fanout in _fanouts_in(method, declared, subclasses)
            ]
            if fanouts:
                found.setdefault(f"{module.rel_path}::{cls.name}", []).extend(fanouts)
        # A dispatch at module level, or in a plain function, still belongs to
        # the file even though no class holds it.
        for func, owner in functions_with_owner(module.tree):
            if owner is not None:
                continue
            fanouts = list(_fanouts_in(func, declared, subclasses))
            if fanouts:
                found.setdefault(f"{module.rel_path}::{func.name}", []).extend(fanouts)
    return found


def _fanouts_in(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    declared: dict[str, list[ast.ClassDef]],
    subclasses: dict[str, list[str]],
) -> Iterator[DispatchFanout]:
    body = list(ast.walk(func))
    seen: set[tuple[str, str]] = set()
    for node in body:
        targets = targets_of(node)
        if not (targets and isinstance(node.value, ast.Call)):
            continue
        name = _assigned_to(targets[0])
        if name is None:
            continue
        for guard in body:
            if not isinstance(guard, ast.If) or not _tests_none(guard.test, name):
                continue
            for statement in ast.walk(guard):
                inner = targets_of(statement)
                if not (
                    inner
                    and _assigned_to(inner[0]) == name
                    and isinstance(statement.value, ast.Name)
                ):
                    continue
                default = statement.value.id
                places = declared.get(default)
                if not places or len(places) != 1 or (name, default) in seen:
                    continue
                seen.add((name, default))
                base = next(
                    (
                        b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                        for b in places[0].bases
                        if isinstance(b, (ast.Name, ast.Attribute))
                    ),
                    "",
                )
                yield DispatchFanout(
                    at=node.lineno,
                    selector=_rendered(node.value),
                    default=default,
                    base=base,
                    candidates=sorted({default, *subclasses.get(base, ())}),
                    announced_by=_announces(func, name),
                )


def _assigned_to(target: ast.expr) -> Optional[str]:
    """``x`` or ``self.x``, the two spellings a dispatch result is kept under."""
    if isinstance(target, ast.Name):
        return target.id
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return f"self.{target.attr}"
    return None


def _tests_none(test: ast.expr, name: str) -> bool:
    """``<name> is None``, alone or as one term of an ``and``."""
    parts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    return any(
        isinstance(part, ast.Compare)
        and part.ops
        and isinstance(part.ops[0], ast.Is)
        and _assigned_to(part.left) == name
        and isinstance(part.comparators[0], ast.Constant)
        and part.comparators[0].value is None
        for part in parts
    )


def _announces(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> str:
    """The literal part of a line that prints *name*'s chosen class.

    Only the literal text, because that is what a query can match: the hole
    is the answer and cannot be in the pattern.  ``adder_gen`` prints
    ``plugin name {self.adder_plugin_class.__name__}``, so ``plugin name ``
    is what to ask the log for.
    """
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr):
            continue
        names = {
            _assigned_to(part.value.value)
            for part in node.values
            if isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Attribute)
            and part.value.attr == "__name__"
        }
        if name not in names:
            continue
        literal = "".join(
            part.value for part in node.values if isinstance(part, ast.Constant)
        ).strip()
        if literal:
            return literal
    return ""


def _rendered(call: ast.Call) -> str:
    try:
        return ast.unparse(call)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return ""


def attach_dispatch(junctions: list[JunctionNode], modules: list[SourceModule]) -> int:
    """Record the run-time class choices made in each junction's class.

    Returns how many junctions got one.  Scoped by the class and settled by
    the arm's line, the same way :func:`worker_uplinks` settles which of two
    workers in one module a line belongs to: ``closer.py`` declares more than
    one class and a key of ``module::class`` alone would hand each the
    other's arms.
    """
    fanouts = dispatch_fanouts(modules)
    if not fanouts:
        return 0
    spans: dict[str, list[tuple[str, tuple[int, int], list[DispatchFanout]]]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            held = fanouts.get(f"{module.rel_path}::{node.name}")
            if held:
                spans.setdefault(module.rel_path, []).append(
                    (node.name, (node.lineno, node.end_lineno or node.lineno), held)
                )
    found = 0
    for junction in junctions:
        module, _, _method = junction.owner.partition("::")
        if junction.anchor is None:
            continue
        for _name, span, held in spans.get(module, ()):
            if span[0] <= junction.anchor.line_start <= span[1]:
                junction.dispatch = list(held)
                found += 1
                break
    return found


def attach(
    junctions: list[JunctionNode],
    modules: list[SourceModule],
    foreign_tables: set[str],
    uplinks: Optional[
        dict[tuple[str, str], list[tuple[str, str, dict[str, str], tuple[int, int]]]]
    ] = None,
) -> tuple[int, int]:
    """Record each junction's entry points.  Returns ``(reached, total)``.

    A junction in an unclassified module reached by nothing is left with no
    entry points rather than being assigned a default.  "Nothing in this map
    starts this" is a real answer and a work item; "presumably a loop" is a
    guess that would make the self-repair property unusable.

    Two kinds of edge, read separately because they are found differently.  A
    call is resolved by name (:func:`reaching_modules`); a dispatched worker is
    not reached by a call at all (:func:`worker_uplinks`), and the second is
    where the knights are -- the code that moves a task's status runs in a
    class its knight constructs, so without it the map can say what starts an
    arm but not what handed the arm its input.
    """
    triggers = classify(modules, foreign_tables)
    inward = reaching_modules(
        modules, through_doors=True, through_inheritance=True, through_dispatch=True
    )
    # Taken from the caller where there is one, so that a build which also
    # reports on the dispatch walks for it once rather than twice.
    if uplinks is None:
        uplinks = worker_uplinks(modules)

    reached = 0
    for junction in junctions:
        owner_module, _, method = junction.owner.partition("::")
        # Keyed by how it arrived as well as by where from.  Without that a
        # facade crossing and a plain call that share a door name collide, and
        # the survivor is whichever was built last -- which silently replaced
        # real argument lists with unread ones and took twelve rows off the
        # differing-arguments report.  The same collision was already possible
        # between a call and a dispatch.
        found: dict[tuple[str, str, Optional[str], str], EntryPoint] = {}
        for kind in triggers.get(owner_module, ()):
            found[(kind, owner_module, None, ARRIVES_BY_CALL)] = EntryPoint(
                trigger=kind, entry=owner_module
            )
        for entry, door, call in inward.get(owner_module, {}).get(method, ()):
            arrival = ARRIVES_BY_CALL if call is not None else ARRIVES_THROUGH_DOOR
            for kind in triggers.get(entry, ()):
                found[(kind, entry, door, arrival)] = EntryPoint(
                    trigger=kind,
                    entry=entry,
                    via=door,
                    # The binding is at the door, which is where the entries
                    # differ -- the message path omits ``minPriority`` there,
                    # not deeper in.
                    arg_binding=_arg_binding(call) if call is not None else {},
                    # A facade crossing carries no call, and saying so is what
                    # keeps ``differing_arguments`` from reading an unread list
                    # as an entry that handed over nothing.
                    reached_by=arrival,
                )
        for entry, via, handover, span in uplinks.get((owner_module, method), ()):
            # The anchor decides which class the line is in.  Where there is
            # none the span cannot be checked, and a junction with no position
            # is not one an investigation can be sent to anyway.
            if junction.anchor and not (
                span[0] <= junction.anchor.line_start <= span[1]
            ):
                continue
            for kind in triggers.get(entry, ()):
                found[(kind, entry, via, ARRIVES_BY_DISPATCH)] = EntryPoint(
                    trigger=kind,
                    entry=entry,
                    via=via,
                    arg_binding=handover,
                    reached_by=ARRIVES_BY_DISPATCH,
                )
        junction.entry_points = [found[key] for key in sorted(found, key=str)]
        if junction.entry_points:
            reached += 1
    return reached, len(junctions)


#: Why a junction ended up with no entry point.  Three, because three is what
#: :func:`attach` can decide from what it already reads -- whether
#: :func:`reaching_modules` resolved a caller, whether :func:`classify` gave any
#: of those callers a trigger, and whether the owner is module-level code.
#: Anything needing a fourth judgment would be this slice guessing, and the
#: report says "unclassified" rather than inventing a bucket for it.
NO_CALLER_RESOLVED = "no caller in another module that the map resolves"
CALLER_STARTS_NOTHING = "a caller is named, but nothing declares what starts it"
RUNS_AT_IMPORT = "runs when its module is imported"
UNCLASSIFIED = "unclassified"


def unreached_reasons(
    junctions: list[JunctionNode],
    modules: list[SourceModule],
    foreign_tables: set[str],
) -> dict[str, list[JunctionNode]]:
    """Return ``{reason: junctions}`` for the ones :func:`attach` left empty.

    ``entry_points == []`` is an honest answer, but on its own it is one number
    covering three different situations, and only one of them is work this
    slice could do:

    ``NO_CALLER_RESOLVED``
        No module other than the owner's own calls it.  Two of the three ways
        that happens are deliberate, and only the first is an absence: the
        corpus really does not call it; or the facade hop was dropped
        (:func:`_outward_call_sites`), so a proxy method the API reaches only
        through ``TaskBuffer`` lands here; or the only callers are in the
        owner's own module, which :func:`reaching_modules` skips because
        "what starts this" is not answered by a sibling -- it only moves the
        question to what starts the sibling.  4 of the 32 junctions in this
        bucket have a ``self.<method>`` call in their own file, and the name
        used to read as though the map had found nothing at all.
    ``CALLER_STARTS_NOTHING``
        A caller is named and carries no trigger, so answering would mean
        asking what starts *it*.  **That is the one-hop rule in the module
        docstring, not a defect**: a name match is a weak edge, and chaining
        weak edges multiplies the error rather than the reach.
    ``RUNS_AT_IMPORT``
        The owner is ``<module>``.  "What starts this" is "who imports it",
        which is a different question with a different answer shape.

    Recomputed rather than carried out of :func:`attach`, which would mean
    widening its return for a reporting concern.  Both passes together are
    about four seconds on the PanDA corpus, against a build in minutes.
    """
    triggers = classify(modules, foreign_tables)
    inward = reaching_modules(
        modules, through_doors=True, through_inheritance=True, through_dispatch=True
    )

    found: dict[str, list[JunctionNode]] = {}
    for junction in junctions:
        if junction.entry_points:
            continue
        owner_module, _, method = junction.owner.partition("::")
        callers = inward.get(owner_module, {}).get(method, ())
        if method == "<module>":
            reason = RUNS_AT_IMPORT
        elif not callers:
            reason = NO_CALLER_RESOLVED
        elif not any(triggers.get(entry) for entry, _door, _call in callers):
            reason = CALLER_STARTS_NOTHING
        else:
            # Unreachable as long as ``attach`` builds an EntryPoint for every
            # (caller, trigger) pair it finds.  Kept so that a future filter
            # there shows up as a number instead of being absorbed by one of
            # the three answers above.
            reason = UNCLASSIFIED
        found.setdefault(reason, []).append(junction)
    return found


def _reaches_from_self(receiver: ast.expr) -> bool:
    """True when *receiver* is this object or a collaborator it holds.

    The same "the receiver is what identifies it" rule
    :func:`_outward_call_sites` turns on for the facade, deciding here which
    calls are *consultations*.  A junction consults what its object holds --
    ``self.taskBufferIF.<method>()`` -- while ``newScanSiteList.append(x)`` and
    ``tmpLog.debug(msg)`` are data and output passing through the function.
    Without the distinction a name match resolves ``append`` to
    ``SQLManager.append`` on 241 junctions, every one of them a list; it is the
    generic reference edge failing in a new place, which this corpus has
    charged for before.

    Two spellings, because the proxy is assembled from mixins and reaches its
    siblings through an accessor rather than an attribute:
    ``get_task_event_module(self).updateInputStatusJedi(...)`` is the same claim
    as ``self.taskBufferIF.<method>()``, and refusing it costs 33 junctions
    their only route to another entity's rows.

    A module-level ``taskBuffer`` global -- the spelling three daemon scripts
    use -- is not accepted: it is a bare name with nothing tying it to this
    object, and taking it would mean trusting a name again.  Measured cost:
    three junctions, reported rather than guessed at.
    """
    while True:
        if isinstance(receiver, ast.Attribute):
            receiver = receiver.value
        elif isinstance(receiver, ast.Call):
            if any(isinstance(a, ast.Name) and a.id == "self" for a in receiver.args):
                return True
            receiver = receiver.func
        else:
            return isinstance(receiver, ast.Name) and receiver.id == "self"


def _consulted_targets(
    modules: list[SourceModule],
) -> dict[str, dict[str, set[str]]]:
    """Return ``{module: {function: qualified targets}}`` -- what each one consults.

    Qualified, because a bare name is not an identity in this corpus and the
    reader of this field joins on it.  Two restrictions decide whether a name
    resolves at all, and they are the ones :func:`reaching_modules` already
    turns on, applied in the other direction: the name means one thing across
    the tree, or the caller imports the module it names.  Without them ``run``
    would make every daemon consult every other.

    A call inside the module itself stays inside it -- ``self.<name>()`` needs
    no resolution, which is why it is recorded even where the name is defined
    nowhere the map can see (proxy mixins inherit plenty).  A key that resolves
    to nothing costs a lookup and claims nothing.

    Which calls count at all is :func:`_reaches_from_self`; the sites come from
    :func:`_outward_call_sites`, which only ever records a call whose ``func``
    is an attribute, so the receiver is always there to ask about.
    """
    defined = definitions(modules)
    sole = sole_definitions(modules)
    imports = {module.rel_path: imported_modules(module) for module in modules}

    targets: dict[str, dict[str, set[str]]] = {}
    for module in modules:
        here = module.rel_path
        for method, callees in _self_calls(module.tree).items():
            targets.setdefault(here, {}).setdefault(method, set()).update(
                f"{here}::{callee}" for callee in callees
            )
        for enclosing, method, call in _outward_call_sites(module):
            if not enclosing or not _reaches_from_self(call.func.value):
                continue
            for home in defined.get(method, ()):
                if home == here:
                    continue
                if sole.get(method) != home and home not in imports[here]:
                    continue
                for name in enclosing:
                    targets.setdefault(here, {}).setdefault(name, set()).add(
                        f"{home}::{method}"
                    )
    return targets


def attach_calls(junctions: list[JunctionNode], modules: list[SourceModule]) -> int:
    """Record what each junction's owner consults.  Returns how many got any.

    Separate from :func:`attach` although both walk the same edges, because the
    two questions they answer are different and one field answering both is how
    ``log_files`` came to mean two things.  ``attach`` asks what *starts* this
    junction and so follows the edges inward and transitively; this asks what
    this junction *consults* and so follows them outward and exactly one hop.

    One hop, not the closure, because the hop is what is being claimed.  The
    arm that sends a task to ``exhausted`` decided on an aggregate over its
    jobs, and the map has no edge for that aggregate -- the write and the read
    sit in two methods with a call between them.  Following further would stop
    being "this junction asks that question" and start being "these two things
    are in the same neighbourhood", which is the shape of join this corpus keeps
    punishing.  Measured: a second hop reaches two junctions the first does not.

    Across modules as well as within one.  Keeping it inside a file was not a
    principle but the shape of :func:`_self_calls`, and it cost the common case:
    a knight decides a task's status and asks the task buffer about that task's
    rows, which is ``self.taskBufferIF.<method>()`` and lands in another
    package every time.
    """
    targets = _consulted_targets(modules)
    found = 0
    for junction in junctions:
        owner_module, _, method = junction.owner.partition("::")
        junction.calls = sorted(targets.get(owner_module, {}).get(method, ()))
        if junction.calls:
            found += 1
    return found


def self_repairing(junctions: list[JunctionNode]) -> dict[str, set[str]]:
    """Return ``{subject: triggers}`` pooled over every junction writing it.

    The property the plan asks for, one level up from the junction: a subject
    only a message can reach is fragile, one a loop can reach comes back by
    itself, and which of those is true decides whether an investigation should
    wait or intervene.
    """
    pooled: dict[str, set[str]] = {}
    for junction in junctions:
        pooled.setdefault(junction.subject, set()).update(
            entry.trigger for entry in junction.entry_points
        )
    return pooled


def differing_arguments(
    junctions: list[JunctionNode],
) -> list[tuple[str, str, dict[str, list[str]]]]:
    """Junctions whose entries do not hand over the same arguments.

    The reason entry points are part of the structure rather than context.
    ``JobGenerator`` reaches ``getTasksToBeProcessed_JEDI`` with ``minPriority``
    and ``maxNumJobs``; the message processors reach the same method with
    neither, so the throttle guard cannot fire on that path at all.  Asked why a
    task was not picked up, the candidate causes are therefore a different set
    depending on which entry ran -- which no amount of reading the branch table
    would reveal.
    """
    found: list[tuple[str, str, dict[str, list[str]]]] = []
    for junction in junctions:
        # Within one way of arriving.  A call binds by keyword and a dispatch
        # by the field the worker's body reads, so the two key sets are drawn
        # from different namespaces: comparing across them reports a
        # difference in spelling as a guard that cannot fire.  Measured: it
        # took the report from 13 rows to 69, and the 56 it added were pairs
        # like ``ContentsFeeder`` -- one entry calling a method directly and
        # one handing the same work to a worker.
        by_arrival: dict[str, dict[str, set[str]]] = {}
        for entry in junction.entry_points:
            if entry.via is not None:
                by_arrival.setdefault(entry.reached_by, {}).setdefault(
                    entry.entry, set()
                ).update(entry.arg_binding)
        for supplied in by_arrival.values():
            if len(supplied) > 1 and len({frozenset(a) for a in supplied.values()}) > 1:
                found.append(
                    (
                        junction.subject,
                        junction.owner,
                        {entry: sorted(args) for entry, args in sorted(supplied.items())},
                    )
                )
    return found


def fragile_subjects(junctions: list[JunctionNode]) -> list[tuple[str, list[str]]]:
    """Subjects no self-repairing trigger reaches, with the triggers that do."""
    return sorted(
        (subject, sorted(kinds))
        for subject, kinds in self_repairing(junctions).items()
        if kinds and not kinds & SELF_REPAIRING_TRIGGERS
    )
