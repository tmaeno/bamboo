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
from typing import Optional

from bamboo.codemap.models import (
    ARRIVES_BY_DISPATCH,
    COMMAND,
    MESSAGE,
    POLLED,
    REQUEST,
    SELF_REPAIRING_TRIGGERS,
    EntryPoint,
    JunctionNode,
    SourceModule,
)
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.pathcond import functions_with_owner
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
        elif isinstance(node, ast.Assign) and _rooted_at_pool(node.value):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
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
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and isinstance(node.value, ast.Name)
            and node.value.id in parameters
        ):
            kept[node.targets[0].attr] = node.value.id
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
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                method = _forwarded_call(node.value, forwarders)
                if method:
                    built.extend(
                        (_bound_name(target), method, node.value)
                        for target in node.targets
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
            queue.append((child, inner_names, inner_pooled))
    sites.extend(_forwarded_sites(module, forwarders or {}))
    return sites


def _outward_calls(
    module: SourceModule,
    forwarders: Optional[dict[str, tuple[str, int]]] = None,
) -> dict[str, ast.Call]:
    """Return ``{method name: first call}`` for the calls *module* really makes.

    :func:`_calls_by_name` with the facade hop removed -- see
    :func:`_outward_call_sites`, which does the reading.  What the door needs is
    the module's whole surface and the *first* call under each name, because
    that is the call whose keyword arguments distinguish one entry from another.
    """
    calls: dict[str, ast.Call] = {}
    for _enclosing, method, call in _outward_call_sites(module, forwarders):
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
) -> dict[str, dict[str, list[tuple[str, str, ast.Call]]]]:
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
    """
    forwarders = forwarder_classes(modules)
    callers: dict[str, list[tuple[str, ast.Call]]] = {}
    for module in modules:
        for name, call in _outward_calls(module, forwarders).items():
            callers.setdefault(name, []).append((module.rel_path, call))

    implemented = sole_definitions(modules)
    imports = {module.rel_path: imported_modules(module) for module in modules}
    inward: dict[str, dict[str, list[tuple[str, str, ast.Call]]]] = {}
    for module in modules:
        edges = _self_calls(module.tree)
        for door in edges:
            unambiguous = implemented.get(door) == module.rel_path
            for entry, call in callers.get(door, ()):
                if entry == module.rel_path:
                    continue
                if not unambiguous and module.rel_path not in imports[entry]:
                    continue
                for method in _downstream(edges, door):
                    inward.setdefault(module.rel_path, {}).setdefault(method, []).append(
                        (entry, door, call)
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
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
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
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id in workers
                ):
                    built.extend((_bound_name(t), node.value) for t in node.targets)
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
    inward = reaching_modules(modules)
    # Taken from the caller where there is one, so that a build which also
    # reports on the dispatch walks for it once rather than twice.
    if uplinks is None:
        uplinks = worker_uplinks(modules)

    reached = 0
    for junction in junctions:
        owner_module, _, method = junction.owner.partition("::")
        found: dict[tuple[str, str, Optional[str]], EntryPoint] = {}
        for kind in triggers.get(owner_module, ()):
            found[(kind, owner_module, None)] = EntryPoint(
                trigger=kind, entry=owner_module
            )
        for entry, door, call in inward.get(owner_module, {}).get(method, ()):
            for kind in triggers.get(entry, ()):
                found[(kind, entry, door)] = EntryPoint(
                    trigger=kind,
                    entry=entry,
                    via=door,
                    # The binding is at the door, which is where the entries
                    # differ -- the message path omits ``minPriority`` there,
                    # not deeper in.
                    arg_binding=_arg_binding(call),
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
                found[(kind, entry, via)] = EntryPoint(
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
