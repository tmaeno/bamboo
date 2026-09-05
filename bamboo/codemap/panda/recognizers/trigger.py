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

from bamboo.codemap.models import EntryPoint, JunctionNode, SourceModule
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.pathcond import functions_with_owner
from bamboo.codemap.panda.recognizers.boundary import endpoint_decorator

POLLED = "polled"
COMMAND = "command"
MESSAGE = "message"
REQUEST = "request"

# Self-repairing triggers.  A subject only these can reach comes back on its
# own; a subject only the others can reach does not.
_SELF_REPAIRING = frozenset({POLLED})

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


def _outward_calls(module: SourceModule) -> dict[str, ast.Call]:
    """Return ``{method name: first call}`` for the calls *module* really makes.

    :func:`_calls_by_name` with the facade hop removed.  ``TaskBuffer`` and
    ``JediTaskBuffer`` borrow a connection from ``self.proxyPool`` and call the
    implementation through it, so a call on a borrowed proxy is the same handoff
    :func:`_forwards_to_itself` already refuses to count as a *definition*,
    refused here as a *call*.

    **The receiver is what identifies it, not the name.**  A facade method may
    forward under a different name --
    ``JediTaskBuffer.checkWaitingTaskPrio_JEDI`` calls
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
    calls: dict[str, ast.Call] = {}
    # Breadth-first over the tree, matching ``_calls_by_name``: which call a
    # name resolves to decides the ``arg_binding`` reported for that entry.
    queue: list[tuple[ast.AST, frozenset[str]]] = [(module.tree, frozenset())]
    while queue:
        node, pooled = queue.pop(0)
        for child in ast.iter_child_nodes(node):
            inner = (
                _pool_bindings(child)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else pooled
            )
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                receiver = child.func.value
                if not (isinstance(receiver, ast.Name) and receiver.id in pooled):
                    calls.setdefault(child.func.attr, child)
            queue.append((child, inner))
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
    defined: dict[str, set[str]] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            if _forwards_to_itself(func):
                continue
            defined.setdefault(func.name, set()).add(module.rel_path)
    return {name: next(iter(where)) for name, where in defined.items() if len(where) == 1}


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


def _self_calls(module: SourceModule) -> dict[str, set[str]]:
    """Return ``{method: the methods it calls on ``self``}`` within one module.

    Needed because the door and the write are rarely the same method.
    ``add_main`` starts ``AdderGen.run``; the ``jobStatus`` writes are in
    ``finalize_job_status`` and ``handle_failed_job``, several ``self`` calls
    further in.  Stopping at the door left 16 of ``setupper_atlas_plugin``'s
    junctions and 11 of ``adder_gen``'s looking as though nothing runs them.

    Within one module ``self.<name>()`` is an unambiguous edge -- no resolution
    is involved, which is why this is followed transitively while the hop
    *between* modules, which rests on a name match, is not.
    """
    edges: dict[str, set[str]] = {}
    for func, _owner in functions_with_owner(module.tree):
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
    callers: dict[str, list[tuple[str, ast.Call]]] = {}
    for module in modules:
        for name, call in _outward_calls(module).items():
            callers.setdefault(name, []).append((module.rel_path, call))

    implemented = sole_definitions(modules)
    imports = {module.rel_path: imported_modules(module) for module in modules}
    inward: dict[str, dict[str, list[tuple[str, str, ast.Call]]]] = {}
    for module in modules:
        edges = _self_calls(module)
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


def attach(
    junctions: list[JunctionNode],
    modules: list[SourceModule],
    foreign_tables: set[str],
) -> tuple[int, int]:
    """Record each junction's entry points.  Returns ``(reached, total)``.

    A junction in an unclassified module reached by nothing is left with no
    entry points rather than being assigned a default.  "Nothing in this map
    starts this" is a real answer and a work item; "presumably a loop" is a
    guess that would make the self-repair property unusable.
    """
    triggers = classify(modules, foreign_tables)
    inward = reaching_modules(modules)

    reached = 0
    for junction in junctions:
        owner_module, _, method = junction.owner.partition("::")
        found: dict[tuple[str, str, Optional[str]], EntryPoint] = {}
        for trigger in triggers.get(owner_module, ()):
            found[(trigger, owner_module, None)] = EntryPoint(
                trigger=trigger, entry=owner_module
            )
        for entry, door, call in inward.get(owner_module, {}).get(method, ()):
            for trigger in triggers.get(entry, ()):
                found[(trigger, entry, door)] = EntryPoint(
                    trigger=trigger,
                    entry=entry,
                    via=door,
                    # The binding is at the door, which is where the entries
                    # differ -- the message path omits ``minPriority`` there,
                    # not deeper in.
                    arg_binding=_arg_binding(call),
                )
        junction.entry_points = [found[key] for key in sorted(found, key=str)]
        if junction.entry_points:
            reached += 1
    return reached, len(junctions)


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
        supplied: dict[str, set[str]] = {}
        for entry in junction.entry_points:
            if entry.via is not None:
                supplied.setdefault(entry.entry, set()).update(entry.arg_binding)
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
        if kinds and not kinds & _SELF_REPAIRING
    )
