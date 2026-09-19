"""Which log file a module's diagnostics land in.

The map's purpose is to say which component's log to read and what to look
for, so the file is part of the answer rather than a detail of fetching it.
It is also what makes the production check practical: PanDA writes one file
per logger, not one per service, so a question about brokerage rejections
goes to ``panda-AtlasProdJobBroker.log`` and comes back narrow instead of
arriving as a truncated slice of everything the service logged.

``PandaLogger.getLogger(name)`` opens ``<logdir>/panda-<name>.log``, and the
corpus names loggers two ways and only two::

    logger = PandaLogger().getLogger(__name__.split(".")[-1])   # the module
    _logger = PandaLogger().getLogger("api_async_process")      # a literal

The split runs along package lines -- JEDI takes the module's own name, the
server mostly spells one out -- but both forms appear in both, so the shape is
what is read, not the package.

**A module can have more than one answer, and that is the interesting case.**
The ``db_proxy_mods`` modules -- 175 of the map's junctions, by far the
largest group -- declare no logger.  They are mixins: ``OraDBProxy.DBProxy``
inherits all thirteen of them and ``JediDBProxy.DBProxy`` inherits that, and
each of those two *does* declare one.  So the same junction writes to
``panda-DBProxy.log`` when the server calls it and ``panda-JediDBProxy.log``
when a knight does.

That is not an ambiguity to be resolved.  It is the JEDI/server crossing
stated exactly: the file depends on which process ran the code, which is a
runtime fact, and naming both candidates is the true answer.  Hence a list
rather than a single name.

**A module with neither is left empty.**  ``JobBrokerBase`` logs through a
``MsgWrapper`` built by whoever instantiated it and is mixed into nothing, so
there is no evidence to read.  A wrong filename would be worse than none: the
query comes back empty, and empty reads as "production never emitted this".
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.models import LogSiteNode, SourceModule
from bamboo.codemap.panda.recognizers import trigger

# PandaLogger.getLogger(name) opens "<logdir>/panda-<name>.log".
_FILENAME = "panda-{}.log"

_GETLOGGER = "getLogger"


def _module_basename(rel_path: str) -> str:
    """``pandajedi/jediorder/JobGenerator.py`` -> ``JobGenerator``."""
    return rel_path.rsplit("/", 1)[-1][: -len(".py")]


def _names_itself(node: ast.expr) -> bool:
    """Whether the argument is the ``__name__.split(".")[-1]`` idiom.

    Matched structurally rather than by unparsing, so that spacing and the
    quote style cannot change the answer.
    """
    if not isinstance(node, ast.Subscript):
        return False
    call = node.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr != "split":
        return False
    receiver = call.func.value
    return isinstance(receiver, ast.Name) and receiver.id == "__name__"


def logger_name(module: SourceModule) -> Optional[str]:
    """The logger this module declares, or None when it declares none.

    The first declaration wins.  A module with two is not a case the corpus
    has, and picking one silently would be a guess; the first is the one at
    module scope in every instance here.
    """
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != _GETLOGGER or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
        if _names_itself(argument):
            return _module_basename(module.rel_path)
    return None


def declared_files(modules: list[SourceModule]) -> dict[str, str]:
    """Return ``{rel_path: log filename}`` for every module that declares one."""
    resolved: dict[str, str] = {}
    for module in modules:
        name = logger_name(module)
        if name:
            resolved[module.rel_path] = _FILENAME.format(name)
    return resolved


def _subclass_edges(modules: list[SourceModule]) -> dict[str, list[tuple[str, str]]]:
    """Return ``{base class name: [(subclass, its module), ...]}``.

    The names of :func:`trigger.subclass_edges`, which does the walking.  Two
    slices read inheritance -- this one to follow a mixin's output to the log
    of whatever inherits it, the trigger slice to find the workers a knight
    dispatches -- and the tree is walked once so the two cannot drift.
    """
    return {
        base: [(node.name, where) for node, where in places]
        for base, places in trigger.subclass_edges(modules).items()
    }


def _classes_in(module: SourceModule) -> list[str]:
    return [n.name for n in ast.walk(module.tree) if isinstance(n, ast.ClassDef)]


def inherited_files(
    modules: list[SourceModule], declared: dict[str, str]
) -> dict[str, list[str]]:
    """Files a mixin's output reaches, through the classes that inherit it.

    Walked transitively, because the chain has two links: the proxy modules
    are mixed into ``OraDBProxy.DBProxy``, which is itself subclassed by
    ``JediDBProxy.DBProxy``.  Stopping at one would name the server's log and
    silently omit JEDI's -- the half most junctions actually run under.
    """
    edges = _subclass_edges(modules)
    reached: dict[str, list[str]] = {}
    for module in modules:
        if module.rel_path in declared:
            continue
        files: set[str] = set()
        seen: set[tuple[str, str]] = set()
        frontier = [(name, module.rel_path) for name in _classes_in(module)]
        while frontier:
            class_name, where = frontier.pop()
            if (class_name, where) in seen:
                continue
            seen.add((class_name, where))
            if where != module.rel_path and where in declared:
                files.add(declared[where])
            frontier.extend(edges.get(class_name, ()))
        if files:
            reached[module.rel_path] = sorted(files)
    return reached


def files_of(
    rel_path: str, declared: dict[str, str], inherited: dict[str, list[str]]
) -> list[str]:
    """The files one module's output can reach, declaration first."""
    if rel_path in declared:
        return [declared[rel_path]]
    return inherited.get(rel_path, [])


def _caller_files(
    owner: str,
    mine: list[str],
    declared: dict[str, str],
    inherited: dict[str, list[str]],
    inward,
) -> list[str]:
    """Files belonging to the modules that reach *owner*, minus its own.

    One reading for the two nodes that carry it.  The enclosing class is part
    of ``owner`` but not of the call graph's keys, which are bare method names
    -- the same last-segment rule the attribution slice uses to name a
    function.
    """
    where, _, method = owner.partition("::")
    reached = inward.get(where, {}).get(method.split(".")[-1], ())
    return sorted(
        {file for entry, _door, _call in reached for file in files_of(entry, declared, inherited)}
        - set(mine)
    )


def attach(fragment, modules: list[SourceModule]) -> tuple[int, int]:
    """Record each node's candidate log files.  Returns ``(resolved, total)``.

    Junctions, filter stages and loop cuts all carry it because each is a thing
    an investigation is pointed at: a stage or a cut says why candidates were
    dropped, a junction says why a value was settled, and none of them can be
    checked without knowing which file to read.

    Junctions additionally get ``caller_log_files``, because for the largest
    group in the map the module that holds the code is not the one that logs
    about it.  A ``db_proxy_mods`` method inherits ``panda-DBProxy.log`` and
    ``panda-JediDBProxy.log`` from the proxy classes that mix it in, and those
    are true -- the SQL comment trace lands there -- but the line an
    investigation greps for, ``set task_status=``, is written by the knight
    that made the call.  Production settles it: of 33 files asked, that line is
    in exactly five -- ContentsFeeder, JobGenerator, PostProcessor,
    TaskCommando and TaskRefiner -- and in neither proxy file.  Each of the
    five reaches a proxy junction as a caller.

    Kept out of ``log_files`` on purpose.  One list would answer "where does
    this code live" and "whose log mentions it" with the same value, and
    conflating those is what made eleven candidates for a pending task look
    indistinguishable when nine of them are separable.

    ``owns_logger`` records which of the two ``log_files`` is.  A reader that
    cannot tell an inherited file from a declared one has no way to know that
    asking the proxy files for a caller's line always returns nothing, and would
    read that nothing as "the junction did not fire" -- for every proxy
    candidate at once.

    The same three facts are recorded for the owners the map names as reading
    or writing a value and that settle nothing themselves -- see
    :func:`_log_sites`.  Done here rather than in a pass of its own because the
    declaration, the inheritance chain and the callers are all already in hand;
    a second pass would be the same three readings again.
    """
    declared = declared_files(modules)
    inherited = inherited_files(modules, declared)
    inward = trigger.reaching_modules(modules)
    resolved = total = 0
    for node in (
        list(fragment.filter_stages) + list(fragment.loop_cuts) + list(fragment.junctions)
    ):
        total += 1
        node.log_files = files_of(node.owner.split("::")[0], declared, inherited)
        if node.log_files:
            resolved += 1

    for junction in fragment.junctions:
        junction.owns_logger = junction.owner.split("::")[0] in declared
        junction.caller_log_files = _caller_files(
            junction.owner, junction.log_files, declared, inherited, inward
        )
    fragment.log_sites.extend(_log_sites(fragment, declared, inherited, inward))
    return resolved, total


def _actor_owners(fragment) -> set[str]:
    """Every ``module::function`` the map names as reading or writing a value.

    Both verbs and both models.  A descent follows ``read_by``, but a question
    about a value that only an ``UPDATE ... WHERE`` acts on still has to say
    which log will show that update running, and the answer is found the same
    way.
    """
    owners: set[str] = set()
    for subject in fragment.subjects:
        for names in list(subject.selected_by.values()) + list(subject.updated_by.values()):
            owners.update(names)
    for entity in fragment.entities:
        owners.update(entity.read_by)
        owners.update(entity.written_by)
    return owners


def _log_sites(
    fragment,
    declared: dict[str, str],
    inherited: dict[str, list[str]],
    inward,
) -> list[LogSiteNode]:
    """Where the readers and writers that own no junction write their diagnostics.

    ``_follow_up`` looks a reader up among the *writers* of the subject, so it
    only ever finds one that happens to settle a value too.  Measured on the
    installed corpus: 279 such owners, 94 of them junctions, and every one of
    the remaining 185 resolves to a file -- 53 to its own and 132 to a
    caller's as well.  The question "which log will say the query ran" was
    therefore unanswerable for two thirds of the readers the map names, not
    because the fact is missing but because nowhere held it.

    Owners that already own a junction are skipped.  Storing the same three
    fields twice is how one of them comes to disagree with the other.
    """
    known = {junction.owner for junction in fragment.junctions}
    sites: list[LogSiteNode] = []
    for owner in sorted(_actor_owners(fragment) - known):
        where = owner.split("::")[0]
        mine = files_of(where, declared, inherited)
        sites.append(
            LogSiteNode(
                map_id=fragment.map_id,
                derived_from=fragment.derived_from,
                name=owner,
                owner=owner,
                log_files=mine,
                caller_log_files=_caller_files(owner, mine, declared, inherited, inward),
                owns_logger=where in declared,
            )
        )
    return sites
