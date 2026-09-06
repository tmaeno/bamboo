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

from bamboo.codemap.models import SourceModule
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

    Keyed by the bare name because that is how a base is written at the point
    of inheritance, qualified or not.  Two classes share the name ``DBProxy``
    -- the server's and JEDI's -- and that collision is load-bearing here
    rather than a nuisance: it is the edge from one to the other.
    """
    edges: dict[str, list[tuple[str, str]]] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if name:
                    edges.setdefault(name, []).append((node.name, module.rel_path))
    return edges


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


def attach(fragment, modules: list[SourceModule]) -> tuple[int, int]:
    """Record each node's candidate log files.  Returns ``(resolved, total)``.

    Junctions and filter stages both carry it because both are things an
    investigation is pointed at: a stage says why candidates were dropped, a
    junction says why a value was settled, and neither can be checked without
    knowing which file to read.

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
    """
    declared = declared_files(modules)
    inherited = inherited_files(modules, declared)
    inward = trigger.reaching_modules(modules)
    resolved = total = 0
    for node in list(fragment.filter_stages) + list(fragment.junctions):
        total += 1
        node.log_files = files_of(node.owner.split("::")[0], declared, inherited)
        if node.log_files:
            resolved += 1

    for junction in fragment.junctions:
        where, _, method = junction.owner.partition("::")
        junction.owns_logger = where in declared
        mine = set(junction.log_files)
        # The enclosing class is part of ``owner`` but not of the call graph's
        # keys, which are bare method names -- the same last-segment rule the
        # attribution slice uses to name a function.
        reached = inward.get(where, {}).get(method.split(".")[-1], ())
        junction.caller_log_files = sorted(
            {
                file
                for entry, _door, _call in reached
                for file in files_of(entry, declared, inherited)
            }
            - mine
        )
    return resolved, total
