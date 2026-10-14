"""Where a call lands, what it is handed, and what comes back.

The walk crossed a function boundary in two ways only: a handover the map had
recorded, and a ``self.<field>`` answered from the class's other methods.  It
never followed a value into the function that returned it or out to the call
that passed it in, so every cross-function shape the corpus has was either a
rule of its own -- a literal returned one hop up, a mapping read out of a
zero-argument method, ``.add()`` on a bare name -- or a stop.  Python has no
end of those shapes, and each case found the next one.

This module is the one place a call is read, three ways:

* :meth:`Calls.callees` -- which function definitions a call can land in,
  read the way a type checker reads it: ``self.m`` through the class and its
  bases, ``obj.m`` through what ``obj`` is declared to be, a bare name through
  the enclosing function, the module and its imports, ``Class(...)`` as its
  ``__init__``.  Never by the method's name alone -- a crossing that matched
  callees by bare name gave 222 read sites the same two false leads -- so a
  call it cannot place comes back with the reason, not a guess.
* :meth:`Calls.bind` -- which expression at the call each parameter holds.
* :meth:`Calls.returns` -- what the function hands back, slot by slot.

What a method changes on its object is read by :meth:`Calls.writes`, so that
``self.result.set_fatal()`` is a write of ``status_code`` like an assignment.

It spends none of the walk's budget, like the declared-type reading it is
built on, and reads the same trees -- :func:`parsed`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from bamboo.codemap.panda import pathcond

#: Why a call could not be placed.  Each is a class of the census, so the
#: count of calls the walk could not enter says what would enter them.
UNPLACED_OUTSIDE = "outside the corpus"
UNPLACED_RECEIVER = "the receiver's class is not declared"
UNPLACED_FORWARDED = "the receiver forwards every call through __getattr__"
UNPLACED_NO_METHOD = "the declared class has no such method"
#: ``self.conn.close()`` on ``WrappedOracleConn``: the class defines no
#: ``close``, and its ``__getattribute__`` hands the name to the connection it
#: wraps -- as a type checker reads it, the name may be supplied there.  A
#: method the class does define is still placed: ``JobSpec``'s
#: ``__getattribute__`` filters values and supplies no name of its own.
UNPLACED_GETATTRIBUTE = "the declared class has no such method, and __getattribute__ may supply one"
UNPLACED_DYNAMIC = "the callee is chosen at run time"
UNPLACED_NAME = "the name is bound to something that is not a function here"
#: ``Interaction.JEDITemporaryError(...)``, ``DataCarouselRequestSpec()``: the
#: class and its bases in the corpus define no ``__init__``, so what building
#: one runs is ``object``'s, ``Exception``'s or a base's outside the corpus.
UNPLACED_NO_INIT = "the class it builds has no __init__ in the corpus"
#: Beside targets, not instead of them: the receiver is one of these classes
#: or another one some way of building it does not show -- or it is one of
#: them that supplies the method only by forwarding the call elsewhere.  Said
#: on its own like the others -- :attr:`TraceStep.open` holds nothing else --
#: and not as a clause that reads only after the targets are named.
UNPLACED_OPEN = "the call may also land somewhere this reading cannot name"
#: What a class says of a name it supplies only by forwarding, the reason
#: that says more first.
_FORWARDING = (UNPLACED_FORWARDED, UNPLACED_GETATTRIBUTE)

#: Where :meth:`Types.candidates` says a value is an instance of a builtin
#: type -- ``tmp_list = []``, ``seen = set()`` -- as ``(BUILTIN, "list")``.
#: Not a file: no class of the corpus is defined there, so no method of one is
#: a call's target.
BUILTIN = "<builtins>"
#: The builtin types a call of their name builds, and what each builds.
_BUILTIN_TYPES = {
    "list": "list",
    "dict": "dict",
    "set": "set",
    "tuple": "tuple",
    "frozenset": "frozenset",
    "str": "str",
    "bytes": "bytes",
    "sorted": "list",
}
#: The standard library's container classes, by the module they are imported from.
_LIBRARY_CONTAINERS = {
    ("collections", "defaultdict"),
    ("collections", "OrderedDict"),
    ("collections", "deque"),
    ("collections", "Counter"),
}
#: The standard library's copies, and the method a class defines to copy
#: itself some other way.  ``copy.copy(x)`` is an object of ``x``'s class --
#: typeshed states ``def copy(x: _T) -> _T`` -- unless that class defines the
#: hook, and then it is what the hook returns.  The map reads the corpus's
#: annotations and not typeshed's, so it is said here.
_COPIES = {("copy", "copy"): "__copy__", ("copy", "deepcopy"): "__deepcopy__"}


#: Parsed modules shared by every reading of the same files.  A walk is made
#: once per arm, per statement and per printed line, and the corpus index
#: behind :meth:`Calls.callers` would otherwise be rebuilt for each.  Keyed by
#: the file's stat, so a tree edited under a reading is read afresh.
_PARSED: dict[tuple[str, int, int], Optional[tuple[str, ast.Module]]] = {}
#: ``callers`` index and answers, per set of roots.
_INDEXES: dict[tuple, dict[str, list["CallSite"]]] = {}
_CALLERS: dict[tuple, list["CallSite"]] = {}
#: The other names each name is imported or bound as, the locals and fields
#: each is bound to, and each class's subclasses, per set of roots --
#: :meth:`Calls.spellings`.
_ALIASES: dict[tuple, dict[str, set[str]]] = {}
_HOLDERS: dict[tuple, dict[str, set[str]]] = {}
_SUBCLASSES: dict[tuple, dict[tuple[str, int], list[tuple[str, ast.ClassDef]]]] = {}
#: Where each class is read as a value, per set of roots -- :meth:`Calls.read_as_value`.
_VALUES: dict[tuple, dict[tuple[str, int], list[str]]] = {}


def parsed(path: Path) -> Optional[tuple[str, ast.Module]]:
    """``(source, tree)`` for *path*, parents attached; ``None`` where it will not read or parse.

    **One tree per file.**  The index behind :meth:`Calls.callers` holds the
    nodes of whatever parse of a file existed when it was built, and a method
    is found by the identity of its node; the walk used to parse each file
    again for itself.  An index built before the walk reached
    ``JediTaskBuffer.py`` held the other parse's methods, ``self.proxyPool``
    could not be typed in them, and the DBProxy methods lost their callers --
    by which walks had run before in the process: 9 of 144 sampled print
    walks.  So every reading takes its tree from here, and the walk's
    re-parsing of each file went with it.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _PARSED:
        try:
            source = path.read_text(errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, ValueError):
            _PARSED[key] = None
        else:
            pathcond.attach_parents(tree)
            _PARSED[key] = (source, tree)
    return _PARSED[key]


# Test code is not the system's behaviour.  It writes the same attributes with
# the same literals, so leaving it in files fixtures as junctions and doubles
# the ambiguous attribution cases -- 88 of 376 modules, and 47% of the writes
# whose spec class could not be settled came from them.  A junction pointing at
# a test would send an investigation to code that never runs in production;
# so would a caller: a JEDI test script's own body builds ``JobGeneratorThread``
# and calls ``doGenerate``, and listed among a function's callers it pushed the
# daemons' calls past the count a list names.
_TEST_PATH = re.compile(r"(^|/)(tests?|jeditest)(/|$)", re.IGNORECASE)
_TEST_FILE = re.compile(r"(^test_|_tests?$)", re.IGNORECASE)


def is_test_module(rel_path: str) -> bool:
    """True when *rel_path* is test code rather than system behaviour."""
    return bool(_TEST_PATH.search(rel_path) or _TEST_FILE.search(Path(rel_path).stem))


def _roots_key(roots: dict[str, Path]) -> tuple:
    return tuple(sorted((package, str(root)) for package, root in roots.items()))


class Target(NamedTuple):
    """One function definition a call can land in."""

    file: str
    func: ast.AST
    cls: Optional[ast.ClassDef]

    @property
    def owner(self) -> str:
        return f"{self.file}::{self.func.name}"  # type: ignore[attr-defined]


class Placed(NamedTuple):
    """Where a call lands: the definitions, or why none is named.

    *on_class*: the call reads its method off a class --
    ``JobBrokerBase.__init__(self, ...)`` -- so a method that takes the
    object is handed it as the call's first argument (:func:`bind`).
    """

    targets: tuple[Target, ...] = ()
    unplaced: str = ""
    on_class: bool = False


#: The owner of a call in a module's own body -- the walk's spelling of the
#: scope a module binds its names in (``trace.MODULE_SCOPE``).
MODULE_SCOPE = "module-scope"

#: How :meth:`Calls.missed` says a function is called from nowhere in the corpus.
_OUTSIDE = "no call in the corpus names it -- reached from outside"
#: ... and that the calls naming it all land outside the corpus.
_LANDS_OUTSIDE = "no call this reading places lands at it"
#: ... and that its class is held as a value something may build from -- :meth:`Calls.read_as_value`.
_READ_AS_VALUE = "no call in the corpus names it; the class is read as a value at {}, and a call of that value may build one"
#: How many of those reads it names before it counts the rest.
_READS = 5


class Route(NamedTuple):
    """One way calls reach a function: the call into it first, out to where the values are written -- :meth:`Calls.routes`."""

    sites: tuple["CallSite", ...]

    @property
    def chain(self) -> tuple[str, ...]:
        """``file:line`` of each call, the call into the function first."""
        return tuple(f"{site.file}:{site.call.lineno}" for site in self.sites)


class CallSite(NamedTuple):
    """One call, in the function that makes it -- or in the module's own body, *func* the module.

    A call in the module's body runs when the file is imported, or under
    ``if __name__ == "__main__":`` when it is run as a script: ``taskBuffer =
    TaskBuffer()`` at the foot of ``TaskBuffer.py``.  It has no caller of its
    own and no parameters, so what it hands a function is written there.
    """

    file: str
    func: ast.AST
    call: ast.Call

    @property
    def in_module(self) -> bool:
        return isinstance(self.func, ast.Module)

    @property
    def owner(self) -> str:
        return f"{self.file}::{MODULE_SCOPE if self.in_module else self.func.name}"  # type: ignore[attr-defined]


def own_nodes(scope: ast.AST):
    """Every node of *scope*'s own body; a nested function or class runs elsewhere."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield child
        yield from own_nodes(child)


def returns(func: ast.AST) -> list[ast.expr]:
    """The values *func* hands back, in source order; a bare ``return`` is ``None``."""
    return [
        node.value if node.value is not None else ast.Constant(value=None)
        for node in own_nodes(func)
        if isinstance(node, ast.Return)
    ]


def _is_static(func: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in getattr(func, "decorator_list", [])
    )


def _is_classmethod(func: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
        for decorator in getattr(func, "decorator_list", [])
    )


def _positional(target: Target, on_class: bool) -> list[ast.arg]:
    """The parameters a call to *target* fills by position.

    Past ``self`` or ``cls`` where the call names the object it is made on.
    A method read off its class (*on_class*, :class:`Placed`) is handed its
    object by position like any other argument --
    ``TaskRefinerBase.extractCommon(self, jediTaskID, ...)`` hands ``self``
    to ``self``, not to ``jediTaskID`` -- while a classmethod is handed its
    class either way.
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if (
        target.cls is not None
        and not _is_static(target.func)
        and positional
        and not (on_class and not _is_classmethod(target.func))
    ):
        positional = positional[1:]
    return positional


def bind(call: ast.Call, target: Target, on_class: bool = False) -> dict[str, Optional[ast.expr]]:
    """``{parameter: the expression the call hands it}``; ``None`` where it is not said.

    Positional, keyword and default, in Python's order.  A ``*args`` or
    ``**kwargs`` at the call makes every parameter it could fill unknown
    rather than guessed; a parameter the call leaves to its default holds the
    default.  ``self`` is left out of a method reached through an instance,
    and bound to the first argument of one read off its class (*on_class*,
    :func:`_positional`).
    """
    args = target.func.args  # type: ignore[attr-defined]
    positional = _positional(target, on_class)
    defaults = dict(
        zip([p.arg for p in positional][len(positional) - len(args.defaults):], args.defaults, strict=True)
    )
    passed = list(call.args)
    spread = any(isinstance(one, ast.Starred) for one in passed)
    named = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    spread_named = any(keyword.arg is None for keyword in call.keywords)
    bound: dict[str, Optional[ast.expr]] = {}
    for index, parameter in enumerate(positional):
        if not spread and index < len(passed) and not isinstance(passed[index], ast.Starred):
            bound[parameter.arg] = passed[index]
        elif parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        elif spread or spread_named:
            bound[parameter.arg] = None
        else:
            bound[parameter.arg] = defaults.get(parameter.arg)
    for parameter, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if parameter.arg in named:
            bound[parameter.arg] = named[parameter.arg]
        else:
            bound[parameter.arg] = None if spread_named else default
    return bound


def said(call: ast.Call, target: Target, on_class: bool = False) -> set[str]:
    """The parameters *call* fills itself, by position or by keyword -- not by default.

    ``self``/``cls`` is not among them: the call names the object it is made
    on, which is a different thing from passing it -- unless the method is
    read off its class (*on_class*), where the object is passed like the
    rest.
    """
    positional = _positional(target, on_class)
    filled = {
        parameter.arg
        for index, parameter in enumerate(positional)
        if index < len(call.args) and not any(isinstance(one, ast.Starred) for one in call.args[: index + 1])
    }
    filled |= {keyword.arg for keyword in call.keywords if keyword.arg}
    return filled


def receiver_parameter(target: Target) -> Optional[str]:
    """``self`` or ``cls`` -- the parameter a method call fills with the object it is made on."""
    args = target.func.args  # type: ignore[attr-defined]
    positional = [*args.posonlyargs, *args.args]
    if target.cls is None or _is_static(target.func) or not positional:
        return None
    return positional[0].arg


class Calls:
    """Calls read against one tree, with a corpus index built the first time it is asked."""

    def __init__(self, types: Types) -> None:
        self.types = types
        self._placed: dict[int, Placed] = {}
        self._by_name: Optional[dict[str, list[CallSite]]] = None
        self._callers: dict[tuple[str, int], list[CallSite]] = {}
        self._missed: dict[tuple[str, int], list[str]] = {}
        #: The parameters :meth:`handed_classes` is reading now: one asked
        #: again on the way is a cycle, and says nothing.
        self._handing: set[tuple[str, int, str]] = set()
        types.handed_to = self.handed_classes

    # -- where a call lands -------------------------------------------------

    def callees(self, file: str, func: ast.AST, call: ast.Call) -> Placed:
        key = id(call)
        if key not in self._placed:
            self._placed[key] = self._place(file, func, call)
        return self._placed[key]

    def _place(self, file: str, func: ast.AST, call: ast.Call) -> Placed:
        called = call.func
        if isinstance(called, ast.Name):
            named = self._named(file, func, called.id)
            if named.unplaced == UNPLACED_NAME and func is not None:
                return self._constructed(file, func, called) or named
            return named
        if not isinstance(called, ast.Attribute):
            return Placed((), UNPLACED_DYNAMIC)
        receiver = called.value
        if isinstance(receiver, (ast.Constant, ast.JoinedStr, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            # ``",".join(...)``, ``f"...".format``: a builtin type's method.
            return Placed((), UNPLACED_OUTSIDE)
        if isinstance(receiver, ast.Name) and receiver.id in ("self", "cls"):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            method = self._method_of(cls, called.attr)
            if method.unplaced == UNPLACED_NO_METHOD and receiver.id == "self":
                return self._constructed(file, func, called) or method
            return method
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
        ):
            cls = self.types.method_class(file, func)
            if cls is None:
                return Placed((), UNPLACED_RECEIVER)
            return self._after(cls, called.attr)
        module = self.types.module_named(file, receiver)
        if module is not None:
            return self._module_member(module, called.attr)
        # ``JobBrokerBase.__init__(self, ...)``: a method read off its class.
        named = self.types.class_named(file, receiver)
        if named is not None:
            return self._method_of(named, called.attr)._replace(on_class=True)
        declared = self.types.declared(file, func, receiver)
        if declared is not None:
            # ``self.conn: WrappedOracleConn | WrappedPostgresConn`` -- each
            # class it is declared as, as for the values below.
            return self._method_of_any(sorted(declared, key=_class_key), called.attr)
        # Nothing declares it -- or it is declared ``Any`` -- so read what it
        # was built from.  Every class it can be is a target; where some way
        # of building it cannot be read, that is said beside them.
        classes, opened = self.types.candidates(file, func, receiver)
        builtin = {name for where, name in classes if where == BUILTIN}
        if builtin and len(builtin) == len(classes) and not opened:
            # ``tmp_list = []`` ... ``tmp_list.append(x)``: a builtin type's
            # method, as on a literal above -- and no call to a corpus
            # function of that name (:meth:`Calls.missed`).
            return Placed((), UNPLACED_OUTSIDE)
        # Beside a class of the corpus, a builtin is as open as it was before
        # it was told apart: the call may land on either.
        opened = opened or bool(builtin)
        built = [
            (where, cls)
            for where, name in sorted(classes)
            if where != BUILTIN and (cls := self.types.class_def(where, name)) is not None
        ]
        placed = self._method_of_any(built, called.attr, opened)
        if placed.targets or placed.unplaced in _FORWARDING:
            return placed
        root = receiver
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and self.types.imported_from_outside(file, root.id):
            return Placed((), UNPLACED_OUTSIDE)
        return Placed((), UNPLACED_RECEIVER)

    def _declared_in(self, file: str, cls: ast.ClassDef, name: str) -> Optional[Target]:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Target(file, node, cls)
        return None

    def _method_of(self, declared: tuple[str, ast.ClassDef], name: str) -> Placed:
        file, cls = declared
        lineage = self.types.lineage(file, cls)
        for where, holder in lineage:
            found = self._declared_in(where, holder, name)
            if found is not None:
                return Placed((found,))
        for hook, reason in (("__getattr__", UNPLACED_FORWARDED), ("__getattribute__", UNPLACED_GETATTRIBUTE)):
            if any(self._declared_in(where, holder, hook) is not None for where, holder in lineage):
                return Placed((), reason)
        return Placed((), UNPLACED_NO_METHOD)

    def _after(self, owned: tuple[str, ast.ClassDef], name: str) -> Placed:
        """Where ``super().<name>(...)`` in a method of *owned* lands.

        At the next class after *owned* in the method resolution order of
        the object's class (:meth:`Types.mro`) that defines *name* -- for an
        object of *owned* itself and of each of its subclasses in the corpus,
        so a mixin's ``super().__init__(log_stream)`` lands at the base it
        names and, on an ``OraDBProxy.DBProxy``, at the next mixin, whose
        constructor no other call reaches.  *owned*'s own order first.
        """
        file, cls = owned
        targets: list[Target] = []
        for where, holder in [owned, *self.subclasses(file, cls)]:
            order = self.types.mro(where, holder)
            at = next((index for index, (_file, one) in enumerate(order) if one is cls), None)
            for place, base in order[at + 1:] if at is not None else ():
                found = self._declared_in(place, base, name)
                if found is not None:
                    if found not in targets:
                        targets.append(found)
                    break
        return Placed(tuple(targets)) if targets else Placed((), UNPLACED_NO_METHOD)

    def _method_of_any(
        self, classes: list[tuple[str, ast.ClassDef]], name: str, opened: bool = False
    ) -> Placed:
        """Where a call of *name* lands on an object that is one of *classes*.

        At the method of each class that has one.  A class that supplies the
        name only by forwarding it -- ``JediTaskBufferInterface``'s
        ``__getattr__``, ``WrappedOracleConn``'s ``__getattribute__`` -- adds
        no definition: what runs is somewhere this reading does not follow, so
        the placement is open beside the definitions found.  Where no class
        has the method, the reason is the forwarding one if any class
        forwards, else that none has it.  *opened* is the caller's own reason
        to say so: some way the object is built cannot be read.
        """
        targets: list[Target] = []
        forwarded: set[str] = set()
        for one in classes:
            placed = self._method_of(one, name)
            targets.extend(target for target in placed.targets if target not in targets)
            if not placed.targets and placed.unplaced in _FORWARDING:
                forwarded.add(placed.unplaced)
        if targets:
            return Placed(tuple(targets), UNPLACED_OPEN if opened or forwarded else "")
        for reason in _FORWARDING:
            if reason in forwarded:
                return Placed((), reason)
        return Placed((), UNPLACED_NO_METHOD)

    def _constructed(self, file: str, func: ast.AST, holder: ast.expr) -> Optional[Placed]:
        """Where a call on a local or a ``self.<field>`` holding a class lands: that class's ``__init__``.

        ``setupper_plugin_class(self.task_buffer, self.jobs, tmp_log, ...)``
        builds whatever ``panda_config.getPlugin`` returned, or
        ``SetupperAtlasPlugin`` where it returned nothing; ``AdderGen`` keeps
        its class in ``self.adder_plugin_class``.  Neither is a function or a
        method, so the call was placed nowhere, and the plugins' constructors
        had no caller: a logger, a job or a task buffer handed to them could
        not be followed past them.  The classes are what the holder is bound
        to (:meth:`Types._values`).

        A binding this cannot read -- ``getPlugin`` returns ``type[Any]`` --
        leaves the call open beside the classes found, and the corpus's other
        subclasses of their base are taken as well: a configuration names a
        plugin of the same family, and those are the ones the corpus has
        (``trigger.dispatch_fanouts`` reads the same set at the same three
        sites).  A parameter holding the class is open the same way, assigned
        or not: ``make(job, cls=None)`` builds whatever class a caller hands
        it.  ``None`` where the holder names no class with a constructor in
        the corpus.
        """
        classes, opened = self.types._values(file, func, holder, 0, instances=False, seen=frozenset())
        arguments = getattr(func, "args", None)
        if isinstance(holder, ast.Name) and arguments is not None:
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            opened = opened or any(one.arg == holder.id for one in every)
        built = [
            (where, cls)
            for where, name in sorted(classes)
            if where != BUILTIN and (cls := self.types.class_def(where, name)) is not None
        ]
        if not built:
            return None
        if opened:
            for where, cls in list(built):
                lineage = self.types.lineage(where, cls)
                if len(lineage) > 1:
                    built += [one for one in self.subclasses(*lineage[1]) if one not in built]
        placed = self._method_of_any(built, "__init__", opened)
        return placed if placed.targets else None

    def _named(self, file: str, func: ast.AST, name: str) -> Placed:
        # A function defined inside this one, or inside one that holds it.
        scope: Optional[ast.AST] = func
        while scope is not None:
            for node in own_nodes(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return Placed((Target(file, node, None),))
            for node in ast.iter_child_nodes(scope):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return Placed((Target(file, node, None),))
            scope = self.types.enclosing_function(file, scope)
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Placed((Target(file, node, None),))
        constructed = self.types.class_named(file, ast.Name(id=name, ctx=ast.Load()))
        if constructed is not None:
            init = self._method_of(constructed, "__init__")
            return init if init.targets else Placed((), UNPLACED_NO_INIT)
        held = self.types.imports(file).get(name)
        if held is not None:
            module, attribute = held
            target = self.types.module_file(module)
            if target is None:
                return Placed((), UNPLACED_OUTSIDE)
            return self._module_member(target, attribute or name)
        return Placed((), UNPLACED_NAME)

    def _module_member(self, file: str, name: str) -> Placed:
        tree = self.types.tree(file)
        if tree is None:
            return Placed((), UNPLACED_OUTSIDE)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return Placed((Target(file, node, None),))
            if isinstance(node, ast.ClassDef) and node.name == name:
                init = self._method_of((file, node), "__init__")
                return init if init.targets else Placed((), UNPLACED_NO_INIT)
        held = self.types.imports(file).get(name)
        if held is not None and held[0]:
            target = self.types.module_file(held[0])
            if target is not None and target != file:
                return self._module_member(target, held[1] or name)
        return Placed((), UNPLACED_NAME)

    # -- who calls a function ----------------------------------------------

    def every_call(self) -> dict[str, list[CallSite]]:
        """Every call in the corpus, by the name it calls -- see :meth:`_index`."""
        return self._index()

    def _index(self) -> dict[str, list[CallSite]]:
        """Every call in the corpus, by the name it calls -- a candidate list only.

        A name is how to *find* the calls worth placing, never how to place
        one: each is resolved by :meth:`callees` before it is kept.

        The calls in each module's own body come after every function's
        (:class:`CallSite`): ``extract_allowed_methods(...)`` at import,
        ``main()`` under ``if __name__ == "__main__":``.  A class's body,
        whose calls the corpus places at none of its functions, and a
        lambda's, which runs when it is called, are not read.
        """
        if self._by_name is not None:
            return self._by_name
        shared = _INDEXES.get(_roots_key(self.types.roots))
        if shared is not None:
            self._by_name = shared
            return shared
        found: dict[str, list[CallSite]] = {}

        def add(file: str, scope: ast.AST) -> None:
            for node in own_nodes(scope):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                name = (
                    called.attr if isinstance(called, ast.Attribute)
                    else called.id if isinstance(called, ast.Name) else ""
                )
                if name:
                    found.setdefault(name, []).append(CallSite(file, scope, node))

        trees = [(file, tree) for file in self.types.corpus_files() if (tree := self.types.tree(file)) is not None]
        for file, tree in trees:
            for func in ast.walk(tree):
                if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(file, func)
        for file, tree in trees:
            add(file, tree)
        self._by_name = found
        _INDEXES[_roots_key(self.types.roots)] = found
        return found

    def callers(self, target: Target) -> list[CallSite]:
        """The calls in the corpus that :meth:`callees` places at *target*.

        Every call :meth:`callees` places at *target* is here -- the
        consistency the tests and the census hold it to.  The candidates are
        the calls spelled with any of :meth:`spellings`; the target's own
        name comes first, so a list that was whole under that name alone
        keeps its order.
        """
        key = (target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in self._callers:
            return self._callers[key]
        shared = (_roots_key(self.types.roots), *key)
        if shared in _CALLERS:
            self._callers[key] = _CALLERS[shared]
            return _CALLERS[shared]
        kept = [
            site
            for name in self.spellings(target)
            for site in self._index().get(name, [])
            if any(
                (placed.file, placed.func.lineno) == key  # type: ignore[attr-defined]
                for placed in self.callees(site.file, site.func, site.call).targets
            )
        ]
        self._callers[key] = kept
        _CALLERS[shared] = kept
        return kept

    def handed(self, site: CallSite, target: Target) -> dict[str, Optional[ast.expr]]:
        """:func:`bind` for *site*, its method read off a class where :meth:`callees` placed it so."""
        return bind(site.call, target, self.callees(site.file, site.func, site.call).on_class)

    def handed_classes(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """The classes every call of *func* hands its parameter *name*, declared ``Any``; ``None`` where one is not declared.

        PanDA writes ``Any`` where no one class it can name covers what
        arrives: ``WorkflowInterface(task_buffer: Any)`` is built with
        JEDI's ``JediTaskBufferInterface`` by its watchdog and with the
        server's ``TaskBuffer`` by ``workflow_api`` -- and naming the JEDI
        class would make ``pandaserver`` import ``pandajedi``
        (``pandaserver.asyncprocess.TaskBufferLike`` says so).  Each call
        says which one it hands, so the parameter holds what the calls hand
        it, read by their declarations: every class any of them hands, as an
        annotation naming them all would.  A ``None`` handed adds nothing,
        as an assignment of ``None`` does not (:meth:`Types.declared`).

        Nothing where a call hands what is not declared, or what a ``*args``
        hides, or where no call is placed: the daemons' ``main(tbuf: Any)``
        is called by the name a configuration gives ``importlib``.
        """
        key = (file, getattr(func, "lineno", 0), name)
        if depth > self.types.DEPTH or key in self._handing:
            return None
        self._handing.add(key)
        try:
            owned = self.types.method_class(file, func)
            target = Target(file, func, owned[1] if owned else None)
            sites = self.callers(target)
            found: set[tuple[str, ast.ClassDef]] = set()
            for site in sites:
                value = self.handed(site, target).get(name)
                if value is None:
                    return None
                if isinstance(value, ast.Constant) and value.value is None:
                    continue
                classes = self.types.declared(site.file, site.func, value, depth + 1)
                if classes is None:
                    return None
                found |= classes
            return frozenset(found) or None
        finally:
            self._handing.discard(key)

    def filled(self, site: CallSite, target: Target) -> set[str]:
        """:func:`said` for *site*, as :meth:`handed`."""
        return said(site.call, target, self.callees(site.file, site.func, site.call).on_class)

    def spellings(self, target: Target) -> list[str]:
        """Every name a call that lands at *target* can be spelled with, its own name first.

        A function's own name.  An ``__init__``'s class name; ``__init__``
        itself, which is how ``super().__init__(...)`` and
        ``Base.__init__(self, ...)`` spell the chain from a subclass's
        constructor; and the name of each subclass that defines no
        ``__init__`` of its own, whose construction runs this one.  Then every
        other name any of those is imported or bound as in the corpus --
        ``from pandaserver.api.v1.http_client import HttpClient as
        HttpClientV1`` built twenty clients no list of ``HttpClient``'s callers
        held.  Last, for an ``__init__``, each local or field a statement binds
        a class of its family to: ``setupper_plugin_class(...)`` builds
        ``SetupperAtlasPlugin`` (:meth:`_constructed`).  A name matching more
        than the calls landing here costs only their placing: :meth:`callees`
        keeps the ones that land.
        """
        name = target.func.name  # type: ignore[attr-defined]
        if name == "__init__" and target.cls is not None:
            seeds = [target.cls.name, *self._built_through(target)]
        else:
            seeds = [name]
        aliases = self._aliases()
        found = list(dict.fromkeys(seeds))
        queue = list(found)
        while queue:
            for other in sorted(aliases.get(queue.pop(0), ())):
                if other not in found:
                    found.append(other)
                    queue.append(other)
        if name == "__init__" and target.cls is not None:
            found.insert(1, "__init__")
            # A local or a field holding the class, or another of its family,
            # is called to build it (:meth:`_constructed`).
            holders = self._holders()
            for one in self._families(target):
                found += [other for other in sorted(holders.get(one, ())) if other not in found]
        return found

    def _families(self, target: Target) -> list[str]:
        """Names of the classes whose construction :meth:`_constructed` may place at *target*, an ``__init__``.

        Its class and the subclasses built through it, and for each the
        other subclasses of its base -- the plugins one configuration may
        name in place of another.
        """
        assert target.cls is not None
        through = set(self._built_through(target))
        classes = [(target.file, target.cls)] + [
            one for one in self._subclasses().get((target.file, target.cls.lineno), []) if one[1].name in through
        ]
        names: list[str] = []
        for where, cls in classes:
            lineage = self.types.lineage(where, cls)
            for _file, one in [(where, cls), *(self.subclasses(*lineage[1]) if len(lineage) > 1 else [])]:
                if one.name not in names:
                    names.append(one.name)
        return names

    def _holders(self) -> dict[str, set[str]]:
        """``{name: each local or self.<field> a statement binds it to}``, across the corpus -- a candidate list only."""
        shared = _roots_key(self.types.roots)
        if shared not in _HOLDERS:
            found: dict[str, set[str]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in ast.walk(tree) if tree is not None else ():
                    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, (ast.Name, ast.Attribute)):
                        continue
                    held = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
                    for one in node.targets if isinstance(node, ast.Assign) else [node.target]:
                        local = (
                            one.id if isinstance(one, ast.Name)
                            else one.attr if isinstance(one, ast.Attribute) and getattr(one.value, "id", "") == "self"
                            else ""
                        )
                        if local and local != held:
                            found.setdefault(held, set()).add(local)
            _HOLDERS[shared] = found
        return _HOLDERS[shared]

    def _built_through(self, target: Target) -> list[str]:
        """Names of *target*'s class's subclasses whose construction lands at *target*, an ``__init__``."""
        assert target.cls is not None
        held = self._subclasses().get((target.file, target.cls.lineno), [])
        return sorted(
            {
                cls.name
                for file, cls in held
                if any(one.func is target.func for one in self._method_of((file, cls), "__init__").targets)
            }
        )

    def subclasses(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """Every corpus class with *cls* among its bases."""
        return list(self._subclasses().get((file, cls.lineno), []))

    def _subclasses(self) -> dict[tuple[str, int], list[tuple[str, ast.ClassDef]]]:
        """``{(file, line of a class): every corpus class with it among its bases}``."""
        shared = _roots_key(self.types.roots)
        if shared not in _SUBCLASSES:
            found: dict[tuple[str, int], list[tuple[str, ast.ClassDef]]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in ast.walk(tree) if tree is not None else ():
                    if isinstance(node, ast.ClassDef):
                        for where, base in self.types.lineage(file, node)[1:]:
                            found.setdefault((where, base.lineno), []).append((file, node))
            _SUBCLASSES[shared] = found
        return _SUBCLASSES[shared]

    def _aliases(self) -> dict[str, set[str]]:
        """``{name: each other name it is imported as or bound to at module level}``, across the corpus."""
        shared = _roots_key(self.types.roots)
        if shared not in _ALIASES:
            found: dict[str, set[str]] = {}
            for file in self.types.corpus_files():
                for local, (_module, attribute) in self.types.imports(file).items():
                    if attribute and local != attribute:
                        found.setdefault(attribute, set()).add(local)
                for local, values in self.types._module_bound(file).items():
                    for value in values:
                        held = value.id if isinstance(value, ast.Name) else value.attr  # type: ignore[attr-defined]
                        if held != local:
                            found.setdefault(held, set()).add(local)
            _ALIASES[shared] = found
        return _ALIASES[shared]

    def opened(self, site: CallSite) -> str:
        """Why *site*'s call may land beyond the definitions it is placed at; ``""`` where it is placed closed.

        :meth:`callers` lists a call placed open beside the ones placed
        closed -- ``self.taskBufferIF.getTaskWithID_JEDI(...)`` lands at
        ``JediTaskBuffer``'s, or wherever the interface forwards it.  A reader
        listing callers says so beside it; a reader taking one definition as
        the call's does not take it.
        """
        placed = self.callees(site.file, site.func, site.call)
        return placed.unplaced if placed.targets else ""

    def hop(self, site: CallSite) -> str:
        """``file:line`` of *site*, and why it may land elsewhere where it is placed open."""
        why = self.opened(site)
        return f"{site.file}:{site.call.lineno}" + (f" (placed open -- {why})" if why else "")

    #: How many chains ``called_from`` lists before it counts the rest.
    CALLED_FROM = 16

    def called_from(self, target: Target) -> list[str]:
        """``file:line`` of each call placed at *target*, and one hop out where its caller is called.

        Sorted, capped at :data:`CALLED_FROM` with a closing ``… N more``.
        A hop placed open says so after its ``file:line`` (:meth:`opened`).
        A call in a module's body is the end of its chain.
        """
        chains: list[str] = []
        for site in self.callers(target):
            at = self.hop(site)
            if site.in_module:
                chains.append(at)
                continue
            owned = self.types.method_class(site.file, site.func)
            outer = self.callers(Target(site.file, site.func, owned[1] if owned else None))
            if outer:
                chains.extend(f"{self.hop(one)} -> {at}" for one in outer)
            else:
                chains.append(at)
        chains = sorted(set(chains))
        if len(chains) > self.CALLED_FROM:
            return [*chains[: self.CALLED_FROM], f"… {len(chains) - self.CALLED_FROM} more"]
        return chains

    #: How many calls out a constant is looked for.
    CONSTANT_HOPS = 3

    def handed_constants(
        self, target: Target, names: set[str], depth: int = 0
    ) -> list[tuple[tuple[str, ...], dict[str, object]]]:
        """Per call placed at *target*: where it came from, and the constants it hands *names*.

        ``(("pkg/tb.py:551", "pkg/cmd.py:143"), {"onlyActive": True})`` -- the
        call into *target* first.  A name the caller hands on from a parameter
        of its own that it never rebinds is looked for at that caller's
        callers, so ``JediTaskBuffer``'s forwarding method is read through to
        ``TaskCommando``.  A call handing none of *names* a constant is left
        out: it runs the function as the general reading already reads it.
        Every call placed at *target* is read, however many there are.
        """
        sites = self.callers(target)
        if not sites:
            return []
        found: list[tuple[tuple[str, ...], dict[str, object]]] = []
        for site in sites:
            bound = self.handed(site, target)
            at = f"{site.file}:{site.call.lineno}"
            values: dict[str, object] = {}
            forwarded: dict[str, str] = {}
            for name in names:
                handed = bound.get(name)
                if isinstance(handed, ast.Constant):
                    values[name] = handed.value
                elif (
                    isinstance(handed, ast.Name)
                    and handed.id in _parameter_names(site.func)
                    and not _rebinds(site.func, handed.id)
                ):
                    forwarded[name] = handed.id
            outer: list[tuple[tuple[str, ...], dict[str, object]]] = []
            if forwarded and depth + 1 < self.CONSTANT_HOPS:
                owned = self.types.method_class(site.file, site.func)
                caller = Target(site.file, site.func, owned[1] if owned else None)
                outer = self.handed_constants(caller, set(forwarded.values()), depth + 1)
            for via, more in outer:
                merged = dict(values)
                merged.update({name: more[held] for name, held in forwarded.items() if held in more})
                found.append(((at, *via), merged))
            if not outer and values:
                found.append(((at,), values))
        return found

    #: How many calls out a route goes, the call into the function counted.
    ROUTE_HOPS = 4

    def routes(self, target: Target, depth: int = 0) -> tuple[list["Route"], list[str]]:
        """Every route of calls into *target*, and why the list may miss some (:meth:`missed`).

        A route starts at a call placed at *target* and goes on outward while
        the caller hands *target* one of its own parameters unchanged -- a
        forwarder, ``TaskBuffer.lockJobsForReassign`` handing on what
        ``copyArchive`` gave it -- so that the outermost call of a route is
        where the values are written down, or where the caller's own callers
        are unknown.  Up to :data:`ROUTE_HOPS` calls out, from every call
        placed at *target*, however many there are.
        """
        sites = self.callers(target)
        stops = list(self.missed(target))
        found: list[Route] = []
        for site in sites:
            handed = self.handed(site, target)
            filled = self.filled(site, target)
            forwards = any(
                isinstance(value, ast.Name)
                and name in filled
                and value.id in _parameter_names(site.func)
                and not _rebinds(site.func, value.id)
                for name, value in handed.items()
            )
            if forwards and depth + 1 < self.ROUTE_HOPS:
                owned = self.types.method_class(site.file, site.func)
                outer, more = self.routes(Target(site.file, site.func, owned[1] if owned else None), depth + 1)
                if not outer:
                    # The forwarder is the route's outermost function: what
                    # it hands on is its parameters' values, unfixed, and no
                    # caller of it is missing from a list it has none in.
                    more = [one for one in more if not one.endswith(_OUTSIDE)]
                stops.extend(one for one in more if one not in stops)
                if outer:
                    found.extend(Route((site, *route.sites)) for route in outer)
                    continue
            found.append(Route((site,)))
        return found, stops

    def missed(self, target: Target) -> list[str]:
        """Why the calls placed at *target* may not be every call reaching it; empty where they are.

        A call spelled with *target*'s name that is placed nowhere may be a
        call to it, so the list is said to stop there -- by reason, and by the
        receiver the call is made on -- though the call may be to another
        function of that name: this can call a complete list incomplete, never
        the other way round.  So is a call placed at other definitions and
        open beside them: ``self.taskBufferIF.getTaskWithID_JEDI(...)`` lands
        at ``JediTaskBuffer``'s, or wherever the interface forwards it.  A
        function no call in the corpus names is reached from outside it: an
        endpoint, a thread's ``run``.

        A call placed outside the corpus is not a call to *target* and is not
        listed: ``tmp_list.append(x)`` on a list, ``os.kill(...)``,
        ``json.dump(...)``.  Where every call by the name lands there, that is
        what is said -- not that nothing names it, and not that it is reached
        from outside: ``ListWithLock.__contains__`` is asked by ``in``, which
        this reading does not read as a call.

        The calls looked at are those spelled with any of :meth:`spellings`.
        One spelled ``__init__`` is a call to a constructor only in a subclass
        of *target*'s class -- where the chain from a subclass runs --
        and ``threading.Thread.__init__(self)`` in any other class is not.
        """
        key = (target.file, target.func.lineno)  # type: ignore[attr-defined]
        if key in self._missed:
            return self._missed[key]
        name = target.func.name  # type: ignore[attr-defined]
        if name == "__init__" and target.cls is not None:
            name = target.cls.name
        placed = self.callers(target)
        why: list[str] = []
        unplaced: dict[str, dict[str, int]] = {}
        elsewhere: dict[str, dict[str, int]] = {}
        outside: dict[str, int] = {}
        for site in (site for spelled in self.spellings(target) for site in self._index().get(spelled, [])):
            if site.func is target.func:
                continue
            landed = self.callees(site.file, site.func, site.call)
            if landed.targets and (not landed.unplaced or any(one.func is target.func for one in landed.targets)):
                continue
            called = site.call.func
            if isinstance(called, ast.Attribute) and called.attr == "__init__" and not self._in_subclass(site, target):
                continue
            if landed.targets:
                held = elsewhere.setdefault(landed.unplaced, {})
            elif landed.unplaced == UNPLACED_OUTSIDE:
                held = outside
            else:
                held = unplaced.setdefault(landed.unplaced or "not placed", {})
            receiver = ast.unparse(site.call.func.value) if isinstance(site.call.func, ast.Attribute) else "(a bare name)"
            held[receiver] = held.get(receiver, 0) + 1
        for reason, receivers in sorted(unplaced.items()):
            why.append(f"{name}: {sum(receivers.values())} call(s) by its name not placed -- {reason} ({_on(receivers)})")
        for reason, receivers in sorted(elsewhere.items()):
            why.append(f"{name}: {sum(receivers.values())} call(s) by its name placed elsewhere -- {reason} ({_on(receivers)})")
        if not placed and not unplaced and not elsewhere:
            if outside:
                why.append(
                    f"{name}: {_LANDS_OUTSIDE} -- {sum(outside.values())} call(s) by its name land "
                    f"{UNPLACED_OUTSIDE} ({_on(outside)})"
                )
            elif read := self.read_as_value(target):
                shown = ", ".join(read[:_READS]) + (f", … {len(read) - _READS} more" if len(read) > _READS else "")
                why.append(f"{name}: {_READ_AS_VALUE.format(shown)}")
            else:
                why.append(f"{name}: {_OUTSIDE}")
        self._missed[key] = why
        return why

    def read_as_value(self, target: Target) -> list[str]:
        """``file:line`` of each place a class whose construction runs *target*, an ``__init__``, is read as a value.

        ``self.adder_plugin_class = AdderAtlasPlugin`` holds the class, and
        ``self.adder_plugin_class(job, ...)`` builds one: no call names
        ``AdderAtlasPlugin``, and it is not reached from outside.  A read a
        call of the value may build from -- bound, passed, returned, held in
        a container -- and not one that cannot: a base list, ``isinstance``,
        ``super()``'s argument, an ``except`` clause, a comparison, an
        annotation.  Empty for any other function.
        """
        if target.cls is None or getattr(target.func, "name", "") != "__init__":
            return []
        held = self._subclasses().get((target.file, target.cls.lineno), [])
        built = [(target.file, target.cls)] + [
            (file, cls)
            for file, cls in held
            if any(one.func is target.func for one in self._method_of((file, cls), "__init__").targets)
        ]
        read = self._values_read()
        return sorted({where for file, cls in built for where in read.get((file, cls.lineno), [])})

    def _values_read(self) -> dict[tuple[str, int], list[str]]:
        """``{(file, line of a class): file:line of each read of it as a value a call may build from}``."""
        shared = _roots_key(self.types.roots)
        if shared not in _VALUES:
            found: dict[tuple[str, int], list[str]] = {}
            for file in self.types.corpus_files():
                tree = self.types.tree(file)
                for node in ast.walk(tree) if tree is not None else ():
                    if (
                        isinstance(node, (ast.Name, ast.Attribute))
                        and isinstance(node.ctx, ast.Load)
                        and _may_build_from(node)
                        and (named := self.types.class_named(file, node)) is not None
                    ):
                        found.setdefault((named[0], named[1].lineno), []).append(f"{file}:{node.lineno}")
            _VALUES[shared] = found
        return _VALUES[shared]

    def _in_subclass(self, site: CallSite, target: Target) -> bool:
        """Whether *site* is made in a method of a strict subclass of *target*'s class."""
        held = self.types.method_class(site.file, site.func)
        return held is not None and any(cls is target.cls for _where, cls in self.types.lineage(*held)[1:])

    # -- what a method changes on its object --------------------------------

    def writes(self, target: Target, depth: int = 0) -> list[tuple[str, ast.expr, ast.AST, Target]]:
        """``(field, value, site, where)`` for each write *target* makes to its own object.

        An assignment to ``self.<field>`` (``+=`` included), a container on it
        filled -- ``self.<field>.append(x)``, ``self.<field>[k] = x`` -- and,
        through a call on ``self``, what the called method writes.  So
        ``set_fatal()`` is ``status_code = self.rc_fatal`` at the call that
        makes it, which is what a reader of ``status_code`` needs.
        """
        found: list[tuple[str, ast.expr, ast.AST, Target]] = []
        for node in own_nodes(target.func):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None:
                    continue
                for held in targets:
                    field = _self_field(held)
                    if field:
                        found.append((field, value, node, target))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                field = _self_field(node.func.value)
                if field and node.func.attr in FILLERS and node.args:
                    found.append((field, node.args[-1], node, target))
                elif (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and depth < 3
                    and target.cls is not None
                ):
                    for inner in self._method_of((target.file, target.cls), node.func.attr).targets:
                        found.extend(self.writes(inner, depth + 1))
        return found


#: How many receivers :func:`_on` names before it counts the rest.
_RECEIVERS = 5


def _on(receivers: dict[str, int]) -> str:
    """``self.taskBufferIF 19, tbIF 2`` -- the receivers calls were made on, most first, the first five.

    A method named ``append`` or ``update`` shares its name with every
    container's: the calls on ``varMap`` and ``tmp_list`` that may be calls to
    it run to 168 receivers and eleven thousand characters, so past
    :data:`_RECEIVERS` the rest are counted.
    """
    ranked = sorted(receivers.items(), key=lambda one: (-one[1], one[0]))
    said = ", ".join(f"{receiver} {count}" for receiver, count in ranked[:_RECEIVERS])
    return said + (f", … {len(ranked) - _RECEIVERS} more" if len(ranked) > _RECEIVERS else "")


#: The methods that put a value into a container.  The removals take one out
#: and so answer no question about where a value came from.
FILLERS = ("append", "add", "update", "extend", "setdefault", "insert")


def _parameter_names(func: ast.AST) -> set[str]:
    args = getattr(func, "args", None)
    if args is None:
        return set()
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _rebinds(func: ast.AST, name: str) -> bool:
    """Whether *func*'s own body binds *name* anywhere -- so a parameter's value may not be the one handed in.

    A :class:`pathcond.Binding` outside a nested scope, or a ``del``.  Read
    off the names' store context before, which counted a comprehension's own
    variable and a bare annotation and missed ``except ... as``.
    """
    if any(not binding.nested for binding in pathcond.bindings(func, name)):
        return True
    return any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Del)
        for node in own_nodes(func)
    )


def _self_field(node: ast.AST) -> str:
    """``status_code`` for ``self.status_code``, ``x`` for ``self.x[k]``; else ``""``."""
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return ""


#: Calls whose arguments are classes to test against, not to build.
_CLASS_TESTS = frozenset({"isinstance", "issubclass", "super"})


def _may_build_from(node: ast.expr) -> bool:
    """Whether *node*, a name or attribute read, holds a value a later call may build from -- :meth:`Calls.read_as_value`.

    Not a call's callee or an attribute's object -- those use it here -- and
    not a place that only tests or declares a class: a base list, a class
    statement's keywords, ``isinstance``/``issubclass``/``super``, an
    ``except`` clause, a comparison, an annotation.
    """
    parent = getattr(node, "parent", None)
    if isinstance(parent, ast.Call) and parent.func is node:
        return False
    if isinstance(parent, ast.Attribute) and parent.value is node:
        return False
    if isinstance(parent, ast.Compare):
        return False
    child: ast.AST = node
    while parent is not None and not isinstance(parent, ast.stmt):
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id in _CLASS_TESTS:
            return False
        if isinstance(parent, ast.ExceptHandler):
            return False
        if isinstance(parent, ast.arg) and child is parent.annotation:
            return False
        child, parent = parent, getattr(parent, "parent", None)
    if isinstance(parent, ast.ClassDef) and child in (*parent.bases, *parent.keywords):
        return False
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not None and child is parent.returns:
        return False
    if isinstance(parent, ast.AnnAssign) and child is parent.annotation:
        return False
    if isinstance(parent, ast.ExceptHandler):
        return False
    return True


def _unquoted(annotation: ast.expr) -> ast.expr:
    """"SiteMapper" as the expression it quotes; anything else as it is."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            return ast.parse(annotation.value.strip(), mode="eval").body
        except SyntaxError:
            return annotation
    return annotation

def _element_annotation(annotation: ast.expr) -> Optional[ast.expr]:
    """The element type a container annotation states: ``dict[K, V]`` -> ``V``, ``list[T]`` -> ``T``."""
    annotation = _unquoted(annotation)
    if not isinstance(annotation, ast.Subscript):
        return None
    wrapper = annotation.value
    name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
    inner = annotation.slice
    if name in ("dict", "Dict", "Mapping", "MutableMapping", "defaultdict", "OrderedDict"):
        return inner.elts[1] if isinstance(inner, ast.Tuple) and len(inner.elts) == 2 else None
    if name in ("list", "List", "Sequence", "set", "Set", "frozenset", "Iterable"):
        return inner
    return None


#: Containers a loop takes its target from, by the annotation's name -- :meth:`Types._iterated`.
_MAPPINGS = frozenset({"dict", "Dict", "Mapping", "MutableMapping", "defaultdict", "OrderedDict"})
_SEQUENCES = frozenset(
    {"list", "List", "Sequence", "set", "Set", "frozenset", "FrozenSet", "Iterable", "Iterator", "Generator"}
)


def _slot_of(annotation: ast.expr, slot: Optional[int]) -> Optional[ast.expr]:
    """Slot *slot* of a ``tuple[A, B]`` annotation; ``None`` for any other, or ``tuple[A, ...]``."""
    annotation = _unquoted(annotation)
    if (
        slot is None
        or not isinstance(annotation, ast.Subscript)
        or getattr(annotation.value, "id", getattr(annotation.value, "attr", "")) not in ("tuple", "Tuple")
        or not isinstance(annotation.slice, ast.Tuple)
        or slot >= len(annotation.slice.elts)
        or any(isinstance(one, ast.Constant) and one.value is Ellipsis for one in annotation.slice.elts)
    ):
        return None
    return annotation.slice.elts[slot]


#: What a context manager is declared to yield its element through.
_YIELDING = frozenset(
    {"Iterator", "Generator", "ContextManager", "AbstractContextManager", "Iterable"}
)

#: The classes a declaration states -- one, or each member of a union.
Classes = frozenset[tuple[str, ast.ClassDef]]


def _one(classes: Optional[Classes]) -> Optional[tuple[str, ast.ClassDef]]:
    """The class of a declaration that states exactly one."""
    if classes is None or len(classes) != 1:
        return None
    return next(iter(classes))


def _class_key(one: tuple[str, ast.ClassDef]) -> tuple[str, str, int]:
    return one[0], one[1].name, one[1].lineno


def _union_members(annotation: ast.expr) -> Optional[list[ast.expr]]:
    """The members of ``A | B``, ``Union[A, B]`` or ``Optional[A]``; ``None`` for anything else."""
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return [annotation.left, annotation.right]
    if isinstance(annotation, ast.Subscript):
        wrapper = annotation.value
        name = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
        if name == "Optional":
            return [annotation.slice]
        if name == "Union":
            inner = annotation.slice
            return list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
    return None


def _module_statements(body: list[ast.stmt]):
    """Every statement run at module level: the module's own, and those under its ``if``, ``try`` and ``with``."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            yield from _module_statements(node.body)
            yield from _module_statements(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _module_statements(node.body)
            for handler in node.handlers:
                yield from _module_statements(handler.body)
            yield from _module_statements(node.orelse)
            yield from _module_statements(node.finalbody)
        elif isinstance(node, ast.With):
            yield from _module_statements(node.body)


class Types:
    """What class the code *declares* a receiver to be.

    Read the way a type checker reads it -- an annotation, a constructor, the
    return annotation of the method called -- and nothing else.  The walk
    needs it for one question, which system is on the other side of a call:
    ``self.site_mapper.getSite(x)`` and ``self.siteMapper.getSite(x)`` are one
    interface spelled two ways, and matching the spelling classified one and
    walked straight through the other.  What the object *is* does not depend
    on what someone called it.

    Where the code declares ``Any``, a type checker learns nothing more; this
    asks the calls what they hand (:meth:`Calls.handed_classes`), since PanDA
    writes ``Any`` where the classes that arrive differ by caller.

    Its own record of what it read, apart from the walk's: reading a base
    class or the interface's own module to learn a type is not a step on the
    path, so it does not spend the walk's file budget.  The trees are the
    same ones -- :func:`parsed` -- because ``pathcond.class_of`` finds a
    method by identity.
    """

    #: How deep a declaration is followed through fields, locals and calls.
    #: JEDI's ``self.siteMapper`` is four links from what declares it -- the
    #: field's assignment, the call, ``self.taskBufferIF``'s assignment, the
    #: ``__init__`` parameter's annotation -- and each of those is two levels
    #: here, so four stopped one short of the answer.
    DEPTH = 8

    #: How many classes a lineage holds before it stops.  ``DBProxy`` is
    #: thirteen mixins and itself; at twelve the last two were cut, and 38
    #: ``proxy.<method>()`` calls read as "the declared class has no such
    #: method".
    LINEAGE = 40

    def __init__(self, roots: dict[str, Path], walked: dict[str, tuple[str, ast.Module]]):
        self.roots = roots
        self.walked = walked
        self.parsed: dict[str, Optional[ast.Module]] = {}
        self._imports: dict[str, dict[str, tuple[str, str]]] = {}
        self._bound: dict[str, dict[str, list[ast.expr]]] = {}
        self._assigned: dict[int, dict[str, list[ast.expr]]] = {}
        # (file, id(func), name, depth) -> (func, classes): see ``declared``.
        self._names: dict[tuple, tuple[ast.AST, Optional[Classes]]] = {}
        self._files: dict[int, str] = {}
        self._mro: dict[int, tuple[ast.ClassDef, list[tuple[str, ast.ClassDef]]]] = {}
        #: The classes the calls of a function hand its parameter declared
        #: ``Any`` -- :meth:`Calls.handed_classes`, set by the :class:`Calls`
        #: that reads this tree; ``None`` reads the declaration alone.
        self.handed_to: Optional[Callable[[str, ast.AST, str, int], Optional[Classes]]] = None

    def assigned(self, func: ast.AST) -> dict[str, list[ast.expr]]:
        """``pathcond.assigned_expressions``, once per function."""
        if id(func) not in self._assigned:
            self._assigned[id(func)] = pathcond.assigned_expressions(func)
        return self._assigned[id(func)]

    # -- modules -----------------------------------------------------------

    def tree(self, file: str) -> Optional[ast.Module]:
        if file in self.walked:
            return self.walked[file][1]
        if file not in self.parsed:
            package, _, relative = file.partition("/")
            root = self.roots.get(package)
            held = parsed(root / relative) if root is not None and relative else None
            self.parsed[file] = held[1] if held is not None else None
        return self.parsed[file]

    def file_of(self, node: ast.AST) -> Optional[str]:
        """The corpus file *node* belongs to, if it is in a tree this reader holds."""
        module = node
        while getattr(module, "parent", None) is not None:
            module = module.parent  # type: ignore[attr-defined]
        if not isinstance(module, ast.Module):
            return None
        if id(module) not in self._files:
            self._files.update({id(tree): file for file, (_sha, tree) in self.walked.items()})
            self._files.update({id(tree): file for file, tree in self.parsed.items() if tree is not None})
        return self._files.get(id(module))

    def corpus_files(self) -> list[str]:
        """Every module of the corpus, as the map spells a file; tests left out, by the build's rule (:func:`is_test_module`)."""
        found: list[str] = []
        for package, root in sorted(self.roots.items()):
            for path in sorted(root.rglob("*.py")):
                file = f"{package}/{path.relative_to(root)}"
                if not is_test_module(file):
                    found.append(file)
        return found

    def enclosing_function(self, file: str, node: ast.AST) -> Optional[ast.AST]:
        """The function *node* is nested in, or ``None`` at module or class scope."""
        parent = getattr(node, "parent", None)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return parent
            if isinstance(parent, (ast.ClassDef, ast.Module)):
                return None
            parent = getattr(parent, "parent", None)
        return None

    def method_class(self, file: str, func: ast.AST) -> Optional[tuple[str, ast.ClassDef]]:
        """The class whose ``self`` *func* reads: its own, or its enclosing method's.

        ``getJobs`` builds its ``WHERE`` in a closure, ``build_where_clause``,
        which calls ``self.construct_where_clause`` -- the ``self`` of the
        method it is nested in.
        """
        node: Optional[ast.AST] = func
        while node is not None:
            parent = getattr(node, "parent", None)
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return file, parent
            if isinstance(parent, ast.Module) or parent is None:
                break
            node = parent
        tree = self.tree(file)
        if tree is not None:
            cls = pathcond.class_of(tree, func)
            if cls is not None:
                return file, cls
        return None

    def imported_from_outside(self, file: str, name: str) -> bool:
        """Whether *name* in *file* is imported from a module this corpus does not hold."""
        held = self.imports(file).get(name)
        if held is None:
            return False
        module = held[0]
        return module.split(".")[0] not in self.roots

    def module_named(self, file: str, expression: ast.expr) -> Optional[str]:
        """The file of the module *expression* names in *file*, when it names one."""
        if not isinstance(expression, ast.Name):
            return None
        held = self.imports(file).get(expression.id)
        if held is None:
            return None
        module, attribute = held
        if not attribute:
            return self.module_file(module)
        return self.module_file(f"{module}.{attribute}")

    def module_file(self, dotted: str) -> Optional[str]:
        """``pandaserver.brokerage.SiteMapper`` -> its file, when the tree holds it."""
        parts = dotted.split(".")
        if not parts or parts[0] not in self.roots:
            return None
        for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
            if self.tree(candidate) is not None:
                return candidate
        return None

    def imports(self, file: str) -> dict[str, tuple[str, str]]:
        """``{local name: (module, attribute)}`` for every import in *file*.

        Every import, wherever it sits: ``if TYPE_CHECKING:`` is where PanDA
        puts the ones only an annotation needs, and those are the ones this
        reads.  A relative import is resolved against the file's own package.
        """
        if file in self._imports:
            return self._imports[file]
        found: dict[str, tuple[str, str]] = {}
        tree = self.tree(file)
        package = file[: -len(".py")].split("/")[:-1] if file.endswith(".py") else []
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        found[alias.asname] = (alias.name, "")
                    else:
                        head = alias.name.split(".")[0]
                        found.setdefault(head, (head, ""))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                    module = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    module = node.module or ""
                for alias in node.names:
                    found[alias.asname or alias.name] = (module, alias.name)
        self._imports[file] = found
        return found

    # -- classes -----------------------------------------------------------

    def class_named(self, file: str, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression* names in *file*: declared there, imported, or bound to a name there.

        A name the module neither declares nor imports is read from what the
        module binds it to -- ``_Proxied: Any = JediTaskBuffer``, so that
        ``class JediTaskBufferInterface(_Proxied)`` has its base.  The value
        is what the name is, whatever the annotation says: ``Any`` says
        nothing.  A binding to no class of the corpus -- ``_Proxied =
        object`` in the other branch -- is passed over; two bindings to two
        corpus classes name neither.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == expression.id:
                    return file, node
            held = self.imports(file).get(expression.id)
            if held is None:
                return self._aliased(file, expression.id, depth)
            module, attribute = held
            if not attribute:
                return None
            # ``from pkg.mod import Cls`` -- or ``from pkg import mod``, a module.
            target = self.module_file(module)
            if target is not None:
                return self.class_named(target, ast.Name(id=attribute, ctx=ast.Load()), depth + 1)
            return None
        if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
            # ``SiteMapper.SiteMapper`` -- a module imported, then its class.
            held = self.imports(file).get(expression.value.id)
            if held is None:
                return None
            module, attribute = held
            dotted = f"{module}.{attribute}" if attribute else module
            target = self.module_file(dotted)
            if target is None:
                return None
            return self.class_named(target, ast.Name(id=expression.attr, ctx=ast.Load()), depth + 1)
        return None

    def _aliased(self, file: str, name: str, depth: int) -> Optional[tuple[str, ast.ClassDef]]:
        """The corpus class every module-level binding of *name* in *file* that names one names."""
        found = {
            named
            for value in self._module_bound(file).get(name, [])
            if (named := self.class_named(file, value, depth + 1)) is not None
        }
        return found.pop() if len(found) == 1 else None

    def _module_bound(self, file: str) -> dict[str, list[ast.expr]]:
        """``{name: [value]}`` for each ``name = value`` / ``name: T = value`` run at module level, the value a name or ``a.b``."""
        if file not in self._bound:
            found: dict[str, list[ast.expr]] = {}
            tree = self.tree(file)
            for node in _module_statements(tree.body) if tree is not None else ():
                if isinstance(node, ast.Assign):
                    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                    names = [node.target.id]
                else:
                    continue
                if isinstance(node.value, (ast.Name, ast.Attribute)):
                    for one in names:
                        found.setdefault(one, []).append(node.value)
            self._bound[file] = found
        return self._bound[file]

    def annotated(self, file: str, annotation: Optional[ast.expr]) -> Optional[tuple[str, ast.ClassDef]]:
        """The class an annotation states, where it states exactly one -- :meth:`annotated_all`."""
        return _one(self.annotated_all(file, annotation))

    def annotated_all(self, file: str, annotation: Optional[ast.expr]) -> Optional[Classes]:
        """Every corpus class an annotation states, through quotes, ``Optional`` and unions.

        ``WrappedOracleConn | WrappedPostgresConn`` is both; ``Foo | None`` and
        ``Foo | str`` are ``Foo`` -- a member naming no class of the corpus
        says nothing this reading can follow.  ``None`` where it states none.
        """
        if annotation is None:
            return None
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            found: set[tuple[str, ast.ClassDef]] = set()
            for member in members:
                if isinstance(member, ast.Constant) and member.value is None:
                    continue
                found |= self.annotated_all(file, member) or frozenset()
            return frozenset(found) or None
        if isinstance(annotation, ast.Subscript):
            return None
        named = self.class_named(file, annotation)
        return frozenset({named}) if named is not None else None

    def states_any(self, file: str, annotation: Optional[ast.expr], depth: int = 0) -> bool:
        """Whether *annotation* declares ``Any`` and nothing else, ``| None`` aside -- spelled so, or by an alias of it.

        ``TaskBufferLike: TypeAlias = Any`` in ``pandaserver.asyncprocess``
        is one: the alias says why the parameter is left open.
        """
        if annotation is None or depth > 2:
            return False
        annotation = _unquoted(annotation)
        members = _union_members(annotation)
        if members is not None:
            rest = [one for one in members if not (isinstance(one, ast.Constant) and one.value is None)]
            return bool(rest) and all(self.states_any(file, one, depth) for one in rest)
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == "Any"
        if not isinstance(annotation, ast.Name):
            return False
        held = self.imports(file).get(annotation.id)
        if annotation.id == "Any":
            return held is None or held[0] == "typing"
        where, alias = file, annotation.id
        if held is not None:
            where, alias = self.module_file(held[0]) or "", held[1] or annotation.id
        tree = self.tree(where) if where else None
        for node in tree.body if tree is not None else ():
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == alias:
                return node.value is not None and self.states_any(where, node.value, depth + 1)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == alias for t in node.targets):
                return self.states_any(where, node.value, depth + 1)
        return False

    def lineage(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its bases, nearest first, followed across files through imports."""
        order: list[tuple[str, ast.ClassDef]] = [(file, cls)]
        queue = [(file, base) for base in cls.bases]
        seen = {(file, cls.name)}
        while queue and len(order) < self.LINEAGE:
            where, base = queue.pop(0)
            found = self.class_named(where, pathcond.unsubscripted(base))
            if found is None or (found[0], found[1].name) in seen:
                continue
            seen.add((found[0], found[1].name))
            order.append(found)
            queue.extend((found[0], one) for one in found[1].bases)
        return order

    def mro(self, file: str, cls: ast.ClassDef) -> list[tuple[str, ast.ClassDef]]:
        """*cls* and its corpus bases in the order Python resolves a method on an object of *cls*.

        :meth:`lineage` reads the bases breadth first, and in the corpus that
        finds the same definition of every name a lookup asks.  ``super()``
        needs the order itself: in a class with several bases it runs the
        class after the caller's in the order of the object's class, so
        ``OraDBProxy.DBProxy``'s thirteen mixins each hand the call on to the
        next.  C3, as Python builds it; a base outside the corpus stands for
        itself, its own bases unknown.  Where the bases admit no such order,
        :meth:`lineage`.
        """
        held = self._mro.get(id(cls))
        if held is not None and held[0] is cls:
            return held[1]
        # A class that reaches itself through its bases reads its lineage.
        self._mro[id(cls)] = (cls, self.lineage(file, cls))
        found = self._linearized(file, cls)
        order = found if found is not None else self.lineage(file, cls)
        self._mro[id(cls)] = (cls, order)
        return order

    def _linearized(self, file: str, cls: ast.ClassDef) -> Optional[list[tuple[str, ast.ClassDef]]]:
        """C3 over *cls*'s bases, ``None`` where they admit no order."""
        held: dict[object, tuple[str, ast.ClassDef]] = {}

        def token(one: tuple[str, ast.ClassDef]) -> object:
            held[id(one[1])] = one
            return id(one[1])

        direct: list[object] = []
        sequences: list[list[object]] = []
        for base in cls.bases:
            found = self.class_named(file, pathcond.unsubscripted(base))
            if found is None:
                outside = f"<outside>{ast.unparse(base)}"
                direct.append(outside)
                sequences.append([outside])
                continue
            direct.append(token(found))
            sequences.append([token(one) for one in self.mro(*found)])
        sequences.append(list(direct))
        merged: list[object] = [token((file, cls))]
        while True:
            sequences = [one for one in sequences if one]
            if not sequences:
                break
            for sequence in sequences:
                head = sequence[0]
                if not any(head in other[1:] for other in sequences):
                    break
            else:
                return None
            merged.append(head)
            for sequence in sequences:
                if sequence[0] == head:
                    del sequence[0]
        return [held[one] for one in merged if one in held]

    def method(self, file: str, cls: ast.ClassDef, name: str):
        for where, holder in self.lineage(file, cls):
            for node in holder.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    return where, node
        return None

    # -- declarations ------------------------------------------------------

    def of_field(self, file: str, cls: ast.ClassDef, field: str, depth: int) -> Optional[Classes]:
        """``self.<field>``'s classes: its annotation in the class or a base, else what every assignment says."""
        lineage = self.lineage(file, cls)
        for where, holder in lineage:
            for node in holder.body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == field
                ):
                    return self.annotated_all(where, node.annotation)
            for method in pathcond.methods_of(holder):
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return self.annotated_all(where, node.annotation)
        if depth >= self.DEPTH:
            return None
        found: set[Classes] = set()
        for where, holder in lineage:
            for method in pathcond.methods_of(holder):
                for expression in pathcond.attribute_expressions(method, field):
                    if isinstance(expression, ast.Constant) and expression.value is None:
                        continue
                    one = self.declared(where, method, expression, depth + 1)
                    if one is None:
                        return None
                    found.add(one)
        return found.pop() if len(found) == 1 else None

    def _with_held(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """The classes ``with <call> as <name>`` binds, from what the call is declared to yield.

        Every ``TaskBuffer`` method reaches the database through ``with
        self.proxyPool.get() as proxy``, and ``get`` is declared ``->
        Iterator["DBProxy.DBProxy"]``: the object is the element, not the
        iterator.
        """
        for node in ast.walk(func):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                held = item.optional_vars
                if not (isinstance(held, ast.Name) and held.id == name):
                    continue
                context = item.context_expr
                if not (isinstance(context, ast.Call) and isinstance(context.func, ast.Attribute)):
                    return self.declared(file, func, context, depth + 1)
                receiver = context.func.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], context.func.attr)
                if found is None:
                    return None
                where, method = found
                returned = _unquoted(method.returns) if method.returns is not None else None
                if isinstance(returned, ast.Subscript):
                    wrapper = returned.value
                    wrapped = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
                    if wrapped in _YIELDING:
                        inner = returned.slice
                        if isinstance(inner, ast.Tuple) and inner.elts:
                            inner = inner.elts[0]
                        return self.annotated_all(where, inner)
                return self.annotated_all(where, returned)
        return None

    def _annotation_of(
        self, file: str, func: ast.AST, expression: ast.expr, seen: frozenset = frozenset()
    ) -> Optional[tuple[str, ast.expr]]:
        """The annotation *expression* was declared with: a field's, a parameter's or a local's."""
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            return self._field_annotation(holder, expression.attr) if holder is not None else None
        if isinstance(expression, ast.Name):
            args = getattr(func, "args", None)
            for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
                if argument.arg == expression.id and argument.annotation is not None:
                    return file, argument.annotation
            for node in ast.walk(func):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == expression.id:
                    return file, node.annotation
            # ``target = self.nuclei`` or ``target = self.satellites``: an
            # alias carries what every value it is bound to was declared as.
            bound = self.assigned(func).get(expression.id, [])
            if bound and expression.id not in seen:
                found = [
                    self._annotation_of(file, func, value, seen | {expression.id}) for value in bound
                ]
                texts = {ast.unparse(one[1]) for one in found if one is not None}
                if len(texts) == 1 and all(one is not None for one in found):
                    return found[0]
        return None

    def _field_annotation(self, holder: tuple[str, ast.ClassDef], field: str) -> Optional[tuple[str, ast.expr]]:
        """The annotation *holder*'s class or a base declares *field* with: in the class body or as ``self.<field>: T``."""
        for where, cls in self.lineage(*holder):
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == field:
                    return where, node.annotation
            for method in pathcond.methods_of(cls):
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                        and node.target.attr == field
                    ):
                        return where, node.annotation
        return None

    def _function_named(self, file: str, name: str) -> Optional[tuple[str, ast.AST]]:
        """The module-level function *name* is in *file*: defined there or imported."""
        tree = self.tree(file)
        if tree is None:
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return file, node
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        other = self.tree(target)
        for node in other.body if other is not None else ():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == held[1]:
                return target, node
        return None

    def _outside_instance(self, file: str, name: str, depth: int) -> Optional[Classes]:
        """The classes of a name this function does not bind: a module's own instance.

        ``taskBuffer = TaskBuffer()`` at the foot of ``TaskBuffer.py`` and
        ``rucioAPI = RucioAPI()`` at the foot of ``ddm.py`` are what every
        daemon imports and calls; the module's own assignment says what it is.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        for node in tree.body:
            # ``global_task_buffer: TaskBuffer = None``: the annotation is the
            # declaration, whatever the placeholder value is.
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                declared = self.annotated_all(file, node.annotation)
                if declared is not None:
                    return declared
        values = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ]
        if values:
            found = {self.declared(file, tree, value, depth + 1) for value in values}
            return found.pop() if len(found) == 1 else None
        return self._imported_instance(file, name, depth)

    def _imported_instance(self, file: str, name: str, depth: int) -> Optional[Classes]:
        """:meth:`_outside_instance` of the module *file* imports *name* from."""
        held = self.imports(file).get(name)
        if held is None or not held[1]:
            return None
        target = self.module_file(held[0])
        if target is None or target == file:
            return None
        return self._outside_instance(target, held[1], depth + 1)

    def _of_name(self, file: str, func: ast.AST, name: str, depth: int) -> Optional[Classes]:
        """:meth:`declared` for a bare name, read in *func*."""
        args = getattr(func, "args", None)
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs) if args else ():
            if argument.arg == name:
                if self.handed_to is not None and self.states_any(file, argument.annotation):
                    return self.handed_to(file, func, name, depth)
                return self.annotated_all(file, argument.annotation)
        for node in ast.walk(func):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                return self.annotated_all(file, node.annotation)
        # Every assignment, less the ``None`` a name is cleared with:
        # ``siteMapper = SiteMapper(taskBuffer)`` and, at the end of the
        # daemon, ``siteMapper = None``.  The evidence for a local is all in
        # this function, so it is read rather than asked for; the answer
        # is classes only where every assignment says the same ones.  So
        # are the bindings an assignment is not (:meth:`_other_bindings`).
        bound = self._assigned_here(func, name)
        others = self._other_bindings(file, func, name, depth)
        held = self._with_held(file, func, name, depth)
        if not bound and held is None and not others:
            # A name a module's own body reads and does not bind is one it
            # imports: ``initializer.init()`` and ``taskBuffer.init(...)`` at
            # the top of ``panda.py``, a module's caller (:class:`CallSite`).
            if isinstance(func, ast.Module):
                return self._imported_instance(file, name, depth)
            return self._outside_instance(file, name, depth)
        found = {self.declared(file, func, value, depth + 1) for value in bound}
        found.update(one for one in others if one is not None)
        if held is not None:
            found.add(held)
        return found.pop() if len(found) == 1 else None

    def _assigned_here(self, func: ast.AST, name: str) -> list[ast.expr]:
        """What *func*'s own assignments bind *name* to, less the ``None`` it is cleared with.

        Not a function's, lambda's or class's nested in it: ``taskBuffer =
        ...`` inside ``add_main.main``'s inner function binds that function's
        own name.
        """
        nested = {id(one.value) for one in pathcond.bindings(func, name) if one.nested and one.value is not None}
        return [
            value
            for value in self.assigned(func).get(name, [])
            if not (isinstance(value, ast.Constant) and value.value is None) and id(value) not in nested
        ]

    def _other_bindings(self, file: str, func: ast.AST, name: str, depth: int) -> list[Optional[Classes]]:
        """What each binding of *name* in *func* that is not an assignment makes it, by its declarations; ``None`` for one they do not say.

        A loop's target is an element of what it iterates
        (:meth:`_iterated`): ``for file_spec in job.Files``, ``Files:
        list[FileSpec]``.  A name unpacked from a call is its slot of the
        ``tuple[...]`` the callee returns; ``:=`` is an assignment;
        ``except X as e`` is an ``X``.  Read by the declarations, as the
        rest of this class reads -- not by which binding reaches the call.
        A binding they do not type adds nothing and takes nothing away, as
        before these were read: ``for tmpNucleusSpec in nucleusList.items()``
        on an undeclared ``nucleusList`` beside a loop over ``dict[str,
        NucleusSpec]``.  Saying "open" for it lost 65 placements whose
        readings were right (X8: 33 closed, 32 builtin), where the 154 the
        rule left closed held no wrong one (X7).  A ``with`` is
        :meth:`_with_held`'s; a nested scope's binding is its own.

        A parameter is not one of these, nor an import inside the function:
        where the function also assigns the name, the assignments answer
        alone.  ``add_main.main``'s ``taskBuffer = tbuf`` is a ``TaskBuffer``
        by ``tbuf = TaskBuffer()`` further down, while the daemon may pass the
        interface proxy (``daemons/utils.py``) -- which hands the call by its
        name to a ``TaskBuffer`` in another process, so the method, the
        arguments and the constants are the ones placed here and the process
        is not.  18 placements rest on such a name (X8), 6 of them behind
        that proxy.
        """
        found: list[Optional[Classes]] = []
        for one in pathcond.bindings(func, name):
            if one.nested:
                continue
            if one.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED, pathcond.BIND_AUGMENTED, pathcond.BIND_WITH):
                if not one.unpacked or one.kind == pathcond.BIND_WITH:
                    continue
                found.append(self._unpacked(file, func, one.value, one.slot, depth))
            elif one.kind == pathcond.BIND_WALRUS:
                found.append(self.declared(file, func, one.value, depth + 1) if one.value is not None else None)
            elif one.kind == pathcond.BIND_EXCEPT:
                found.append(self._caught(file, one.site))
            elif one.kind == pathcond.BIND_FOR:
                loop = one.site
                found.append(
                    self._iterated(file, func, loop.iter, one.slot if one.unpacked else None, one.unpacked, depth)  # type: ignore[attr-defined]
                )
            else:
                found.append(None)
        return found

    def _unpacked(
        self, file: str, func: ast.AST, value: Optional[ast.expr], slot: Optional[int], depth: int
    ) -> Optional[Classes]:
        """The classes slot *slot* of *value* is declared as: a tuple literal's element, or a call's ``tuple[...]``."""
        if value is None or slot is None:
            return None
        if isinstance(value, (ast.Tuple, ast.List)):
            return self.declared(file, func, value.elts[slot], depth + 1) if slot < len(value.elts) else None
        returned = self._returned(file, func, value, depth)
        if returned is None:
            return None
        where, annotation = returned
        element = _slot_of(annotation, slot)
        return self.annotated_all(where, element) if element is not None else None

    def _returned(self, file: str, func: ast.AST, value: ast.expr, depth: int) -> Optional[tuple[str, ast.expr]]:
        """``(file, return annotation)`` of the function *value*, a call, lands in, read as :meth:`declared` reads it."""
        if not isinstance(value, ast.Call):
            return None
        called = value.func
        if isinstance(called, ast.Name):
            function = self._function_named(file, called.id)
            definition = function[1] if function is not None else None
            where = function[0] if function is not None else file
        elif isinstance(called, ast.Attribute):
            receiver = called.value
            if isinstance(receiver, ast.Name) and receiver.id == "self":
                owner = self.method_class(file, func)
            else:
                owner = self.of_expression(file, func, receiver, depth + 1)
            found = self.method(owner[0], owner[1], called.attr) if owner is not None else None
            where, definition = found if found is not None else (file, None)
        else:
            return None
        returns = getattr(definition, "returns", None)
        return (where, returns) if returns is not None else None

    def _caught(self, file: str, handler: ast.AST) -> Optional[Classes]:
        """The corpus classes ``except X as e`` binds ``e`` to; ``None`` where one is not the corpus's."""
        caught = getattr(handler, "type", None)
        members = caught.elts if isinstance(caught, ast.Tuple) else [caught] if caught is not None else []
        found = [self.class_named(file, one) for one in members]
        if not found or any(one is None for one in found):
            return None
        return frozenset(found)  # type: ignore[arg-type]

    def _iterated(
        self, file: str, func: ast.AST, iterable: ast.expr, slot: Optional[int], unpacked: bool, depth: int
    ) -> Optional[Classes]:
        """The classes a loop's target is, from what it iterates; slot *slot* of each element where it is unpacked.

        ``list(x)``, ``sorted(x)``, ``reversed(x)`` iterate what ``x`` does;
        ``enumerate(x)``'s second slot is ``x``'s element.  ``d.items()``,
        ``.values()`` and ``.keys()`` on a ``dict[K, V]`` are ``(K, V)``,
        ``V`` and ``K``; a container is its element, a mapping its key.
        """
        if isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and len(iterable.args) == 1:
            if iterable.func.id in ("list", "sorted", "reversed", "tuple", "set", "frozenset") and not unpacked:
                return self._iterated(file, func, iterable.args[0], None, False, depth)
            if iterable.func.id == "enumerate" and unpacked and slot == 1:
                return self._iterated(file, func, iterable.args[0], None, False, depth)
            return None
        view = ""
        source: ast.expr = iterable
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Attribute)
            and iterable.func.attr in ("items", "values", "keys")
            and not iterable.args
        ):
            view, source = iterable.func.attr, iterable.func.value
        declared = self._annotation_of(file, func, source)
        if declared is None and isinstance(source, ast.Attribute):
            # ``job.Files``, on a ``job`` declared a ``JobSpec``.
            owner = self.of_expression(file, func, source.value, depth + 1)
            declared = self._field_annotation(owner, source.attr) if owner is not None else None
        if declared is None:
            declared = self._returned(file, func, source, depth)
        if declared is None:
            return None
        where, annotation = declared
        annotation = _unquoted(annotation)
        if not isinstance(annotation, ast.Subscript):
            return None
        wrapper = annotation.value
        kind = wrapper.attr if isinstance(wrapper, ast.Attribute) else getattr(wrapper, "id", "")
        inner = annotation.slice
        if kind in _MAPPINGS:
            if not (isinstance(inner, ast.Tuple) and len(inner.elts) == 2):
                return None
            key, value = inner.elts
            if view == "items":
                element = (key, value)[slot] if unpacked and slot in (0, 1) else None
            else:
                element = value if view == "values" else key
                element = None if unpacked else element
        elif kind in _SEQUENCES and not view:
            element = inner if not unpacked else _slot_of(inner, slot)
        else:
            element = None
        return self.annotated_all(where, element) if element is not None else None

    def of_expression(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[tuple[str, ast.ClassDef]]:
        """The class *expression*, read in *func* of *file*, is declared to be, where it is declared one.

        The walk's typer asks this, and so does the owner of a method whose
        return annotation is read: a call on an object declared as two
        classes returns what either method does, and that is not read.
        """
        return _one(self.declared(file, func, expression, depth))

    def declared(self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0) -> Optional[Classes]:
        """The classes *expression*, read in *func* of *file*, is declared to be.

        One, or each member of a union.  A union is read wherever a single
        class is -- a field's, a parameter's or a local's annotation, a
        module's instance, what a ``with`` yields, a container's element, a
        return annotation -- and where every assignment of a name or field
        must say the same, it must say the same classes.
        """
        if depth > self.DEPTH:
            return None
        tree = self.tree(file)
        if tree is None:
            return None
        if isinstance(expression, ast.Name):
            # What a name is depends on the function, the name and how deep
            # the question already is -- not on which ``Name`` asked.
            # ``JobGenerator`` rebinds locals from calls on locals that are
            # rebound too, and reading each afresh cost the product of their
            # assignment counts: a census stopped in that one file for hours.
            # The function is kept with the answer, so its id is not reused.
            key = (file, id(func), expression.id, depth)
            hit = self._names.get(key)
            if hit is not None and hit[0] is func:
                return hit[1]
            answer = self._of_name(file, func, expression.id, depth)
            self._names[key] = (func, answer)
            return answer
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            cls = pathcond.class_of(tree, func)
            return self.of_field(file, cls, expression.attr, depth + 1) if cls is not None else None
        if isinstance(expression, ast.Subscript):
            # ``self.nuclei[name]``, where ``self.nuclei: dict[str, NucleusSpec]``.
            container = self._annotation_of(file, func, expression.value)
            if container is not None:
                where, annotation = container
                element = _element_annotation(annotation)
                if element is not None:
                    return self.annotated_all(where, element)
            return None
        if isinstance(expression, ast.Call):
            called = expression.func
            constructed = self.class_named(file, called) if isinstance(called, (ast.Name, ast.Attribute)) else None
            if constructed is not None:
                return frozenset({constructed})
            hook = self._copy_hook(file, called)
            if hook is not None:
                # ``jobSpec = copy.copy(job)``: what ``job`` is declared to be.
                copied = self.declared(file, func, expression.args[0], depth + 1) if expression.args else None
                return self._copied(copied, hook)
            if isinstance(called, ast.Name):
                # ``get_task_utils_module(self)``, declared ``-> TaskUtilsModule``.
                function = self._function_named(file, called.id)
                if function is not None:
                    where, definition = function
                    return self.annotated_all(where, definition.returns)
                return None
            if isinstance(called, ast.Attribute):
                receiver = called.value
                if isinstance(receiver, ast.Name) and receiver.id == "self":
                    owner = self.method_class(file, func)
                else:
                    owner = self.of_expression(file, func, receiver, depth + 1)
                if owner is None:
                    return None
                found = self.method(owner[0], owner[1], called.attr)
                if found is None:
                    return None
                where, method = found
                return self.annotated_all(where, method.returns)
        return None

    # -- what a value can be, where nothing declares it --------------------

    def candidates(
        self, file: str, func: ast.AST, expression: ast.expr, depth: int = 0
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """``({(file, class name)}, open)``: the classes *expression* can be an instance of.

        Asked where the declaration says nothing -- no annotation, or ``Any``.
        ``AdderGen`` declares ``self.adder_plugin: Any`` and builds it as
        ``self.adder_plugin_class(...)``, where ``adder_plugin_class`` is what
        ``get_plugin_class`` returns: ``AdderAtlasPlugin``, or a class the
        configuration names.  Read from the values rather than the
        declaration, that is ``{AdderAtlasPlugin}`` and *open* -- there is
        another way, and this reading cannot say which class it is.
        """
        declared = self.declared(file, func, expression, 0)
        if declared is not None:
            return frozenset((where, cls.name) for where, cls in declared), False
        return self._values(file, func, expression, depth, instances=True, seen=frozenset())

    def class_def(self, file: str, name: str) -> Optional[ast.ClassDef]:
        tree = self.tree(file)
        for node in ast.walk(tree) if tree is not None else ():
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def _values(
        self,
        file: str,
        func: ast.AST,
        expression: ast.expr,
        depth: int,
        *,
        instances: bool,
        seen: frozenset,
    ) -> tuple[frozenset[tuple[str, str]], bool]:
        """Classes an expression holds -- instances of, or the class objects themselves."""
        if depth > self.DEPTH:
            return frozenset(), True
        key = (file, id(func), ast.unparse(expression), instances)
        if key in seen:
            return frozenset(), False
        seen = seen | {key}
        found: set[tuple[str, str]] = set()
        opened = False

        def add(result: tuple[frozenset, bool]) -> None:
            nonlocal opened
            found.update(result[0])
            opened = opened or result[1]

        if isinstance(expression, ast.Constant) and expression.value is None:
            return frozenset(), False
        if instances and (builtin := self._builtin_instance(file, func, expression)):
            return frozenset({(BUILTIN, builtin)}), False
        if isinstance(expression, ast.Call):
            hook = self._copy_hook(file, expression.func) if instances else None
            if hook is not None:
                # A copy is an instance of what it copies; a class that copies
                # itself with the hook is what the hook returns, unread here.
                if not expression.args:
                    return frozenset(), True
                copied, opened = self._values(file, func, expression.args[0], depth + 1, instances=True, seen=seen)
                for where, name in copied:
                    cls = self.class_def(where, name) if where != BUILTIN else None
                    if cls is not None and self.method(where, cls, hook) is not None:
                        opened = True
                    else:
                        found.add((where, name))
                return frozenset(found), opened
            if instances:
                # An instance of whatever class the callee is: ``Cls(...)``,
                # or ``self.plugin_class(...)`` holding a class.
                add(self._values(file, func, expression.func, depth + 1, instances=False, seen=seen))
                return frozenset(found), opened
            # A call that returns a class: what its body returns.
            owner = self._owner_of(file, func, expression.func)
            if owner is None:
                return frozenset(), True
            where, method = owner
            for node in ast.walk(method):
                if isinstance(node, ast.Return) and node.value is not None:
                    add(self._values(where, method, node.value, depth + 1, instances=False, seen=seen))
            return frozenset(found), opened
        if isinstance(expression, ast.Name):
            if not instances:
                named = self.class_named(file, expression)
                if named is not None:
                    return frozenset({(named[0], named[1].name)}), False
            bound = self._assigned_here(func, expression.id)
            # A loop's target, a name unpacked or caught: the classes their
            # declarations say, as :meth:`declared` reads them; one they do
            # not type adds nothing (:meth:`_other_bindings`).
            others = self._other_bindings(file, func, expression.id, depth)
            if not bound and not others:
                return frozenset(), True
            for value in bound:
                # What a binding is declared as, where it is: ``site_spec_tmp
                # = self.siteSpecList[name]`` is a ``SiteSpec`` beside a loop
                # over a ``dict[Any, Any]`` -- read as :meth:`_of_name` reads
                # each binding, and from what it is built where nothing
                # declares it.
                typed = self.declared(file, func, value, depth + 1) if instances else None
                if typed is not None:
                    found.update((where, cls.name) for where, cls in typed)
                    continue
                add(self._values(file, func, value, depth + 1, instances=instances, seen=seen))
            readable = [one for one in others if one is not None and instances]
            if not bound and not readable:
                return frozenset(), True
            for one in readable:
                found.update((where, cls.name) for where, cls in one)
            return frozenset(found), opened
        if (
            isinstance(expression, ast.Attribute)
            and isinstance(expression.value, ast.Name)
            and expression.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return frozenset(), True
            any_value = False
            for where, cls in self.lineage(*holder):
                for method in pathcond.methods_of(cls):
                    for value in pathcond.attribute_expressions(method, expression.attr):
                        if isinstance(value, ast.Constant) and value.value is None:
                            continue
                        any_value = True
                        add(self._values(where, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened or not any_value
        if isinstance(expression, ast.Attribute):
            # ``obj.field`` where ``obj`` is one of some classes: what each assigns it.
            owners, owner_open = self.candidates(file, func, expression.value, depth + 1)
            opened = owner_open
            for where, name in owners:
                if where == BUILTIN:
                    # ``self.task_setupper_map = {}`` ... ``.get(key)``: what
                    # a builtin's method returns, or its field holds, is not
                    # read here.
                    opened = True
                    continue
                cls = self.class_def(where, name)
                if cls is None:
                    continue
                typed = self.of_field(where, cls, expression.attr, depth + 1)
                if typed is not None and instances:
                    found.update((one, held.name) for one, held in typed)
                    continue
                for place, holder in self.lineage(where, cls):
                    for method in pathcond.methods_of(holder):
                        for value in pathcond.attribute_expressions(method, expression.attr):
                            if isinstance(value, ast.Constant) and value.value is None:
                                continue
                            add(self._values(place, method, value, depth + 1, instances=instances, seen=seen))
            return frozenset(found), opened
        return frozenset(), True

    def _builtin_instance(self, file: str, func: ast.AST, expression: ast.expr) -> str:
        """The builtin type *expression* builds an instance of -- ``"list"`` for ``[]`` or ``sorted(x)`` -- or ``""``.

        A literal, a comprehension, a string constant; a call of a builtin
        type's name the module and the function leave unbound; a container
        class of the standard library, imported.  ``tmp_list.append(x)`` on
        such a value is a builtin's method, not a call to a corpus function
        named ``append`` -- ``ListWithLock.append`` had a thousand calls by
        its name that could not be told from it.
        """
        literal = (
            ((ast.List, ast.ListComp), "list"),
            ((ast.Dict, ast.DictComp), "dict"),
            ((ast.Set, ast.SetComp), "set"),
            ((ast.Tuple,), "tuple"),
            ((ast.JoinedStr,), "str"),
        )
        for shapes, kind in literal:
            if isinstance(expression, shapes):
                return kind
        if isinstance(expression, ast.Constant):
            return type(expression.value).__name__
        if not isinstance(expression, ast.Call):
            return ""
        called = expression.func
        if isinstance(called, ast.Name):
            held = self.imports(file).get(called.id)
            if held is not None:
                return held[1] if held in _LIBRARY_CONTAINERS else ""
            if called.id not in _BUILTIN_TYPES or called.id in self.assigned(func):
                return ""
            tree = self.tree(file)
            for node in tree.body if tree is not None else ():
                defined = (
                    [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    else [one.id for one in getattr(node, "targets", []) if isinstance(one, ast.Name)]
                )
                if called.id in defined:
                    return ""
            return _BUILTIN_TYPES[called.id]
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            held = self.imports(file).get(called.value.id)
            if held is not None and not held[1] and (held[0], called.attr) in _LIBRARY_CONTAINERS:
                return called.attr
        return ""

    def _copy_hook(self, file: str, called: ast.expr) -> Optional[str]:
        """The hook a class would copy itself with, where *called* is one of :data:`_COPIES`."""
        imported = self.imports(file)
        if isinstance(called, ast.Name):
            held = imported.get(called.id)
            return _COPIES.get(held) if held is not None else None
        if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name):
            held = imported.get(called.value.id)
            if held is not None and not held[1]:
                return _COPIES.get((held[0], called.attr))
        return None

    def _copied(self, copied: Optional[Classes], hook: str) -> Optional[Classes]:
        """What a copy of an object of the classes *copied* is declared to be.

        The same classes; where one defines *hook*, what the hook is declared
        to return, and nothing where it declares nothing.
        """
        if copied is None:
            return None
        found: set[tuple[str, ast.ClassDef]] = set()
        for where, cls in copied:
            hooked = self.method(where, cls, hook)
            if hooked is None:
                found.add((where, cls))
                continue
            returned = self.annotated_all(hooked[0], hooked[1].returns)
            if returned is None:
                return None
            found |= returned
        return frozenset(found)

    def _owner_of(self, file: str, func: ast.AST, called: ast.expr) -> Optional[tuple[str, ast.AST]]:
        """The function *called* names, for a call on ``self`` or a plain name."""
        if (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and called.value.id == "self"
        ):
            holder = self.method_class(file, func)
            if holder is None:
                return None
            return self.method(holder[0], holder[1], called.attr)
        return None

