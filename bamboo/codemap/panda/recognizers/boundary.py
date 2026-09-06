"""Boundary recognizer -- where values enter from a system this map does not cover.

A boundary is generated from the *receiving* side alone, so it exists before
the far side is ever mapped.  That is what makes adding another system's map a
binding operation rather than a re-derivation: the attachment point is already
there, and until something attaches to it an unbound boundary is still a
complete answer ("the pilot reported 1099") as well as a concrete work item.

PanDA declares its entry points outright.  ``@request_validation(...)`` marks a
function as an HTTP endpoint and states, as literal keyword arguments, the
conditions the request must satisfy before the handler runs -- transport
security, caller role, HTTP method, task ownership.  Reading that decorator is
far more reliable than inferring endpoints from a ``req`` first parameter, and
it recovers something the handler body never shows: **the first place a request
can be rejected**.  A command that seems to have vanished may have been turned
away here, before any junction saw it.

Boundaries move.  Between two PanDA releases the pilot's entry moved file
entirely -- ``jobdispatcher/JobDispatcher.py::updateJob`` became
``api/v1/pilot_api.py::update_job`` and the old module disappeared -- while
remaining the same boundary.  Identity is therefore ``(system, interface)``;
the anchor merely records where the evidence was found this time.

**Not every boundary is an endpoint.**  PanDA and DEFT talk through shared
database tables, and the code says so: every statement against them is
qualified with ``panda_config.schemaDEFT``, which ``panda_config`` declares as
``ATLAS_DEFT``.  A table read and written across that qualifier is a channel
between two systems whether or not anything HTTP is involved, and it fails
differently from an endpoint -- the row is either there or it is not, which
makes non-arrival directly checkable instead of merely absent from a log.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Optional

from bamboo.codemap.models import Anchor, BoundaryNode, CoverageStat, SourceModule
from bamboo.codemap.panda import sql
from bamboo.codemap.panda.pathcond import functions_with_owner

SLICE_NAME = "boundary"
CHANNEL_SLICE_NAME = "db-channel"

_DECORATOR = "request_validation"

# The module name is how the code names its caller, but only some callers are
# separate *systems* -- ones with their own release cycle, whose version this
# map's stamp does not cover.  The rest are clients of PanDA (users,
# pandaclient, DEFT) reached through the same HTTP surface; they cross a trust
# boundary but not a version boundary.
_SYSTEM_BY_MODULE = {
    "pilot_api": "pilot",
    "harvester_api": "harvester",
    "idds_api": "idds",
}
_DEFAULT_SYSTEM = "client"

# Parameters that identify the far side's build at run time.  A system whose
# version moves independently of this map cannot share its stamp, so a
# conclusion drawn across such a boundary has to say which build reported it.
_VERSION_BINDING_PARAMS = frozenset(
    {"pilot_id", "pilotID", "scheduler_id", "schedulerID", "pilot_version", "harvester_id"}
)

# Never carried across the boundary: the request object itself and the
# framework's own plumbing.
_NON_PAYLOAD_PARAMS = frozenset({"req", "self", "cls"})


def _system_of(rel_path: str) -> str:
    return _SYSTEM_BY_MODULE.get(Path(rel_path).stem, _DEFAULT_SYSTEM)


def endpoint_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[ast.Call]:
    """Return the ``@request_validation(...)`` call decorating *func*, if any."""
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
        if name == _DECORATOR:
            return decorator
    return None


def _access_conditions(decorator: ast.Call) -> dict[str, Any]:
    """Return the literal keyword arguments the decorator declares.

    Only literals are kept.  ``task_buffer=lambda: global_task_buffer`` is
    wiring, not a condition, and recording it as one would put a callable where
    a precondition is expected.
    """
    conditions: dict[str, Any] = {}
    for keyword in decorator.keywords:
        if keyword.arg is None:
            continue
        if isinstance(keyword.value, ast.Constant):
            conditions[keyword.arg] = keyword.value.value
    return conditions


def _parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return the payload parameter names, in declaration order."""
    args = func.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return [n for n in names if n not in _NON_PAYLOAD_PARAMS]


def _logged_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return identifiers the function interpolates into its own f-strings.

    This is the second, independent statement of what the endpoint receives:
    the signature declares it, and the arrival log repeats it.  Comparing the
    two is what lets the extraction be graded without production data.
    """
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr):
            continue
        for value in node.values:
            if not isinstance(value, ast.FormattedValue):
                continue
            target = value.value
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[BoundaryNode], list[CoverageStat]]:
    """Extract boundaries and per-file coverage.

    Coverage counts decorated endpoints as candidates and those whose payload
    could be read as explained, so an endpoint declaring no parameters at all
    is not counted as a miss -- it carries nothing across.
    """
    boundaries: list[BoundaryNode] = []
    coverage: list[CoverageStat] = []

    for module in modules:
        candidates = 0
        explained = 0
        for node in ast.walk(module.tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator = endpoint_decorator(node)
            if decorator is None:
                continue
            candidates += 1
            carried = _parameters(node)
            accepts_arbitrary = node.args.kwarg is not None
            if not carried and not accepts_arbitrary:
                # Nothing crosses: an introspection endpoint taking only the
                # request itself is an entry point but not a boundary.
                continue
            explained += 1

            logged = _logged_names(node)
            system = _system_of(module.rel_path)
            interface = f"{Path(module.rel_path).stem}::{node.name}"
            boundaries.append(
                BoundaryNode(
                    map_id=map_id,
                    derived_from=derived_from,
                    name=BoundaryNode.make_name(map_id, system, interface),
                    system=system,
                    # An HTTP endpoint is a caller handing over values, so the
                    # far side's state is visible in the record.  A broker,
                    # which carries causation and leaves nothing behind when a
                    # message is lost, is a different recognizer.
                    kind="reports_state",
                    interface=interface,
                    carried_values=carried,
                    access_conditions=_access_conditions(decorator),
                    observable_values=[c for c in carried if c in logged],
                    accepts_arbitrary=accepts_arbitrary,
                    version_binding=[c for c in carried if c in _VERSION_BINDING_PARAMS],
                    anchor=Anchor(
                        package=module.package,
                        file=module.rel_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        blob_sha=module.blob_sha,
                    ),
                )
            )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )
    return boundaries, coverage


# --------------------------------------------------------------------------- #
# shared-table channels
# --------------------------------------------------------------------------- #

# Which schemas belong to somebody else.  Stated here rather than derived
# because "does this system release on its own cycle" is a fact about the
# deployment, not about the source -- the same reason ``_SYSTEM_BY_MODULE``
# above is a map and not an inference.  What *is* derived is everything that
# follows from it: the schema's real name, which tables live under it, which
# columns cross, and in which direction.
_FOREIGN_SCHEMAS = {
    "DEFT": "deft",
    "GRISLI": "grisli",
    "EI": "eventindex",
}

_SCHEMA_ATTRIBUTE = "schema"


def schema_names(modules: list[SourceModule]) -> dict[str, str]:
    """Return ``{suffix: schema name}`` from ``panda_config``'s declarations.

    ``tmpSelf.__dict__["schemaDEFT"] = "ATLAS_DEFT"`` is the one place the code
    says which Oracle schema each name means.  Read rather than assumed so that
    the boundary's interface carries the schema an operator would actually type
    into a query.
    """
    found: dict[str, str] = {}
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                continue
            for target in node.targets:
                key = target.slice if isinstance(target, ast.Subscript) else None
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if key.value.startswith(_SCHEMA_ATTRIBUTE) and key.value != _SCHEMA_ATTRIBUTE:
                    found[key.value[len(_SCHEMA_ATTRIBUTE) :]] = node.value.value
    return found


def _qualifying_schema(
    func: ast.FunctionDef | ast.AsyncFunctionDef, variable: str, known: dict[str, str]
) -> Optional[str]:
    """Return the schema suffix a statement is qualified with, when just one is.

    A statement joining a DEFT table to a PanDA one mentions two, and which
    side of the boundary a column sits on is then no longer readable from the
    qualifier.  Those are skipped rather than guessed; the coverage line counts
    them.
    """
    suffixes = {
        suffix
        for expression in sql.interpolations(func, variable)
        for suffix in known
        if expression.endswith(f"{_SCHEMA_ATTRIBUTE}{suffix}")
    }
    return suffixes.pop() if len(suffixes) == 1 else None


class _Channel:
    """What crossed one foreign table, accumulated over the whole corpus.

    Columns are pooled case-insensitively.  SQL identifiers are, and the same
    column really is written ``COMM_CMD`` in one statement and ``comm_cmd`` in
    another -- listing both would claim the channel carries two things where it
    carries one.  The first spelling seen is kept rather than a normalised one,
    since nothing in the source says which case the table was declared with.
    """

    def __init__(self, system: str, interface: str) -> None:
        self.system = system
        self.interface = interface
        self.received: dict[str, str] = {}
        self.sent: dict[str, str] = {}
        self.operations: set[str] = set()
        self.anchor: Optional[Anchor] = None

    def receives(self, columns: Iterable[str]) -> None:
        for column in columns:
            self.received.setdefault(column.lower(), column)

    def hands_over(self, columns: Iterable[str]) -> None:
        for column in columns:
            self.sent.setdefault(column.lower(), column)


#: What a boundary's ``system`` says when the source does not name one.  Kept
#: as a value rather than left blank so that it reads as an open question in
#: the node itself -- "bind this" -- the same way an unbound pilot boundary
#: does.
UNRESOLVED_SYSTEM = "unresolved"


def tables_never_written(modules: list[SourceModule]) -> set[str]:
    """Tables this map only ever reads.

    A dependency the map cannot explain: whatever keeps the table current is
    outside the source being read, so its freshness is not something any branch
    table can account for.
    """
    written: set[str] = set()
    read: set[str] = set()
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            for run in sql.executions(func):
                written.update(write.table for write in sql.writes(run.sql))
                written.update(sql.deletes(run.sql))
                read.update(table for table, _columns in sql.reads(run.sql))
                read.update(sql.joins(run.sql))
    return {table for table in read if table not in written and table != "{}"}


def extract_selection_gates(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
    never_written: set[str],
    already_known: set[str],
) -> list[BoundaryNode]:
    """A boundary per table that bounds a query's reach and nothing here writes.

    The other extraction reads *which schema* a statement is qualified with, so
    it finds the tables another system owns by name.  This one reads a
    different signal and finds tables that are nominally PanDA's own::

        FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID

    ``JEDI_AUX_Status_MinTaskID`` sits in JEDI's own schema, is joined by
    thirty-one functions, and **no statement in the corpus writes it**.  A task
    below its watermark is invisible to all of them whatever its status is, so
    when the table goes stale the map's account of why nothing picked a task up
    is wrong in a way no branch condition shows.  That is exactly what an
    unbound boundary is for: a complete answer -- the value comes from outside
    -- and a work item.

    ``system`` is left unresolved rather than guessed.  The schema qualifier
    says which database the table is in, not who maintains it, and inventing a
    name here would put a claim where the source is silent.
    """
    seen: dict[str, _Channel] = {}
    for module in modules:
        for func, _owner in functions_with_owner(module.tree):
            for run in sql.executions(func):
                for table in sql.joins(run.sql):
                    if table not in never_written or table in already_known:
                        continue
                    channel = _channel_for(
                        seen, UNRESOLVED_SYSTEM, "", table, module, run.call
                    )
                    channel.receives(sql.joined_columns(run.sql, table))
                    channel.operations.add("SELECT")
    return [
        BoundaryNode(
            map_id=map_id,
            derived_from=derived_from,
            name=BoundaryNode.make_name(map_id, channel.system, channel.interface),
            system=channel.system,
            kind="reports_state",
            transport="shared_table",
            interface=channel.interface,
            carried_values=sorted(channel.received.values()),
            # Nothing here writes it: that emptiness is the finding, not a gap
            # in the reading.
            handed_over=[],
            operations=sorted(channel.operations),
            anchor=channel.anchor,
        )
        for channel in sorted(seen.values(), key=lambda c: c.interface)
    ]


def extract_shared_tables(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[BoundaryNode], list[CoverageStat]]:
    """Extract a boundary per table PanDA shares with another system."""
    known = schema_names(modules)
    channels: dict[str, _Channel] = {}
    coverage: list[CoverageStat] = []

    for module in modules:
        candidates = 0
        explained = 0
        for func, _owner in functions_with_owner(module.tree):
            seen: set[tuple[str, str]] = set()
            for run in sql.executions(func):
                if (run.variable, run.sql) in seen:
                    continue
                seen.add((run.variable, run.sql))
                suffix = _qualifying_schema(func, run.variable, known)
                if suffix is None or suffix not in _FOREIGN_SCHEMAS:
                    continue
                candidates += 1
                if _absorb(
                    channels,
                    system=_FOREIGN_SCHEMAS[suffix],
                    schema=known[suffix],
                    module=module,
                    node=run.call,
                    statement=run.sql,
                ):
                    explained += 1
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=CHANNEL_SLICE_NAME,
                    file=module.rel_path,
                    candidates=candidates,
                    explained=explained,
                )
            )

    boundaries = [
        BoundaryNode(
            map_id=map_id,
            derived_from=derived_from,
            name=BoundaryNode.make_name(map_id, channel.system, channel.interface),
            system=channel.system,
            # The row is either in the table or it is not, so what failed to
            # cross leaves the same evidence as what crossed -- the test the
            # kind axis draws, and the opposite of a broker.
            kind="reports_state",
            transport="shared_table",
            interface=channel.interface,
            carried_values=sorted(channel.received.values()),
            handed_over=sorted(channel.sent.values()),
            operations=sorted(channel.operations),
            anchor=channel.anchor,
        )
        for channel in sorted(channels.values(), key=lambda c: c.interface)
    ]
    return boundaries, coverage


def _absorb(
    channels: dict[str, _Channel],
    *,
    system: str,
    schema: str,
    module: SourceModule,
    node: ast.Call,
    statement: str,
) -> bool:
    """Fold one statement into its table's channel.  True if anything was read."""
    found = False
    for table, columns in sql.reads(statement):
        channel = _channel_for(channels, system, schema, table, module, node)
        channel.receives(columns)
        channel.operations.add("SELECT")
        found = True
    for write in sql.writes(statement):
        channel = _channel_for(channels, system, schema, write.table, module, node)
        channel.hands_over(write.columns)
        channel.operations.add(write.kind.upper())
        found = True
    for table in sql.deletes(statement):
        channel = _channel_for(channels, system, schema, table, module, node)
        channel.operations.add("DELETE")
        found = True
    return found


def _channel_for(
    channels: dict[str, _Channel],
    system: str,
    schema: str,
    table: str,
    module: SourceModule,
    node: ast.Call,
) -> _Channel:
    # No schema when the qualifier is interpolated and names PanDA's own -- the
    # table name is then the whole identity, which is how the rest of the map
    # spells a table anyway.
    interface = f"{schema}.{table}" if schema else table
    channel = channels.get(interface)
    if channel is None:
        channel = _Channel(system, interface)
        channel.anchor = Anchor(
            package=module.package,
            file=module.rel_path,
            line_start=node.lineno,
            line_end=node.end_lineno,
            blob_sha=module.blob_sha,
        )
        channels[interface] = channel
    return channel
