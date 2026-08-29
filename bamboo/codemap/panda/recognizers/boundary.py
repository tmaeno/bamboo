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
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

from bamboo.codemap.models import Anchor, BoundaryNode, CoverageStat, SourceModule

SLICE_NAME = "boundary"

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


def _endpoint_decorator(func: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[ast.Call]:
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
            decorator = _endpoint_decorator(node)
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
