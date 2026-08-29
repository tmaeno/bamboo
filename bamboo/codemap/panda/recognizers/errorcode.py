"""Value-enumeration recognizer.

Extracts module-level ``NAME = <value>`` constants -- the shape PanDA uses for
its error codes (``EC_Kill = 100``) and event-service states
(``ST_discarded = 5``) -- into a ``(namespace, value)`` index.

This is the cheapest slice in the map and the most reliable, because the code
declares the vocabulary outright: a name, a value, and usually a comment
saying what it means, all in a fixed shape.  Nothing has to be inferred.  That
is the general rule the map exploits -- how fragile a recognizer is tracks how
little the code declares about itself, and an explicit enumeration declares
everything.

Two things this deliberately does *not* do:

* It does not treat every module constant as an enumeration entry.  A declared
  *value set* (``FINAL_TASK_STATUSES = [...]``) is a vocabulary oracle for
  checking extraction, not an index entry, and counting the two together makes
  an error-code module indistinguishable from a config module.
* It does not key on the number alone.  ``taskbuffer.EC_Kill`` and
  ``jobdispatcher.EC_Watcher`` are both ``100``; which one a field holds is
  decided by the namespace it was read from.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from bamboo.codemap.models import Anchor, CoverageStat, SourceModule, ValueEnumNode

SLICE_NAME = "value-enum"

# Names that read as enumeration members.  Upper-case is the usual convention;
# the ``ST_``/``EC_`` prefixes catch the mixed-case members that PanDA writes
# for event-service states and error codes.
_PREFIXES = ("EC_", "ST_")


def _is_enum_name(name: str) -> bool:
    return name.isupper() or name.startswith(_PREFIXES)


def _module_namespace(rel_path: str) -> str:
    """Return ``<subsystem>.<module>`` for a package-relative path."""
    parts = Path(rel_path).parts
    module = Path(rel_path).stem
    return f"{parts[1]}.{module}" if len(parts) >= 3 else module


def _name_prefix(constant: str) -> Optional[str]:
    """Return the leading ``WORD_`` segment of a constant name, if any."""
    head, sep, _rest = constant.partition("_")
    return head if sep and head else None


def _namespace_of(rel_path: str, constant: str, grouped_prefixes: set[str]) -> str:
    """Return the namespace that makes ``(namespace, value)`` identify one entry.

    Scoping to the subsystem is too coarse: ``taskbuffer/`` holds unrelated
    enumerations whose values legitimately collide.  Scoping to the module is
    still too coarse, because one module can declare two of them --
    ``EventServiceUtils`` defines event states ``ST_ready..ST_reserved_get``
    *and* task types ``TASK_NORMAL..TASK_FINE_GRAINED``, so ``0`` means two
    different things in one file.

    Python offers no enum block here, so the shared ``ST_`` / ``TASK_`` / ``EC_``
    prefix *is* how the code declares the grouping -- the same "read what the
    code states about itself" principle the whole slice rests on.  A prefix is
    only treated as a group when at least two constants share it, so a lone
    ``MESSAGE_JSON`` is not given a namespace of its own.

    Mapping a *field* to the enumeration that decodes it (``taskBufferErrorCode``
    -> the ``EC_`` group in ``taskbuffer.ErrorCode``) is a separate, explicit
    binding; it is not recoverable from the constant's location.
    """
    module_ns = _module_namespace(rel_path)
    prefix = _name_prefix(constant)
    if prefix and prefix in grouped_prefixes:
        return f"{module_ns}.{prefix}"
    return module_ns


def _leading_comment(lines: list[str], lineno: int) -> Optional[str]:
    """Return the comment block immediately above a 1-indexed *lineno*.

    ``ast`` discards comments, but they carry the only human-readable meaning
    an enumeration entry has (``# killed`` above ``EC_Kill = 100``).  Walking
    back over contiguous ``#`` lines recovers it without a second parser.
    """
    collected: list[str] = []
    idx = lineno - 2  # 0-indexed line above the assignment
    while idx >= 0:
        stripped = lines[idx].strip()
        if stripped.startswith("#"):
            collected.append(stripped.lstrip("#").strip())
            idx -= 1
            continue
        if not stripped:
            # A blank line separates one entry's comment from the previous
            # entry, so stop rather than absorbing the neighbour's text.
            break
        break
    if not collected:
        return None
    return " ".join(reversed(collected)) or None


def _iter_module_constants(tree: ast.Module) -> Iterable[tuple[str, ast.Assign]]:
    """Yield ``(name, node)`` for module-level constant assignments."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_enum_name(target.id):
                yield target.id, node
                break


def extract(
    modules: list[SourceModule],
    map_id: str,
    derived_from: str,
) -> tuple[list[ValueEnumNode], list[CoverageStat]]:
    """Extract value enumerations and per-file coverage.

    Args:
        modules:      Every parsed module in the snapshot.
        map_id:       Map this fragment belongs to.
        derived_from: Version stamp recorded on every node.

    Returns:
        ``(value_enums, coverage)``.  Coverage counts module-level constants as
        candidates and enumeration entries as explained, so a module of
        computed or collection-valued constants shows a low ratio -- the signal
        that it is a different kind of file, not a failed extraction.
    """
    enums: list[ValueEnumNode] = []
    coverage: list[CoverageStat] = []
    references: Counter = Counter()

    # Reference counting runs over every module, not just the defining one:
    # a constant is only meaningfully part of the index if something reads it.
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Attribute) and _is_enum_name(node.attr):
                references[node.attr] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if _is_enum_name(node.id):
                    references[node.id] += 1

    for module in modules:
        pkg, rel, tree = module.package, module.rel_path, module.tree
        lines = module.source.splitlines()
        candidates = 0
        explained = 0
        # A prefix only counts as an enumeration group when several constants
        # share it, so singletons stay in the module namespace.
        prefix_counts = Counter(
            p for name, _ in _iter_module_constants(tree) if (p := _name_prefix(name))
        )
        grouped_prefixes = {p for p, n in prefix_counts.items() if n >= 2}
        for name, node in _iter_module_constants(tree):
            value_node = node.value
            is_scalar = (
                isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, (int, str))
                and not isinstance(value_node.value, bool)
            )
            is_collection = isinstance(value_node, (ast.List, ast.Tuple, ast.Set, ast.Dict))
            if not (is_scalar or is_collection):
                # Computed values (calls, arithmetic, name lookups) are not
                # declarations at all, so they are not a coverage miss -- they
                # are simply a different kind of statement and stay out of the
                # denominator.
                continue
            candidates += 1
            if not is_scalar:
                # A collection literal is a declared *value set* -- a vocabulary
                # oracle rather than an index entry.  Counted as a candidate so
                # a file full of them reads as "different kind of declaration",
                # not as full coverage.
                continue
            explained += 1
            # The namespace is the index key and may end with the enumeration's
            # prefix; the display name stays module-scoped so it does not read
            # as ``taskbuffer.ErrorCode.EC.EC_Kill``.
            ns = _namespace_of(rel, name, grouped_prefixes)
            enums.append(
                ValueEnumNode(
                    map_id=map_id,
                    derived_from=derived_from,
                    name=ValueEnumNode.make_name(_module_namespace(rel), name),
                    namespace=ns,
                    constant=name,
                    value=value_node.value,
                    comment=_leading_comment(lines, node.lineno),
                    anchor=Anchor(
                        package=pkg,
                        file=rel,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        blob_sha=module.blob_sha,
                    ),
                    references=references.get(name, 0),
                )
            )
        if candidates:
            coverage.append(
                CoverageStat(
                    slice_name=SLICE_NAME,
                    file=rel,
                    candidates=candidates,
                    explained=explained,
                )
            )
    return enums, coverage
