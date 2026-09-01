"""Checks that every backend shipped here still satisfies the backend contract.

The contract is an ABC, so Python already enforces it -- but only at the moment
something is instantiated, and nothing instantiated the example backend.  Adding
``merge_map_node`` and ``clear_map`` to ``GraphDatabaseBackend`` for the Code Map
therefore broke ``InMemoryGraphBackend``, the reference implementation the
database-plugin pages tell a contributor to copy, and 818 tests stayed green.

These instantiate every backend in the package so that the ABC gets to do its job
in CI rather than in a plugin author's editor.  Instantiation is deliberately the
whole test: a hand-maintained list of required method names would be a third
expression of the contract, free to drift from the other two.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterator

import pytest

import bamboo.database.backends as backends_pkg
from bamboo.database.base import GraphDatabaseBackend, VectorDatabaseBackend
from bamboo.models.graph_element import CODE_MAP_NODE_TYPES, NodeType


def _shipped_backends() -> Iterator[type]:
    """Every concrete backend under ``bamboo.database.backends``, examples included.

    Discovered rather than listed: a backend added to the package is covered
    without anyone remembering to add it here, which is the failure this file
    exists to stop repeating.
    """
    for info in pkgutil.walk_packages(
        backends_pkg.__path__, prefix=f"{backends_pkg.__name__}."
    ):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != info.name:
                continue
            if not issubclass(obj, (GraphDatabaseBackend, VectorDatabaseBackend)):
                continue
            if inspect.isabstract(obj):
                continue
            yield obj


SHIPPED = sorted(_shipped_backends(), key=lambda c: c.__name__)


def test_the_package_ships_backends_to_check() -> None:
    """Guards the discovery itself: a walk that silently finds nothing would
    make every check below pass without checking anything."""
    names = [c.__name__ for c in SHIPPED]
    assert "Neo4jBackend" in names
    assert "InMemoryGraphBackend" in names


@pytest.mark.parametrize("backend_class", SHIPPED, ids=lambda c: c.__name__)
def test_a_shipped_backend_implements_its_whole_contract(backend_class: type) -> None:
    """Constructing is the check -- an ABC refuses a subclass with a method missing.

    None of these connect in ``__init__`` (the graph client connects lazily on
    first query), so this costs nothing and needs no services.
    """
    backend_class()


def test_the_code_map_labels_match_the_nodes_the_extraction_writes() -> None:
    """``CODE_MAP_NODE_TYPES`` decides what a rebuild is allowed to delete.

    The node models declare the same fact independently -- each carries the
    label it is stored under -- so the two can be compared.  They disagreed
    once already: ``clear_map`` was written before ``FilterStage`` existed and
    kept deleting only four of the five, so stages dropped by a new source
    version stayed in the database forever.
    """
    from bamboo.codemap import models as codemap_models
    from bamboo.models.graph_element import BaseNode

    declared = {
        obj.model_fields["node_type"].default
        for _, obj in inspect.getmembers(codemap_models, inspect.isclass)
        if issubclass(obj, BaseNode) and "node_type" in obj.model_fields
    }

    assert declared == set(CODE_MAP_NODE_TYPES), (
        "the Code Map label set and the node models disagree: "
        f"only in the set {sorted(t.value for t in CODE_MAP_NODE_TYPES - declared)}, "
        f"only in the models {sorted(t.value for t in declared - CODE_MAP_NODE_TYPES)}"
    )


def _subject(version: str):
    """One Code Map node, identical apart from the release it was derived from."""
    from bamboo.codemap.models import SubjectNode

    return SubjectNode(
        name="JediTaskSpec.status",
        map_id="panda",
        derived_from=version,
        spec_class="JediTaskSpec",
        attribute="status",
    )


def test_the_code_map_namespace_is_a_strict_part_of_the_labels() -> None:
    """The separation is what lets a rebuild leave incident knowledge alone, so
    a Code Map label that is not a real ``NodeType`` would silently never match."""
    assert CODE_MAP_NODE_TYPES < set(NodeType)


async def test_clearing_one_map_leaves_the_incident_graph_alone() -> None:
    """The behaviour the label separation exists for, on the backend that can be
    exercised without a database."""
    from bamboo.database.backends.examples.in_memory_backend import (
        InMemoryGraphBackend,
    )
    from bamboo.models.graph_element import CauseNode

    backend = InMemoryGraphBackend()
    await backend.create_node(CauseNode(name="a cause a human validated"))
    await backend.merge_map_node(_subject("1.0.2"))

    assert await backend.clear_map("panda") == 1
    remaining = [n.name for n in backend.nodes.values()]
    assert remaining == ["a cause a human validated"]


async def test_rebuilding_a_map_updates_the_node_and_records_both_versions() -> None:
    """Identity is the semantic signature, so the same subject seen in a later
    release is the same node -- not a duplicate, and not a lost history."""
    from bamboo.database.backends.examples.in_memory_backend import (
        InMemoryGraphBackend,
    )

    backend = InMemoryGraphBackend()
    first = await backend.merge_map_node(_subject("1.0.2"))
    second = await backend.merge_map_node(_subject("1.0.3"))

    assert first == second
    assert len(backend.nodes) == 1
    assert backend.nodes[first].metadata["valid_for"] == ["1.0.2", "1.0.3"]
