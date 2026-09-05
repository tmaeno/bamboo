"""Reading a stored Code Map back.

These run against the in-memory example backend rather than mocks, so the write
and the read meet each other: a mock would happily hand back whatever the test
put in and prove nothing about the encoding, which is the half most likely to
drift.  That the reference backend is exercised here at all is a side benefit --
it is the thing that silently stopped working the last time the interface grew.
"""

from __future__ import annotations

import pytest

from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    Anchor,
    Branch,
    EntryPoint,
    FilterStageNode,
    JunctionNode,
    MapFragment,
    SubjectNode,
)
from bamboo.codemap.store import store_fragment
from bamboo.database.backends.examples.in_memory_backend import InMemoryGraphBackend

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"


def _subject(attribute: str = "status") -> SubjectNode:
    return SubjectNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"JediTaskSpec.{attribute}",
        spec_class="JediTaskSpec",
        attribute=attribute,
        criteria=["3:closed-literal-set"],
    )


def _junction(owner: str, *branches: Branch, subject: str = "JediTaskSpec.status") -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=JunctionNode.make_name(MAP_ID, subject, owner),
        subject=subject,
        owner=owner,
        branches=list(branches),
        log_files=[f"panda-{owner.split('/')[-1].split('.')[0]}.log"],
        entry_points=[EntryPoint(trigger="polled", entry=owner)],
        anchor=Anchor(package="pandajedi", file=owner.split("::")[0], line_start=42),
    )


def _stage(label: str, tag: str, order: int, owner: str = "b.py::doBrokerage") -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=FilterStageNode.make_name(MAP_ID, owner, f"{label}|{tag}"),
        owner=owner,
        criteria_tag=tag,
        funnel_label=label,
        order=order,
        conditions=["site_value > limit"],
    )


async def _stored(fragment: MapFragment) -> CodeMap:
    """Write *fragment* through the real store, and hand back a reader for it.

    The backend stands in for the client, which is a façade whose signatures
    mirror it one for one -- the same substitution the existing store tests make
    with a mock, except that this one actually stores and reads.
    """
    backend = InMemoryGraphBackend()
    await backend.connect()
    await store_fragment(fragment, backend)
    return CodeMap(backend, map_id=MAP_ID)


async def test_a_junction_survives_the_round_trip_unchanged():
    """The encoding and the decoding are two expressions of one shape.

    Nested values cannot be stored as they are -- Neo4j holds primitives and
    arrays -- so branches, entry points and anchors go in as JSON and have to
    come back out.  Comparing the model to itself across the trip is the only
    thing that says the two halves still agree.
    """
    original = _junction(
        "pandajedi/jediorder/JobGenerator.py::runImpl",
        Branch(outcome="pending", path_condition=["not tasks"], tier=1),
        Branch(outcome="runtime(newStatus)", tier=2),
    )
    code_map = await _stored(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, subjects=[_subject()], junctions=[original])
    )

    (read_back,) = await code_map.writers_of("JediTaskSpec.status")

    assert read_back.branches == original.branches
    assert read_back.entry_points == original.entry_points
    assert read_back.anchor == original.anchor
    assert read_back.model_dump() == original.model_dump()


async def test_the_writers_of_a_subject_are_all_of_them():
    """The fan-out is the system's, not the map's -- so nothing is ranked away."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction("a.py::f", Branch(outcome="pending")),
            _junction("b.py::g", Branch(outcome="ready")),
            _junction("c.py::h", Branch(outcome="broken"), subject="JediTaskSpec.oldStatus"),
        ],
    )
    code_map = await _stored(fragment)

    assert [j.owner for j in await code_map.writers_of("JediTaskSpec.status")] == [
        "a.py::f",
        "b.py::g",
    ]


async def test_an_observed_value_eliminates_the_writers_that_cannot_produce_it():
    """Pruning is elimination, so it has to be exact about what it removes."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction("can.py::f", Branch(outcome="pending")),
            _junction("cannot.py::g", Branch(outcome="ready")),
        ],
    )
    code_map = await _stored(fragment)

    assert [j.owner for j in await code_map.producers_of("JediTaskSpec.status", "pending")] == [
        "can.py::f"
    ]


async def test_a_run_time_writer_is_never_eliminated_by_an_observed_value():
    """Tier 2 means the writer is known and the value is not.

    Dropping it would turn an incomplete candidate set into a confident wrong
    answer, which is the one failure elimination cannot recover from -- the
    real writer is simply gone and the survivors look unanimous.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[_junction("runtime.py::f", Branch(outcome="runtime(newStatus)", tier=2))],
    )
    code_map = await _stored(fragment)

    assert len(await code_map.producers_of("JediTaskSpec.status", "pending")) == 1


async def test_a_passthrough_moves_the_question_to_the_field_it_came_from():
    """The reference lives inside JSON-encoded branches, so this is the hop
    that a property match in the query cannot make on its own."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(), _subject("oldStatus")],
        junctions=[
            _junction("copy.py::f", Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2)),
            _junction("src.py::g", Branch(outcome="pending"), subject="JediTaskSpec.oldStatus"),
        ],
    )
    code_map = await _stored(fragment)

    carried = await code_map.carried_from("JediTaskSpec.status")

    assert [j.owner for j in carried["JediTaskSpec.oldStatus"]] == ["src.py::g"]


async def test_a_value_carried_from_outside_the_map_ends_the_walk():
    """An empty list is an answer: it arrived with the task."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[_junction("copy.py::f", Branch(outcome="passthrough(JediTaskSpec.taskPriority)"))],
    )
    code_map = await _stored(fragment)

    assert (await code_map.carried_from("JediTaskSpec.status")) == {
        "JediTaskSpec.taskPriority": []
    }


async def test_a_chain_comes_back_in_the_order_the_source_runs_it():
    """Stages are cumulative, so their order is part of the answer."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("disk check", "-disk", 2), _stage("memory check", "-lowmemory", 1)],
    )
    code_map = await _stored(fragment)

    assert [s.criteria_tag for s in await code_map.chain("b.py::doBrokerage")] == [
        "-lowmemory",
        "-disk",
    ]


async def test_a_tag_from_a_log_line_finds_every_stage_that_emits_it():
    """One tag can belong to two stages -- ``AtlasProdJobBroker`` emits
    ``-disk`` from both "disk check" and "Storage check" -- so folding them
    together would lose one of the two cuts."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("disk check", "-disk", 1), _stage("Storage check", "-disk", 2)],
    )
    code_map = await _stored(fragment)

    assert {s.funnel_label for s in await code_map.stage_for_tag("-disk")} == {
        "disk check",
        "Storage check",
    }


async def test_reading_refuses_a_label_outside_the_code_map_namespace():
    """The separation is what stops a Code Map read reaching incident knowledge."""
    backend = InMemoryGraphBackend()
    await backend.connect()

    with pytest.raises(ValueError, match="not a Code Map label"):
        await backend.find_map_nodes("Cause", MAP_ID)


async def test_which_log_to_read_is_answered_per_writer():
    """The question the map exists for. Package does not decide it -- the same
    proxy code logs to JEDI's file or the server's depending on who called.

    Two lists, because for the largest group of writers the file that holds the
    code is not the file that mentions it: the proxy mixins own
    ``panda-DBProxy.log`` and the knight that called them writes
    ``set task_status=``.  Merging them would answer both questions with one
    value and make eleven separable candidates look identical.
    """
    knight = _junction("pandajedi/jediorder/JobGenerator.py::runImpl", Branch(outcome="pending"))
    proxy = _junction(
        "pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py::makeTaskPending_JEDI",
        Branch(outcome="pending"),
    )
    proxy.log_files = ["panda-DBProxy.log", "panda-JediDBProxy.log"]
    proxy.caller_log_files = ["panda-AtlasTaskWithholderWatchDog.log"]
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[knight, proxy],
    )
    code_map = await _stored(fragment)

    assert await code_map.log_files_for("JediTaskSpec.status") == {
        knight.owner: {"own": ["panda-JobGenerator.log"], "caller": []},
        proxy.owner: {
            "own": ["panda-DBProxy.log", "panda-JediDBProxy.log"],
            "caller": ["panda-AtlasTaskWithholderWatchDog.log"],
        },
    }
