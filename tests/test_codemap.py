"""Code Map extraction, gates, and namespace-scoped storage.

The value-enumeration slice is the map's simplest: the code declares a name, a
value and usually a comment, all in a fixed shape, so nothing has to be
inferred.  These tests pin the parts that were *not* obvious and that real
PanDA source forced corrections to:

* what counts as one enumeration (a module can declare two, with overlapping
  values, and the shared name prefix is the only thing that separates them);
* that a constant nothing reads is not an index entry;
* that rebuilding a Code Map cannot take the incident graph with it.
"""

from __future__ import annotations

import ast
from unittest.mock import AsyncMock

import pytest

from bamboo.codemap import gates
from bamboo.codemap.models import (
    Anchor,
    Branch,
    JunctionNode,
    MapFragment,
    ValueEnumNode,
)
from bamboo.codemap.panda.recognizers import errorcode
from bamboo.models.graph_element import NodeType

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"


def _module(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py"):
    """Build the ``(package, rel, tree, source)`` tuple the recognizer takes."""
    return (rel.split("/")[0], rel, ast.parse(source), source)


def _extract(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py"):
    return errorcode.extract([_module(source, rel)], MAP_ID, VERSION)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def test_extracts_name_value_and_comment():
    """The comment above a constant is its only human-readable meaning.

    ``ast`` throws comments away, so they are recovered from the source text;
    without them an index entry is a bare number with no explanation.
    """
    enums, _ = _extract("# error code\n\n# killed\nEC_Kill = 100\n")

    assert len(enums) == 1
    entry = enums[0]
    assert entry.constant == "EC_Kill"
    assert entry.value == 100
    assert entry.comment == "killed"
    assert entry.node_type is NodeType.VALUE_ENUM
    assert entry.map_id == MAP_ID
    assert entry.derived_from == VERSION


def test_comment_does_not_leak_across_a_blank_line():
    """A blank line ends one entry's comment; otherwise it absorbs its neighbour's."""
    enums, _ = _extract("# killed\nEC_Kill = 100\n\nEC_Transfer = 101\n")

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].comment == "killed"
    assert by_name["EC_Transfer"].comment is None


def test_prefix_separates_two_enumerations_in_one_module():
    """One module can declare two enumerations whose values overlap.

    ``EventServiceUtils`` defines event states *and* task types, so ``0`` means
    two different things in one file.  Python has no enum block here, so the
    shared prefix is how the code states the grouping -- keying on the module
    alone would make ``(namespace, value)`` ambiguous.
    """
    source = (
        "ST_ready = 0\nST_sent = 1\n"
        "TASK_NORMAL = 0\nTASK_EVENT_SERVICE = 1\n"
    )
    enums, _ = _extract(source, "pandaserver/taskbuffer/EventServiceUtils.py")

    namespaces = {e.constant: e.namespace for e in enums}
    assert namespaces["ST_ready"] != namespaces["TASK_NORMAL"]
    assert namespaces["ST_ready"] == namespaces["ST_sent"]
    assert namespaces["ST_ready"].endswith(".ST")
    # The display name stays module-scoped so it does not read as "...EC.EC_Kill".
    assert {e.name for e in enums} == {
        "taskbuffer.EventServiceUtils.ST_ready",
        "taskbuffer.EventServiceUtils.ST_sent",
        "taskbuffer.EventServiceUtils.TASK_NORMAL",
        "taskbuffer.EventServiceUtils.TASK_EVENT_SERVICE",
    }


def test_lone_prefixed_constant_is_not_given_its_own_namespace():
    """A prefix only groups when several constants share it."""
    enums, _ = _extract("MESSAGE_JSON = 'bad json'\nLATENCY = 'latency'\n",
                        "pandaserver/api/v1/common.py")

    assert {e.namespace for e in enums} == {"api.common"}


def test_declared_value_set_is_not_an_index_entry():
    """A collection literal is a vocabulary oracle, not a name<->value pair.

    Counting the two together makes an error-code module and a config module
    look like the same kind of file, which is what hid the distinction in the
    first place.
    """
    source = "EC_Kill = 100\nFINAL_TASK_STATUSES = ['done', 'failed']\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    # Both are declarations, so both are candidates -- the value set shows up
    # as an unexplained candidate rather than vanishing from the denominator.
    assert coverage[0].candidates == 2
    assert coverage[0].explained == 1


def test_computed_constants_stay_out_of_the_denominator():
    """A computed value is not a failed extraction, it is a different statement."""
    source = "EC_Kill = 100\nTIMEOUT = compute_timeout()\nLIMIT = 3 * 60\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    assert coverage[0].candidates == 1
    assert coverage[0].ratio == 1.0


def test_references_are_counted_across_the_whole_corpus():
    """A constant belongs in a decoding index only if something reads it."""
    definition = _module("EC_Kill = 100\nEC_Unused = 999\n")
    user = _module(
        "from x import ErrorCode\nif code == ErrorCode.EC_Kill:\n    pass\n",
        "pandaserver/dataservice/user.py",
    )
    enums, _ = errorcode.extract([definition, user], MAP_ID, VERSION)

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].references >= 1
    assert by_name["EC_Unused"].references == 0


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def _enum(constant: str, value, namespace: str = "taskbuffer.ErrorCode.EC", references: int = 1):
    return ValueEnumNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"{namespace}.{constant}",
        namespace=namespace,
        constant=constant,
        value=value,
        references=references,
        anchor=Anchor(package="pandaserver", file="x.py", line_start=1),
    )


def test_gate_5_flags_unreferenced_constants():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[_enum("EC_Kill", 100), _enum("EC_Dead", 206, references=0)],
    )
    result = gates.gate_5_value_enum_referenced(fragment)

    assert not result.passed
    assert result.checked == 2
    assert any("EC_Dead" in f for f in result.failures)


def test_gate_5b_allows_reuse_across_namespaces():
    """Reused numbers are the reason the key carries a namespace at all."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("EC_Kill", 100, "taskbuffer.ErrorCode.EC"),
            _enum("EC_Watcher", 100, "jobdispatcher.ErrorCode.EC"),
        ],
    )
    assert gates.gate_5b_namespace_disambiguates(fragment).passed


def test_gate_5b_rejects_collision_inside_one_namespace():
    """Within one namespace a repeated value makes the key identify nothing."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("ST_ready", 0, "taskbuffer.EventServiceUtils"),
            _enum("TASK_NORMAL", 0, "taskbuffer.EventServiceUtils"),
        ],
    )
    result = gates.gate_5b_namespace_disambiguates(fragment)

    assert not result.passed
    assert "ST_ready" in result.failures[0]


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def test_content_hash_ignores_the_anchor():
    """A gloss keyed on position would be thrown away by unrelated edits.

    Between two PanDA releases the pilot boundary moved file entirely while
    remaining the same boundary, so anything derived from a junction has to be
    keyed on what it means, not where it was found.
    """
    branches = [Branch(outcome="assigning", path_condition=["cloud is None"])]
    here = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="a.py", line_start=10),
    )
    moved = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="b.py", line_start=999),
    )
    assert here.content_hash() == moved.content_hash()


def test_content_hash_changes_when_the_condition_changes():
    common = {
        "map_id": MAP_ID,
        "derived_from": VERSION,
        "name": "n",
        "subject": "JediTaskSpec.status",
        "owner": "m::f",
    }
    before = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 1"])], **common)
    after = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 2"])], **common)

    assert before.content_hash() != after.content_hash()


# --------------------------------------------------------------------------- #
# storage namespace
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_store_fragment_replaces_only_its_own_version():
    """A rebuild clears this map at this version, never the incident graph.

    ``clear_all`` would take human-validated incident knowledge with it, and
    other versions must survive so an old incident can still be explained
    against the code that was running when it happened.
    """
    fragment = MapFragment(
        map_id=MAP_ID, derived_from=VERSION, value_enums=[_enum("EC_Kill", 100)]
    )
    graph_db = AsyncMock()
    graph_db.clear_map.return_value = 3

    from bamboo.codemap.store import store_fragment

    written = await store_fragment(fragment, graph_db)

    graph_db.clear_map.assert_awaited_once_with(MAP_ID, VERSION)
    graph_db.clear_all.assert_not_awaited()
    assert written["value_enums"] == 1
    assert graph_db.merge_map_node.await_count == 1


@pytest.mark.asyncio
async def test_store_fragment_can_keep_existing_versions():
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, value_enums=[_enum("EC_Kill", 100)])
    graph_db = AsyncMock()

    from bamboo.codemap.store import store_fragment

    await store_fragment(fragment, graph_db, replace_version=False)

    graph_db.clear_map.assert_not_awaited()


def test_code_map_labels_are_distinct_from_incident_labels():
    """Label separation is what makes a namespace-scoped delete possible."""
    code_map = {NodeType.JUNCTION_POINT, NodeType.BOUNDARY, NodeType.SUBJECT, NodeType.VALUE_ENUM}
    incident = {NodeType.SYMPTOM, NodeType.CAUSE, NodeType.RESOLUTION, NodeType.PROCEDURE}

    assert not {n.value for n in code_map} & {n.value for n in incident}
