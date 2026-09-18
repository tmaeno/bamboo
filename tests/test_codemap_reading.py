"""Choosing the code a reader is given, and noticing when it is the wrong code.

The unit under test is a *choice*, so the tests are about the choice being the
map's and not the asker's: the same junction must select the same text, the
line must win over the name where a file reuses one, and a tree that is not the
one the map was built from must be said out loud rather than quietly read.
"""

from __future__ import annotations

import ast

from bamboo.codemap import reading
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    JunctionNode,
    MapFragment,
    SourceModule,
)

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"

TWO_OF_A_NAME = '''\
class Early:
    def run(self):
        spec.status = "early"


class Late:
    def run(self):
        if held:
            spec.status = "late"
        return spec
'''


def _module(source: str, rel_path: str = "pkg/mod.py") -> SourceModule:
    return SourceModule(
        package="pkg",
        rel_path=rel_path,
        tree=ast.parse(source),
        source=source,
        blob_sha=blob_sha(source),
    )


def test_the_line_picks_the_function_where_the_name_is_ambiguous() -> None:
    """``_read_method`` takes the first function of a name; the map has the line.

    38 files in the corpus define some name more than once, 99 times over.  It
    happens to be harmless for all but three of 497 junctions -- which is
    exactly why a name-based reader looks like it works.
    """
    module = _module(TWO_OF_A_NAME)

    early = reading.region_in(
        module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::Early.run", line=3
    )
    late = reading.region_in(
        module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=9
    )

    assert early is not None and late is not None
    assert 'spec.status = "early"' in early.text
    assert 'spec.status = "early"' not in late.text
    assert late.line_start == 7
    assert early.gloss_key != late.gloss_key


def test_arms_of_one_function_share_a_key() -> None:
    """The sharing is the reason this is keyed at all: read once, explain all."""
    module = _module(TWO_OF_A_NAME)

    first = reading.region_in(
        module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=8
    )
    second = reading.region_in(
        module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=9
    )

    assert first is not None and second is not None
    assert first.gloss_key == second.gloss_key


def test_the_key_changes_when_the_function_does() -> None:
    """A reading is only about the text it was shown."""
    one = _module(TWO_OF_A_NAME)
    other = _module(TWO_OF_A_NAME.replace('"late"', '"later"'))

    before = reading.region_in(
        one.source, one.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=9
    )
    after = reading.region_in(
        other.source, other.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=9
    )

    assert before is not None and after is not None
    assert before.gloss_key != after.gloss_key


def test_a_region_read_from_another_snapshot_says_so() -> None:
    """The failure this catches is silent and happened on the first real run.

    A map built from the installed distribution was read against a checkout and
    the report marked a line that exists in both versions and means different
    things in each.  Line drift is one of the two skew symptoms no gate catches.
    """
    module = _module(TWO_OF_A_NAME)

    same = reading.region_in(
        module.source,
        module.tree,
        file="pkg/mod.py",
        owner="pkg/mod.py::Late.run",
        line=9,
        expected_sha=module.blob_sha,
    )
    drifted = reading.region_in(
        module.source,
        module.tree,
        file="pkg/mod.py",
        owner="pkg/mod.py::Late.run",
        line=9,
        expected_sha=blob_sha("something else entirely"),
    )

    assert same is not None and not same.off_version
    assert drifted is not None and drifted.off_version


def test_a_write_outside_any_function_gets_no_key() -> None:
    """Module-scope writes have nothing to hand a reader, and five are real.

    Reported rather than filled in: an empty key that means "no function here"
    and one that means "the reader was not run" would be the same value, which
    is how a missing reading hides inside a present one.
    """
    source = "DEFAULT = Spec()\nDEFAULT.status = 'offline'\n"
    module = _module(source)

    assert (
        reading.region_in(
            module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::<module>", line=2
        )
        is None
    )


def test_marking_keeps_the_whole_function() -> None:
    """Marking, not excerpting.

    Cutting the text down to the marked lines is the thing the measurement
    rules out: only 35 of 927 arms refer to nothing outside their own guard.
    """
    module = _module(TWO_OF_A_NAME)
    region = reading.region_in(
        module.source, module.tree, file="pkg/mod.py", owner="pkg/mod.py::Late.run", line=9
    )

    assert region is not None
    marked = region.marked([9])

    assert marked.count("\n") + 1 == region.line_end - region.line_start + 1
    assert ">>>     9" in marked
    assert "    7  " in marked


def _junction(line: int, name: str) -> JunctionNode:
    return JunctionNode(
        name=JunctionNode.make_name(MAP_ID, "Spec.status", name),
        map_id=MAP_ID,
        derived_from=VERSION,
        subject="Spec.status",
        owner="pkg/mod.py::Late.run",
        branches=[Branch(outcome="late", line=line)],
        anchor=Anchor(package="pkg", file="pkg/mod.py", line_start=line),
    )


def test_attach_keys_every_node_it_can_locate() -> None:
    """Eager, because the choice of input belongs to the map and not the asker."""
    module = _module(TWO_OF_A_NAME)
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        junctions=[_junction(8, "a"), _junction(9, "b")],
    )

    keyed = reading.attach(fragment, [module])

    assert keyed == 2
    assert fragment.junctions[0].gloss_key
    # Both arms are in one function, so one reading answers for both.
    assert fragment.junctions[0].gloss_key == fragment.junctions[1].gloss_key
