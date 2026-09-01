"""Checks that keep the documentation site from drifting away from the code.

Prose cannot be checked automatically and is not what these are for.  What can
be checked is the places where a page enumerates something the code also
enumerates -- those go stale silently, and this repository has already paid for
that twice: ``config.py`` points at a ``docs/AGENTS.md`` that no longer exists,
and the graph schema page claimed eighteen node types for months after the Code
Map added five more.

So the rule these encode is narrow: **where a page lists what an enum declares,
the list has to be complete.**  A type nobody documented is a type nobody knows
they can query.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bamboo.models.graph_element import NodeType, RelationType

SCHEMA_PAGE = (
    Path(__file__).resolve().parent.parent
    / "website"
    / "src"
    / "content"
    / "docs"
    / "architecture"
    / "schema.md"
)


@pytest.fixture(scope="module")
def schema_text() -> str:
    if not SCHEMA_PAGE.exists():  # pragma: no cover - only if the site moves
        pytest.skip(f"{SCHEMA_PAGE} is not in this checkout")
    return SCHEMA_PAGE.read_text()


def test_every_node_type_is_documented(schema_text: str) -> None:
    """A node type absent from the schema page is a label nobody can query for.

    Whole-line matching against the page's ``- Name: description`` form rather
    than a substring search: ``Task`` appears inside ``Task_Feature``, so a
    substring test would pass on a page that never mentions it.
    """
    documented = {
        line.split(":", 1)[0].removeprefix("- ").strip()
        for line in schema_text.splitlines()
        if line.startswith("- ") and ":" in line
    }
    missing = sorted(t.value for t in NodeType if t.value not in documented)

    assert not missing, f"NodeType values missing from schema.md: {missing}"


def test_every_relation_type_is_documented(schema_text: str) -> None:
    """The same for edges, which is the half a reader writing Cypher needs."""
    documented = {
        line.split(":", 1)[0].removeprefix("- ").strip()
        for line in schema_text.splitlines()
        if line.startswith("- ") and ":" in line
    }
    missing = sorted(r.value for r in RelationType if r.value not in documented)

    assert not missing, f"RelationType values missing from schema.md: {missing}"


def test_the_stated_counts_match_the_enums(schema_text: str) -> None:
    """The page states totals in two places, and a total is the thing a reader
    trusts without counting.  Both were wrong."""
    assert f"**{len(list(NodeType))} node types**" in schema_text
    assert f"**{len(list(RelationType))} relationship types**" in schema_text
    assert f"### Node types ({len(list(NodeType))})" in schema_text
    assert f"### Relationship types ({len(list(RelationType))})" in schema_text
