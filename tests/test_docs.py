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

from bamboo.database.base import GraphDatabaseBackend, VectorDatabaseBackend
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


# ---------------------------------------------------------------------------
# The database-plugin pages, which tell a contributor what to implement.
# ---------------------------------------------------------------------------

PLUGIN_DOCS = SCHEMA_PAGE.parent.parent / "database-plugins"

_BACKENDS = (
    ("GraphDatabaseBackend", GraphDatabaseBackend),
    ("VectorDatabaseBackend", VectorDatabaseBackend),
)


@pytest.fixture(scope="module")
def plugin_pages() -> dict[str, str]:
    if not PLUGIN_DOCS.is_dir():  # pragma: no cover - only if the site moves
        pytest.skip(f"{PLUGIN_DOCS} is not in this checkout")
    return {p.name: p.read_text() for p in PLUGIN_DOCS.glob("*.md")}


@pytest.mark.parametrize("label,backend", _BACKENDS, ids=lambda x: getattr(x, "__name__", x))
def test_the_interface_page_lists_every_method_a_backend_must_have(
    plugin_pages: dict[str, str], label: str, backend: type
) -> None:
    """A short list here is not a cosmetic problem.

    Every method is abstract, so a contributor who implements exactly what the
    page lists gets a ``TypeError`` on instantiation.  The page listed 8 of 19
    for the graph backend and 6 of 9 for the vector one.
    """
    page = plugin_pages["implementation.md"]
    missing = sorted(m for m in backend.__abstractmethods__ if f"`{m}(" not in page)

    assert not missing, f"{label} methods missing from implementation.md: {missing}"


def test_the_stated_interface_sizes_match_the_abstract_classes(
    plugin_pages: dict[str, str],
) -> None:
    """Counts, which a reader believes without counting, in both pages."""
    graph = len(GraphDatabaseBackend.__abstractmethods__)
    vector = len(VectorDatabaseBackend.__abstractmethods__)

    implementation = plugin_pages["implementation.md"]
    assert f"### GraphDatabaseBackend ({graph} methods)" in implementation
    assert f"### VectorDatabaseBackend ({vector} methods)" in implementation
    assert f"{graph} and {vector} abstract methods respectively" in implementation

    checklist = plugin_pages["checklist.md"]
    assert f"`GraphDatabaseBackend` interface with {graph} methods" in checklist
    assert f"`VectorDatabaseBackend` interface with {vector} methods" in checklist
    assert f"Full implementation of all {graph} methods" in checklist
