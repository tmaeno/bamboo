"""Persist a Code Map fragment to the graph database.

The Code Map shares Neo4j with the incident graph but occupies its own labels.
That separation is load-bearing rather than cosmetic: the Code Map is derived
and disposable, rebuilt whenever the source version moves, while the incident
graph holds human-validated knowledge that cannot be regenerated.  Rebuilding
one must never touch the other, which is why writes go through
``clear_map`` / ``merge_map_node`` instead of the wholesale ``clear_all``.
"""

from __future__ import annotations

import logging

from bamboo.codemap.models import MapFragment
from bamboo.database.graph_database_client import GraphDatabaseClient

logger = logging.getLogger(__name__)


async def store_fragment(
    fragment: MapFragment,
    graph_db: GraphDatabaseClient,
    replace_version: bool = True,
) -> dict[str, int]:
    """Write *fragment* to the graph database.

    Args:
        fragment:        What a plugin produced.
        graph_db:        Connected client.
        replace_version: Drop this map's nodes for this exact version first, so
            a rebuild of the same version is idempotent.  Other versions are
            left alone -- several can coexist, which is what lets an incident
            from months ago be explained against the code that was running
            then, and what makes cross-version diffing possible at all.

    Returns:
        Counts of what was written, per node kind.
    """
    written = {
        "value_enums": 0,
        "subjects": 0,
        "junctions": 0,
        "boundaries": 0,
        "filter_stages": 0,
    }

    if replace_version:
        removed = await graph_db.clear_map(fragment.map_id, fragment.derived_from)
        logger.info(
            "store_fragment: cleared %d existing node(s) for %s @ %s",
            removed, fragment.map_id, fragment.derived_from,
        )

    for enum in fragment.value_enums:
        await graph_db.merge_map_node(enum)
        written["value_enums"] += 1
    for subject in fragment.subjects:
        await graph_db.merge_map_node(subject)
        written["subjects"] += 1
    for junction in fragment.junctions:
        await graph_db.merge_map_node(junction)
        written["junctions"] += 1
    for boundary in fragment.boundaries:
        await graph_db.merge_map_node(boundary)
        written["boundaries"] += 1
    for stage in fragment.filter_stages:
        await graph_db.merge_map_node(stage)
        written["filter_stages"] += 1

    logger.info("store_fragment: wrote %r", written)
    return written
