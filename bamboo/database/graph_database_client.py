"""Graph database client: thin façade over the pluggable backend.

:class:`GraphDatabaseClient` is the single interface used by the rest of the
application to interact with the graph database.  It delegates every call to
the :class:`~bamboo.database.base.GraphDatabaseBackend` selected by
:func:`~bamboo.database.factory.get_graph_backend`.

Usage::

    client = GraphDatabaseClient()
    await client.connect()
    try:
        node_id = await client.get_or_create_canonical_node(node, node.name)
    finally:
        await client.close()
"""

import logging
from typing import Any

from bamboo.database.factory import get_graph_backend
from bamboo.models.graph_element import BaseNode, GraphRelationship

logger = logging.getLogger(__name__)


class GraphDatabaseClient:
    """Façade over the configured :class:`~bamboo.database.base.GraphDatabaseBackend`.

    All method signatures mirror those of the backend interface.  See
    :class:`~bamboo.database.base.GraphDatabaseBackend` for full
    documentation of each operation.
    """

    def __init__(self):
        self._backend = get_graph_backend()

    async def connect(self):
        """Open the backend connection."""
        await self._backend.connect()

    async def close(self):
        """Close the backend connection."""
        await self._backend.close()

    async def create_node(self, node: BaseNode) -> str:
        """Create a node unconditionally and return its ID."""
        return await self._backend.create_node(node)

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Merge on *canonical_name*: return existing ID or create new node."""
        return await self._backend.get_or_create_canonical_node(node, canonical_name)

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a directed relationship between two nodes."""
        return await self._backend.create_relationship(relationship)

    async def find_causes(
        self,
        symptoms: list[str] = None,
        task_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find candidate causes ranked by evidence breadth across clue types."""
        return await self._backend.find_causes(
            symptoms=symptoms,
            task_features=task_features,
            environment_factors=environment_factors,
            components=components,
            limit=limit,
        )

    async def increment_cause_frequency(self, cause_id: str):
        """Increment the frequency counter on a cause node."""
        return await self._backend.increment_cause_frequency(cause_id)

    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update the running success-rate statistic on a resolution node."""
        return await self._backend.update_resolution_success_rate(
            resolution_id, success
        )
