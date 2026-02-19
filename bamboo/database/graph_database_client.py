"""Graph database client with pluggable backend support."""

import logging
from typing import Any

from bamboo.database.factory import get_graph_backend
from bamboo.models.graph_element import BaseNode, GraphRelationship

logger = logging.getLogger(__name__)


class GraphDatabaseClient:
    """Client for graph database operations with pluggable backend support."""

    def __init__(self):
        """Initialize graph database client."""
        self._backend = get_graph_backend()

    async def connect(self):
        """Establish connection to graph database backend."""
        await self._backend.connect()

    async def close(self):
        """Close connection."""
        await self._backend.close()

    async def create_node(self, node: BaseNode) -> str:
        """Create a node in the graph."""
        return await self._backend.create_node(node)

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Get existing node by canonical name or create new one."""
        return await self._backend.get_or_create_canonical_node(node, canonical_name)

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between nodes."""
        return await self._backend.create_relationship(relationship)

    async def find_causes(
        self,
        errors: list[str] = None,
        task_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find possible causes ranked by total evidence across all clue types."""
        return await self._backend.find_causes(
            symptoms=errors,
            task_features=task_features,
            environment_factors=environment_factors,
            components=components,
            limit=limit,
        )

    async def increment_cause_frequency(self, cause_id: str):
        """Increment the frequency counter for a cause."""
        return await self._backend.increment_cause_frequency(cause_id)

    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update resolution success rate based on feedback."""
        return await self._backend.update_resolution_success_rate(
            resolution_id, success
        )
