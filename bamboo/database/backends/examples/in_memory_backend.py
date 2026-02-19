"""
Example: In-Memory Database Backend

This is a simple example backend that stores data in memory.
Useful for testing and development without external dependencies.

To use this backend:
1. Add to config: GRAPH_DATABASE_BACKEND=in_memory
2. Or register programmatically:
   from bamboo.database.factory import register_graph_backend
   from bamboo.database.backends.examples.in_memory_backend import InMemoryGraphBackend
   register_graph_backend("in_memory", InMemoryGraphBackend)
"""

import logging
from typing import Any, Dict, Set

from bamboo.database.base import GraphDatabaseBackend
from bamboo.models.graph_element import BaseNode, GraphRelationship

logger = logging.getLogger(__name__)


class InMemoryGraphBackend(GraphDatabaseBackend):
    """In-memory graph database backend for testing and development."""

    def __init__(self):
        """Initialize in-memory backend."""
        self.nodes: Dict[str, BaseNode] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        self.node_index: Dict[str, Set[str]] = {}  # Index by type for quick lookups
        self.connected = False

    async def connect(self):
        """Establish connection (no-op for in-memory)."""
        self.connected = True
        logger.info("Connected to in-memory graph backend")

    async def close(self):
        """Close connection (no-op for in-memory)."""
        self.connected = False
        logger.info("In-memory graph backend closed")
        self.nodes.clear()
        self.relationships.clear()
        self.node_index.clear()

    async def create_node(self, node: BaseNode) -> str:
        """Create a node in memory."""
        if not node.id:
            import uuid

            node.id = str(uuid.uuid4())

        self.nodes[node.id] = node

        # Update index
        node_type = str(node.node_type)
        if node_type not in self.node_index:
            self.node_index[node_type] = set()
        self.node_index[node_type].add(node.id)

        logger.debug(f"Created node: {node.id} ({node.name})")
        return node.id

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Get existing node by canonical name or create new one."""
        node_type = str(node.node_type)

        # Search for existing node with canonical name
        if node_type in self.node_index:
            for node_id in self.node_index[node_type]:
                if self.nodes[node_id].name == canonical_name:
                    return node_id

        # Create new node with canonical name
        node.name = canonical_name
        return await self.create_node(node)

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between nodes."""
        # Verify both nodes exist
        if relationship.source_id not in self.nodes:
            logger.warning(f"Source node {relationship.source_id} not found")
            return False

        if relationship.target_id not in self.nodes:
            logger.warning(f"Target node {relationship.target_id} not found")
            return False

        rel_id = f"{relationship.source_id}-{relationship.relation_type}-{relationship.target_id}"
        self.relationships[rel_id] = relationship

        logger.debug(
            f"Created relationship: {relationship.source_id} "
            f"-[{relationship.relation_type}]-> {relationship.target_id}"
        )
        return True

    async def find_causes_by_error(
        self, error_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find possible causes for a given error."""
        results = []

        # Find error nodes matching the name
        for node_id, node in self.nodes.items():
            if "ERROR" in str(node.node_type) and (
                error_name.lower() in node.name.lower()
                or error_name.lower() in (node.description or "").lower()
            ):
                # Find causes connected via 'indicate' relationship
                for rel_id, rel in self.relationships.items():
                    if rel.source_id == node_id and "indicate" in str(
                        rel.relation_type
                    ):
                        cause_node = self.nodes.get(rel.target_id)
                        if cause_node and "CAUSE" in str(cause_node.node_type):
                            # Find resolutions
                            resolutions = []
                            for rel_id2, rel2 in self.relationships.items():
                                if (
                                    rel2.source_id == cause_node.id
                                    and "solved_by" in str(rel2.relation_type)
                                ):
                                    res_node = self.nodes.get(rel2.target_id)
                                    if res_node:
                                        resolutions.append(
                                            {
                                                "id": res_node.id,
                                                "name": res_node.name,
                                                "description": res_node.description,
                                            }
                                        )

                            results.append(
                                {
                                    "cause_id": cause_node.id,
                                    "cause_name": cause_node.name,
                                    "cause_description": cause_node.description,
                                    "confidence": getattr(
                                        cause_node, "confidence", 1.0
                                    ),
                                    "frequency": getattr(cause_node, "frequency", 1),
                                    "resolutions": resolutions,
                                }
                            )

        return results[:limit]

    async def find_causes_by_features(
        self, features: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find possible causes based on task features."""
        results = []
        matched_causes = {}

        # Find feature nodes and their connected causes
        for node_id, node in self.nodes.items():
            if "FEATURE" in str(node.node_type) and node.name in features:
                # Find causes connected via 'contribute_to' relationship
                for rel_id, rel in self.relationships.items():
                    if rel.source_id == node_id and "contribute_to" in str(
                        rel.relation_type
                    ):
                        cause_node = self.nodes.get(rel.target_id)
                        if cause_node and "CAUSE" in str(cause_node.node_type):
                            if cause_node.id not in matched_causes:
                                matched_causes[cause_node.id] = {
                                    "cause": cause_node,
                                    "matching_features": [node.name],
                                }
                            else:
                                matched_causes[cause_node.id][
                                    "matching_features"
                                ].append(node.name)

        # Build results
        for cause_id, data in matched_causes.items():
            cause_node = data["cause"]

            # Find resolutions
            resolutions = []
            for rel_id, rel in self.relationships.items():
                if rel.source_id == cause_node.id and "solved_by" in str(
                    rel.relation_type
                ):
                    res_node = self.nodes.get(rel.target_id)
                    if res_node:
                        resolutions.append(
                            {
                                "id": res_node.id,
                                "name": res_node.name,
                                "description": res_node.description,
                            }
                        )

            results.append(
                {
                    "cause_id": cause_node.id,
                    "cause_name": cause_node.name,
                    "cause_description": cause_node.description,
                    "confidence": getattr(cause_node, "confidence", 1.0),
                    "frequency": getattr(cause_node, "frequency", 1),
                    "matching_features": data["matching_features"],
                    "resolutions": resolutions,
                }
            )

        # Sort by number of matching features
        results.sort(
            key=lambda x: (len(x["matching_features"]), x["frequency"]),
            reverse=True,
        )

        return results[:limit]

    async def increment_cause_frequency(self, cause_id: str):
        """Increment the frequency counter for a cause."""
        if cause_id in self.nodes:
            node = self.nodes[cause_id]
            if hasattr(node, "frequency"):
                node.frequency += 1
                logger.debug(f"Incremented frequency for {cause_id}: {node.frequency}")

    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update resolution success rate based on feedback."""
        if resolution_id in self.nodes:
            node = self.nodes[resolution_id]

            if not hasattr(node, "total_attempts"):
                node.total_attempts = 0
            if not hasattr(node, "successful_attempts"):
                node.successful_attempts = 0

            node.total_attempts += 1
            if success:
                node.successful_attempts += 1

            node.success_rate = (
                node.successful_attempts / node.total_attempts
                if node.total_attempts > 0
                else 0
            )
            logger.debug(
                f"Updated success rate for {resolution_id}: {node.success_rate}"
            )
