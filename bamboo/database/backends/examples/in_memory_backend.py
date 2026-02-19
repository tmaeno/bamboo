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

    async def find_causes(
        self,
        errors: list[str] = None,
        task_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find possible causes ranked by total evidence across all clue types."""
        # Map each clue type to its node-type substring and relationship type
        clue_groups = [
            (errors or [], "ERROR", "indicate"),
            (task_features or [], "FEATURE", "contribute_to"),
            (environment_factors or [], "ENVIRONMENT", "contribute_to"),
            (components or [], "COMPONENT", "contribute_to"),
        ]

        # cause_id -> {"cause": node, "match_score": int, "resolutions": list}
        matched: dict[str, dict] = {}

        for clue_values, node_type_substr, rel_type in clue_groups:
            if not clue_values:
                continue
            for node_id, node in self.nodes.items():
                if node_type_substr not in str(node.node_type):
                    continue
                if node.name not in clue_values:
                    continue
                # Follow the relationship to a Cause node
                for rel in self.relationships.values():
                    if rel.source_id != node_id or rel_type not in str(
                        rel.relation_type
                    ):
                        continue
                    cause_node = self.nodes.get(rel.target_id)
                    if not cause_node or "CAUSE" not in str(cause_node.node_type):
                        continue
                    if cause_node.id not in matched:
                        matched[cause_node.id] = {
                            "cause": cause_node,
                            "match_score": 0,
                            "resolutions": [],
                        }
                    # Each clue type contributes at most +1 to match_score
                    matched[cause_node.id]["match_score"] += 1

        # Collect resolutions for each matched cause
        for cause_id, data in matched.items():
            for rel in self.relationships.values():
                if rel.source_id != cause_id or "solved_by" not in str(
                    rel.relation_type
                ):
                    continue
                res_node = self.nodes.get(rel.target_id)
                if res_node:
                    data["resolutions"].append(
                        {
                            "id": res_node.id,
                            "name": res_node.name,
                            "description": res_node.description,
                        }
                    )

        results = [
            {
                "cause_id": data["cause"].id,
                "cause_name": data["cause"].name,
                "cause_description": data["cause"].description,
                "confidence": getattr(data["cause"], "confidence", 1.0),
                "frequency": getattr(data["cause"], "frequency", 1),
                "match_score": data["match_score"],
                "resolutions": data["resolutions"],
            }
            for data in matched.values()
        ]
        results.sort(
            key=lambda x: (x["match_score"], x["frequency"], x["confidence"]),
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
