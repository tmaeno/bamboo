"""Neo4j graph database backend implementation."""

import logging
from typing import Any

try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError as e:
    raise ImportError(
        "Neo4j backend requires 'neo4j' package. "
        "Install it with: pip install neo4j"
    ) from e

from bamboo.config import get_settings
from bamboo.database.base import GraphDatabaseBackend
from bamboo.models.graph_element import (
    BaseNode,
    GraphRelationship,
    NodeType,
)

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphDatabaseBackend):
    """Neo4j implementation of graph database backend."""

    def __init__(self):
        """Initialize Neo4j backend."""
        self.settings = get_settings()
        self.driver = None

    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
            )
            await self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")
            await self._create_indexes()
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def _create_indexes(self):
        """Create necessary indexes and constraints."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            # Create constraints for unique node IDs
            for node_type in NodeType:
                constraint_query = f"""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type.value})
                REQUIRE n.id IS UNIQUE
                """
                await session.run(constraint_query)

            # Create indexes for common queries
            index_queries = [
                "CREATE INDEX IF NOT EXISTS FOR (n:Cause) ON (n.confidence)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Cause) ON (n.frequency)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Error) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Resolution) ON (n.success_rate)",
            ]
            for query in index_queries:
                await session.run(query)

    async def create_node(self, node: BaseNode) -> str:
        """Create a node in Neo4j."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = f"""
            CREATE (n:{node.node_type.value} $properties)
            RETURN n.id as id
            """
            properties = node.model_dump(exclude={"node_type"})
            if not properties.get("id"):
                import uuid

                properties["id"] = str(uuid.uuid4())

            result = await session.run(query, properties=properties)
            record = await result.single()
            return record["id"]

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Get existing node by canonical name or create new one."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            # Try to find existing node with canonical name
            find_query = f"""
            MATCH (n:{node.node_type.value} {{name: $name}})
            RETURN n.id as id
            """

            result = await session.run(find_query, name=canonical_name)
            record = await result.single()

            if record:
                return record["id"]

            # Create new node with canonical name
            node.name = canonical_name
            return await self.create_node(node)

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between nodes."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = (
                """
            MATCH (source {id: $source_id})
            MATCH (target {id: $target_id})
            CREATE (source)-[r:%s $properties]->(target)
            RETURN r
            """
                % relationship.relation_type.value
            )

            properties = relationship.properties.copy()
            properties["confidence"] = relationship.confidence

            result = await session.run(
                query,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=properties,
            )
            return await result.single() is not None

    async def find_causes_by_error(
        self, error_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find possible causes for a given error."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = """
            MATCH (e:Error)-[:indicate]->(c:Cause)
            WHERE e.name CONTAINS $error_name OR e.description CONTAINS $error_name
            OPTIONAL MATCH (c)-[:solved_by]->(r:Resolution)
            RETURN c.id as cause_id, c.name as cause_name, c.description as cause_description,
                   c.confidence as confidence, c.frequency as frequency,
                   collect({id: r.id, name: r.name, description: r.description, 
                           steps: r.steps, success_rate: r.success_rate}) as resolutions
            ORDER BY c.frequency DESC, c.confidence DESC
            LIMIT $limit
            """
            result = await session.run(query, error_name=error_name, limit=limit)
            records = await result.values()
            return [
                {
                    "cause_id": record[0],
                    "cause_name": record[1],
                    "cause_description": record[2],
                    "confidence": record[3],
                    "frequency": record[4],
                    "resolutions": record[5],
                }
                for record in records
            ]

    async def find_causes_by_features(
        self, features: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Find possible causes based on task features."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = """
            MATCH (f:Feature)-[:contribute_to]->(c:Cause)
            WHERE f.name IN $features
            OPTIONAL MATCH (c)-[:solved_by]->(r:Resolution)
            RETURN c.id as cause_id, c.name as cause_name, c.description as cause_description,
                   c.confidence as confidence, c.frequency as frequency,
                   collect(DISTINCT f.name) as matching_features,
                   collect({id: r.id, name: r.name, description: r.description,
                           steps: r.steps, success_rate: r.success_rate}) as resolutions
            ORDER BY size(matching_features) DESC, c.frequency DESC
            LIMIT $limit
            """
            result = await session.run(query, features=features, limit=limit)
            records = await result.values()
            return [
                {
                    "cause_id": record[0],
                    "cause_name": record[1],
                    "cause_description": record[2],
                    "confidence": record[3],
                    "frequency": record[4],
                    "matching_features": record[5],
                    "resolutions": record[6],
                }
                for record in records
            ]

    async def increment_cause_frequency(self, cause_id: str):
        """Increment the frequency counter for a cause."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = """
            MATCH (c:Cause {id: $cause_id})
            SET c.frequency = c.frequency + 1
            RETURN c.frequency as frequency
            """
            await session.run(query, cause_id=cause_id)

    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update resolution success rate based on feedback."""
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = """
            MATCH (r:Resolution {id: $resolution_id})
            SET r.total_attempts = COALESCE(r.total_attempts, 0) + 1,
                r.successful_attempts = COALESCE(r.successful_attempts, 0) + $success_increment,
                r.success_rate = toFloat(COALESCE(r.successful_attempts, 0) + $success_increment) / 
                                 toFloat(COALESCE(r.total_attempts, 0) + 1)
            RETURN r.success_rate as success_rate
            """
            await session.run(
                query,
                resolution_id=resolution_id,
                success_increment=1 if success else 0,
            )



