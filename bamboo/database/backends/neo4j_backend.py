"""Neo4j graph database backend implementation."""

import logging
from typing import Any

try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError as e:
    raise ImportError(
        "Neo4j backend requires 'neo4j' package. " "Install it with: pip install neo4j"
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
                "CREATE INDEX IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
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

    async def find_causes(
        self,
        symptoms: list[str] = None,
        task_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find possible causes ranked by total evidence across all clue types.

        Each clue type that links to a cause contributes +1 to its match_score,
        so causes corroborated by multiple clue types rank above those matched
        by only one.
        """
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            query = """
            // --- Symptom clues ---
            OPTIONAL MATCH (e:Symptom)-[:indicate]->(c1:Cause)
            WHERE e.name IN $symptoms
            WITH collect(DISTINCT c1) AS symptom_causes

            // --- Task-feature clues ---
            OPTIONAL MATCH (f:Feature)-[:contribute_to]->(c2:Cause)
            WHERE f.name IN $task_features
            WITH symptom_causes, collect(DISTINCT c2) AS feature_causes

            // --- Environment clues ---
            OPTIONAL MATCH (env:Environment)-[:contribute_to]->(c3:Cause)
            WHERE env.name IN $environment_factors
            WITH symptom_causes, feature_causes, collect(DISTINCT c3) AS env_causes

            // --- Component clues ---
            OPTIONAL MATCH (comp:Component)-[:contribute_to]->(c4:Cause)
            WHERE comp.name IN $components
            WITH symptom_causes, feature_causes, env_causes,
                 collect(DISTINCT c4) AS comp_causes

            // --- Union all matched causes and score by clue-type breadth ---
            WITH symptom_causes + feature_causes + env_causes + comp_causes AS all_causes,
                 symptom_causes, feature_causes, env_causes, comp_causes
            UNWIND all_causes AS c
            WITH DISTINCT c,
                 (CASE WHEN c IN symptom_causes      THEN 1 ELSE 0 END +
                  CASE WHEN c IN feature_causes      THEN 1 ELSE 0 END +
                  CASE WHEN c IN env_causes          THEN 1 ELSE 0 END +
                  CASE WHEN c IN comp_causes         THEN 1 ELSE 0 END) AS match_score

            OPTIONAL MATCH (c)-[:solved_by]->(r:Resolution)
            RETURN c.id          AS cause_id,
                   c.name        AS cause_name,
                   c.description AS cause_description,
                   c.confidence  AS confidence,
                   c.frequency   AS frequency,
                   match_score,
                   collect({id: r.id, name: r.name, description: r.description,
                            steps: r.steps, success_rate: r.success_rate}) AS resolutions
            ORDER BY match_score DESC, c.frequency DESC, c.confidence DESC
            LIMIT $limit
            """
            result = await session.run(
                query,
                symptoms=symptoms or [],
                task_features=task_features or [],
                environment_factors=environment_factors or [],
                components=components or [],
                limit=limit,
            )
            records = await result.values()
            return [
                {
                    "cause_id": record[0],
                    "cause_name": record[1],
                    "cause_description": record[2],
                    "confidence": record[3],
                    "frequency": record[4],
                    "match_score": record[5],
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


