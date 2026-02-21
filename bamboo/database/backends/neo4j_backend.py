"""Neo4j graph database backend implementation.

Provides :class:`Neo4jBackend`, a concrete :class:`GraphDatabaseBackend`
that persists the Bamboo knowledge graph in a Neo4j database using the
async Python driver.

Connection is established via :meth:`Neo4jBackend.connect` (called by
:class:`~bamboo.database.graph_database_client.GraphDatabaseClient`).
All methods require an open driver; call :meth:`Neo4jBackend.close` when done.
"""

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
from bamboo.models.graph_element import BaseNode, GraphRelationship, NodeType

logger = logging.getLogger(__name__)


class Neo4jBackend(GraphDatabaseBackend):
    """Async Neo4j implementation of :class:`GraphDatabaseBackend`.

    Uses the official ``neo4j`` async driver.  All public methods open a
    short-lived session so they are safe to call concurrently.
    """

    def __init__(self):
        self.settings = get_settings()
        self.driver = None

    async def connect(self):
        """Open the Neo4j driver, verify connectivity, and create indexes."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
            )
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self.settings.neo4j_uri)
            await self._create_indexes()
        except Neo4jError as exc:
            logger.error("Failed to connect to Neo4j: %s", exc)
            raise

    async def close(self):
        """Close the Neo4j driver and release all connections."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def _create_indexes(self):
        """Create uniqueness constraints and performance indexes.

        Idempotent — uses ``IF NOT EXISTS`` so re-running on an already
        initialised database is safe.
        """
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            for node_type in NodeType:
                await session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type.value}) "
                    "REQUIRE n.id IS UNIQUE"
                )
            for query in [
                "CREATE INDEX IF NOT EXISTS FOR (n:Cause) ON (n.confidence)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Cause) ON (n.frequency)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Resolution) ON (n.success_rate)",
            ]:
                await session.run(query)

    async def create_node(self, node: BaseNode) -> str:
        """Create a new node unconditionally and return its ID.

        A UUID is generated if ``node.id`` is not already set.

        .. note::
            Prefer :meth:`get_or_create_canonical_node` to avoid duplicates
            for canonical node types (Cause, Resolution, Symptom, etc.).
        """
        import uuid
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            properties = node.model_dump(exclude={"node_type"})
            if not properties.get("id"):
                properties["id"] = str(uuid.uuid4())
            result = await session.run(
                f"CREATE (n:{node.node_type.value} $properties) RETURN n.id AS id",
                properties=properties,
            )
            record = await result.single()
            return record["id"]

    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Merge on canonical name: return existing ID or create and return new ID.

        Looks up a node of the same type with ``name = canonical_name``.  If
        found, returns its ID without modification.  If not found, sets
        ``node.name = canonical_name`` and creates a new node.

        Args:
            node:           Source node whose properties are used when creating.
            canonical_name: Stable name used as the merge key.

        Returns:
            The node's ID string.
        """
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            result = await session.run(
                f"MATCH (n:{node.node_type.value} {{name: $name}}) RETURN n.id AS id",
                name=canonical_name,
            )
            record = await result.single()
            if record:
                return record["id"]
            node.name = canonical_name
            return await self.create_node(node)

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a directed relationship between two nodes.

        Nodes are looked up by their ``id`` property.  Returns ``False``
        (without raising) if either node cannot be found.

        Args:
            relationship: Edge descriptor.

        Returns:
            ``True`` if the relationship was created.
        """
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            properties = relationship.properties.copy()
            properties["confidence"] = relationship.confidence
            result = await session.run(
                f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                CREATE (source)-[r:{relationship.relation_type.value} $properties]->(target)
                RETURN r
                """,
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

