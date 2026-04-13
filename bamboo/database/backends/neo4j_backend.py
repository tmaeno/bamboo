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
        "Neo4j backend requires 'neo4j' package. " "Install it with: pip install neo4j"
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
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
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
                "CREATE INDEX IF NOT EXISTS FOR (n:Task_Feature) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Aggregated_Job_Feature) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Aggregated_Job_Feature) ON (n.attribute)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Job_Instance) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Job_Instance) ON (n.site)",
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

        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
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
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
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
        """Merge a directed relationship between two nodes and track provenance.

        Uses MERGE on ``(source)-[rel_type]->(target)`` so the same logical edge
        is stored only once regardless of how many tasks share it.  Each call
        records the relationship's ``graph_id`` (stored in
        ``relationship.properties["graph_id"]``) in a deduplicated list on the
        edge, and derives ``frequency`` from that list's length.  Re-processing
        the same task (same ``graph_id``) is therefore idempotent.

        Nodes are looked up by their ``id`` property.  Returns ``False``
        (without raising) if either node cannot be found.

        Args:
            relationship: Edge descriptor.  ``relationship.properties["graph_id"]``
                          should be set by the caller before invoking this method.

        Returns:
            ``True`` if the relationship was merged/created successfully.
        """
        graph_id = relationship.properties.pop("graph_id", "")
        parameters = relationship.properties.pop("parameters", None)
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            result = await session.run(
                f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{relationship.relation_type.value}]->(target)
                ON CREATE SET
                    r.confidence  = $confidence,
                    r.graph_ids   = CASE WHEN $graph_id <> '' THEN [$graph_id] ELSE [] END,
                    r.frequency   = CASE WHEN $graph_id <> '' THEN 1 ELSE 0 END,
                    r.parameters  = CASE WHEN $parameters IS NOT NULL THEN [$parameters] ELSE [] END
                ON MATCH SET
                    r.graph_ids   = CASE
                        WHEN $graph_id = '' OR $graph_id IN r.graph_ids THEN r.graph_ids
                        ELSE r.graph_ids + $graph_id
                    END,
                    r.frequency   = size(CASE
                        WHEN $graph_id = '' OR $graph_id IN r.graph_ids THEN r.graph_ids
                        ELSE r.graph_ids + $graph_id
                    END),
                    r.confidence  = CASE
                        WHEN $graph_id IN r.graph_ids THEN r.confidence
                        ELSE (r.confidence * r.frequency + $confidence) /
                             (r.frequency + 1)
                    END,
                    r.parameters  = CASE
                        WHEN $parameters IS NULL OR $graph_id IN coalesce(r.graph_ids, [])
                            THEN coalesce(r.parameters, [])
                        ELSE coalesce(r.parameters, []) + [$parameters]
                    END
                RETURN r
                """,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                confidence=relationship.confidence,
                graph_id=graph_id,
                parameters=parameters,
            )
            return await result.single() is not None

    async def find_common_pattern(
        self,
        graph_ids: list[str],
        min_occurrences: int = 2,
    ) -> list[dict]:
        """Return edges shared across at least *min_occurrences* of the given graphs.

        Only edges whose ``graph_ids`` list contains at least *min_occurrences*
        values from *graph_ids* are returned.  The endpoint nodes are included
        inline.  Isolated common nodes (nodes that are common but have no
        qualifying edges) are naturally excluded.

        Args:
            graph_ids:        List of graph IDs to intersect (derived from
                              ``KnowledgeAccumulator._deterministic_id``).
            min_occurrences:  Minimum number of graphs that must share an edge
                              for it to appear in the result.

        Returns:
            List of dicts with keys ``src_name``, ``src_type``, ``tgt_name``,
            ``tgt_type``, ``rel_type``, ``occurrence_count``, ``confidence``.
            Ordered by ``occurrence_count DESC``.
        """
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            result = await session.run(
                """
                MATCH (src)-[r]->(tgt)
                WITH src, tgt, type(r) AS rel_type, r.confidence AS confidence,
                     r.graph_ids AS all_graph_ids,
                     [gid IN r.graph_ids WHERE gid IN $graph_ids] AS matched_ids
                WHERE size(matched_ids) >= $min_occurrences
                RETURN
                    src.name AS src_name,
                    labels(src)[0] AS src_type,
                    tgt.name AS tgt_name,
                    labels(tgt)[0] AS tgt_type,
                    rel_type,
                    confidence,
                    size(matched_ids) AS occurrence_count
                ORDER BY occurrence_count DESC
                """,
                graph_ids=graph_ids,
                min_occurrences=min_occurrences,
            )
            return [dict(record) async for record in result]

    async def remove_graph_id(self, graph_id: str) -> dict[str, int]:
        """Remove a graph_id's contribution from all relationships and clean up.

        Step 1: for every relationship whose ``graph_ids`` list contains
        *graph_id*, remove it from the list.  Delete the relationship entirely
        if the list becomes empty (no other task shares it).

        Step 2: delete any nodes that are now fully isolated (no remaining edges).

        Returns:
            Dict with ``rels_affected`` (updated or deleted) and ``nodes_removed``.
        """
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            # Step 1: update / delete relationships in one pass.
            rel_result = await session.run(
                """
                MATCH ()-[r]->()
                WHERE $graph_id IN r.graph_ids
                WITH r,
                     [gid IN r.graph_ids WHERE gid <> $graph_id] AS remaining
                CALL {
                    WITH r, remaining
                    FOREACH (_ IN CASE WHEN size(remaining) = 0 THEN [1] ELSE [] END |
                        DELETE r
                    )
                    FOREACH (_ IN CASE WHEN size(remaining) > 0 THEN [1] ELSE [] END |
                        SET r.graph_ids = remaining,
                            r.frequency = size(remaining)
                    )
                }
                RETURN count(r) AS rels_affected
                """,
                graph_id=graph_id,
            )
            rel_record = await rel_result.single()
            rels_affected = rel_record["rels_affected"] if rel_record else 0

            # Step 2: delete nodes with no remaining edges.
            node_result = await session.run(
                """
                MATCH (n)
                WHERE NOT (n)--()
                DELETE n
                RETURN count(n) AS nodes_removed
                """
            )
            node_record = await node_result.single()
            nodes_removed = node_record["nodes_removed"] if node_record else 0

        logger.info(
            "Neo4j: removed graph_id=%s — %d rel(s) updated/deleted, %d node(s) removed",
            graph_id,
            rels_affected,
            nodes_removed,
        )
        return {"rels_affected": rels_affected, "nodes_removed": nodes_removed}

    async def clear_all(self) -> None:
        """Delete every node and relationship from the Neo4j database."""
        async with self.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j: all nodes and relationships deleted")

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
            OPTIONAL MATCH (f:Task_Feature)-[:contribute_to]->(c2:Cause)
            WHERE f.name IN $task_features
            WITH symptom_causes, collect(DISTINCT c2) AS feature_causes

            // --- Environment clues ---
            OPTIONAL MATCH (env:Environment)-[:contribute_to]->(c3:Cause)
            WHERE env.name IN $environment_factors
            WITH symptom_causes, feature_causes,
                 collect(DISTINCT c3) AS env_causes

            // --- Component clues ---
            OPTIONAL MATCH (comp:Component)-[:originated_from]->(c4:Cause)
            WHERE comp.name IN $components
            WITH symptom_causes, feature_causes, env_causes,
                 collect(DISTINCT c4) AS comp_causes

            // --- Union all matched causes and score by clue-type breadth ---
            WITH symptom_causes + feature_causes + env_causes + comp_causes AS all_causes,
                 symptom_causes, feature_causes, env_causes, comp_causes
            UNWIND all_causes AS c
            WITH DISTINCT c,
                 (CASE WHEN c IN symptom_causes  THEN 1 ELSE 0 END +
                  CASE WHEN c IN feature_causes  THEN 1 ELSE 0 END +
                  CASE WHEN c IN env_causes      THEN 1 ELSE 0 END +
                  CASE WHEN c IN comp_causes     THEN 1 ELSE 0 END) AS match_score

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

    async def find_procedures_for_causes(
        self, cause_names: list[str]
    ) -> list[dict[str, Any]]:
        """Return Procedure nodes linked to the given causes via investigated_by edges.

        Each result includes the procedure's strategy_type, the accumulated
        parameter sets from all contributing incidents, and the edge frequency
        (how many incidents confirmed this procedure for this cause).

        Results are ordered by frequency descending so the most-evidenced
        procedures appear first.

        Args:
            cause_names: Canonical cause names to look up.

        Returns:
            List of dicts with keys ``cause_name``, ``procedure_name``,
            ``strategy_type``, ``parameters``, ``frequency``.
        """
        async with self.driver.session(
            database=self.settings.neo4j_database
        ) as session:
            result = await session.run(
                """
                MATCH (c:Cause)-[r:investigated_by]->(p:Procedure)
                WHERE c.name IN $cause_names
                RETURN c.name             AS cause_name,
                       p.name             AS procedure_name,
                       p.strategy_type    AS strategy_type,
                       p.description      AS description,
                       coalesce(r.parameters, []) AS parameters,
                       coalesce(r.frequency, 1)   AS frequency
                ORDER BY frequency DESC
                """,
                cause_names=cause_names,
            )
            return [dict(record) async for record in result]

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
