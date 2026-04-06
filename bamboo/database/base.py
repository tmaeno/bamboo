"""Abstract base classes for pluggable database backends.

The database layer is split into two independent abstractions:

- :class:`GraphDatabaseBackend` – stores the structured knowledge graph
  (nodes, relationships) and supports graph traversal queries.  The default
  implementation is :class:`~bamboo.database.backends.neo4j_backend.Neo4jBackend`.

- :class:`VectorDatabaseBackend` – stores embedded text vectors for semantic
  similarity search.  The default implementation is
  :class:`~bamboo.database.backends.qdrant_backend.QdrantBackend`.

New backends are registered via
:func:`~bamboo.database.factory.register_graph_backend` /
:func:`~bamboo.database.factory.register_vector_backend` and selected through
the ``GRAPH_DATABASE_BACKEND`` / ``VECTOR_DATABASE_BACKEND`` configuration keys.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from bamboo.models.graph_element import BaseNode, GraphRelationship


class GraphDatabaseBackend(ABC):
    """Interface for graph database implementations.

    Implementations must be safe to use as async context managers: call
    :meth:`connect` before any other method and :meth:`close` when done.
    """

    @abstractmethod
    async def connect(self):
        """Open the database connection and create any required indexes."""
        pass

    @abstractmethod
    async def close(self):
        """Close the database connection and release resources."""
        pass

    @abstractmethod
    async def create_node(self, node: BaseNode) -> str:
        """Unconditionally create a new node and return its assigned ID.

        .. note::
            Prefer :meth:`get_or_create_canonical_node` to avoid duplicates.
        """
        pass

    @abstractmethod
    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Return the ID of an existing node with *canonical_name*, or create it.

        Merges on the ``name`` property: if a node of the same type and name
        already exists its ID is returned without modification; otherwise the
        node is created with ``node.name = canonical_name`` and its new ID is
        returned.

        Args:
            node:           Node object whose properties are used when creating.
            canonical_name: The stable, deduplicated name to merge on.

        Returns:
            The node's ID string.
        """
        pass

    @abstractmethod
    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a directed relationship between two nodes.

        Args:
            relationship: Edge descriptor with ``source_id``, ``target_id``,
                          ``relation_type``, and optional properties.

        Returns:
            ``True`` if the relationship was created, ``False`` otherwise.
        """
        pass

    @abstractmethod
    async def find_causes(
        self,
        symptoms: list[str] = None,
        task_features: list[str] = None,
        job_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find candidate causes ranked by evidence breadth.

        Each clue type (symptom, task feature, job feature, environment,
        component) that links to a cause contributes +1 to its
        ``match_score``, so causes corroborated by multiple clue types rank
        above those matched by only one.

        Args:
            symptoms:             Symptom node names.
            task_features:        Task-feature node names.
            job_features:         Job-feature node names (aggregated execution
                                  patterns, e.g. ``"site_failure_rate=AGLT2:high"``).
            environment_factors:  Environment node names.
            components:           Component node names.
            limit:                Maximum number of causes to return.

        Returns:
            List of cause dicts ordered by ``match_score DESC``,
            ``frequency DESC``, ``confidence DESC``.  Each dict includes
            ``cause_id``, ``cause_name``, ``cause_description``,
            ``confidence``, ``frequency``, ``match_score``, and
            ``resolutions``.
        """
        pass

    @abstractmethod
    async def find_common_pattern(
        self,
        graph_ids: list[str],
        min_occurrences: int = 2,
    ) -> list[dict[str, Any]]:
        """Return edges shared across at least *min_occurrences* of the given graphs.

        Args:
            graph_ids:       List of graph IDs to intersect.
            min_occurrences: Minimum number of graphs that must share an edge.

        Returns:
            List of edge dicts ordered by ``occurrence_count DESC``.
        """
        pass

    @abstractmethod
    async def remove_graph_id(self, graph_id: str) -> dict[str, int]:
        """Remove a graph_id's contribution from all relationships and clean up.

        For each relationship whose ``graph_ids`` list contains *graph_id*:
        - Remove *graph_id* from the list and recompute ``frequency``.
        - Delete the relationship if ``graph_ids`` becomes empty.
        Then delete any nodes that have become fully isolated as a result.

        Args:
            graph_id: The graph ID to remove (derived from task_id + status).

        Returns:
            Dict with keys ``rels_affected`` and ``nodes_removed``.
        """
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """Delete every node and relationship in the database.

        .. warning::
            Irreversible.  Intended for development / testing resets only.
        """
        pass

    @abstractmethod
    async def increment_cause_frequency(self, cause_id: str):
        """Increment the ``frequency`` counter on a cause node by 1.

        Args:
            cause_id: The cause node's ID.
        """
        pass

    @abstractmethod
    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update the running success-rate statistic on a resolution node.

        Uses an incremental formula::

            success_rate = successful_attempts / total_attempts

        Args:
            resolution_id: The resolution node's ID.
            success:        ``True`` if this application was successful.
        """
        pass


class VectorDatabaseBackend(ABC):
    """Interface for vector database implementations."""

    @abstractmethod
    async def connect(self):
        """Open the database connection and ensure the collection exists."""
        pass

    @abstractmethod
    async def close(self):
        """Close the database connection."""
        pass

    @abstractmethod
    async def upsert_section_vector(
        self,
        vector_id: str,
        embedding: list[float],
        content: str,
        section: str,
        metadata: dict[str, Any],
    ) -> str:
        """Insert or update a vector point.

        Points are partitioned by *section* in the payload so that
        :meth:`search_similar` can filter to a single section efficiently.

        Args:
            vector_id: Stable, deterministic ID for the point.  Re-inserting
                       with the same ID updates the existing point (upsert).
            embedding: The embedding vector.
            content:   The text that was embedded; stored in the payload.
            section:   Logical partition name (e.g. ``"Task_Context"``,
                       ``"canonical_node::Cause"``, ``"Summary"``).
            metadata:  Arbitrary key-value payload stored alongside the vector.

        Returns:
            The ``vector_id`` that was upserted.
        """
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return the *limit* most similar points above *score_threshold*.

        Args:
            query_embedding:   The query vector.
            limit:             Maximum number of results.
            score_threshold:   Minimum cosine similarity; points below this
                               are excluded.
            filter_conditions: Optional equality filters applied to payload
                               fields before scoring (e.g.
                               ``{"section": "Summary"}``).

        Returns:
            List of result dicts with keys ``id``, ``score``, ``content``,
            ``entry``, and ``metadata``.
        """
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Return ``True`` if the backing collection/index exists and is ready.

        Used as a pre-flight check before the first vector operation so that
        callers can emit a clear, actionable error rather than catching an
        opaque low-level exception from the database driver.
        """
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """Drop and recreate the entire collection (all vectors deleted).

        .. warning::
            Irreversible.  Intended for development / testing resets only.
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete the point with the given ID.

        Returns:
            ``True`` if deleted, ``False`` if the point did not exist.
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single point by ID.

        Returns:
            The point dict, or ``None`` if not found.
        """
        pass

    @abstractmethod
    async def get_summaries_by_graph_ids(
        self, graph_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch ``Summary`` section entries for the given graph IDs.

        Used as the second step of the two-step vector retrieval pattern:

        1. :meth:`search_similar` finds matching node-description points and
           returns their ``graph_id`` metadata values.
        2. This method fetches the corresponding ``Summary`` entries so the
           caller gets the full narrative context of each matched past case.

        Args:
            graph_ids: List of graph IDs to fetch summaries for.

        Returns:
            List of summary dicts (same structure as :meth:`search_similar`
            results).
        """
        pass
