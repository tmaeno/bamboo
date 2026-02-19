"""Abstract base classes for database backends."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from bamboo.models.graph_element import BaseNode, GraphRelationship


class GraphDatabaseBackend(ABC):
    """Abstract base class for graph database implementations."""

    @abstractmethod
    async def connect(self):
        """Establish connection to graph database."""
        pass

    @abstractmethod
    async def close(self):
        """Close connection."""
        pass

    @abstractmethod
    async def create_node(self, node: BaseNode) -> str:
        """Create a node in the graph."""
        pass

    @abstractmethod
    async def get_or_create_canonical_node(
        self, node: BaseNode, canonical_name: str
    ) -> str:
        """Get existing node by canonical name or create new one."""
        pass

    @abstractmethod
    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between nodes."""
        pass

    @abstractmethod
    async def find_causes(
        self,
        errors: list[str] = None,
        task_features: list[str] = None,
        environment_factors: list[str] = None,
        components: list[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find possible causes ranked by how many clue types they match.

        Causes that match across multiple clue types (e.g. both an error and
        a component) score higher than causes matching only one type, producing
        more accurate ranking than merging separate single-type query results.

        Args:
            errors: Error node names extracted from the knowledge graph.
            task_features: Task-feature node names extracted from the knowledge graph.
            environment_factors: Environment node names extracted from the knowledge graph.
            components: Component node names extracted from the knowledge graph.
            limit: Maximum number of causes to return.

        Returns:
            List of cause dicts ordered by match_score DESC, frequency DESC,
            confidence DESC.  Each dict includes a ``match_score`` field
            indicating how many distinct clue types contributed to the match.
        """
        pass

    @abstractmethod
    async def increment_cause_frequency(self, cause_id: str):
        """Increment the frequency counter for a cause."""
        pass

    @abstractmethod
    async def update_resolution_success_rate(self, resolution_id: str, success: bool):
        """Update resolution success rate based on feedback."""
        pass


class VectorDatabaseBackend(ABC):
    """Abstract base class for vector database implementations."""

    @abstractmethod
    async def connect(self):
        """Establish connection to vector database."""
        pass

    @abstractmethod
    async def close(self):
        """Close connection."""
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
        """Insert or update a document in the vector database."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a specific document by ID."""
        pass

    @abstractmethod
    async def get_summaries_by_graph_ids(
        self, graph_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Retrieve Summary entries for the given graph IDs.

        Used in the two-step vector retrieval pattern:
          1. Search unstructured node descriptions to find matching graph_ids.
          2. Call this method to fetch the full summaries for those graphs.
        """
        pass
