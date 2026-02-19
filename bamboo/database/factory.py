"""Database factory for plugin-based backend selection."""

import logging
from typing import Type

from bamboo.config import get_settings
from bamboo.database.base import GraphDatabaseBackend, VectorDatabaseBackend

logger = logging.getLogger(__name__)

# Registry of available backends
_graph_backends: dict[str, Type[GraphDatabaseBackend]] = {}
_vector_backends: dict[str, Type[VectorDatabaseBackend]] = {}


def register_graph_backend(name: str, backend_class: Type[GraphDatabaseBackend]):
    """Register a graph database backend."""
    _graph_backends[name.lower()] = backend_class
    logger.debug(f"Registered graph backend: {name}")


def register_vector_backend(name: str, backend_class: Type[VectorDatabaseBackend]):
    """Register a vector database backend."""
    _vector_backends[name.lower()] = backend_class
    logger.debug(f"Registered vector backend: {name}")


def get_graph_backend() -> GraphDatabaseBackend:
    """Get configured graph database backend instance."""
    settings = get_settings()
    backend_name = settings.graph_database_backend.lower()

    if backend_name not in _graph_backends:
        raise ValueError(
            f"Graph database backend '{backend_name}' not found. "
            f"Available backends: {list(_graph_backends.keys())}"
        )

    backend_class = _graph_backends[backend_name]
    logger.info(f"Loading graph database backend: {backend_name}")
    return backend_class()


def get_vector_backend() -> VectorDatabaseBackend:
    """Get configured vector database backend instance."""
    settings = get_settings()
    backend_name = settings.vector_database_backend.lower()

    if backend_name not in _vector_backends:
        raise ValueError(
            f"Vector database backend '{backend_name}' not found. "
            f"Available backends: {list(_vector_backends.keys())}"
        )

    backend_class = _vector_backends[backend_name]
    logger.info(f"Loading vector database backend: {backend_name}")
    return backend_class()


def list_graph_backends() -> list[str]:
    """List all registered graph database backends."""
    return list(_graph_backends.keys())


def list_vector_backends() -> list[str]:
    """List all registered vector database backends."""
    return list(_vector_backends.keys())


# Register built-in backends
def _register_builtin_backends():
    """Register built-in backend implementations."""
    # Register Neo4j backend
    try:
        from bamboo.database.backends.neo4j_backend import Neo4jBackend

        register_graph_backend("neo4j", Neo4jBackend)
    except ImportError as e:
        logger.debug(f"Neo4j backend not available: {e}")

    # Register Qdrant backend
    try:
        from bamboo.database.backends.qdrant_backend import QdrantBackend

        register_vector_backend("qdrant", QdrantBackend)
    except ImportError as e:
        logger.debug(f"Qdrant backend not available: {e}")


# Auto-register built-in backends on import
_register_builtin_backends()
