"""Database backend factory and registry.

Backends are registered by name at import time via
:func:`register_graph_backend` / :func:`register_vector_backend`.
The active backend is selected by reading the
``GRAPH_DATABASE_BACKEND`` / ``VECTOR_DATABASE_BACKEND`` configuration keys.

Built-in backends registered automatically:

===========  =============================================
Name         Class
===========  =============================================
``neo4j``    :class:`~bamboo.database.backends.neo4j_backend.Neo4jBackend`
``qdrant``   :class:`~bamboo.database.backends.qdrant_backend.QdrantBackend`
===========  =============================================
"""

import logging
from typing import Type

from bamboo.config import get_settings
from bamboo.database.base import GraphDatabaseBackend, VectorDatabaseBackend

logger = logging.getLogger(__name__)

# Registries: lower-cased name → backend class
_graph_backends: dict[str, Type[GraphDatabaseBackend]] = {}
_vector_backends: dict[str, Type[VectorDatabaseBackend]] = {}


def register_graph_backend(name: str, backend_class: Type[GraphDatabaseBackend]):
    """Register a graph database backend under *name*.

    Args:
        name:          Lower-cased registry key (e.g. ``"neo4j"``).
        backend_class: Concrete :class:`GraphDatabaseBackend` subclass.
    """
    _graph_backends[name.lower()] = backend_class
    logger.debug("Registered graph backend: %s", name)


def register_vector_backend(name: str, backend_class: Type[VectorDatabaseBackend]):
    """Register a vector database backend under *name*.

    Args:
        name:          Lower-cased registry key (e.g. ``"qdrant"``).
        backend_class: Concrete :class:`VectorDatabaseBackend` subclass.
    """
    _vector_backends[name.lower()] = backend_class
    logger.debug("Registered vector backend: %s", name)


def get_graph_backend() -> GraphDatabaseBackend:
    """Return an unconnected :class:`GraphDatabaseBackend` instance.

    The backend name is read from the ``GRAPH_DATABASE_BACKEND`` config key.
    Call :meth:`~GraphDatabaseBackend.connect` on the returned instance before
    using it.

    Returns:
        A freshly instantiated :class:`GraphDatabaseBackend`.

    Raises:
        ValueError: If the configured backend name is not registered.
    """
    settings = get_settings()
    backend_name = settings.graph_database_backend.lower()
    if backend_name not in _graph_backends:
        raise ValueError(
            f"Graph database backend {backend_name!r} not registered. "
            f"Available: {sorted(_graph_backends.keys())}"
        )
    logger.info("Loading graph database backend: %s", backend_name)
    return _graph_backends[backend_name]()


def get_vector_backend() -> VectorDatabaseBackend:
    """Return an unconnected :class:`VectorDatabaseBackend` instance.

    The backend name is read from the ``VECTOR_DATABASE_BACKEND`` config key.
    Call :meth:`~VectorDatabaseBackend.connect` on the returned instance before
    using it.

    Returns:
        A freshly instantiated :class:`VectorDatabaseBackend`.

    Raises:
        ValueError: If the configured backend name is not registered.
    """
    settings = get_settings()
    backend_name = settings.vector_database_backend.lower()
    if backend_name not in _vector_backends:
        raise ValueError(
            f"Vector database backend {backend_name!r} not registered. "
            f"Available: {sorted(_vector_backends.keys())}"
        )
    logger.info("Loading vector database backend: %s", backend_name)
    return _vector_backends[backend_name]()


def list_graph_backends() -> list[str]:
    """Return the names of all registered graph database backends."""
    return sorted(_graph_backends.keys())


def list_vector_backends() -> list[str]:
    """Return the names of all registered vector database backends."""
    return sorted(_vector_backends.keys())


# ---------------------------------------------------------------------------
# Auto-registration of built-in backends
# ---------------------------------------------------------------------------

def _register_builtin_backends():
    """Register built-in backend implementations.  Called once at import time."""
    try:
        from bamboo.database.backends.neo4j_backend import Neo4jBackend
        register_graph_backend("neo4j", Neo4jBackend)
    except ImportError as exc:
        logger.debug("Neo4j backend not available: %s", exc)

    try:
        from bamboo.database.backends.qdrant_backend import QdrantBackend
        register_vector_backend("qdrant", QdrantBackend)
    except ImportError as exc:
        logger.debug("Qdrant backend not available: %s", exc)


_register_builtin_backends()
