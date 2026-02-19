"""Database connections and operations with pluggable backend support."""

from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.database.vector_database_client import VectorDatabaseClient
from bamboo.database.factory import (
    get_graph_backend,
    get_vector_backend,
    register_graph_backend,
    register_vector_backend,
    list_graph_backends,
    list_vector_backends,
)

__all__ = [
    "GraphDatabaseClient",
    "VectorDatabaseClient",
    "get_graph_backend",
    "get_vector_backend",
    "register_graph_backend",
    "register_vector_backend",
    "list_graph_backends",
    "list_vector_backends",
]
