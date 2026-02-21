"""Application settings loaded from environment variables / ``.env`` file.

All configuration is expressed as a single :class:`Settings` Pydantic model.
The :func:`get_settings` factory is cached so the ``.env`` file is parsed
only once per process.

Environment variables are case-insensitive and can be set directly or via a
``.env`` file in the project root.  See ``README.md`` for the full list.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application configuration.

    Attributes:
        openai_api_key:           OpenAI API key (required when
                                  ``llm_provider="openai"``).
        anthropic_api_key:        Anthropic API key (required when
                                  ``llm_provider="anthropic"``).
        llm_provider:             LLM backend — ``"openai"`` or
                                  ``"anthropic"``.
        llm_model:                Model name passed to the LLM SDK.
        extraction_strategy:      Which :class:`ExtractionStrategy` to use.
        graph_database_backend:   Graph DB backend name (``"neo4j"``).
        vector_database_backend:  Vector DB backend name (``"qdrant"``).
        neo4j_uri:                Bolt URI for Neo4j.
        neo4j_username:           Neo4j username.
        neo4j_password:           Neo4j password.
        neo4j_database:           Neo4j database name.
        qdrant_url:               Qdrant HTTP URL.
        qdrant_api_key:           Qdrant API key (empty for local instances).
        qdrant_collection_name:   Qdrant collection used for all vectors.
        log_level:                Python logging level name (e.g. ``"INFO"``).
        embedding_model:          OpenAI embedding model name.
        embedding_dimension:      Dimension of the embedding vectors; must
                                  match ``embedding_model``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"

    # Extraction
    extraction_strategy: Literal[
        "llm", "rule_based", "jira", "github", "generic", "panda"
    ] = "llm"

    # Database backend selection
    graph_database_backend: Literal["neo4j"] = "neo4j"
    vector_database_backend: Literal["qdrant"] = "qdrant"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "graph_db"
    neo4j_password: str = "password"
    neo4j_database: str = "graph_db"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "bamboo_knowledge"

    # Application
    log_level: str = "INFO"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton.

    The ``.env`` file (if present) is read on the first call; subsequent calls
    return the cached instance without re-reading the file.
    """
    return Settings()
