"""Application settings loaded from environment variables / ``.env`` file.

All configuration is expressed as a single :class:`Settings` Pydantic model.
The :func:`get_settings` factory is cached so the ``.env`` file is parsed
only once per process.

Environment variables are case-insensitive and can be set directly or via a
``.env`` file in the project root.  See ``README.md`` for the full list.
"""

import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application configuration.

    Attributes:
        llm_api_key:              API key for the configured LLM provider
                                  (OpenAI or Anthropic).
        embeddings_api_key:       API key for the embeddings provider.
                                  Not used when ``embeddings_provider="local"``.
                                  Leave empty when ``llm_provider="openai"``
                                  and ``embeddings_provider="openai"`` —
                                  ``llm_api_key`` is reused automatically via
                                  :attr:`effective_embeddings_api_key`.
        embeddings_provider:      Embeddings backend: ``"openai"`` (default,
                                  requires API key) or ``"local"`` (free,
                                  runs ``sentence-transformers`` in-process,
                                  no API key needed).  Anthropic does not
                                  provide an embeddings API.
        llm_provider:             LLM backend — ``"openai"``, ``"anthropic"``,
                                  or ``"ollama"`` (free, runs locally via
                                  Ollama; no API key needed).
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

    @field_validator(
        "llm_api_key", "embeddings_api_key", "llm_provider", "embeddings_provider",
        "llm_model", "extraction_strategy", "graph_database_backend",
        "vector_database_backend", "neo4j_uri", "neo4j_username", "neo4j_password",
        "neo4j_database", "qdrant_url", "qdrant_api_key", "qdrant_collection_name",
        "log_level", "embedding_model",
        mode="before",
    )
    @classmethod
    def _strip_inline_comments(cls, value: Any) -> Any:
        """Strip inline ``# …`` comments from string values read from ``.env``.

        ``python-dotenv`` does not remove inline comments, so a line such as::

            QDRANT_API_KEY=  # Optional, leave empty for local instance

        would otherwise set the field to ``"# Optional, leave empty for local instance"``.
        """
        if isinstance(value, str):
            return re.sub(r"\s*#.*$", "", value).strip()
        return value

    # LLM
    llm_api_key: str = ""  # not required when llm_provider="ollama"
    embeddings_api_key: str = ""  # falls back to llm_api_key when empty
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    embeddings_provider: Literal["openai", "local"] = "openai"
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

    @property
    def effective_embeddings_api_key(self) -> str:
        """Return the API key to use for embeddings.

        Falls back to ``llm_api_key`` when ``embeddings_api_key`` is not set,
        which is the common case when ``llm_provider="openai"`` and both the
        LLM and embeddings share the same key.  Not used when
        ``embeddings_provider="local"``.
        """
        return self.embeddings_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton.

    The ``.env`` file (if present) is read on the first call; subsequent calls
    return the cached instance without re-reading the file.
    """
    return Settings()
