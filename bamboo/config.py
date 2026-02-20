"""Configuration management for Bamboo."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"

    # Extraction Configuration
    extraction_strategy: Literal[
        "llm", "rule_based", "jira", "github", "generic", "panda"
    ] = "llm"

    # Database Backend Configuration
    graph_database_backend: Literal["neo4j"] = "neo4j"
    vector_database_backend: Literal["qdrant"] = "qdrant"

    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "graph_db"
    neo4j_password: str = "password"
    neo4j_database: str = "graph_db"

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "bamboo_knowledge"

    # Application Settings
    log_level: str = "INFO"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
