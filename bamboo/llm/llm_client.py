"""LLM client initialization and utilities."""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from bamboo.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    """Get configured LLM instance."""
    settings = get_settings()

    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.7,
        )
    elif settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


@lru_cache
def get_embeddings() -> Embeddings:
    """Get configured embeddings instance."""
    settings = get_settings()

    # Currently only OpenAI embeddings are supported
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
