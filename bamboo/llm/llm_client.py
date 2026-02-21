"""LLM client initialisation and provider abstraction.

Provides two cached factory functions used throughout the application:

- :func:`get_llm`        – returns the configured chat model.
- :func:`get_embeddings` – returns the configured embeddings model.

Both functions are cached with :func:`functools.lru_cache` so the underlying
SDK client is constructed only once per process.  Supported LLM providers are
``"openai"`` (default) and ``"anthropic"``.  Embeddings always use OpenAI.
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from bamboo.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    """Return the configured LangChain chat model (cached).

    The provider and model name are read from :class:`~bamboo.config.Settings`.
    Supported providers: ``"openai"``, ``"anthropic"``.

    Returns:
        A :class:`langchain_core.language_models.BaseChatModel` instance.

    Raises:
        ValueError: If ``llm_provider`` is not ``"openai"`` or ``"anthropic"``.
    """
    settings = get_settings()

    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.7,
        )
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.7,
        )
    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}. "
        "Supported values: 'openai', 'anthropic'."
    )


@lru_cache
def get_embeddings() -> Embeddings:
    """Return the configured LangChain embeddings model (cached).

    Currently only OpenAI embeddings are supported.  The model name and API
    key are read from :class:`~bamboo.config.Settings`.

    Returns:
        A :class:`langchain_core.embeddings.Embeddings` instance.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
