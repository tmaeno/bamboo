"""LLM client initialisation and provider abstraction.

Provides two cached factory functions used throughout the application:

- :func:`get_llm`        – returns the configured chat model.
- :func:`get_embeddings` – returns the configured embeddings model.

Both functions are cached with :func:`functools.lru_cache` so the underlying
SDK client is constructed only once per process.

**LLM providers** (``LLM_PROVIDER``):
  - ``"openai"`` (default) — OpenAI chat models (e.g. ``gpt-4o``).
  - ``"anthropic"``        — Anthropic Claude models.

**Embeddings providers** (``EMBEDDINGS_PROVIDER``):
  - ``"openai"`` (default) — ``OpenAIEmbeddings``; requires an API key.
    Anthropic does not provide an embeddings API, so this is the paid
    cloud option regardless of ``LLM_PROVIDER``.
  - ``"local"``            — ``HuggingFaceEmbeddings`` backed by
    ``sentence-transformers`` running in-process.  Completely free, no API
    key required.  Set ``EMBEDDING_MODEL`` to a sentence-transformers model
    name, e.g. ``all-MiniLM-L6-v2`` (fast, 384-dim) or
    ``all-mpnet-base-v2`` (more accurate, 768-dim).  Requires
    ``pip install sentence-transformers langchain-huggingface``.
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
            api_key=settings.llm_api_key,
            temperature=0.7,
        )
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=0.7,
        )
    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}. "
        "Supported values: 'openai', 'anthropic'."
    )


@lru_cache
def get_embeddings() -> Embeddings:
    """Return the configured LangChain embeddings model (cached).

    The backend is selected by ``EMBEDDINGS_PROVIDER``:

    - ``"openai"`` — :class:`langchain_openai.OpenAIEmbeddings`.
      Uses :attr:`~bamboo.config.Settings.effective_embeddings_api_key`
      (falls back to ``LLM_API_KEY`` when ``EMBEDDINGS_API_KEY`` is unset).
    - ``"local"`` — :class:`langchain_huggingface.HuggingFaceEmbeddings`
      backed by ``sentence-transformers`` running in-process.
      ``EMBEDDING_MODEL`` must be a sentence-transformers model name, e.g.
      ``all-MiniLM-L6-v2`` (fast, 384-dim) or ``all-mpnet-base-v2``
      (more accurate, 768-dim).  No API key required.

    Returns:
        A :class:`langchain_core.embeddings.Embeddings` instance.

    Raises:
        ValueError: If ``embeddings_provider`` is unrecognised.
        ImportError: If ``embeddings_provider="local"`` but
                     ``sentence-transformers`` / ``langchain-huggingface``
                     are not installed.
    """
    settings = get_settings()

    if settings.embeddings_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.effective_embeddings_api_key,
        )

    if settings.embeddings_provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "EMBEDDINGS_PROVIDER=local requires 'sentence-transformers' and "
                "'langchain-huggingface'.\n"
                "Install them with:  "
                "pip install sentence-transformers langchain-huggingface"
            ) from exc
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)

    raise ValueError(
        f"Unsupported embeddings provider: {settings.embeddings_provider!r}. "
        "Supported values: 'openai', 'local'."
    )
