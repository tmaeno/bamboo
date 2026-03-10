"""LLM client initialisation and provider abstraction.

Provides two cached factory functions used throughout the application:

- :func:`get_llm`        – returns the configured chat model.
- :func:`get_embeddings` – returns the configured embeddings model.

Both functions are cached with :func:`functools.lru_cache` so the underlying
SDK client is constructed only once per process.

**LLM providers** (``LLM_PROVIDER``):
  - ``"openai"`` (default) — OpenAI chat models (e.g. ``gpt-4o``).
    Requires ``LLM_API_KEY``.
  - ``"anthropic"``        — Anthropic Claude models via the Anthropic API.
    Requires a *Claude API* subscription and ``LLM_API_KEY``.
    Note: a claude.ai consumer subscription does **not** grant API access.
  - ``"ollama"``           — Local models served by Ollama (e.g. ``llama3``,
    ``mistral``, ``gemma3``).  Completely free, no API key required.
    Requires Ollama to be installed and running (``ollama serve``).
    Set ``LLM_MODEL`` to the pulled model name (e.g. ``llama3.2``).
    Requires ``pip install langchain-ollama``.

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

**Fully free setup** (no API keys at all)::

    LLM_PROVIDER=ollama
    LLM_MODEL=llama3.2
    EMBEDDINGS_PROVIDER=local
    EMBEDDING_MODEL=all-MiniLM-L6-v2
    EMBEDDING_DIMENSION=384

    pip install langchain-ollama sentence-transformers langchain-huggingface
    # or: pip install "bamboo[local]"
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
    Supported providers: ``"openai"``, ``"anthropic"``, ``"ollama"``.

    Returns:
        A :class:`langchain_core.language_models.BaseChatModel` instance.

    Raises:
        ValueError:  If ``llm_provider`` is not a supported value.
        ImportError: If ``llm_provider="ollama"`` but ``langchain-ollama``
                     is not installed.
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
    if settings.llm_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "LLM_PROVIDER=ollama requires 'langchain-ollama'.\n"
                "Install it with:  pip install langchain-ollama\n"
                "Also make sure Ollama is installed and running: https://ollama.com"
            ) from exc
        return ChatOllama(
            model=settings.llm_model,
            temperature=0.7,
        )
    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}. "
        "Supported values: 'openai', 'anthropic', 'ollama'."
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
        ValueError:  If ``embeddings_provider`` is unrecognised.
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
                "Install them with:  pip install 'bamboo[local-embeddings]'"
            ) from exc
        try:
            return HuggingFaceEmbeddings(model_name=settings.embedding_model)
        except (NameError, AttributeError) as exc:
            # transformers>=4.45 breaks with torch<2.4:
            #   NameError: name 'nn' is not defined
            raise RuntimeError(
                f"Failed to initialise local embeddings ({exc}).\n\n"
                "This is a torch / transformers version conflict.\n"
                "Fix by pinning transformers:\n\n"
                "  pip install 'transformers<4.45'\n\n"
                "This pins transformers<4.45, which is compatible with torch 2.2.x.\n\n"
            ) from exc

    raise ValueError(
        f"Unsupported embeddings provider: {settings.embeddings_provider!r}. "
        "Supported values: 'openai', 'local'."
    )
