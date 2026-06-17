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

import contextlib
import io
import logging
from functools import lru_cache
from typing import Any, Callable, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from bamboo.config import get_settings

logger = logging.getLogger(__name__)

# Cloud windows are large enough that the tool budget rarely binds, and the provider
# APIs don't reliably expose the value, so a built-in constant is used. Ollama's
# *served* window is auto-detected at runtime instead.
_CLOUD_CONTEXT_WINDOW = {"openai": 128_000, "anthropic": 200_000}
# Conservative fallback when an Ollama probe fails or the model isn't loaded yet.
_OLLAMA_FALLBACK_CONTEXT = 8192
# Detected Ollama context windows, keyed by (base_url, model). Only successful
# detections are cached (a not-yet-loaded model self-corrects on the next turn).
_CTX_CACHE: dict[tuple[str, str], int] = {}


def _ollama_model_matches(entry_name: str, model: str) -> bool:
    """Match an ``/api/ps`` entry name to ``llm_model``, tolerating an implicit
    ``:latest`` tag on either side (e.g. ``qwen3.6`` vs ``qwen3.6:latest``)."""
    if not entry_name:
        return False
    return entry_name == model or entry_name.split(":")[0] == model.split(":")[0]


def _detect_ollama_context_window(settings: Any) -> int:
    base_url = (
        getattr(settings, "ollama_base_url", "") or "http://localhost:11434"
    ).rstrip("/")
    model = getattr(settings, "llm_model", "") or ""
    key = (base_url, model)
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(f"{base_url}/api/ps", timeout=2.0)
        resp.raise_for_status()
        for entry in (resp.json() or {}).get("models", []):
            if _ollama_model_matches(entry.get("model") or entry.get("name") or "", model):
                ctx = int(entry.get("context_length") or 0)
                if ctx > 0:
                    _CTX_CACHE[key] = ctx
                    logger.debug(
                        "resolve_context_window: ollama served context_length=%d for %r",
                        ctx, model,
                    )
                    return ctx
        logger.debug(
            "resolve_context_window: %r not loaded in /api/ps; fallback %d",
            model, _OLLAMA_FALLBACK_CONTEXT,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to fallback, never break a turn
        logger.debug(
            "resolve_context_window: /api/ps probe failed (%s); fallback %d",
            exc, _OLLAMA_FALLBACK_CONTEXT,
        )
    return _OLLAMA_FALLBACK_CONTEXT


def resolve_context_window(settings: Optional[Any] = None) -> int:
    """Return the model's usable context window in tokens.

    ``settings.llm_context_window > 0`` is an explicit override. Otherwise it is
    auto-detected per provider:

    * ``ollama`` — the *served* window from ``GET {ollama_base_url}/api/ps`` for the
      loaded ``llm_model`` (cached). If the model isn't loaded yet or the server is
      unreachable, a conservative fallback is returned and the next call self-corrects.
    * ``openai`` / ``anthropic`` — a built-in constant (their windows are large enough
      that the tool budget rarely binds, and the APIs don't expose the value reliably).

    Model-scoped on purpose: the tool-selection budget and (later) a generic
    truncation guard consume the same value, so it lives here, not in tool selection.
    """
    settings = settings or get_settings()
    override = int(getattr(settings, "llm_context_window", 0) or 0)
    if override > 0:
        return override
    provider = getattr(settings, "llm_provider", "openai")
    if provider == "ollama":
        return _detect_ollama_context_window(settings)
    return _CLOUD_CONTEXT_WINDOW.get(provider, _OLLAMA_FALLBACK_CONTEXT)


def _build_llm(temperature: float) -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=temperature,
        )
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=temperature,
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
            temperature=temperature,
            reasoning=settings.ollama_reasoning,
        )
    raise ValueError(
        f"Unsupported LLM provider: {settings.llm_provider!r}. "
        "Supported values: 'openai', 'anthropic', 'ollama'."
    )


@lru_cache
def get_llm() -> BaseChatModel:
    """Return the chat model for generative tasks (summarisation, email).

    Uses temperature=0.7 for natural-sounding output.
    """
    return _build_llm(temperature=0.7)


@lru_cache
def get_summary_llm() -> BaseChatModel:
    """Return the chat model for summarisation tasks.

    Uses ``LLM_SUMMARY_TEMPERATURE`` (default 0.3) — low enough to produce
    consistent summaries across repeated runs while still allowing natural prose.
    """
    return _build_llm(temperature=get_settings().llm_summary_temperature)


@lru_cache
def get_extraction_llm() -> BaseChatModel:
    """Return the chat model for structured extraction tasks.

    Uses temperature=0.0 for deterministic, reproducible JSON output.
    Extraction at temperature > 0 causes the same input to produce different
    node counts on repeated runs.
    """
    return _build_llm(temperature=0.0)


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
                "Install them with:  pip install 'bamboo[local]'"
            ) from exc
        try:
            _sink = io.StringIO()
            with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
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


@lru_cache
def get_reranker():
    """Return the configured cross-encoder reranker, or None if RERANKER_MODEL is unset.

    Set ``RERANKER_MODEL`` to a sentence-transformers cross-encoder model name, e.g.
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``, to enable post-retrieval reranking.
    Requires ``pip install sentence-transformers``.
    """
    settings = get_settings()
    if not settings.reranker_model:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "RERANKER_MODEL requires 'sentence-transformers'.\n"
            "Install it with:  pip install sentence-transformers"
        ) from exc
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
        return CrossEncoder(settings.reranker_model)


@lru_cache
def get_token_counter() -> Callable[[str], int]:
    """Return a ``text -> token count`` function for the configured chat model.

    Used to bound the orchestration prompt's tool block against the model's
    context window (see :mod:`bamboo.agents.helpers.tool_selection`). It tries the
    model's real tokenizer and falls back to a deliberately *over*-counting char
    heuristic, so the budget never under-counts and overflows a small local
    context.

    - ``openai``     → ``tiktoken`` for the model (or ``cl100k_base``).
    - other providers → a Hugging Face tokenizer for ``LLM_MODEL`` when it
      resolves to a loadable repo; short Ollama names that aren't repos fall
      through to the heuristic.
    - fallback       → ``ceil(len(text) / 3)`` (conservative over-count).
    """
    settings = get_settings()

    if settings.llm_provider == "openai":
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(settings.llm_model)
            except Exception:  # noqa: BLE001 — unknown model name
                enc = tiktoken.get_encoding("cl100k_base")
            return lambda s: len(enc.encode(s))
        except Exception:  # noqa: BLE001 — tiktoken unavailable
            pass

    try:
        from transformers import AutoTokenizer

        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            tok = AutoTokenizer.from_pretrained(settings.llm_model)
        return lambda s: len(tok.encode(s))
    except Exception:  # noqa: BLE001 — not a loadable repo / transformers absent
        pass

    return lambda s: (len(s) + 2) // 3
