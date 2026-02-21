"""LLM sub-package: clients, embeddings, and prompt templates.

Public API
----------
``get_llm()``
    Return the cached :class:`~langchain_core.language_models.BaseChatModel`.

``get_embeddings()``
    Return the cached :class:`~langchain_core.embeddings.Embeddings`.

All prompt template constants are also re-exported here for convenient
single-import access across the rest of the package.
"""

from .llm_client import get_embeddings, get_llm
from .prompts import (
    CAUSE_IDENTIFICATION_PROMPT,
    CAUSE_RESOLUTION_CANONICALIZE_PROMPT,
    EMAIL_EXTRACTION_PROMPT,
    EMAIL_GENERATION_PROMPT,
    EXTRACTION_PROMPT,
    SUMMARIZATION_PROMPT,
)

__all__ = [
    "get_llm",
    "get_embeddings",
    "EXTRACTION_PROMPT",
    "EMAIL_EXTRACTION_PROMPT",
    "CAUSE_RESOLUTION_CANONICALIZE_PROMPT",
    "ERROR_CATEGORY_LABEL_PROMPT",
    "SUMMARIZATION_PROMPT",
    "CAUSE_IDENTIFICATION_PROMPT",
    "EMAIL_GENERATION_PROMPT",
]
