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

from .llm_client import get_embeddings, get_extraction_llm, get_llm, get_summary_llm
from .prompts import (
    BROKERAGE_LOG_EXTRACTION_PROMPT,
    CAUSE_IDENTIFICATION_PROMPT,
    CAUSE_RESOLUTION_CANONICALIZE_PROMPT,
    DESCRIPTION_CANONICALIZE_PROMPT,
    EXPLORATION_GAP_ANALYSIS_PROMPT,
    EXPLORATION_PLAN_PROMPT,
    EXPLORER_TOOL_SELECTION_PROMPT,
    JOB_DIAG_NORMALIZE_PROMPT,
    EMAIL_EXTRACTION_PROMPT,
    EMAIL_GENERATION_PROMPT,
    KNOWLEDGE_REVIEW_PROMPT,
    TASK_ERROR_CATEGORY_LABEL_PROMPT,
    EXTRACTION_PROMPT,
    LOG_EXTRACTION_PROMPT,
    SUMMARIZATION_PROMPT,
)

__all__ = [
    "get_llm",
    "get_extraction_llm",
    "get_summary_llm",
    "get_embeddings",
    "BROKERAGE_LOG_EXTRACTION_PROMPT",
    "DESCRIPTION_CANONICALIZE_PROMPT",
    "EXTRACTION_PROMPT",
    "EMAIL_EXTRACTION_PROMPT",
    "LOG_EXTRACTION_PROMPT",
    "CAUSE_RESOLUTION_CANONICALIZE_PROMPT",
    "TASK_ERROR_CATEGORY_LABEL_PROMPT",
    "SUMMARIZATION_PROMPT",
    "CAUSE_IDENTIFICATION_PROMPT",
    "EMAIL_GENERATION_PROMPT",
    "JOB_DIAG_NORMALIZE_PROMPT",
    "KNOWLEDGE_REVIEW_PROMPT",
    "EXPLORATION_GAP_ANALYSIS_PROMPT",
    "EXPLORATION_PLAN_PROMPT",
    "EXPLORER_TOOL_SELECTION_PROMPT",
]
