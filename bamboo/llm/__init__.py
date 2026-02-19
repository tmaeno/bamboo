"""LLM integrations and utilities."""

from .llm_client import get_embeddings, get_llm
from .prompts import (
    CAUSE_IDENTIFICATION_PROMPT,
    EMAIL_GENERATION_PROMPT,
    EXTRACTION_PROMPT,
    SUMMARIZATION_PROMPT,
)

__all__ = [
    "get_llm",
    "get_embeddings",
    "EXTRACTION_PROMPT",
    "SUMMARIZATION_PROMPT",
    "CAUSE_IDENTIFICATION_PROMPT",
    "EMAIL_GENERATION_PROMPT",
]
