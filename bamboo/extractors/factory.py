"""Extraction strategy factory and registry.

Strategies are registered by name at import time via
:func:`register_extraction_strategy`.  The active strategy is resolved by
:func:`get_extraction_strategy`, which first checks for an exact name match
and then falls back to calling :meth:`~bamboo.extractors.base.ExtractionStrategy.supports_system`
on each registered strategy.

Built-in strategies registered automatically:

============  ====================================================
Name          Class
============  ====================================================
``panda``     :class:`~bamboo.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`
``llm``       ``LLMExtractionStrategy`` (optional dependency)
``rule_based`` / ``jira`` / ``github`` / ``generic``
              ``RuleBasedExtractionStrategy`` (optional dependency)
============  ====================================================
"""

import logging
from typing import Type

from bamboo.config import get_settings
from bamboo.extractors.base import ExtractionStrategy

logger = logging.getLogger(__name__)

# Registry: lower-cased name → strategy class
_extraction_strategies: dict[str, Type[ExtractionStrategy]] = {}


def register_extraction_strategy(name: str, strategy_class: Type[ExtractionStrategy]):
    """Register an extraction strategy under *name*.

    If a strategy is already registered under the same name it is silently
    replaced, which allows tests to inject mock implementations.

    Args:
        name:           Lower-cased registry key (e.g. ``"panda"``).
        strategy_class: Concrete :class:`ExtractionStrategy` subclass.
    """
    _extraction_strategies[name.lower()] = strategy_class
    logger.debug("Registered extraction strategy: %s", name)


def get_extraction_strategy(strategy: str = None) -> ExtractionStrategy:
    """Return an instantiated :class:`ExtractionStrategy`.

    Resolution order:

    1. If *strategy* (or ``EXTRACTION_STRATEGY`` from config when *None*) is an
       exact key in the registry, return an instance of that class.
    2. Otherwise iterate the registry and return the first strategy whose
       :meth:`~ExtractionStrategy.supports_system` returns ``True`` for the
       given name.

    Args:
        strategy: Strategy name or system identifier.  ``None`` uses the
                  ``EXTRACTION_STRATEGY`` configuration value.

    Returns:
        A freshly instantiated :class:`ExtractionStrategy`.

    Raises:
        ValueError: If no registered strategy matches *strategy*.
    """
    settings = get_settings()
    strategy = strategy or settings.extraction_strategy

    if strategy in _extraction_strategies:
        logger.info("Using extraction strategy: %s", strategy)
        return _extraction_strategies[strategy]()

    for strategy_class in _extraction_strategies.values():
        instance = strategy_class()
        if instance.supports_system(strategy):
            logger.info(
                "Using extraction strategy '%s' for system: %s", instance.name, strategy
            )
            return instance

    raise ValueError(
        f"No extraction strategy found for: {strategy!r}. "
        f"Available strategies: {sorted(_extraction_strategies.keys())}"
    )


def list_extraction_strategies() -> list[dict]:
    """Return metadata for every registered extraction strategy.

    Returns:
        List of dicts with keys ``name``, ``id``, ``description``,
        ``supports``.
    """
    result = []
    for name, strategy_class in _extraction_strategies.items():
        instance = strategy_class()
        result.append(
            {
                "name": instance.name,
                "id": name,
                "description": instance.description,
                "supports": (
                    "all systems"
                    if instance.supports_system("generic")
                    else "structured systems only"
                ),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Auto-registration of built-in strategies
# ---------------------------------------------------------------------------


def _register_builtin_strategies():
    """Register built-in extraction strategies.  Called once at import time."""
    try:
        from bamboo.extractors.llm_strategy import LLMExtractionStrategy

        register_extraction_strategy("llm", LLMExtractionStrategy)
    except ImportError as exc:
        logger.debug("LLM strategy not available: %s", exc)

    try:
        from bamboo.extractors.rule_based_strategy import RuleBasedExtractionStrategy

        register_extraction_strategy("rule_based", RuleBasedExtractionStrategy)
        register_extraction_strategy("jira", RuleBasedExtractionStrategy)
        register_extraction_strategy("github", RuleBasedExtractionStrategy)
        register_extraction_strategy("generic", RuleBasedExtractionStrategy)
    except ImportError as exc:
        logger.debug("Rule-based strategy not available: %s", exc)

    try:
        from bamboo.extractors.panda_knowledge_extractor import PandaKnowledgeExtractor

        register_extraction_strategy("panda", PandaKnowledgeExtractor)
    except ImportError as exc:
        logger.debug("Panda strategy not available: %s", exc)


_register_builtin_strategies()
