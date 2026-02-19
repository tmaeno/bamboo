"""Extraction strategy factory and registry."""

import logging
from typing import Type

from bamboo.config import get_settings
from bamboo.extractors.base import ExtractionStrategy

logger = logging.getLogger(__name__)

# Registry of available extraction strategies
_extraction_strategies: dict[str, Type[ExtractionStrategy]] = {}


def register_extraction_strategy(name: str, strategy_class: Type[ExtractionStrategy]):
    """Register an extraction strategy.

    Args:
        name: Unique name for the strategy
        strategy_class: Class implementing ExtractionStrategy
    """
    _extraction_strategies[name.lower()] = strategy_class
    logger.debug(f"Registered extraction strategy: {name}")


def get_extraction_strategy(strategy: str = None) -> ExtractionStrategy:
    """Get appropriate extraction strategy.

    Args:
        strategy: Strategy name (llm, rule_based, jira, github, generic, etc.)
                 If None, uses EXTRACTION_STRATEGY from configuration.

    Returns:
        ExtractionStrategy instance

    Raises:
        ValueError: If no suitable strategy found
    """
    settings = get_settings()
    strategy = strategy or settings.extraction_strategy

    # First try explicit strategy name
    if strategy in _extraction_strategies:
        strategy_class = _extraction_strategies[strategy]
        logger.info(f"Using extraction strategy: {strategy}")
        return strategy_class()

    # Try to find strategy that supports this system
    for strategy_class in _extraction_strategies.values():
        instance = strategy_class()
        if instance.supports_system(strategy):
            logger.info(f"Using extraction strategy {instance.name} for: {strategy}")
            return instance

    raise ValueError(
        f"No extraction strategy found for: {strategy}. "
        f"Available strategies: {list(_extraction_strategies.keys())}"
    )


def list_extraction_strategies() -> list[dict]:
    """List all registered extraction strategies with details.

    Returns:
        List of dicts with strategy info
    """
    strategies = []
    for name, strategy_class in _extraction_strategies.items():
        strategy = strategy_class()
        strategies.append(
            {
                "name": strategy.name,
                "id": name,
                "description": strategy.description,
                "supports": "all systems"
                if strategy.supports_system("generic")
                else "structured systems only",
            }
        )
    return strategies


# Register built-in strategies
def _register_builtin_strategies():
    """Register built-in extraction strategies."""
    try:
        from bamboo.extractors.llm_strategy import LLMExtractionStrategy

        register_extraction_strategy("llm", LLMExtractionStrategy)
    except ImportError as e:
        logger.debug(f"LLM strategy not available: {e}")

    try:
        from bamboo.extractors.rule_based_strategy import (
            RuleBasedExtractionStrategy,
        )

        register_extraction_strategy("rule_based", RuleBasedExtractionStrategy)
        register_extraction_strategy("jira", RuleBasedExtractionStrategy)
        register_extraction_strategy("github", RuleBasedExtractionStrategy)
        register_extraction_strategy("generic", RuleBasedExtractionStrategy)
    except ImportError as e:
        logger.debug(f"Rule-based strategy not available: {e}")


# Auto-register built-in strategies on import
_register_builtin_strategies()
