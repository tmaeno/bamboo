"""Abstract base classes for extraction strategy plugins.

An :class:`ExtractionStrategy` converts raw incident data (email thread,
structured task fields, external metadata) into a :class:`KnowledgeGraph`.
New strategies are registered via :func:`~bamboo.extractors.factory.register_extraction_strategy`
and selected at runtime through the ``EXTRACTION_STRATEGY`` configuration key.
"""

from abc import ABC, abstractmethod
from typing import Any

from bamboo.models.knowledge_entity import KnowledgeGraph


class ExtractionStrategy(ABC):
    """Plugin interface for knowledge-graph extraction.

    Implement this class to add support for a new data source or extraction
    approach.  Register the implementation with
    :func:`~bamboo.extractors.factory.register_extraction_strategy` so it can
    be selected via configuration.

    Subclasses must implement :meth:`extract`, :meth:`supports_system`,
    :attr:`name`, and :attr:`description`.
    """

    @abstractmethod
    async def extract(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract a knowledge graph from the provided input sources.

        Implementations should treat all three arguments as optional: a caller
        may supply only ``task_data``, only ``email_text``, or any combination.

        Args:
            email_text:    Email thread or communication text.  May be empty.
            task_data:     Structured task/issue data as a flat dict.  May be
                           ``None``.
            external_data: External metadata (logs, metrics, JIRA fields, etc.)
                           as a flat dict.  May be ``None``.

        Returns:
            :class:`KnowledgeGraph` containing the extracted nodes and
            relationships.  Node IDs are assigned by the caller
            (:class:`~bamboo.extractors.knowledge_graph_extractor.KnowledgeGraphExtractor`)
            after extraction.
        """
        pass

    @abstractmethod
    def supports_system(self, system_type: str) -> bool:
        """Return ``True`` if this strategy can handle *system_type*.

        Used by the factory to auto-select a strategy when no explicit name is
        configured.

        Args:
            system_type: Lower-cased system identifier, e.g. ``"panda"``,
                         ``"jira"``, ``"generic"``.

        Returns:
            ``True`` if the strategy supports this system type.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique identifier for this strategy (e.g. ``"panda"``)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown in strategy listings."""
        pass
