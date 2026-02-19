"""Abstract base classes for extraction strategy plugins."""

from abc import ABC, abstractmethod
from typing import Any

from bamboo.models.knowledge_entity import KnowledgeGraph


class ExtractionStrategy(ABC):
    """Abstract base class for extraction strategies.

    Different extraction strategies can be used depending on:
    - Target system type (task management, CI/CD, monitoring, etc.)
    - Data format and structure
    - Available fields and attributes
    - Extraction rules and patterns
    """

    @abstractmethod
    async def extract(
        self,
        email_text: str = "",
        task_data: dict[str, Any] = None,
        external_data: dict[str, Any] = None,
    ) -> KnowledgeGraph:
        """Extract knowledge graph from sources using this strategy.

        Args:
            email_text: Email thread or communication text
            task_data: Structured task/issue data
            external_data: External information (logs, metrics, etc.)

        Returns:
            KnowledgeGraph with extracted nodes and relationships
        """
        pass

    @abstractmethod
    def supports_system(self, system_type: str) -> bool:
        """Check if this strategy supports the given system type.

        Args:
            system_type: System identifier (e.g., 'jira', 'github', 'generic')

        Returns:
            True if strategy can handle this system
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extraction strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this strategy."""
        pass
