"""Abstract base classes for extraction strategy plugins.

An :class:`ExtractionStrategy` converts raw incident data (email thread,
structured task fields, external metadata, task-level log output, job-level
log output, and raw job attributes) into a :class:`KnowledgeGraph`.  New
strategies are registered via
:func:`~bamboo.extractors.factory.register_extraction_strategy` and selected
at runtime through the ``EXTRACTION_STRATEGY`` configuration key.

Log-level distinction
---------------------
Two separate log parameters are accepted:

``task_logs``
    Task-level logs produced by orchestration services (e.g. JEDI,
    Harvester).  These describe scheduling decisions, dataset handling, and
    task-level errors.  Every extracted node is tagged
    ``log_level="task"`` in its metadata.

``job_logs``
    Job-level logs produced by execution workers (e.g. pilot logs, Athena /
    payload logs).  These describe what happened inside individual job
    processes.  Every extracted node is tagged ``log_level="job"`` in its
    metadata.

Keeping the two separate preserves provenance: a
:class:`~bamboo.models.graph_element.ComponentNode` originating from a pilot
log is clearly an execution-layer component, while one from a JEDI log is a
scheduling-layer component.
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
        task_logs: dict[str, str] = None,
        job_logs: dict[str, str] = None,
        jobs_data: list[dict[str, Any]] = None,
    ) -> KnowledgeGraph:
        """Extract a knowledge graph from the provided input sources.

        Implementations should treat all arguments as optional: a caller may
        supply any combination of sources.

        Args:
            email_text:    Email thread or communication text.  May be empty.
            task_data:     Structured task/issue data as a flat dict.  May be
                           ``None``.
            external_data: External metadata (metrics, JIRA fields, etc.) as a
                           flat dict.  May be ``None``.
            task_logs:     *Task-level* log output from orchestration services
                           (e.g. JEDI, Harvester), keyed by source name
                           (e.g. ``{"jedi": "...", "harvester": "..."}``)
                           May be ``None``.  Extracted nodes are tagged
                           ``log_level="task"`` in their metadata.
            job_logs:      *Job-level* log output from execution workers
                           (e.g. pilot, Athena/payload), keyed by a stable
                           source name such as ``"pilot"`` or ``"payload"``
                           (NOT a raw PanDA job ID).  May be ``None``.
                           Extracted nodes are tagged ``log_level="job"``
                           in their metadata.
            jobs_data:     List of raw job attribute dicts (one per PanDA job).
                           Used by :class:`PandaJobDataAggregator` to derive
                           aggregated :class:`JobFeatureNode` values, dominant
                           error signals, and representative context texts.
                           May be ``None``.

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
