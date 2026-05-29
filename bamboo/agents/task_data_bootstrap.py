"""Shared task_data → initial-graph-skeleton helper.

This module exposes :func:`bootstrap_initial_graph`, a thin wrapper around
:meth:`bamboo.agents.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor._bootstrap_from_task_data`
that turns a PanDA ``task_data`` dict (plus optional ``external_data``) into
the deterministic initial set of graph nodes — Symptom from ``errorDialog`` and
``status``; Task_Feature nodes for discrete/continuous/split-rule/cli-argument
fields; Task_Context for unstructured prose; ComponentNode for submission-tool
names.

It is the single source of truth for the "what graph do we get for free from a
task_data dict?" question and is reused by:

* :class:`bamboo.agents.extractors.panda_knowledge_extractor.PandaKnowledgeExtractor`
  during populate's full extraction (its ``extract()`` calls
  ``_bootstrap_from_task_data`` first, then layers email + log-source LLM
  extraction on top).
* :class:`bamboo.agents.investigation_session.InvestigationOrchestrator`
  at session start (``investigate`` doesn't have an email or task-log input
  modality — the dialog itself replaces those).

The investigate path passes ``extract_embedded_logs=False`` so the optional
HTML-log-link fetch path inside ``errorDialog`` handling is skipped — investigate
captures logs explicitly via the human-driven dialog loop instead.
"""

from __future__ import annotations

from typing import Any, Optional

from bamboo.models.graph_element import GraphRelationship


async def bootstrap_initial_graph(
    task_data: Optional[dict[str, Any]] = None,
    external_data: Optional[dict[str, Any]] = None,
    *,
    error_classifier: Any = None,
    extract_embedded_logs: bool = False,
) -> tuple[list, list[GraphRelationship]]:
    """Build the deterministic initial graph skeleton from task_data + external_data.

    Args:
        task_data:               PanDA task fields. ``errorDialog`` becomes a
                                 canonical SymptomNode (via the error
                                 classifier); ``status`` becomes a
                                 deterministic ``TaskStatusXxx`` SymptomNode;
                                 other fields are classified into Task_Feature
                                 (discrete / continuous-bucketed / split-rule /
                                 cli-argument) or Task_Context (unstructured
                                 prose). Sensitive identity fields are
                                 pseudonymised. Unrecognised keys are silently
                                 skipped.
        external_data:           Each k/v becomes a Task_Feature node tagged
                                 ``source="external_data"``.
        error_classifier:        Optional pre-built
                                 :class:`~bamboo.agents.extractors.panda_knowledge_extractor.ErrorCategoryClassifier`.
                                 When ``None`` a default classifier is created
                                 (which lazily provisions its VectorDB-backed
                                 store on first call). Pass an explicit
                                 instance for tests that want to inject a
                                 mock.
        extract_embedded_logs:   When ``True``, HTML log URLs found in
                                 ``errorDialog`` are fetched and run through
                                 the LLM log-extraction prompt — same behavior
                                 as populate. When ``False`` (the default for
                                 investigate), the embedded-log fetch is
                                 skipped and the human can request logs
                                 explicitly via the dialog loop. Investigate
                                 uses False; populate uses True.

    Returns:
        ``(nodes, relationships)``. ``relationships`` is empty unless
        ``extract_embedded_logs=True`` produced edges from the embedded log
        extraction.
    """
    # Lazy import to avoid a circular: PandaKnowledgeExtractor is the canonical
    # owner of the field-classification tables and helper methods; this module
    # is a thin wrapper for callers that don't already hold an extractor
    # instance.
    from bamboo.agents.extractors.panda_knowledge_extractor import (  # noqa: PLC0415
        PandaKnowledgeExtractor,
    )

    extractor = PandaKnowledgeExtractor(error_classifier=error_classifier)
    return await extractor._bootstrap_from_task_data(
        task_data=task_data,
        external_data=external_data,
        extract_embedded_logs=extract_embedded_logs,
    )
