"""Knowledge review agent: gap analysis of extracted graphs before DB commit.

:class:`KnowledgeReviewer` sits between extraction and storage.  It calls an
LLM to evaluate whether the extracted :class:`~bamboo.models.knowledge_entity.KnowledgeGraph`
adequately describes the incident — identifying structural, specificity, and
contextual gaps rather than cross-checking against source text.

Gaps are grounded in either the graph itself (structural implications) or the
available task context (e.g. nJobsFailed > 0 but no JOB_FEATURE nodes).  The
LLM is instructed not to speculate beyond what the graph and context imply.

Failure modes are handled defensively: any LLM or parse error causes the
reviewer to return ``approved=True`` (fail-open), so a reviewer malfunction
never blocks the accumulation pipeline.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from bamboo.llm import KNOWLEDGE_REVIEW_PROMPT, get_extraction_llm
from bamboo.models.knowledge_entity import KnowledgeGraph
from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)

# Maximum characters to include per source excerpt sent to the reviewer.
_MAX_EMAIL_CHARS = 2000
_MAX_LOG_CHARS = 1000

# Task fields forwarded to the LLM as incident context for gap detection.
_TASK_SUMMARY_KEYS = (
    "jediTaskID",
    "status",
    "errorDialog",
    "retryID",
    "transUses",
    "prodSourceLabel",
    "taskName",
    "taskType",
    "nJobs",
    "nJobsFinished",
    "nJobsFailed",
)


@dataclass
class ReviewResult:
    """Outcome of one review pass.

    Attributes:
        approved:               ``True`` if the graph adequately describes the incident.
        feedback:               Actionable instructions for the extractor (empty when approved).
        confidence:             Reviewer's confidence in its verdict (0.0–1.0).
        issues:                 List of identified gaps (empty when approved).
        relevant_feature_nodes: Feature node names (Task_Feature / Job_Feature) that the
                                reviewer identified as directly causally relevant based on
                                domain documentation.  Used by the accumulator to create
                                explicit ``contribute_to`` edges rather than blanket ones.
    """

    approved: bool
    feedback: str = ""
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    relevant_feature_nodes: list[str] = field(default_factory=list)


class KnowledgeReviewer:
    """LLM-based incident gap analyzer for extracted knowledge graphs.

    Call :meth:`review` after extraction and before storing to Neo4j/Qdrant.
    If the result is not approved, pass ``result.feedback`` back to the
    extractor as ``review_feedback`` and retry.
    """

    async def review(
        self,
        graph: KnowledgeGraph,
        sources: dict[str, str],
        task_data: dict[str, Any] | None = None,
        available_tools: list | None = None,
        doc_hints: dict[str, str] | None = None,
    ) -> ReviewResult:
        """Evaluate the extracted graph for incident completeness.

        Identifies gaps that are structurally implied by the graph or
        contextually supported by ``task_data``.  Does not require source
        text to be present.

        Args:
            graph:            The extracted :class:`KnowledgeGraph` to evaluate.
            sources:          Mapping of source name → text excerpt (optional context
                              hints).  Pass the result of :func:`build_sources_summary`.
            task_data:        Raw task fields from PanDA — used to detect contextual
                              gaps (e.g. many failed jobs but no JOB_FEATURE nodes).
            available_tools:  List of :class:`~bamboo.mcp.McpTool` descriptors that
                              the explorer can invoke to fill gaps.  When provided,
                              the reviewer annotates each issue with the tool that
                              could resolve it (e.g. ``"→ resolvable with get_failed_job_details"``).

        Returns:
            :class:`ReviewResult` with the verdict and corrective feedback.
            Always returns ``approved=True`` on any internal error (fail-open).
        """
        if not graph.nodes:
            say("Graph is empty — skipping review.")
            return ReviewResult(approved=True, confidence=1.0)

        graph_summary = _serialise_graph(graph)
        task_summary = _build_task_summary(task_data or {})

        sources_summary = _serialise_sources(sources) if sources else "(none)"
        domain_hints = _join_doc_hints(doc_hints)

        show_block("reviewer: task context", task_summary)
        if doc_hints:
            show_block("reviewer: panda docs hints", domain_hints)

        try:
            llm = get_extraction_llm()
            say(f"Analysing graph for incident gaps and node relevance ({len(graph.nodes)} nodes)...")
            with thinking("Working"):
                response = await llm.ainvoke(
                    KNOWLEDGE_REVIEW_PROMPT.format(
                        graph_summary=graph_summary,
                        task_summary=task_summary,
                        sources_summary=sources_summary,
                        domain_hints=domain_hints,
                        available_tools=_format_available_tools(available_tools),
                    )
                )
            result = _parse_review_response(response.content)
            if result.approved:
                say(f"Graph approved (confidence {result.confidence:.0%}).")
            else:
                say(
                    f"Graph rejected (confidence {result.confidence:.0%}) — "
                    f"{len(result.issues)} gap(s) found."
                )
                if result.issues:
                    show_block(
                        "reviewer gaps",
                        "\n".join(f"• {i}" for i in result.issues),
                    )
            if result.relevant_feature_nodes:
                show_block(
                    "reviewer: causally relevant features",
                    "\n".join(f"• {n}" for n in result.relevant_feature_nodes),
                )
            return result
        except Exception:
            logger.exception("KnowledgeReviewer: LLM call failed — failing open (approved=True)")
            return ReviewResult(approved=True, confidence=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_sources_summary(
    email_text: str = "",
    task_logs: dict[str, str] | None = None,
    job_logs: dict[str, str] | None = None,
    doc_hints: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build truncated source excerpts for use as optional reviewer context.

    Args:
        email_text: Raw email thread.
        task_logs:  Task-level logs keyed by source name.
        job_logs:   Job-level logs keyed by source name.
        doc_hints:  PanDA documentation hints keyed by query string (already
                    rendered as plain text — not JSON).

    Returns:
        Dict mapping source label → truncated text.
    """
    sources: dict[str, str] = {}
    if email_text and email_text.strip():
        sources["email"] = email_text[:_MAX_EMAIL_CHARS]
    for name, text in (task_logs or {}).items():
        if text and text.strip():
            sources[f"task_log:{name}"] = text[:_MAX_LOG_CHARS]
    for name, text in (job_logs or {}).items():
        if text and text.strip():
            sources[f"job_log:{name}"] = text[:_MAX_LOG_CHARS]
    for query, text in (doc_hints or {}).items():
        if text and text.strip():
            sources[f"panda_docs:{query[:40]}"] = text[:_MAX_LOG_CHARS * 3]
    return sources


def _join_doc_hints(doc_hints: dict[str, str] | None) -> str:
    """Concatenate all doc hint values into a single prompt string."""
    if not doc_hints:
        return "(none)"
    return "\n\n".join(v for v in doc_hints.values() if v)


def _format_available_tools(tools: list | None) -> str:
    """Format a list of :class:`~bamboo.mcp.McpTool` descriptors for the reviewer prompt."""
    if not tools:
        return "(none — no explorer configured)"
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def _build_task_summary(task_data: dict[str, Any]) -> str:
    """Return a compact JSON string of task fields relevant to gap detection."""
    subset: dict[str, Any] = {k: task_data[k] for k in _TASK_SUMMARY_KEYS if k in task_data}
    if subset.get("errorDialog"):
        subset["errorDialog"] = str(subset["errorDialog"])[:500]
    if not subset:
        return "(no task context available)"
    return json.dumps(subset, indent=2, default=str)


def _serialise_graph(graph: KnowledgeGraph) -> str:
    """Produce a compact text representation of the graph for the reviewer."""
    from collections import Counter

    type_counts = Counter(n.node_type.value for n in graph.nodes)
    lines = ["Node counts: " + ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items()))]
    lines.append("")
    lines.append("Nodes:")
    for n in graph.nodes:
        desc_snippet = (n.description or "")[:120]
        lines.append(f"  [{n.node_type.value}] {n.name!r}  — {desc_snippet}")
    lines.append("")
    lines.append("Relationships:")
    for r in graph.relationships:
        lines.append(f"  {r.source_id!r} --[{r.relation_type.value}]--> {r.target_id!r}")
    return "\n".join(lines)


def _serialise_sources(sources: dict[str, str]) -> str:
    """Format source excerpts for the reviewer prompt."""
    parts = []
    for label, text in sources.items():
        parts.append(f"=== {label} ===\n{text}")
    return "\n\n".join(parts)


def _parse_review_response(response: str) -> ReviewResult:
    """Parse the LLM's JSON review response, failing open on any error."""
    text = response.strip()
    if "```json" in text:
        text = text[text.find("```json") + 7 : text.rfind("```")].strip()
    elif "```" in text:
        text = text[text.find("```") + 3 : text.rfind("```")].strip()
    try:
        data = json.loads(text)
        return ReviewResult(
            approved=bool(data.get("approved", True)),
            feedback=str(data.get("feedback", "")),
            confidence=float(data.get("confidence", 1.0)),
            issues=[str(i) for i in data.get("issues", [])],
            relevant_feature_nodes=[str(n) for n in data.get("relevant_feature_nodes", [])],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("KnowledgeReviewer: failed to parse review response: %s", exc)
        return ReviewResult(approved=True, confidence=0.0)
