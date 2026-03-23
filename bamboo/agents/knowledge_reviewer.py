"""Knowledge review agent: evaluates extracted graphs before DB commit.

:class:`KnowledgeReviewer` sits between extraction and storage.  It calls an
LLM to compare the extracted :class:`~bamboo.models.knowledge_entity.KnowledgeGraph`
against truncated originals of the source text and decides whether the
extraction is good enough to store, or whether it should be retried with
corrective feedback.

Failure modes are handled defensively: any LLM or parse error causes the
reviewer to return ``approved=True`` (fail-open), so a reviewer malfunction
never blocks the accumulation pipeline.
"""

import json
import logging
from dataclasses import dataclass, field

from bamboo.llm import KNOWLEDGE_REVIEW_PROMPT, get_extraction_llm
from bamboo.models.knowledge_entity import KnowledgeGraph

logger = logging.getLogger(__name__)

# Maximum characters to include per source excerpt sent to the reviewer.
_MAX_EMAIL_CHARS = 2000
_MAX_LOG_CHARS = 1000


@dataclass
class ReviewResult:
    """Outcome of one review pass.

    Attributes:
        approved:   ``True`` if the graph adequately captures the sources.
        feedback:   Actionable instructions for the extractor (empty when approved).
        confidence: Reviewer's confidence in its verdict (0.0–1.0).
        issues:     List of specific problems found (empty when approved).
    """

    approved: bool
    feedback: str = ""
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)


class KnowledgeReviewer:
    """LLM-based quality gate for extracted knowledge graphs.

    Call :meth:`review` after extraction and before storing to Neo4j/Qdrant.
    If the result is not approved, pass ``result.feedback`` back to the
    extractor as ``review_feedback`` and retry.
    """

    async def review(
        self,
        graph: KnowledgeGraph,
        sources: dict[str, str],
    ) -> ReviewResult:
        """Evaluate the extracted graph against the original sources.

        Args:
            graph:   The extracted :class:`KnowledgeGraph` to evaluate.
            sources: Mapping of source name → text excerpt.  Each value is
                     already truncated by the caller; pass the result of
                     :func:`build_sources_summary`.

        Returns:
            :class:`ReviewResult` with the verdict and corrective feedback.
            Always returns ``approved=True`` on any internal error (fail-open).
        """
        llm_extracted_count = sum(
            1
            for n in graph.nodes
            if (n.metadata or {}).get("log_source") or (n.metadata or {}).get("source") == "email"
        )
        if not sources or llm_extracted_count == 0:
            # No LLM-extracted content to review — nothing to critique.
            return ReviewResult(approved=True, confidence=1.0)

        graph_summary = _serialise_graph(graph)
        sources_summary = _serialise_sources(sources)

        try:
            llm = get_extraction_llm()
            response = await llm.ainvoke(
                KNOWLEDGE_REVIEW_PROMPT.format(
                    graph_summary=graph_summary,
                    sources_summary=sources_summary,
                )
            )
            return _parse_review_response(response.content)
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
) -> dict[str, str]:
    """Build truncated source excerpts suitable for the reviewer.

    Only text-based sources that were passed through the LLM extractor are
    included.  Structured sources (task_data, external_data) are deterministic
    and do not need review.

    Args:
        email_text: Raw email thread.
        task_logs:  Task-level logs keyed by source name.
        job_logs:   Job-level logs keyed by source name.

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
    return sources


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
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("KnowledgeReviewer: failed to parse review response: %s", exc)
        return ReviewResult(approved=True, confidence=0.0)
