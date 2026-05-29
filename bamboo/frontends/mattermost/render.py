"""Pure Mattermost message/attachment builders — no network, no client.

Kept dependency-free and side-effect-free so it can be unit-tested directly and
reused by both the Phase 1 poster and the Phase 2 ``MattermostInteractionIO``
render methods.  Functions here return plain dicts in Mattermost's
Slack-compatible *message attachment* format.
"""

from __future__ import annotations

from typing import Any

# Confidence band → attachment bar color (hex).
_COLOR_HIGH = "#2eb886"  # green
_COLOR_MEDIUM = "#daa038"  # amber
_COLOR_LOW = "#d24b4b"  # red


def confidence_color(confidence: float) -> str:
    """Map an analysis confidence in [0, 1] to an attachment bar color."""
    if confidence >= 0.7:
        return _COLOR_HIGH
    if confidence >= 0.4:
        return _COLOR_MEDIUM
    return _COLOR_LOW


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def analysis_attachment(result: Any) -> dict[str, Any]:
    """Render an :class:`~bamboo.models.knowledge_entity.AnalysisResult` as a
    Mattermost message attachment.

    Accepts any object exposing the ``AnalysisResult`` attributes
    (``task_id``, ``root_cause``, ``confidence``, ``resolution``, ``explanation``,
    ``unmatched_symptoms``, ``capability_gaps``) so tests can pass a lightweight
    stand-in.
    """
    task_id = getattr(result, "task_id", "") or "(unknown)"
    root_cause = getattr(result, "root_cause", "") or "(no root cause determined)"
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    resolution = getattr(result, "resolution", "") or "(none suggested)"
    explanation = getattr(result, "explanation", "") or ""
    unmatched = list(getattr(result, "unmatched_symptoms", []) or [])
    gaps = list(getattr(result, "capability_gaps", []) or [])

    fields: list[dict[str, Any]] = [
        {"title": "Root cause", "value": _clip(root_cause, 700), "short": False},
        {"title": "Confidence", "value": f"{confidence:.0%}", "short": True},
        {"title": "Task", "value": str(task_id), "short": True},
        {"title": "Suggested resolution", "value": _clip(resolution, 700), "short": False},
    ]
    if unmatched:
        fields.append(
            {
                "title": "Novel symptoms (no KB match)",
                "value": _clip("\n".join(f"• {s}" for s in unmatched), 700),
                "short": False,
            }
        )
    if gaps:
        gap_lines = [
            f"• {g.get('investigation', '?')} → needs: {g.get('suggested_tool_capability', '?')}"
            for g in gaps
            if isinstance(g, dict)
        ]
        if gap_lines:
            fields.append(
                {
                    "title": "Capability gaps",
                    "value": _clip("\n".join(gap_lines), 700),
                    "short": False,
                }
            )

    attachment: dict[str, Any] = {
        "fallback": f"bamboo analysis for task {task_id}: {_clip(root_cause, 120)}",
        "color": confidence_color(confidence),
        "title": f"bamboo analysis — task {task_id}",
        "fields": fields,
        "footer": "bamboo",
    }
    if explanation.strip():
        # The full reasoning narrative goes in the attachment text (secondary,
        # below the structured fields).
        attachment["text"] = _clip(explanation, 3500)
    return attachment


def analysis_message(result: Any) -> dict[str, Any]:
    """Return the ``props`` payload (``{"attachments": [...]}``) for a post."""
    return {"attachments": [analysis_attachment(result)]}
