"""Prescription composer: translates navigator analysis into typed action steps.

:class:`PrescriptionComposer` is a pure prescription producer — it outputs
only "what to do" and nothing else.  Email drafting is handled by the
separate :class:`~bamboo.agents.email_drafter.EmailDrafter` so that future
consumers (auto-optimizer, ticket creator, etc.) can also consume the
prescription without changes to this class.

Pipeline inside :meth:`PrescriptionComposer.compose`:

1. Classify the resolution into an action type
   (resubmit / contact_admin / fix_code / wait / other).
2. Fetch any resources required for that action type via the MCP client
   (e.g. CLI docs for *resubmit*).
3. Ask the LLM to compose concrete action steps from the analysis and
   fetched resources.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.llm import (
    PRESCRIPTION_CLASSIFY_PROMPT,
    PRESCRIPTION_COMPOSE_PROMPT,
    get_llm,
)
from bamboo.models.knowledge_entity import AnalysisResult
from bamboo.utils.sanitize import sanitize_for_llm
from bamboo.utils.narrator import thinking

logger = logging.getLogger(__name__)


class PrescriptionComposer:
    """Produces a typed prescription from navigator analysis.

    The composer is resolution-type-aware: it classifies the resolution,
    fetches any required resources (e.g. CLI help text for resubmission),
    and composes concrete action steps.

    Args:
        mcp_client: An MCP client exposing at least ``fetch_cli_options``.
    """

    def __init__(self, mcp_client):
        self._mcp = mcp_client
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    async def compose(
        self, task_data: dict[str, Any], result: AnalysisResult
    ) -> dict[str, Any] | None:
        """Produce a typed prescription dict.

        Returns:
            Dict with keys ``action_type``, ``hints``, ``notes``, and
            optionally ``command_template`` (for *resubmit* type).
            Returns ``None`` if the result carries no resolution.
        """
        if not result.resolution:
            logger.info("PrescriptionComposer: no resolution — skipping")
            return None

        components = result.metadata.get("components", [])
        investigation = result.metadata.get("investigation", {})

        classification = await self._classify(result, components, investigation)
        action_type = classification.get("action_type", "other")
        logger.info("PrescriptionComposer: action_type=%s", action_type)

        resources = await self._fetch_resources(
            action_type,
            classification.get("component"),
        )

        return await self._compose(task_data, result, action_type, resources, investigation)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _classify(
        self,
        result: AnalysisResult,
        components: list[str],
        investigation: dict,
    ) -> dict[str, Any]:
        """Phase 1: classify resolution type and determine needed resources."""
        prompt = PRESCRIPTION_CLASSIFY_PROMPT.format(
            root_cause=result.root_cause,
            resolution=result.resolution,
            investigation=json.dumps(investigation, indent=2, default=str),
            components=", ".join(components) if components else "(none)",
        )
        messages = [
            SystemMessage(content="You are a PanDA task failure analyst."),
            HumanMessage(content=prompt),
        ]
        with thinking("Classifying prescription type"):
            response = await self.llm.ainvoke(messages)
        return _parse_json(response.content, default={"action_type": "other"})

    async def _fetch_resources(
        self, action_type: str, component: str | None
    ) -> dict[str, Any]:
        """Phase 2: fetch resources required for the action type."""
        resources: dict[str, Any] = {}
        if action_type == "resubmit" and component in ("prun", "pathena"):
            try:
                cli_result = await self._mcp.execute(
                    "fetch_cli_options", component=component
                )
                if "error" not in cli_result:
                    resources["cli_docs"] = cli_result.get(component, "")
                    logger.info(
                        "PrescriptionComposer: fetched CLI docs for %s (%d chars)",
                        component,
                        len(resources["cli_docs"]),
                    )
                else:
                    logger.warning(
                        "PrescriptionComposer: fetch_cli_options error: %s",
                        cli_result["error"],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("PrescriptionComposer: failed to fetch CLI docs: %s", exc)
        return resources

    async def _compose(
        self,
        task_data: dict[str, Any],
        result: AnalysisResult,
        action_type: str,
        resources: dict[str, Any],
        investigation: dict,
    ) -> dict[str, Any]:
        """Phase 3: compose concrete action steps."""
        cli_docs_section = ""
        if resources.get("cli_docs"):
            cli_docs_section = (
                f"CLI REFERENCE (--help output):\n{resources['cli_docs']}"
            )

        task_summary = json.dumps(
            sanitize_for_llm(task_data), indent=2, default=str
        )
        prompt = PRESCRIPTION_COMPOSE_PROMPT.format(
            task_summary=task_summary,
            root_cause=result.root_cause,
            resolution=result.resolution,
            investigation=json.dumps(investigation, indent=2, default=str),
            action_type=action_type,
            cli_docs_section=cli_docs_section,
        )
        messages = [
            SystemMessage(content="You are a PanDA task failure analyst."),
            HumanMessage(content=prompt),
        ]
        with thinking("Composing prescription"):
            response = await self.llm.ainvoke(messages)

        prescription = _parse_json(
            response.content,
            default={"action_type": action_type, "hints": [], "notes": ""},
        )
        prescription.setdefault("action_type", action_type)
        return prescription


def _parse_json(text: str, default: dict) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text[text.find("```json") + 7 : text.rfind("```")].strip()
    elif "```" in text:
        text = text[text.find("```") + 3 : text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("PrescriptionComposer: failed to parse LLM response: %s", exc)
        return default
