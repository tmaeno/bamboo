"""Email drafter: one downstream consumer of the prescription.

:class:`EmailDrafter` combines the navigator's root-cause analysis with the
prescription from :class:`~bamboo.agents.prescription_composer.PrescriptionComposer`
to produce a single coherent email for the task owner.

Keeping this separate from the composer means future consumers
(auto-optimizer, ticket creator, etc.) can also act on the prescription
without touching email logic.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from bamboo.llm import PRESCRIPTION_EMAIL_PROMPT, get_llm
from bamboo.models.knowledge_entity import AnalysisResult
from bamboo.utils.narrator import thinking

logger = logging.getLogger(__name__)


class EmailDrafter:
    """Drafts a resolution email from navigator analysis and prescription."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    async def draft(
        self,
        task_data: dict[str, Any],
        result: AnalysisResult,
        prescription: dict[str, Any] | None,
        domain_hints: str = "(none)",
    ) -> str:
        """Draft a resolution email.

        Args:
            task_data:    Raw task fields (``jediTaskID``, ``description`` used).
            result:       Navigator analysis result.
            prescription: Typed prescription from :class:`PrescriptionComposer`,
                          or ``None`` if none was produced.
            domain_hints: Domain documentation string.

        Returns:
            Email body as a plain-text string.
        """
        logger.info("EmailDrafter: drafting resolution email")

        raw_task_id = task_data.get("jediTaskID") or "unknown"
        task_status = task_data.get("status")
        task_id = f"{raw_task_id}:{task_status}" if task_status else str(raw_task_id)

        analysis_dict = {
            "root_cause": result.root_cause,
            "confidence": result.confidence,
            "resolution": result.resolution,
            "reasoning": result.explanation,
            "investigation_data": result.metadata.get("investigation_data", {}),
        }

        prescription_text = (
            json.dumps(prescription, indent=2, default=str)
            if prescription
            else "(no prescription available)"
        )

        prompt = PRESCRIPTION_EMAIL_PROMPT.format(
            task_id=task_id,
            task_description=task_data.get("description", ""),
            domain_hints=domain_hints,
            analysis=json.dumps(analysis_dict, indent=2, default=str),
            prescription=prescription_text,
        )
        messages = [
            SystemMessage(
                content="You are an expert at technical communication and customer support."
            ),
            HumanMessage(content=prompt),
        ]
        with thinking("Drafting email"):
            response = await self.llm.ainvoke(messages)
        return response.content
