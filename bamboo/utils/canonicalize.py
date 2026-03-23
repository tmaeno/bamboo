"""Description canonicalization for knowledge-graph nodes.

:func:`canonicalize_descriptions` is a shared async utility that strips
task-instance-specific tokens from node description fields using the LLM.

It is called by :class:`~bamboo.agents.extractors.knowledge_graph_extractor.KnowledgeGraphExtractor`
after every extraction so the canonicalization applies uniformly regardless of
which strategy produced the nodes.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from bamboo.utils.narrator import thinking

if TYPE_CHECKING:
    from bamboo.models.graph_element import BaseNode

logger = logging.getLogger(__name__)


async def canonicalize_descriptions(nodes: list[BaseNode]) -> None:
    """Rewrite task-instance-specific tokens in node descriptions in-place.

    All non-empty ``node.description`` values are batched into a **single** LLM
    call.  The model is asked to rewrite each description to be task-agnostic
    (removing URLs, IDs, run-specific counts, dataset names, filenames, etc.)
    while preserving semantic meaning.  Descriptions that are already generic
    are returned unchanged.

    Falls back silently to the original descriptions if the LLM call fails or
    returns an unparseable response, so extraction is never blocked by this
    step.

    Args:
        nodes: List of graph nodes to process (modified in-place).
    """
    from langchain_core.messages import HumanMessage

    from bamboo.llm import DESCRIPTION_CANONICALIZE_PROMPT, get_extraction_llm
    from bamboo.utils.narrator import say

    nodes_with_desc = [n for n in nodes if n.description]
    if not nodes_with_desc:
        return

    say("Canonicalizing node descriptions...")
    descriptions = [n.description for n in nodes_with_desc]
    prompt = DESCRIPTION_CANONICALIZE_PROMPT.format(
        descriptions_json=json.dumps(descriptions, ensure_ascii=False)
    )
    try:
        with thinking("Working"):
            response = await get_extraction_llm().ainvoke(
                [HumanMessage(content=prompt)]
            )
        raw = response.content.strip()
        # Strip optional markdown code fences the model may add
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines() if not line.startswith("```")
            ).strip()
        rewritten: list[str] = json.loads(raw)
        if len(rewritten) != len(nodes_with_desc):
            raise ValueError(
                f"Expected {len(nodes_with_desc)} items, got {len(rewritten)}"
            )
        changed = 0
        for node, new_desc in zip(nodes_with_desc, rewritten):
            if new_desc and new_desc != node.description:
                node.description = new_desc
                changed += 1
        logger.debug(
            "canonicalize_descriptions: updated %d/%d description(s)",
            changed,
            len(nodes_with_desc),
        )
    except Exception as exc:
        logger.warning(
            "canonicalize_descriptions: failed (%s) — keeping original descriptions",
            exc,
        )
