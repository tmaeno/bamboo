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


_BATCH_SIZE = 20


async def canonicalize_descriptions(
    nodes: list[BaseNode],
    cache: dict[str, str] | None = None,
) -> None:
    """Rewrite task-instance-specific tokens in node descriptions in-place.

    Non-empty ``node.description`` values are processed in batches of
    :data:`_BATCH_SIZE` to keep each LLM response small and reliable.  The
    model is asked to rewrite each description to be task-agnostic (removing
    URLs, IDs, run-specific counts, dataset names, filenames, etc.) while
    preserving semantic meaning.  Descriptions that are already generic are
    returned unchanged.

    Falls back silently to the original descriptions for any batch where the
    LLM call fails or returns an unparseable / wrong-length response, so
    extraction is never blocked by this step.

    Args:
        nodes:  List of graph nodes to process (modified in-place).
        cache:  Optional dict mapping original description → canonical
                description.  Descriptions found in the cache are applied
                immediately without an LLM call; new results are stored in
                the cache so subsequent calls (e.g. retry passes) skip them.
                Pass ``self._desc_cache`` from :class:`KnowledgeGraphExtractor`
                to share the cache across all retry passes within one run.
    """
    from langchain_core.messages import HumanMessage

    from bamboo.llm import DESCRIPTION_CANONICALIZE_PROMPT, get_extraction_llm
    from bamboo.utils.narrator import say

    nodes_with_desc = [n for n in nodes if n.description]
    if not nodes_with_desc:
        return

    # Resolve cache: use caller-supplied dict or a local throwaway one.
    _cache: dict[str, str] = cache if cache is not None else {}

    # Apply cached rewrites immediately; collect only uncached nodes for LLM.
    to_canonicalize: list[BaseNode] = []
    for node in nodes_with_desc:
        if node.description in _cache:
            node.description = _cache[node.description]
        else:
            to_canonicalize.append(node)

    n_cached = len(nodes_with_desc) - len(to_canonicalize)

    if not to_canonicalize:
        say(f"Canonicalizing node descriptions... (all {n_cached} cached, skipping LLM)")
        return

    if n_cached:
        say(f"Canonicalizing node descriptions ({len(to_canonicalize)} new, {n_cached} cached)...")
    else:
        say("Canonicalizing node descriptions...")

    llm = get_extraction_llm()
    total_changed = 0

    for batch_start in range(0, len(to_canonicalize), _BATCH_SIZE):
        batch = to_canonicalize[batch_start : batch_start + _BATCH_SIZE]
        original_descs = [n.description for n in batch]
        prompt = DESCRIPTION_CANONICALIZE_PROMPT.format(
            descriptions_json=json.dumps(original_descs, ensure_ascii=False)
        )
        try:
            with thinking("Working"):
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.startswith("```")
                ).strip()
            rewritten: list[str] = json.loads(raw)
            if len(rewritten) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} items, got {len(rewritten)}"
                )
            for node, orig, new_desc in zip(batch, original_descs, rewritten):
                if new_desc:
                    _cache[orig] = new_desc
                    if new_desc != node.description:
                        node.description = new_desc
                        total_changed += 1
        except Exception as exc:
            logger.warning(
                "canonicalize_descriptions: batch %d–%d failed (%s) — keeping original descriptions",
                batch_start,
                batch_start + len(batch) - 1,
                exc,
            )

    logger.debug(
        "canonicalize_descriptions: updated %d/%d description(s)",
        total_changed,
        len(nodes_with_desc),
    )
