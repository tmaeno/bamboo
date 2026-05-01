"""Context-prefetching utilities: PanDA docs + source-code analysis.

Three public async functions:

``prefetch_panda_docs(task_data, email_text)``
    BM25 ReadTheDocs search, splitRule param lookup, gdpconfig param lookup.
    Returns a ``dict[str, str]`` of hint entries keyed by query string.

``prefetch_panda_source(task_data)``
    Runs PandaSourceNavigator on the task errorDialog to produce a code-level
    explanation.  Returns an empty dict if pandaserver/pandajedi are not
    installed or no errorDialog is present.

``prefetch_panda_context(task_data, email_text)``
    Calls both concurrently and returns the merged dict.  This is the
    preferred entry point for callers that want full context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from bamboo.utils.narrator import say, show_block, thinking

logger = logging.getLogger(__name__)


async def prefetch_panda_docs(
    task_data: dict[str, Any], email_text: str = ""
) -> dict[str, str]:
    """Fetch PanDA documentation hints before the first extraction pass.

    Derives search queries from key ``task_data`` fields (status,
    errorDialog, splitRule sub-rule keys) and an optional ``email_text``,
    then calls ``search_panda_docs`` via PandaMcpClient.

    For ``errorDialog`` and ``email_text`` an LLM call extracts 2-5 focused
    search terms.  Falls back to the raw 120-char errorDialog string if the
    LLM call fails.

    Returns a ``doc_hints`` dict: keys are query strings, values are plain
    rendered text (``"[Title] snippet\\n\\n[Title2] snippet2"``).
    Returns an empty dict on any error so the caller is unaffected.
    """
    error_dialog = task_data.get("errorDialog", "") or ""
    plain_error = ""
    if error_dialog:
        plain_error = re.sub(r"<[^>]+>", " ", error_dialog)
        plain_error = re.sub("#[^ ]+", " ", plain_error)
        plain_error = " ".join(plain_error.split())

    doc_query = ""
    if plain_error or email_text:
        try:
            from bamboo.llm import DOC_SEARCH_KEYWORDS_PROMPT, get_extraction_llm  # noqa: PLC0415
            from langchain_core.messages import HumanMessage  # noqa: PLC0415

            llm = get_extraction_llm()
            prompt = DOC_SEARCH_KEYWORDS_PROMPT.format(
                error_dialog=plain_error[:500].rsplit(None, 1)[0] if plain_error else "(none)",
                email_text=email_text[:500].rsplit(None, 1)[0] if email_text else "(none)",
            )
            with thinking("Extracting doc search keywords"):
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines() if not line.startswith("```")
                ).strip()
            keywords: list[str] = json.loads(raw)
            clean = [k for k in keywords if k and isinstance(k, str)]
            if clean:
                doc_query = " ".join(clean)
                say(f"Doc search keywords: {clean}")
        except Exception as exc:
            logger.warning(
                "prefetch_panda_docs: keyword extraction failed (%s) — falling back to raw errorDialog",
                exc,
            )

        if not doc_query and plain_error:
            doc_query = plain_error[:120]

    status = task_data.get("status", "")
    parts: list[str] = []
    if status:
        parts.append(f"task status {status}")
    if doc_query:
        parts.append(doc_query)

    if not parts:
        return {}

    combined_query = " ".join(parts)

    from bamboo.mcp.panda_mcp_client import PandaMcpClient, _fetch_task_params_table  # noqa: PLC0415

    panda_client = PandaMcpClient()
    doc_hints: dict[str, str] = {}
    try:
        results = await panda_client.execute("search_panda_docs", query=combined_query)
        if isinstance(results, list) and results:
            rendered = "\n\n".join(
                f"[{e.get('title', '')}] {e.get('snippet', '')}".strip()
                for e in results if e.get("snippet")
            )
            if rendered:
                doc_hints[combined_query] = rendered
                say(
                    f"Pre-fetched {len(results)} PanDA doc section(s) "
                    f"for query: {combined_query!r}"
                )
                show_block(f"doc_hints: {combined_query}", rendered, max_lines=120)
    except Exception as exc:
        logger.warning(
            "prefetch_panda_docs: search_panda_docs failed for query=%r: %s",
            combined_query,
            exc,
        )

    split_rule_str = task_data.get("splitRule", "") or ""
    if split_rule_str:
        table = await _fetch_task_params_table()
        if table:
            found: dict[str, str] = {}
            for sub_rule in split_rule_str.split(","):
                sub_rule = sub_rule.strip()
                if "=" in sub_rule:
                    key = sub_rule.split("=", 1)[0].strip()
                    if key and key not in found and key in table:
                        found[key] = table[key]
            if found:
                doc_hints["splitRule params"] = "\n".join(
                    f"- {k}: {v}" for k, v in found.items()
                )
                say(
                    f"Looked up {len(found)} splitRule description(s) "
                    "from task_params.rst"
                )
                show_block("doc_hints: splitRule params", doc_hints["splitRule params"], max_lines=120)

    # Scan all fetched doc_hints text for gdpconfig UPPERCASE key names and
    # append their descriptions — same targeted-lookup approach as splitRule.
    if doc_hints:
        from bamboo.mcp.panda_mcp_client import _fetch_gdpconfig_table  # noqa: PLC0415

        gdp_table = await _fetch_gdpconfig_table()
        if gdp_table:
            all_text = " ".join(doc_hints.values())
            candidates = set(re.findall(r"\b[A-Z][A-Z0-9_]*[A-Z0-9]\b", all_text))
            found_gdp: dict[str, str] = {}
            for word in candidates:
                if word in gdp_table:
                    found_gdp[word] = gdp_table[word]
                else:
                    base = re.sub(r"_<[^>]+>$", "", word)
                    if base != word and base in gdp_table:
                        found_gdp[base] = gdp_table[base]
            if found_gdp:
                doc_hints["gdpconfig params"] = "\n".join(
                    f"- {k}: {v}" for k, v in sorted(found_gdp.items())
                )
                say(f"Looked up {len(found_gdp)} gdpconfig param(s) from gdpconfig.rst")
                show_block("doc_hints: gdpconfig params", doc_hints["gdpconfig params"], max_lines=120)

    return doc_hints


async def prefetch_panda_source(task_data: dict[str, Any]) -> dict[str, str]:
    """Return source-code analysis hint from PandaSourceNavigator.

    Returns an empty dict if pandaserver/pandajedi are not installed or if
    no errorDialog is present.  Never raises — errors are silently swallowed
    so the caller's pipeline is unaffected.
    """
    from bamboo.agents.panda_source_navigator import PandaSourceNavigator, _get_pkg_roots  # noqa: PLC0415

    error_dialog = (task_data.get("errorDialog") or "").strip()
    if not error_dialog or not _get_pkg_roots():
        return {}
    try:
        result = await PandaSourceNavigator().navigate(error_dialog)
    except Exception as exc:
        logger.warning("prefetch_panda_source: navigator failed (%s)", exc)
        return {}
    if not result or result.startswith("No methods found") or result.startswith("Neither"):
        return {}
    return {"source code analysis": f"Source code analysis:\n{result}"}


async def prefetch_panda_context(
    task_data: dict[str, Any], email_text: str = ""
) -> dict[str, str]:
    """Fetch both PanDA doc hints and source-code analysis concurrently.

    Returns a merged ``dict[str, str]`` suitable for use as ``doc_hints``
    throughout the pipeline.  Either sub-fetch may return empty without
    affecting the other.
    """
    doc_hints, src_hints = await asyncio.gather(
        prefetch_panda_docs(task_data, email_text),
        prefetch_panda_source(task_data),
    )
    return {**doc_hints, **src_hints}
