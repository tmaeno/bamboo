"""Diagnostic script: show raw LLM output for doc-node classification.

Fetches a single ReadTheDocs page, parses it into DocNode objects using the
same code as the index builder, then runs the classification prompt through
the extraction LLM and prints the raw response before any JSON parsing.

Usage::

    # Default: terminology page, all nodes
    python -m bamboo.scripts.panda.test_doc_classification

    # Only nodes whose title contains "Task"
    python -m bamboo.scripts.panda.test_doc_classification --filter "Task"

    # Different page
    python -m bamboo.scripts.panda.test_doc_classification --url https://panda-wms.readthedocs.io/en/latest/architecture/architecture.html

    # Also print the full prompt sent to the LLM
    python -m bamboo.scripts.panda.test_doc_classification --show-prompt
"""

from __future__ import annotations

import asyncio
import json
import os

import click

_DEFAULT_URL = (
    "https://panda-wms.readthedocs.io/en/latest/terminology/terminology.html"
)
_DOCS_HTML_BASE = "https://panda-wms.readthedocs.io/en/latest"


def _url_to_rst_path(url: str) -> str:
    rel = url.removeprefix(_DOCS_HTML_BASE + "/").removesuffix(".html")
    return f"docs/source/{rel}.rst"


@click.command()
@click.option(
    "--url",
    default=_DEFAULT_URL,
    show_default=True,
    help="ReadTheDocs HTML page to test.",
)
@click.option(
    "--filter",
    "title_filter",
    default=None,
    metavar="TEXT",
    help="Only test nodes whose title contains TEXT (case-insensitive).",
)
@click.option(
    "--show-prompt",
    is_flag=True,
    default=False,
    help="Print the full prompt sent to the LLM.",
)
def main(url: str, title_filter: str | None, show_prompt: bool) -> None:
    """Show raw LLM classification output for doc nodes."""
    asyncio.run(_run(url, title_filter, show_prompt))
    os._exit(0)


async def _run(url: str, title_filter: str | None, show_prompt: bool) -> None:
    import httpx
    from bs4 import BeautifulSoup
    from langchain_core.messages import HumanMessage, SystemMessage

    from bamboo.agents.panda_doc_navigator import PandaDocNavigator
    from bamboo.llm import PANDA_DOC_SUMMARIZE_SYSTEM, PANDA_DOC_SUMMARIZE_USER, get_extraction_llm

    click.echo(f"Fetching {url}")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

    rst_path = _url_to_rst_path(url)
    nav = PandaDocNavigator()
    nodes = nav._parse_page_to_nodes(rst_path, html, BeautifulSoup)
    click.echo(f"Parsed {len(nodes)} node(s)")

    node_map = {n.id: n for n in nodes}

    def root_title(node) -> str:
        current = node
        while current.parent_id:
            parent = node_map.get(current.parent_id)
            if parent is None:
                break
            current = parent
        return current.title

    if title_filter:
        nodes = [n for n in nodes if title_filter.lower() in n.title.lower()]
        click.echo(f"Filtered to {len(nodes)} node(s) matching {title_filter!r}")

    llm = get_extraction_llm()

    for node in nodes:
        page_title = root_title(node)
        label = f"{page_title} › {node.title}" if page_title != node.title else node.title
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Node:    {label}")
        click.echo(f"Level:   {node.level}  |  Content: {len(node.content)} chars")
        preview = node.content[:120].replace("\n", " ")
        click.echo(f"Preview: {preview!r}")

        if not node.content.strip():
            click.echo("[empty content — skipping LLM call]")
            continue

        user_content = PANDA_DOC_SUMMARIZE_USER.format(
            page_title=page_title,
            title=node.title,
            content=node.content[:3000],
        )

        if show_prompt:
            click.echo(f"\n--- SYSTEM ---\n{PANDA_DOC_SUMMARIZE_SYSTEM}\n--- USER ---\n{user_content}\n--- END PROMPT ---")

        resp = await llm.ainvoke(
            [
                SystemMessage(content=PANDA_DOC_SUMMARIZE_SYSTEM),
                HumanMessage(content=user_content),
            ]
        )
        raw = resp.content.strip()
        click.echo(f"\nRaw LLM response:\n{raw}")

        stripped = raw
        if stripped.startswith("```"):
            stripped = "\n".join(
                line for line in stripped.splitlines() if not line.startswith("```")
            ).strip()
        try:
            parsed = json.loads(stripped)
            doc_type = parsed.get("doc_type", "(missing)")
            summary = parsed.get("summary", "")[:100]
            click.echo(f"\nParsed  -> doc_type: {doc_type!r}  summary: {summary!r}")
        except json.JSONDecodeError as exc:
            click.echo(f"\nJSON parse FAILED: {exc}")


if __name__ == "__main__":
    main()
