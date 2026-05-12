"""Inspect Qdrant panda_docs collection payload directly.

Scrolls all stored points and shows doc_type, level, title, and summary
without loading embeddings or LLMs.  Use this to verify that --rebuild-docs
actually wrote the expected doc_type values into the index.

Usage::

    # Show all nodes with summary statistics
    python -m bamboo.scripts.panda.inspect_doc_index

    # Only nodes whose title contains "Task"
    python -m bamboo.scripts.panda.inspect_doc_index --filter "Task"

    # Only concept-tagged nodes
    python -m bamboo.scripts.panda.inspect_doc_index --concept-only
"""

from __future__ import annotations

import asyncio
import os

import click

_COLLECTION = "panda_docs"


@click.command()
@click.option(
    "--filter",
    "title_filter",
    default=None,
    metavar="TEXT",
    help="Only show nodes whose title contains TEXT (case-insensitive).",
)
@click.option(
    "--concept-only",
    is_flag=True,
    default=False,
    help="Only show nodes with doc_type == 'concept'.",
)
def main(title_filter: str | None, concept_only: bool) -> None:
    """Dump Qdrant panda_docs payload to stdout."""
    asyncio.run(_run(title_filter, concept_only))
    os._exit(0)


async def _run(title_filter: str | None, concept_only: bool) -> None:
    from qdrant_client import AsyncQdrantClient

    from bamboo.config import get_settings

    settings = get_settings()
    kwargs: dict = {"url": settings.qdrant_url, "check_compatibility": False}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key

    client = AsyncQdrantClient(**kwargs)
    try:
        collections = await client.get_collections()
        names = {c.name for c in collections.collections}
        if _COLLECTION not in names:
            click.echo(f"Collection '{_COLLECTION}' does not exist — run --rebuild-docs first.")
            return

        all_points: list = []
        offset = None
        while True:
            points, next_offset = await client.scroll(
                collection_name=_COLLECTION,
                offset=offset,
                limit=200,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset
    finally:
        await client.close()

    total = len(all_points)
    concept_count = sum(1 for p in all_points if p.payload.get("doc_type") == "concept")
    other_count = total - concept_count

    click.echo(f"Collection '{_COLLECTION}': {total} nodes")
    click.echo(f"  concept: {concept_count}   other: {other_count}")
    click.echo("")

    payloads = [p.payload for p in all_points]
    payloads.sort(key=lambda p: (p.get("url", ""), p.get("level", 0)))

    id_to_title: dict[str, str] = {p.get("node_id", ""): p.get("title", "") for p in payloads}

    shown = 0
    for p in payloads:
        doc_type = p.get("doc_type", "other")
        title = p.get("title", "")
        level = p.get("level", 0)
        summary = (p.get("summary") or "")[:100]
        parent_id = p.get("parent_id")
        parent_title = id_to_title.get(parent_id, "") if parent_id else ""

        if concept_only and doc_type != "concept":
            continue
        if title_filter and title_filter.lower() not in title.lower():
            continue

        label = f"{parent_title} › {title}" if parent_title else title
        tag = "[concept]" if doc_type == "concept" else "[other]  "
        click.echo(f"{tag} L{level}  {label}")
        if summary:
            click.echo(f"           {summary}")
        shown += 1

    if title_filter or concept_only:
        click.echo(f"\n{shown} node(s) shown.")


if __name__ == "__main__":
    main()
