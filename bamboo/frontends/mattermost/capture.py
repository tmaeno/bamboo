"""Capture-from-thread — turn a Mattermost incident discussion into knowledge.

The MM-native form of ``bamboo populate``: the thread transcript takes the place
of the email the CLI ingests.  Ops summarise cause + resolution, bamboo extracts
(dry-run) so the graph can be **reviewed inline** before it is stored, then
persists via the shared :meth:`KnowledgeAccumulator.store_extracted` commit path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from bamboo.frontends.base import InteractionIO

logger = logging.getLogger(__name__)


def build_email_text(transcript: str, cause: str, resolution: Optional[str]) -> str:
    """Compose the ingestion text from the thread transcript + ops summary.

    The explicit cause/resolution are appended so the extractor reliably picks
    them up even if they were stated loosely in the discussion.
    """
    parts = [transcript.strip(), ""]
    if cause:
        parts.append(f"Cause: {cause.strip()}")
    if resolution:
        parts.append(f"Resolution: {resolution.strip()}")
    return "\n".join(p for p in parts if p is not None).strip()


async def _classify_nodes(graph: Any, graph_db: Any) -> list[tuple[str, str, str]]:
    """Per-node ``(label, name, "new"|"merge")`` for the inline review diff."""
    rows: list[tuple[str, str, str]] = []
    for n in graph.nodes:
        nt = getattr(n, "node_type", None)
        label = nt.value if hasattr(nt, "value") else str(nt or "Node")
        existing = None
        if graph_db is not None:
            try:
                existing = await graph_db.get_node_description(label, n.name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("classify get_node_description(%s,%s) failed: %s", label, n.name, exc)
        rows.append((label, n.name, "merge" if existing is not None else "new"))
    return rows


async def run_capture(
    io: InteractionIO,
    *,
    transcript: str,
    task_id: Optional[int],
    accumulator: Any,
    graph_db: Any = None,
    mcp_client: Any = None,
) -> bool:
    """Drive a capture-from-thread session. Returns True if knowledge was stored."""
    # 1) Optionally enrich with structured task data — via the shared fetch seam
    #    (same one analyze/investigate/populate use). Capture only needs the dict,
    #    not the MCP tool client.
    task_data: Optional[dict[str, Any]] = None
    if task_id is not None:
        from bamboo.agents.deps import resolve_task_data  # noqa: PLC0415

        try:
            task_data = await resolve_task_data(task_id)
        except Exception as exc:  # noqa: BLE001
            io.notice(f"[yellow]Could not fetch task_data for {task_id}: {exc}[/yellow]")

    # 2) Ops summary (the curation judgment).
    cause = await io.ask("[bold]What was the cause?[/bold]")
    resolution = await io.ask(
        "[bold]What was the resolution? (optional, blank to skip)[/bold]", default=""
    )
    resolution = resolution or None

    # 3) Extract without storing — this is the preview for the inline review gate.
    io.notice("[dim]Extracting knowledge from the discussion…[/dim]")
    email_text = build_email_text(transcript, cause, resolution)
    result = await accumulator.process_knowledge(
        email_text=email_text, task_data=task_data, dry_run=True
    )
    graph = result.graph
    if not graph.nodes:
        io.notice("[yellow]No knowledge could be extracted from this thread.[/yellow]")
        return False

    # 4) Inline review gate: show the diff, then confirm.
    rows = await _classify_nodes(graph, graph_db)
    edges = [
        (
            r.source_id,
            r.target_id,
            r.relation_type.value if hasattr(r.relation_type, "value") else str(r.relation_type),
        )
        for r in graph.relationships
    ]
    io.diff(rows, edge_count=len(graph.relationships), edges=edges)
    if not await io.confirm("Ingest this discussion into the knowledge base?", default=True):
        io.notice("[yellow]Capture cancelled.[/yellow]")
        return False

    # 5) Store the exact previewed graph via the shared commit path.
    await accumulator.store_extracted(
        graph, summary=result.summary, key_insights=result.key_insights
    )
    io.notice(
        f"[green]✓ Captured {len(graph.nodes)} node(s), "
        f"{len(graph.relationships)} edge(s) into the knowledge base.[/green]"
    )
    return True
