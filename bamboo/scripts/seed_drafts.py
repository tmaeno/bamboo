"""Generate seeding drafts for bamboo commissioning.

Reads a CSV of problematic PanDA tasks, checks coverage against the Qdrant DB and
an approved-email library, then generates structured JSON draft files that a human
reviews before batch population.

Three tiers:
  - Skipped:          Current PanDA status differs from CSV status.
  - DB-covered:       Qdrant already has a similar Symptom vector.
  - Approved-matched: approved_email_drafts/ has a reviewed draft for a similar error.
  - New:              No coverage found; LLM draft generated from scratch.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from bamboo.utils.logging import setup_logging

console = Console()
logger = logging.getLogger(__name__)

_FEATURE_DIFF_FIELDS = ["taskType", "prodSourceLabel", "site", "coreCount", "splitRule"]

_DRAFT_PROMPT = """\
A PanDA task has failed. You are helping to seed a knowledge base for an automated \
analysis system called Bamboo.

Task information:
  Status:       {status}
  Error:        {error_dialog}
  Task name:    {task_name}
  Source label: {prod_source_label}

Relevant PanDA documentation:
{panda_docs}

Write a structured investigation email that an expert support engineer would write \
after diagnosing this failure. Respond with ONLY a valid JSON object — no prose, no \
markdown fences — with exactly these four string/array fields:

{{
  "background": "1-2 sentences: what the task does and what went wrong",
  "cause": "The specific root cause (technical, actionable — no vague generalities)",
  "resolution": "How to fix or work around the issue",
  "procedure": ["step 1: ...", "step 2: ...", "step 3: ..."]
}}

The procedure list should contain 2-5 concrete investigation steps.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-10)


async def _canonicalize(raw_error: str) -> str:
    """Return a canonicalized (task-instance-agnostic) version of the error string."""
    from bamboo.models.graph_element import SymptomNode
    from bamboo.utils.canonicalize import canonicalize_descriptions

    node = SymptomNode(name="symptom", description=raw_error)
    await canonicalize_descriptions([node])
    return node.description or raw_error


async def _fetch_panda_docs(task_data: dict[str, Any]) -> str:
    """Return a short block of relevant PanDA doc snippets for the given task."""
    from langchain_core.messages import HumanMessage

    from bamboo.llm import DOC_SEARCH_KEYWORDS_PROMPT, get_extraction_llm
    from bamboo.mcp.panda_mcp_client import PandaMcpClient

    error_dialog = task_data.get("errorDialog", "") or ""
    if not error_dialog:
        return "(no error dialog)"

    plain = re.sub(r"<[^>]+>", " ", error_dialog)
    plain = " ".join(plain.split())

    keywords: list[str] = [plain[:120]]
    try:
        llm = get_extraction_llm()
        prompt = DOC_SEARCH_KEYWORDS_PROMPT.format(
            error_dialog=plain[:500],
            email_text="(none)",
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```")).strip()
        parsed = json.loads(raw)
        if parsed and isinstance(parsed, list):
            keywords = [k for k in parsed if isinstance(k, str)][:5]
    except Exception as exc:
        logger.debug("keyword extraction failed: %s", exc)

    panda = PandaMcpClient()
    parts: list[str] = []
    for kw in keywords:
        try:
            results = await panda.execute("search_panda_docs", query=kw)
            if isinstance(results, list):
                for entry in results[:2]:
                    snippet = entry.get("snippet", "")
                    title = entry.get("title", "")
                    if snippet:
                        parts.append(f"[{title}] {snippet}".strip())
        except Exception as exc:
            logger.debug("search_panda_docs failed for %r: %s", kw, exc)

    return "\n\n".join(parts) if parts else "(no relevant docs found)"


async def _generate_email_body(task_data: dict[str, Any], panda_docs: str) -> dict[str, Any]:
    """Call LLM to generate a structured email_body dict."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from bamboo.llm import get_llm

    prompt = _DRAFT_PROMPT.format(
        status=task_data.get("status", ""),
        error_dialog=task_data.get("errorDialog", ""),
        task_name=task_data.get("taskName", ""),
        prod_source_label=task_data.get("prodSourceLabel", ""),
        panda_docs=panda_docs,
    )
    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(content="You are an expert PanDA system administrator."),
            HumanMessage(content=prompt),
        ]
    )
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```")).strip()
    return json.loads(raw)


def _feature_diff(new_task: dict[str, Any], source_task: dict[str, Any]) -> str:
    diffs = [
        f"{f}: {source_task.get(f)!r}→{new_task.get(f)!r}"
        for f in _FEATURE_DIFF_FIELDS
        if new_task.get(f) != source_task.get(f)
    ]
    if not diffs:
        return "No key task feature differences detected."
    return "Key differences vs. source: " + ", ".join(diffs)


def _load_approved_entries(approved_dir: Path) -> list[dict[str, Any]]:
    """Load all approved draft entries that have a stored embedding."""
    entries = []
    for f in sorted(approved_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "errorDialog_embedding" in data:
                data["_file"] = str(f)
                entries.append(data)
        except Exception as exc:
            logger.warning("skipping malformed approved entry %s: %s", f, exc)
    return entries


def _find_best_approved_match(
    embedding: list[float],
    approved_entries: list[dict[str, Any]],
    threshold: float,
) -> tuple[str, float, dict[str, Any]] | None:
    """Return (file_path, score, entry) for the best match above threshold, else None."""
    best_score = -1.0
    best_entry: dict[str, Any] | None = None
    for entry in approved_entries:
        score = _cosine_sim(embedding, entry["errorDialog_embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_score >= threshold and best_entry is not None:
        return best_entry["_file"], best_score, best_entry
    return None


def _greedy_cluster(
    items: list[tuple[Any, list[float]]],
    threshold: float,
) -> list[list[Any]]:
    """Greedy cosine-similarity clustering. Returns list of clusters (each is a list of items)."""
    assigned = [False] * len(items)
    clusters: list[list[Any]] = []
    for i, (item_i, emb_i) in enumerate(items):
        if assigned[i]:
            continue
        cluster = [item_i]
        assigned[i] = True
        for j, (item_j, emb_j) in enumerate(items):
            if assigned[j]:
                continue
            if _cosine_sim(emb_i, emb_j) >= threshold:
                cluster.append(item_j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def _make_draft(
    *,
    rep_task: dict[str, Any],
    task_ids: list[int],
    email_body: dict[str, Any],
    errorDialog_canonical: str,
    review_hint: str,
    matched_from: str | None,
) -> dict[str, Any]:
    return {
        "reviewed": False,
        "review_hint": review_hint,
        "matched_from": matched_from,
        "task_ids": task_ids,
        "task_data": rep_task,
        "errorDialog_canonical": errorDialog_canonical,
        "email_body": email_body,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------


async def _run(
    csv_path: str,
    output_dir: str,
    approved_dir: str,
    threshold: float,
    concurrency: int,
    skip_existing: bool,
    verbose: bool,
) -> None:
    from bamboo.database.vector_database_client import VectorDatabaseClient
    from bamboo.llm import get_embeddings
    from bamboo.utils.panda_client import fetch_task_data

    setup_logging()
    from bamboo.utils.narrator import set_narrator

    set_narrator(console, verbose=verbose)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    approved = Path(approved_dir)
    approved.mkdir(parents=True, exist_ok=True)

    # -- Read CSV ---------------------------------------------------------
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    if not rows:
        console.print("[yellow]CSV is empty — nothing to do.[/yellow]")
        return

    task_id_col = next((c for c in rows[0] if c.lower() in ("jedtaskid", "jeditaskid", "task_id", "taskid")), None)
    status_col = next((c for c in rows[0] if c.lower() == "status"), None)
    if not task_id_col or not status_col:
        console.print("[red]CSV must have jediTaskID and status columns.[/red]")
        sys.exit(1)

    console.print(f"[bold]Reading {len(rows)} rows from {csv_path}[/bold]")

    # -- Phase A: Fetch task data + consistency check ---------------------
    sem = asyncio.Semaphore(concurrency)

    async def _fetch_and_check(row: dict[str, str]) -> dict[str, Any] | None:
        task_id = int(row[task_id_col])
        csv_status = row[status_col].strip()
        async with sem:
            try:
                task_data = await fetch_task_data(task_id)
            except Exception as exc:
                console.print(f"[yellow]  Task {task_id}: fetch failed ({exc}) — skipping[/yellow]")
                return None
        live_status = task_data.get("status", "")
        if live_status != csv_status:
            console.print(
                f"[yellow]  Task {task_id}: status changed "
                f"({csv_status!r} → {live_status!r}) — skipping[/yellow]"
            )
            return None
        return task_data

    with console.status("[bold green]Fetching task data from PanDA..."):
        fetched = await asyncio.gather(*[_fetch_and_check(r) for r in rows])

    valid_tasks: list[dict[str, Any]] = [t for t in fetched if t is not None]
    n_skipped = len(rows) - len(valid_tasks)
    console.print(f"  {len(valid_tasks)} tasks valid, {n_skipped} skipped (status changed)")

    if not valid_tasks:
        console.print("[yellow]No valid tasks to process.[/yellow]")
        return

    # -- Phase B: Exact-string dedup + canonicalize -----------------------
    exact_groups: dict[str, list[dict[str, Any]]] = {}
    for task in valid_tasks:
        key = task.get("errorDialog") or ""
        exact_groups.setdefault(key, []).append(task)

    console.print(
        f"  {len(valid_tasks)} tasks → {len(exact_groups)} unique errorDialogs after exact dedup"
    )

    with console.status("[bold green]Canonicalizing errorDialogs..."):
        canonical_map: dict[str, str] = {}
        for raw in exact_groups:
            canonical_map[raw] = await _canonicalize(raw)

    # -- Phase C: Coverage check ------------------------------------------
    vector_db = VectorDatabaseClient()
    await vector_db.connect()
    embeddings_model = get_embeddings()
    approved_entries = _load_approved_entries(approved)

    db_covered_count = 0
    approved_matched_groups: dict[str, tuple[str, float, dict[str, Any], list[dict[str, Any]]]] = {}
    new_groups: dict[str, list[dict[str, Any]]] = {}
    group_embeddings: dict[str, list[float]] = {}

    try:
        with console.status("[bold green]Checking coverage (DB + approved drafts)..."):
            for raw_error, tasks in exact_groups.items():
                # Embed raw errorDialog (consistent with how Qdrant stores Symptom vectors)
                embedding = await embeddings_model.aembed_query(raw_error or "(empty)")
                group_embeddings[raw_error] = embedding

                # DB check
                hits = await vector_db.search_similar(
                    query_embedding=embedding,
                    limit=1,
                    score_threshold=threshold,
                    filter_conditions={"section": "Symptom"},
                )
                if hits:
                    db_covered_count += len(tasks)
                    continue

                # Approved-draft check
                match = _find_best_approved_match(embedding, approved_entries, threshold)
                if match:
                    matched_file, score, matched_entry = match
                    approved_matched_groups[raw_error] = (matched_file, score, matched_entry, tasks)
                    continue

                new_groups[raw_error] = tasks
    finally:
        await vector_db.close()

    console.print(
        f"  {db_covered_count} tasks DB-covered, "
        f"{sum(len(v[3]) for v in approved_matched_groups.values())} approved-matched, "
        f"{sum(len(v) for v in new_groups.values())} new"
    )

    # -- Phase D: Cluster --------------------------------------------------
    # Cluster approved-matched by which approved file they matched (same file = same cluster)
    # then further cluster by within-group similarity
    approved_by_file: dict[str, list[tuple[str, float, dict[str, Any], dict[str, Any]]]] = {}
    for raw_error, (matched_file, score, matched_entry, tasks) in approved_matched_groups.items():
        approved_by_file.setdefault(matched_file, []).append(
            (raw_error, score, matched_entry, tasks)
        )

    # Cluster new groups by embedding similarity
    if new_groups:
        new_items = [(raw, group_embeddings[raw], tasks) for raw, tasks in new_groups.items()]
        # Cluster error dialogs that are similar to each other
        clusterable = [(raw, emb) for raw, emb, _ in new_items]
        raw_clusters = _greedy_cluster(clusterable, threshold)
        # Map clusters back to tasks
        raw_to_tasks = {raw: tasks for raw, _, tasks in new_items}
        new_clusters: list[tuple[str, list[dict[str, Any]]]] = []
        for cluster_raws in raw_clusters:
            all_tasks = []
            rep_raw = cluster_raws[0]
            for raw in cluster_raws:
                all_tasks.extend(raw_to_tasks[raw])
            new_clusters.append((rep_raw, all_tasks))
    else:
        new_clusters = []

    # -- Phase E: Generate draft files ------------------------------------
    n_approved_drafts = 0
    n_new_drafts = 0

    # Approved-matched drafts
    for matched_file, groups in approved_by_file.items():
        # Flatten all tasks for this matched file
        all_tasks = [t for _, _, _, tasks in groups for t in tasks]
        rep_task = all_tasks[0]
        task_ids = [t["jediTaskID"] for t in all_tasks]
        score = groups[0][1]

        matched_entry = groups[0][2]
        source_task_data = matched_entry.get("task_data", {})
        diff_str = _feature_diff(rep_task, source_task_data)

        hint = (
            f"Pre-filled from {matched_file} (errorDialog similarity {score:.2f}). "
            f"{diff_str} "
            f"Verify the cause is still correct for these differences before approving."
        )

        out_file = output / f"task_{rep_task['jediTaskID']}.json"
        if skip_existing and out_file.exists():
            n_approved_drafts += 1
            continue

        draft = _make_draft(
            rep_task=rep_task,
            task_ids=task_ids,
            email_body=matched_entry["email_body"],
            errorDialog_canonical=canonical_map.get(rep_task.get("errorDialog", ""), ""),
            review_hint=hint,
            matched_from=matched_file,
        )
        out_file.write_text(json.dumps(draft, indent=2, default=str))
        n_approved_drafts += 1

    # New drafts
    for rep_raw, cluster_tasks in new_clusters:
        rep_task = cluster_tasks[0]
        task_ids = [t["jediTaskID"] for t in cluster_tasks]

        out_file = output / f"task_{rep_task['jediTaskID']}.json"
        if skip_existing and out_file.exists():
            n_new_drafts += 1
            continue

        with console.status(f"[bold green]Generating draft for task {rep_task['jediTaskID']}..."):
            panda_docs = await _fetch_panda_docs(rep_task)
            try:
                email_body = await _generate_email_body(rep_task, panda_docs)
            except Exception as exc:
                logger.warning("email generation failed for task %s: %s", rep_task.get("jediTaskID"), exc)
                email_body = {
                    "background": f"Task {rep_task.get('jediTaskID')} failed with: {rep_task.get('errorDialog', '')}",
                    "cause": "(DRAFT GENERATION FAILED — fill in manually)",
                    "resolution": "",
                    "procedure": [],
                }

        draft = _make_draft(
            rep_task=rep_task,
            task_ids=task_ids,
            email_body=email_body,
            errorDialog_canonical=canonical_map.get(rep_raw, ""),
            review_hint="Generated from scratch — full review required: check all sections carefully.",
            matched_from=None,
        )
        out_file.write_text(json.dumps(draft, indent=2, default=str))
        n_new_drafts += 1

    # -- Summary ----------------------------------------------------------
    table = Table(title="seed-drafts summary")
    table.add_column("Category", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Draft files", justify="right")
    table.add_row("Skipped (status changed)", str(n_skipped), "—")
    table.add_row("DB-covered (no action needed)", str(db_covered_count), "—")
    table.add_row(
        "Approved-matched (pre-filled, quick review)",
        str(sum(len(v[3]) for v in approved_matched_groups.values())),
        str(n_approved_drafts),
    )
    table.add_row(
        "New (full review required)",
        str(sum(len(tasks) for _, tasks in new_clusters)),
        str(n_new_drafts),
    )
    console.print(table)
    console.print(f"\nDrafts written to [bold]{output_dir}/[/bold]")
    console.print(
        "Next: review draft files, set [bold]\"reviewed\": true[/bold], "
        "then run [bold]bamboo batch-populate[/bold]."
    )


# ---------------------------------------------------------------------------
# Click entry point
# ---------------------------------------------------------------------------


@click.command("seed-drafts")
@click.option(
    "--csv",
    "csv_path",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with jediTaskID and status columns.",
)
@click.option(
    "--output",
    "output_dir",
    default="drafts",
    show_default=True,
    type=click.Path(),
    help="Directory to write draft JSON files.",
)
@click.option(
    "--approved",
    "approved_dir",
    default="approved_email_drafts",
    show_default=True,
    type=click.Path(),
    help="Directory containing the approved email draft library.",
)
@click.option(
    "--similarity-threshold",
    default=0.85,
    show_default=True,
    type=float,
    help="Cosine similarity threshold for DB coverage and approved-draft matching.",
)
@click.option(
    "--concurrency",
    default=5,
    show_default=True,
    type=int,
    help="Maximum concurrent PanDA fetch requests.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip draft files that already exist in the output directory.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(csv_path, output_dir, approved_dir, similarity_threshold, concurrency, skip_existing, verbose):
    """Generate seeding drafts for bamboo commissioning.

    Reads a CSV of problematic PanDA tasks and classifies each into one of four
    tiers: skipped (status changed), DB-covered, approved-matched (pre-filled
    from library), or new (LLM-generated draft).

    Examples:

    \b
      bamboo seed-drafts --csv tasks.csv
      bamboo seed-drafts --csv tasks.csv --output my_drafts/ --approved approved/
      bamboo seed-drafts --csv tasks.csv --similarity-threshold 0.90
    """
    asyncio.run(
        _run(
            csv_path=csv_path,
            output_dir=output_dir,
            approved_dir=approved_dir,
            threshold=similarity_threshold,
            concurrency=concurrency,
            skip_existing=skip_existing,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    main()
