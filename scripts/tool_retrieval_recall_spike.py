#!/usr/bin/env python
"""Recall spike for the tool-selection design (the plan's validation gate).

This is a **throwaway diagnostic**, not production code. It answers the one
load-bearing question before Phase 1 is tuned for a large catalogue: *does
embedding retrieval actually surface the tools an investigation needs?* — using
the project's configured (local) embeddings, so the numbers reflect production.

It measures two signals:

* **Source #2 (tool-description retrieval)** — embeds each tool's
  ``name + description + param names`` and, for each labelled symptom, checks
  whether the human-chosen tools land in the top-K. Measurable now (tools exist).
  Run this against the *large* catalogue (configure ``MCP_SERVERS_CONFIG``) for a
  meaningful number — a ~14-tool catalogue is below budget and never retrieves.

* **Source #1 (validated past investigations)** — best-effort, held-out: over the
  procedures already in the graph (``find_all_procedures``), hide one, embed the
  rest's ``trigger_signals + code_summary``, query with the held-out's text, and
  check whether the held-out procedure's tools appear in the retrieved
  procedures' tool sets. Early on this corpus is tiny; that's expected.

Usage::

    python scripts/tool_retrieval_recall_spike.py --labels labels.json --source 2
    python scripts/tool_retrieval_recall_spike.py --source 1            # needs Neo4j
    python scripts/tool_retrieval_recall_spike.py --catalogue tools.json --labels labels.json

``--labels`` JSON (for source #2): a list of objects::

    [{"symptom": "stage-out failed at the endpoint", "tools": ["get_failed_job_log_summary", "fetch_linked_log_files"]},
     {"symptom": "scout phase exhausted", "tools": ["get_scout_job_details"]}]

``--catalogue`` JSON (optional; otherwise the live MCP client is used): a list of
``{"name": ..., "description": ..., "parameters_schema": {...}}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from bamboo.llm.llm_client import get_embeddings

# --- shared helpers ----------------------------------------------------------


def _param_names(schema: dict[str, Any]) -> str:
    return ", ".join(((schema or {}).get("properties") or {}).keys())


def _tool_text(name: str, description: str, schema: dict[str, Any]) -> str:
    """The text source #2 embeds for a tool (mirror of ToolSelector)."""
    params = _param_names(schema)
    suffix = f" (args: {params})" if params else ""
    return f"{name}: {description}{suffix}"


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _recall(expected: set[str], retrieved: list[str]) -> float:
    if not expected:
        return float("nan")
    return len(expected & set(retrieved)) / len(expected)


# --- source #2: tool-description retrieval -----------------------------------


def _load_catalogue_from_file(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters_schema": t.get("parameters_schema", {}),
        }
        for t in data
    ]


async def _load_catalogue_live() -> list[dict[str, Any]]:
    from bamboo.config import get_settings  # noqa: PLC0415
    from bamboo.mcp.factory import build_mcp_client  # noqa: PLC0415

    client = build_mcp_client(get_settings())
    await client.connect()
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters_schema": t.parameters_schema,
        }
        for t in client.list_tools()
    ]


async def run_source2(args: argparse.Namespace, emb: Any) -> None:
    print("\n=== Source #2: tool-description retrieval ===")
    if args.catalogue:
        catalogue = _load_catalogue_from_file(Path(args.catalogue))
    else:
        catalogue = await _load_catalogue_live()
    if not catalogue:
        print("  no tools in catalogue — skipping")
        return
    if not args.labels:
        print("  --labels required for source #2 — skipping")
        return
    labels = json.loads(Path(args.labels).read_text())

    names = [t["name"] for t in catalogue]
    texts = [_tool_text(t["name"], t.get("description", ""), t.get("parameters_schema", {})) for t in catalogue]
    print(f"  embedding {len(texts)} tool descriptions …")
    vecs = _normalise(np.array(emb.embed_documents(texts), dtype=np.float32))

    for k in sorted({min(k, len(names)) for k in args.k}):
        recalls = []
        misses: list[str] = []
        for label in labels:
            q = np.array(emb.embed_query(label["symptom"]), dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            sims = vecs @ q
            top = [names[i] for i in np.argsort(-sims)[:k]]
            expected = set(label["tools"])
            r = _recall(expected, top)
            recalls.append(r)
            if expected - set(top):
                misses.append(f"{label['symptom'][:50]!r} missed {sorted(expected - set(top))}")
        mean = float(np.nanmean(recalls)) if recalls else float("nan")
        print(f"  recall@{k:<3} = {mean:.2f}  over {len(labels)} symptoms")
        if k == max(min(kk, len(names)) for kk in args.k) and misses:
            print("    misses at largest K:")
            for m in misses:
                print(f"      - {m}")


# --- source #1: validated past investigations (held-out) ---------------------


def _proc_tools(proc: dict[str, Any]) -> set[str]:
    """Best-effort tool set for a procedure row from find_all_procedures."""
    code = proc.get("orchestration_code") or ""
    if code:
        try:
            from bamboo.agents.helpers.orchestration import (
                referenced_tool_names,  # noqa: PLC0415
            )

            return set(referenced_tool_names(code))
        except Exception:  # noqa: BLE001
            pass
    # fall back to a signature-derived tool_name like proc__a__b
    name = proc.get("procedure_name") or ""
    if name.startswith("proc__"):
        return set(name[len("proc__"):].split("__"))
    return set()


def _proc_text(proc: dict[str, Any]) -> str:
    triggers = " ".join(proc.get("trigger_signals") or [])
    summary = proc.get("code_summary") or proc.get("description") or ""
    return f"{triggers} {summary}".strip()


async def run_source1(args: argparse.Namespace, emb: Any) -> None:
    print("\n=== Source #1: validated past investigations (held-out) ===")
    from bamboo.database.graph_database_client import (
        GraphDatabaseClient,  # noqa: PLC0415
    )

    graph = GraphDatabaseClient()
    try:
        await graph.connect()
        procs = await graph.find_all_procedures(limit=500, include_tentative=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  graph DB unavailable ({exc}) — skipping")
        return
    finally:
        try:
            await graph.close()
        except Exception:  # noqa: BLE001
            pass

    usable = [(p, _proc_text(p), _proc_tools(p)) for p in procs]
    usable = [(p, t, tools) for (p, t, tools) in usable if t and tools]
    if len(usable) < 3:
        print(f"  only {len(usable)} usable procedures — corpus too small for a held-out read")
        return
    print(f"  {len(usable)} usable procedures; running leave-one-out …")

    texts = [t for (_p, t, _tools) in usable]
    vecs = _normalise(np.array(emb.embed_documents(texts), dtype=np.float32))
    top_n = args.top_procedures
    recalls = []
    for i, (_p, _t, expected) in enumerate(usable):
        mask = np.ones(len(usable), dtype=bool)
        mask[i] = False
        sims = vecs[mask] @ vecs[i]
        idx = [j for j in range(len(usable)) if j != i]
        ranked = [idx[j] for j in np.argsort(-sims)[:top_n]]
        retrieved_tools: set[str] = set()
        for j in ranked:
            retrieved_tools |= usable[j][2]
        recalls.append(_recall(expected, list(retrieved_tools)))
    mean = float(np.nanmean(recalls)) if recalls else float("nan")
    print(f"  held-out recall (union of top-{top_n} similar procs) = {mean:.2f}  over {len(usable)} procedures")


# --- entrypoint --------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", help="JSON file of {symptom, tools} pairs (source #2)")
    parser.add_argument("--catalogue", help="JSON file of tools; default = live MCP client")
    parser.add_argument("--source", choices=["1", "2", "both"], default="both")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 40], help="top-K values for source #2")
    parser.add_argument("--top-procedures", type=int, default=5, help="top-N similar procedures unioned (source #1)")
    args = parser.parse_args()

    emb = get_embeddings()
    if args.source in ("2", "both"):
        await run_source2(args, emb)
    if args.source in ("1", "both"):
        await run_source1(args, emb)


if __name__ == "__main__":
    asyncio.run(_main())
