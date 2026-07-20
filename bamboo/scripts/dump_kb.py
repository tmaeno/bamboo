#!/usr/bin/env python
"""Export the env-derivable KB snapshot artifacts for the batch pipeline.

``bamboo dump-kb --out DIR`` connects to the SAME Qdrant / Neo4j the local bamboo
run uses (resolved through ``get_settings()`` / ``.env``) and writes:

  - ``qdrant-<collection>.snapshot``  — one Qdrant Snapshot-API export per collection.
    Every collection is included, so the doc-navigator's ``panda_docs`` /
    ``panda_docs_meta`` travel alongside the KB collection.
  - ``metadata.json``    — embedding model/dimension + collection list + live server versions

Because it goes through the Snapshot API it needs only ``QDRANT_URL`` / api-key — never
the Qdrant server's on-disk storage dir — so it works against a local Docker Qdrant or a
managed one alike. The batch container recovers each snapshot on startup (a repeated
``qdrant --snapshot <file>:<collection>``; see ``deploy/batch/run-analyze.sh``).

The Neo4j graph dump stays a separate **offline** ``neo4j-admin database dump`` step
(it needs a stopped DB + data-dir access, which a bolt URL cannot provide); this
command reads the live Neo4j version for the manifest and prints the exact dump command
with the DB name filled in from config.

See ``website/src/content/docs/guides/batch.md``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console

from bamboo.utils.logging import setup_logging

console = Console()


async def _neo4j_version(settings) -> str:
    """Read the Neo4j server version over bolt (best-effort; ``""`` on failure).

    Mirrors the driver construction in
    :class:`~bamboo.database.backends.neo4j_backend.Neo4jBackend` so the version
    reflects exactly the deployment the local run talks to.
    """
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        return ""
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            # Select the kernel component explicitly: dbms.components() can return more
            # than one row (extra components), which would both warn on .single() and
            # risk returning a non-kernel version.
            result = await session.run(
                "CALL dbms.components() YIELD name, versions "
                "WHERE name = 'Neo4j Kernel' "
                "RETURN versions[0] AS v"
            )
            record = await result.single()
            return str(record["v"]) if record and record.get("v") else ""
    except Exception as exc:  # noqa: BLE001 — version is informational metadata
        console.print(f"[yellow]⚠  could not read Neo4j version: {exc}[/yellow]")
        return ""
    finally:
        await driver.close()


async def _run(out_dir: str) -> int:
    setup_logging()
    from bamboo.config import get_settings
    from bamboo.database.backends.qdrant_backend import QdrantBackend
    from bamboo.database.factory import get_vector_backend

    settings = get_settings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if settings.embeddings_provider != "local":
        console.print(
            f"[yellow]⚠  EMBEDDINGS_PROVIDER={settings.embeddings_provider!r} — the batch "
            "image bakes local embeddings, so this snapshot's metadata will fail the "
            "restore-time consistency guard. Populate the KB with EMBEDDINGS_PROVIDER=local "
            "for batch use.[/yellow]"
        )

    # --- Qdrant snapshot via the Snapshot API (URL/collection only — no storage dir) ---
    backend = get_vector_backend()
    if not isinstance(backend, QdrantBackend):
        raise click.ClickException(
            f"Vector backend {type(backend).__name__} has no snapshot export; "
            "the batch KB dump currently supports Qdrant only."
        )
    try:
        qmeta = await backend.export_all_snapshots(str(out))
    finally:
        await backend.close()
    collections = qmeta.get("qdrant_collections", [])
    for entry in collections:
        console.print(
            f"[green]✓[/green] Qdrant snapshot ({entry['collection']}) "
            f"→ {out / entry['snapshot_file']}"
        )

    # --- Neo4j server version (read live over bolt; the dump itself stays offline) ---
    neo4j_version = await _neo4j_version(settings)

    # --- metadata.json — every field from real state, no hand-typed values ---
    metadata = {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "neo4j_database": settings.neo4j_database,
        "neo4j_version": neo4j_version,
        # Primary KB collection — used by the restore-time embedding-consistency guard.
        "qdrant_collection": qmeta.get(
            "primary_collection", settings.qdrant_collection_name
        ),
        # Every collection + its snapshot file — the batch restore recovers them all.
        "qdrant_collections": collections,
        "qdrant_version": qmeta.get("qdrant_version", ""),
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    console.print(f"[green]✓[/green] metadata.json → {meta_path}")

    # --- Neo4j dump — offline step the operator runs against a STOPPED database ---
    console.print(
        "\n[bold]Next: dump Neo4j offline[/bold] (stop the database first), then stage "
        f"the file as [cyan]{settings.neo4j_database}.dump[/cyan] under {out}/:\n"
        f"  neo4j-admin database dump {settings.neo4j_database} --to-path={out}"
    )
    return 0


@click.command("dump-kb")
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write qdrant-<collection>.snapshot files + metadata.json (created if missing).",
)
def main(out_dir):
    """Export the env-derivable KB snapshot artifacts (Qdrant snapshot + metadata.json).

    Reads Qdrant / Neo4j / embedding config from the same .env the local bamboo run
    uses, so the staged snapshot matches your populated deployment. The Neo4j graph
    dump remains a separate offline ``neo4j-admin database dump`` step (printed at the
    end).

    \b
      bamboo dump-kb --out /tmp/kb
    """
    rc = asyncio.run(_run(out_dir))
    sys.exit(rc)


if __name__ == "__main__":
    main()
