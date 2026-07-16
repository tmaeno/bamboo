---
title: "Batch Analysis via Container"
---

Run `bamboo analyze` as a **self-contained batch job** on a compute slot that has no pre-deployed services,
and only offers non-root container execution — on either a CPU-only or a GPU queue, from one container.

## How it works

`analyze` needs Neo4j + Qdrant + an LLM + embeddings. It's agentic (the LLM drives
retrieval mid-run), so the knowledge base must live **on the node** next to the LLM —
you can't precompute it. The design:

- **One lean image, two targets** (`Dockerfile`):
  - `bamboo` — the app, configured by env, talks to external services (also the
    standalone Docker artifact).
  - `bamboo-batch-analyze` — `FROM bamboo`, adds Neo4j + Qdrant + Ollama (copied from
    official images) + the entry script. Converted to a `.sif`.
- **Large, changing artifacts stay out of the image**, staged on the shared filesystem
  and mounted read-only: the **Ollama model** (`/models`) and the **KB snapshot**
  (`/kb`). Update them without rebuilding the image.
- **`bamboo batch-analyze`** processes many tasks per container invocation so the
  costly service + model startup is paid **once**, not per task.


## One-time setup (on a networked host)

```bash
# 1. Build & publish the images (or use CI: .github/workflows/build-images.yml)
docker build --target bamboo               -t bamboo .
docker build --target bamboo-batch-analyze -t bamboo-batch-analyze .
apptainer build bamboo-batch-analyze.sif docker-daemon://bamboo-batch-analyze:latest

# 2. Stage the LLM model onto shared storage (mounted read-only at /models).
#    MODEL defaults to LLM_MODEL from your local .env (Ollama config) — this host needs
#    the bamboo repo + .env importable. Pass MODEL to override.
SHARED=/shared deploy/batch/stage-model.sh
# or, explicitly: SHARED=/shared MODEL=llama3.2:3b deploy/batch/stage-model.sh

# 3. Build & stage the KB snapshot (mounted read-only at /kb) — see "Build the KB snapshot" below
```

### Build the KB snapshot

The batch container restores the KB from three files under `/kb`. Produce `qdrant.snapshot`
and `metadata.json` with **`bamboo dump-kb`** — it reads your Qdrant/Neo4j/embedding config from
the same `.env` your populated deployment uses, so the snapshot matches it exactly. The Neo4j
graph dump stays a separate offline step (it needs a stopped DB + data-dir access, which a bolt
URL can't provide). Stage all three files to the shared path mounted read-only at `/kb`.

| File | What it is | Restored by `run-analyze.sh` |
|------|------------|------------------------------|
| `graph_db.dump` | Neo4j offline dump, named for the batch `NEO4J_DATABASE` (default `graph_db`) | `neo4j-admin database load graph_db --from-path=/kb` |
| `qdrant.snapshot` | Qdrant Snapshot-API export of the collection | recovered on startup (`qdrant --snapshot <file>:<collection>`) |
| `metadata.json` | embedding model/dimension, collection, Neo4j/Qdrant versions | the KB metadata guard (embeddings + versions) |

**Qdrant snapshot + metadata.json** — from a host that can reach your Qdrant (and Neo4j, for the
version stamp):

```bash
bamboo dump-kb --out /tmp/kb   # writes qdrant.snapshot + metadata.json from your .env
```

`dump-kb` uses the Qdrant **Snapshot API** (only `QDRANT_URL`/collection/api-key — never Qdrant's
on-disk storage dir), so it works against a local Docker Qdrant or a managed instance, with no
service stop. `metadata.json` is filled entirely from real state: `embedding_model`/
`embedding_dimension`/`neo4j_database`/`qdrant_collection` from your config, plus
`neo4j_version`/`qdrant_version` read live from the servers.

**Neo4j dump** (offline — the database must be stopped). `dump-kb` prints this command with your
DB name filled in; with `neo4j-admin` on the deployment host:

```bash
neo4j-admin database dump graph_db --to-path=/tmp/kb   # writes graph_db.dump
```

The dump can equally come from a version-matched `neo4j` container over the data dir, the Neo4j
Desktop **Dump** menu, or a managed-console export. Two rules hold regardless: the file must be
named to match the batch `NEO4J_DATABASE` (default `graph_db`, i.e. `graph_db.dump` — rename it if
your source DB differs), and the Neo4j version must match the batch image's `NEO4J_VERSION`
(dump/load is version-sensitive).

Then stage `/tmp/kb` to the shared filesystem path you mount read-only at `/kb`.

> Initial recipe — refine once the restore round-trip (`load` + snapshot-recover + query) is
> verified on your deployment.

**Metadata guard (critical):** `run-analyze.sh` refuses to run on a mismatch between the
snapshot's `metadata.json` and the batch image. The embedding model + dimension
(`EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` build args) MUST match how the KB was populated — vector
search silently degrades otherwise. The Neo4j version must match the image's `NEO4J_VERSION`
(dump/load is version-strict → hard fail); the Qdrant major version is checked best-effort against
the image's `QDRANT_VERSION` (a mismatch only warns, since the boot-time snapshot recover is the
real gate). `bamboo dump-kb` stamps all these values for you.

## Submitting a job

Stage task-data `*.json` files into an input dir, then:

```bash
SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out \
  deploy/batch/submit.sh          # CPU queue (LLM_MODEL from your .env; set it to override)
# GPU queue: also export USE_GPU=1   (adds --nv; Ollama auto-detects the GPU)
```

One result JSON is written per task to `OUT_DIR`; a failing task gets a
`*.error.json` sidecar and the job exits non-zero (the batch still completes the
others). A SLURM wrapper example is in `deploy/batch/submit.sh`.

### Live PanDA fetch (optional)

Fully offline, stage task JSON and use `--input-dir`. If PanDA egress is granted you
can fetch live with `--task-id`; pass the OIDC token via the **`file:` form** so it
never lands in env/argv:

```bash
PANDA_TOKEN_FILE=~/.panda/token PANDA_AUTH_VO=<vo> deploy/batch/submit.sh
```

OIDC ID tokens are short-lived — a queued job may outlive its token. Mount `~/.panda`
for in-job refresh (needs IdP egress too) or use a long-lived X.509 proxy.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Two-target image (`bamboo`, `bamboo-batch-analyze`) |
| `deploy/batch/run-analyze.sh` | In-container entry: boot stack, restore KB, run batch, tear down |
| `deploy/batch/stage-model.sh` | Pull the Ollama model into shared storage |
| `deploy/batch/submit.sh` | Example Apptainer submission (CPU/GPU) |
| `.github/workflows/build-images.yml` | CI: build + push images, optional `.sif` |
