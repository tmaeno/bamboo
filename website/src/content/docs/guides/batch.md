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
  and mounted read-only: the **Ollama model** (`/models`), the **local embedding model**
  (`/embeddings`), and the **KB snapshot** (`/kb`). Update them without rebuilding the image.
- **`bamboo batch-analyze`** processes many tasks per container invocation so the
  costly service + model startup is paid **once**, not per task.


## One-time setup (on a networked host)

```bash
# 1. Build & publish the images (or use CI: .github/workflows/build-images.yml)
docker build --target bamboo               -t bamboo .
docker build --target bamboo-batch-analyze -t bamboo-batch-analyze .
apptainer build bamboo-batch-analyze.sif docker-daemon://bamboo-batch-analyze:latest

# 2. Stage the LLM model onto shared storage (mounted read-only at /models).
#    Ships with `pip install bamboo`. --model defaults to LLM_MODEL from your config
#    (Ollama), else qwen3.6; out dir = $MODELS_OUT or ${SHARED:-/shared}/bamboo/ollama.
#    Writes a bamboo-model.json manifest so LLM_MODEL is derived at run time — you do NOT
#    set it at submit.
SHARED=/shared bamboo stage-model
# explicit model:  SHARED=/shared bamboo stage-model --model qwen3.6

# 3. Stage the local embedding model onto shared storage (mounted read-only at /embeddings).
#    Needs the `[local]` extra (sentence-transformers). --model defaults to EMBEDDING_MODEL
#    from your config when EMBEDDINGS_PROVIDER=local, else all-MiniLM-L6-v2; out dir =
#    $EMBEDDINGS_OUT or ${SHARED:-/shared}/bamboo/embeddings. If RERANKER_MODEL is set in your
#    config, that cross-encoder is staged alongside the embedding model too.
SHARED=/shared bamboo stage-embeddings
# explicit model:  SHARED=/shared bamboo stage-embeddings --model all-mpnet-base-v2

# 4. Build & stage the KB snapshot (mounted read-only at /kb) — see "Build the KB snapshot" below
```

### Build the KB snapshot

The batch container restores the KB from three files under `/kb`. Produce `qdrant.snapshot`
and `metadata.json` with **`bamboo dump-kb`** — it reads your Qdrant/Neo4j/embedding config from
the same `.env` your populated deployment uses, so the snapshot matches it exactly. The Neo4j
graph dump stays a separate offline step (it needs a stopped DB + data-dir access, which a bolt
URL can't provide). Stage all three files to the shared path mounted read-only at `/kb`.

| File | What it is | Restored by `run-analyze.sh` |
|------|------------|------------------------------|
| `neo4j.dump` | Neo4j offline dump, named for the batch `NEO4J_DATABASE` (default `neo4j`) | `neo4j-admin database load neo4j --from-path=/kb` |
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
# If you are running Neo4j Desktop, the DBMS may use the default block store format
# (Enterprise Edition). Neo4j Community Edition does not support the block format,
# so you must first convert the database store format to aligned before dumping it.
# First navigate to the DBMS directory, then run:
neo4j-admin database migrate neo4j --to-format=aligned

neo4j-admin database dump neo4j --to-path=/tmp/kb   # writes neo4j.dump
```

The dump can equally come from a version-matched `neo4j` container over the data dir, the Neo4j
Desktop **Dump** menu, or a managed-console export. Two rules hold regardless: the file must be
named to match the batch `NEO4J_DATABASE` (default `neo4j`, i.e. `neo4j.dump` — rename it if
your source DB differs), and the Neo4j version must match or lower than the batch image's `NEO4J_VERSION`
(Downgrade is not supported).

Then stage `/tmp/kb` to the shared filesystem path you mount read-only at `/kb`.

**Metadata guard (critical):** `run-analyze.sh` derives the embedding model + dimension
(`EMBEDDING_MODEL`/`EMBEDDING_DIMENSION`) straight from the snapshot's `metadata.json`, so query
embeddings match how the KB was populated *by construction* — no silent vector degradation. It
cross-checks that against the staged `/embeddings` manifest and fails early on a mismatch; a
`/embeddings` dir missing that model then fails loudly at offline load. `RERANKER_MODEL` is
derived from the `/embeddings` manifest and `LLM_MODEL` from the `/models` manifest, so no model
name is set at submit time (export one to override for a run). Still *compared* against the image:
the Neo4j version must match `NEO4J_VERSION` (dump/load is version-strict → hard fail), and the
Qdrant major version is checked best-effort against `QDRANT_VERSION` (a mismatch only warns, since
the boot-time snapshot recover is the real gate). `bamboo dump-kb` stamps all these values for you.

## Testing locally

Before queuing on a cluster, dress-rehearse the **exact container** on your workstation — it runs
the same ENTRYPOINT (`run-analyze.sh`) SLURM will, so it catches the batch-specific failures (KB
`metadata.json` mismatch, Neo4j/Qdrant version skew, a bad snapshot, a mis-staged model) where
they're cheap to debug. Reuse the `bamboo-batch-analyze` image, staged model, and KB snapshot you
built above; point the mounts at your local paths and run it under Docker:

```bash
docker run --rm \
  -v $PWD/in:/in:ro \
  -v $PWD/out:/out \
  -v ${SHARED:-/shared}/bamboo/ollama:/models:ro \
  -v ${SHARED:-/shared}/bamboo/embeddings:/embeddings:ro \
  -v ${SHARED:-/shared}/bamboo/kb:/kb:ro \
  bamboo-batch-analyze      # models derived from the staged manifests/KB; add --gpus all for GPU
# override a model for a one-off run with e.g. -e LLM_MODEL=qwen3.6
```

`run-analyze.sh` runs its full sequence — the KB metadata guard, `neo4j-admin database load`,
Qdrant snapshot recover, `ollama serve`, then `bamboo batch-analyze` over `/in` — and tears the
stack down on exit. Success is one result JSON per task in `./out`; a `*.error.json` sidecar plus a
non-zero exit flags a failing task. This is the same round-trip `submit.sh` performs, minus the
Apptainer launcher.

> A Docker run won't reproduce every Apptainer detail — rootless arbitrary-uid execution, the
> shared host netns (why `run-analyze.sh` allocates free ports), and `--nv` GPU wiring. Treat it as
> a functional dress rehearsal, and still do one real cluster smoke test before relying on the queue.

## Submitting a job

Stage task-data `*.json` files into an input dir, then:

```bash
SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out \
  deploy/batch/submit.sh          # CPU queue (models derived from staged artifacts; export LLM_MODEL to override)
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
| `bamboo stage-model` / `bamboo stage-embeddings` | Stage the LLM / embedding (+ reranker) models onto shared storage |
| `deploy/batch/submit.sh` | Example Apptainer submission (CPU/GPU) |
| `.github/workflows/build-images.yml` | CI: build + push images, optional `.sif` |
