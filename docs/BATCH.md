# Batch Analysis via Container

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

# 2. Stage the LLM model onto shared storage (mounted read-only at /models)
SHARED=/shared MODEL=llama3.2:3b deploy/batch/stage-model.sh

# 3. Build & stage the KB snapshot (mounted read-only at /kb) — see "Build the KB snapshot" below
```

### Build the KB snapshot

The batch container restores the KB from three files under `/kb`. Produce them from your
existing, populated Neo4j + Qdrant deployment and stage them to the shared path mounted
read-only at `/kb`:

| File | What it is | Restored by `run-analyze.sh` |
|------|------------|------------------------------|
| `graph_db.dump` | Neo4j offline dump, named for the batch `NEO4J_DATABASE` (default `graph_db`) | `neo4j-admin database load graph_db --from-path=/kb` |
| `qdrant_storage.tar.zst` (or `.tar.gz`) | tar of the Qdrant storage-dir **contents** | untarred into the storage path |
| `metadata.json` | embedding model/dimension + component versions | the embedding-consistency guard |

**Neo4j dump** (offline — the database must be stopped). With `neo4j-admin` on the deployment host:

```bash
neo4j-admin database dump neo4j --to-path=/tmp/kb   # writes <db>.dump
mv /tmp/kb/neo4j.dump /tmp/kb/graph_db.dump         # rename to the batch NEO4J_DATABASE
```

The dump can equally come from a version-matched `neo4j` container over the data dir, the Neo4j
Desktop **Dump** menu, or a managed-console export. Two rules hold regardless: the file must be
named `graph_db.dump` (`database load` finds it by target name), and the Neo4j version must match
the batch image's `NEO4J_VERSION` (dump/load is version-sensitive).

**Qdrant storage** (quiesce/stop the service for a consistent copy):

```bash
tar --use-compress-program=zstd -cf /tmp/kb/qdrant_storage.tar.zst -C <qdrant-storage-dir> .
```

**metadata.json** — values must match how the KB was populated:

```bash
cat >/tmp/kb/metadata.json <<'EOF'
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "neo4j_version": "5.26",
  "neo4j_database": "graph_db",
  "qdrant_version": "v1"
}
EOF
```

Then stage `/tmp/kb` to the shared filesystem path you mount read-only at `/kb`.

> Initial recipe — refine once the restore round-trip (`load` + untar + query) is verified on
> your deployment.

**Embedding consistency (critical):** the embedding model + dimension baked into the
image (`EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` build args) MUST match the values in the
snapshot's `metadata.json` — i.e. how the deployment that produced the dump was populated.
The entry script refuses to run on a mismatch; vector search silently degrades otherwise.

## Submitting a job

Stage task-data `*.json` files into an input dir, then:

```bash
SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out LLM_MODEL=llama3.2:3b \
  deploy/batch/submit.sh          # CPU queue
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
