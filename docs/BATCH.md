# Batch Analysis via Container

Run `bamboo analyze` as a **self-contained batch job** on a compute slot that has no pre-deployed services,
and only offers non-root container execution — on either a CPU-only or a GPU queue, from one container.

## How it works

`analyze` needs Neo4j + Qdrant + an LLM + embeddings. It's agentic (the LLM drives
retrieval mid-run), so the knowledge base must live **on the node** next to the LLM —
you can't precompute it. The design:

- **One lean image, two targets** ([Dockerfile](../Dockerfile)):
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

# 3. Build & stage the KB snapshot (mounted read-only at /kb)
#    Edit the populate step inside the script for your inputs.
SHARED=/shared deploy/batch/build-kb-snapshot.sh
```

**Embedding consistency (critical):** the embedding model + dimension baked into the
image (`EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` build args) MUST match those used in
`build-kb-snapshot.sh`. The snapshot's `metadata.json` records them and the entry
script refuses to run on a mismatch — vector search silently degrades otherwise.

## Submitting a job

Stage task-data `*.json` files into an input dir, then:

```bash
SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out LLM_MODEL=llama3.2:3b \
  deploy/batch/submit.sh          # CPU queue
# GPU queue: also export USE_GPU=1   (adds --nv; Ollama auto-detects the GPU)
```

One result JSON is written per task to `OUT_DIR`; a failing task gets a
`*.error.json` sidecar and the job exits non-zero (the batch still completes the
others). A SLURM wrapper example is in [submit.sh](batch/submit.sh).

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
| [Dockerfile](../Dockerfile) | Two-target image (`bamboo`, `bamboo-batch-analyze`) |
| [deploy/batch/run-analyze.sh](batch/run-analyze.sh) | In-container entry: boot stack, restore KB, run batch, tear down |
| [deploy/batch/build-kb-snapshot.sh](batch/build-kb-snapshot.sh) | Build + stage the KB snapshot |
| [deploy/batch/stage-model.sh](batch/stage-model.sh) | Pull the Ollama model into shared storage |
| [deploy/batch/submit.sh](batch/submit.sh) | Example Apptainer submission (CPU/GPU) |
| [.github/workflows/build-images.yml](../.github/workflows/build-images.yml) | CI: build + push images, optional `.sif` |
