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
  - `bamboo-batch` — `FROM bamboo`, adds Neo4j + Qdrant + Ollama (copied from
    official images) + the entry script. Converted to a `.sif`.
- **Large, changing artifacts stay out of the image**, staged on the shared filesystem
  and mounted read-only: the **Ollama model** (`/models`), the **local embedding model**
  (`/embeddings`), and the **KB snapshot** (`/kb`). Update them without rebuilding the image.
  Where a job is handed one archive instead of a shared filesystem, the same three arrive
  as a **sandbox tarball** (`BAMBOO_SANDBOX`) — see "Single-tarball sandbox" below.
- **`bamboo batch-analyze`** processes many tasks per container invocation so the
  costly service + model startup is paid **once**, not per task.


## One-time setup (on a networked host)

```bash
# 1. Build & publish the images (or use CI: .github/workflows/build-images.yml)
docker build --target bamboo       -t bamboo .
docker build --target bamboo-batch -t bamboo-batch .
apptainer build bamboo-batch.sif docker-daemon://bamboo-batch:latest

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

The batch container restores the KB from the files under `/kb`. Produce the Qdrant snapshots
and `metadata.json` with **`bamboo dump-kb`** — it reads your Qdrant/Neo4j/embedding config from
the same `.env` your populated deployment uses, so the snapshot matches it exactly. `dump-kb`
exports **every** Qdrant collection (one snapshot file each), so auxiliary collections — notably
the doc-navigator's `panda_docs` / `panda_docs_meta` — travel with the KB, not just the main one.
The Neo4j graph dump stays a separate offline step (it needs a stopped DB + data-dir access, which
a bolt URL can't provide). Stage all of them to the shared path mounted read-only at `/kb`.

| File | What it is | Restored by `entrypoint.sh` |
|------|------------|------------------------------|
| `neo4j.dump` | Neo4j offline dump, named for the batch `NEO4J_DATABASE` (default `neo4j`) | `neo4j-admin database load neo4j --from-path=/kb` |
| `qdrant-<collection>.snapshot` | Qdrant Snapshot-API export, one file per collection (all collections, incl. the doc-navigator's `panda_docs` / `panda_docs_meta`) | each recovered on startup (a repeated `qdrant --snapshot <file>:<collection>`) |
| `metadata.json` | embedding model/dimension, the collection list + primary collection, Neo4j/Qdrant versions | the KB metadata guard (embeddings + versions) |

The doc-navigator index is served **as-is** in the container: `entrypoint.sh` sets
`DOC_INDEX_FREEZE=1`, so the navigator never reaches out to GitHub or the LLM to rebuild doc
summaries — it loads the staged `panda_docs` / `panda_docs_meta` collections directly. (Set
`DOC_INDEX_FREEZE=0` to allow an in-container rebuild if you ever need one.)

**Qdrant snapshot + metadata.json** — from a host that can reach your Qdrant (and Neo4j, for the
version stamp):

```bash
bamboo dump-kb --out /tmp/kb   # writes qdrant-<collection>.snapshot (one per collection) + metadata.json from your .env
```

`dump-kb` uses the Qdrant **Snapshot API** (only `QDRANT_URL`/api-key — never Qdrant's on-disk
storage dir), so it works against a local Docker Qdrant or a managed instance, with no service
stop; it enumerates and snapshots every collection. `metadata.json` is filled entirely from real
state: `embedding_model`/`embedding_dimension`/`neo4j_database` from your config, the
`qdrant_collections` list (each collection + its snapshot file) plus the primary `qdrant_collection`,
and `neo4j_version`/`qdrant_version` read live from the servers.

**Neo4j dump** (offline — the database must be stopped). `dump-kb` prints this command with your
DB name filled in; with `neo4j-admin` on the deployment host:

```bash
# If you are running Neo4j Desktop, the DBMS may use the default block store format
# (Enterprise Edition). Neo4j Community Edition does not support the block format,
# so you must first convert the database store format to aligned before dumping it.
# To convert the format, navigate to the DBMS directory, then run:
neo4j-admin database migrate neo4j --to-format=aligned

neo4j-admin database dump neo4j --to-path=/tmp/kb   # writes neo4j.dump
```

The dump can equally come from a version-matched `neo4j` container over the data dir, the Neo4j
Desktop **Dump** menu, or a managed-console export. Two rules hold regardless: the file must be
named to match the batch `NEO4J_DATABASE` (default `neo4j`, i.e. `neo4j.dump` — rename it if
your source DB differs), and the Neo4j version must match or lower than the batch image's `NEO4J_VERSION`
(Downgrade is not supported).

Then stage `/tmp/kb` to the shared filesystem path you mount read-only at `/kb`.

**Metadata guard (critical):** `entrypoint.sh` derives the embedding model + dimension
(`EMBEDDING_MODEL`/`EMBEDDING_DIMENSION`) straight from the snapshot's `metadata.json`, so query
embeddings match how the KB was populated *by construction* — no silent vector degradation. It
cross-checks that against the staged `/embeddings` manifest and fails early on a mismatch; a
`/embeddings` dir missing that model then fails loudly at offline load. `RERANKER_MODEL` is
derived from the `/embeddings` manifest and `LLM_MODEL` from the `/models` manifest, so no model
name is set at submit time (export one to override for a run). Still *compared* against the image:
the Neo4j version must match `NEO4J_VERSION` (dump/load is version-strict → hard fail), and the
Qdrant major version is checked best-effort against `QDRANT_VERSION` (a mismatch only warns, since
the boot-time snapshot recover is the real gate). `bamboo dump-kb` stamps all these values for you.

## Single-tarball sandbox (alternative to the three mounts)

Some deployments hand a job **one archive** rather than three shared-filesystem
directories. Set **`BAMBOO_SANDBOX`** to a tarball whose top level holds the same
component directories and the container reads them from there instead:

```bash
$ tar tzf sandbox.tgz
models/…            # what `bamboo stage-model` produces      → OLLAMA_MODELS
kb/…                # what `bamboo dump-kb` produces (+ the .dump) → the KB snapshot
embeddings/…        # what `bamboo stage-embeddings` produces  → HF_HOME
in/…                # optional: the task-data *.json           → --input-dir
```

Build one by pointing the same staging commands at a common directory, then tarring its
*contents* (`-C sandbox .`) — nothing new to learn, only different `--out` paths:

```bash
bamboo stage-model      --out sandbox/models
bamboo stage-embeddings --out sandbox/embeddings
cp -a /tmp/kb           sandbox/kb        # the dir built in "Build the KB snapshot" above
tar czf sandbox.tgz -C sandbox .
```

Running it is the same workload either way, so the commands live with the rest: locally under
"Testing locally" below, on a cluster under "Submitting a job".

What to know about it:

- **Opt-in.** With `BAMBOO_SANDBOX` unset nothing changes; a tarball that happens to sit
  next to the job is never picked up.
- **Per-component.** Only the components actually in the archive are redirected — a `kb/`-only
  tarball composes with `/models` + `/embeddings` still mounted. Every component is logged
  either way (`sandbox: kb -> …` / `sandbox: no models/ — keeping OLLAMA_MODELS=/models`), so a
  partial sandbox that didn't cover what you meant is visible in the job log. An archive with
  none of the four names is rejected before anything boots.
- **It costs scratch space.** The archive is expanded under `BAMBOO_WORK` next to the restored
  Neo4j/Qdrant data, so mount node-local scratch at `/work` — otherwise it lands in the
  container's writable layer. Teardown removes it with the rest of the scratch dir. Pass an
  **already-extracted directory** instead of a tarball and it is read in place, with no copy.
- **Model derivation is unchanged**, just relative to the sandbox: `LLM_MODEL` from
  `models/bamboo-model.json`, `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` from `kb/metadata.json`,
  `RERANKER_MODEL` from `embeddings/bamboo-embeddings.json`. The metadata guard above applies
  as-is.
- **Formats:** `.tgz`/`.tar.gz`, `.tar.bz2`, `.tar.xz`, plain `.tar` — detected from the archive,
  not its name. **Not `.tar.zst`**: the `Dockerfile` purges `zstd` after installing Ollama.
  A single wrapper directory (`tar czf sandbox.tgz sandbox/`) is tolerated.

## Testing locally

Before queuing on a cluster, dress-rehearse the **exact container** on your workstation — it runs
the same ENTRYPOINT (`entrypoint.sh`) SLURM will, so it catches the batch-specific failures (KB
`metadata.json` mismatch, Neo4j/Qdrant version skew, a bad snapshot, a mis-staged model) where
they're cheap to debug. Reuse the `bamboo-batch` image, staged model, and KB snapshot you
built above; point the mounts at your local paths and run it under Docker:

```bash
docker run --rm \
  -v $PWD/in:/in:ro \
  -v $PWD/out:/out \
  -v ${SHARED:-/shared}/bamboo/ollama:/models:ro \
  -v ${SHARED:-/shared}/bamboo/embeddings:/embeddings:ro \
  -v ${SHARED:-/shared}/bamboo/kb:/kb:ro \
  bamboo-batch batch-analyze  # models derived from the staged manifests/KB; add --gpus all for GPU
# override a model for a one-off run with e.g. -e LLM_MODEL=qwen3.6
# portable run: add  -v $PWD/.env:/app/.env:ro  to inject keys/settings (see "Portable mode" below)
```

The `batch-analyze` subcommand runs the full sequence — the KB metadata guard,
`neo4j-admin database load`, Qdrant snapshot recover, `ollama serve`, then `bamboo batch-analyze`
over `/in` — and tears the stack down on exit. Success is one result JSON per task in `./out`; a
`*.error.json` sidecar plus a non-zero exit flags a failing task. This is the same round-trip
`submit.sh` performs, minus the Apptainer launcher.

Rehearsing a **sandbox** job (see "Single-tarball sandbox" above) is the same command with the
mounts collapsed into the archive — no `/models`, `/embeddings`, `/kb` or `/in`, `/out` the only
writable mount, and `/work` the scratch the archive is expanded into:

```bash
docker run --rm \
  -v $PWD/sandbox.tgz:/sandbox.tgz:ro -e BAMBOO_SANDBOX=/sandbox.tgz \
  -v $PWD/work:/work -e BAMBOO_WORK=/work \
  -v $PWD/out:/out \
  bamboo-batch batch-analyze
```

The task data comes from the archive's `in/` here, and the `sandbox: in -> …` line in the log is
what confirms it: `batch-analyze` builds its argv *after* the boot that expands the sandbox, so
`--input-dir` follows the archive. Let it — passing `--input-dir /in` yourself pins the literal
mount and quietly ignores the sandbox's `in/`.

Name the workload: **with no subcommand the container prints usage and exits non-zero** rather than
guessing one. `bamboo batch-analyze` is not the only batch command bamboo has — `batch-populate` is
another — so it gets no implied position; every other command is reached with `exec` (below).
`bamboo-batch help` lists the full set (`batch-analyze`, `exec`, `shell`, `setup`, `teardown`, `help`)
without booting anything.

> A Docker run won't reproduce every Apptainer detail — rootless arbitrary-uid execution, the
> shared host netns (why `entrypoint.sh` allocates free ports), and `--nv` GPU wiring. Treat it as
> a functional dress rehearsal, and still do one real cluster smoke test before relying on the queue.

### Interactive debugging

To poke at `bamboo` by hand against the real staged KB — the fastest way to validate the
scaffolded pieces or to debug RAG/agent behavior — use `shell`, or split the boot phase off with
`setup`/`teardown`. The simplest entry boots the stack and drops you into a shell (add
`-it` for a TTY); the exported env already points `bamboo` at the live services, and the stack is
torn down when you `exit`:

```bash
docker run -it --rm \
  -v $PWD/in:/in:ro -v $PWD/out:/out \
  -v ${SHARED:-/shared}/bamboo/ollama:/models:ro \
  -v ${SHARED:-/shared}/bamboo/embeddings:/embeddings:ro \
  -v ${SHARED:-/shared}/bamboo/kb:/kb:ro \
  bamboo-batch shell
# optional: add  -v $PWD/.env:/app/.env:ro  so `bamboo verify` finds a .env (✓) — see "Portable mode"
# inside the shell:
#   bamboo verify
#   bamboo analyze --task-id 123 …      # single task against the live stack
#   bamboo batch-analyze --input-dir /in --output-dir /out
#   exit                                # tears the stack down
```

#### One-shot commands (`exec`)

When you want *one* command run against the staged KB — from a script, a CI step, or just without a
TTY — use `exec` instead of `shell`. It boots the stack, runs the command, tears the stack down, and
**exits with the command's status** (`shell` is interactive-only and discards it). Arguments are
passed to the command verbatim, so quoted free text survives intact:

```bash
docker run --rm \
  -v $PWD/.env:/app/.env:ro \
  -v $PWD/in:/in:ro -v $PWD/out:/out \
  -v ${SHARED:-/shared}/bamboo/ollama:/models:ro \
  -v ${SHARED:-/shared}/bamboo/embeddings:/embeddings:ro \
  -v ${SHARED:-/shared}/bamboo/kb:/kb:ro \
  bamboo-batch exec bamboo verify

# free-text arguments are safe — argv is not re-parsed by a shell
docker run --rm … bamboo-batch exec bamboo investigate "why did task 123 fail"
# pipes, redirects and && need an explicit shell:
docker run --rm … bamboo-batch exec bash -lc 'bamboo verify | tee /out/verify.log'

# a non-analyze batch command — note -v …/kb:/kb:rw (see below)
docker run --rm … -v ${SHARED:-/shared}/bamboo/kb:/kb:rw \
  bamboo-batch exec bash -lc 'bamboo batch-populate --drafts /out/drafts --yes && bamboo dump-kb --out /kb'
```

`exec` is how every bamboo command other than `batch-analyze` runs in this image — `batch-populate`,
`dump-kb`, `investigate`, `verify`. One thing to know for the write path: the container restores the KB
into node-local scratch and **teardown deletes it**, so a command that modifies the KB only leaves a
trace if you snapshot it back out (`bamboo dump-kb --out /kb`) — which needs `/kb` mounted `:rw`
instead of the read-only mount the analyze path uses.

Setup chatter goes to stderr, so the command's stdout stays clean and pipeable
(`… exec bamboo verify > verify.txt` captures only `bamboo`'s output). A command that does not exist
is rejected before the stack boots, so a typo costs a second rather than a few minutes. The trade-off
is cost: each `exec` pays a full stack boot (KB load, snapshot recover, `ollama serve`), so several
commands in a row are cheaper inside one `shell` session. Passing arguments to `shell` is an error
that points you here.

Or drive the pieces yourself (same single container session — the services live in that
session's process namespace, so `setup`/`teardown` **must share one `docker run`**):

```bash
docker run -it --rm … --entrypoint bash bamboo-batch
  $ /opt/bamboo/entrypoint.sh setup          # boot the stack, leave it running
  $ source /work/bamboo-batch.env            # or ${TMPDIR:-/tmp}/bamboo-batch.env without -v …:/work
  $ bamboo investigate …                     # bamboo now sees the live stack
  $ bamboo batch-analyze --input-dir /in --output-dir /out
  $ /opt/bamboo/entrypoint.sh teardown       # kill services + remove scratch
```

`setup` writes the derived env (service URLs, model identities) and service PIDs to a state file
(`$BAMBOO_WORK/bamboo-batch.env`, default `/work/bamboo-batch.env` under `submit.sh`). Sourcing it is
what points `bamboo` at the live stack; `teardown` reads it back for the service PIDs, so you never
re-type the ports or model names.

### Portable mode: injecting runtime config

The same image doubles as a portable execution environment: mount your own `.env` at `/app/.env`
(`/app` is the container's working directory) to inject credentials and settings at launch —
no rebuild, nothing baked into the image:

```bash
docker run --rm \
  -v $PWD/.env:/app/.env:ro \
  -v $PWD/in:/in:ro -v $PWD/out:/out \
  -v ${SHARED:-/shared}/bamboo/ollama:/models:ro \
  -v ${SHARED:-/shared}/bamboo/embeddings:/embeddings:ro \
  -v ${SHARED:-/shared}/bamboo/kb:/kb:ro \
  bamboo-batch batch-analyze   # + your read-only .env mounted at /app/.env
```

`bamboo` loads `/app/.env` itself (`config._find_env_file`, so `bamboo verify` reports
`✓ .env loaded from: /app/.env` instead of the otherwise-benign `✗ .env file not found`).

**What a mounted `.env` does and does not control.** Precedence is: values passed with `-e` win,
then the image's baked batch defaults and the vars the container derives from the staged KB/model.
So `.env` is for **keys / tokens / settings**, not for provider or model/DB selection:

- **Honored** — `PANDA_*`, `SSL_CERT_FILE`, `LOG_LEVEL`, `MATTERMOST_*`, `MCP_SERVERS_CONFIG`,
  the tool-selection knobs, etc.
- **Ignored (stay as the container decides)** — `LLM_PROVIDER`/`EMBEDDINGS_PROVIDER` and the
  `*_OFFLINE` flags (baked to the bundled `ollama`/`local` stack), and the KB/manifest-derived
  `LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `NEO4J_DATABASE`,
  `QDRANT_COLLECTION_NAME`, and the Neo4j/Qdrant/Ollama service URLs.

This is deliberate: it keeps the bundled stack and KB integrity intact even if you mount a `.env`
copied straight from `.env.example` (whose `LLM_PROVIDER=openai`, `EMBEDDING_MODEL=…`, `LLM_MODEL=…`
lines are simply ignored here). To point the batch image at a cloud LLM or external databases
instead of the bundled stack, use the standalone `bamboo` image with `--env-file .env`,
which is configured entirely by env — not this batch image.

## Submitting a job

Stage task-data `*.json` files into an input dir, then:

```bash
SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out \
  deploy/batch/submit.sh          # CPU queue (models derived from staged artifacts; export LLM_MODEL to override)
# GPU queue: also export USE_GPU=1   (adds --nv; Ollama auto-detects the GPU)
```

For a **sandbox** job (see "Single-tarball sandbox" above), `SANDBOX=` replaces `SHARED=` and
`IN_DIR=` — the archive carries the models, embeddings, KB and task data, and `submit.sh` does
the binding for you:

```bash
SANDBOX=$PWD/sandbox.tgz OUT_DIR=$PWD/out deploy/batch/submit.sh
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
| `Dockerfile` | Two-target image (`bamboo`, `bamboo-batch`) |
| `deploy/batch/entrypoint.sh` | In-container entry: boot stack, restore KB, run the workload, tear down (subcommands: `batch-analyze`/`exec`/`shell`/`setup`/`teardown`/`help`) |
| `bamboo stage-model` / `bamboo stage-embeddings` | Stage the LLM / embedding (+ reranker) models onto shared storage |
| `deploy/batch/submit.sh` | Example Apptainer submission (CPU/GPU; `SANDBOX=` for a single-tarball job) |
| `.github/workflows/build-images.yml` | CI: build + push images, optional `.sif` |
