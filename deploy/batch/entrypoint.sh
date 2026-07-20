#!/usr/bin/env bash
# entrypoint.sh — entry point for the air-gapped batch container (Image 2).
#
# Boots Neo4j + Qdrant + Ollama on localhost from read-only shared-FS mounts,
# restores the KB into node-local scratch, runs `bamboo batch-analyze` over the
# staged tasks, then tears everything down. Designed to run rootless under
# Apptainer (you are an arbitrary uid; the .sif and /models /kb are read-only).
#
# Subcommands (the boot phase and the batch phase are separable so you can stop
# mid-way and poke at `bamboo` by hand inside one container session):
#
#   entrypoint.sh                 full run: setup → batch → teardown (default; any
#                                 extra args are forwarded to `bamboo batch-analyze`).
#   entrypoint.sh run [args]      explicit alias of the default.
#   entrypoint.sh setup           boot the stack + restore the KB, then LEAVE IT
#                                 RUNNING. Persists a state env-file (see below).
#   entrypoint.sh batch [args]    run `bamboo batch-analyze` against an already
#                                 booted stack (sources the state env-file).
#   entrypoint.sh teardown        kill the services and remove the scratch dir.
#   entrypoint.sh shell           boot the stack, then drop into an interactive
#                                 shell with the env exported; teardown on exit.
#
# Interactive debugging (ONE container session — services live in that session's
# process namespace, so `setup`/`batch`/`teardown` must share a single `docker run`):
#   docker run -it … bamboo-batch-analyze shell          # boot + interactive bash
#   # or drive the pieces yourself:
#   docker run -it --entrypoint bash … bamboo-batch-analyze
#     $ /opt/bamboo/entrypoint.sh setup
#     $ source "${BAMBOO_WORK:-/tmp}/bamboo-batch.env"    # bamboo now sees the stack
#     $ bamboo analyze … / bamboo investigate … / bamboo verify
#     $ /opt/bamboo/entrypoint.sh batch
#     $ /opt/bamboo/entrypoint.sh teardown
#
# Mounts (see deploy/batch/submit.sh):
#   /in         (ro)  directory of task-data *.json files
#   /out        (rw)  one result JSON per task is written here
#   /kb         (ro)  KB snapshot: <db>.dump + qdrant-<collection>.snapshot (one per
#                     collection) + metadata.json
#   /models     (ro)  Ollama models dir (OLLAMA_MODELS) + bamboo-model.json manifest
#   /embeddings (ro)  local HF cache (HF_HOME): embedding model + optional reranker + manifest
#   /work       (rw)  node-local scratch (optional; falls back to $TMPDIR)
#
# ⚠ SCAFFOLD — UNVERIFIED. Grep "VERIFY:" for spots the Phase 0 spike must confirm
#   (rootless Neo4j wiring, admin subcommand syntax, readiness probes).
set -euo pipefail

# Read a top-level string field from a JSON file; empty if the file/field is absent. Used to
# derive model identities from the staged artifacts (stage manifests + KB metadata.json).
_json_get() {
  python - "$1" "$2" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2], "") or "")
except Exception:
    pass
PY
}

# --------------------------------------------------------------------------- #
# Config (override via env / APPTAINERENV_*)
#
# Only pure *defaults* live at top level so every subcommand resolves the same
# paths and the same state-file location. Everything that allocates a resource
# (scratch dir, free ports) or derives from the staged KB happens in do_setup.
# --------------------------------------------------------------------------- #
IN_DIR="${BAMBOO_IN:-/in}"
OUT_DIR="${BAMBOO_OUT:-/out}"
KB_DIR="${BAMBOO_KB:-/kb}"
WORK_ROOT="${BAMBOO_WORK:-${TMPDIR:-/tmp}}"

# Where do_setup persists the derived env + service PIDs so `batch`/`teardown`/an
# interactive shell can pick the stack up. Deterministic given WORK_ROOT, so all
# subcommands in one session agree on the path without being told it.
BAMBOO_STATE_FILE="${BAMBOO_STATE_FILE:-${WORK_ROOT%/}/bamboo-batch.env}"

export OLLAMA_MODELS="${OLLAMA_MODELS:-/models}"
export HF_HOME="${HF_HOME:-/embeddings}"
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export EMBEDDINGS_PROVIDER="${EMBEDDINGS_PROVIDER:-local}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Freeze the PanDA doc index: never re-fetch from GitHub or re-summarize with the LLM;
# load the pre-built index staged in the panda_docs / panda_docs_meta Qdrant collections
# as-is. An explicit env still wins if a run ever needs to rebuild in-container.
export DOC_INDEX_FREEZE="${DOC_INDEX_FREEZE:-1}"
# NEO4J_DATABASE is derived from the KB metadata.json below (falling back to the
# built-in `neo4j`) so the load target always matches the dump the KB was built
# with. An explicit NEO4J_DATABASE env still wins.
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
# Print the tail of a service log so a readiness failure is diagnosable *before*
# teardown's `rm -rf "${WORK}"` wipes it.
dump_tail() { log "---- last 60 lines of $1 ----"; tail -n 60 "$1" 2>/dev/null | sed 's/^/  | /' >&2; log "---- end $1 ----"; }

# --------------------------------------------------------------------------- #
# Scratch + teardown (must survive SIGKILL/walltime: kill the process group,
# remove scratch). Services run in their own process groups (setsid) so they
# outlive do_setup returning; teardown kills those groups.
#
# In-process (full run / shell) WORK and PIDS are already set. Standalone
# `teardown` loads them from the state env-file first.
# --------------------------------------------------------------------------- #
WORK=""
PIDS=()
teardown() {
  local rc=$?
  if [[ -z "${WORK}" && -f "${BAMBOO_STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${BAMBOO_STATE_FILE}" || true
    WORK="${BAMBOO_WORK_ACTIVE:-}"
    read -ra PIDS <<<"${BAMBOO_SERVICE_PIDS:-}"
  fi
  log "tearing down (rc=$rc)…"
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && { kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true; }
  done
  wait 2>/dev/null || true
  [[ -n "${WORK}" ]] && rm -rf "${WORK}" 2>/dev/null || true
  rm -f "${BAMBOO_STATE_FILE}" 2>/dev/null || true
  log "done."
}

# --------------------------------------------------------------------------- #
# Persist the derived env + service PIDs so a later `batch`/`teardown` or an
# interactive shell can source the live stack. `printf %q` escaping keeps values
# re-sourceable. Only emit the optional model vars when they were actually set.
# --------------------------------------------------------------------------- #
persist_env() {
  {
    printf '# bamboo batch stack — written by entrypoint.sh setup; source to reach the live stack.\n'
    local k
    for k in NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD NEO4J_DATABASE \
             QDRANT_URL QDRANT_COLLECTION_NAME \
             OLLAMA_BASE_URL OLLAMA_MODELS \
             LLM_PROVIDER LLM_MODEL \
             EMBEDDINGS_PROVIDER EMBEDDING_MODEL EMBEDDING_DIMENSION RERANKER_MODEL \
             HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DOC_INDEX_FREEZE; do
      [[ -n "${!k:-}" ]] && printf 'export %s=%q\n' "$k" "${!k}"
    done
    # Internal bookkeeping for teardown (not exported into bamboo's env).
    printf 'BAMBOO_WORK_ACTIVE=%q\n' "${WORK}"
    printf 'BAMBOO_SERVICE_PIDS=%q\n' "${PIDS[*]:-}"
  } >"${BAMBOO_STATE_FILE}"
  log "wrote stack state to ${BAMBOO_STATE_FILE}"
}

# Source the state env-file when the stack env isn't already in this process
# (i.e. a standalone `batch` after a separate `setup`). No-op for the full run.
load_state() {
  [[ -n "${NEO4J_URI:-}" ]] && return 0
  [[ -f "${BAMBOO_STATE_FILE}" ]] || die "no stack state at ${BAMBOO_STATE_FILE} — run 'entrypoint.sh setup' first"
  # shellcheck disable=SC1090
  source "${BAMBOO_STATE_FILE}"
}

# --------------------------------------------------------------------------- #
# do_setup — boot the localhost stack and restore the KB into scratch.
#
# On entry it guards a partial boot (teardown on EXIT/INT/TERM); on success it
# clears that trap so the caller owns teardown (a standalone `setup` leaves the
# stack running; `run`/`shell` re-arm the trap themselves).
# --------------------------------------------------------------------------- #
do_setup() {
  trap 'teardown' EXIT INT TERM

  WORK="$(mktemp -d "${WORK_ROOT%/}/bamboo.XXXXXX")"
  mkdir -p "${WORK}/neo4j/data" "${WORK}/neo4j/logs" "${WORK}/neo4j/run" \
           "${WORK}/neo4j/conf" "${WORK}/qdrant/storage" "${WORK}/ollama" "${OUT_DIR}"

  # ------------------------------------------------------------------------- #
  # Free-port allocation — Apptainer shares the host netns, so co-scheduled jobs
  # would otherwise collide on 7687/6333/11434.
  # ------------------------------------------------------------------------- #
  free_port() { python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }
  BOLT_PORT="$(free_port)"
  QDRANT_PORT="$(free_port)"
  OLLAMA_PORT="$(free_port)"

  export NEO4J_URI="bolt://127.0.0.1:${BOLT_PORT}"
  export QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"
  export OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
  log "ports: bolt=${BOLT_PORT} qdrant=${QDRANT_PORT} ollama=${OLLAMA_PORT}"

  # LLM_MODEL: an explicit env wins; otherwise derive it from the staged /models manifest
  # (bamboo stage-model writes bamboo-model.json), so the model choice travels with the staged
  # files and nothing needs to be set at submit time.
  : "${LLM_MODEL:=$(_json_get "${OLLAMA_MODELS}/bamboo-model.json" llm_model)}"
  : "${LLM_MODEL:?no LLM_MODEL and no ${OLLAMA_MODELS}/bamboo-model.json — stage a model with 'bamboo stage-model'}"
  export LLM_MODEL

  # ------------------------------------------------------------------------- #
  # KB metadata guard. Two different rigor levels on purpose:
  #   • Embedding model — HARD FAIL. Correctness-critical: a mismatched model means query
  #     vectors won't match the stored vectors (silent garbage / dimension errors), and it's a
  #     staging mistake, not engine drift. The model/dim themselves are DERIVED from
  #     metadata.json below (the KB is their source of truth); this only cross-checks the staged
  #     /embeddings manifest so a mis-staged dir fails here with a clear message rather than as a
  #     cryptic offline HF load error later.
  #   • Neo4j / Qdrant engine versions — WARN only. The real gates are the boot-time
  #     `neo4j-admin database load` and Qdrant `--snapshot` recover, which run under
  #     `set -euo pipefail` and fail the job loudly on a genuine incompatibility. We can't verify
  #     the exact cross-version behavior here (this file is a scaffold), so we surface drift as a
  #     warning rather than block on a version-string that may well be compatible.
  # NEO4J_VERSION/QDRANT_VERSION are baked into the image (Dockerfile); the KB values are what
  # `bamboo dump-kb` stamps into metadata.json.
  # ------------------------------------------------------------------------- #
  local META EMB_MANIFEST
  META="${KB_DIR}/metadata.json"
  EMB_MANIFEST="${HF_HOME}/bamboo-embeddings.json"
  if [[ -f "${META}" ]]; then
    python - "$META" "$EMB_MANIFEST" <<'PY' || die "staged embeddings do not match the KB — re-stage the embedding model"
import json, os, sys
meta = json.load(open(sys.argv[1]))
try:
    man = json.load(open(sys.argv[2]))
except Exception:
    man = {}

def _diff(want, got):
    return bool(want) and bool(got) and str(want) != str(got)

# Engine-version comparison only: the image bakes the Docker tag form (`v1.18.3`)
# while the engine/metadata reports it bare (`1.18.3`), so drop a leading `v` on
# both sides before comparing — otherwise the drift warning mis-fires even when the
# versions match. NOT applied to model names (a leading 'v' there is significant).
def _diffver(want, got):
    strip = lambda s: str(s)[1:] if str(s)[:1] == "v" else str(s)
    return _diff(strip(want), strip(got))

# Embedding model — correctness-critical (query vs stored vectors must agree). HARD FAIL.
if _diff(meta.get("embedding_model", ""), man.get("embedding_model", "")):
    print(f"staged embedding model {man.get('embedding_model')!r} != KB {meta.get('embedding_model')!r}", file=sys.stderr); sys.exit(1)

# Engine versions — the boot-time `neo4j-admin database load` and Qdrant `--snapshot` recover
# are the real gates (they fail loudly on a genuine incompatibility). We can't verify exact
# cross-version behavior here, so WARN on drift rather than block.
if _diffver(os.environ.get("NEO4J_VERSION", ""), meta.get("neo4j_version", "")):
    print(f"WARNING: neo4j version drift: image={os.environ.get('NEO4J_VERSION')} kb={meta.get('neo4j_version')} — dump load may fail at boot", file=sys.stderr)
if _diffver(os.environ.get("QDRANT_VERSION", ""), meta.get("qdrant_version", "")):
    print(f"WARNING: qdrant version drift: image={os.environ.get('QDRANT_VERSION')} kb={meta.get('qdrant_version')} — snapshot recover may fail at boot", file=sys.stderr)
PY
    log "KB metadata checks passed"
  else
    log "WARNING: no ${META} — skipping KB metadata checks (recommend stamping it)"
  fi

  # ------------------------------------------------------------------------- #
  # Derive model identities from the staged artifacts (an explicit env value wins for each):
  #   EMBEDDING_MODEL / EMBEDDING_DIMENSION / QDRANT_COLLECTION_NAME ← KB metadata.json
  #   RERANKER_MODEL                                                 ← /embeddings manifest
  # Deriving the collection from the snapshot keeps the populate-time and batch-time
  # collections — and the --snapshot target — matched. HF resolves the on-disk model files via
  # HF_HOME=/embeddings; a missing model then fails loudly offline.
  # ------------------------------------------------------------------------- #
  : "${EMBEDDING_MODEL:=$(_json_get "${META}" embedding_model)}"
  : "${EMBEDDING_DIMENSION:=$(_json_get "${META}" embedding_dimension)}"
  : "${QDRANT_COLLECTION_NAME:=$(_json_get "${META}" qdrant_collection)}"
  : "${QDRANT_COLLECTION_NAME:=bamboo_knowledge}"
  : "${NEO4J_DATABASE:=$(_json_get "${META}" neo4j_database)}"
  : "${NEO4J_DATABASE:=neo4j}"
  : "${RERANKER_MODEL:=$(_json_get "${EMB_MANIFEST}" reranker_model)}"
  export QDRANT_COLLECTION_NAME
  export NEO4J_DATABASE
  [[ -n "${EMBEDDING_MODEL:-}" ]]    && export EMBEDDING_MODEL
  [[ -n "${EMBEDDING_DIMENSION:-}" ]] && export EMBEDDING_DIMENSION
  [[ -n "${RERANKER_MODEL:-}" ]]    && export RERANKER_MODEL

  # Sanity: scratch has room (best-effort; df may be absent in minimal images).
  if command -v df >/dev/null 2>&1; then
    local avail_kb
    avail_kb="$(df -Pk "${WORK}" | awk 'NR==2{print $4}')"
    log "scratch free: $(( avail_kb / 1024 )) MB at ${WORK}"
  fi

  # ------------------------------------------------------------------------- #
  # Restore KB into writable scratch
  # ------------------------------------------------------------------------- #
  log "restoring Neo4j dump…"
  # VERIFY: Neo4j 5 admin syntax + dump filename (<db>.dump). See the /kb contract in docs/BATCH.md.
  # The Neo4j 5 CLI (`neo4j`, `neo4j-admin`) reads its conf dir from $NEO4J_CONF —
  # export it so both the offline `database load` and `neo4j console` pick up the
  # scratch neo4j.conf below (without it they use $NEO4J_HOME/conf and write to the
  # default /opt/neo4j/data, an image symlink -> /data that isn't writable here).
  export NEO4J_CONF="${WORK}/neo4j/conf"
  cp -r "${NEO4J_HOME}/conf/." "${NEO4J_CONF}/" 2>/dev/null || true
  # The stock image conf ships an active `server.http.enabled=true`; appending our
  # own value below would make Neo4j 5 reject the file ("declared multiple times").
  # Strip any active declaration of every key we override, then append ours. The
  # regex is anchored at line start (with optional leading whitespace), so commented
  # `#server.…` lines are left untouched.
  sed -i -E '/^[[:space:]]*(server\.directories\.data|server\.directories\.logs|server\.directories\.run|server\.directories\.transaction\.logs\.root|server\.bolt\.listen_address|server\.http\.enabled|dbms\.security\.procedures\.unrestricted)[[:space:]]*=/d' "${NEO4J_CONF}/neo4j.conf"
  cat >>"${NEO4J_CONF}/neo4j.conf" <<EOF
server.directories.data=${WORK}/neo4j/data
server.directories.logs=${WORK}/neo4j/logs
server.directories.run=${WORK}/neo4j/run
server.directories.transaction.logs.root=${WORK}/neo4j/data/transactions
server.bolt.listen_address=:${BOLT_PORT}
server.http.enabled=false
dbms.security.procedures.unrestricted=apoc.*
EOF

  neo4j-admin dbms set-initial-password "${NEO4J_PASSWORD}" >/dev/null 2>&1 || true
  neo4j-admin database load "${NEO4J_DATABASE}" \
    --from-path="${KB_DIR}" --overwrite-destination=true

  log "locating Qdrant snapshot(s)…"
  # The KB ships one Qdrant Snapshot-API file per collection (produced by `bamboo dump-kb`,
  # see docs/BATCH.md), listed in metadata.json under "qdrant_collections". Qdrant recovers
  # each into the fresh storage dir on startup via a repeated --snapshot flag below — no
  # extraction needed. Building ALL collections (not just the KB one) is what carries the
  # doc-navigator's panda_docs / panda_docs_meta into the container.
  QDRANT_SNAPSHOT_ARGS=()
  local snap_file coll snap_path
  QDRANT_COLL_COUNT=0
  while IFS=$'\t' read -r snap_file coll; do
    [[ -n "${snap_file}" && -n "${coll}" ]] || continue
    snap_path="${KB_DIR}/${snap_file}"
    [[ -f "${snap_path}" ]] || die "missing Qdrant snapshot ${snap_path} (collection '${coll}')"
    QDRANT_SNAPSHOT_ARGS+=(--snapshot "${snap_path}:${coll}")
    QDRANT_COLL_COUNT=$((QDRANT_COLL_COUNT + 1))
  done < <(python - "${META}" <<'PY' 2>/dev/null || true
import json, sys
try:
    meta = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for e in meta.get("qdrant_collections", []) or []:
    f, c = e.get("snapshot_file"), e.get("collection")
    if f and c:
        print(f"{f}\t{c}")
PY
)
  # Legacy fallback: a KB dumped before multi-collection support ships a single
  # qdrant.snapshot with no "qdrant_collections" list.
  if [[ ${QDRANT_COLL_COUNT} -eq 0 ]]; then
    local legacy="${KB_DIR}/qdrant.snapshot"
    [[ -f "${legacy}" ]] || die "no Qdrant snapshots in ${KB_DIR} (neither qdrant_collections nor qdrant.snapshot)"
    QDRANT_SNAPSHOT_ARGS+=(--snapshot "${legacy}:${QDRANT_COLLECTION_NAME}")
    QDRANT_COLL_COUNT=1
    log "using legacy single-collection snapshot for '${QDRANT_COLLECTION_NAME}'"
  fi

  # ------------------------------------------------------------------------- #
  # Launch services (each in its own process group via setsid for clean teardown)
  # ------------------------------------------------------------------------- #
  log "starting neo4j…"
  setsid neo4j console >"${WORK}/neo4j/console.log" 2>&1 & PIDS+=($!)

  log "starting qdrant (recovering ${QDRANT_COLL_COUNT} collection snapshot(s))…"
  QDRANT__SERVICE__HTTP_PORT="${QDRANT_PORT}" \
  QDRANT__STORAGE__STORAGE_PATH="${WORK}/qdrant/storage" \
    setsid qdrant "${QDRANT_SNAPSHOT_ARGS[@]}" \
    >"${WORK}/qdrant/qdrant.log" 2>&1 & PIDS+=($!)

  log "starting ollama…"
  OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" HOME="${WORK}/ollama" \
    setsid ollama serve >"${WORK}/ollama/ollama.log" 2>&1 & PIDS+=($!)

  # ------------------------------------------------------------------------- #
  # Readiness (fail fast on timeout)
  # ------------------------------------------------------------------------- #
  wait_tcp()  { for _ in $(seq "${2:-120}"); do (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0; sleep 1; done; return 1; }
  wait_http() { for _ in $(seq "${3:-120}"); do curl -fsS "$2" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

  log "waiting for services (bolt=${BOLT_PORT} qdrant=${QDRANT_PORT} ollama=${OLLAMA_PORT})…"
  wait_tcp  "${BOLT_PORT}" 180 || { dump_tail "${WORK}/neo4j/console.log"; die "neo4j not ready"; }
  wait_http "${QDRANT_PORT}" "http://127.0.0.1:${QDRANT_PORT}/readyz" 120 || { dump_tail "${WORK}/qdrant/qdrant.log"; die "qdrant not ready"; }
  wait_http "${OLLAMA_PORT}" "http://127.0.0.1:${OLLAMA_PORT}/api/tags" 120 || { dump_tail "${WORK}/ollama/ollama.log"; die "ollama not ready"; }
  log "all services ready"

  persist_env
  # Hand teardown control back to the caller: a standalone `setup` leaves the stack
  # running; `run`/`shell` re-arm their own EXIT trap. A mid-setup failure above
  # instead trips the guard trap and cleans up the half-booted stack.
  trap - EXIT INT TERM
}

# --------------------------------------------------------------------------- #
# do_batch — run the batch (deps + in-process models warm across every task).
# --------------------------------------------------------------------------- #
do_batch() {
  log "running bamboo batch-analyze…"
  set +e
  bamboo batch-analyze --input-dir "${IN_DIR}" --output-dir "${OUT_DIR}" "${@}"
  local rc=$?
  set -e
  log "batch-analyze exited rc=${rc}"
  return "${rc}"
}

# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
cmd_run() {                     # default: boot → batch → teardown (backward compatible)
  do_setup
  trap 'teardown' EXIT INT TERM
  local rc=0
  do_batch "$@" || rc=$?
  exit "${rc}"                  # teardown fires via the EXIT trap
}

cmd_setup() {                   # boot the stack and leave it running
  do_setup
  log "stack is up. To use it:"
  log "  source ${BAMBOO_STATE_FILE}   # then run bamboo … against the live stack"
  log "  entrypoint.sh batch [args]    # run bamboo batch-analyze"
  log "  entrypoint.sh teardown        # kill services + remove scratch"
}

cmd_batch() {                   # run batch-analyze against an already booted stack
  load_state
  do_batch "$@"
}

cmd_teardown() {                # kill services + remove scratch (loads state if needed)
  teardown
}

cmd_shell() {                   # boot the stack, drop into an interactive shell, teardown on exit
  do_setup
  trap 'teardown' EXIT INT TERM
  log "stack is up — dropping into an interactive shell. Type 'exit' to tear down."
  log "  (bamboo already sees the stack via the exported env; e.g. 'bamboo verify')"
  bash -i || true               # child, not exec, so the EXIT trap runs teardown on return
}

# --------------------------------------------------------------------------- #
# Dispatch. Only the literal tokens below are subcommands; anything else (e.g.
# `--task-id 123`) falls through to the default full run with all args forwarded
# to `bamboo batch-analyze` — preserving the pre-subcommand invocation contract.
#
# Consume the subcommand into $sub, then shift it off *only if* one was given
# (a bare `shift` with no positional params fails and would trip `set -e`, which
# is the common CMD [] / no-args case).
# --------------------------------------------------------------------------- #
sub="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
case "${sub}" in
  run)      cmd_run "$@" ;;
  setup)    cmd_setup "$@" ;;
  batch)    cmd_batch "$@" ;;
  teardown) cmd_teardown "$@" ;;
  shell)    cmd_shell "$@" ;;
  *)        cmd_run "${sub}" "$@" ;;   # not a subcommand → forward all args to batch-analyze
esac
