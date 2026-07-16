#!/usr/bin/env bash
# run-analyze.sh — entry point for the air-gapped batch container (Image 2).
#
# Boots Neo4j + Qdrant + Ollama on localhost from read-only shared-FS mounts,
# restores the KB into node-local scratch, runs `bamboo batch-analyze` over the
# staged tasks, then tears everything down. Designed to run rootless under
# Apptainer (you are an arbitrary uid; the .sif and /models /kb are read-only).
#
# Mounts (see deploy/batch/submit.sh):
#   /in      (ro)  directory of task-data *.json files
#   /out     (rw)  one result JSON per task is written here
#   /kb      (ro)  KB snapshot: <db>.dump + qdrant.snapshot + metadata.json
#   /models  (ro)  Ollama models dir (OLLAMA_MODELS)
#   /work    (rw)  node-local scratch (optional; falls back to $TMPDIR)
#
# ⚠ SCAFFOLD — UNVERIFIED. Grep "VERIFY:" for spots the Phase 0 spike must confirm
#   (rootless Neo4j wiring, admin subcommand syntax, readiness probes).
set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via env / APPTAINERENV_*)
# --------------------------------------------------------------------------- #
IN_DIR="${BAMBOO_IN:-/in}"
OUT_DIR="${BAMBOO_OUT:-/out}"
KB_DIR="${BAMBOO_KB:-/kb}"
WORK_ROOT="${BAMBOO_WORK:-${TMPDIR:-/tmp}}"
: "${LLM_MODEL:?set LLM_MODEL to a model present under /models (OLLAMA_MODELS)}"

export OLLAMA_MODELS="${OLLAMA_MODELS:-/models}"
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export EMBEDDINGS_PROVIDER="${EMBEDDINGS_PROVIDER:-local}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-graph_db}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-graph_db}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
export QDRANT_COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-bamboo_knowledge}"

log() { printf '[run-analyze] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --------------------------------------------------------------------------- #
# Scratch + teardown (must survive SIGKILL/walltime: kill the process group,
# remove scratch). We run services in our own process group and kill it on exit.
# --------------------------------------------------------------------------- #
WORK="$(mktemp -d "${WORK_ROOT%/}/bamboo.XXXXXX")"
PIDS=()
cleanup() {
  local rc=$?
  log "tearing down (rc=$rc)…"
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  rm -rf "${WORK}" 2>/dev/null || true
  log "done."
}
trap cleanup EXIT INT TERM

mkdir -p "${WORK}/neo4j/data" "${WORK}/neo4j/logs" "${WORK}/neo4j/run" \
         "${WORK}/neo4j/conf" "${WORK}/qdrant/storage" "${WORK}/ollama" "${OUT_DIR}"

# --------------------------------------------------------------------------- #
# Free-port allocation — Apptainer shares the host netns, so co-scheduled jobs
# would otherwise collide on 7687/6333/11434.
# --------------------------------------------------------------------------- #
free_port() { python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }
BOLT_PORT="$(free_port)"
QDRANT_PORT="$(free_port)"
OLLAMA_PORT="$(free_port)"

export NEO4J_URI="bolt://127.0.0.1:${BOLT_PORT}"
export QDRANT_URL="http://127.0.0.1:${QDRANT_PORT}"
export OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
log "ports: bolt=${BOLT_PORT} qdrant=${QDRANT_PORT} ollama=${OLLAMA_PORT}"

# --------------------------------------------------------------------------- #
# KB metadata guard — refuse to run on an incompatible snapshot. Embedding model/dim
# mismatch silently corrupts vector search; Neo4j dump/load is version-strict; Qdrant
# snapshot recover is major-version sensitive (image pins the rolling 'v1' tag).
# EMBEDDING_MODEL/DIMENSION and NEO4J_VERSION/QDRANT_VERSION are baked into the image
# (Dockerfile) and compared against the values stamped by `bamboo dump-kb`.
# --------------------------------------------------------------------------- #
META="${KB_DIR}/metadata.json"
if [[ -f "${META}" ]]; then
  python - "$META" <<'PY' || die "KB metadata mismatch — rebuild the snapshot or image"
import json, os, sys
meta = json.load(open(sys.argv[1]))

def _diff(want, got):
    return bool(want) and bool(got) and str(want) != str(got)

# Embedding model + dimension — hard requirement (vector search silently degrades).
if _diff(os.environ.get("EMBEDDING_MODEL", ""), meta.get("embedding_model", "")):
    print(f"embedding model mismatch: image={os.environ.get('EMBEDDING_MODEL')} kb={meta.get('embedding_model')}", file=sys.stderr); sys.exit(1)
if _diff(os.environ.get("EMBEDDING_DIMENSION", ""), meta.get("embedding_dimension", "")):
    print(f"embedding dim mismatch: image={os.environ.get('EMBEDDING_DIMENSION')} kb={meta.get('embedding_dimension')}", file=sys.stderr); sys.exit(1)

# Neo4j — dump/load is version-strict → hard fail on mismatch.
if _diff(os.environ.get("NEO4J_VERSION", ""), meta.get("neo4j_version", "")):
    print(f"neo4j version mismatch: image={os.environ.get('NEO4J_VERSION')} kb={meta.get('neo4j_version')}", file=sys.stderr); sys.exit(1)

# Qdrant — image pins the rolling 'v1' tag, so compare MAJOR only and WARN (the boot-
# time snapshot recover is the hard gate).
def _major(v):
    return str(v).lstrip("v").split(".")[0] if v else ""
wq, gq = _major(os.environ.get("QDRANT_VERSION", "")), _major(meta.get("qdrant_version", ""))
if wq and gq and wq != gq:
    print(f"WARNING: qdrant major mismatch: image={wq} kb={gq} — snapshot recover may fail", file=sys.stderr)
PY
  log "KB metadata checks passed"
else
  log "WARNING: no ${META} — skipping KB metadata checks (recommend stamping it)"
fi

# Collection name comes from the snapshot's metadata (falls back to the env default),
# so the populate-time and batch-time collections — and the --snapshot target — match.
if [[ -f "${META}" ]]; then
  COLL="$(python - "$META" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1])).get("qdrant_collection", "") or "")
except Exception:
    pass
PY
)"
  [[ -n "${COLL}" ]] && export QDRANT_COLLECTION_NAME="${COLL}"
fi

# Sanity: scratch has room (best-effort; df may be absent in minimal images).
if command -v df >/dev/null 2>&1; then
  avail_kb="$(df -Pk "${WORK}" | awk 'NR==2{print $4}')"
  log "scratch free: $(( avail_kb / 1024 )) MB at ${WORK}"
fi

# --------------------------------------------------------------------------- #
# Restore KB into writable scratch
# --------------------------------------------------------------------------- #
log "restoring Neo4j dump…"
# VERIFY: Neo4j 5 admin syntax + dump filename (<db>.dump). See the /kb contract in docs/BATCH.md.
NEO4J_CONF="${WORK}/neo4j/conf"
cp -r "${NEO4J_HOME}/conf/." "${NEO4J_CONF}/" 2>/dev/null || true
cat >>"${NEO4J_CONF}/neo4j.conf" <<EOF
server.directories.data=${WORK}/neo4j/data
server.directories.logs=${WORK}/neo4j/logs
server.directories.run=${WORK}/neo4j/run
server.bolt.listen_address=:${BOLT_PORT}
server.http.enabled=false
dbms.security.procedures.unrestricted=apoc.*
EOF
export NEO4J_CONF_DIR="${NEO4J_CONF}"   # VERIFY: env name neo4j honours for conf dir

neo4j-admin dbms set-initial-password "${NEO4J_PASSWORD}" >/dev/null 2>&1 || true
neo4j-admin database load "${NEO4J_DATABASE}" \
  --from-path="${KB_DIR}" --overwrite-destination=true

log "locating Qdrant snapshot…"
# The KB snapshot ships a Qdrant Snapshot-API file (produced by `bamboo dump-kb`, see
# docs/BATCH.md); Qdrant recovers it into the fresh storage dir on startup via the
# --snapshot flag below — no extraction needed here.
QDRANT_SNAPSHOT="${KB_DIR}/qdrant.snapshot"
[[ -f "${QDRANT_SNAPSHOT}" ]] || die "no Qdrant snapshot at ${QDRANT_SNAPSHOT}"

# --------------------------------------------------------------------------- #
# Launch services (each in its own process group via setsid for clean teardown)
# --------------------------------------------------------------------------- #
log "starting neo4j…"
setsid neo4j console >"${WORK}/neo4j/console.log" 2>&1 & PIDS+=($!)

log "starting qdrant (recovering snapshot for '${QDRANT_COLLECTION_NAME}')…"
QDRANT__SERVICE__HTTP_PORT="${QDRANT_PORT}" \
QDRANT__STORAGE__STORAGE_PATH="${WORK}/qdrant/storage" \
  setsid qdrant --snapshot "${QDRANT_SNAPSHOT}:${QDRANT_COLLECTION_NAME}" \
  >"${WORK}/qdrant/qdrant.log" 2>&1 & PIDS+=($!)

log "starting ollama…"
OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" HOME="${WORK}/ollama" \
  setsid ollama serve >"${WORK}/ollama/ollama.log" 2>&1 & PIDS+=($!)

# --------------------------------------------------------------------------- #
# Readiness (fail fast on timeout)
# --------------------------------------------------------------------------- #
wait_tcp()  { for _ in $(seq "${2:-120}"); do (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && return 0; sleep 1; done; return 1; }
wait_http() { for _ in $(seq "${3:-120}"); do curl -fsS "$2" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

wait_tcp  "${BOLT_PORT}" 180                                  || die "neo4j not ready (see ${WORK}/neo4j/console.log)"
wait_http "${QDRANT_PORT}" "http://127.0.0.1:${QDRANT_PORT}/readyz" 120 || die "qdrant not ready"
wait_http "${OLLAMA_PORT}" "http://127.0.0.1:${OLLAMA_PORT}/api/tags" 120 || die "ollama not ready"
log "all services ready"

# --------------------------------------------------------------------------- #
# Run the batch (deps + in-process models warm across every task)
# --------------------------------------------------------------------------- #
log "running bamboo batch-analyze…"
set +e
bamboo batch-analyze --input-dir "${IN_DIR}" --output-dir "${OUT_DIR}" "${@}"
rc=$?
set -e
log "batch-analyze exited rc=${rc}"
exit "${rc}"
