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
#   /kb      (ro)  KB snapshot: <db>.dump + qdrant_storage.tar.zst + metadata.json
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
# Embedding-consistency guard — refuse to run if the baked embedding model/dim
# differs from what built the KB (else vector search silently returns garbage).
# --------------------------------------------------------------------------- #
META="${KB_DIR}/metadata.json"
if [[ -f "${META}" ]]; then
  python - "$META" <<'PY' || die "KB embedding metadata mismatch — rebuild the snapshot or image"
import json, os, sys
meta = json.load(open(sys.argv[1]))
want_model = os.environ.get("EMBEDDING_MODEL", "")
want_dim = str(os.environ.get("EMBEDDING_DIMENSION", ""))
got_model = str(meta.get("embedding_model", ""))
got_dim = str(meta.get("embedding_dimension", ""))
if want_model and got_model and want_model != got_model:
    print(f"model mismatch: image={want_model} kb={got_model}", file=sys.stderr); sys.exit(1)
if want_dim and got_dim and want_dim != got_dim:
    print(f"dim mismatch: image={want_dim} kb={got_dim}", file=sys.stderr); sys.exit(1)
PY
  log "embedding-consistency check passed"
else
  log "WARNING: no ${META} — skipping embedding-consistency check (recommend stamping it)"
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
# VERIFY: Neo4j 5 admin syntax + dump filename (<db>.dump) produced by build-kb-snapshot.sh.
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

log "restoring Qdrant storage…"
# build-kb-snapshot.sh ships the whole storage dir as a tarball (simplest robust path).
if   [[ -f "${KB_DIR}/qdrant_storage.tar.zst" ]]; then tar --use-compress-program=unzstd -xf "${KB_DIR}/qdrant_storage.tar.zst" -C "${WORK}/qdrant/storage"
elif [[ -f "${KB_DIR}/qdrant_storage.tar.gz"  ]]; then tar -xzf "${KB_DIR}/qdrant_storage.tar.gz" -C "${WORK}/qdrant/storage"
else die "no qdrant storage tarball under ${KB_DIR}"; fi

# --------------------------------------------------------------------------- #
# Launch services (each in its own process group via setsid for clean teardown)
# --------------------------------------------------------------------------- #
log "starting neo4j…"
setsid neo4j console >"${WORK}/neo4j/console.log" 2>&1 & PIDS+=($!)

log "starting qdrant…"
QDRANT__SERVICE__HTTP_PORT="${QDRANT_PORT}" \
QDRANT__STORAGE__STORAGE_PATH="${WORK}/qdrant/storage" \
  setsid qdrant >"${WORK}/qdrant/qdrant.log" 2>&1 & PIDS+=($!)

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
