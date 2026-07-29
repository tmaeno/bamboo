#!/usr/bin/env bash
# entrypoint.sh — entry point for the batch container (Image 2).
#
# Serves two use cases from one image:
#   • air-gapped batch — `batch-analyze`; bundled Neo4j + Qdrant + Ollama, no keys.
#   • portable run     — mount your own config at /app/.env
#                        (`-v $PWD/.env:/app/.env:ro`) to inject keys/settings
#                        (PANDA_*, SSL_CERT_FILE, LOG_LEVEL, …). See below.
#
# Boots Neo4j + Qdrant + Ollama on localhost from read-only shared-FS mounts,
# restores the KB into node-local scratch, runs the requested workload against that
# stack, then tears everything down. Designed to run rootless under Apptainer (you
# are an arbitrary uid; the .sif and /models /kb are read-only).
#
# Subcommands — each names the workload it runs; the container never guesses one
# (`bamboo batch-analyze` is only one of bamboo's batch commands, batch-populate is
# another, so no argument / an unknown token / --help print usage instead of booting):
#
#   entrypoint.sh batch-analyze [args]
#                                 the batch job: `bamboo batch-analyze --input-dir /in
#                                 --output-dir /out [args]`, then teardown. What
#                                 deploy/batch/submit.sh runs. Any OTHER bamboo command
#                                 goes through `exec` (see below).
#   entrypoint.sh exec <cmd…>     run <cmd…> against the stack (argv verbatim, no shell
#                                 re-parse), teardown on exit; exits with <cmd…>'s
#                                 status. Needs no TTY, so it is the scriptable form of
#                                 `shell`. For pipes/redirects: exec bash -lc '…'.
#   entrypoint.sh shell           drop into an interactive shell with the env exported;
#                                 teardown on exit.
#   entrypoint.sh setup           boot the stack + restore the KB, then LEAVE IT
#                                 RUNNING. Persists a state env-file (see below).
#   entrypoint.sh teardown        kill the services and remove the scratch dir.
#   entrypoint.sh help            print usage (booting nothing).
#
# Interactive debugging (ONE container session — services live in that session's
# process namespace, so `setup`/`teardown` must share a single `docker run`):
#   docker run -it … bamboo-batch shell                  # boot + interactive bash
#   # one-shot, no TTY needed, exit code is the command's:
#   docker run … bamboo-batch exec bamboo verify
#   docker run … bamboo-batch exec bash -lc 'bamboo verify | tee /out/verify.log'
#   # a non-analyze batch command (needs /kb mounted :rw for the write-back):
#   docker run … bamboo-batch exec bash -lc 'bamboo batch-populate … && bamboo dump-kb --out /kb'
#   # or drive the pieces yourself:
#   docker run -it --entrypoint bash … bamboo-batch
#     $ /opt/bamboo/entrypoint.sh setup
#     $ source "${BAMBOO_WORK:-/tmp}/bamboo-batch.env"    # bamboo now sees the stack
#     $ bamboo analyze … / bamboo investigate … / bamboo verify
#     $ bamboo batch-analyze --input-dir /in --output-dir /out
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
# Sandbox — the staged inputs handed over as ONE archive instead of separate mounts.
# Point BAMBOO_SANDBOX at a tarball (or an already-extracted directory) whose top level
# holds any of  models/  kb/  embeddings/  in/ :
#   docker run … -v $PWD/sandbox.tgz:/sandbox.tgz:ro -e BAMBOO_SANDBOX=/sandbox.tgz \
#                -v $PWD/work:/work -e BAMBOO_WORK=/work  bamboo-batch exec bamboo …
# Each component present in the sandbox re-points its path; components absent from it keep
# their mount default, so a kb/-only tarball composes with mounted /models + /embeddings.
# A tarball is expanded into the job's scratch dir alongside the restored Neo4j/Qdrant
# data, so mount node-local scratch at /work (BAMBOO_WORK) to keep that off the container
# writable layer. Nothing is auto-detected: with BAMBOO_SANDBOX unset the mounts stand
# exactly as before.
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

# Printed on `help` (to stdout) and ahead of a dispatch error (redirected to stderr
# by the caller). Every subcommand names the workload it runs: the container never
# guesses one, because `batch-analyze` is not the only batch command bamboo has.
usage() {
  cat <<'USAGE'
bamboo-batch — bamboo plus a bundled Neo4j + Qdrant + Ollama stack in one container.
Each subcommand boots the stack from the read-only mounts, runs its workload, and
tears the stack down (setup/teardown split it apart for interactive debugging).

Usage:  <image> <subcommand> [args…]

  batch-analyze [args…]  the batch job: `bamboo batch-analyze --input-dir /in
                         --output-dir /out [args…]`. This is what deploy/batch/submit.sh
                         runs. Any OTHER bamboo command goes through `exec`, e.g.
                         `exec bamboo batch-populate …`, `exec bamboo dump-kb …`.
  exec <cmd…>            run <cmd…> against the stack; exits with its status. argv is
                         passed verbatim (quoted free text is safe); for pipes,
                         redirects or && use `exec bash -lc '…'`.
  shell                  drop into an interactive bash with the stack env exported.
  setup                  boot the stack and LEAVE IT RUNNING; writes a state env-file.
                         Then: source it and run bamboo directly in the same session.
  teardown               kill the services and remove the scratch dir.
  help                   this message.

Mounts:  /in (ro) task-data *.json   /out (rw) results   /kb (ro) KB snapshot
         /models (ro) Ollama models  /embeddings (ro) HF cache   /work (rw, optional)
Sandbox: BAMBOO_SANDBOX=<tarball|dir> supplies those inputs as one archive whose top
         level holds any of models/ kb/ embeddings/ in/ — each component present wins
         over its mount, each absent one keeps the mount. Opt-in; nothing is guessed.
Portable mode: mount your own .env at /app/.env to inject keys/settings (not the
         provider/model/service vars — those stay derived from the staged KB/model).
USAGE
}

# Optional portable-mode config: mount your own .env at /app/.env
# (`-v $PWD/.env:/app/.env:ro`, /app is WORKDIR). We do NOT parse it here — bamboo
# loads it itself (config._find_env_file, override=False), so it supplies keys /
# tokens / settings (PANDA_*, SSL_CERT_FILE, LOG_LEVEL, …). The batch-managed vars
# (provider, LLM_MODEL, EMBEDDING_*, service URLs) are exported by do_setup *before*
# bamboo runs, so override=False keeps them derived from the staged KB/model — a .env
# copied from .env.example can't flip the container off its bundled stack. This is
# just a heads-up so those ignored .env lines aren't mistaken for a bug.
if [[ -f /app/.env ]]; then
  log "runtime .env detected at /app/.env — supplies keys/settings to bamboo;"
  log "  batch-managed vars (provider, LLM_MODEL, EMBEDDING_*, service URLs) stay derived."
fi

# --------------------------------------------------------------------------- #
# Scratch + teardown (must survive SIGKILL/walltime: kill the process group,
# remove scratch). Services run in their own process groups (setsid) so they
# outlive do_setup returning; teardown kills those groups.
#
# In-process (batch-analyze / exec / shell) WORK and PIDS are already set.
# Standalone `teardown` loads them from the state env-file first.
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
# Persist the derived env + service PIDs so a later `teardown`, or a shell in the
# same session, can source the live stack. `printf %q` escaping keeps values
# re-sourceable. Only emit the optional model vars when they were actually set.
#
# BAMBOO_SANDBOX/BAMBOO_IN/BAMBOO_KB are in the list for the `setup` → source → run
# bamboo by hand flow (see the header): with a sandbox the inputs live in scratch, so
# `--input-dir /in` would be wrong. They are unset without a sandbox, and the loop skips
# unset vars, so a non-sandbox state file is unchanged.
# --------------------------------------------------------------------------- #
persist_env() {
  {
    printf '# bamboo batch stack — written by entrypoint.sh setup; source to reach the live stack.\n'
    local k
    for k in NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD NEO4J_DATABASE \
             QDRANT_URL QDRANT_COLLECTION_NAME \
             OLLAMA_BASE_URL OLLAMA_HOST OLLAMA_MODELS \
             LLM_PROVIDER LLM_MODEL \
             EMBEDDINGS_PROVIDER EMBEDDING_MODEL EMBEDDING_DIMENSION RERANKER_MODEL \
             HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE DOC_INDEX_FREEZE \
             BAMBOO_SANDBOX BAMBOO_IN BAMBOO_KB; do
      [[ -n "${!k:-}" ]] && printf 'export %s=%q\n' "$k" "${!k}"
    done
    # Internal bookkeeping for teardown (not exported into bamboo's env).
    printf 'BAMBOO_WORK_ACTIVE=%q\n' "${WORK}"
    printf 'BAMBOO_SERVICE_PIDS=%q\n' "${PIDS[*]:-}"
  } >"${BAMBOO_STATE_FILE}"
  log "wrote stack state to ${BAMBOO_STATE_FILE}"
}

# --------------------------------------------------------------------------- #
# resolve_sandbox — accept the staged inputs as ONE archive instead of four mounts.
#
# BAMBOO_SANDBOX names a tarball, or a directory that is already the extracted form of
# one (used in place, no copy). Its top level holds any of models/ kb/ embeddings/ in/;
# each component present re-points the corresponding path and each absent one keeps the
# value it already had, so a partial sandbox composes with the /models /kb /embeddings
# /in mounts. Strictly opt-in: with BAMBOO_SANDBOX unset this is a no-op, which is how a
# stray tarball beside the job is kept from ever changing what a run reads.
#
# Must run after WORK exists and before anything reads OLLAMA_MODELS / HF_HOME / KB_DIR /
# IN_DIR — hence the call site at the top of do_setup. Extracting into WORK means
# teardown's `rm -rf "${WORK}"` removes it for free (walltime kills included), and what
# the services see is writable, unlike the :ro mounts.
# --------------------------------------------------------------------------- #
SANDBOX_COMPONENTS=(models kb embeddings in)

resolve_sandbox() {
  local src="${BAMBOO_SANDBOX:-}"
  [[ -n "${src}" ]] || return 0

  local root
  if [[ -d "${src}" ]]; then
    root="${src%/}"
    log "sandbox: using already-extracted directory ${root}"
  elif [[ -f "${src}" ]]; then
    command -v tar >/dev/null 2>&1 || die "sandbox: no tar in this image, cannot expand ${src}"
    root="${WORK}/sandbox"
    mkdir -p "${root}"
    # Size + headroom before the fact: a multi-GB archive filling scratch is the likeliest
    # failure here, and tar's "No space left on device" says nothing about how close it was.
    if command -v du >/dev/null 2>&1 && command -v df >/dev/null 2>&1; then
      log "sandbox: $(du -m "${src}" | awk '{print $1}') MB archive," \
          "$(( $(df -Pk "${root}" | awk 'NR==2{print $4}') / 1024 )) MB free at ${root}"
    fi
    log "sandbox: extracting ${src} -> ${root}"
    # No suffix logic: tar detects the compression from the archive itself. gzip/bzip2/xz
    # are all present in the image; .tar.zst is NOT — the Dockerfile purges zstd after
    # installing Ollama, so that purge is the line to change if zstd sandboxes ever appear.
    # --no-same-owner: the archive carries the staging host's uid/gid, while under rootless
    # Apptainer we are an arbitrary uid that must own what it extracts. (Already the
    # non-root default; explicit so the Docker-as-root path behaves identically.)
    tar -xf "${src}" -C "${root}" --no-same-owner
  else
    die "BAMBOO_SANDBOX=${src} is neither a file nor a directory"
  fi

  # True when $1 directly holds at least one component directory.
  _sandbox_has_component() {
    local c
    for c in "${SANDBOX_COMPONENTS[@]}"; do [[ -d "$1/${c}" ]] && return 0; done
    return 1
  }
  # Tolerate one wrapper level (`tar czf sandbox.tgz sandbox/` rather than `… -C sandbox .`),
  # but only when the top level names no component at all AND the single directory below it
  # does — so this can never shadow a real component, and a wrong archive is reported
  # against its own top level rather than against whatever we happened to descend into.
  # Settling this before the mapping keeps a bad archive from printing four "keeping …"
  # lines that read as if the sandbox had been accepted.
  if ! _sandbox_has_component "${root}"; then
    local subs=("${root}"/*/)
    if [[ ${#subs[@]} -eq 1 ]] && _sandbox_has_component "${subs[0]%/}"; then
      root="${subs[0]%/}"
      log "sandbox: no component at the top level — descending into ${root}"
    else
      die "sandbox ${src} holds none of ${SANDBOX_COMPONENTS[*]} at its top level (found: $(ls -A "${root}" 2>/dev/null | tr '\n' ' '))"
    fi
  fi

  # Map each component independently. Both directions are logged: a quietly kept /models
  # is exactly what you need to see when a partial sandbox didn't cover what you meant.
  if [[ -d "${root}/models" ]]; then
    export OLLAMA_MODELS="${root}/models"
    log "sandbox: models     -> ${OLLAMA_MODELS}"
  else
    log "sandbox: no models/ — keeping OLLAMA_MODELS=${OLLAMA_MODELS}"
  fi
  if [[ -d "${root}/embeddings" ]]; then
    export HF_HOME="${root}/embeddings"
    log "sandbox: embeddings -> ${HF_HOME}"
  else
    log "sandbox: no embeddings/ — keeping HF_HOME=${HF_HOME}"
  fi
  # KB_DIR/IN_DIR are script-local; BAMBOO_KB/BAMBOO_IN are exported alongside so
  # persist_env carries the sandbox paths into a `setup`-then-source session.
  if [[ -d "${root}/kb" ]]; then
    KB_DIR="${root}/kb"; export BAMBOO_KB="${KB_DIR}"
    log "sandbox: kb         -> ${KB_DIR}"
  else
    log "sandbox: no kb/ — keeping KB_DIR=${KB_DIR}"
  fi
  if [[ -d "${root}/in" ]]; then
    IN_DIR="${root}/in"; export BAMBOO_IN="${IN_DIR}"
    log "sandbox: in         -> ${IN_DIR}"
  else
    log "sandbox: no in/ — keeping IN_DIR=${IN_DIR}"
  fi
}

# --------------------------------------------------------------------------- #
# do_setup — boot the localhost stack and restore the KB into scratch.
#
# On entry it guards a partial boot (teardown on EXIT/INT/TERM); on success it
# clears that trap so the caller owns teardown (a standalone `setup` leaves the
# stack running; `exec`/`shell` re-arm the trap themselves).
# --------------------------------------------------------------------------- #
do_setup() {
  trap 'teardown' EXIT INT TERM

  WORK="$(mktemp -d "${WORK_ROOT%/}/bamboo.XXXXXX")"
  # OUT_DIR is deliberately NOT created here: `bamboo batch-analyze` creates its own
  # --output-dir, it is the only thing that writes there, and most subcommands never touch
  # it — an unbound /out would otherwise abort the whole boot on the read-only rootfs of a
  # rootless Apptainer run, over a directory the workload doesn't use.
  mkdir -p "${WORK}/neo4j/data" "${WORK}/neo4j/logs" "${WORK}/neo4j/run" \
           "${WORK}/neo4j/conf" "${WORK}/qdrant/storage" "${WORK}/ollama"

  # A sandbox may supply models/ kb/ embeddings/ in/ as one archive; resolve it before
  # anything below reads OLLAMA_MODELS / HF_HOME / KB_DIR / IN_DIR.
  resolve_sandbox

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
  # bamboo itself reads OLLAMA_BASE_URL; OLLAMA_HOST is exported so the `ollama`
  # CLI in an interactive `shell` (ollama ps/list) reaches the bundled server
  # instead of the default 11434, where nothing is listening.
  export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
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
  # VERIFY: Neo4j 5 admin syntax + dump filename (<db>.dump). See the /kb contract in the Batch guide.
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
  # see the Batch guide), listed in metadata.json under "qdrant_collections". Qdrant recovers
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
  # running; `exec`/`shell` re-arm their own EXIT trap. A mid-setup failure above
  # instead trips the guard trap and cleans up the half-booted stack.
  trap - EXIT INT TERM
}

# --------------------------------------------------------------------------- #
# Subcommands
#
# `exec` and `batch-analyze` share the two halves below rather than one calling the
# other, so rc propagation, teardown and stdout separation still hold by construction
# for both — while each keeps control of *when* its argv is built. That matters for
# batch-analyze: a sandbox re-points IN_DIR inside do_setup, so expanding it before the
# boot would silently pin the job to the /in mount.
# --------------------------------------------------------------------------- #

# Boot the stack and hand teardown to the EXIT trap.
# Setup writes to stdout (`neo4j-admin database load`), which would corrupt a piped
# command's output — the point of exec is that `docker run … exec bamboo … > file` works.
# A redirect on a function call is not a subshell, so WORK/PIDS still propagate.
_boot() {
  do_setup >&2
  trap 'teardown' EXIT INT TERM
}

# Run argv verbatim against the booted stack and exit with its status.
_run_and_exit() {
  log "stack is up — running: $*"
  local rc=0
  "$@" || rc=$?                 # argv verbatim: quoting preserved, no shell re-parse
  log "command exited rc=${rc}"
  exit "${rc}"                  # teardown fires via the EXIT trap
}

cmd_setup() {                   # boot the stack and leave it running
  do_setup
  log "stack is up. To use it:"
  log "  source ${BAMBOO_STATE_FILE}   # then run bamboo … against the live stack, e.g."
  log "  bamboo batch-analyze --input-dir ${IN_DIR} --output-dir ${OUT_DIR}"
  log "  entrypoint.sh teardown        # kill services + remove scratch"
}

cmd_teardown() {                # kill services + remove scratch (loads state if needed)
  teardown
}

cmd_shell() {                   # boot the stack, drop into an interactive shell, teardown on exit
  # `shell` is interactive-only. Extra args used to be silently dropped, which made
  # `shell bamboo verify` boot the stack and sit at a prompt as if nothing was asked.
  [[ $# -eq 0 ]] || die "'shell' takes no arguments — use 'exec $*' to run a command non-interactively"
  do_setup
  trap 'teardown' EXIT INT TERM
  log "stack is up — dropping into an interactive shell. Type 'exit' to tear down."
  log "  (bamboo already sees the stack via the exported env; e.g. 'bamboo verify')"
  bash -i || true               # child, not exec, so the EXIT trap runs teardown on return
}

cmd_exec() {                    # boot the stack, run one command against it, teardown on exit
  [[ $# -gt 0 ]] || die "exec needs a command, e.g. 'exec bamboo verify' (pipes/redirects: exec bash -lc '…')"
  # Resolve the command *before* booting: a typo would otherwise cost the full stack
  # boot (minutes) only to come back as a 127 from a command that was never going to run.
  command -v "$1" >/dev/null 2>&1 \
    || die "not an executable: '$1' (exec runs argv directly; for shell syntax use exec bash -lc '…')"
  _boot
  _run_and_exit "$@"
}

cmd_batch_analyze() {           # the batch job: `bamboo batch-analyze` wired to the /in → /out paths
  # Answer --help without booting anything (click never invokes the callback for it).
  local a
  for a in "$@"; do
    [[ "${a}" == "-h" || "${a}" == "--help" ]] && { bamboo batch-analyze --help; exit 0; }
  done
  # Boot first, THEN build the argv: a sandbox (BAMBOO_SANDBOX) can re-point IN_DIR inside
  # do_setup, so expanding it here would pin the job to the /in mount and ignore the
  # sandbox's in/. `bamboo` needs no `command -v` pre-flight the way exec's argv does —
  # it is this image's own entry point, not something a user typed.
  _boot
  _run_and_exit bamboo batch-analyze --input-dir "${IN_DIR}" --output-dir "${OUT_DIR}" "$@"
}

# --------------------------------------------------------------------------- #
# Dispatch. Every subcommand names the workload it runs, and nothing is implied:
# no argument, an unknown token and --help all print usage instead of booting the
# stack and guessing a job. `bamboo batch-analyze` is only *one* of bamboo's batch
# commands (batch-populate is another), so it gets no privileged position here —
# the others are reached with `exec bamboo <cmd> …`.
#
# Consume the subcommand into $sub, then shift it off *only if* one was given
# (a bare `shift` with no positional params fails and would trip `set -e`, which
# is the CMD [] / no-args case).
# --------------------------------------------------------------------------- #
sub="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "${sub}" in
  batch-analyze)  cmd_batch_analyze "$@" ;;
  setup)          cmd_setup "$@" ;;
  teardown)       cmd_teardown "$@" ;;
  shell)          cmd_shell "$@" ;;
  exec)           cmd_exec "$@" ;;
  help|-h|--help) usage ;;
  run|batch)      usage >&2       # migration aid: both used to mean batch-analyze
                  die "'${sub}' no longer exists — use 'batch-analyze' (boots the stack, runs the /in → /out job, tears down)" ;;
  "")             usage >&2
                  die "no subcommand given — 'batch-analyze' for the batch job, 'exec <cmd…>' for anything else" ;;
  -*)             usage >&2       # options used to be forwarded to batch-analyze implicitly
                  die "unknown option '${sub}' — options belong to a subcommand, e.g. 'batch-analyze ${sub} …'" ;;
  *)              usage >&2
                  die "unknown subcommand '${sub}' — to run it as a command use 'exec ${sub} …'" ;;
esac
