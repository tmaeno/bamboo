#!/usr/bin/env bash
# submit.sh — example launch of the batch container under Apptainer.
#
# The SAME .sif runs on CPU and GPU queues; the only difference is the --nv flag.
# Shown standalone here; a SLURM wrapper is in comments.
#
# Model names are NOT passed here: they are derived in-container from the staged
# artifacts — LLM_MODEL from /models (bamboo stage-model), EMBEDDING_MODEL/DIMENSION from
# /kb/metadata.json, RERANKER_MODEL from /embeddings (bamboo stage-embeddings). Export any
# of LLM_MODEL / EMBEDDING_MODEL / RERANKER_MODEL to override the derived value for a run.
#
# This passes the `batch-analyze` subcommand explicitly: boot the stack, run
# `bamboo batch-analyze` over /in, tear down. The container does not guess a workload —
# with no subcommand it prints usage and exits non-zero. To debug interactively instead,
# use `shell` (or `setup`/`teardown`); to run a single command against the booted stack
# non-interactively, use `exec <cmd…>` — see the entrypoint header
# (deploy/batch/entrypoint.sh) and the Batch Analysis guide, "Interactive debugging".
#
# ⚠ SCAFFOLD — UNVERIFIED. Adjust SHARED/SCRATCH paths and scheduler to your site.
set -euo pipefail

SIF="${SIF:-bamboo-batch.sif}"
SHARED="${SHARED:-/shared}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/bamboo.$$}"     # node-local scratch
IN_DIR="${IN_DIR:-$PWD/in}"                          # staged task-data *.json
OUT_DIR="${OUT_DIR:-$PWD/out}"
USE_GPU="${USE_GPU:-0}"                              # 1 on a GPU queue

mkdir -p "${SCRATCH}" "${OUT_DIR}"

binds=(
  "${IN_DIR}:/in:ro"
  "${OUT_DIR}:/out"
  "${SCRATCH}:/work"
  "${SHARED}/bamboo/ollama:/models:ro"
  "${SHARED}/bamboo/embeddings:/embeddings:ro"
  "${SHARED}/bamboo/kb:/kb:ro"
)
bind_arg="$(IFS=, ; echo "${binds[*]}")"

apptainer_args=(run --cleanenv
  --bind "${bind_arg}"
  --env "BAMBOO_WORK=/work"
)
# Optional per-run overrides — forwarded only when explicitly set (else derived in-container).
for _var in LLM_MODEL EMBEDDING_MODEL RERANKER_MODEL; do
  [[ -n "${!_var:-}" ]] && apptainer_args+=(--env "${_var}=${!_var}")
done
[[ "${USE_GPU}" == "1" ]] && apptainer_args+=(--nv)

# --- Optional: live PanDA fetch (--task-id) needs OIDC creds. Pass the token via
#     the file: form so it never lands in env/argv (see the Batch guide). ---
if [[ -n "${PANDA_TOKEN_FILE:-}" ]]; then
  apptainer_args+=(
    --bind "${PANDA_TOKEN_FILE}:/run/panda/token:ro"
    --env "PANDA_AUTH=oidc"
    --env "PANDA_AUTH_VO=${PANDA_AUTH_VO:-}"
    --env "PANDA_AUTH_ID_TOKEN=file:/run/panda/token"
  )
fi

echo "[submit] apptainer ${apptainer_args[*]} ${SIF} batch-analyze"
apptainer "${apptainer_args[@]}" "${SIF}" batch-analyze

# ---------------------------------------------------------------------------
# SLURM example (CPU queue):
#   #!/bin/bash
#   #SBATCH -p cpu -c 8 --mem=24G -t 02:00:00
#   export SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out   # models derived from staged artifacts
#   srun deploy/batch/submit.sh
# GPU queue: add `#SBATCH -p gpu --gres=gpu:1` and `export USE_GPU=1`.
# ---------------------------------------------------------------------------
