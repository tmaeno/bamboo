#!/usr/bin/env bash
# submit.sh — example launch of the batch container under Apptainer.
#
# The SAME .sif runs on CPU and GPU queues; the only difference is the --nv flag
# (and possibly LLM_MODEL). Shown standalone here; a SLURM wrapper is in comments.
#
# ⚠ SCAFFOLD — UNVERIFIED. Adjust SHARED/SCRATCH paths, model, and scheduler to
#   your site.
set -euo pipefail

SIF="${SIF:-bamboo-batch-analyze.sif}"
SHARED="${SHARED:-/shared}"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/bamboo.$$}"     # node-local scratch
IN_DIR="${IN_DIR:-$PWD/in}"                          # staged task-data *.json
OUT_DIR="${OUT_DIR:-$PWD/out}"
LLM_MODEL="${LLM_MODEL:-llama3.2:3b}"
USE_GPU="${USE_GPU:-0}"                              # 1 on a GPU queue

mkdir -p "${SCRATCH}" "${OUT_DIR}"

binds=(
  "${IN_DIR}:/in:ro"
  "${OUT_DIR}:/out"
  "${SCRATCH}:/work"
  "${SHARED}/bamboo/ollama:/models:ro"
  "${SHARED}/bamboo/kb:/kb:ro"
)
bind_arg="$(IFS=, ; echo "${binds[*]}")"

apptainer_args=(run --cleanenv
  --bind "${bind_arg}"
  --env "LLM_MODEL=${LLM_MODEL}"
  --env "BAMBOO_WORK=/work"
)
[[ "${USE_GPU}" == "1" ]] && apptainer_args+=(--nv)

# --- Optional: live PanDA fetch (--task-id) needs OIDC creds. Pass the token via
#     the file: form so it never lands in env/argv (see docs/BATCH.md). ---
if [[ -n "${PANDA_TOKEN_FILE:-}" ]]; then
  apptainer_args+=(
    --bind "${PANDA_TOKEN_FILE}:/run/panda/token:ro"
    --env "PANDA_AUTH=oidc"
    --env "PANDA_AUTH_VO=${PANDA_AUTH_VO:-}"
    --env "PANDA_AUTH_ID_TOKEN=file:/run/panda/token"
  )
fi

echo "[submit] apptainer ${apptainer_args[*]} ${SIF}"
apptainer "${apptainer_args[@]}" "${SIF}"

# ---------------------------------------------------------------------------
# SLURM example (CPU queue):
#   #!/bin/bash
#   #SBATCH -p cpu -c 8 --mem=24G -t 02:00:00
#   export SHARED=/shared IN_DIR=$PWD/in OUT_DIR=$PWD/out LLM_MODEL=llama3.2:3b
#   srun deploy/batch/submit.sh
# GPU queue: add `#SBATCH -p gpu --gres=gpu:1` and `export USE_GPU=1`.
# ---------------------------------------------------------------------------
