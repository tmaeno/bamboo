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
# Set SANDBOX=/path/to/sandbox.tgz when the staged inputs arrive as a single archive whose
# top level holds models/ kb/ embeddings/ (and optionally in/) instead of as the three
# ${SHARED}/bamboo/* directories — it is bound in and BAMBOO_SANDBOX points the container at
# it. Model names stay derived exactly as above, just from inside the sandbox.
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
USE_GPU="${USE_GPU:-0}"                              # 1 on a GPU queue: adds --nv and makes a
                                                     # CPU fallback fatal (BAMBOO_REQUIRE_GPU;
                                                     # set it to 0 to warn instead)
SANDBOX="${SANDBOX:-}"                               # optional: one archive (or an already
                                                     # extracted dir) carrying models/ kb/
                                                     # embeddings/ [in/] instead of the
                                                     # three ${SHARED}/bamboo/* mounts

mkdir -p "${SCRATCH}" "${OUT_DIR}"

binds=("${OUT_DIR}:/out" "${SCRATCH}:/work")
SANDBOX_DEST=""
if [[ -n "${SANDBOX}" ]]; then
  # With a sandbox the staged binds are dropped rather than added to: a site that ships one
  # archive has no ${SHARED}/bamboo/* tree, and binding a path that doesn't exist just makes
  # Apptainer fail. A directory is read in place; a tarball is expanded into /work in-container
  # (hence SCRATCH needs room for it on top of the restored KB). A partial sandbox — say kb/
  # only, with /models and /embeddings still on shared storage — is an in-container feature:
  # add those two binds by hand for that case.
  [[ -d "${SANDBOX}" ]] && SANDBOX_DEST=/sandbox || SANDBOX_DEST=/sandbox.tgz
  binds+=("${SANDBOX}:${SANDBOX_DEST}:ro")
  # in/ may come from the sandbox, so a local one is bound only if it is actually there.
  [[ -d "${IN_DIR}" ]] && binds+=("${IN_DIR}:/in:ro")
else
  binds+=(
    "${IN_DIR}:/in:ro"
    "${SHARED}/bamboo/ollama:/models:ro"
    "${SHARED}/bamboo/embeddings:/embeddings:ro"
    "${SHARED}/bamboo/kb:/kb:ro"
  )
fi
bind_arg="$(IFS=, ; echo "${binds[*]}")"

apptainer_args=(run --cleanenv
  --bind "${bind_arg}"
  --env "BAMBOO_WORK=/work"
)
[[ -n "${SANDBOX_DEST}" ]] && apptainer_args+=(--env "BAMBOO_SANDBOX=${SANDBOX_DEST}")
# Optional per-run overrides — forwarded only when explicitly set (else derived in-container).
#
# CUDA_VISIBLE_DEVICES / NVIDIA_VISIBLE_DEVICES are in this list because --cleanenv drops
# them: a scheduler that allocates a subset of a multi-GPU node communicates that choice
# through the environment, and without forwarding it the container sees every device on the
# node unless the site also isolates by cgroup. Forwarded only when set, so nothing is
# invented on a single-GPU or CPU queue.
for _var in LLM_MODEL EMBEDDING_MODEL RERANKER_MODEL \
            CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES; do
  [[ -n "${!_var:-}" ]] && apptainer_args+=(--env "${_var}=${!_var}")
done
# --nv wires the host driver in; BAMBOO_REQUIRE_GPU makes the container *verify* it was
# actually used. Ollama answers /api/tags and generates happily on the CPU when its CUDA
# runtime fails to load, so a GPU-queue job would otherwise just finish an order of
# magnitude late with nothing in the log saying why. Asking for a GPU queue and silently
# getting CPU is never the wanted outcome — so on a GPU queue it is a hard failure.
[[ "${USE_GPU}" == "1" ]] && apptainer_args+=(--nv --env "BAMBOO_REQUIRE_GPU=${BAMBOO_REQUIRE_GPU:-1}")

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
# GPU queue: add `#SBATCH -p gpu --gres=gpu:1` and `export USE_GPU=1`. Look for the
#   `accelerator: gpu …` line in the job log; to diagnose a CPU fallback without paying a
#   full stack boot, run `apptainer exec --nv … ${SIF} /opt/bamboo/entrypoint.sh gpu-check`.
# Single-archive inputs: `export SANDBOX=$PWD/sandbox.tgz` instead of SHARED (and give
# SCRATCH room for the expanded archive on top of the restored KB).
# ---------------------------------------------------------------------------
