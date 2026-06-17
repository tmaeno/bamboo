#!/usr/bin/env bash
# stage-model.sh — one-time pull of the Ollama LLM model into shared storage, on a
# NETWORKED host. The batch container then mounts this dir read-only at /models.
#
# Usage:  MODEL=llama3.2:3b ./stage-model.sh
#         (override MODELS_OUT / SHARED as needed)
#
# Uses a local `ollama` if present, otherwise the official ollama Docker image.
# ⚠ SCAFFOLD — UNVERIFIED.
set -euo pipefail

MODEL="${MODEL:-${1:-llama3.2:3b}}"          # pick one tolerable on CPU, faster on GPU
MODELS_OUT="${MODELS_OUT:-${SHARED:-/shared}/bamboo/ollama}"
mkdir -p "${MODELS_OUT}"

echo "[stage-model] pulling '${MODEL}' into ${MODELS_OUT}"
if command -v ollama >/dev/null 2>&1; then
  OLLAMA_MODELS="${MODELS_OUT}" ollama pull "${MODEL}"
else
  # Run a throwaway server + pull, models land in the mounted dir.
  cname="bamboo-stage-ollama-$$"
  trap 'docker rm -f "${cname}" >/dev/null 2>&1 || true' EXIT
  docker run -d --name "${cname}" -v "${MODELS_OUT}:/root/.ollama/models" ollama/ollama >/dev/null
  for _ in $(seq 60); do docker exec "${cname}" ollama list >/dev/null 2>&1 && break; sleep 2; done
  docker exec "${cname}" ollama pull "${MODEL}"
fi

echo "[stage-model] ✓ '${MODEL}' staged. Set LLM_MODEL=${MODEL} at submit time."
