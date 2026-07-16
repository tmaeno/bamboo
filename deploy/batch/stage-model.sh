#!/usr/bin/env bash
# stage-model.sh — one-time pull of the Ollama LLM model into shared storage, on a
# NETWORKED host. The batch container then mounts this dir read-only at /models.
#
# Usage:  MODEL=llama3.2:3b ./stage-model.sh   # explicit model (env or positional arg)
#         ./stage-model.sh                      # derive from the local bamboo .env
#         (override MODELS_OUT / SHARED / BAMBOO_PY as needed)
#
# With no explicit MODEL / positional arg, the model is taken from the local bamboo
# config (LLM_MODEL in .env, via get_settings()) — but ONLY when LLM_PROVIDER=ollama,
# since the batch path is Ollama-only. Point BAMBOO_PY at the interpreter that runs
# bamboo locally (default: python); falls back to llama3.2:3b if bamboo isn't importable.
#
# Uses a local `ollama` if present, otherwise the official ollama Docker image.
# ⚠ SCAFFOLD — UNVERIFIED.
set -euo pipefail

# Absent an explicit MODEL=/$1, adopt LLM_MODEL from the local bamboo config so the
# staged model matches the local run — guarded to Ollama configs (see header).
if [ -z "${MODEL:-}" ] && [ -z "${1:-}" ]; then
  BAMBOO_PY="${BAMBOO_PY:-python}"
  derived="$("${BAMBOO_PY}" - <<'PY' 2>/dev/null || true
from bamboo.config import get_settings
s = get_settings()
if s.llm_provider == "ollama":
    print(s.llm_model)
PY
)"
  if [ -n "${derived}" ]; then
    MODEL="${derived}"
    echo "[stage-model] using LLM_MODEL from local bamboo config: ${MODEL}"
  fi
fi
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
