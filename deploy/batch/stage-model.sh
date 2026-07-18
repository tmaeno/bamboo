#!/usr/bin/env bash
# stage-model.sh — thin wrapper around `bamboo stage-model`, kept for the repo/batch flow
# and its documented invocation. The real implementation lives in the CLI, so it also
# ships with `pip install bamboo` (no repo checkout needed).
#
# Usage:  ./stage-model.sh                    # model from your bamboo config (Ollama)
#         MODEL=qwen3.6 ./stage-model.sh   # explicit (env)
#         ./stage-model.sh qwen3.6         # explicit (positional)
#         (MODELS_OUT / SHARED flow through to the CLI via env; set BAMBOO to override
#          the interpreter/entrypoint, default `bamboo`)
#
# ⚠ SCAFFOLD — UNVERIFIED.
set -euo pipefail

_model="${MODEL:-${1:-}}"
if [ -n "${_model}" ]; then
  exec "${BAMBOO:-bamboo}" stage-model --model "${_model}"
else
  exec "${BAMBOO:-bamboo}" stage-model
fi
