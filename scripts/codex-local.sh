#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${1:-}"
shift || true
LOCAL_PORT="${CODEX_LOCAL_PORT:-8001}"

if [[ -z "${MODEL_ID}" || "${MODEL_ID}" == -* ]]; then
    printf '%s\n' 'Usage: scripts/codex-local.sh MODEL [CODEX_ARGS...]' >&2
    printf '%s\n' 'Example: scripts/codex-local.sh qwen38-27b-ud-q3-v3' >&2
    exit 2
fi

cleanup() {
    "${SCRIPT_DIR}/stop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

CODEX_LOCAL_PORT="${LOCAL_PORT}" "${SCRIPT_DIR}/run.sh" "${MODEL_ID}"

CODEX_BIN="${CODEX_BIN:-codex}"
"${CODEX_BIN}" \
    -c model_provider=local-llama \
    -c model="${MODEL_ID}" \
    -c 'model_providers.local-llama.name="Local llama.cpp"' \
    -c "model_providers.local-llama.base_url=\"http://127.0.0.1:${LOCAL_PORT}/v1\"" \
    -c 'model_providers.local-llama.wire_api="responses"' \
    "$@"
