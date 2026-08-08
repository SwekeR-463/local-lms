#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: scripts/benchmark.sh [MODEL]'
    exit 0
fi
MODEL_ARG=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then MODEL_ARG="$1"; shift; fi
load_model_profile "${MODEL_ARG}"
[[ "$#" -eq 0 ]] || die "Usage: scripts/benchmark.sh [MODEL]"

require_command curl
require_command jq
ensure_results_dir
curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null || die "server is not healthy at http://${HOST}:${PORT}"

PROMPT_FILE="${PROJECT_DIR}/config/prompts/benchmark.txt"
REQUEST="$(mktemp)"
RESPONSE="$(mktemp)"
trap 'rm -f "${REQUEST}" "${RESPONSE}"' EXIT
jq -n --arg model "${MODEL_ID}" --rawfile prompt "${PROMPT_FILE}" --argjson seed "${SEED:-42}" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:512,temperature:0,seed:$seed}' >"${REQUEST}"
curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary "@${REQUEST}" \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${RESPONSE}"
jq -e '(.choices[0].message.reasoning_content // "") + (.choices[0].message.content // .choices[0].text // "") | length > 0' "${RESPONSE}" >/dev/null || die "benchmark returned no completion"

RSS_KB=0
PID_FILE="${RESULTS_DIR}/server.pid"
if pid_is_ours "${PID_FILE}"; then RSS_KB="$(ps -o rss= -p "$(<"${PID_FILE}")" | awk '{print $1+0}')"; fi
if is_macos; then
    HARDWARE="$(sysctl -n machdep.cpu.brand_string); $(human_bytes "$(sysctl -n hw.memsize)") unified memory"
else
    HARDWARE="$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')"
    if command -v nvidia-smi >/dev/null; then HARDWARE+="; $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)"; fi
fi

OUTPUT="${RESULTS_DIR}/benchmark-${MODEL_ID}-$(timestamp).json"
jq --arg model_id "${MODEL_ID}" --arg model_name "${MODEL_NAME:-${MODEL_ID}}" \
    --arg model_file "${MODEL_FILE:-${MODEL_PATH:-}}" --arg hardware "${HARDWARE}" \
    --argjson context "${CTX_SIZE:-${PREFERRED_CONTEXT}}" --argjson rss_kb "${RSS_KB}" \
    --arg cache_k "${CACHE_TYPE_K:-q8_0}" --arg cache_v "${CACHE_TYPE_V:-$(default_cache_v)}" --arg spec_type "${SPEC_TYPE:-none}" --arg run_at "$(iso_timestamp)" \
    '{model_id:$model_id,model_name:$model_name,model_file:$model_file,run_at:$run_at,hardware:$hardware,context:$context,cache_type_k:$cache_k,cache_type_v:$cache_v,spec_type:$spec_type,rss_kb:$rss_kb,prompt_tokens:(.usage.prompt_tokens // 0),completion_tokens:(.usage.completion_tokens // 0),prompt_ms:(.timings.prompt_ms // 0),generation_ms:(.timings.predicted_ms // 0),prompt_tokens_per_second:(.timings.prompt_per_second // 0),generation_tokens_per_second:(.timings.predicted_per_second // 0),draft_tokens:(.timings.draft_n // 0),draft_tokens_accepted:(.timings.draft_n_accepted // 0),draft_acceptance:(if (.timings.draft_n // 0) > 0 then .timings.draft_n_accepted / .timings.draft_n else 0 end),response:([.choices[0].message.reasoning_content,.choices[0].message.content,.choices[0].text] | map(select(. != null and . != "")) | join("\n\n"))}' \
    "${RESPONSE}" >"${OUTPUT}"
log "Benchmark saved: ${OUTPUT}"
jq '{model_id,context,rss_kb,prompt_tokens_per_second,generation_tokens_per_second}' "${OUTPUT}"
