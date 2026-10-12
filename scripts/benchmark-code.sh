#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: scripts/benchmark-code.sh [MODEL] [--sampling greedy|profile]'
    printf '%s\n' '  --sampling greedy   suite defaults: temperature 0, low reasoning (default, keeps earlier results comparable)'
    printf '%s\n' '  --sampling profile  use the sampling parameters prescribed in config/models/<id>.env'
    exit 0
fi
MODEL_ARG="${1:-}"
[[ -n "${MODEL_ARG}" && "${MODEL_ARG}" != -* ]] && shift
SAMPLING="greedy"
while (($#)); do
    case "$1" in
        --sampling) shift; SAMPLING="${1:?missing value for --sampling}" ;;
        -h|--help)
            printf '%s\n' 'Usage: scripts/benchmark-code.sh [MODEL] [--sampling greedy|profile]'
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done
case "${SAMPLING}" in
    greedy|profile) ;;
    *) die "unknown --sampling mode: ${SAMPLING}" ;;
esac
load_model_profile "${MODEL_ARG}"

require_command curl
require_command jq
require_command python3
curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null || die "server is not healthy at http://${HOST}:${PORT}"
EXPECTED_MODEL="$(resolve_model_path)"
SERVER_MODEL="$(curl --fail --silent "http://${HOST}:${PORT}/v1/models" | jq -r '.data[0].id // empty')"
[[ "${SERVER_MODEL}" == "${EXPECTED_MODEL}" ]] || die "server loaded ${SERVER_MODEL:-unknown}, expected ${EXPECTED_MODEL}"

SUITE_DIR="${PROJECT_DIR}/benchmark-data/short-python"
MAX_TOKENS="${BENCHMARK_MAX_TOKENS:-1024}"
REASONING_EFFORT="${BENCHMARK_REASONING_EFFORT:-low}"
SAMPLING_JSON='{}'
if [[ "${SAMPLING}" == profile ]]; then
    [[ -z "${TEMPERATURE:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${TEMPERATURE}" '. + {temperature:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${TOP_P:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${TOP_P}" '. + {top_p:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${TOP_K:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${TOP_K}" '. + {top_k:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${MIN_P:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${MIN_P}" '. + {min_p:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${PRESENCE_PENALTY:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${PRESENCE_PENALTY}" '. + {presence_penalty:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${REPEAT_PENALTY:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${REPEAT_PENALTY}" '. + {repetition_penalty:$v}' <<<"${SAMPLING_JSON}")"
    [[ -z "${CHAT_TEMPLATE_KWARGS:-}" ]] || SAMPLING_JSON="$(jq -c --argjson v "${CHAT_TEMPLATE_KWARGS}" '. + {chat_template_kwargs:$v}' <<<"${SAMPLING_JSON}")"
    [[ "${SAMPLING_JSON}" != '{}' ]] || die "--sampling profile requires sampling parameters in ${MODEL_CONFIG}"
    PROFILE_EFFORT="$(jq -r '.chat_template_kwargs.reasoning_effort // empty' <<<"${SAMPLING_JSON}")"
    [[ -z "${PROFILE_EFFORT}" ]] || REASONING_EFFORT="${PROFILE_EFFORT}"
else
    SAMPLING_JSON="$(jq -c -n --arg effort "${REASONING_EFFORT}" '{temperature:0,chat_template_kwargs:{reasoning_effort:$effort}}')"
fi
OUTPUT_DIR="$(unique_path "${RESULTS_DIR}/short-python-${MODEL_ID}-$(timestamp)")"
mkdir -p "${OUTPUT_DIR}"
OUTCOMES="${OUTPUT_DIR}/tasks.jsonl"
: >"${OUTCOMES}"

jq -r '.tasks[].id' "${SUITE_DIR}/manifest.json" | while IFS= read -r task; do
    task_dir="${SUITE_DIR}/tasks/${task}"
    response="${OUTPUT_DIR}/${task}-response.json"
    candidate="${OUTPUT_DIR}/${task}.py"
    evaluation="${OUTPUT_DIR}/${task}-evaluation.log"
    request="${OUTPUT_DIR}/${task}-request.json"

    jq -n --arg model "${MODEL_ID}" --rawfile prompt "${task_dir}/prompt.txt" \
        --argjson max_tokens "${MAX_TOKENS}" --argjson sampling "${SAMPLING_JSON}" \
        '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,seed:42} + $sampling' >"${request}"
    curl --fail --silent --show-error --max-time "${BENCHMARK_TIMEOUT_SECONDS:-1800}" -H 'Content-Type: application/json' --data-binary "@${request}" \
        "http://${HOST}:${PORT}/v1/chat/completions" >"${response}"

    python3 - "${response}" "${candidate}" <<'PY'
import json, re, sys
message = json.load(open(sys.argv[1]))["choices"][0]
text = message.get("message", {}).get("content") or message.get("text") or ""
match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
open(sys.argv[2], "w").write((match.group(1) if match else text).strip() + "\n")
PY

    passed=false
    if python3 "${task_dir}/evaluate.py" "${candidate}" >"${evaluation}" 2>&1; then passed=true; fi
    jq -n --arg task "${task}" --argjson passed "${passed}" \
        --arg response "$(basename "${response}")" --arg candidate "$(basename "${candidate}")" \
        --arg evaluation "$(basename "${evaluation}")" \
        --slurpfile api "${response}" \
        '{task:$task,passed:$passed,response:$response,candidate:$candidate,evaluation:$evaluation,finish_reason:($api[0].choices[0].finish_reason // ""),prompt_tokens:($api[0].usage.prompt_tokens // 0),completion_tokens:($api[0].usage.completion_tokens // 0),generation_tokens_per_second:($api[0].timings.predicted_per_second // 0)}' >>"${OUTCOMES}"
    log "${task}: ${passed}"
done

SUMMARY="${OUTPUT_DIR}/summary.json"
jq -s --arg model_id "${MODEL_ID}" --arg model_file "${MODEL_FILE:-${MODEL_PATH:-}}" --arg server_model "${SERVER_MODEL}" \
    --arg suite_id "short-python-v1" --arg run_at "$(iso_timestamp)" --arg effort "${REASONING_EFFORT}" --argjson max_tokens "${MAX_TOKENS}" \
    --arg sampling_mode "${SAMPLING}" --argjson sampling "${SAMPLING_JSON}" \
    '{model_id:$model_id,model_file:$model_file,server_model:$server_model,suite_id:$suite_id,run_at:$run_at,sampling_mode:$sampling_mode,sampling:$sampling,settings:{temperature:0,seed:42,max_tokens:$max_tokens,reasoning_effort:$effort},total:length,passed:(map(select(.passed))|length),tasks:.}' \
    "${OUTCOMES}" >"${SUMMARY}"
log "Short Python benchmark saved: ${SUMMARY}"
jq '{model_id,passed,total}' "${SUMMARY}"
