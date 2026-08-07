#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
MODEL_ARG=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then MODEL_ARG="$1"; shift; fi
load_model_profile "${MODEL_ARG}"

MODE="short"
case "${1:-}" in
    --short|"") MODE=short ;;
    --accepted) MODE=accepted ;;
    --preferred) MODE=preferred ;;
    --quality) MODE=quality ;;
    -h|--help) printf '%s\n' 'Usage: scripts/validate.sh [MODEL] [--short|--accepted|--preferred|--quality]'; exit 0 ;;
    *) die "unknown validation mode: $1" ;;
esac

require_command curl
require_command jq
SERVER="$(resolve_server_bin)"
MODEL="$(resolve_model_path)"
[[ -n "${MODEL}" ]] || die "model is required"

curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null 2>&1 || die "server is not healthy at http://${HOST}:${PORT}"

case "${MODE}" in
    short) CONTEXT=8192 ;;
    accepted) CONTEXT="${CTX_SIZE:-${MIN_ACCEPTED_CONTEXT}}" ;;
    preferred) CONTEXT="${PREFERRED_CONTEXT}" ;;
    quality) CONTEXT=8192 ;;
esac

if [[ "${MODE}" == accepted && "${CONTEXT}" -lt "${MIN_ACCEPTED_CONTEXT}" ]]; then
    die "selected context ${CONTEXT} is below minimum accepted context ${MIN_ACCEPTED_CONTEXT}"
fi

if ((CONTEXT > 8192)); then
    prompt_chars=$((CONTEXT * 4))
    PROMPT="$(head -c "${prompt_chars}" < <(yes 'context validation token; '))"
    PROMPT+=$'\nReturn a short deterministic acknowledgement.'
else
    PROMPT="Validate ${MODEL_NAME:-${MODEL_ID}} server output at approximately ${CONTEXT} tokens. Return a short deterministic acknowledgement."
fi
REQUEST="${RESULTS_DIR}/validation-$(timestamp).json"
mkdir -p "${RESULTS_DIR}"
PROMPT_FILE="${RESULTS_DIR}/validation-prompt-$(timestamp).txt"
printf '%s' "${PROMPT}" >"${PROMPT_FILE}"
jq -n --rawfile prompt "${PROMPT_FILE}" --argjson seed "${SEED:-42}" \
    '{messages:[{role:"user",content:$prompt}],max_tokens:256,temperature:0,seed:$seed}' >"${REQUEST}"
RESPONSE="${REQUEST%.json}-response.json"
curl --fail --silent --show-error -H 'Content-Type: application/json' \
    --data-binary "@${REQUEST}" "http://${HOST}:${PORT}/v1/chat/completions" >"${RESPONSE}"

if ! jq -e '(.choices | length > 0) and ((.choices[0].message.content // .choices[0].text // "") | length > 0)' "${RESPONSE}" >/dev/null; then
    die "response did not contain usable text: ${RESPONSE}"
fi

log "${MODE} validation passed at configured context ${CONTEXT}; response: ${RESPONSE}"
record_log "${MODEL_ID}: ${MODE} validation passed at context ${CONTEXT}"
