#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
MODEL_ARG=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then MODEL_ARG="$1"; shift; fi
load_model_profile "${MODEL_ARG}"

FOREGROUND=0
while (($#)); do
    case "$1" in
        --foreground) FOREGROUND=1 ;;
        --skip-preflight) SKIP_PREFLIGHT=1 ;;
        -h|--help)
            printf '%s\n' 'Usage: scripts/run.sh [MODEL] [--foreground] [--skip-preflight]'
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

SERVER="$(resolve_server_bin)"
MODEL="$(resolve_model_path)"
ensure_results_dir

if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
    "${SCRIPT_DIR}/preflight.sh" "${MODEL_ID}"
fi
port_is_free "${HOST}" "${PORT}" || die "port ${HOST}:${PORT} is already in use"

HELP="$(server_help "${SERVER}")"
ARGS=(-m "${MODEL}")
add_flag_value() {
    local flag="$1"
    local value="$2"
    if has_flag "${HELP}" "${flag}"; then ARGS+=("${flag}" "${value}"); fi
}
add_flag_value "--host" "${HOST}"
add_flag_value "--port" "${PORT}"
add_flag_value "--ctx-size" "${CTX_SIZE:-${PREFERRED_CONTEXT}}"
add_flag_value "--threads" "${THREADS:-$(runtime_threads)}"
add_flag_value "--threads-batch" "${THREADS_BATCH:-$(default_threads_batch)}"
add_flag_value "--parallel" "${PARALLEL:-1}"
add_flag_value "--cache-type-k" "${CACHE_TYPE_K:-q8_0}"
add_flag_value "--cache-type-v" "${CACHE_TYPE_V:-$(default_cache_v)}"
add_flag_value "--ubatch-size" "${UBATCH_SIZE:-$(default_batch_size)}"
add_flag_value "--batch-size" "${BATCH_SIZE:-$(default_batch_size)}"
add_flag_value "--n-cpu-moe" "${N_CPU_MOE:-$(default_cpu_moe)}"
add_flag_value "--seed" "${SEED:-42}"
[[ -z "${TEMPERATURE:-}" ]] || add_flag_value "--temp" "${TEMPERATURE}"
[[ -z "${TOP_P:-}" ]] || add_flag_value "--top-p" "${TOP_P}"
[[ -z "${TOP_K:-}" ]] || add_flag_value "--top-k" "${TOP_K}"
if [[ -n "${DRAFT_MODEL_FILE:-}" ]]; then
    DRAFT_MODEL_DIR="$(project_path "${DRAFT_MODEL_DIR:-models}")"
    DRAFT_MODEL="${DRAFT_MODEL_DIR}/$(basename "${DRAFT_MODEL_FILE}")"
    [[ -f "${DRAFT_MODEL}" ]] || die "draft model does not exist: ${DRAFT_MODEL}"
    add_flag_value "--model-draft" "${DRAFT_MODEL}"
fi
if [[ -n "${SPEC_TYPE:-}" ]]; then
    add_flag_value "--spec-type" "${SPEC_TYPE}"
    add_flag_value "--spec-draft-n-max" "${SPEC_DRAFT_N_MAX:-1}"
    add_flag_value "--spec-draft-n-min" "${SPEC_DRAFT_N_MIN:-0}"
    add_flag_value "--spec-draft-p-min" "${SPEC_DRAFT_P_MIN:-0.75}"
    add_flag_value "--spec-ngram-mod-n-min" "${SPEC_NGRAM_MOD_N_MIN:-8}"
    add_flag_value "--spec-ngram-mod-n-max" "${SPEC_NGRAM_MOD_N_MAX:-24}"
    add_flag_value "--spec-ngram-mod-n-match" "${SPEC_NGRAM_MOD_N_MATCH:-48}"
fi

if has_flag "${HELP}" "-ngl"; then ARGS+=(-ngl "${GPU_LAYERS:-99}"); fi
if has_flag "${HELP}" "-fa"; then ARGS+=(-fa on); fi
if has_flag "${HELP}" "--jinja"; then ARGS+=(--jinja); fi
if has_flag "${HELP}" "--metrics"; then ARGS+=(--metrics); fi

if [[ "${EXPOSE_NETWORK:-0}" == 1 ]]; then
    warn "network exposure enabled: ${HOST}:${PORT}"
fi

PID_FILE="${PROJECT_DIR}/results/server.pid"
LOG_FILE="${RESULTS_DIR}/server-$(timestamp).log"
[[ ! -e "${PID_FILE}" ]] || ! pid_is_ours "${PID_FILE}" || die "project server appears to already be running"

log "Launching ${SERVER}"
log "Model: ${MODEL_NAME:-${MODEL_ID}} (${MODEL})"
log "Context: ${CTX_SIZE:-${PREFERRED_CONTEXT}}, K/V: ${CACHE_TYPE_K:-q8_0}/${CACHE_TYPE_V:-$(default_cache_v)}"
log "Log: ${LOG_FILE}"

if ((FOREGROUND)); then
    exec "${SERVER}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi

"${SERVER}" "${ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${PID_FILE}"
sleep 1
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    die "server exited during startup; inspect ${LOG_FILE}"
fi
log "Server started with PID ${SERVER_PID}: http://${HOST}:${PORT}"
record_log "${MODEL_ID}: server started with PID ${SERVER_PID}; log ${LOG_FILE#"${PROJECT_DIR}/"}"
