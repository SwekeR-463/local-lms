#!/usr/bin/env bash

set -Eeuo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${COMMON_DIR}/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/config/default.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONFIG_FILE}"
fi

RESULTS_DIR="${RESULTS_DIR:-results}"
if [[ "${RESULTS_DIR}" != /* ]]; then
    RESULTS_DIR="${PROJECT_DIR}/${RESULTS_DIR}"
fi

timestamp() {
    date '+%Y%m%d-%H%M%S'
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
    printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
    printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

project_path() {
    local value="$1"
    if [[ "${value}" = /* ]]; then
        printf '%s\n' "${value}"
    else
        printf '%s/%s\n' "${PROJECT_DIR}" "${value}"
    fi
}

ensure_results_dir() {
    mkdir -p "${RESULTS_DIR}"
}

load_optional_config() {
    local file="$1"
    if [[ -f "${file}" ]]; then
        # shellcheck disable=SC1090
        source "${file}"
    fi
}

server_help() {
    local server_bin="$1"
    "${server_bin}" --help 2>&1 || true
}

has_flag() {
    local help_text="$1"
    local flag="$2"
    grep -Eq -- "(^|[[:space:],])${flag}([=[:space:],]|$)" <<<"${help_text}"
}

resolve_server_bin() {
    if [[ -n "${SERVER_BIN:-}" ]]; then
        [[ -x "${SERVER_BIN}" ]] || die "SERVER_BIN is not executable: ${SERVER_BIN}"
        printf '%s\n' "${SERVER_BIN}"
        return
    fi

    local candidate
    for candidate in \
        "${PROJECT_DIR}/.cache/llama-cpp-turboquant/build/bin/llama-server" \
        "${PROJECT_DIR}/.cache/llama-cpp-turboquant/build/bin/lm-server-tq" \
        "${PROJECT_DIR}/build/bin/llama-server" \
        "${PROJECT_DIR}/build/bin/lm-server-tq"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return
    fi
    if command -v lm-server-tq >/dev/null 2>&1; then
        command -v lm-server-tq
        return
    fi
    die "could not locate llama-server; run scripts/build.sh or set SERVER_BIN"
}

resolve_model_path() {
    local model_path="${MODEL_PATH:-}"
    if [[ -n "${model_path}" ]]; then
        if [[ "${model_path}" != /* ]]; then
            model_path="${PROJECT_DIR}/${model_path}"
        fi
        [[ -f "${model_path}" ]] || die "MODEL_PATH does not exist: ${model_path}"
        printf '%s\n' "${model_path}"
        return
    fi

    [[ -n "${MODEL_FILE:-}" ]] || die "MODEL_PATH or MODEL_FILE must be set; run scripts/download-model.sh"
    local model_dir="${MODEL_DIR:-${PROJECT_DIR}/models}"
    [[ "${model_dir}" = /* ]] || model_dir="${PROJECT_DIR}/${model_dir}"
    local candidate="${model_dir}/${MODEL_FILE}"
    [[ -f "${candidate}" ]] || die "model file does not exist: ${candidate}"
    printf '%s\n' "${candidate}"
}

pid_is_ours() {
    local pid_file="$1"
    local pid
    [[ -f "${pid_file}" ]] || return 1
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq -- "${PROJECT_DIR}"
}

port_is_free() {
    local host="$1"
    local port="$2"
    if command -v ss >/dev/null 2>&1; then
        ! ss -H -ltn "sport = :${port}" | grep -q .
        return
    fi
    if command -v lsof >/dev/null 2>&1; then
        ! lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t | grep -q .
        return
    fi
    warn "neither ss nor lsof is installed; cannot verify whether ${host}:${port} is free"
    return 0
}

json_escape() {
    sed 's/\\/\\\\/g; s/"/\\"/g'
}

record_log() {
    local message="$*"
    local log_file="${PROJECT_DIR}/log.md"
    [[ -f "${log_file}" ]] || return 0
    printf '\n- `%s` — %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${message}" >>"${log_file}"
}
