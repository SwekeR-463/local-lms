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

is_macos() {
    [[ "$(uname -s)" == Darwin ]]
}

cpu_threads() {
    if is_macos; then sysctl -n hw.ncpu; else nproc; fi
}

runtime_threads() {
    if is_macos; then
        local level
        for level in 0 1 2; do
            if [[ "$(sysctl -n "hw.perflevel${level}.name" 2>/dev/null || true)" == Performance ]]; then
                sysctl -n "hw.perflevel${level}.physicalcpu"
                return
            fi
        done
        cpu_threads
    else
        printf '%s\n' 8
    fi
}

default_threads_batch() {
    if is_macos; then runtime_threads; else printf '%s\n' 12; fi
}

default_batch_size() {
    if is_macos; then printf '%s\n' 1024; else printf '%s\n' 512; fi
}

default_cpu_moe() {
    printf '%s\n' 0
}

default_cache_v() {
    if is_macos; then printf '%s\n' turbo3; else printf '%s\n' turbo4; fi
}

file_size() {
    if is_macos; then stat -f '%z' "$1"; else stat -c '%s' "$1"; fi
}

human_bytes() {
    awk -v bytes="$1" 'BEGIN { split("B KiB MiB GiB TiB", u); while (bytes >= 1024 && i < 4) { bytes /= 1024; i++ } printf "%.1f %s\n", bytes, u[i+1] }'
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}

epoch_ms() {
    perl -MTime::HiRes=time -e 'printf "%.0f\n", time * 1000'
}

iso_timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

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

load_model_profile() {
    MODEL_ID="${1:-${DEFAULT_MODEL:-}}"
    [[ "${MODEL_ID}" =~ ^[a-zA-Z0-9._-]+$ ]] || die "invalid model ID: ${MODEL_ID}"
    MODEL_CONFIG="${PROJECT_DIR}/config/models/${MODEL_ID}.env"
    [[ -f "${MODEL_CONFIG}" ]] || die "unknown model '${MODEL_ID}'; available: $(find "${PROJECT_DIR}/config/models" -name '*.env' -exec basename {} .env \; | sort | tr '\n' ' ')"
    # shellcheck disable=SC1090
    source "${MODEL_CONFIG}"
    load_optional_config "${PROJECT_DIR}/config/runtime.env"
    load_optional_config "${PROJECT_DIR}/config/local/${MODEL_ID}.env"
}

selected_config() {
    printf '%s/config/local/%s.env\n' "${PROJECT_DIR}" "${MODEL_ID}"
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
        "${PROJECT_DIR}/.cache/llama.cpp/build/bin/llama-server" \
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
    ps -ww -p "${pid}" -o command= 2>/dev/null | grep -Fq -- "${PROJECT_DIR}"
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
