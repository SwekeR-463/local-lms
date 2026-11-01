#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

case "${1:-}" in
    -h|--help)
        printf '%s\n' 'Usage: scripts/build.sh'
        printf '%s\n' 'Build llama-server with Metal on macOS, or CUDA/Vulkan on Linux.'
        exit 0
        ;;
    "") ;;
    *) die "unknown option: $1" ;;
esac

require_command git
require_command cmake

SOURCE_DIR="$(project_path "${SOURCE_DIR}")"
BUILD_DIR="$(project_path "${BUILD_DIR}")"
REPO_URL="${TURBOQUANT_REPO:-https://github.com/TheTom/llama-cpp-turboquant.git}"
REPO_BRANCH="${TURBOQUANT_BRANCH:-feature/turboquant-kv-cache}"

# TURBOQUANT_COMMIT pins the fork to one commit so every device builds the same
# source. Set it empty (TURBOQUANT_COMMIT= scripts/build.sh) to track the branch tip.
TURBOQUANT_COMMIT="${TURBOQUANT_COMMIT-}"

fetch_commit() {
    # GitHub serves any reachable SHA; fall back to the full branch if a host refuses.
    git -C "${SOURCE_DIR}" fetch --quiet --depth 1 origin "${TURBOQUANT_COMMIT}" 2>/dev/null \
        || git -C "${SOURCE_DIR}" fetch --quiet origin "${REPO_BRANCH}"
}

mkdir -p "$(dirname "${SOURCE_DIR}")"
if [[ -z "${TURBOQUANT_COMMIT}" ]]; then
    warn "TURBOQUANT_COMMIT is empty: building the ${REPO_BRANCH} tip, which is not reproducible"
    if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
        log "Cloning llama.cpp TurboQuant (${REPO_BRANCH})"
        git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${SOURCE_DIR}"
    else
        log "Using existing TurboQuant source: ${SOURCE_DIR}"
    fi
else
    if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
        log "Fetching llama.cpp TurboQuant at pinned commit ${TURBOQUANT_COMMIT:0:12}"
        git init --quiet "${SOURCE_DIR}"
        git -C "${SOURCE_DIR}" remote add origin "${REPO_URL}"
    fi
    if [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD 2>/dev/null || true)" != "${TURBOQUANT_COMMIT}" ]]; then
        [[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=no)" ]] \
            || die "${SOURCE_DIR} has local changes; commit or discard them before pinning to ${TURBOQUANT_COMMIT:0:12}"
        log "Checking out pinned commit ${TURBOQUANT_COMMIT:0:12}"
        fetch_commit
        git -C "${SOURCE_DIR}" checkout --quiet --detach "${TURBOQUANT_COMMIT}" \
            || die "pinned commit ${TURBOQUANT_COMMIT} not found in ${REPO_URL}"
    else
        log "TurboQuant source already at pinned commit ${TURBOQUANT_COMMIT:0:12}"
    fi
fi

if is_macos; then
    log "Building Metal backend for Apple Silicon"
    cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=ON \
        -DGGML_ACCELERATE=ON
else
    if [[ -n "${CUDA_ARCHITECTURES:-}" ]]; then
        CUDA_ARGS=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}")
    else
        CUDA_ARGS=()
    fi

    NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
    if [[ -z "${NVCC_BIN}" ]]; then
        for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
            if [[ -x "${candidate}" ]]; then
                NVCC_BIN="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${NVCC_BIN}" ]]; then
        export CUDACXX="${NVCC_BIN}"
        export PATH="$(dirname "${NVCC_BIN}"):${PATH}"
        log "CUDA compiler: $(nvcc --version | tail -n 1)"
        cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_CUDA=ON \
            "${CUDA_ARGS[@]}"
    elif [[ "${ALLOW_VULKAN_FALLBACK:-0}" == 1 ]]; then
        require_command vulkaninfo
        log "CUDA compiler unavailable; using Vulkan fallback"
        cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_VULKAN=ON
    else
        die "nvcc not found; install CUDA toolkit or set ALLOW_VULKAN_FALLBACK=1"
    fi
fi

cmake --build "${BUILD_DIR}" --config Release --target llama-server -j"$(cpu_threads)"
SERVER="${BUILD_DIR}/bin/llama-server"
if [[ ! -x "${SERVER}" ]]; then
    SERVER="$(find "${BUILD_DIR}" -type f \( -name llama-server -o -name lm-server-tq \) -perm -111 -print -quit)"
fi
[[ -n "${SERVER}" && -x "${SERVER}" ]] || die "build completed but no server executable was found"

printf 'SERVER_BIN=%s\n' "${SERVER}" >"${PROJECT_DIR}/config/runtime.env"
printf 'BUILD_SOURCE=%s\n' "${SOURCE_DIR}" >>"${PROJECT_DIR}/config/runtime.env"
printf 'BUILD_DIR=%s\n' "${BUILD_DIR}" >>"${PROJECT_DIR}/config/runtime.env"
printf 'BUILD_COMMIT=%s\n' "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" >>"${PROJECT_DIR}/config/runtime.env"

log "Server built: ${SERVER}"
record_log "llama.cpp TurboQuant build completed; server recorded in config/runtime.env"
