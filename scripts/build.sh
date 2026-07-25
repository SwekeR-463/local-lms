#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

case "${1:-}" in
    -h|--help)
        printf '%s\n' 'Usage: scripts/build.sh'
        printf '%s\n' 'Build the TurboQuant llama-server with CUDA, or Vulkan when explicitly enabled.'
        exit 0
        ;;
    "") ;;
    *) die "unknown option: $1" ;;
esac

require_command git
require_command cmake

SOURCE_DIR="$(project_path "${SOURCE_DIR}")"
BUILD_DIR="$(project_path "${BUILD_DIR}")"
REPO_URL="https://github.com/TheTom/llama-cpp-turboquant.git"
REPO_BRANCH="${TURBOQUANT_BRANCH:-feature/turboquant-kv-cache}"

mkdir -p "$(dirname "${SOURCE_DIR}")"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    log "Cloning TurboQuant fork (${REPO_BRANCH})"
    git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${SOURCE_DIR}"
else
    log "Using existing TurboQuant source: ${SOURCE_DIR}"
fi

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
else
    if [[ "${ALLOW_VULKAN_FALLBACK:-0}" == 1 ]]; then
        require_command vulkaninfo
        log "CUDA compiler unavailable; using Vulkan fallback"
        cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DGGML_VULKAN=ON
    else
        die "nvcc not found; install CUDA toolkit or set ALLOW_VULKAN_FALLBACK=1"
    fi
fi

cmake --build "${BUILD_DIR}" --config Release --target llama-server -j"$(nproc)"
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
record_log "TurboQuant build completed; server recorded in config/runtime.env"
