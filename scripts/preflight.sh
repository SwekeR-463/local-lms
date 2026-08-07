#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
MODEL_ARG=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then MODEL_ARG="$1"; shift; fi
load_model_profile "${MODEL_ARG}"

ALLOW_MISSING_MODEL=0
ALLOW_MISSING_RUNTIME=0
DIAGNOSE=0

usage() {
    cat <<'EOF'
Usage: scripts/preflight.sh [MODEL] [options]

Options:
  --allow-missing-model    Check the host before the model is downloaded.
  --allow-missing-runtime  Check the host before the server is built.
  --diagnose               Print diagnostics without failing on known blockers.
EOF
}

while (($#)); do
    case "$1" in
        --allow-missing-model) ALLOW_MISSING_MODEL=1 ;;
        --allow-missing-runtime) ALLOW_MISSING_RUNTIME=1 ;;
        --diagnose) DIAGNOSE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

ensure_results_dir
PREFLIGHT_FILE="${RESULTS_DIR}/preflight-$(timestamp).txt"
exec > >(tee "${PREFLIGHT_FILE}") 2>&1

log "Starting preflight"
log "Project: ${PROJECT_DIR}"
log "Model profile: ${MODEL_ID} (${MODEL_CONFIG})"

required_commands=(cmake git curl awk sed perl)
for command_name in "${required_commands[@]}"; do
    if command -v "${command_name}" >/dev/null 2>&1; then
        log "command ok: ${command_name} ($(command -v "${command_name}"))"
    else
        if ((DIAGNOSE)); then
            warn "missing command: ${command_name}"
        else
            die "required command missing: ${command_name}"
        fi
    fi
done

if command -v jq >/dev/null 2>&1; then
    log "command ok: jq ($(command -v jq))"
else
    if ((DIAGNOSE)); then warn "missing command: jq"; else die "required command missing: jq"; fi
fi

if command -v g++ >/dev/null 2>&1; then
    log "compiler: $(g++ --version | head -n 1)"
elif command -v clang++ >/dev/null 2>&1; then
    log "compiler: $(clang++ --version | head -n 1)"
else
    if ((DIAGNOSE)); then warn "no C++ compiler found"; else die "no C++ compiler found"; fi
fi

log "OS: $(uname -srm)"
if is_macos; then
    log "CPU: $(sysctl -n machdep.cpu.brand_string)"
    log "CPU cores: $(cpu_threads) total, $(runtime_threads) performance"
    TOTAL_RAM_BYTES="$(sysctl -n hw.memsize)"
    log "Unified memory: $(human_bytes "${TOTAL_RAM_BYTES}")"
    GPU_INFO="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model|Total Number of Cores|Metal Support/ {gsub(/^ +/, "", $1); printf "%s%s: %s", separator, $1, $2; separator=", "}')"
    [[ -n "${GPU_INFO}" ]] || die "Metal GPU not detected"
    log "GPU: ${GPU_INFO}"
    log "Swap: $(sysctl -n vm.swapusage)"
else
    log "CPU: $(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}' || true)"
    log "CPU threads: $(cpu_threads 2>/dev/null || printf unknown)"
    log "RAM: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2 " total, " $7 " available"}' || printf unknown)"
    log "NUMA: $(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/ {gsub(/^ +/, "", $2); print $2; exit}' || printf unknown)"

    GPU_PRESENT=0
    if command -v lspci >/dev/null 2>&1 && lspci | grep -Eiq 'VGA|3D|Display' && lspci | grep -Eiq 'NVIDIA|AMD|Intel'; then
        GPU_PRESENT=1
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)"; then
            log "NVIDIA GPU: ${GPU_INFO}"
        elif ((DIAGNOSE)); then
            warn "NVIDIA PCI device/tool exists but nvidia-smi failed: ${GPU_INFO}"
        else
            die "NVIDIA driver/runtime unavailable: ${GPU_INFO}"
        fi
    elif ((GPU_PRESENT)); then
        if ((DIAGNOSE)); then warn "GPU detected but nvidia-smi is not installed"; else die "GPU detected but nvidia-smi is not installed"; fi
    else
        warn "no supported GPU detected"
    fi

    if swapon --show --noheadings 2>/dev/null | grep -q .; then
        log "Swap: active"
        if [[ "${ALLOW_SWAP:-0}" != 1 && "${DIAGNOSE}" == 0 ]]; then
            die "swap is active; set ALLOW_SWAP=1 only for an explicit diagnostic run"
        elif [[ "${ALLOW_SWAP:-0}" != 1 ]]; then
            warn "swap is active; this run is diagnostic only"
        fi
    else
        log "Swap: inactive"
    fi
fi
log "Disk: $(df -h "${PROJECT_DIR}" | awk 'NR==2 {print $4 " free on " $1}')"

if ! port_is_free "${HOST}" "${PORT}"; then
    die "port ${HOST}:${PORT} is already in use"
fi
log "Port ${HOST}:${PORT}: available"

if [[ -n "${MODEL_PATH:-}" || -n "${MODEL_FILE:-}" ]]; then
    if [[ -n "${MODEL_PATH:-}" ]]; then
        CHECK_MODEL="${MODEL_PATH}"
    else
        CHECK_MODEL="$(project_path "${MODEL_DIR:-models}/${MODEL_FILE}")"
    fi
    if [[ "${CHECK_MODEL}" != /* ]]; then CHECK_MODEL="${PROJECT_DIR}/${CHECK_MODEL}"; fi
    if [[ -f "${CHECK_MODEL}" ]]; then
        MODEL_BYTES="$(file_size "${CHECK_MODEL}")"
        if is_macos; then AVAILABLE_BYTES="$(sysctl -n hw.memsize)"; else AVAILABLE_BYTES="$(awk '/MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo)"; fi
        log "Model: ${CHECK_MODEL} ($(human_bytes "${MODEL_BYTES}"))"
        log "Memory budget: $(human_bytes "${AVAILABLE_BYTES}")"
        if ((MODEL_BYTES > AVAILABLE_BYTES)); then
            warn "model file is larger than currently available RAM; GPU offload may reduce CPU residency, but this is high risk"
        fi
    elif ((ALLOW_MISSING_MODEL == 0 && DIAGNOSE == 0)); then
        die "model file not found: ${CHECK_MODEL}"
    else
        warn "model file not found yet: ${CHECK_MODEL}"
    fi
else
    warn "MODEL_PATH/MODEL_FILE not set; model checks skipped"
fi

if SERVER="$(resolve_server_bin 2>/dev/null)"; then
    log "Server: ${SERVER}"
    HELP="$(server_help "${SERVER}")"
    for flag in --ctx-size --cache-type-k --cache-type-v --n-cpu-moe; do
        if has_flag "${HELP}" "${flag}"; then log "server flag supported: ${flag}"; else warn "server flag not found: ${flag}"; fi
    done
else
    if ((ALLOW_MISSING_RUNTIME == 0 && DIAGNOSE == 0)); then die "server binary not found"; fi
    warn "server binary not found yet"
fi

log "Preflight complete: ${PREFLIGHT_FILE}"
record_log "preflight completed; report saved at ${PREFLIGHT_FILE#"${PROJECT_DIR}/"}"
