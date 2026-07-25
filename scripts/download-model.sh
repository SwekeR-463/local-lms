#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"

require_command curl
require_command jq

ensure_results_dir

RESOLVE_ONLY=0
case "${1:-}" in
    --resolve-only) RESOLVE_ONLY=1; shift ;;
    -h|--help)
        printf '%s\n' 'Usage: scripts/download-model.sh [--resolve-only]'
        exit 0
        ;;
esac
[[ "$#" -eq 0 ]] || die "unknown arguments: $*"

API_URL="https://huggingface.co/api/models/${MODEL_REPO}"
MODEL_DIR="$(project_path "${MODEL_DIR:-models}")"
mkdir -p "${MODEL_DIR}"

log "Querying ${API_URL}"
MODEL_JSON="$(curl --fail --silent --show-error --location "${API_URL}")" || die "could not query model repository"
mapfile -t FILES < <(jq -r '.siblings[]?.rfilename // empty' <<<"${MODEL_JSON}")
((${#FILES[@]} > 0)) || die "repository has no visible files: ${MODEL_REPO}"

contains_file() {
    local wanted="$1"
    local item
    for item in "${FILES[@]}"; do
        [[ "${item}" == "${wanted}" ]] && return 0
    done
    return 1
}

MODEL_FILE_RESOLVED="${MODEL_FILE:-}"
if [[ -n "${MODEL_FILE_RESOLVED}" ]]; then
    contains_file "${MODEL_FILE_RESOLVED}" || die "MODEL_FILE is not present in ${MODEL_REPO}: ${MODEL_FILE_RESOLVED}"
else
    profile="${MODEL_PROFILE:-I-Mini}"
    case "${profile}" in
        I-Mini) patterns=('APEX-I-Mini.gguf' 'I-Mini.gguf') ;;
        Mini) patterns=('APEX-Mini.gguf' '-Mini.gguf') ;;
        I-Compact) patterns=('APEX-I-Compact.gguf' 'I-Compact.gguf') ;;
        Compact) patterns=('APEX-Compact.gguf' '-Compact.gguf') ;;
        MTP-I-Compact) patterns=('APEX-MTP-I-Compact.gguf' 'MTP-I-Compact.gguf') ;;
        *) die "unsupported MODEL_PROFILE: ${profile}" ;;
    esac

    for suffix in "${patterns[@]}"; do
        for item in "${FILES[@]}"; do
            if [[ "${item}" == *.gguf && "${item}" == *"${suffix}" ]]; then
                MODEL_FILE_RESOLVED="${item}"
                break 2
            fi
        done
    done
    [[ -n "${MODEL_FILE_RESOLVED}" ]] || die "could not find a ${profile} GGUF in ${MODEL_REPO}; visible files: $(printf '%s ' "${FILES[@]}")"
fi

if [[ "${MODEL_FILE_RESOLVED}" == */* ]]; then
    relative_name="${MODEL_FILE_RESOLVED}"
    output_name="$(basename "${MODEL_FILE_RESOLVED}")"
else
    relative_name="${MODEL_FILE_RESOLVED}"
    output_name="${MODEL_FILE_RESOLVED}"
fi

OUTPUT_PATH="${MODEL_DIR}/${output_name}"
DOWNLOAD_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${relative_name}?download=true"
log "Selected model file: ${MODEL_FILE_RESOLVED}"
log "Destination: ${OUTPUT_PATH}"

if ((RESOLVE_ONLY)); then
    REMOTE_SIZE="$(curl --fail --silent --show-error --location --head "${DOWNLOAD_URL}" | tr -d '\r' | awk -F': ' 'tolower($1) == "content-length" {value=$2} END {print value+0}')"
    log "Remote size: ${REMOTE_SIZE} bytes"
    exit 0
fi

if [[ -f "${OUTPUT_PATH}" && -s "${OUTPUT_PATH}" ]]; then
    log "File already exists; skipping download"
else
    curl --fail --location --continue-at - --output "${OUTPUT_PATH}.part" "${DOWNLOAD_URL}"
    mv "${OUTPUT_PATH}.part" "${OUTPUT_PATH}"
fi

[[ -s "${OUTPUT_PATH}" ]] || die "downloaded model is empty: ${OUTPUT_PATH}"
MODEL_BYTES="$(stat -c '%s' "${OUTPUT_PATH}")"
MODEL_SHA256="$(sha256sum "${OUTPUT_PATH}" | awk '{print $1}')"
MODEL_SIZE="$(jq -r --arg file "${MODEL_FILE_RESOLVED}" '.siblings[] | select(.rfilename == $file) | (.size // 0)' <<<"${MODEL_JSON}")"
MODEL_SHA256_HF="$(jq -r --arg file "${MODEL_FILE_RESOLVED}" '.siblings[] | select(.rfilename == $file) | (.lfs.sha256 // .sha // "")' <<<"${MODEL_JSON}")"
if [[ "${MODEL_SIZE}" == "0" || -z "${MODEL_SIZE}" ]]; then
    MODEL_SIZE="$(curl --fail --silent --show-error --location --head "${DOWNLOAD_URL}" | tr -d '\r' | awk -F': ' 'tolower($1) == "content-length" {value=$2} END {print value+0}')"
fi

jq -n \
    --arg repo "${MODEL_REPO}" \
    --arg profile "${MODEL_PROFILE:-explicit}" \
    --arg file "${MODEL_FILE_RESOLVED}" \
    --arg path "${OUTPUT_PATH}" \
    --arg bytes "${MODEL_BYTES}" \
    --arg advertised_bytes "${MODEL_SIZE}" \
    --arg sha256 "${MODEL_SHA256}" \
    --arg advertised_sha256 "${MODEL_SHA256_HF}" \
    --arg downloaded_at "$(date --iso-8601=seconds)" \
    '{repository:$repo, profile:$profile, filename:$file, path:$path, bytes:($bytes|tonumber), advertised_bytes:($advertised_bytes|tonumber), sha256:$sha256, advertised_sha256:$advertised_sha256, downloaded_at:$downloaded_at}' \
    >"${RESULTS_DIR}/model.json"

MODEL_FILE="${output_name}"
cat >"${PROJECT_DIR}/config/model.env" <<EOF
MODEL_REPO=${MODEL_REPO}
MODEL_PROFILE=${MODEL_PROFILE:-explicit}
MODEL_FILE=${MODEL_FILE}
MODEL_PATH=${OUTPUT_PATH}
EOF

log "Model ready: $(numfmt --to=iec "${MODEL_BYTES}")"
record_log "model resolver selected ${MODEL_FILE} (${MODEL_BYTES} bytes); metadata saved to results/model.json"
