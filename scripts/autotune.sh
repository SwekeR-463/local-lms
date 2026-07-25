#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
load_optional_config "${PROJECT_DIR}/config/runtime.env"
load_optional_config "${PROJECT_DIR}/config/model.env"

require_command curl
require_command jq
require_command ps

QUICK=0
RESUME=0
DRY_RUN=0
MAX_CONTEXT="${PREFERRED_CONTEXT}"
while (($#)); do
    case "$1" in
        --quick) QUICK=1 ;;
        --resume) RESUME=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --max-context) shift; MAX_CONTEXT="${1:?missing value for --max-context}" ;;
        -h|--help)
            printf '%s\n' 'Usage: scripts/autotune.sh [--quick] [--resume] [--dry-run] [--max-context TOKENS]'
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

SERVER="$(resolve_server_bin)"
MODEL="$(resolve_model_path)"
HELP="$(server_help "${SERVER}")"
ensure_results_dir
RUN_ROOT="${RESULTS_DIR}/autotune-$(timestamp)"
SUMMARY="${RESULTS_DIR}/autotune-summary.json"
mkdir -p "${RUN_ROOT}"

if [[ "${RESUME}" == 1 && -f "${SUMMARY}" ]]; then
    RUN_ROOT="$(jq -r '.run_root // empty' "${SUMMARY}")"
    [[ -d "${RUN_ROOT}" ]] || die "resume summary points to missing directory: ${RUN_ROOT}"
fi

if [[ "${QUICK}" == 1 ]]; then
    contexts=(8192 32768 65536)
    kv_pairs=("q8_0,turbo4" "q8_0,turbo3")
    cpu_moe_values=(30 32)
    batch_values=(512)
    ubatch_values=(512)
else
    contexts=(8192 16384 32768 65536 98304 131072)
    kv_pairs=("q8_0,turbo4" "q8_0,turbo3" "q8_0,turbo2" "f16,turbo4")
    cpu_moe_values=(32 28 36 24)
    batch_values=(512 1024 2048)
    ubatch_values=(256 512 1024)
fi

filtered_contexts=()
for context in "${contexts[@]}"; do
    if ((context <= MAX_CONTEXT)); then filtered_contexts+=("${context}"); fi
done
contexts=("${filtered_contexts[@]}")
((${#contexts[@]} > 0)) || die "no context candidates at or below ${MAX_CONTEXT}"

candidate_args() {
    local context="$1" k_type="$2" v_type="$3" cpu_moe="$4" batch="$5" ubatch="$6"
    local -n output_ref="$7"
    output_ref=(-m "${MODEL}")
    add_candidate_value() {
        local flag="$1" value="$2"
        if has_flag "${HELP}" "${flag}"; then output_ref+=("${flag}" "${value}"); fi
    }
    add_candidate_value --host "${HOST}"
    add_candidate_value --port "${PORT}"
    add_candidate_value --ctx-size "${context}"
    add_candidate_value --threads "${THREADS:-8}"
    add_candidate_value --threads-batch "${THREADS_BATCH:-12}"
    add_candidate_value --parallel "${PARALLEL:-1}"
    add_candidate_value --cache-type-k "${k_type}"
    add_candidate_value --cache-type-v "${v_type}"
    add_candidate_value --ubatch-size "${ubatch}"
    add_candidate_value --batch-size "${batch}"
    add_candidate_value --n-cpu-moe "${cpu_moe}"
    add_candidate_value --seed "${SEED:-42}"
    if has_flag "${HELP}" "-ngl"; then output_ref+=(-ngl "${GPU_LAYERS:-99}"); fi
    if has_flag "${HELP}" "-fa"; then output_ref+=(-fa on); fi
    if has_flag "${HELP}" "--jinja"; then output_ref+=(--jinja); fi
    if has_flag "${HELP}" "--metrics"; then output_ref+=(--metrics); fi
}

wait_ready() {
    local deadline=$((SECONDS + 90))
    while ((SECONDS < deadline)); do
        if curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}

make_prompt() {
    local tokens="$1"
    local chars=$((tokens * 4))
    head -c "${chars}" < <(yes 'function main() { return "context benchmark"; } // ')
}

run_candidate() {
    local context="$1" k_type="$2" v_type="$3" cpu_moe="$4" batch="$5" ubatch="$6" index="$7"
    local candidate_dir="${RUN_ROOT}/candidate-${index}"
    local log_file="${candidate_dir}/server.log"
    local request_file="${candidate_dir}/request.json"
    local result_file="${candidate_dir}/result.json"
    mkdir -p "${candidate_dir}"

    local -a args
    candidate_args "${context}" "${k_type}" "${v_type}" "${cpu_moe}" "${batch}" "${ubatch}" args
    printf '%q ' "${SERVER}" "${args[@]}" >"${candidate_dir}/command.txt"
    printf '\n' >>"${candidate_dir}/command.txt"

    if ((DRY_RUN)); then
        log "DRY RUN candidate ${index}: ${context} ${k_type}/${v_type} n-cpu-moe=${cpu_moe} batch=${batch} ubatch=${ubatch}"
        return 0
    fi

    port_is_free "${HOST}" "${PORT}" || die "port ${HOST}:${PORT} is occupied by a non-project process"
    local pid rss_kb vram_mb status accepted response_file prompt request_started ended_at elapsed completion_tokens
    local started_at="$(date +%s%3N)"
    "${SERVER}" "${args[@]}" >"${log_file}" 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" >"${candidate_dir}/pid"

    if wait_ready; then status="ready"; else status="startup-timeout"; fi
    rss_kb="$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{print $1+0}')"
    vram_mb="unknown"
    if command -v nvidia-smi >/dev/null 2>&1; then
        vram_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || printf unknown)"
    fi

    accepted=false
    completion_tokens=0
    elapsed=0
    if [[ "${status}" == ready ]]; then
        prompt="$(make_prompt "${context}")"
        printf '%s' "${prompt}" >"${candidate_dir}/prompt.txt"
        jq -n --arg model "kat-coder" --rawfile prompt "${candidate_dir}/prompt.txt" --argjson seed "${SEED:-42}" \
            '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:32,temperature:0,seed:$seed}' >"${request_file}"
        response_file="${candidate_dir}/response.json"
        request_started="$(date +%s%3N)"
        if curl --fail --silent --show-error --max-time "${TUNE_TIMEOUT_SECONDS}" \
            -H 'Content-Type: application/json' --data-binary "@${request_file}" \
            "http://${HOST}:${PORT}/v1/chat/completions" >"${response_file}" 2>"${candidate_dir}/curl.err"; then
            accepted=true
            completion_tokens="$(jq -r '.usage.completion_tokens // 0' "${response_file}" 2>/dev/null || printf 0)"
            ended_at="$(date +%s%3N)"
            elapsed=$((ended_at - request_started))
        else
            status="http-failure"
            ended_at="$(date +%s%3N)"
            elapsed=$((ended_at - request_started))
        fi
    else
        ended_at="$(date +%s%3N)"
        elapsed=$((ended_at - started_at))
    fi

    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    jq -n \
        --argjson index "${index}" --argjson context "${context}" \
        --arg k_type "${k_type}" --arg v_type "${v_type}" \
        --argjson cpu_moe "${cpu_moe}" --argjson batch "${batch}" --argjson ubatch "${ubatch}" \
        --arg status "${status}" --argjson accepted "${accepted}" \
        --arg rss_kb "${rss_kb:-0}" --arg vram_mb "${vram_mb}" \
        --argjson elapsed_ms "${elapsed}" --argjson completion_tokens "${completion_tokens}" \
        --arg command "$(<"${candidate_dir}/command.txt")" --arg run_root "${RUN_ROOT}" \
        '{index:$index,context:$context,cache_type_k:$k_type,cache_type_v:$v_type,n_cpu_moe:$cpu_moe,batch_size:$batch,ubatch_size:$ubatch,status:$status,accepted:$accepted,rss_kb:($rss_kb|tonumber),vram_mb:(if $vram_mb=="unknown" then null else ($vram_mb|tonumber) end),elapsed_ms:$elapsed_ms,completion_tokens:$completion_tokens,command:$command,run_root:$run_root}' \
        >"${result_file}"
    cat "${result_file}"
}

results=()
index=0
for context in "${contexts[@]}"; do
    for kv in "${kv_pairs[@]}"; do
        IFS=',' read -r k_type v_type <<<"${kv}"
        for cpu_moe in "${cpu_moe_values[@]}"; do
            for batch in "${batch_values[@]}"; do
                for ubatch in "${ubatch_values[@]}"; do
                    index=$((index + 1))
                    result="$(run_candidate "${context}" "${k_type}" "${v_type}" "${cpu_moe}" "${batch}" "${ubatch}" "${index}")"
                    if ((DRY_RUN)); then continue; fi
                    results+=("${result}")
                done
            done
        done
    done
done

if ((DRY_RUN)); then
    log "Dry run complete: ${index} candidates"
    exit 0
fi

printf '%s\n' "${results[@]}" | jq -s --arg run_root "${RUN_ROOT}" --arg preferred "${PREFERRED_CONTEXT}" --arg minimum "${MIN_ACCEPTED_CONTEXT}" \
    '{run_root:$run_root,preferred_context:($preferred|tonumber),minimum_accepted_context:($minimum|tonumber),candidates:.}' >"${SUMMARY}"

WINNER="$(jq -r --argjson minimum "${MIN_ACCEPTED_CONTEXT}" '
    [.candidates[] | select(.accepted == true and .context >= $minimum)]
    | sort_by([.context, -(.completion_tokens / ([.elapsed_ms,1] | max))])
    | last // empty
' "${SUMMARY}")"

if [[ -z "${WINNER}" ]]; then
    warn "No candidate reached the minimum accepted context"
    record_log "autotune completed without a candidate at or above ${MIN_ACCEPTED_CONTEXT}; see ${SUMMARY#"${PROJECT_DIR}/"}"
    exit 1
fi

WINNER_CONTEXT="$(jq -r '.context' <<<"${WINNER}")"
jq --argjson winner "${WINNER}" '. + {winner:$winner, status:(if $winner.context >= .preferred_context then "success-preferred-context" else "success-minimum-context" end)}' "${SUMMARY}" >"${SUMMARY}.tmp"
mv "${SUMMARY}.tmp" "${SUMMARY}"

jq -r '"CTX_SIZE=" + (.context|tostring), "CACHE_TYPE_K=" + .cache_type_k, "CACHE_TYPE_V=" + .cache_type_v, "N_CPU_MOE=" + (.n_cpu_moe|tostring), "BATCH_SIZE=" + (.batch_size|tostring), "UBATCH_SIZE=" + (.ubatch_size|tostring)' <<<"${WINNER}" >"${PROJECT_DIR}/config/selected.env"
log "Selected context: ${WINNER_CONTEXT}"
log "Summary: ${SUMMARY}"
record_log "autotune selected ${WINNER_CONTEXT}-token configuration; summary saved at ${SUMMARY#"${PROJECT_DIR}/"}"
