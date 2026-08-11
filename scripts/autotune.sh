#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/common.sh"
MODEL_ARG=""
if [[ -n "${1:-}" && "${1}" != -* ]]; then MODEL_ARG="$1"; shift; fi
load_model_profile "${MODEL_ARG}"

require_command curl
require_command jq
require_command ps

QUICK=0
RESUME=0
DRY_RUN=0
CPU_MOE_SWEEP=0
MIN_CONTEXT=0
MAX_CONTEXT="${PREFERRED_CONTEXT}"
while (($#)); do
    case "$1" in
        --quick) QUICK=1 ;;
        --resume) RESUME=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --cpu-moe-sweep) CPU_MOE_SWEEP=1 ;;
        --min-context) shift; MIN_CONTEXT="${1:?missing value for --min-context}" ;;
        --max-context) shift; MAX_CONTEXT="${1:?missing value for --max-context}" ;;
        -h|--help)
            printf '%s\n' 'Usage: scripts/autotune.sh [MODEL] [--quick] [--cpu-moe-sweep] [--resume] [--dry-run] [--min-context TOKENS] [--max-context TOKENS]'
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
if ((CPU_MOE_SWEEP)); then
    RUN_ROOT="${RESULTS_DIR}/cpu-moe-${MODEL_ID}-$(timestamp)"
    SUMMARY="${RESULTS_DIR}/cpu-moe-${MODEL_ID}-summary.json"
    MAX_TOKENS=128
else
    RUN_ROOT="${RESULTS_DIR}/autotune-${MODEL_ID}-$(timestamp)"
    SUMMARY="${RESULTS_DIR}/autotune-${MODEL_ID}-summary.json"
    MAX_TOKENS=32
fi
mkdir -p "${RUN_ROOT}"

if [[ "${RESUME}" == 1 && -f "${SUMMARY}" ]]; then
    RUN_ROOT="$(jq -r '.run_root // empty' "${SUMMARY}")"
    [[ -d "${RUN_ROOT}" ]] || die "resume summary points to missing directory: ${RUN_ROOT}"
fi

if is_macos; then
    cpu_moe_values=(0)
    if [[ "${QUICK}" == 1 ]]; then
        contexts=(8192 32768 65536)
        kv_pairs=("q8_0,turbo3")
        batch_values=(1024)
        ubatch_values=(1024)
    else
        contexts=(8192 16384 32768 65536 98304 131072)
        kv_pairs=("q8_0,turbo3" "q8_0,turbo4")
        batch_values=(1024 2048)
        ubatch_values=(1024 2048)
    fi
elif [[ "${QUICK}" == 1 ]]; then
    contexts=(8192 32768 65536)
    kv_pairs=("q8_0,turbo4" "q8_0,turbo3")
    cpu_moe_values=(0)
    batch_values=(512)
    ubatch_values=(512)
else
    contexts=(8192 16384 32768 65536 98304 131072)
    kv_pairs=("q8_0,turbo4" "q8_0,turbo3" "q8_0,turbo2" "f16,turbo4")
    cpu_moe_values=(0 8 16 32)
    batch_values=(512 1024 2048)
    ubatch_values=(256 512 1024)
fi

[[ -z "${AUTOTUNE_KV_PAIRS:-}" ]] || read -r -a kv_pairs <<<"${AUTOTUNE_KV_PAIRS}"
[[ -z "${AUTOTUNE_BATCH_VALUES:-}" ]] || read -r -a batch_values <<<"${AUTOTUNE_BATCH_VALUES}"
[[ -z "${AUTOTUNE_UBATCH_VALUES:-}" ]] || read -r -a ubatch_values <<<"${AUTOTUNE_UBATCH_VALUES}"

if ((CPU_MOE_SWEEP)); then
    cpu_moe_values=(0 8 16 24 32 40)
    kv_pairs=("${CACHE_TYPE_K:-q8_0},${CACHE_TYPE_V:-$(default_cache_v)}")
    batch_values=("${BATCH_SIZE:-$(default_batch_size)}")
    ubatch_values=("${UBATCH_SIZE:-$(default_batch_size)}")
fi

filtered_contexts=()
for context in "${contexts[@]}"; do
    if ((context >= MIN_CONTEXT && context <= MAX_CONTEXT)); then filtered_contexts+=("${context}"); fi
done
contexts=("${filtered_contexts[@]}")
((${#contexts[@]} > 0)) || die "no context candidates between ${MIN_CONTEXT} and ${MAX_CONTEXT}"

candidate_args() {
    local context="$1" k_type="$2" v_type="$3" cpu_moe="$4" batch="$5" ubatch="$6"
    CANDIDATE_ARGS=(-m "${MODEL}")
    add_candidate_value() {
        local flag="$1" value="$2"
        if has_flag "${HELP}" "${flag}"; then CANDIDATE_ARGS+=("${flag}" "${value}"); fi
    }
    add_candidate_value --host "${HOST}"
    add_candidate_value --port "${PORT}"
    add_candidate_value --ctx-size "${context}"
    add_candidate_value --threads "${THREADS:-$(runtime_threads)}"
    add_candidate_value --threads-batch "${THREADS_BATCH:-$(default_threads_batch)}"
    add_candidate_value --parallel "${PARALLEL:-1}"
    add_candidate_value --cache-type-k "${k_type}"
    add_candidate_value --cache-type-v "${v_type}"
    add_candidate_value --ubatch-size "${ubatch}"
    add_candidate_value --batch-size "${batch}"
    add_candidate_value --n-cpu-moe "${cpu_moe}"
    add_candidate_value --seed "${SEED:-42}"
    if [[ -n "${SPEC_TYPE:-}" ]]; then
        add_candidate_value --spec-type "${SPEC_TYPE}"
        add_candidate_value --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-1}"
        add_candidate_value --spec-draft-n-min "${SPEC_DRAFT_N_MIN:-0}"
        add_candidate_value --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0.75}"
        add_candidate_value --spec-ngram-mod-n-min "${SPEC_NGRAM_MOD_N_MIN:-8}"
        add_candidate_value --spec-ngram-mod-n-max "${SPEC_NGRAM_MOD_N_MAX:-24}"
        add_candidate_value --spec-ngram-mod-n-match "${SPEC_NGRAM_MOD_N_MATCH:-48}"
    fi
    if has_flag "${HELP}" "-ngl"; then CANDIDATE_ARGS+=(-ngl "${GPU_LAYERS:-99}"); fi
    if has_flag "${HELP}" "-fa"; then CANDIDATE_ARGS+=(-fa on); fi
    if has_flag "${HELP}" "--jinja"; then CANDIDATE_ARGS+=(--jinja); fi
    if has_flag "${HELP}" "--metrics"; then CANDIDATE_ARGS+=(--metrics); fi
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

    candidate_args "${context}" "${k_type}" "${v_type}" "${cpu_moe}" "${batch}" "${ubatch}"
    local -a args=("${CANDIDATE_ARGS[@]}")
    printf '%q ' "${SERVER}" "${args[@]}" >"${candidate_dir}/command.txt"
    printf '\n' >>"${candidate_dir}/command.txt"

    if ((DRY_RUN)); then
        log "DRY RUN candidate ${index}: ${context} ${k_type}/${v_type} n-cpu-moe=${cpu_moe} batch=${batch} ubatch=${ubatch}"
        return 0
    fi

    port_is_free "${HOST}" "${PORT}" || die "port ${HOST}:${PORT} is occupied by a non-project process"
    local pid rss_kb rss_after vram_mb status accepted response_file prompt request_started ended_at elapsed completion_tokens prompt_tokens prompt_tps generation_tps
    local started_at="$(epoch_ms)"
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
    prompt_tokens=0
    prompt_tps=0
    generation_tps=0
    elapsed=0
    if [[ "${status}" == ready ]]; then
        prompt="$(make_prompt "${context}")"
        printf '%s' "${prompt}" >"${candidate_dir}/prompt.txt"
        jq -n --arg model "${MODEL_ID}" --rawfile prompt "${candidate_dir}/prompt.txt" --argjson seed "${SEED:-42}" --argjson max_tokens "${MAX_TOKENS}" \
            '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,temperature:0,seed:$seed}' >"${request_file}"
        response_file="${candidate_dir}/response.json"
        request_started="$(epoch_ms)"
        if curl --fail --silent --show-error --max-time "${TUNE_TIMEOUT_SECONDS}" \
            -H 'Content-Type: application/json' --data-binary "@${request_file}" \
            "http://${HOST}:${PORT}/v1/chat/completions" >"${response_file}" 2>"${candidate_dir}/curl.err"; then
            accepted=true
            completion_tokens="$(jq -r '.usage.completion_tokens // 0' "${response_file}" 2>/dev/null || printf 0)"
            prompt_tokens="$(jq -r '.usage.prompt_tokens // 0' "${response_file}" 2>/dev/null || printf 0)"
            prompt_tps="$(jq -r '.timings.prompt_per_second // 0' "${response_file}" 2>/dev/null || printf 0)"
            generation_tps="$(jq -r '.timings.predicted_per_second // 0' "${response_file}" 2>/dev/null || printf 0)"
            ended_at="$(epoch_ms)"
            elapsed=$((ended_at - request_started))
        else
            status="http-failure"
            ended_at="$(epoch_ms)"
            elapsed=$((ended_at - request_started))
        fi
    else
        ended_at="$(epoch_ms)"
        elapsed=$((ended_at - started_at))
    fi

    rss_after="$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{print $1+0}')"
    ((rss_after > rss_kb)) && rss_kb="${rss_after}"
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    jq -n \
        --argjson index "${index}" --argjson context "${context}" \
        --arg k_type "${k_type}" --arg v_type "${v_type}" \
        --argjson cpu_moe "${cpu_moe}" --argjson batch "${batch}" --argjson ubatch "${ubatch}" \
        --arg status "${status}" --argjson accepted "${accepted}" \
        --arg rss_kb "${rss_kb:-0}" --arg vram_mb "${vram_mb}" \
        --argjson elapsed_ms "${elapsed}" --argjson completion_tokens "${completion_tokens}" \
        --argjson prompt_tokens "${prompt_tokens}" --argjson prompt_tps "${prompt_tps}" --argjson generation_tps "${generation_tps}" \
        --arg command "$(<"${candidate_dir}/command.txt")" --arg run_root "${RUN_ROOT}" \
        '{index:$index,context:$context,cache_type_k:$k_type,cache_type_v:$v_type,n_cpu_moe:$cpu_moe,batch_size:$batch,ubatch_size:$ubatch,status:$status,accepted:$accepted,rss_kb:($rss_kb|tonumber),vram_mb:(if $vram_mb=="unknown" then null else ($vram_mb|tonumber) end),elapsed_ms:$elapsed_ms,prompt_tokens:$prompt_tokens,completion_tokens:$completion_tokens,prompt_tokens_per_second:$prompt_tps,generation_tokens_per_second:$generation_tps,command:$command,run_root:$run_root}' \
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
                    ((ubatch <= batch)) || continue
                    index=$((index + 1))
                    if ((DRY_RUN)); then
                        run_candidate "${context}" "${k_type}" "${v_type}" "${cpu_moe}" "${batch}" "${ubatch}" "${index}"
                        continue
                    fi
                    results+=("$(run_candidate "${context}" "${k_type}" "${v_type}" "${cpu_moe}" "${batch}" "${ubatch}" "${index}")")
                done
            done
        done
    done
done

if ((DRY_RUN)); then
    log "Dry run complete: ${index} candidates"
    exit 0
fi

printf '%s\n' "${results[@]}" | jq -s --arg model_id "${MODEL_ID}" --arg model_name "${MODEL_NAME:-${MODEL_ID}}" --arg model_file "${MODEL_FILE:-${MODEL_PATH:-}}" --arg run_root "${RUN_ROOT}" --arg preferred "${PREFERRED_CONTEXT}" --arg minimum "${MIN_ACCEPTED_CONTEXT}" \
    '{model_id:$model_id,model_name:$model_name,model_file:$model_file,run_root:$run_root,preferred_context:($preferred|tonumber),minimum_accepted_context:($minimum|tonumber),candidates:.}' >"${SUMMARY}"

if ((CPU_MOE_SWEEP)); then
    jq '. + {mode:"cpu-moe-sweep",winners_by_context:([.candidates[] | select(.accepted == true)] | group_by(.context) | map(sort_by(.generation_tokens_per_second) | last))}' \
        "${SUMMARY}" >"${SUMMARY}.tmp"
    mv "${SUMMARY}.tmp" "${SUMMARY}"
    WINNER="$(jq -r '.winners_by_context | sort_by(.context) | last // empty' "${SUMMARY}")"
else
    WINNER="$(jq -r --argjson minimum "${MIN_ACCEPTED_CONTEXT}" '
        [.candidates[] | select(.accepted == true and .context >= $minimum)]
        | sort_by([.context, (.completion_tokens / ([.elapsed_ms,1] | max))])
        | last // empty
    ' "${SUMMARY}")"
fi

if [[ -z "${WINNER}" ]]; then
    warn "No candidate reached the requested context"
    record_log "autotune completed without a winner; see ${SUMMARY#"${PROJECT_DIR}/"}"
    exit 1
fi

WINNER_CONTEXT="$(jq -r '.context' <<<"${WINNER}")"
jq --argjson winner "${WINNER}" '. + {winner:$winner, status:(if $winner.context >= .preferred_context then "success-preferred-context" else "success-minimum-context" end)}' "${SUMMARY}" >"${SUMMARY}.tmp"
mv "${SUMMARY}.tmp" "${SUMMARY}"

mkdir -p "${PROJECT_DIR}/config/local"
jq -r '"CTX_SIZE=" + (.context|tostring), "CACHE_TYPE_K=" + .cache_type_k, "CACHE_TYPE_V=" + .cache_type_v, "N_CPU_MOE=" + (.n_cpu_moe|tostring), "BATCH_SIZE=" + (.batch_size|tostring), "UBATCH_SIZE=" + (.ubatch_size|tostring)' <<<"${WINNER}" >"$(selected_config)"
log "Selected context: ${WINNER_CONTEXT}"
log "Summary: ${SUMMARY}"
if ((CPU_MOE_SWEEP)); then
    record_log "CPU MoE sweep completed; per-context winners saved at ${SUMMARY#"${PROJECT_DIR}/"}"
else
    record_log "autotune selected ${WINNER_CONTEXT}-token configuration; summary saved at ${SUMMARY#"${PROJECT_DIR}/"}"
fi
