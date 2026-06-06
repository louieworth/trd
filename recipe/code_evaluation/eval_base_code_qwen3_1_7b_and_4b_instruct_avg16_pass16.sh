#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VERL_ROOT"

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export TASK="code"

BENCHMARK_SCRIPT="$VERL_ROOT/recipe/code_evaluation/benchmark_code_model.sh"

# Code base eval: Qwen3-1.7B and Qwen3-4B-Instruct-2507, avg@16/pass@16.
export DATASETS="${DATASETS:-humaneval_plus mbpp_plus livecodebench_v6}"
export PASS_K="${PASS_K:-16}"
export CODE_EVAL_TEMPERATURE="${CODE_EVAL_TEMPERATURE:-0.6}"
export CODE_EVAL_TOP_P="${CODE_EVAL_TOP_P:-0.95}"
export CODE_EVAL_MAX_PROMPT_TOKENS="${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}"
export CODE_EVAL_MAX_RESPONSE_TOKENS="${CODE_EVAL_MAX_RESPONSE_TOKENS:-16384}"
export CODE_EVAL_MAX_MODEL_LEN="${CODE_EVAL_MAX_MODEL_LEN:-$((CODE_EVAL_MAX_PROMPT_TOKENS + CODE_EVAL_MAX_RESPONSE_TOKENS))}"
export CODE_EVAL_MAX_NUM_SEQS="${CODE_EVAL_MAX_NUM_SEQS:-128}"

# benchmark_code_model.sh uses GEN_TP for vLLM tensor parallelism.
detect_visible_gpu_count() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]; then
        local count=0 gpu_id
        for gpu_id in ${CUDA_VISIBLE_DEVICES//,/ }; do
            if [ -n "$gpu_id" ]; then
                count=$((count + 1))
            fi
        done
        if [ "$count" -gt 0 ]; then
            printf '%s\n' "$count"
            return 0
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
            printf '%s\n' "$count"
            return 0
        fi
    fi

    printf '1\n'
}

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-$(detect_visible_gpu_count)}"
export GEN_TP="${GEN_TP:-$NGPUS_PER_NODE}"

QWEN3_1_7B_BASE_MODEL_PATH="${QWEN3_1_7B_BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
QWEN3_4B_INSTRUCT_BASE_MODEL_PATH="${QWEN3_4B_INSTRUCT_BASE_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

BASE_EVAL_RESULTS_DIR="${BASE_EVAL_RESULTS_DIR:-$VERL_ROOT/results/base/code}"
BASE_EVAL_OUTPUT_BASE_DIR="${BASE_EVAL_OUTPUT_BASE_DIR:-$VERL_ROOT/gen_results/eval/code/base}"
BASE_EVAL_LOG_DIR="${BASE_EVAL_LOG_DIR:-$VERL_ROOT/outputs/OPD/code/base_eval_logs}"
mkdir -p "$BASE_EVAL_RESULTS_DIR" "$BASE_EVAL_OUTPUT_BASE_DIR" "$BASE_EVAL_LOG_DIR"
BASE_EVAL_LOG="$BASE_EVAL_LOG_DIR/eval_base_code_avg16_pass16_$(date +%Y%m%d_%H%M%S).log"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$BASE_EVAL_LOG"
}

hf_snapshot_dir_name() {
    local model_id="$1"
    printf 'models--%s\n' "${model_id//\//--}"
}

resolve_model_path() {
    local requested_path="$1"
    local model_name="$2"
    local env_name="$3"

    if [ -d "$requested_path" ]; then
        printf '%s\n' "$requested_path"
        return 0
    fi

    if [[ "$requested_path" == */* ]]; then
        local repo_dir snapshots_dir snapshot cache_dir
        repo_dir="$(hf_snapshot_dir_name "$requested_path")"
        local cache_dirs=()
        if [ -n "${TRANSFORMERS_CACHE:-}" ]; then cache_dirs+=("$TRANSFORMERS_CACHE"); fi
        if [ -n "${HF_HOME:-}" ]; then cache_dirs+=("$HF_HOME/hub"); fi
        if [ -n "${HOME:-}" ]; then cache_dirs+=("$HOME/.cache/huggingface/hub"); fi
        cache_dirs+=("${HF_HOME:-$HOME/.cache/huggingface}/hub")

        for cache_dir in "${cache_dirs[@]}"; do
            snapshots_dir="$cache_dir/$repo_dir/snapshots"
            if [ ! -d "$snapshots_dir" ]; then continue; fi
            snapshot="$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
            if [ -n "$snapshot" ] && [ -d "$snapshot" ]; then
                printf '%s\n' "$snapshot"
                return 0
            fi
        done
    fi

    {
        echo "ERROR: local model directory not found for $model_name: $requested_path"
        echo "Set $env_name to a local Hugging Face model directory, for example a snapshots/<revision> path."
    } >&2
    return 1
}

run_base_eval() {
    local model_name="$1"
    local requested_path="$2"
    local env_name="$3"
    local model_path results_file results_csv output_dir

    model_path="$(resolve_model_path "$requested_path" "$model_name" "$env_name")"
    results_file="$BASE_EVAL_RESULTS_DIR/${model_name}_code_avg16_pass16.json"
    results_csv="${results_file%.json}.csv"
    output_dir="$BASE_EVAL_OUTPUT_BASE_DIR/$model_name"
    mkdir -p "$output_dir" "$(dirname "$results_file")"

    log "=========================================="
    log "START base code eval: $model_name"
    log "Model path: $model_path"
    log "Datasets: $DATASETS"
    log "PASS_K=$PASS_K temperature=$CODE_EVAL_TEMPERATURE top_p=$CODE_EVAL_TOP_P"
    log "max_prompt=$CODE_EVAL_MAX_PROMPT_TOKENS max_response=$CODE_EVAL_MAX_RESPONSE_TOKENS max_model_len=$CODE_EVAL_MAX_MODEL_LEN max_num_seqs=$CODE_EVAL_MAX_NUM_SEQS"
    log "NGPUS_PER_NODE=$NGPUS_PER_NODE GEN_TP=$GEN_TP"
    log "Results: $results_file"
    log "Output: $output_dir"
    log "=========================================="

    (
        cd "$VERL_ROOT"
        EVAL_BASE_MODEL_NAME="$model_name" \
        EVAL_MODEL_NAME="$model_name" \
        EVAL_RESULTS_FILE="$results_file" \
        EVAL_RESULTS_CSV_FILE="$results_csv" \
        EVAL_OUTPUT_DIR="$output_dir" \
        bash "$BENCHMARK_SCRIPT" "$model_path"
    ) 2>&1 | tee -a "$BASE_EVAL_LOG"

    log "DONE base code eval: $model_name"
}

log "Starting code-only base performance evaluation"
log "Benchmark: $BENCHMARK_SCRIPT"
log "Log: $BASE_EVAL_LOG"

run_base_eval "Qwen3-1.7B" "$QWEN3_1_7B_BASE_MODEL_PATH" "QWEN3_1_7B_BASE_MODEL_PATH"
run_base_eval "Qwen3-4B-Instruct-2507" "$QWEN3_4B_INSTRUCT_BASE_MODEL_PATH" "QWEN3_4B_INSTRUCT_BASE_MODEL_PATH"

log "All requested code base evaluations completed"
