#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_PATHS=("$@")
if [ ${#MODEL_PATHS[@]} -eq 0 ] && [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS=("$MODEL_PATH")
fi
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "ERROR: specify at least one model path" >&2
    exit 1
fi
for mp in "${MODEL_PATHS[@]}"; do
    [ -d "$mp" ] || { echo "ERROR: model not found: $mp" >&2; exit 1; }
done

DATASETS_TO_TEST=${DATASETS:-"humaneval_plus mbpp_plus livecodebench_v6"}
PASS_K=${PASS_K:-16}
TEMPERATURE=${CODE_EVAL_TEMPERATURE:-0.6}
TOP_P=${CODE_EVAL_TOP_P:-0.95}
MAX_PROMPT_TOKENS=${CODE_EVAL_MAX_PROMPT_TOKENS:-2048}
MAX_TOKENS=${CODE_EVAL_MAX_RESPONSE_TOKENS:-${CODE_EVAL_MAX_TOKENS:-16384}}
MAX_MODEL_LEN=${CODE_EVAL_MAX_MODEL_LEN:-$((MAX_PROMPT_TOKENS + MAX_TOKENS))}
CODE_EVAL_MAX_NUM_SEQS=${CODE_EVAL_MAX_NUM_SEQS:-128}
CODE_EVAL_LCB_ENFORCE_EAGER=${CODE_EVAL_LCB_ENFORCE_EAGER:-false}
GEN_TP=${GEN_TP:-${NGPUS_PER_NODE:-1}}
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-results}
GEN_OUTPUT_BASE_DIR=${GEN_OUTPUT_BASE_DIR:-gen_results/code_eval}
WRITE_RESULTS_CSV=${WRITE_RESULTS_CSV:-true}

require_module() {
    "$PYTHON_BIN" - "$1" <<'PYMOD'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PYMOD
}

update_result() {
    "$PYTHON_BIN" - "$@" <<'PYUPDATE'
import json
import os
import sys
results_file, model_name, dataset, avg, passed = sys.argv[1:]
avg = None if avg == '' else float(avg)
passed = None if passed == '' else float(passed)
data = {}
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
pass_k = os.environ.get("PASS_K", "4")
entry = data.setdefault(model_name, {})
if avg is not None:
    entry[f"{dataset}_avg{pass_k}"] = avg
if passed is not None:
    entry[f"{dataset}_pass{pass_k}"] = passed
tmp = results_file + '.tmp'
os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
with open(tmp, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
os.replace(tmp, results_file)
PYUPDATE
}

evalplus_dataset() {
    local model_path="$1" model_name="$2" results_file="$3" out_dir="$4" dataset_name="$5" evalplus_name="$6"
    require_module evalplus || { echo "ERROR: evalplus is not installed. Install: pip install --upgrade 'evalplus[vllm] @ git+https://github.com/evalplus/evalplus'" >&2; exit 1; }
    mkdir -p "$out_dir/evalplus"
    local root="$out_dir/evalplus"
    local metrics avg pass
    if metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_evalplus_metrics.py" --root "$root" --dataset "$evalplus_name" --pass_k "$PASS_K" 2>/dev/null); then
        echo "Existing EvalPlus $evalplus_name eval results found under $root; skipping generation/evaluation."
    else
        "$PYTHON_BIN" "$SCRIPT_DIR/run_evalplus_vllm.py" \
            --dataset "$evalplus_name" \
            --model "$model_path" \
            --root "$root" \
            --tp "$GEN_TP" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_tokens "$MAX_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$CODE_EVAL_MAX_NUM_SEQS" \
            --n_samples "$PASS_K" \
            --trust_remote_code \
            ${EVALPLUS_EXTRA_ARGS:-}
        metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_evalplus_metrics.py" --root "$root" --dataset "$evalplus_name" --pass_k "$PASS_K")
    fi
    avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
    pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
    update_result "$results_file" "$model_name" "$dataset_name" "$avg" "$pass"
}

lcb_dataset() {
    local model_path="$1" model_name="$2" results_file="$3" out_dir="$4"
    local verl_root="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
    local lcb_repo="${LCB_REPO:-$verl_root/recipe/code_evaluation/external/LiveCodeBench}"
    [ -d "$lcb_repo" ] || { echo "ERROR: LiveCodeBench repo not found: $lcb_repo" >&2; exit 1; }
    PYTHONPATH="$lcb_repo${PYTHONPATH:+:$PYTHONPATH}" require_module lcb_runner || { echo "ERROR: lcb_runner is not importable. Install: $PYTHON_BIN -m pip install -e $lcb_repo" >&2; exit 1; }
    local lcb_model_key="${LCB_MODEL_KEY:-Qwen/Qwen3-235B-A22B}"
    local enforce_eager_arg=""
    if [ "$CODE_EVAL_LCB_ENFORCE_EAGER" = "true" ]; then
        enforce_eager_arg="--enforce_eager"
    fi
    mkdir -p "$out_dir/livecodebench"
    touch "$out_dir/livecodebench/.run_start"
    (
        cd "$lcb_repo"
        PYTHONPATH="$lcb_repo${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m lcb_runner.runner.main \
            --model "$lcb_model_key" \
            --local_model_path "$model_path" \
            --trust_remote_code \
            --scenario codegeneration \
            --evaluate \
            --release_version release_v6 \
            --n "$PASS_K" \
            --temperature "$TEMPERATURE" \
            --top_p "$TOP_P" \
            --max_tokens "$MAX_TOKENS" \
            --max_model_len "$MAX_MODEL_LEN" \
            --max_num_seqs "$CODE_EVAL_MAX_NUM_SEQS" \
            --tensor_parallel_size "$GEN_TP" \
            $enforce_eager_arg \
            ${LCB_EXTRA_ARGS:-}
    )
    find "$lcb_repo/output" -type f \
        \( -name '*_eval.json' -o -name '*_eval_all.json' -o -name 'codegeneration_*.json' \) \
        -newer "$out_dir/livecodebench/.run_start" -exec cp --parents {} "$out_dir/livecodebench" \; 2>/dev/null || true
    local metrics avg pass
    metrics=$("$PYTHON_BIN" "$SCRIPT_DIR/extract_lcb_metrics.py" --root "$out_dir/livecodebench" --pass_k "$PASS_K")
    avg=$(echo "$metrics" | awk -F= '/^avg=/{print $2}')
    pass=$(echo "$metrics" | awk -F= '/^pass=/{print $2}')
    update_result "$results_file" "$model_name" "livecodebench_v6" "$avg" "$pass"
}

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    FULL_MODEL_DIR=$(dirname "$MODEL_PATH")
    FULL_MODEL_NAME=$(basename "$FULL_MODEL_DIR")
    BASE_MODEL_NAME="${EVAL_BASE_MODEL_NAME:-$(echo "$FULL_MODEL_NAME" | sed -E 's/_kl_.*$//')}"
    MODEL_NAME="${EVAL_MODEL_NAME:-$FULL_MODEL_NAME}"
    GEN_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$GEN_OUTPUT_BASE_DIR/$MODEL_NAME}"
    RESULTS_FILE="${EVAL_RESULTS_FILE:-$RESULTS_BASE_DIR/${BASE_MODEL_NAME}_code.json}"
    mkdir -p "$GEN_OUTPUT_DIR" "$(dirname "$RESULTS_FILE")"

    echo "################################################################################"
    echo "# Code Model Benchmark"
    echo "# Model:      $MODEL_PATH"
    echo "# Model Name: $MODEL_NAME"
    echo "# Datasets:   $DATASETS_TO_TEST"
    echo "# Results:    $RESULTS_FILE"
    echo "# Sampling:   pass_k=$PASS_K temperature=$TEMPERATURE top_p=$TOP_P max_prompt_tokens=$MAX_PROMPT_TOKENS max_response_tokens=$MAX_TOKENS max_model_len=$MAX_MODEL_LEN max_num_seqs=$CODE_EVAL_MAX_NUM_SEQS lcb_enforce_eager=$CODE_EVAL_LCB_ENFORCE_EAGER"
    echo "################################################################################"

    for ds in $DATASETS_TO_TEST; do
        case "$ds" in
            humaneval_plus|humaneval+) evalplus_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" humaneval_plus humaneval ;;
            mbpp_plus|mbpp+) evalplus_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" mbpp_plus mbpp ;;
            livecodebench_v6|lcb_v6|livecodebench) lcb_dataset "$MODEL_PATH" "$MODEL_NAME" "$RESULTS_FILE" "$GEN_OUTPUT_DIR" ;;
            *) echo "ERROR: unsupported code dataset: $ds" >&2; exit 1 ;;
        esac
    done

    if [ "$WRITE_RESULTS_CSV" = "true" ]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/results_json_to_csv.py" \
            --results_file "$RESULTS_FILE" \
            --output_file "${EVAL_RESULTS_CSV_FILE:-${RESULTS_FILE%.json}.csv}"
    fi
    echo "Results saved to: $RESULTS_FILE"
done
