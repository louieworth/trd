#!/bin/bash
# OPSD CODE Qwen3-4B-Instruct-2507 - Forward KL on y_o with per-token clip.
# This wrapper owns task/model/length/variant defaults and calls run_kl_training.sh directly.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

export TASK="code"
export DISTILL_MODE="opsd"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Instruct-2507}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-4B-Instruct-2507}"
export TEACHER_MODEL_PATH=""
unset TEACHER_MODEL

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to your local TACO training dataset path or HF dataset name}"
export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-BAAI/TACO}"
export EVAL_DATASETS="${EVAL_DATASETS:-humaneval_plus,mbpp_plus,livecodebench_v6}"
export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:?Set EVAL_DATASETS_DIR to your prepared evaluation datasets directory}"
export EVAL_RECIPE_DIR="${EVAL_RECIPE_DIR:-$VERL_ROOT/recipe/code_evaluation}"
export EVAL_BENCHMARK_SCRIPT="${EVAL_BENCHMARK_SCRIPT:-benchmark_code_model.sh}"

export BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export Y_R_PROMPT_LENGTH="${Y_R_PROMPT_LENGTH:-22528}"
export MAX_LENGTH="${MAX_LENGTH:-22528}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-24576}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"

export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="y_o"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-vanilla}"
export KL_TOKEN_CLIP="${KL_TOKEN_CLIP:-0.1}"
export TOP_K="0"

export TEMPERATURE="${TEMPERATURE:-1.0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"


exec bash "$VERL_ROOT/recipe/opd/run/run_kl_training.sh" "$@"
