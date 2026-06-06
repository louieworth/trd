#!/bin/bash
# OPD MATH Qwen3-1.7B - Forward KL on y_r.
# This wrapper owns task/model/length/variant defaults and calls run_kl_training.sh directly.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

export TASK="math"
export DISTILL_MODE="opd"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
export STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-Qwen/Qwen3-8B}"
export TEACHER_MODEL="${TEACHER_MODEL:-Qwen3-8B}"

export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:?Set TRAIN_DATA_PATH to your local math training dataset path or HF dataset name}"
export TRAIN_DATA_SOURCE="${TRAIN_DATA_SOURCE:-deepscaleR}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25,hmmt25,beyondaime,amobench}"
export EVAL_DATASETS_DIR="${EVAL_DATASETS_DIR:?Set EVAL_DATASETS_DIR to your prepared evaluation datasets directory}"
export EVAL_RECIPE_DIR="${EVAL_RECIPE_DIR:-$VERL_ROOT/recipe/math_evaluation}"
export EVAL_BENCHMARK_SCRIPT="${EVAL_BENCHMARK_SCRIPT:-benchmark_kl_model.sh}"

export BASE_PROMPT_LENGTH="${BASE_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export EXPERT_SOLUTION_PROMPT_LENGTH="${EXPERT_SOLUTION_PROMPT_LENGTH:-4096}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-18432}"
export Y_R_PROMPT_LENGTH="${Y_R_PROMPT_LENGTH:-18432}"
export MAX_LENGTH="${MAX_LENGTH:-34816}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-34816}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"

export KL_TYPE="forward"
export KL_METHOD="full_vocab"
export Y_MODE="y_r"
export TEACHER_TRAINING_PROMPT="${TEACHER_TRAINING_PROMPT:-refine}"
export KL_TOKEN_CLIP="0"
export TOP_K="0"

export TEMPERATURE="${TEMPERATURE:-1.0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export RUN_EVAL_AFTER_TRAINING="${RUN_EVAL_AFTER_TRAINING:-true}"


exec bash "$VERL_ROOT/recipe/opd/run/run_kl_training.sh" "$@"
