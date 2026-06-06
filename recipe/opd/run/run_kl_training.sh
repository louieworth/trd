#!/bin/bash
# Single-update OPD/OPSD KL training entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(dirname "$SCRIPT_DIR")"
VERL_ROOT="$(dirname "$(dirname "$RECIPE_DIR")")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$VERL_ROOT"
export VERL_ROOT
export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
export OTEL_METRICS_EXPORTER="${OTEL_METRICS_EXPORTER:-none}"
export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-none}"
export OTEL_LOGS_EXPORTER="${OTEL_LOGS_EXPORTER:-none}"

# 1. Configuration
CONFIG_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/opd_run_config.XXXXXX")"
"$PYTHON_BIN" "$SCRIPT_DIR/utils/run_config.py" env > "$CONFIG_ENV_FILE"
source "$CONFIG_ENV_FILE"
rm -f "$CONFIG_ENV_FILE"

mkdir -p "$EPOCH_OUTPUT_DIR/logs" "$EPOCH_MODEL_SAVE_DIR" "$GEN_RESULTS_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/utils/run_config.py" print

# 2. Rollout
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_training_data.py"

# 3. Update
"$PYTHON_BIN" "$SCRIPT_DIR/launch_training.py"

# 4. Evaluation
"$PYTHON_BIN" "$SCRIPT_DIR/run_evaluation.py"

echo "=========================================="
echo "Done"
echo "Run name:       $MODEL_RUN_NAME"
echo "Model:          $EPOCH_MODEL_SAVE_DIR/hf_merged"
echo "Gen results:    $GEN_RESULTS_DIR"
echo "Training logs:  $EPOCH_OUTPUT_DIR"
echo "Results file:   $RESULTS_FILE"
echo "=========================================="
