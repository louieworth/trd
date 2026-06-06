#!/usr/bin/env python3
"""Run the final evaluation for a single merged KL checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

from eval_status import missing_items  # noqa: E402
from run_config import build_env  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent


def run_with_log(args: list[str], log_file: Path, env: dict[str, str], cwd: str) -> None:
    print("Running evaluation command:")
    print(" ".join(args))
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    values = build_env()
    if values["RUN_EVAL_AFTER_TRAINING"] != "true":
        return

    eval_model_path = Path(values["EPOCH_MODEL_SAVE_DIR"]) / "hf_merged"
    if not (eval_model_path / "config.json").is_file():
        raise SystemExit(f"ERROR: eval requested but merged model is missing: {eval_model_path}")

    verl_root = Path(os.environ.get("VERL_ROOT", Path.cwd())).resolve()
    eval_output_dir = values["EVAL_OUTPUT_DIR"] or str(
        verl_root / "gen_results" / "eval" / values["TASK"] / values["RESULTS_MODEL_KEY"]
    )
    missing = missing_items(
        values["TASK"],
        values["RESULTS_FILE"],
        values["RESULTS_MODEL_KEY"],
        values["EVAL_DATASETS"],
        values["PASS_K"],
        eval_output_dir,
    )
    if not missing:
        print(f"Evaluation already complete: {values['RESULTS_MODEL_KEY']}")
        return

    log_dir = Path(values["EPOCH_OUTPUT_DIR"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.update({
        "NGPUS_PER_NODE": values["NGPUS_PER_NODE"],
        "NNODES": values["NNODES"],
        "GEN_TP": values["EVAL_GEN_TP"],
        "EVAL_DATASETS_DIR": values["EVAL_DATASETS_DIR"],
        "DATASETS": " ".join(missing),
        "PASS_K": values["PASS_K"],
        "EVAL_BASE_MODEL_NAME": values["RESULTS_BASE_MODEL_NAME"],
        "EVAL_MODEL_NAME": values["RESULTS_MODEL_KEY"],
        "EVAL_RESULTS_FILE": values["RESULTS_FILE"],
        "EVAL_OUTPUT_DIR": eval_output_dir,
    })
    benchmark = Path(values["EVAL_RECIPE_DIR"]) / values["EVAL_BENCHMARK_SCRIPT"]
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    run_with_log(["bash", str(benchmark), str(eval_model_path)], log_file, env, str(verl_root))


if __name__ == "__main__":
    main()
