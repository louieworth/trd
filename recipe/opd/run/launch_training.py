#!/usr/bin/env python3
"""Launch the single KL update and mirror logs to the epoch output directory."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

from run_config import build_env, write_training_config  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR.parent


def add_optional(args: list[str], flag: str, value: str) -> None:
    if value:
        args.extend([flag, value])


def build_torchrun_command(values: dict[str, str]) -> list[str]:
    args = [
        "torchrun",
        f"--nproc-per-node={values['NGPUS_PER_NODE']}",
        f"--nnodes={values['NNODES']}",
        f"--node-rank={values['NODE_RANK']}",
        f"--master-addr={values['MASTER_ADDR']}",
        f"--master-port={values['MASTER_PORT']}",
        str(RECIPE_DIR / "run_training.py"),
        "--nnodes",
        values["NNODES"],
        "--n_gpus_per_node",
        values["NGPUS_PER_NODE"],
        "--task",
        values["TASK"],
        "--distill_mode",
        values["DISTILL_MODE"],
        "--kl_type",
        values["KL_TYPE"],
        "--kl_method",
        values["KL_METHOD"],
        "--kl_token_clip",
        values["KL_TOKEN_CLIP"],
        "--beta",
        values["BETA"],
        "--temperature",
        values["TEMPERATURE"],
        "--student_model_path",
        values["MODEL_PATH"],
        "--base_model_name",
        values["MODEL_NAME"],
        "--use_lora",
        values["USE_LORA"],
        "--lora_rank",
        values["LORA_RANK"],
        "--lora_alpha",
        values["LORA_ALPHA"],
        "--learning_rate",
        values["LEARNING_RATE"],
        "--train_batch_size",
        values["TRAIN_BATCH_SIZE"],
        "--gradient_accumulation_steps",
        values["GRADIENT_ACCUMULATION_STEPS"],
        "--total_epochs",
        values["TRAIN_EPOCHS_PER_ROUND"],
        "--max_length",
        values["MAX_LENGTH"],
        "--warmup_steps_ratio",
        values["WARMUP_RATIO"],
        "--weight_decay",
        values["WEIGHT_DECAY"],
        "--min_lr_ratio",
        "0.1",
        "--data_path",
        values["TRAINING_DATA_PATH"],
        "--num_workers",
        values["NUM_WORKERS"],
        "--fsdp_strategy",
        values["FSDP_STRATEGY"],
        "--fsdp_size",
        values["FSDP_SIZE"],
        "--ulysses_sequence_parallel_size",
        values["SP_SIZE"],
        "--max_token_len_per_gpu",
        values["MAX_TOKEN_LEN_PER_GPU"],
        "--use_torch_compile",
        values["USE_TORCH_COMPILE"],
        "--param_offload",
        values["PARAM_OFFLOAD"],
        "--optimizer_offload",
        values["OPTIMIZER_OFFLOAD"],
        "--offload_policy",
        values["OFFLOAD_POLICY"],
        "--epoch_index",
        "1",
        "--output_dir",
        values["EPOCH_OUTPUT_DIR"],
        "--model_save_dir",
        values["EPOCH_MODEL_SAVE_DIR"],
        "--gen_results_dir",
        values["GEN_RESULTS_DIR"],
        "--wandb_project",
        values["WANDB_PROJECT"],
        "--wandb_run_name",
        values["WANDB_RUN_NAME"] or values["RESULTS_MODEL_KEY"],
        "--save_merged_model",
        values["SAVE_MERGED_MODEL"],
        "--save_steps",
        values["SAVE_STEPS"],
        "--max_ckpt_to_keep",
        values["KEEP_LAST_N_CHECKPOINTS"],
        "--eval_datasets",
        values["EVAL_DATASETS"],
        "--eval_datasets_dir",
        values["EVAL_DATASETS_DIR"],
        "--use_initial_response",
        values["USE_INITIAL_RESPONSE"],
        "--prompt_truncation",
        values["PROMPT_TRUNCATION"],
        "--top_k",
        values["TOP_K"],
    ]
    add_optional(args, "--teacher_model_path", values["TEACHER_MODEL_PATH"])
    add_optional(args, "--corrected_responses_path", values["CORRECTED_RESPONSES_PATH"])
    add_optional(args, "--max_samples", values["MAX_SAMPLES"])
    return args


def run_with_log(args: list[str], log_file: Path, env: dict[str, str]) -> None:
    print("Running training command:")
    print(" ".join(args))
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    values = build_env()
    output_dir = Path(values["EPOCH_OUTPUT_DIR"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(values["EPOCH_MODEL_SAVE_DIR"]).mkdir(parents=True, exist_ok=True)
    write_training_config(values)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["WANDB_MODE"] = values["WANDB_MODE"]
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    run_with_log(build_torchrun_command(values), log_file, env)


if __name__ == "__main__":
    main()
