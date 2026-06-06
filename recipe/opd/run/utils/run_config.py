#!/usr/bin/env python3
"""Build and print the normalized environment for a single OPD/OPSD run."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None else value


def _require(name: str, message: str) -> None:
    if not _env(name):
        raise SystemExit(f"ERROR: {name} is required. {message}")


def _path_component(value: str) -> str:
    value = (value or "unknown").rstrip("/").split("/")[-1]
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value or "unknown"


def _int_env(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be an integer (got: {raw})") from exc


def _require_one_of(name: str, value: str, allowed: Iterable[str]) -> None:
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        joined = ", ".join(allowed_values)
        raise SystemExit(f"ERROR: {name} must be one of: {joined} (got: {value})")


def build_env() -> dict[str, str]:
    task = _env("TASK", "math")
    distill_mode = _env("DISTILL_MODE", "opsd")
    _require_one_of("TASK", task, ("math", "code"))
    _require_one_of("DISTILL_MODE", distill_mode, ("opsd", "opd"))

    y_mode = _env("Y_MODE", "y_o")
    if y_mode == "y_raw":
        print("WARNING: Y_MODE=y_raw is deprecated; use y_o.", file=sys.stderr)
        y_mode = "y_o"
    elif y_mode == "y_cor":
        print("WARNING: Y_MODE=y_cor is deprecated; use y_r.", file=sys.stderr)
        y_mode = "y_r"
    _require_one_of("Y_MODE", y_mode, ("y_o", "y_r"))

    model_path = _env("MODEL_PATH", "Qwen/Qwen3-8B")
    model_name = _path_component(_env("MODEL_NAME", model_path))
    teacher_model_path = _env("TEACHER_MODEL_PATH")
    if distill_mode == "opd":
        _require("TEACHER_MODEL_PATH", "OPD needs a teacher model different from the student.")
    teacher_model_name = _path_component(_env("TEACHER_MODEL") or teacher_model_path or model_path)

    teacher_training_prompt = _env("TEACHER_TRAINING_PROMPT")
    if teacher_training_prompt:
        _require_one_of("TEACHER_TRAINING_PROMPT", teacher_training_prompt, ("vanilla", "refine"))
        use_initial_response = "true" if teacher_training_prompt == "refine" else "false"
    elif y_mode == "y_r":
        teacher_training_prompt = "refine"
        use_initial_response = "true"
    else:
        teacher_training_prompt = "vanilla"
        use_initial_response = "false"
    if y_mode == "y_o" and use_initial_response == "true":
        raise SystemExit("ERROR: TEACHER_TRAINING_PROMPT=refine is incompatible with Y_MODE=y_o.")

    for name, message in {
        "TRAIN_DATA_PATH": "Set it to a local Hugging Face dataset directory or dataset name.",
        "TRAIN_DATA_SOURCE": "Set a short source tag used in generated parquet metadata.",
        "EVAL_DATASETS": "Set comma-separated evaluation dataset names.",
        "EVAL_DATASETS_DIR": "Set the local directory containing prepared eval datasets.",
        "EVAL_RECIPE_DIR": "Set to recipe/math_evaluation or recipe/code_evaluation.",
        "EVAL_BENCHMARK_SCRIPT": "Set the benchmark shell script inside EVAL_RECIPE_DIR.",
        "BASE_PROMPT_LENGTH": "Set by the concrete wrapper.",
        "MAX_RESPONSE_LENGTH": "Set by the concrete wrapper.",
        "EXPERT_SOLUTION_PROMPT_LENGTH": "Set by the concrete wrapper.",
        "MAX_PROMPT_LENGTH": "Set by the concrete wrapper.",
        "Y_R_PROMPT_LENGTH": "Set by the concrete wrapper.",
        "MAX_LENGTH": "Set by the concrete wrapper.",
    }.items():
        _require(name, message)

    verl_root = Path(_env("VERL_ROOT", Path.cwd().as_posix())).resolve()
    output_dir = Path(_env("OUTPUT_DIR", str(verl_root / "outputs")))
    model_save_dir = Path(_env("MODEL_SAVE_DIR", str(verl_root / "checkpoints")))
    gen_results_root = Path(_env("GEN_RESULTS_ROOT", str(verl_root / "gen_results")))

    base_prompt_length = _int_env("BASE_PROMPT_LENGTH", 2048)
    max_response_length = _int_env("MAX_RESPONSE_LENGTH", 16384)
    y_r_prompt_length = _int_env("Y_R_PROMPT_LENGTH", 18432)
    max_length = _int_env("MAX_LENGTH", 18432)
    max_token_len_per_gpu = _int_env("MAX_TOKEN_LEN_PER_GPU", 24576)
    max_token_len_per_gpu = max(max_token_len_per_gpu, max_length)
    rollout_buffer = _int_env("ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER", 0)
    rollout_max_num_batched_tokens = _int_env("ROLLOUT_MAX_NUM_BATCHED_TOKENS", 65536)

    y_o_rollout_max_model_len = base_prompt_length + max_response_length + rollout_buffer
    y_r_rollout_max_model_len = y_r_prompt_length + max_response_length + rollout_buffer
    y_o_batched_tokens = max(rollout_max_num_batched_tokens, y_o_rollout_max_model_len)
    y_r_batched_tokens = max(rollout_max_num_batched_tokens, y_r_rollout_max_model_len)

    task_upper = task.upper()
    distill_upper = distill_mode.upper()
    kl_type = _env("KL_TYPE", "reverse")
    kl_method = _env("KL_METHOD", "monte_carlo")
    kl_token_clip = _env("KL_TOKEN_CLIP", "0")
    beta = _env("BETA", "0")
    top_k = _int_env("TOP_K", 0)
    run_date = _env("RUN_DATE", datetime.now().strftime("%Y%m%d-%H%M%S"))
    clip_tag = f"clip{kl_token_clip.replace('.', '')}"
    beta_tag = f"_beta{beta.replace('.', '')}" if kl_type == "jsd" else ""
    topk_tag = f"_topk{top_k}" if top_k > 0 else ""
    run_descriptor = (
        f"{y_mode}_kl_{kl_type}_{kl_method}_{clip_tag}"
        f"{beta_tag}{topk_tag}_{teacher_training_prompt}_{run_date}"
    )
    if distill_mode == "opd":
        model_run_name = f"teacher{teacher_model_name}_{run_descriptor}"
    else:
        model_run_name = run_descriptor

    use_lora = _env("USE_LORA", "true")
    results_tuning_suffix = "_LORA" if use_lora.lower() in {"true", "1", "yes", "y"} else "_NO_LORA"
    results_model_key = f"{model_name}_{distill_upper}_{task_upper}_{model_run_name}{results_tuning_suffix}"
    results_base_model_name = f"{distill_upper}/{task}/{model_name}"
    results_file = _env("EVAL_RESULTS_FILE", str(verl_root / "results" / distill_upper / task / f"{model_name}_{task}.json"))

    output_base_dir = output_dir / distill_upper / task / model_name / model_run_name
    model_save_base_dir = model_save_dir / f"{distill_upper}_{task_upper}" / model_name / model_run_name
    gen_results_dir = gen_results_root / distill_upper / task / model_name / model_run_name
    epoch_output_dir = output_base_dir / "epoch1"
    epoch_model_save_dir = model_save_base_dir / "epoch1"
    task_file_prefix = task

    data_path = _env("DATA_PATH")
    if data_path:
        training_data_path = data_path
    elif y_mode == "y_r":
        training_data_path = str(gen_results_dir / f"{task_file_prefix}_y_r_{distill_mode}_{teacher_model_name}_responses.parquet")
    else:
        training_data_path = str(gen_results_dir / f"{task_file_prefix}_y_o_responses.parquet")

    return {
        "TASK": task,
        "DISTILL_MODE": distill_mode,
        "KL_TYPE": kl_type,
        "KL_METHOD": kl_method,
        "KL_TOKEN_CLIP": kl_token_clip,
        "TEMPERATURE": _env("TEMPERATURE", "1.0"),
        "BETA": beta,
        "TOP_K": str(top_k),
        "Y_MODE": y_mode,
        "MODEL_PATH": model_path,
        "MODEL_NAME": model_name,
        "TEACHER_MODEL_PATH": teacher_model_path,
        "TEACHER_MODEL_NAME": teacher_model_name,
        "TEACHER_TRAINING_PROMPT": teacher_training_prompt,
        "USE_INITIAL_RESPONSE": use_initial_response,
        "TRAIN_DATA_PATH": _env("TRAIN_DATA_PATH"),
        "TRAIN_DATA_SOURCE": _env("TRAIN_DATA_SOURCE"),
        "DATA_PATH": data_path,
        "CORRECTED_RESPONSES_PATH": _env("CORRECTED_RESPONSES_PATH"),
        "MAX_SAMPLES": _env("MAX_SAMPLES"),
        "PROMPT_TRUNCATION": _env("PROMPT_TRUNCATION", "true"),
        "USE_LORA": use_lora,
        "LORA_RANK": _env("LORA_RANK", "64"),
        "LORA_ALPHA": _env("LORA_ALPHA", "128"),
        "LEARNING_RATE": _env("LEARNING_RATE", "5e-6"),
        "TRAIN_BATCH_SIZE": _env("TRAIN_BATCH_SIZE", "1"),
        "GRADIENT_ACCUMULATION_STEPS": _env("GRADIENT_ACCUMULATION_STEPS", "16"),
        "TRAIN_EPOCHS_PER_ROUND": _env("TRAIN_EPOCHS_PER_ROUND", "1"),
        "WARMUP_RATIO": _env("WARMUP_RATIO", "0.1"),
        "WEIGHT_DECAY": _env("WEIGHT_DECAY", "0.005"),
        "NUM_WORKERS": _env("NUM_WORKERS", "4"),
        "NGPUS_PER_NODE": _env("NGPUS_PER_NODE", "8"),
        "NNODES": _env("NNODES", "1"),
        "NODE_RANK": _env("NODE_RANK", "0"),
        "MASTER_ADDR": _env("MASTER_ADDR", "localhost"),
        "MASTER_PORT": _env("MASTER_PORT", "29500"),
        "EVAL_GEN_TP": _env("EVAL_GEN_TP", _env("NGPUS_PER_NODE", "8")),
        "FSDP_STRATEGY": _env("FSDP_STRATEGY", "fsdp2"),
        "FSDP_SIZE": _env("FSDP_SIZE", "-1"),
        "SP_SIZE": _env("SP_SIZE", "1"),
        "MAX_TOKEN_LEN_PER_GPU": str(max_token_len_per_gpu),
        "USE_TORCH_COMPILE": _env("USE_TORCH_COMPILE", "true"),
        "PARAM_OFFLOAD": _env("PARAM_OFFLOAD", "false"),
        "OPTIMIZER_OFFLOAD": _env("OPTIMIZER_OFFLOAD", "false"),
        "OFFLOAD_POLICY": _env("OFFLOAD_POLICY", "false"),
        "MODEL_SAVE_DIR": str(model_save_dir),
        "OUTPUT_DIR": str(output_dir),
        "GEN_RESULTS_ROOT": str(gen_results_root),
        "WANDB_PROJECT": _env("WANDB_PROJECT", "opd-kl-training"),
        "WANDB_RUN_NAME": _env("WANDB_RUN_NAME"),
        "WANDB_MODE": _env("WANDB_MODE", "offline"),
        "SAVE_MERGED_MODEL": _env("SAVE_MERGED_MODEL", "true"),
        "SAVE_STEPS": _env("SAVE_STEPS", "100"),
        "KEEP_LAST_N_CHECKPOINTS": _env("KEEP_LAST_N_CHECKPOINTS", "1"),
        "RUN_EVAL_AFTER_TRAINING": _env("RUN_EVAL_AFTER_TRAINING", "true"),
        "PASS_K": _env("PASS_K", "16"),
        "EVAL_DATASETS": _env("EVAL_DATASETS"),
        "EVAL_DATASETS_DIR": _env("EVAL_DATASETS_DIR"),
        "EVAL_RECIPE_DIR": _env("EVAL_RECIPE_DIR"),
        "EVAL_BENCHMARK_SCRIPT": _env("EVAL_BENCHMARK_SCRIPT"),
        "EVAL_RESULTS_FILE": _env("EVAL_RESULTS_FILE"),
        "EVAL_OUTPUT_DIR": _env("EVAL_OUTPUT_DIR"),
        "ROLLOUT_CHAT_TEMPLATE_TOKEN_BUFFER": str(rollout_buffer),
        "Y_O_ROLLOUT_MAX_MODEL_LEN": str(y_o_rollout_max_model_len),
        "Y_R_ROLLOUT_MAX_MODEL_LEN": str(y_r_rollout_max_model_len),
        "ROLLOUT_MAX_NUM_SEQS": _env("ROLLOUT_MAX_NUM_SEQS", "64"),
        "ROLLOUT_GPU_MEMORY_UTILIZATION": _env("ROLLOUT_GPU_MEMORY_UTILIZATION", "0.85"),
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(rollout_max_num_batched_tokens),
        "Y_O_ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(y_o_batched_tokens),
        "Y_R_ROLLOUT_MAX_NUM_BATCHED_TOKENS": str(y_r_batched_tokens),
        "TASK_UPPER": task_upper,
        "DISTILL_UPPER": distill_upper,
        "DISTILL_TASK_FAMILY": f"{distill_upper}_{task_upper}",
        "RUN_DATE": run_date,
        "MODEL_RUN_NAME": model_run_name,
        "RESULTS_MODEL_KEY": results_model_key,
        "RESULTS_BASE_MODEL_NAME": results_base_model_name,
        "RESULTS_FILE": results_file,
        "OUTPUT_BASE_DIR": str(output_base_dir),
        "MODEL_SAVE_BASE_DIR": str(model_save_base_dir),
        "GEN_RESULTS_DIR": str(gen_results_dir),
        "EPOCH_OUTPUT_DIR": str(epoch_output_dir),
        "EPOCH_MODEL_SAVE_DIR": str(epoch_model_save_dir),
        "TASK_FILE_PREFIX": task_file_prefix,
        "TRAINING_DATA_PATH": training_data_path,
        "BASE_PROMPT_LENGTH": str(base_prompt_length),
        "MAX_RESPONSE_LENGTH": str(max_response_length),
        "EXPERT_SOLUTION_PROMPT_LENGTH": _env("EXPERT_SOLUTION_PROMPT_LENGTH"),
        "MAX_PROMPT_LENGTH": _env("MAX_PROMPT_LENGTH"),
        "Y_R_PROMPT_LENGTH": str(y_r_prompt_length),
        "MAX_LENGTH": str(max_length),
    }


def emit_shell_env(values: dict[str, str]) -> None:
    for key in sorted(values):
        print(f"export {key}={shlex.quote(values[key])}")


def print_config(values: dict[str, str]) -> None:
    print("==========================================")
    print("Single-Update OPD/OPSD KL Training")
    print("==========================================")
    print(f"Task:            {values['TASK']}")
    print(f"Distill mode:    {values['DISTILL_MODE']}")
    print(f"Student model:   {values['MODEL_PATH']}")
    print(f"Teacher model:   {values['TEACHER_MODEL_PATH'] or '<same as student>'}")
    print(
        "Variant:         "
        f"y={values['Y_MODE']} kl={values['KL_TYPE']}/{values['KL_METHOD']} "
        f"clip={values['KL_TOKEN_CLIP']} top_k={values['TOP_K']}"
    )
    print(f"Train data:      {values['TRAIN_DATA_PATH']}")
    print(f"Training parquet:{values['TRAINING_DATA_PATH']}")
    print(f"Output dir:      {values['EPOCH_OUTPUT_DIR']}")
    print(f"Model save dir:  {values['EPOCH_MODEL_SAVE_DIR']}")
    print(f"Gen results dir: {values['GEN_RESULTS_DIR']}")
    print(f"Eval:            {values['RUN_EVAL_AFTER_TRAINING']} ({values['EVAL_DATASETS']})")
    print("==========================================")


def write_training_config(values: dict[str, str]) -> None:
    output = Path(values["EPOCH_OUTPUT_DIR"]) / "training_config.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "TASK",
        "DISTILL_MODE",
        "MODEL_RUN_NAME",
        "RESULTS_MODEL_KEY",
        "MODEL_PATH",
        "TEACHER_MODEL_PATH",
        "TRAINING_DATA_PATH",
        "TRAIN_DATA_PATH",
        "GEN_RESULTS_DIR",
        "EPOCH_OUTPUT_DIR",
        "EPOCH_MODEL_SAVE_DIR",
        "KL_TYPE",
        "KL_METHOD",
        "KL_TOKEN_CLIP",
        "TOP_K",
        "TEMPERATURE",
        "BASE_PROMPT_LENGTH",
        "Y_R_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "MAX_LENGTH",
    ]
    names = {
        "MODEL_PATH": "student_model_path",
        "TEACHER_MODEL_PATH": "teacher_model_path",
        "EPOCH_OUTPUT_DIR": "output_dir",
        "EPOCH_MODEL_SAVE_DIR": "model_save_dir",
    }
    with output.open("w", encoding="utf-8") as f:
        for key in keys:
            yaml_key = names.get(key, key.lower())
            value = values[key] if key != "TEACHER_MODEL_PATH" else values[key] or values["MODEL_PATH"]
            f.write(f"{yaml_key}: {value}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["env", "print", "write-training-config"])
    args = parser.parse_args()
    values = build_env()

    if args.command == "env":
        emit_shell_env(values)
    elif args.command == "print":
        print_config(values)
    else:
        write_training_config(values)


if __name__ == "__main__":
    main()
