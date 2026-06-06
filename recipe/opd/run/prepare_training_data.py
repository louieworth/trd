#!/usr/bin/env python3
"""Prepare y_o/y_r parquet data for the single-update KL run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(UTILS_DIR))

from run_config import build_env  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
RECIPE_DIR = SCRIPT_DIR.parent
GENERATION_DIR = RECIPE_DIR / "generation"


def file_exists_and_nonempty(path: str) -> bool:
    target = Path(path)
    return target.is_file() and target.stat().st_size > 0


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print(" ".join(args))
    subprocess.run(args, check=True, env=env)


def filter_prompt_parquet(path: str, max_prompt_tokens: str, tokenizer_path: str, label: str) -> None:
    if not file_exists_and_nonempty(path):
        return
    if int(max_prompt_tokens) <= 0:
        return
    run_command([
        sys.executable,
        str(SCRIPT_DIR / "utils" / "prompt_filter.py"),
        path,
        max_prompt_tokens,
        tokenizer_path,
        label,
    ])


def generate_with_vllm(
    values: dict[str, str],
    model_path: str,
    prompts: str,
    output: str,
    prompt_length: str,
    max_model_len: str,
    max_batched_tokens: str,
) -> None:
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    run_command([
        sys.executable,
        "-m",
        "verl.trainer.main_generation_server",
        f"trainer.nnodes={values['NNODES']}",
        f"trainer.n_gpus_per_node={values['NGPUS_PER_NODE']}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.trust_remote_code=true",
        "actor_rollout_ref.rollout.temperature=0.6",
        "actor_rollout_ref.rollout.top_p=0.95",
        "actor_rollout_ref.rollout.top_k=20",
        f"actor_rollout_ref.rollout.prompt_length={prompt_length}",
        f"actor_rollout_ref.rollout.response_length={values['MAX_RESPONSE_LENGTH']}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={values['ROLLOUT_GPU_MEMORY_UTILIZATION']}",
        f"actor_rollout_ref.rollout.max_num_seqs={values['ROLLOUT_MAX_NUM_SEQS']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_batched_tokens}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.n=1",
        f"data.train_files=['{prompts}']",
        "data.prompt_key=prompt",
        f"+data.output_path={output}",
    ], env=env)


def prepare_y_o_prompts(values: dict[str, str], output_file: str) -> None:
    if not file_exists_and_nonempty(output_file):
        args = [
            sys.executable,
            str(GENERATION_DIR / "y_o_prepare.py"),
            "--input_path",
            values["TRAIN_DATA_PATH"],
            "--output_file",
            output_file,
            "--task",
            values["TASK"],
            "--data_source",
            values["TRAIN_DATA_SOURCE"],
        ]
        if values["MAX_SAMPLES"]:
            args.extend(["--max_samples", values["MAX_SAMPLES"]])
        run_command(args)
    filter_prompt_parquet(output_file, values["BASE_PROMPT_LENGTH"], values["MODEL_PATH"], f"y_o {values['TASK']} prompts")


def ensure_y_o_responses(values: dict[str, str]) -> str:
    prompts = str(Path(values["GEN_RESULTS_DIR"]) / f"{values['TASK_FILE_PREFIX']}_y_o_prompts.parquet")
    responses = str(Path(values["GEN_RESULTS_DIR"]) / f"{values['TASK_FILE_PREFIX']}_y_o_responses.parquet")
    if not file_exists_and_nonempty(responses):
        prepare_y_o_prompts(values, prompts)
        generate_with_vllm(
            values,
            values["MODEL_PATH"],
            prompts,
            responses,
            values["BASE_PROMPT_LENGTH"],
            values["Y_O_ROLLOUT_MAX_MODEL_LEN"],
            values["Y_O_ROLLOUT_MAX_NUM_BATCHED_TOKENS"],
        )
    filter_prompt_parquet(responses, values["BASE_PROMPT_LENGTH"], values["MODEL_PATH"], f"y_o {values['TASK']} responses")
    return responses


def prepare_y_r(values: dict[str, str]) -> None:
    y_o_output = ensure_y_o_responses(values)
    y_r_prompts = str(
        Path(values["GEN_RESULTS_DIR"])
        / f"{values['TASK_FILE_PREFIX']}_y_r_{values['DISTILL_MODE']}_{values['TEACHER_MODEL_NAME']}_prompts.parquet"
    )
    if not file_exists_and_nonempty(y_r_prompts):
        run_command([
            sys.executable,
            str(GENERATION_DIR / "y_r_prepare.py"),
            "--y_o_output",
            y_o_output,
            "--task",
            values["TASK"],
            "--distill_mode",
            values["DISTILL_MODE"],
            "--output_file",
            y_r_prompts,
        ])

    y_r_model = values["TEACHER_MODEL_PATH"] if values["DISTILL_MODE"] == "opd" else values["MODEL_PATH"]
    filter_prompt_parquet(y_r_prompts, values["Y_R_PROMPT_LENGTH"], y_r_model, f"y_r {values['TASK']} prompts")
    generate_with_vllm(
        values,
        y_r_model,
        y_r_prompts,
        values["TRAINING_DATA_PATH"],
        values["Y_R_PROMPT_LENGTH"],
        values["Y_R_ROLLOUT_MAX_MODEL_LEN"],
        values["Y_R_ROLLOUT_MAX_NUM_BATCHED_TOKENS"],
    )
    filter_prompt_parquet(values["TRAINING_DATA_PATH"], values["Y_R_PROMPT_LENGTH"], y_r_model, f"y_r {values['TASK']} responses")


def main() -> None:
    values = build_env()
    Path(values["GEN_RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)
    if values["DATA_PATH"]:
        if not file_exists_and_nonempty(values["DATA_PATH"]):
            raise SystemExit(f"ERROR: DATA_PATH does not exist or is empty: {values['DATA_PATH']}")
        print(f"Using provided training parquet: {values['DATA_PATH']}")
        return
    if file_exists_and_nonempty(values["TRAINING_DATA_PATH"]):
        print(f"Training parquet already exists: {values['TRAINING_DATA_PATH']}")
        return
    if values["Y_MODE"] == "y_o":
        ensure_y_o_responses(values)
    else:
        prepare_y_r(values)


if __name__ == "__main__":
    main()
