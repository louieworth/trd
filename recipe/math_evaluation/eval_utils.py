#!/usr/bin/env python3
#
# Evaluation utilities for KL Training

import asyncio
import itertools
import json
import logging
import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

from verl.trainer.generation_server_env import (
    build_generation_server_runtime_env,
    temporarily_clear_torch_launch_env,
)
from verl.trainer.main_generation_server import generate, start_server

logger = logging.getLogger(__name__)

DATASET_ALIASES = {
    "bytedance-seed/beyondaime": "beyondaime",
    "beyondaime": "beyondaime",
    "meituan-longcat/amo-bench": "amobench",
    "amo-bench": "amobench",
    "amobench": "amobench",
    "openai/gsm8k": "gsm8k",
    "gsm8k": "gsm8k",
}

DATASET_RESULT_NAMES = {
    "gsm8k": "openai/gsm8k",
}

EVAL_DATASET_ROOTS = {
    "aime24": "aime24/aime24_test.parquet",
    "aime25": "aime25/aime25_test.parquet",
    "math500": "math500/math500_test.parquet",
    "hmmt23": "hmmt23/hmmt23_test.parquet",
    "hmmt24": "hmmt24/hmmt24_test.parquet",
    "hmmt25": "hmmt25/hmmt25_test.parquet",
    "amc23": "amc23/amc23_test.parquet",
    "beyondaime": "beyondaime/beyondaime_test.parquet",
    "amobench": "amobench/amobench_test.parquet",
    "gsm8k": "gsm8k/gsm8k_test.parquet",
}


def normalize_eval_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    return DATASET_ALIASES.get(normalized, normalized)


def get_result_data_source_name(dataset_name: str) -> str:
    normalized_name = normalize_eval_dataset_name(dataset_name)
    return DATASET_RESULT_NAMES.get(normalized_name, normalized_name)


def build_result_key(dataset_name: str, pass_k: int) -> str:
    result_name = get_result_data_source_name(dataset_name)
    return f"{result_name}_pass{pass_k}_generation_pass_{pass_k}"


def build_legacy_result_key(dataset_name: str, pass_k: int) -> str | None:
    if pass_k != 1:
        return None
    result_name = get_result_data_source_name(dataset_name)
    return f"{result_name}_generation_pass_1"


def find_existing_result_value(model_results: dict, dataset_name: str, pass_k: int):
    key_candidates = [build_result_key(dataset_name, pass_k)]
    legacy_key = build_legacy_result_key(dataset_name, pass_k)
    if legacy_key is not None:
        key_candidates.append(legacy_key)

    for key in key_candidates:
        if key in model_results:
            return model_results[key]
    return None


def resolve_eval_dataset_paths(dataset_names: list[str], datasets_dir: str) -> dict[str, str]:
    resolved_paths: dict[str, str] = {}
    for dataset_name in dataset_names:
        normalized_name = normalize_eval_dataset_name(dataset_name)
        if normalized_name in EVAL_DATASET_ROOTS:
            resolved_paths[normalized_name] = os.path.join(datasets_dir, EVAL_DATASET_ROOTS[normalized_name])
    return resolved_paths


def _build_generation_config(
    model_path: str,
    tokenizer_path: Optional[str],
    *,
    temperature: float,
    top_p: float,
    pass_k: int,
    prompt_length: int,
    response_length: int,
    max_model_len: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
):
    model_config = {
        "path": model_path,
        "trust_remote_code": True,
    }
    if tokenizer_path:
        model_config["tokenizer_path"] = tokenizer_path

    return OmegaConf.create(
        {
            "trainer": {
                "nnodes": nnodes,
                "n_gpus_per_node": n_gpus_per_node,
            },
            "actor_rollout_ref": {
                "model": model_config,
                "rollout": {
                    "_target_": "verl.workers.config.RolloutConfig",
                    "name": "vllm",
                    "mode": "async",
                    "nnodes": nnodes,
                    "n_gpus_per_node": n_gpus_per_node,
                    "temperature": temperature,
                    "top_k": -1,
                    "top_p": top_p,
                    "do_sample": True,
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "max_model_len": max_model_len,
                    "tensor_model_parallel_size": tensor_model_parallel_size,
                    "pipeline_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "gpu_memory_utilization": 0.95,
                    "n": pass_k,
                    "dtype": "bfloat16",
                    "enforce_eager": False,
                },
            },
        }
    )


def _generate_responses_with_server(
    server_addresses: list[str],
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    prompt_key: str,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    dataset = pd.read_parquet(dataset_path)
    chat_lst = dataset[prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst, dtype=object)

    gen_results = asyncio.run(
        generate(
            server_addresses,
            model_path,
            pass_k,
            {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            },
            chat_numpy,
        )
    )

    results = list(itertools.chain.from_iterable(gen_results))
    responses = np.array([result.choices[0].message.content for result in results], dtype=object)
    responses = np.reshape(responses, (-1, pass_k)).tolist()

    dataset["responses"] = responses
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_parquet(output_path)


def generate_responses_with_server(
    server_addresses: list[str],
    model_path: str,
    dataset_path: str,
    output_path: str,
    *,
    prompt_key: str,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    _generate_responses_with_server(
        server_addresses,
        model_path,
        dataset_path,
        output_path,
        prompt_key=prompt_key,
        pass_k=pass_k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def _evaluate_generated_output(
    dataset_name: str,
    gen_output: str,
    *,
    prompt_key: str,
    pass_k: int,
    output_json_path: Optional[str],
    model_name: Optional[str],
    model_path: Optional[str],
) -> float:
    eval_cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_eval",
        f"data.path={gen_output}",
        f"data.prompt_key={prompt_key}",
        "custom_reward_function.path=recipe/math_evaluation/compute_score.py",
        "custom_reward_function.name=compute_score_data_source",
        "sample_aggregation=pass_at_k",
    ]

    if output_json_path and model_name:
        eval_cmd.extend(
            [
                f"+output_json_path={output_json_path}",
                f"+model_name={model_name}",
                f"+pass_k={pass_k}",
                f"+model_path={model_path}",
            ]
        )

    subprocess.run(eval_cmd, check=True, capture_output=False)

    if output_json_path and model_name and os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            results = json.load(f)
        if model_name in results:
            value = find_existing_result_value(results[model_name], dataset_name, pass_k)
            if value is not None:
                return value

    return 0.0


def evaluate_generated_output(
    dataset_name: str,
    gen_output: str,
    *,
    prompt_key: str,
    pass_k: int,
    output_json_path: Optional[str],
    model_name: Optional[str],
    model_path: Optional[str],
) -> float:
    return _evaluate_generated_output(
        dataset_name,
        gen_output,
        prompt_key=prompt_key,
        pass_k=pass_k,
        output_json_path=output_json_path,
        model_name=model_name,
        model_path=model_path,
    )


def slice_responses_in_generation_file(source_path: str, target_path: str, pass_k: int):
    dataset = pd.read_parquet(source_path)
    dataset["responses"] = dataset["responses"].apply(lambda responses: list(responses[:pass_k]))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    dataset.to_parquet(target_path)


def launch_generation_server(
    model_path: str,
    tokenizer_path: Optional[str],
    *,
    temperature: float,
    top_p: float,
    pass_k: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_model_parallel_size: int,
):
    prompt_length = int(os.environ.get("EVAL_PROMPT_LENGTH", "4096"))
    response_length = int(os.environ.get("EVAL_RESPONSE_LENGTH", "38912"))
    max_model_len = int(os.environ.get("EVAL_MAX_MODEL_LEN", str(prompt_length + response_length)))

    config = _build_generation_config(
        model_path,
        tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        pass_k=pass_k,
        prompt_length=prompt_length,
        response_length=response_length,
        max_model_len=max_model_len,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )

    if ray.is_initialized():
        ray.shutdown()

    with temporarily_clear_torch_launch_env():
        ray.init(runtime_env=build_generation_server_runtime_env())
        server_handles, server_addresses = asyncio.run(start_server(config))
        return server_handles, server_addresses, response_length


def shutdown_generation_server(server_handles: list):
    for server_handle in server_handles:
        try:
            ray.kill(server_handle, no_restart=True)
        except Exception:
            pass
    if ray.is_initialized():
        ray.shutdown()


def run_evaluation_suite(
    model_path: str,
    dataset_paths: dict[str, str],
    output_dir: str,
    *,
    pass_k: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    nnodes: int = 1,
    n_gpus_per_node: int = 4,
    tensor_model_parallel_size: int = 1,
    output_json_path: Optional[str] = None,
    model_name: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    prompt_key: str = "prompt",
) -> dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)

    server_handles = []
    eval_results: dict[str, float] = {}

    try:
        server_handles, server_addresses, response_length = launch_generation_server(
            model_path,
            tokenizer_path,
            temperature=temperature,
            top_p=top_p,
            pass_k=pass_k,
            nnodes=nnodes,
            n_gpus_per_node=n_gpus_per_node,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )

        for dataset_name, dataset_path in dataset_paths.items():
            if not dataset_path or not os.path.exists(dataset_path):
                logger.warning("Skipping evaluation dataset %s because %s does not exist", dataset_name, dataset_path)
                continue

            gen_output = os.path.join(output_dir, f"{dataset_name}_pass{pass_k}_generation.parquet")

            if os.path.exists(gen_output) and os.path.getsize(gen_output) > 0:
                logger.info("  Reusing existing generated responses for %s: %s", dataset_name, gen_output)
            else:
                logger.info("  Generating responses for %s with pass@%s...", dataset_name, pass_k)
                generate_responses_with_server(
                    server_addresses,
                    model_path,
                    dataset_path,
                    gen_output,
                    prompt_key=prompt_key,
                    pass_k=pass_k,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=response_length,
                )

            logger.info("  Computing scores for %s...", dataset_name)
            accuracy = evaluate_generated_output(
                dataset_name,
                gen_output,
                prompt_key=prompt_key,
                pass_k=pass_k,
                output_json_path=output_json_path,
                model_name=model_name,
                model_path=model_path,
            )
            eval_results[dataset_name] = accuracy

        return eval_results
    finally:
        shutdown_generation_server(server_handles)


def run_evaluation_on_dataset(
    model_path: str,
    dataset_name: str,
    dataset_path: str,
    output_dir: str,
    pass_k: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    ngpus: int = 4,
    output_json_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> float:
    try:
        results = run_evaluation_suite(
            model_path,
            {dataset_name: dataset_path},
            output_dir,
            pass_k=pass_k,
            temperature=temperature,
            top_p=top_p,
            n_gpus_per_node=ngpus,
            output_json_path=output_json_path,
            model_name=model_name,
        )
        return results.get(dataset_name, 0.0)
    except subprocess.CalledProcessError as e:
        logger.error("  Evaluation failed for %s: %s", dataset_name, e)
        return 0.0


def parse_accuracy_from_output(output: str) -> float:
    import re

    match = re.search(r"accuracy[:\\s]+([0-9.]+)", output, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0


def save_eval_results(
    results: Dict[str, float],
    output_path: str,
    model_name: str,
    config: Dict = None,
    model_path: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    formatted_results = {}
    for dataset, accuracy in results.items():
        key = f"{dataset}_pass1_generation_pass_1"
        formatted_results[key] = accuracy

    existing_results = dict(all_results.get(model_name, {}))
    existing_results["model_path"] = model_path
    existing_results.update(formatted_results)
    all_results[model_name] = existing_results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info("Evaluation results saved to: %s", output_path)

    if config:
        detailed_path = output_path.replace(".json", "_detailed.json")
        detailed_output = {
            "model_name": model_name,
            "model_path": model_path,
            "results": formatted_results,
            "config": config,
        }
        with open(detailed_path, "w") as f:
            json.dump(detailed_output, f, indent=4)
        logger.info("Detailed results saved to: %s", detailed_path)
