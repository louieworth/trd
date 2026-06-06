# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import json
import os
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(config, data_source, response_lst, reward_data):
    reward_fn = get_custom_reward_fn(config)
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    aggregation = config.get("sample_aggregation", "pass_at_k")

    if aggregation == "mean":
        aggregated_score = float(np.mean(score_lst))
    elif aggregation == "pass_at_k":
        aggregated_score = float(any(score > 0 for score in score_lst))
    else:
        raise ValueError(f"Unknown sample aggregation mode: {aggregation}")

    return data_source, aggregated_score


def format_eval_results(metric_dict: dict[str, float], pass_k: int) -> dict[str, float]:
    formatted_results = {}
    for key, value in metric_dict.items():
        data_source = key.removeprefix("test_score/")
        formatted_results[f"{data_source}_pass{pass_k}_generation_pass_{pass_k}"] = float(value)
    return formatted_results


def save_eval_results(
    metric_dict: dict[str, float],
    output_json_path: str,
    model_name: str,
    pass_k: int,
    model_path: str | None = None,
):
    formatted_results = format_eval_results(metric_dict, pass_k=pass_k)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    all_results = {}
    if os.path.exists(output_json_path) and os.path.getsize(output_json_path) > 0:
        try:
            with open(output_json_path) as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print(f"[save_eval_results] warning: {output_json_path} corrupted; starting fresh")
            all_results = {}

    model_results = dict(all_results.get(model_name, {}))
    model_results["model_path"] = model_path
    model_results.update(formatted_results)
    all_results[model_name] = model_results

    tmp_path = output_json_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(all_results, f, indent=4)
    os.replace(tmp_path, output_json_path)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray_init_kwargs = OmegaConf.to_container(config.ray_kwargs.get("ray_init", {}), resolve=True)
        if ray_init_kwargs.get("address") or os.environ.get("RAY_ADDRESS"):
            ray_init_kwargs = dict(ray_init_kwargs)
            ray_init_kwargs.pop("num_cpus", None)
            ray_init_kwargs.pop("num_gpus", None)
        ray.init(**ray_init_kwargs)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    # Create remote tasks
    remote_tasks = [
        process_item.remote(config, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)

    output_json_path = config.get("output_json_path")
    model_name = config.get("model_name")
    if output_json_path and model_name:
        save_eval_results(
            metric_dict,
            output_json_path=output_json_path,
            model_name=model_name,
            pass_k=config.get("pass_k", 1),
            model_path=config.get("model_path"),
        )


if __name__ == "__main__":
    main()
