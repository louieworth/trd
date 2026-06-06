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
Generate responses given a dataset of prompts
"""

import os
import re

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
import pyarrow.lib
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from verl.trainer.generation_server_env import (
    build_generation_server_runtime_env,
    temporarily_clear_torch_launch_env,
)
from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


def read_parquet_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except pyarrow.lib.ArrowNotImplementedError:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = [
            batch.to_pandas()
            for batch in parquet_file.iter_batches(batch_size=1024)
        ]
        if not batches:
            return pd.DataFrame()
        return pd.concat(batches, axis=0, ignore_index=True)


async def start_server(config):
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, max_retries=5, retry_base_delay=2.0, **chat_complete_request):
    extra_headers = chat_complete_request.pop("extra_headers", {})
    for attempt in range(max_retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            try:
                async with session.post(
                    url=f"http://{server_address}/v1/chat/completions",
                    headers={"Authorization": "Bearer token-abc123", **extra_headers},
                    json=chat_complete_request,
                ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        match = re.search(
                            r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
                            error_msg,
                        )
                        current_max_tokens = chat_complete_request.get("max_tokens")
                        if resp.status == 400 and match and current_max_tokens is not None:
                            max_model_len = int(match.group(1))
                            prompt_tokens = int(match.group(2))
                            adjusted_max_tokens = max(1, max_model_len - prompt_tokens)
                            if adjusted_max_tokens < int(current_max_tokens):
                                chat_complete_request["max_tokens"] = adjusted_max_tokens
                                print(
                                    f"[Adjust max_tokens] Request to {server_address}: "
                                    f"{current_max_tokens} -> {adjusted_max_tokens} "
                                    f"for prompt_tokens={prompt_tokens}, max_model_len={max_model_len}"
                                )
                                continue
                        raise RuntimeError(f"Server returned {resp.status}: {error_msg}")
                    return ChatCompletion(**data)
            finally:
                await session.close()
        except Exception as e:
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                print(f"[Retry {attempt + 1}/{max_retries}] Request to {server_address} failed: {e}. Retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                raise


async def submit_indexed_request(request_index: int, server_address: str, **chat_complete_request):
    result = await submit_request(server_address, **chat_complete_request)
    return request_index, result


async def generate_per_replica(
    server_address,
    model_path: str,
    n_samples: int,
    sampling_params: dict,
    chat_lst: list,
    progress_bar: tqdm | None = None,
    max_concurrency: int | None = None,
):
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    if not chat_complete_request:
        return []

    if max_concurrency is None:
        max_concurrency = int(os.environ.get("EVAL_MAX_CONCURRENCY", os.environ.get("GENERATION_MAX_CONCURRENCY", "128")))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def submit_with_limit(request_index: int, req: dict):
        if semaphore is None:
            return await submit_indexed_request(request_index, server_address, **req)
        async with semaphore:
            return await submit_indexed_request(request_index, server_address, **req)

    tasks = [
        asyncio.create_task(submit_with_limit(request_index, req))
        for request_index, req in enumerate(chat_complete_request)
    ]
    results = [None] * len(tasks)

    try:
        for task in asyncio.as_completed(tasks):
            request_index, result = await task
            results[request_index] = result
            if progress_bar is not None:
                progress_bar.update(1)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    total_requests = len(chat_numpy) * n_samples
    with tqdm(total=total_requests, desc="Generating responses", dynamic_ncols=True) as progress_bar:
        results = await asyncio.gather(
            *[
                generate_per_replica(
                    server_addresses[i],
                    model_path,
                    n_samples,
                    sampling_params,
                    chat_sub_array[i],
                    progress_bar=progress_bar,
                )
                for i in range(num_replicas)
            ]
        )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    with temporarily_clear_torch_launch_env():
        ray_address = os.environ.get("RAY_ADDRESS") or None
        ray.init(address=ray_address, runtime_env=build_generation_server_runtime_env())

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        n_samples = config.actor_rollout_ref.rollout.n

        if config.actor_rollout_ref.rollout.temperature == 0.0:
            assert n_samples == 1, "When temperature=0, n_samples must be 1."
        assert n_samples >= 1, "n_samples should always >= 1"

        sampling_params = {
            "temperature": config.actor_rollout_ref.rollout.temperature,
            "top_p": config.actor_rollout_ref.rollout.top_p,
            # "top_k": config.actor_rollout_ref.rollout.top_k,
            "max_tokens": config.actor_rollout_ref.rollout.response_length,
        }

        from omegaconf import ListConfig

        train_files = config.data.train_files
        if not isinstance(train_files, list | ListConfig):
            train_files = [train_files]

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

        datasets = []
        for train_file in train_files:
            dataset = read_parquet_compat(train_file)
            datasets.append(dataset)

        # concat dataset
        dataset = pd.concat(datasets, axis=0, ignore_index=True)
        chat_lst = dataset[config.data.prompt_key].tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]
        chat_numpy = np.array(chat_lst)

        # start native server
        server_handles, server_addresses = asyncio.run(start_server(config))

        # run generate
        gen_results = asyncio.run(
            generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, chat_numpy)
        )

        # reshape results into a numpy array
        import itertools

        results = list(itertools.chain.from_iterable(gen_results))

        # extract content from results
        results = np.array([result.choices[0].message.content for result in results])
        results = np.reshape(results, (-1, n_samples))

        assert results.shape == (len(chat_lst), n_samples)

    results = results.tolist()

    # add to the data frame
    dataset["responses"] = results

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
