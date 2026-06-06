#!/usr/bin/env python3
#
# Prepare BeyondAIME, AMO-Bench, and GSM8K evaluation datasets in verl parquet format.

import argparse
import os
import re
import sys
from copy import deepcopy

import datasets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from verl.utils.reward_score.math_reward import remove_boxed

MATH_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def normalize_math_answer(answer_raw):
    if isinstance(answer_raw, str):
        answer = answer_raw.strip()
        if answer.startswith("\\boxed"):
            try:
                return remove_boxed(answer)
            except Exception:
                return answer
        return answer
    return str(answer_raw)


def extract_gsm8k_solution(answer_raw):
    match = re.search(r"####\s*([\-0-9\.,]+)", answer_raw)
    if match is None:
        raise ValueError(f"Failed to extract GSM8K solution from answer: {answer_raw}")
    return match.group(1).replace(",", "").replace("$", "")


def append_boxed_instruction(question_raw):
    return f"{question_raw} {MATH_BOXED_INSTRUCTION}"


def append_gsm8k_instruction(question_raw):
    return f"{question_raw} {GSM8K_INSTRUCTION}"


def build_amobench_try_list(question_id):
    if question_id == 5:
        return [
            "n=1", "n=2", "n=3", "n=4", "n=5", "n=6", "n=7", "n=8", "n=9", "n=10",
            "n=11", "n=12", "n=13", "n=14", "n=15", "n=16", "n=17", "n=18", "n=19", "n=20",
        ]
    if question_id == 37:
        return [
            "a=2,b=3,c=4", "a=3,b=4,c=5", "a=4,b=5,c=6", "a=5,b=6,c=7", "a=6,b=7,c=8",
            "a=7,b=8,c=9", "a=8,b=9,c=10", "a=9,b=10,c=11", "a=10,b=11,c=12",
            "a=11,b=12,c=13", "a=12,b=13,c=14", "a=13,b=14,c=15", "a=14,b=15,c=16",
            "a=15,b=16,c=17", "a=16,b=17,c=18", "a=17,b=18,c=19", "a=18,b=19,c=20",
        ]
    return []


def extract_amobench_parser_ground_truth(example):
    ground_truth = {
        "answer": example["answer"],
        "answer_type": example["answer_type"],
        "question_id": example["question_id"],
        "prompt": example["prompt"],
    }
    try_list = build_amobench_try_list(example["question_id"])
    if try_list:
        ground_truth["try_list"] = try_list
    return ground_truth


def is_amobench_parser_example(example):
    return example["answer_type"] != "description"


DATASET_CONFIGS = {
    "beyondaime": {
        "hf_name": "ByteDance-Seed/BeyondAIME",
        "split": "test",
        "prompt_key": "problem",
        "answer_key": "answer",
        "prompt_builder": append_boxed_instruction,
        "answer_extractor": normalize_math_answer,
        "data_source": "beyondaime",
    },
    "amobench": {
        "hf_name": "meituan-longcat/AMO-Bench",
        "split": "test",
        "prompt_key": "prompt",
        "answer_key": "answer",
        "prompt_builder": lambda prompt_raw: prompt_raw,
        "answer_extractor": extract_amobench_parser_ground_truth,
        "answer_extractor_uses_example": True,
        "filter_fn": is_amobench_parser_example,
        "data_source": "amobench",
    },
    "gsm8k": {
        "hf_name": "openai/gsm8k",
        "config_name": "main",
        "split": "test",
        "prompt_key": "question",
        "answer_key": "answer",
        "prompt_builder": append_gsm8k_instruction,
        "answer_extractor": extract_gsm8k_solution,
        "data_source": "openai/gsm8k",
    },
}


def resolve_dataset_id(local_dataset_root, hf_name):
    if local_dataset_root is None:
        return hf_name
    return os.path.join(local_dataset_root, hf_name)


def load_raw_dataset(config, local_dataset_root=None):
    dataset_id = resolve_dataset_id(local_dataset_root, config["hf_name"])
    config_name = config.get("config_name")
    if config_name:
        return datasets.load_dataset(dataset_id, config_name, split=config["split"])
    return datasets.load_dataset(dataset_id, split=config["split"])


def make_map_fn(config):
    prompt_key = config["prompt_key"]
    answer_key = config["answer_key"]
    prompt_builder = config["prompt_builder"]
    answer_extractor = config["answer_extractor"]
    answer_extractor_uses_example = config.get("answer_extractor_uses_example", False)
    data_source = config["data_source"]

    def process_fn(example, idx):
        extra_info = deepcopy(example)
        question_raw = extra_info[prompt_key]
        answer_raw = extra_info[answer_key]
        ground_truth = answer_extractor(extra_info) if answer_extractor_uses_example else answer_extractor(answer_raw)

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_builder(question_raw),
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                **extra_info,
                "index": idx,
                "question": question_raw,
                "answer": answer_raw,
            },
        }
        return data

    return process_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BeyondAIME, AMO-Bench, and GSM8K evaluation datasets in verl parquet format."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="beyondaime,amobench,gsm8k",
        help="Comma-separated dataset names. Supported: beyondaime, amobench, gsm8k",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local root directory containing raw HF dataset repos laid out by namespace/name.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="./data/eval",
        help="Base directory for the prepared parquet files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared parquet files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    selected_datasets = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]

    for dataset_name in selected_datasets:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        config = DATASET_CONFIGS[dataset_name]
        save_dir = os.path.join(os.path.expanduser(args.local_save_dir), dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{dataset_name}_test.parquet")

        if os.path.exists(output_path) and not args.overwrite:
            print(f"Dataset {dataset_name} already exists at {output_path}, skipping...")
            continue

        print(f"Loading {config['hf_name']} ({config['split']})...")
        dataset = load_raw_dataset(config, local_dataset_root=args.local_dataset_path)
        if "filter_fn" in config:
            dataset = dataset.filter(config["filter_fn"])
        dataset = dataset.map(make_map_fn(config), with_indices=True, remove_columns=dataset.column_names)
        dataset.to_parquet(output_path)
        print(f"Saved {dataset_name} to {output_path}")


if __name__ == "__main__":
    main()
