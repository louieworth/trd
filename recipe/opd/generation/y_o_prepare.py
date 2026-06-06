#!/usr/bin/env python3
#
# y_o Data Preparation: Prepare data for initial response generation
# Input: DeepScaleR dataset with problem, answer, solution
# Output: Parquet file with prompts, expert_cot in extra_info

import os
import argparse
import datasets
from verl.utils.reward_score.math_reward import remove_boxed

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
code_instruction_following = (
    "You will be given a programming problem. Write a correct Python program that solves it. "
    "Return only the code inside a single ```python code block."
)


def _first_solution(solutions_raw):
    if not solutions_raw:
        return ""
    if isinstance(solutions_raw, list):
        return solutions_raw[0] if solutions_raw else ""
    try:
        import json
        parsed = json.loads(solutions_raw)
        if isinstance(parsed, list):
            return parsed[0] if parsed else ""
    except Exception:
        pass
    return str(solutions_raw)


def _build_taco_prompt(example):
    question = example.get("question", "")
    starter_code = example.get("starter_code", "") or ""
    prompt = question.strip()
    if starter_code.strip():
        prompt += "\n\nStarter code:\n```python\n" + starter_code.strip() + "\n```"
    return prompt + "\n\n" + code_instruction_following


def make_map_fn_y_o_code(data_source="BAAI/TACO", index_offset=0):
    """Prepare TACO examples for y_o code generation."""
    def process_fn(example, idx):
        question_raw = example.get("question", "")
        prompt_content = _build_taco_prompt(example)
        expert_solution = _first_solution(example.get("solutions", ""))
        extra_info = {
            "problem": question_raw,
            "expert_cot": expert_solution,
            "starter_code": example.get("starter_code", ""),
            "difficulty": example.get("difficulty", ""),
            "source": example.get("source", ""),
            "url": example.get("url", ""),
            "index": index_offset + idx,
            "task": "code",
        }
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": "code",
            "extra_info": extra_info,
        }
    return process_fn


def make_map_fn_y_o(question_key="problem", data_source="deepscaleR", index_offset=0):
    """
    Prepare data for y_o generation.
    Preserves expert solution for y_r correction.
    """
    def process_fn(example, idx):
        # Extract problem and answer
        question_raw = example[question_key]
        answer_raw = example['answer']

        # Store expert CoT for y_r
        extra_info = {}
        extra_info['expert_cot'] = example.get('solution', '')
        extra_info[question_key] = question_raw
        extra_info['index'] = index_offset + idx

        # Store original answer in extra_info
        extra_info['answer'] = answer_raw

        # Construct prompt with instruction
        question = question_raw + " " + instruction_following

        # Extract ground truth answer
        try:
            solution = remove_boxed(answer_raw)
        except Exception:
            solution = answer_raw

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": extra_info,
        }
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="./data/train/DeepScaleR-Preview-Dataset-Update",
        help="Path to DeepScaleR dataset directory",
    )
    parser.add_argument(
        "--output_file",
        default="gen_results/y_o/deepscaleR_y_o.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--data_source",
        default="deepscaleR",
        help="Data source name",
    )
    parser.add_argument(
        "--question_key",
        default="problem",
        help="Key name for question in dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start offset into the prepared dataset before prompt mapping.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process from start_index. Omit for all remaining rows.",
    )
    parser.add_argument(
        "--task",
        choices=["math", "code"],
        default="math",
        help="Task-specific prompt/schema adapter.",
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_path}")
    # Try load_from_disk first (for datasets saved with save_to_disk)
    # Fallback to load_dataset for HuggingFace hub datasets
    try:
        ds_loaded = datasets.load_from_disk(args.input_path)
        # If it's a DatasetDict, get the 'train' split
        if isinstance(ds_loaded, datasets.DatasetDict):
            ds_raw = ds_loaded['train']
        else:
            ds_raw = ds_loaded
    except (ValueError, FileNotFoundError):
        # Not a disk-saved dataset, try loading from HuggingFace hub
        ds_raw = datasets.load_dataset(args.input_path, split="train")

    # Limit samples for testing if max_samples is specified
    if args.max_samples is not None:
        print(f"Limiting to first {args.max_samples} samples for testing...")
        ds_raw = ds_raw.select(range(min(args.max_samples, len(ds_raw))))

    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0 when provided")

    if args.start_index or args.num_samples is not None:
        start = min(args.start_index, len(ds_raw))
        stop = len(ds_raw) if args.num_samples is None else min(start + args.num_samples, len(ds_raw))
        print(f"Selecting sample range [{start}, {stop}) from {len(ds_raw)} prepared examples...")
        ds_raw = ds_raw.select(range(start, stop))

    print(f"Processing {len(ds_raw)} examples...")
    ds_processed = ds_raw.map(
        (make_map_fn_y_o_code(
            data_source=args.data_source,
            index_offset=args.start_index,
        ) if args.task == "code" else make_map_fn_y_o(
            question_key=args.question_key,
            data_source=args.data_source,
            index_offset=args.start_index,
        )),
        with_indices=True,
        remove_columns=[col for col in ds_raw.column_names if col not in ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']],
    )

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to: {args.output_file}")
    ds_processed.to_parquet(args.output_file)
    print("y_o data preparation completed!")
