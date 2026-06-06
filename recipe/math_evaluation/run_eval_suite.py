#!/usr/bin/env python3

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, REPO_ROOT)

from recipe.opd._tokenizer_compat import apply_qwen2_tokenizer_vllm_compat

from recipe.math_evaluation.eval_utils import resolve_eval_dataset_paths, run_evaluation_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one model across multiple datasets with one server load")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="aime24,aime25,math500,hmmt25")
    parser.add_argument("--datasets_dir", type=str, default="./data/eval")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--pass_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=4)
    parser.add_argument("--gen_tp", type=int, default=1)
    return parser.parse_args()


def main():
    pythonpath_entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if SCRIPT_DIR not in pythonpath_entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([SCRIPT_DIR, *pythonpath_entries])

    apply_qwen2_tokenizer_vllm_compat()

    args = parse_args()
    dataset_names = [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
    dataset_paths = resolve_eval_dataset_paths(dataset_names, args.datasets_dir)

    results = run_evaluation_suite(
        args.model_path,
        dataset_paths,
        args.output_dir,
        pass_k=args.pass_k,
        temperature=args.temperature,
        top_p=args.top_p,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_model_parallel_size=args.gen_tp,
        output_json_path=args.results_file,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path or None,
    )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
