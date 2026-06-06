#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recipe.math_evaluation.eval_utils import (
    build_result_key,
    evaluate_generated_output,
    find_existing_result_value,
    generate_responses_with_server,
    launch_generation_server,
    normalize_eval_dataset_name,
    resolve_eval_dataset_paths,
    shutdown_generation_server,
    slice_responses_in_generation_file,
)
from recipe.math_evaluation.results_json_to_csv import default_base_results_file, default_csv_path, write_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Batch benchmark KL models from a JSON config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the benchmark JSON config.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned work without launching evaluation.")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_model_path(model_path: str) -> str:
    resolved_path = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isdir(os.path.join(resolved_path, "hf_merged")):
        resolved_path = os.path.join(resolved_path, "hf_merged")
    if not os.path.isdir(resolved_path):
        raise FileNotFoundError(f"Model path not found: {resolved_path}")
    return resolved_path


def extract_model_identity(model_path: str) -> tuple[str, str, str]:
    full_model_dir = os.path.dirname(model_path)
    full_model_name = os.path.basename(full_model_dir)

    epoch_suffix = ""
    if re.fullmatch(r"epoch\d+", full_model_name):
        epoch_suffix = f"_{full_model_name}"
        full_model_dir = os.path.dirname(full_model_dir)
        full_model_name = os.path.basename(full_model_dir)

    base_model_name = re.sub(r"_kl_.*$", "", full_model_name)
    model_name = f"{full_model_name}{epoch_suffix}"
    return full_model_name, base_model_name, model_name


def load_existing_results(results_file: str) -> dict:
    if not os.path.exists(results_file):
        return {}
    return load_json(results_file)


def find_reusable_generation_file(output_dir: str, dataset_name: str, min_pass_k: int) -> tuple[str | None, int | None]:
    if not os.path.isdir(output_dir):
        return None, None

    pattern = re.compile(rf"^{re.escape(dataset_name)}_pass(\d+)_generation\.parquet$")
    reusable_candidates = []
    for file_name in os.listdir(output_dir):
        match = pattern.fullmatch(file_name)
        if match:
            pass_k = int(match.group(1))
            if pass_k >= min_pass_k:
                reusable_candidates.append((pass_k, os.path.join(output_dir, file_name)))

    if not reusable_candidates:
        return None, None

    reusable_candidates.sort()
    selected_pass_k, selected_path = reusable_candidates[0]
    return selected_path, selected_pass_k


def print_model_summary(results_file: str, model_name: str, dataset_names: list[str], pass_k_values: list[int]):
    results = load_existing_results(results_file)
    model_results = results.get(model_name, {})
    print(f"\nResults for {model_name}:")
    for dataset_name in dataset_names:
        normalized_name = normalize_eval_dataset_name(dataset_name)
        for pass_k in pass_k_values:
            value = find_existing_result_value(model_results, normalized_name, pass_k)
            if value is not None:
                print(f"  {normalized_name} pass@{pass_k}: {value:.2%}")
            else:
                print(f"  {normalized_name} pass@{pass_k}: (missing)")


def truthy_config(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def write_comparison_csv(results_file: str, model_config: dict, config: dict, base_model_name: str):
    if not truthy_config(model_config.get("write_csv", config.get("write_csv", True))):
        return

    csv_file = os.path.abspath(
        os.path.expanduser(model_config.get("csv_file", config.get("csv_file", default_csv_path(results_file))))
    )
    base_results_file = os.path.abspath(
        os.path.expanduser(
            model_config.get(
                "base_results_file",
                config.get("base_results_file", default_base_results_file(results_file, base_model_name)),
            )
        )
    )
    output_file = write_results_csv(
        results_file,
        csv_file,
        base_results_file=base_results_file,
        base_model_name=base_model_name,
    )
    print(f"CSV comparison: {output_file}")


def evaluate_from_generation_file(
    dataset_name: str,
    source_output_path: str,
    source_pass_k: int,
    target_pass_values: list[int],
    *,
    output_dir: str,
    results_file: str,
    model_name: str,
):
    for pass_k in sorted(target_pass_values):
        if pass_k > source_pass_k:
            raise ValueError(
                f"Cannot derive pass@{pass_k} from {source_output_path} which only contains pass@{source_pass_k} generations."
            )

        if pass_k == source_pass_k:
            current_output_path = source_output_path
        else:
            current_output_path = os.path.join(output_dir, f"{dataset_name}_pass{pass_k}_generation.parquet")
            slice_responses_in_generation_file(source_output_path, current_output_path, pass_k)

        accuracy = evaluate_generated_output(
            dataset_name,
            current_output_path,
            prompt_key="prompt",
            pass_k=pass_k,
            output_json_path=results_file,
            model_name=model_name,
        )
        print(f"  Completed {dataset_name} pass@{pass_k}: {accuracy:.2%}")


def main():
    args = parse_args()
    config = load_json(os.path.abspath(os.path.expanduser(args.config)))

    dataset_names = [normalize_eval_dataset_name(name) for name in config["datasets"]]
    pass_k_values = sorted({int(value) for value in config["pass_k_values"]})
    datasets_dir = config.get("datasets_dir", "./data/eval")
    dataset_paths = resolve_eval_dataset_paths(dataset_names, datasets_dir)

    missing_dataset_names = [name for name in dataset_names if name not in dataset_paths]
    if missing_dataset_names:
        raise ValueError(f"Unsupported dataset names in config: {missing_dataset_names}")

    default_temperature = float(config.get("temperature", 0.6))
    default_top_p = float(config.get("top_p", 0.95))
    default_nnodes = int(config.get("nnodes", 1))
    default_n_gpus_per_node = int(config.get("n_gpus_per_node", 4))
    default_gen_tp = int(config.get("gen_tp", 1))

    for model_config in config["models"]:
        model_path = normalize_model_path(model_config["model_path"])
        full_model_name, base_model_name, model_name = extract_model_identity(model_path)
        model_name = model_config.get("model_name", model_name)
        base_model_name = model_config.get("base_model_name", base_model_name)
        results_file = os.path.abspath(
            os.path.expanduser(model_config.get("results_file", config.get("results_file", f"results/{base_model_name}/results.json")))
        )
        output_dir = os.path.abspath(
            os.path.expanduser(model_config.get("output_dir", os.path.join("gen_results", "eval", full_model_name)))
        )
        tokenizer_path = model_config.get("tokenizer_path") or config.get("tokenizer_path")
        if tokenizer_path:
            tokenizer_path = os.path.abspath(os.path.expanduser(tokenizer_path))

        model_results = load_existing_results(results_file).get(model_name, {})
        missing_passes_by_dataset: dict[str, list[int]] = {}

        print("")
        print("################################################################################")
        print(f"# Model: {model_path}")
        print(f"# Model Name: {model_name}")
        print(f"# Results: {results_file}")
        print(f"# Output: {output_dir}")
        print("################################################################################")

        for dataset_name in dataset_names:
            missing_pass_values = [
                pass_k for pass_k in pass_k_values if find_existing_result_value(model_results, dataset_name, pass_k) is None
            ]
            if missing_pass_values:
                missing_passes_by_dataset[dataset_name] = missing_pass_values

        if not missing_passes_by_dataset:
            print("All requested results already exist. Skipping model.")
            print_model_summary(results_file, model_name, dataset_names, pass_k_values)
            write_comparison_csv(results_file, model_config, config, base_model_name)
            continue

        print("Pending evaluations:")
        for dataset_name, missing_pass_values in missing_passes_by_dataset.items():
            pending_passes = " ".join(f"pass@{pass_k}" for pass_k in missing_pass_values)
            print(f"  {dataset_name}: {pending_passes}")

        if args.dry_run:
            continue

        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        server_handles = []
        server_addresses = []
        try:
            datasets_requiring_generation = {}
            for dataset_name, missing_pass_values in missing_passes_by_dataset.items():
                reusable_output_path, reusable_pass_k = find_reusable_generation_file(
                    output_dir,
                    dataset_name,
                    max(missing_pass_values),
                )
                if reusable_output_path:
                    print(f"Reusing existing generation file for {dataset_name}: {reusable_output_path}")
                    evaluate_from_generation_file(
                        dataset_name,
                        reusable_output_path,
                        reusable_pass_k,
                        missing_pass_values,
                        output_dir=output_dir,
                        results_file=results_file,
                        model_name=model_name,
                    )
                else:
                    datasets_requiring_generation[dataset_name] = missing_pass_values

            if datasets_requiring_generation:
                max_requested_pass_k = max(max(pass_values) for pass_values in datasets_requiring_generation.values())
                server_handles, server_addresses = launch_generation_server(
                    model_path,
                    tokenizer_path,
                    temperature=float(model_config.get("temperature", default_temperature)),
                    top_p=float(model_config.get("top_p", default_top_p)),
                    pass_k=max_requested_pass_k,
                    nnodes=int(model_config.get("nnodes", default_nnodes)),
                    n_gpus_per_node=int(model_config.get("n_gpus_per_node", default_n_gpus_per_node)),
                    tensor_model_parallel_size=int(model_config.get("gen_tp", default_gen_tp)),
                )

                for dataset_name, missing_pass_values in datasets_requiring_generation.items():
                    required_pass_k = max(missing_pass_values)
                    output_path = os.path.join(output_dir, f"{dataset_name}_pass{required_pass_k}_generation.parquet")
                    print(f"Generating {dataset_name} with pass@{required_pass_k} ...")
                    generate_responses_with_server(
                        server_addresses,
                        model_path,
                        dataset_paths[dataset_name],
                        output_path,
                        prompt_key="prompt",
                        pass_k=required_pass_k,
                        temperature=float(model_config.get("temperature", default_temperature)),
                        top_p=float(model_config.get("top_p", default_top_p)),
                    )
                    evaluate_from_generation_file(
                        dataset_name,
                        output_path,
                        required_pass_k,
                        missing_pass_values,
                        output_dir=output_dir,
                        results_file=results_file,
                        model_name=model_name,
                    )
        finally:
            shutdown_generation_server(server_handles)

        print_model_summary(results_file, model_name, dataset_names, pass_k_values)
        write_comparison_csv(results_file, model_config, config, base_model_name)


if __name__ == "__main__":
    main()
