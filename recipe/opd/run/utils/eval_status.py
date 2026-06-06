#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def parse_datasets(raw):
    return [d.strip() for chunk in raw.replace(',', ' ').split() for d in [chunk.strip()] if d]


def has_number(entry, key):
    value = entry.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_entry(results_file, model_name):
    try:
        with open(results_file) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    entry = results.get(model_name)
    return entry if isinstance(entry, dict) else {}


def evalplus_file_exists(root, name):
    return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(f"evalplus/{name}/*eval_results*.json"))


def lcb_file_exists(root):
    return any(path.is_file() and path.stat().st_size > 0 for path in (root / "livecodebench").rglob("*_eval_all.json"))


def missing_for_code(entry, root, datasets):
    missing = []
    for ds in datasets:
        normalized = ds.replace('+', '_plus')
        if normalized in {"humaneval_plus", "humaneval"}:
            if not (has_number(entry, "humaneval_plus_avg4") and has_number(entry, "humaneval_plus_pass4") and evalplus_file_exists(root, "humaneval")):
                missing.append("humaneval_plus")
        elif normalized in {"mbpp_plus", "mbpp"}:
            if not (has_number(entry, "mbpp_plus_avg4") and has_number(entry, "mbpp_plus_pass4") and evalplus_file_exists(root, "mbpp")):
                missing.append("mbpp_plus")
        elif normalized in {"livecodebench_v6", "lcb_v6", "livecodebench"}:
            if not (has_number(entry, "livecodebench_v6_avg4") and has_number(entry, "livecodebench_v6_pass4") and lcb_file_exists(root)):
                missing.append("livecodebench_v6")
        else:
            missing.append(ds)
    return missing


def missing_for_math(entry, root, datasets, pass_k):
    missing = []
    for ds in datasets:
        prefix = f"openai/{ds}" if ds == "gsm8k" else ds
        if pass_k == 16:
            required = [
                f"{prefix}_avg_pass1_generation_pass_16",
                f"{prefix}_pass8_generation_pass_16",
                f"{prefix}_pass16_generation_pass_16",
            ]
            parquet = root / f"{ds}_pass16_generation.parquet"
            if not (all(has_number(entry, key) for key in required) and parquet.is_file() and parquet.stat().st_size > 0):
                missing.append(ds)
        elif not any(k.startswith(f"{prefix}_") for k in entry):
            missing.append(ds)
    return missing


def missing_items(task, results_file, model_name, datasets_raw, pass_k_raw, output_dir):
    pass_k = int(pass_k_raw or 0)
    datasets = parse_datasets(datasets_raw)
    root = Path(output_dir)
    entry = load_entry(results_file, model_name)
    if task == "code":
        return missing_for_code(entry, root, datasets)
    return missing_for_math(entry, root, datasets, pass_k)


def main() -> None:
    mode, task, results_file, model_name, datasets_raw, pass_k_raw, output_dir = sys.argv[1:]
    missing = missing_items(task, results_file, model_name, datasets_raw, pass_k_raw, output_dir)
    if mode == "missing":
        print(" ".join(missing))
        return
    if missing:
        print("Eval incomplete:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)
    print(f"Eval complete for {model_name}")


if __name__ == "__main__":
    main()
