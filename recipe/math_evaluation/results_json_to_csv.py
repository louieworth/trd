#!/usr/bin/env python3
"""Convert math evaluation results JSON into comparison CSV files.

The JSON remains the source of truth. This script derives a flat table with one
row per evaluated model and stable metadata columns parsed from the OPD/OPSD
model key. The exported metric columns are intentionally compact: five avg@16
columns, then five pass@16 columns for the canonical math eval datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any


META_COLUMNS = [
    "model",
    "task",
    "is_lora",
    "teacher",
    "y_mode",
    "kl_type",
    "kl_method",
    "clip_ratio",
    "top_K",
    "multi_turn",
]

SUMMARY_DATASETS = [
    "aime24",
    "aime25",
    "hmmt25",
    "beyondaime",
    "amobench",
]

AVG16_COLUMNS = [f"{dataset}_avg@16" for dataset in SUMMARY_DATASETS]
PASS16_COLUMNS = [f"{dataset}_pass@16" for dataset in SUMMARY_DATASETS]
SUMMARY_COLUMNS = AVG16_COLUMNS + PASS16_COLUMNS

DATASET_ORDER = SUMMARY_DATASETS + [
    "math500",
    "gsm8k",
    "openai/gsm8k",
]

METRIC_ORDER = [
    "avg_pass1_generation_pass_16",
    "pass16_generation_pass_16",
    "pass1_generation_pass_1",
]


def load_results(path: str | os.PathLike[str] | None) -> OrderedDict[str, dict[str, Any]]:
    if not path or not os.path.exists(path):
        return OrderedDict()
    with open(path) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top level in {path}")
    return OrderedDict((str(k), v if isinstance(v, dict) else {}) for k, v in data.items())


def default_csv_path(results_file: str | os.PathLike[str]) -> str:
    return str(Path(results_file).with_suffix(".csv"))



def default_base_results_file(results_file: str | os.PathLike[str], base_model_name: str | None = None) -> str | None:
    if not base_model_name:
        base_model_name = Path(results_file).stem

    current = Path(results_file).resolve()
    for parent in [current.parent, *current.parents]:
        if parent.name == "results":
            return str(parent / "base" / f"{base_model_name}.json")
    return str(current.parent.parent / "base" / f"{base_model_name}.json")


def format_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def parse_clip_value(raw_value: str) -> str:
    if not raw_value:
        return ""
    if raw_value == "0" or "." in raw_value:
        return raw_value
    if raw_value.isdigit() and raw_value.startswith("0"):
        return f"0.{raw_value[1:].rstrip('0') or '0'}"
    return raw_value


def metric_sort_key(metric_name: str) -> tuple[int, int, str]:
    dataset_rank = len(DATASET_ORDER)
    metric_rank = len(METRIC_ORDER)
    for idx, dataset in enumerate(DATASET_ORDER):
        prefix = f"{dataset}_"
        if metric_name.startswith(prefix):
            dataset_rank = idx
            suffix = metric_name[len(prefix) :]
            if suffix in METRIC_ORDER:
                metric_rank = METRIC_ORDER.index(suffix)
            return dataset_rank, metric_rank, metric_name
    return dataset_rank, metric_rank, metric_name


def collect_metric_columns(*result_sets: OrderedDict[str, dict[str, Any]]) -> list[str]:
    # Kept for compatibility with older callers; exports now use SUMMARY_COLUMNS.
    metric_names: set[str] = set()
    for results in result_sets:
        for entry in results.values():
            for key, value in entry.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, (int, float, str, bool)) or value is None:
                    metric_names.add(key)
    return sorted(metric_names, key=metric_sort_key)


def parse_model_key(model_key: str, fallback_model: str | None = None) -> dict[str, str]:
    row = {
        "model": fallback_model or "",
        "task": "",
        "is_lora": "",
        "teacher": "",
        "y_mode": "",
        "kl_type": "",
        "kl_method": "",
        "clip_ratio": "",
        "top_K": "",
        "multi_turn": "",
    }

    normalized_key = model_key.upper()
    if "_MATH_" in normalized_key:
        row["task"] = "math"
    elif "_CODE_" in normalized_key:
        row["task"] = "code"

    if "NO_LORA" in normalized_key or model_key.endswith("_base"):
        row["is_lora"] = "false"
    elif "LORA" in normalized_key:
        row["is_lora"] = "true"

    if model_key.endswith("_base"):
        base_name = model_key[: -len("_base")]
        base_name = re.sub(r"_(MATH|CODE)$", "", base_name, flags=re.IGNORECASE)
        row["model"] = base_name
        row["teacher"] = "base"
        row["task"] = row["task"] or "math"
        row["y_mode"] = "base"
        row["kl_type"] = "base"
        return row

    parts = model_key.split("_")
    if parts:
        row["model"] = parts[0]

    if "teacher" in model_key:
        teacher_match = re.search(r"(?:^|_)teacher([^_]+)", model_key)
        if teacher_match:
            row["teacher"] = teacher_match.group(1)

    if "_y_o_" in model_key:
        row["y_mode"] = "y_o"
    elif "_y_r_" in model_key:
        row["y_mode"] = "y_r"

    kl_match = re.search(r"_kl_([^_]+)(?:_([^_]+(?:_[^_]+)*))?_clip", model_key)
    if kl_match:
        row["kl_type"] = kl_match.group(1)
        row["kl_method"] = kl_match.group(2) or ""

    clip_match = re.search(r"_clip([^_]+)", model_key)
    if clip_match:
        row["clip_ratio"] = parse_clip_value(clip_match.group(1))

    topk_match = re.search(r"_topk([0-9]+)", model_key)
    row["top_K"] = topk_match.group(1) if topk_match else "0"

    ms_match = re.search(r"_ms([0-9]+)(?:_|$)", model_key)
    if ms_match:
        row["multi_turn"] = ms_match.group(1)

    return row


def rows_for_results(
    results: OrderedDict[str, dict[str, Any]],
    metric_columns: list[str],
    *,
    fallback_model: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for model_key, entry in results.items():
        row = parse_model_key(model_key, fallback_model=fallback_model)
        for dataset in SUMMARY_DATASETS:
            row[f"{dataset}_avg@16"] = format_scalar(entry.get(f"{dataset}_avg_pass1_generation_pass_16", ""))
        for dataset in SUMMARY_DATASETS:
            row[f"{dataset}_pass@16"] = format_scalar(entry.get(f"{dataset}_pass16_generation_pass_16", ""))
        rows.append(row)
    return rows



def write_results_csv(
    results_file: str | os.PathLike[str],
    output_file: str | os.PathLike[str] | None = None,
    *,
    base_results_file: str | os.PathLike[str] | None = None,
    base_model_name: str | None = None,
) -> str:
    results_file = str(results_file)
    output_file = str(output_file or default_csv_path(results_file))
    base_results_file = str(base_results_file or default_base_results_file(results_file, base_model_name))

    results = load_results(results_file)
    base_results = load_results(base_results_file)
    metric_columns = SUMMARY_COLUMNS
    fieldnames = META_COLUMNS + metric_columns

    rows: list[dict[str, Any]] = []
    rows.extend(rows_for_results(base_results, metric_columns, fallback_model=base_model_name))

    base_keys = set(base_results.keys())
    non_base_results = OrderedDict((k, v) for k, v in results.items() if k not in base_keys)
    rows.extend(rows_for_results(non_base_results, metric_columns, fallback_model=base_model_name))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".results-", suffix=".csv", dir=os.path.dirname(output_file) or ".")
    os.close(fd)
    try:
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.chmod(tmp, 0o664)
        os.replace(tmp, output_file)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert results JSON to comparison CSV files.")
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--base_results_file", default=None)
    parser.add_argument("--base_model_name", default=None)
    args = parser.parse_args()

    output_file = write_results_csv(
        args.results_file,
        args.output_file,
        base_results_file=args.base_results_file,
        base_model_name=args.base_model_name,
    )
    print(f"Wrote CSV comparison: {output_file}")


if __name__ == "__main__":
    main()
