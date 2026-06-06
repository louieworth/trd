#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PASS = "pass"


def find_json(root: Path, dataset: str) -> Path:
    dataset_root = root / dataset
    candidates = []
    if dataset_root.exists():
        candidates.extend(sorted(dataset_root.rglob("*eval_results*.json")))
    candidates.extend(
        path
        for path in sorted(root.rglob("*eval_results*.json"))
        if dataset in path.parts
    )
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"No EvalPlus {dataset} eval_results json found under {root}")


def is_correct(sample: dict) -> bool:
    base = str(sample.get("base_status", "")).lower()
    plus = sample.get("plus_status")
    if plus is None:
        return base == PASS
    return base == PASS and str(plus).lower() == PASS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pass_k", type=int, default=4)
    args = parser.parse_args()
    path = find_json(Path(args.root), args.dataset)
    with open(path) as f:
        data = json.load(f)

    eval_by_task = data.get("eval", {})
    if not eval_by_task:
        raise SystemExit(f"No EvalPlus eval entries found in {path}")

    avg_values = []
    pass_values = []
    for samples in eval_by_task.values():
        if not samples:
            continue
        limited = samples[: args.pass_k]
        correct = sum(1 for sample in limited if is_correct(sample))
        avg_values.append(correct / len(limited))
        pass_values.append(1.0 if correct > 0 else 0.0)
    if not avg_values:
        raise SystemExit(f"No EvalPlus samples found in {path}")
    print(f"avg={sum(avg_values) / len(avg_values)}")
    print(f"pass={sum(pass_values) / len(pass_values)}")


if __name__ == "__main__":
    main()
