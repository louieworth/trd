#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_eval_all(root: Path) -> Path:
    candidates = sorted(root.rglob("*_eval_all.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit(f"No LiveCodeBench *_eval_all.json found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pass_k", type=int, default=4)
    args = parser.parse_args()
    path = find_eval_all(Path(args.root))
    with open(path) as f:
        rows = json.load(f)
    avg_values = []
    pass_values = []
    for row in rows:
        graded = row.get("graded_list") or []
        if not graded:
            continue
        limited = graded[: args.pass_k]
        correct = sum(1 for value in limited if bool(value))
        avg_values.append(correct / len(limited))
        pass_values.append(1.0 if correct > 0 else 0.0)
    if not avg_values:
        raise SystemExit(f"No LiveCodeBench graded_list entries found in {path}")
    print(f"avg={sum(avg_values) / len(avg_values)}")
    print(f"pass={sum(pass_values) / len(pass_values)}")


if __name__ == "__main__":
    main()
