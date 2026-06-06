#!/usr/bin/env python3
"""Read a pass@16 generation parquet, score each of the 16 responses, and write
avg_pass@1 / pass@8 / pass@16 into results.json under {model_name}.

Keys written:
  {bench}_avg_pass1_generation_pass_16  — mean of per-response correctness over all 16
  {bench}_pass8_generation_pass_16      — "any of first 8 correct" per question, then mean
  {bench}_pass16_generation_pass_16     — "any of 16 correct" per question, then mean
"""
import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd

from recipe.math_evaluation.compute_score import compute_score_data_source


def score_row(resp_str: str, gt, data_source: str) -> float:
    try:
        return float(compute_score_data_source(data_source, str(resp_str), gt))
    except Exception:
        return 0.0


def aggregate_bench(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    per_q_scores = []     # list of list of 16 scores per question
    for _, row in df.iterrows():
        ds = row.get("data_source", "deepscaleR")
        rm = row.get("reward_model")
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        resps = row.get("responses")
        if resps is None or len(resps) == 0:
            per_q_scores.append([0.0] * 16)
            continue
        scores = [score_row(r, gt, ds) for r in resps]
        # pad to 16 if fewer samples
        while len(scores) < 16:
            scores.append(0.0)
        per_q_scores.append(scores[:16])
    mat = np.array(per_q_scores)  # shape (n, 16)
    avg_pass1 = float(mat.mean())
    pass8 = float((mat[:, :8].max(axis=1) >= 1.0).mean())
    pass16 = float((mat[:, :16].max(axis=1) >= 1.0).mean())
    return {"avg_pass1": avg_pass1, "pass8": pass8, "pass16": pass16, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="dir with {bench}_pass16_generation.parquet")
    ap.add_argument("--results_file", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_path", default="")
    ap.add_argument("--datasets", required=True, help="comma-separated")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    updates = {}
    for ds in datasets:
        parq = os.path.join(args.gen_dir, f"{ds}_pass16_generation.parquet")
        if not os.path.exists(parq):
            print(f"  SKIP {ds}: parquet not found at {parq}")
            continue
        r = aggregate_bench(parq)
        print(f"  {ds:12s}  n={r['n']}  avg_pass1={r['avg_pass1']:.4f}  pass8={r['pass8']:.4f}  pass16={r['pass16']:.4f}")
        # openai/ prefix for gsm8k
        prefix = f"openai/{ds}" if ds == "gsm8k" else ds
        updates[f"{prefix}_avg_pass1_generation_pass_16"] = r["avg_pass1"]
        updates[f"{prefix}_pass8_generation_pass_16"] = r["pass8"]
        updates[f"{prefix}_pass16_generation_pass_16"] = r["pass16"]

    # read results.json, update, atomic write
    with open(args.results_file) as f:
        results = json.load(f)
    entry = results.setdefault(args.model_name, {})
    entry["model_path"] = args.model_path or None
    entry.update(updates)

    d = os.path.dirname(args.results_file)
    fd, tmp = tempfile.mkstemp(prefix=".results-", suffix=".json", dir=d)
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        os.chmod(tmp, 0o664)
        os.replace(tmp, args.results_file)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"\nWrote {len(updates)} keys to {args.model_name} in {args.results_file}")


if __name__ == "__main__":
    main()
