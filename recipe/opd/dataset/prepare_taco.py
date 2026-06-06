#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import datasets


def first_solution(raw):
    if not raw:
        return ""
    if isinstance(raw, list):
        return raw[0] if raw else ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed[0] if parsed else ""
    except Exception:
        pass
    return str(raw)


def load_taco(split: str):
    from huggingface_hub import snapshot_download

    repo_dir = snapshot_download(
        repo_id="BAAI/TACO",
        repo_type="dataset",
        allow_patterns=[f"ALL/{split}-*.parquet"],
    )
    parquet_dir = Path(repo_dir) / "ALL"
    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No BAAI/TACO parquet files found for split={split} under {parquet_dir}")
    return datasets.load_dataset("parquet", data_files=[str(path) for path in files], split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache BAAI/TACO and write a local dataset directory for OPD code training.")
    parser.add_argument("--output_dir", default="./data/train/TACO")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--analyze_model", default="", help="Optional tokenizer/model path used to estimate prompt/solution lengths.")
    args = parser.parse_args()

    ds = load_taco(args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    ds.to_parquet(str(output_dir / f"{args.split}.parquet"))

    summary = {
        "dataset": "BAAI/TACO",
        "config": "ALL",
        "split": args.split,
        "num_rows": len(ds),
        "output_dir": str(output_dir),
        "columns": list(ds.column_names),
    }

    if args.analyze_model:
        from transformers import AutoTokenizer
        from recipe.opd.generation.y_o_prepare import code_instruction_following

        tokenizer = AutoTokenizer.from_pretrained(args.analyze_model, trust_remote_code=True)
        prompt_lens = []
        solution_lens = []
        for ex in ds:
            prompt = ex.get("question", "").strip()
            starter = ex.get("starter_code", "") or ""
            if starter.strip():
                prompt += "\n\nStarter code:\n```python\n" + starter.strip() + "\n```"
            prompt += "\n\n" + code_instruction_following
            prompt_lens.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
            solution_lens.append(len(tokenizer.encode(first_solution(ex.get("solutions", "")), add_special_tokens=False)))

        def pct(values, p):
            values = sorted(values)
            if not values:
                return 0
            return values[min(len(values) - 1, int(round((p / 100) * (len(values) - 1))))]

        summary["token_length_stats"] = {
            "prompt": {"p50": pct(prompt_lens, 50), "p90": pct(prompt_lens, 90), "p95": pct(prompt_lens, 95), "p99": pct(prompt_lens, 99), "max": max(prompt_lens or [0])},
            "solution": {"p50": pct(solution_lens, 50), "p90": pct(solution_lens, 90), "p95": pct(solution_lens, 95), "p99": pct(solution_lens, 99), "max": max(solution_lens or [0])},
        }

    with open(output_dir / "opd_taco_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
