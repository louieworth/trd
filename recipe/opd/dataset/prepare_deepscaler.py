#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets


REQUIRED_COLUMNS = {"problem", "answer", "solution"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/cache DeepScaleR and write a local dataset directory for OPD/OPSD math training."
    )
    parser.add_argument("--output_dir", default="./data/train/DeepScaleR")
    parser.add_argument("--dataset", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    ds = datasets.load_dataset(args.dataset, split=args.split)
    missing = sorted(REQUIRED_COLUMNS.difference(ds.column_names))
    if missing:
        raise ValueError(f"DeepScaleR dataset is missing required columns: {missing}")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    ds.to_parquet(str(output_dir / f"{args.split}.parquet"))

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_rows": len(ds),
        "output_dir": str(output_dir),
        "columns": list(ds.column_names),
        "required_columns": sorted(REQUIRED_COLUMNS),
    }
    with open(output_dir / "opd_deepscaler_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
