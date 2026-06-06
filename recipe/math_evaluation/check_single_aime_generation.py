#!/usr/bin/env python3
"""Small HF generation check for one local model on selected AIME rows."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe.math_evaluation.compute_score import compute_score_data_source


def short_text(text: str, max_chars: int = 2400) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rows", default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--num-return", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.to("cuda:0")
    model.eval()

    df = pd.read_parquet(args.dataset)
    rows = [int(x) for x in args.rows.split(",") if x.strip()]

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "rows": rows,
        "max_new_tokens": args.max_new_tokens,
        "num_return": args.num_return,
        "items": [],
    }

    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = model.generation_config.pad_token_id or tokenizer.pad_token_id or tokenizer.eos_token_id

    for idx in rows:
        row = df.iloc[idx]
        chat_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(chat_text, return_tensors="pt").to("cuda:0")
        generation_kwargs = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "num_return_sequences": args.num_return,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if args.top_k is not None:
            generation_kwargs["top_k"] = args.top_k
        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                **generation_kwargs,
            )

        input_len = int(encoded["input_ids"].shape[-1])
        ground_truth = row["reward_model"]["ground_truth"]
        generations = []
        for seq in outputs:
            gen_ids = seq[input_len:].detach().cpu().tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
            try:
                score = float(compute_score_data_source(row["data_source"], text, ground_truth))
            except Exception as exc:
                score = 0.0
                score_error = repr(exc)
            else:
                score_error = None
            generations.append(
                {
                    "generated_tokens": len(gen_ids),
                    "first_token_id": gen_ids[0] if gen_ids else None,
                    "score": score,
                    "score_error": score_error,
                    "contains_boxed": "\\boxed" in text,
                    "contains_ground_truth": str(ground_truth) in text,
                    "text": short_text(text),
                    "raw": short_text(raw),
                }
            )

        result["items"].append(
            {
                "row": idx,
                "ground_truth": ground_truth,
                "input_tokens": input_len,
                "generations": generations,
            }
        )

        with out.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"row={idx} scores={[g['score'] for g in generations]} tokens={[g['generated_tokens'] for g in generations]}", flush=True)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
