#!/usr/bin/env python3
import os
import sys
import tempfile
import pandas as pd
from transformers import AutoTokenizer


def prompt_to_text(prompt):
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, (list, tuple)):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(prompt)


def main() -> None:
    path, max_tokens, tokenizer_path, label = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    df = pd.read_parquet(path)
    if "prompt" not in df.columns or len(df) == 0:
        print(f"[Prompt length filter] {label}: no prompt rows to filter ({path})")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = [len(tokenizer.encode(prompt_to_text(x), add_special_tokens=False)) for x in df["prompt"]]
    keep = [n <= max_tokens for n in lengths]
    dropped = len(df) - sum(keep)
    if dropped == 0:
        max_seen = max(lengths) if lengths else 0
        print(f"[Prompt length filter] {label}: kept all {len(df)} rows (max_prompt_tokens={max_tokens}, max_seen={max_seen})")
        return

    filtered = df.loc[keep].reset_index(drop=True)
    out_dir = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".prompt_filter_", suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        filtered.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    max_seen = max(lengths) if lengths else 0
    print(
        f"[Prompt length filter] {label}: dropped {dropped}/{len(df)} "
        f"({dropped / len(df):.4%}) rows with prompt_tokens>{max_tokens}; "
        f"max_seen={max_seen}; wrote {len(filtered)} rows -> {path}"
    )


if __name__ == "__main__":
    main()
