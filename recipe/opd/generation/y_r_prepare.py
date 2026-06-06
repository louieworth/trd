#!/usr/bin/env python3
#
# y_r prompt preparation — build y_r rewrite prompts for every y_o row.
#
# A "y_r" example is the teacher's rewrite of the student's y_o attempt.
# This script takes the y_o parquet (student rollouts) and produces the
# prompt that the teacher will rollout against to generate y_r.
#
# The generation prompt is FIXED (always the refine variant — teacher reads
# the initial response). The two templates differ only by whether they show
# the expert solution:
#
#   --distill_mode opsd  →  refine OPSD: π_T(·|x, y*, y_o)
#   --distill_mode opd   →  refine OPD:  π_T(·|x, y_o)
#
# The training-time teacher conditioning is a separate knob (driven by
# TEACHER_TRAINING_PROMPT in run_kl_training.sh / use_initial_response in
# config.py) and may be vanilla or refine independently.

import argparse
import os
from typing import Any, Dict

import datasets

instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
code_instruction_following = "Return only the corrected Python code inside a single ```python code block."


# refine OPSD: teacher sees problem + expert solution + initial response.
PROMPT_TEMPLATE_OPSD_REFINE = """
Your task is to rewrite your mathematical solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
{EXPERT_SOLUTION}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Review the reference solution to understand the target reasoning and method
2. Rewrite your solution so it is consistent with the reference solution
3. Keep useful parts of your original structure and style when appropriate
4. Output ONLY the rewritten solution
"""

# refine OPD: teacher sees problem + initial response (no expert reference).
PROMPT_TEMPLATE_OPD_REFINE = """
Your task is to rewrite your mathematical solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Preserve the overall structure and reasoning path of your original solution
2. Identify and fix errors in computation or logic
3. Keep correct intermediate steps and meaningful work
4. Output ONLY the rewritten solution
"""



PROMPT_TEMPLATE_CODE_OPSD_REFINE = """
Your task is to rewrite your Python solution using the reference solution as guidance.

**Problem:**
{PROBLEM}

**Reference Solution:**
```python
{EXPERT_SOLUTION}
```

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

PROMPT_TEMPLATE_CODE_OPD_REFINE = """
Your task is to rewrite your Python solution.

**Problem:**
{PROBLEM}

**Your Initial Solution:**
{INITIAL_RESPONSE}

**Instructions:**
1. Fix correctness issues and edge cases
2. Preserve useful parts of the original approach when appropriate
3. Output ONLY the rewritten Python solution
"""

def _normalize_reward_value(val: Any) -> int:
    if isinstance(val, list):
        val = val[0] if val else 0
    try:
        return int(val)
    except Exception:
        return 0


def make_map_fn(distill_mode: str, task: str = "math"):
    def process_fn(example: Dict[str, Any]):
        extra_info = (example.get("extra_info", {}) or {}).copy()
        extra_info["reward"] = _normalize_reward_value(extra_info.get("reward", 0))

        responses_val = example.get("responses", [""])
        initial_response = responses_val[0] if isinstance(responses_val, list) else responses_val
        extra_info["initial_response"] = initial_response

        problem = extra_info.get("problem", "")
        expert_solution = extra_info.get("expert_cot", "")

        if task == "code":
            if distill_mode == "opd":
                prompt_content = (
                    PROMPT_TEMPLATE_CODE_OPD_REFINE
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_response)
                )
            else:
                prompt_content = (
                    PROMPT_TEMPLATE_CODE_OPSD_REFINE
                    .replace("{PROBLEM}", problem)
                    .replace("{INITIAL_RESPONSE}", initial_response)
                    .replace("{EXPERT_SOLUTION}", expert_solution)
                )
            prompt_content = prompt_content + " " + code_instruction_following
        elif distill_mode == "opd":
            prompt_content = (
                PROMPT_TEMPLATE_OPD_REFINE
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
            )
            prompt_content = prompt_content + " " + instruction_following
        else:  # opsd
            prompt_content = (
                PROMPT_TEMPLATE_OPSD_REFINE
                .replace("{PROBLEM}", problem)
                .replace("{INITIAL_RESPONSE}", initial_response)
                .replace("{EXPERT_SOLUTION}", expert_solution)
            )
            prompt_content = prompt_content + " " + instruction_following

        row = {
            "data_source": example.get("data_source", "deepscaleR"),
            "prompt": [{"role": "user", "content": prompt_content}],
            "ability": example.get("ability", "code" if task == "code" else "math"),
            "extra_info": extra_info,
        }
        if task != "code":
            row["reward_model"] = example.get("reward_model")
        return row

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--y_o_output",
        required=True,
        help="y_o output parquet (with responses and reward).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output parquet for y_r prompts.",
    )
    parser.add_argument(
        "--task",
        choices=["math", "code"],
        default="math",
        help="Task-specific rewrite prompt adapter.",
    )
    parser.add_argument(
        "--distill_mode",
        choices=["opsd", "opd"],
        default="opsd",
        help="opsd: teacher sees expert solution + initial response. "
             "opd:  teacher sees only initial response (no expert reference).",
    )
    args = parser.parse_args()

    print(f"Loading y_o output: {args.y_o_output}")
    ds_raw = datasets.load_dataset("parquet", data_files=args.y_o_output, split="train")

    print(f"Building y_r prompts ({len(ds_raw)} rows, refine {args.distill_mode.upper()})...")
    ds_out = ds_raw.map(
        function=make_map_fn(args.distill_mode, args.task),
        remove_columns=["responses"],
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(ds_out)} y_r prompts to: {args.output_file}")
    ds_out.to_parquet(args.output_file)
    print("y_r prompt preparation completed.")


if __name__ == "__main__":
    main()
