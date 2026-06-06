# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_SOLUTION_CLIP_CHARS = 300


def _normalize_number(s):
    """Strip commas and dollar signs; return None for empty/punctuation-only strings."""
    s = s.replace(",", "").replace("$", "").strip()
    return s if s not in ("", ".") else None


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # Try the prompt-requested "#### <num>" format first.
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if solutions:
            return _normalize_number(solutions[-1])

        # Fallback: \boxed{<content>}. Non-instruct models (e.g. base Qwen3-8B)
        # ignore the "####" instruction and default to LaTeX boxed format on
        # math data. Accept it rather than marking them all as 0.
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", solution_str)
        if boxed:
            nums = re.findall(r"\-?[0-9\.\,]+", boxed[-1])
            for n in reversed(nums):
                norm = _normalize_number(n)
                if norm is not None:
                    return norm

        # Last-resort fallback: the final number in the clipped tail.
        nums = re.findall(r"\-?[0-9\.\,]+", solution_str)
        for n in reversed(nums):
            norm = _normalize_number(n)
            if norm is not None:
                return norm
        return None

    # method == "flexible"
    nums = re.findall(r"\-?[0-9\.\,]+", solution_str)
    for n in reversed(nums):
        norm = _normalize_number(n)
        if norm is not None:
            return norm
    return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
