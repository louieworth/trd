#!/usr/bin/env python3

from __future__ import annotations

import json

from math_verify import parse, verify
from sympy import solve

ANSWER_PREFIX_LIST = [
    "### the final answer is:",
    "### the final answer:",
    "### final answer is:",
    "### final answer:",
    "### the final answer is",
    "### the final answer",
    "### final answer is",
    "### final answer",
]
ANSWER_PREFIX_LIST_WO_HASHTAG = [prefix[4:] for prefix in ANSWER_PREFIX_LIST]
THINK_POSTFIX_LIST = ["</think>", "</longcat_think>"]
CUT_LIST = ["\\medskip", "\n---"]
REMOVE_LIST = [
    "\\bigl",
    "\\bigr",
    "\\Bigl",
    "\\Bigr",
    "\\biggl",
    "\\biggr",
    "\\Biggl",
    "\\Biggr",
    "\\bigg",
    "\\Bigg",
    "\\big",
    "\\Big",
    "\\left",
    "\\right",
]
REPLACE_LIST = [
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
    ("（", "("),
    ("）", ")"),
    ("，", ", "),
    ("：", ": "),
    ("；", "; "),
    ("。", ". "),
    ("！", "! "),
    ("？", "? "),
    ("…", "..."),
    ("–", "-"),
    ("−", "-"),
]


def pred_cut(pred_extract: str) -> str:
    for pattern in CUT_LIST:
        pred_extract = pred_extract.split(pattern)[0].strip()
    return pred_extract


def pred_extractor(pred: str, answer_type: str) -> str:
    pred_extract = pred.replace("：", ": ")

    for think_postfix in THINK_POSTFIX_LIST:
        pred_extract = pred_extract.split(think_postfix)[-1].strip()

    pred_extract_lower = pred_extract.lower()
    for prefix in ANSWER_PREFIX_LIST + ANSWER_PREFIX_LIST_WO_HASHTAG:
        if prefix in pred_extract_lower:
            suffix_lower = pred_extract_lower.split(prefix)[-1]
            pred_extract = pred_extract[-len(suffix_lower):].strip()
            pred_extract_lower = pred_extract.lower()
            break

    if answer_type != "description":
        for pattern in REMOVE_LIST:
            pred_extract = pred_extract.replace(pattern, "")

    for pattern, replacement in REPLACE_LIST:
        pred_extract = pred_extract.replace(pattern, replacement)

    while " }" in pred_extract:
        pred_extract = pred_extract.replace(" }", "}")
    while ".}" in pred_extract:
        pred_extract = pred_extract.replace(".}", "}")

    if answer_type in ["number", "variable", "set"]:
        pred_extract = pred_extract.replace("\\,", "")
        pred_extract = pred_extract.replace("\\;", "")
        pred_extract = pred_extract.replace("\n", " ")

    if answer_type in ["number", "variable"]:
        pred_extract = pred_extract.replace(",", "")
        pred_extract = pred_extract.replace("\\{", "(").replace("\\}", ")").replace("\\[", "(").replace("\\]", ")")

    return pred_extract.strip()


def verify_number_set_answer(pred_extract: str, answer: str) -> bool:
    pred_parse = parse(pred_extract)
    gold_parse = parse(answer)
    verify_result = verify(gold_parse, pred_parse, float_rounding=4) or verify(
        pred_parse, gold_parse, float_rounding=4
    )

    if pred_parse and "=" in pred_parse[-1]:
        pred_last_str = pred_parse[-1].split("=")[-1]
        pred_last_str_parse = parse("\\boxed{" + pred_last_str + "}")
        verify_last_result = verify(gold_parse, pred_last_str_parse, float_rounding=4) or verify(
            pred_last_str_parse, gold_parse, float_rounding=4
        )
        verify_result = verify_result or verify_last_result

    return bool(verify_result)


def verify_variable_answer(pred_extract: str, answer: str, try_list: list[str]) -> bool:
    pred_parse_ori = parse(pred_extract)
    if not pred_parse_ori:
        return False

    pred_parse_str = pred_parse_ori[-1]
    pred_parse_str = pred_parse_str.split("\\qquad")[-2].strip() if "\\qquad" in pred_parse_str else pred_parse_str
    pred_parse_str = pred_parse_str.split("\\quad")[-2].strip() if "\\quad" in pred_parse_str else pred_parse_str
    pred_parse_str = pred_parse_str.split("=")[-1]

    gold_parse_ori = parse(answer)
    gold_parse_str = gold_parse_ori[-1].split("=")[-1]

    for try_str in try_list:
        pred_parse_equ = parse("\\boxed{" + try_str + ", y=" + pred_parse_str + "}")
        gold_parse_equ = parse("\\boxed{" + try_str + ", y=" + gold_parse_str + "}")

        if not pred_parse_equ or not gold_parse_equ:
            return False

        pred_parse_solve = solve(pred_parse_equ[0])
        gold_parse_solve = solve(gold_parse_equ[0])
        if not pred_parse_solve or not gold_parse_solve:
            return False

        if isinstance(pred_parse_solve, list):
            pred_parse_solve = pred_parse_solve[0]
        if isinstance(gold_parse_solve, list):
            gold_parse_solve = gold_parse_solve[0]

        pred_parse_solve_y = None
        gold_parse_solve_y = None

        try:
            for symbol, value in pred_parse_solve.items():
                if str(symbol) == "y":
                    pred_parse_solve_y = value
            for symbol, value in gold_parse_solve.items():
                if str(symbol) == "y":
                    gold_parse_solve_y = value
        except Exception:
            return False

        if pred_parse_solve_y is None or gold_parse_solve_y is None:
            return False

        pred_parse_solve_y = pred_parse_solve_y.evalf()
        gold_parse_solve_y = gold_parse_solve_y.evalf()

        verify_result = verify(gold_parse_solve_y, pred_parse_solve_y, float_rounding=8) or verify(
            pred_parse_solve_y, gold_parse_solve_y, float_rounding=8
        )
        if not verify_result:
            return False

    return True


def compute_score(solution_str, ground_truth) -> float:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    if not isinstance(ground_truth, dict):
        return 0.0

    answer_type = ground_truth.get("answer_type")
    answer = ground_truth.get("answer")
    if answer_type not in {"number", "set", "variable"} or answer is None:
        return 0.0

    pred_extract = pred_extractor(solution_str, answer_type)

    try:
        if answer_type in {"number", "set"}:
            result = verify_number_set_answer(pred_extract, answer)
        else:
            result = verify_variable_answer(pred_extract, answer, ground_truth.get("try_list", []))
    except Exception:
        result = False

    if not result:
        pred_extract_cut = pred_cut(pred_extract)
        try:
            if answer_type in {"number", "set"}:
                result = verify_number_set_answer(pred_extract_cut, answer)
            else:
                result = verify_variable_answer(pred_extract_cut, answer, ground_truth.get("try_list", []))
        except Exception:
            result = False

    return 1.0 if result else 0.0
