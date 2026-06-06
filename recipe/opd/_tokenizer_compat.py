#!/usr/bin/env python3

"""Compatibility patches for tokenizer/runtime mismatches in KL recipes."""


def apply_qwen2_tokenizer_vllm_compat() -> bool:
    """Backfill ``all_special_tokens_extended`` expected by vLLM.

    In the current environment, Transformers 5.3.0 loads Qwen3 text tokenizers
    as ``Qwen2Tokenizer``. vLLM 0.11.0 assumes this tokenizer exposes
    ``all_special_tokens_extended``, but that attribute is absent and causes
    server startup to fail. Returning ``all_special_tokens`` is sufficient for
    this rollout path.
    """

    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    except Exception:
        return False

    if hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
        return False

    def _all_special_tokens_extended(self):
        return list(self.all_special_tokens)

    Qwen2Tokenizer.all_special_tokens_extended = property(_all_special_tokens_extended)
    return True
