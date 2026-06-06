"""Runtime compatibility patches loaded automatically by Python.

vLLM 0.11 still reads ``all_special_tokens_extended`` from tokenizer
instances. Transformers 5.x no longer exposes that attribute on Qwen
tokenizers, so vLLM EngineCore subprocesses fail during startup unless the
attribute is restored before vLLM initializes its tokenizer cache.
"""

import os

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _patch_numpy_compat():
    """Restore NumPy aliases still used by older Megatron-Core releases."""

    try:
        import numpy as np
    except Exception:
        return

    if not hasattr(np, "product"):
        np.product = np.prod


_patch_numpy_compat()


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)


def _patch_vllm_raw_full_vocab_logprobs():
    """Keep full-vocab logprobs in vocab order and avoid materializing indices.

    OPD full-vocab KL does not need per-token top-k indices: the implicit index
    is the vocab column. vLLM's prompt-logprobs path normally calls topk even
    when all vocab entries are requested, producing a huge int32 index matrix.
    This patch is enabled only for the teacher server via an env flag.
    """

    import numpy as np
    import torch

    from vllm.v1.outputs import LogprobsTensors
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def is_full_vocab_num_logprobs(value, vocab_size=None):
        try:
            value = int(value)
        except Exception:
            return False
        if value < 0:
            return True
        return vocab_size is not None and value >= int(vocab_size)

    def resolve_runner_vocab_size(runner):
        vocab_size = getattr(runner, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)

        model_config = getattr(runner, "model_config", None)
        if model_config is not None and hasattr(model_config, "get_vocab_size"):
            return int(model_config.get_vocab_size())

        input_batch = getattr(runner, "input_batch", None)
        vocab_size = getattr(input_batch, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)

        return None

    original_gather_logprobs = Sampler.gather_logprobs
    if not getattr(original_gather_logprobs, "_opd_raw_full_vocab", False):

        def gather_logprobs(logprobs, num_logprobs, token_ids):
            if is_full_vocab_num_logprobs(num_logprobs, logprobs.size(-1)):
                empty_ids = torch.empty((logprobs.size(0), 0), dtype=torch.int32, device=logprobs.device)
                ranks = torch.zeros((logprobs.size(0),), dtype=torch.int32, device=logprobs.device)
                return LogprobsTensors(empty_ids, logprobs, ranks)
            return original_gather_logprobs(logprobs, num_logprobs, token_ids)

        gather_logprobs._opd_raw_full_vocab = True
        Sampler.gather_logprobs = staticmethod(gather_logprobs)

    if not getattr(RequestState.make_sampling_metadata, "_opd_raw_full_vocab", False):
        original_make_sampling_metadata = RequestState.make_sampling_metadata

        def make_sampling_metadata(self, idx_mapping, idx_mapping_np, pos):
            metadata = original_make_sampling_metadata(self, idx_mapping, idx_mapping_np, pos)
            num_logprobs = self.num_logprobs[idx_mapping_np]
            if num_logprobs.size and np.all(num_logprobs < 0):
                metadata.max_num_logprobs = -1
            return metadata

        make_sampling_metadata._opd_raw_full_vocab = True
        RequestState.make_sampling_metadata = make_sampling_metadata

    prompt_method_name = (
        "_get_prompt_logprobs_dict"
        if hasattr(GPUModelRunner, "_get_prompt_logprobs_dict")
        else "_get_prompt_logprobs"
    )
    original_get_prompt_logprobs = getattr(GPUModelRunner, prompt_method_name)

    if getattr(original_get_prompt_logprobs, "_opd_raw_full_vocab", False):
        return

    def get_prompt_logprobs(self, hidden_states, num_scheduled_tokens):
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        vocab_size = resolve_runner_vocab_size(self)
        if not num_prompt_logprobs_dict or not any(
            is_full_vocab_num_logprobs(v, vocab_size) for v in num_prompt_logprobs_dict.values()
        ):
            return original_get_prompt_logprobs(self, hidden_states, num_scheduled_tokens)

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict = {}
        completed_prefill_reqs = []

        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            if not is_full_vocab_num_logprobs(num_prompt_logprobs, vocab_size):
                return original_get_prompt_logprobs(self, hidden_states, num_scheduled_tokens)

            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                continue

            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                continue

            num_prompt_tokens = len(request.prompt_token_ids)

            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                logprobs_tensors = LogprobsTensors(
                    torch.empty(0, dtype=torch.int32, device="cpu"),
                    torch.empty((num_prompt_tokens - 1, vocab_size), dtype=torch.float32, device="cpu"),
                    torch.empty(0, dtype=torch.int32, device="cpu"),
                )
                in_progress_dict[req_id] = logprobs_tensors

            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)
            logprobs = self.sampler.compute_logprobs(logits)

            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)

        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    get_prompt_logprobs._opd_raw_full_vocab = True
    setattr(GPUModelRunner, prompt_method_name, get_prompt_logprobs)


if str(os.environ.get("OPD_PATCH_VLLM_FULL_VOCAB_RAW", "")).lower() in {"1", "true", "yes", "y"}:
    try:
        _patch_vllm_raw_full_vocab_logprobs()
    except Exception as exc:
        print(f"[sitecustomize] failed to patch vLLM raw full-vocab logprobs: {exc}", flush=True)
