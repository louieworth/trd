#!/usr/bin/env python3
#
# KL divergence trainer backed by verl FSDP TrainingWorker engines.

import glob
import json
import logging
import math
import os
import subprocess
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from verl.trainer.config.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

from .config import KLTrainingConfig
from .dataset.data_utils import create_kl_dataloader
from recipe.math_evaluation.eval_utils import run_evaluation_suite
from .kl_utils import compute_kl_divergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_eval_tag(config: KLTrainingConfig) -> str:
    clip_tag = f"clip{str(config.kl_token_clip).replace('.', '')}"
    return f"kl_{config.kl_type}_{config.kl_method}_{config.prompt_mode_tag}_{clip_tag}"


def build_eval_model_name(config: KLTrainingConfig) -> str:
    base_model_name = config.base_model_name or config.student_model_path.split("/")[-1]
    return f"{base_model_name}_{build_eval_tag(config)}_epoch{config.epoch_index}"


class KLTrainer:
    """Token-level KL training using verl's sharded FSDP engines."""

    def __init__(self, config: KLTrainingConfig, *, sync_only: bool = False):
        self.config = config
        self.global_step = 0
        self.epoch = 0
        self.sync_only = sync_only

        self.local_rank, self.rank, self.world_size = initialize_global_process_group()
        self.config.local_rank = self.local_rank
        self.config.world_size = self.world_size

        if self.sync_only:
            self.tokenizer = None
            self.train_dataloader = None
            self.total_training_steps = 1
        else:
            self.tokenizer = self._load_tokenizer()
            self.train_dataloader = self._create_dataloader()
            self.total_training_steps = (
                math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
                * self.config.total_epochs
            )

        self.student_worker = self._build_student_worker()
        self.student_engine = self.student_worker.engine
        if self.sync_only:
            self.teacher_worker = None
            self.teacher_engine = None
        else:
            self.teacher_worker = self._build_teacher_worker()
            self.teacher_engine = self.teacher_worker.engine

        self.is_logging = (
            self.student_engine.is_mp_src_rank_with_outputs() and self.student_engine.get_data_parallel_rank() == 0
        )

        if not self.sync_only and self.is_logging and HAS_WANDB:
            self._init_wandb()


        logger.info("KL Trainer initialized with verl FSDP backend%s", " (sync-only)" if self.sync_only else "")
        logger.info("  KL Type: %s", self.config.kl_type)
        logger.info("  KL Method: %s", self.config.kl_method)
        logger.info("  Student Model: %s", self.config.student_model_path)
        logger.info("  Teacher Model: %s", self.config.teacher_model_path or self.config.student_model_path)
        logger.info("  World Size: %s", self.world_size)
        logger.info("  FSDP Strategy: %s", self.config.fsdp_strategy)
        logger.info("  FSDP Size: %s", self.config.fsdp_size)
        logger.info("  Max Token Len/GPU: %s", self.config.max_token_len_per_gpu)
        logger.info("  Per-GPU Batch Size: %s", self.config.train_batch_size)
        logger.info("  Grad Accum Steps: %s", self.config.gradient_accumulation_steps)

    def close(self):
        if not self.sync_only and HAS_WANDB and self.is_logging:
            wandb.finish()
        destroy_global_process_group()

    def _init_wandb(self):
        effective_teacher_path = self.config.teacher_model_path or self.config.student_model_path
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            mode=os.environ.get("WANDB_MODE", "offline"),
            config={
                "distill_mode": self.config.distill_mode,
                "task": self.config.task,
                "kl_type": self.config.kl_type,
                "kl_method": self.config.kl_method,
                "temperature": self.config.temperature,
                "prompt_mode": self.config.prompt_mode_tag,
                "use_initial_response": self.config.use_initial_response,
                "student_model_path": self.config.student_model_path,
                "student_model": self.config.student_model_path.rstrip("/").split("/")[-1],
                "teacher_model_path": effective_teacher_path,
                "teacher_model": effective_teacher_path.rstrip("/").split("/")[-1],
                "use_lora": self.config.use_lora,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "train_batch_size_per_gpu": self.config.train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "total_epochs": self.config.total_epochs,
                "max_length": self.config.max_length,
                "max_token_len_per_gpu": self.config.max_token_len_per_gpu,
                "fsdp_strategy": self.config.fsdp_strategy,
                "fsdp_size": self.config.fsdp_size,
            },
        )
        logger.info("Wandb initialized: %s", wandb.run.url)

    def _load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.student_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _create_dataloader(self):
        # Shard data by DP rank, not global rank: under Ulysses SP, ranks in
        # the same SP group must see identical inputs, otherwise the SP
        # all-to-all in attention deadlocks (mesh layout is row-major
        # (dp, sp), so dp_rank = global_rank // sp_size).
        sp_size = self.config.ulysses_sequence_parallel_size
        dp_size = self.world_size // sp_size
        dp_rank = self.rank // sp_size
        return create_kl_dataloader(
            data_path=self.config.data_path,
            tokenizer=self.tokenizer,
            kl_type=self.config.kl_type,
            batch_size=self.config.train_batch_size,
            max_length=self.config.max_length,
            max_samples=self.config.max_samples,
            corrected_responses_path=self.config.corrected_responses_path,
            use_initial_response=self.config.use_initial_response,
            num_workers=self.config.num_workers,
            local_rank=dp_rank,
            world_size=dp_size,
            prompt_truncation=getattr(self.config, "prompt_truncation", False),
            distill_mode=getattr(self.config, "distill_mode", "opsd"),
            task=getattr(self.config, "task", "math"),
        )

    def _build_model_config(self, model_path: str, *, trainable: bool) -> HFModelConfig:
        return HFModelConfig(
            path=model_path,
            tokenizer_path=model_path,
            trust_remote_code=True,
            load_tokenizer=False,
            enable_gradient_checkpointing=trainable and self.config.gradient_checkpointing,
            use_remove_padding=self.config.use_remove_padding,
            lora_rank=self.config.lora_rank if trainable and self.config.use_lora else 0,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
        )

    def _build_wrap_policy(self, model_path: str) -> dict[str, list[str]]:
        model_identifiers = [
            model_path,
            self.config.base_model_name,
            self.config.student_model_path,
            self.config.teacher_model_path,
        ]
        if any(identifier and "Qwen3" in identifier for identifier in model_identifiers):
            return {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]}
        return {}

    def _build_engine_config(self, model_path: str, *, forward_only: bool) -> FSDPEngineConfig:
        return FSDPEngineConfig(
            strategy=self.config.fsdp_strategy,
            fsdp_size=self.config.fsdp_size,
            ulysses_sequence_parallel_size=self.config.ulysses_sequence_parallel_size,
            wrap_policy=self._build_wrap_policy(model_path),
            forward_only=forward_only,
            use_dynamic_bsz=True,
            max_token_len_per_gpu=self.config.max_token_len_per_gpu,
            infer_max_token_len_per_gpu=self.config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=None,
            infer_micro_batch_size_per_gpu=None,
            use_remove_padding=self.config.use_remove_padding,
            use_torch_compile=self.config.use_torch_compile,
            param_offload=self.config.param_offload,
            optimizer_offload=self.config.optimizer_offload,
            offload_policy=self.config.offload_policy,
            dtype="bfloat16" if self.config.bf16 else "float16",
        )

    def _build_optimizer_config(self) -> FSDPOptimizerConfig:
        return FSDPOptimizerConfig(
            lr=self.config.learning_rate,
            lr_warmup_steps_ratio=self.config.warmup_steps_ratio,
            total_training_steps=self.total_training_steps,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            clip_grad=self.config.max_grad_norm,
            min_lr_ratio=self.config.min_lr_ratio,
            lr_scheduler_type="cosine",
        )

    def _build_worker(self, model_path: str, *, trainable: bool) -> TrainingWorker:
        checkpoint_config = CheckpointConfig(
            save_contents=["model", "optimizer", "extra"],
            load_contents=["model", "optimizer", "extra"],
        )
        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self._build_model_config(model_path, trainable=trainable),
            engine_config=self._build_engine_config(model_path, forward_only=not trainable),
            optimizer_config=self._build_optimizer_config(),
            checkpoint_config=checkpoint_config,
        )
        worker = TrainingWorker(config=worker_config)
        worker.reset()
        return worker

    def _build_student_worker(self) -> TrainingWorker:
        return self._build_worker(self.config.student_model_path, trainable=True)

    def _build_teacher_worker(self) -> TrainingWorker:
        teacher_model_path = self.config.teacher_model_path or self.config.student_model_path
        return self._build_worker(teacher_model_path, trainable=False)

    def _needs_logits(self) -> bool:
        """Only full_vocab kl_method needs the full logits tensor; MC (including JSD MC) is logprob-only."""
        return self.config.kl_method == "full_vocab"

    def _common_meta(self, *, return_logits: bool) -> dict[str, Any]:
        return {
            "pad_mode": DatasetPadMode.NO_PADDING,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_remove_padding": self.config.use_remove_padding,
            "use_dynamic_bsz": True,
            "max_token_len_per_gpu": self.config.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": None,
            "calculate_entropy": False,
            "return_logits": return_logits,
        }

    def _make_teacher_batch(self, batch: dict[str, torch.Tensor]) -> TensorDict:
        batch_size = batch["teacher_input_ids"].size(0)
        return tu.get_tensordict(
            tensor_dict={
                "input_ids": batch["teacher_input_ids"],
                "position_ids": batch["teacher_position_ids"],
                "loss_mask": batch["teacher_loss_mask"],
                "temperature": torch.full((batch_size,), self.config.temperature, dtype=torch.float32),
            },
            non_tensor_dict=self._common_meta(return_logits=self._needs_logits()),
        )

    def _make_student_batch(self, batch: dict[str, torch.Tensor], teacher_model_output: dict[str, torch.Tensor]) -> TensorDict:
        batch_size = batch["student_input_ids"].size(0)
        tensor_dict = {
            "input_ids": batch["student_input_ids"],
            "position_ids": batch["student_position_ids"],
            "loss_mask": batch["student_loss_mask"],
            "temperature": torch.full((batch_size,), self.config.temperature, dtype=torch.float32),
            "teacher_log_probs": teacher_model_output["log_probs"],
            "teacher_loss_mask": batch["teacher_loss_mask"],
        }
        if self.config.kl_method == "full_vocab":
            tensor_dict["teacher_logits"] = teacher_model_output["logits"]

        meta = self._common_meta(return_logits=self._needs_logits())
        meta["grad_accum_steps"] = self.config.gradient_accumulation_steps
        return tu.get_tensordict(tensor_dict=tensor_dict, non_tensor_dict=meta)

    @staticmethod
    def _masked_nested_to_padded(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_rows = list(tensor.unbind())
        mask_rows = list(mask.unbind())
        trailing_shape = tensor_rows[0].shape[1:]

        selected_rows = []
        max_len = 0
        for tensor_row, mask_row in zip(tensor_rows, mask_rows, strict=True):
            selected = tensor_row[mask_row.to(torch.bool)]
            selected_rows.append(selected)
            max_len = max(max_len, selected.shape[0])

        # If no row has any response tokens, fall back to a 1-token zero
        # padded tensor with an all-zero mask. Raising here would only kill the
        # offending rank and deadlock the rest of the FSDP collective; instead
        # the caller produces a zero loss that still flows through autograd so
        # every rank stays in lockstep.
        empty = max_len == 0
        if empty:
            max_len = 1

        padded = tensor_rows[0].new_zeros((len(selected_rows), max_len, *trailing_shape))
        padded_mask = torch.zeros((len(selected_rows), max_len), dtype=torch.float32, device=padded.device)

        for idx, selected in enumerate(selected_rows):
            if selected.shape[0] == 0:
                continue
            padded[idx, : selected.shape[0]] = selected
            padded_mask[idx, : selected.shape[0]] = 1.0

        if padded.dim() == 3 and not trailing_shape:
            padded = padded.squeeze(-1)

        return padded, padded_mask

    @staticmethod
    def _all_reduce_in_place(tensors: list[torch.Tensor], dp_group):
        if dp_group is None or dist.get_world_size(group=dp_group) == 1:
            return
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_group)

    def _compute_kl_loss(self, model_output: dict[str, torch.Tensor], data: TensorDict, dp_group=None):
        student_log_probs, mask = self._masked_nested_to_padded(model_output["log_probs"], data["loss_mask"])
        teacher_log_probs, teacher_mask = self._masked_nested_to_padded(data["teacher_log_probs"], data["teacher_loss_mask"])

        # When this rank's micro-batch has no valid response tokens we still
        # have to participate in every FSDP collective, otherwise the other
        # ranks deadlock in backward. Return a zero loss that keeps the
        # autograd link to model_output so backward fires reduce-scatter for
        # all params, with zero gradient contribution.
        if mask.sum() == 0:
            zero_loss = model_output["log_probs"].values().sum() * 0.0
            metrics = {
                "kl_num": 0.0,
                "student_nll_num": 0.0,
                "teacher_nll_num": 0.0,
                "response_tokens": 0.0,
            }
            return zero_loss, metrics

        student_counts = mask.sum(dim=1)
        teacher_counts = teacher_mask.sum(dim=1)
        if not torch.equal(student_counts, teacher_counts):
            raise ValueError("Student and teacher response lengths diverged inside the KL loss.")


        if self._needs_logits():
            from .kl_utils import (
                _forward_kl_chunk,
                _forward_kl_topk_chunk,
                _generalized_jsd_chunk,
                _reverse_kl_chunk,
                _reverse_kl_topk_chunk,
            )
            top_k = int(getattr(self.config, "top_k", 0) or 0)
            if top_k > 0 and self.config.kl_type == "jsd":
                raise ValueError(
                    "top_k > 0 is only supported with kl_type in {forward, reverse}; "
                    "JSD requires the full vocabulary for the mixture distribution."
                )
            # Process logits one sample at a time, freeing each row's logits
            # immediately after use to keep peak memory low on A100-40GB.
            student_logits_nested = model_output["logits"]
            teacher_logits_nested = data["teacher_logits"]
            student_rows = list(student_logits_nested.unbind())
            teacher_rows = list(teacher_logits_nested.unbind())
            student_mask_rows = list(data["loss_mask"].unbind())
            teacher_mask_rows = list(data["teacher_loss_mask"].unbind())
            # Release the original nested tensors so their backing storage
            # can be freed as we consume individual rows below.
            del student_logits_nested, teacher_logits_nested
            if "logits" in model_output:
                del model_output["logits"]
            if "teacher_logits" in data.keys():
                data.pop("teacher_logits")

            kl_chunks = []
            chunk_size = int(os.environ.get("KL_FULL_VOCAB_CHUNK_SIZE") or "512")
            if chunk_size < 1:
                raise ValueError(f"KL_FULL_VOCAB_CHUNK_SIZE must be positive, got {chunk_size}")
            T = self.config.temperature

            for i, (s_logits, t_logits, s_mask, t_mask) in enumerate(zip(
                student_rows, teacher_rows, student_mask_rows, teacher_mask_rows, strict=True
            )):
                s_idx = s_mask.to(torch.bool).nonzero(as_tuple=False).flatten()
                t_idx = t_mask.to(torch.bool).nonzero(as_tuple=False).flatten()
                common = min(s_idx.shape[0], t_idx.shape[0])
                if common == 0:
                    kl_chunks.append(s_logits.new_zeros(1))
                    student_rows[i] = None
                    teacher_rows[i] = None
                    del s_logits, t_logits, s_idx, t_idx
                    continue
                s_idx = s_idx[:common]
                t_idx = t_idx[:common]


                sample_kl = []
                for start in range(0, common, chunk_size):
                    end = min(start + chunk_size, common)
                    s_slice = s_logits.index_select(0, s_idx[start:end])
                    t_slice = t_logits.index_select(0, t_idx[start:end])
                    if T != 1.0:
                        s_slice = s_slice / T
                        t_slice = t_slice / T
                    if top_k > 0:
                        # Top-K teacher local support matching (paper Eq. 7-8).
                        if self.config.kl_type == "reverse":
                            sample_kl.append(_reverse_kl_topk_chunk(s_slice, t_slice, top_k))
                        else:  # forward
                            sample_kl.append(_forward_kl_topk_chunk(t_slice, s_slice, top_k))
                    elif self.config.kl_type == "reverse":
                        sample_kl.append(_reverse_kl_chunk(s_slice, t_slice))
                    elif self.config.kl_type == "forward":
                        sample_kl.append(_forward_kl_chunk(t_slice, s_slice))
                    else:  # jsd
                        sample_kl.append(
                            _generalized_jsd_chunk(s_slice, t_slice, beta=self.config.beta)
                        )
                    del s_slice, t_slice
                kl_chunks.append(torch.cat(sample_kl))
                # Free this row's full logits immediately after all chunks finish.
                student_rows[i] = None
                teacher_rows[i] = None
                del s_logits, t_logits, s_idx, t_idx, sample_kl

            del student_rows, teacher_rows

            max_len = max(c.shape[0] for c in kl_chunks)
            if max_len == 0:
                max_len = 1
            kl_per_position = mask.new_zeros(len(kl_chunks), max_len)
            for idx, c in enumerate(kl_chunks):
                kl_per_position[idx, :c.shape[0]] = c
            mask = mask[:, :max_len]
            student_log_probs = student_log_probs[:, :max_len]
            teacher_log_probs = teacher_log_probs[:, :max_len]
        else:
            kl_per_position = compute_kl_divergence(
                student_log_probs,
                teacher_log_probs,
                mask,
                kl_type=self.config.kl_type,
                kl_method="monte_carlo",
                is_logprobs=True,
                reduction="none",
                beta=self.config.beta,
            )

        # Collect pre-clip per-token KL statistics for tuning kl_token_clip.
        # Quantiles are computed on the local DP rank and averaged across
        # micro-batches in _summarize_step_output; clip_num is additive
        # (summable across micro-batches) and becomes clip_frac after
        # dividing by response_tokens.
        with torch.no_grad():
            valid_kl = kl_per_position[mask.to(torch.bool)].float()
            if valid_kl.numel() > 0:
                kl_p50 = valid_kl.quantile(0.5)
                kl_p95 = valid_kl.quantile(0.95)
                kl_p99 = valid_kl.quantile(0.99)
                kl_max = valid_kl.max()
            else:
                kl_p50 = kl_p95 = kl_p99 = kl_max = torch.zeros((), device=kl_per_position.device)
            clip_threshold = self.config.kl_token_clip or float("inf")
            clip_num = (valid_kl >= clip_threshold).sum() if valid_kl.numel() > 0 else torch.zeros(
                (), dtype=torch.long, device=kl_per_position.device
            )

        if self.config.kl_token_clip and self.config.kl_token_clip > 0:
            kl_per_position = kl_per_position.clamp(max=self.config.kl_token_clip)
        kl_num = (kl_per_position * mask).sum()
        student_nll_num = (-student_log_probs * mask).sum()
        teacher_nll_num = (-teacher_log_probs * mask).sum()
        response_tokens = mask.sum()

        batch_num_tokens = float(data["batch_num_tokens"])
        dp_size = float(data["dp_size"])
        kl_loss = kl_num / max(batch_num_tokens, 1.0) * dp_size
        loss = kl_loss / float(tu.get_non_tensor_data(data, "grad_accum_steps", 1))

        # T1 (per-trajectory mean): unbiased to within-batch length skew —
        # for each row, divide its KL sum by its valid-token count, then
        # average across rows. Length-zero rows are skipped.
        with torch.no_grad():
            kl_row_sum = (kl_per_position * mask).sum(dim=1)
            row_tok = mask.sum(dim=1).clamp(min=1)
            row_means = kl_row_sum / row_tok
            valid_rows = (mask.sum(dim=1) > 0).float()
            kl_per_traj_num = (row_means * valid_rows).sum()
            kl_per_traj_den = valid_rows.sum()

        # Metrics are reported as local-rank values (no per-micro-batch NCCL
        # all_reduce). With dynamic batching each rank sees ~similar token
        # counts, so the logging rank's ratios (kl_loss = kl_num /
        # response_tokens, clip_frac = clip_num / response_tokens) are within
        # ~1% of the global value. Removing the all_reduce eliminates one NCCL
        # collective per micro-batch, which dominated GPU idle time on the
        # full_vocab path.
        metrics = {
            "kl_num": kl_num.detach().float().item(),
            "student_nll_num": student_nll_num.detach().float().item(),
            "teacher_nll_num": teacher_nll_num.detach().float().item(),
            "response_tokens": response_tokens.detach().float().item(),
            "clip_num": clip_num.detach().float().item(),
            "kl_p50": kl_p50.float().item(),
            "kl_p95": kl_p95.float().item(),
            "kl_p99": kl_p99.float().item(),
            "kl_max": kl_max.float().item(),
            # T1 (per-trajectory variant). _summarize_step_output divides
            # the summed numerator by the summed denominator across micro-batches.
            "kl_per_traj_num": float(kl_per_traj_num.item()),
            "kl_per_traj_den": float(kl_per_traj_den.item()),
        }
        return loss, metrics

    @staticmethod
    def _summarize_step_output(output: dict[str, Any]) -> dict[str, float]:
        metrics = output.get("metrics", {})
        kl_num = float(sum(metrics.get("kl_num", [])))
        student_nll_num = float(sum(metrics.get("student_nll_num", [])))
        teacher_nll_num = float(sum(metrics.get("teacher_nll_num", [])))
        response_tokens = float(sum(metrics.get("response_tokens", [])))
        response_tokens = max(response_tokens, 1.0)
        clip_num = float(sum(metrics.get("clip_num", [])))

        def _mean(key: str) -> float:
            values = metrics.get(key, [])
            return float(sum(values) / len(values)) if values else 0.0

        # T1-traj: weighted by trajectory count so the per-row arithmetic mean
        # is preserved across micro-batches.
        kl_per_traj_num = float(sum(metrics.get("kl_per_traj_num", [])))
        kl_per_traj_den = float(sum(metrics.get("kl_per_traj_den", [])))


        return {
            "kl_loss": kl_num / response_tokens,
            "student_perplexity": math.exp(student_nll_num / response_tokens),
            "teacher_perplexity": math.exp(teacher_nll_num / response_tokens),
            "response_tokens": response_tokens,
            "clip_frac": clip_num / response_tokens,
            "kl_p50": _mean("kl_p50"),
            "kl_p95": _mean("kl_p95"),
            "kl_p99": _mean("kl_p99"),
            "kl_max": max(metrics.get("kl_max", [0.0])) if metrics.get("kl_max") else 0.0,
            # T1-traj: arithmetic mean of per-trajectory token-mean KLs.
            "kl_loss_per_traj": (kl_per_traj_num / kl_per_traj_den) if kl_per_traj_den > 0 else 0.0,
        }

    def _save_checkpoint(self):
        ckpt_dir = os.path.join(self.config.model_save_dir, f"global_step_{self.global_step}")
        max_keep = self.config.max_ckpt_to_keep if self.config.max_ckpt_to_keep > 0 else None
        self.student_engine.save_checkpoint(
            local_path=ckpt_dir,
            global_step=self.global_step,
            max_ckpt_to_keep=max_keep,
        )
        if self.config.use_lora and self.rank == 0:
            lora_meta = {
                "r": int(self.config.lora_rank),
                "lora_alpha": int(self.config.lora_alpha),
                "task_type": "CAUSAL_LM",
            }
            lora_meta_path = os.path.join(ckpt_dir, "lora_train_meta.json")
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(lora_meta_path, "w", encoding="utf-8") as f:
                json.dump(lora_meta, f, ensure_ascii=False, indent=4)
            logger.info("Saved LoRA rank/alpha metadata to %s", lora_meta_path)
        logger.info("Checkpoint saved to %s", ckpt_dir)

    def _find_latest_checkpoint(self) -> str:
        ckpt_dirs = glob.glob(os.path.join(self.config.model_save_dir, "global_step_*"))
        if not ckpt_dirs:
            raise FileNotFoundError(f"No FSDP checkpoints found under {self.config.model_save_dir}")
        return max(ckpt_dirs, key=lambda path: int(path.rsplit("_", 1)[-1]))

    def _checkpoint_step_from_path(self, path: str) -> int:
        try:
            return int(path.rstrip("/").rsplit("_", 1)[-1])
        except ValueError:
            return 0

    def _maybe_resume_from_checkpoint(self) -> int:
        """Load an FSDP checkpoint when requested or present.

        ``continue`` preserves the old same-run resume behavior and skips already
        completed optimizer steps. ``initialize`` loads the checkpoint state as the
        starting point for this batch, then resets ``self.global_step`` so this
        short run still performs its optimizer step.
        """
        explicit = (self.config.resume_checkpoint_path or "").strip()
        if explicit:
            latest = explicit
            if not os.path.isdir(latest):
                raise FileNotFoundError(f"resume_checkpoint_path does not exist or is not a directory: {latest}")
        else:
            ckpt_dirs = glob.glob(os.path.join(self.config.model_save_dir, "global_step_*"))
            if not ckpt_dirs:
                if self.rank == 0:
                    print(
                        f"[RESUME] No checkpoint found under {self.config.model_save_dir} -- "
                        f"training from scratch (global_step=0)",
                        flush=True,
                    )
                return 0
            latest = max(ckpt_dirs, key=self._checkpoint_step_from_path)

        step = self._checkpoint_step_from_path(latest)
        mode = self.config.resume_checkpoint_mode
        if self.rank == 0:
            print(
                "\n" + "=" * 70 + "\n"
                f"[RESUME] Loading FSDP checkpoint\n"
                f"[RESUME]   path:        {latest}\n"
                f"[RESUME]   global_step: {step}\n"
                f"[RESUME]   mode:        {mode}\n"
                f"[RESUME]   world_size:  {self.world_size}\n"
                + "=" * 70,
                flush=True,
            )
        logger.info("Loading checkpoint %s (global_step=%s, mode=%s)", latest, step, mode)
        self.student_engine.load_checkpoint(local_path=latest, del_local_after_load=False)
        self.global_step = step if mode == "continue" else 0
        if self.rank == 0:
            print(f"[RESUME] Checkpoint loaded -- trainer global_step={self.global_step}\n", flush=True)
        return self.global_step

    def _broadcast_rank0_error(self, error_message: str | None) -> str | None:
        error_holder = [error_message]
        dist.broadcast_object_list(error_holder, src=0)
        return error_holder[0]

    def _export_hf_model(self):
        latest_ckpt = self._find_latest_checkpoint()
        merged_dir = os.path.join(self.config.model_save_dir, "hf_merged")
        hf_assets_source = self.config.student_model_path
        rank0_exception = None
        rank0_error = None

        if self.rank == 0:
            try:
                os.makedirs(merged_dir, exist_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    latest_ckpt,
                    "--hf_model_config_path",
                    hf_assets_source,
                    "--target_dir",
                    merged_dir,
                ]
                if hf_assets_source:
                    logger.info("Using HF assets from %s for merged export", hf_assets_source)
                cmd.append("--trust-remote-code")
                logger.info("Merging FSDP checkpoint %s -> %s", latest_ckpt, merged_dir)
                subprocess.run(cmd, check=True)
            except Exception as e:
                rank0_exception = e
                rank0_error = f"{type(e).__name__}: {e}"
                logger.exception("Failed to export checkpoint %s to %s", latest_ckpt, merged_dir)

        exported_error = self._broadcast_rank0_error(rank0_error)
        if exported_error is not None:
            raise RuntimeError(f"FSDP checkpoint export failed on rank 0: {exported_error}") from rank0_exception

    def _launch_hf_export_async(self):
        latest_ckpt = self._find_latest_checkpoint()
        merged_dir = os.path.join(self.config.model_save_dir, "hf_merged")
        hf_assets_source = self.config.student_model_path
        rank0_error = None

        if self.rank == 0:
            try:
                os.makedirs(merged_dir, exist_ok=True)
                log_path = os.path.join(self.config.model_save_dir, "hf_export_async.log")
                cmd = [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    latest_ckpt,
                    "--hf_model_config_path",
                    hf_assets_source,
                    "--target_dir",
                    merged_dir,
                    "--trust-remote-code",
                ]
                logger.info("Launching async HF export %s -> %s (log=%s)", latest_ckpt, merged_dir, log_path)
                log_f = open(log_path, "ab", buffering=0)
                subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except Exception as e:
                rank0_error = f"{type(e).__name__}: {e}"
                logger.exception("Failed to launch async export checkpoint %s to %s", latest_ckpt, merged_dir)

        exported_error = self._broadcast_rank0_error(rank0_error)
        if exported_error is not None:
            raise RuntimeError(f"Async FSDP checkpoint export launch failed on rank 0: {exported_error}")

    def train_epoch(self, skip_batches: int = 0) -> dict[str, float]:
        sampler = getattr(self.train_dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.epoch)

        running_kl = 0.0
        running_student_ppl = 0.0
        running_teacher_ppl = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}/{self.config.total_epochs}",
            disable=not self.is_logging,
        )

        self.student_engine.optimizer_zero_grad()
        grad_norm = 0.0
        lr = self.config.learning_rate

        with self.student_engine.train_mode():
            for batch_idx, batch in enumerate(progress_bar):
                if batch_idx < skip_batches:
                    continue
                teacher_batch = self._make_teacher_batch(batch)
                with self.teacher_engine.eval_mode():
                    teacher_output = self.teacher_engine.forward_backward_batch(
                        teacher_batch,
                        loss_function=None,
                        forward_only=True,
                    )

                student_batch = self._make_student_batch(batch, teacher_output["model_output"])

                start_time = time.time()
                train_output = self.student_engine.forward_backward_batch(
                    student_batch,
                    loss_function=self._compute_kl_loss,
                    forward_only=False,
                )
                step_time = time.time() - start_time

                step_metrics = self._summarize_step_output(train_output)
                running_kl += step_metrics["kl_loss"]
                running_student_ppl += step_metrics["student_perplexity"]
                running_teacher_ppl += step_metrics["teacher_perplexity"]
                num_batches += 1

                should_step = (
                    (batch_idx + 1) % self.config.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(self.train_dataloader)
                )

                if should_step:
                    grad_norm = self.student_engine.optimizer_step()
                    self.student_engine.optimizer_zero_grad()
                    lr = self.student_engine.lr_scheduler_step()
                    self.global_step += 1

                    if self.is_logging and HAS_WANDB:
                        log_payload = {
                            "train/kl_loss": step_metrics["kl_loss"],
                            "train/kl_loss_per_traj": step_metrics["kl_loss_per_traj"],
                            "train/student_perplexity": step_metrics["student_perplexity"],
                            "train/teacher_perplexity": step_metrics["teacher_perplexity"],
                            "train/response_tokens": step_metrics["response_tokens"],
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": lr,
                            "train/step_time_sec": step_time,
                            "train/epoch": self.epoch,
                            "train/global_step": self.global_step,
                            "train/kl_p50": step_metrics["kl_p50"],
                            "train/kl_p95": step_metrics["kl_p95"],
                            "train/kl_p99": step_metrics["kl_p99"],
                            "train/kl_max": step_metrics["kl_max"],
                            "train/clip_frac": step_metrics["clip_frac"],
                            "train/kl_token_clip": self.config.kl_token_clip,
                        }
                        wandb.log(log_payload)

                    if self.global_step % self.config.logging_steps == 0 and self.is_logging:
                        logger.info(
                            "step=%s kl=%.4f student_ppl=%.2f teacher_ppl=%.2f grad_norm=%.4f lr=%.2e",
                            self.global_step,
                            step_metrics["kl_loss"],
                            step_metrics["student_perplexity"],
                            step_metrics["teacher_perplexity"],
                            grad_norm,
                            lr,
                        )

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()

                progress_bar.set_postfix(
                    {
                        "kl": f"{running_kl / num_batches:.4f}",
                        "student_ppl": f"{running_student_ppl / num_batches:.2f}",
                        "teacher_ppl": f"{running_teacher_ppl / num_batches:.2f}",
                    }
                )

                del teacher_output
                del train_output

        return {
            "kl_loss": running_kl / max(num_batches, 1),
            "student_perplexity": running_student_ppl / max(num_batches, 1),
            "teacher_perplexity": running_teacher_ppl / max(num_batches, 1),
        }

    def train(self):
        logger.info("Starting KL training")
        logger.info("Total epochs: %s", self.config.total_epochs)
        logger.info("Total optimizer steps: %s", self.total_training_steps)

        resumed_step = self._maybe_resume_from_checkpoint()
        batches_per_epoch = len(self.train_dataloader)
        grad_accum = self.config.gradient_accumulation_steps
        optim_steps_per_epoch = max(1, math.ceil(batches_per_epoch / grad_accum))
        start_epoch = resumed_step // optim_steps_per_epoch
        steps_done_in_epoch = resumed_step % optim_steps_per_epoch
        initial_skip_batches = steps_done_in_epoch * grad_accum
        if resumed_step > 0:
            logger.info(
                "Resume plan: start_epoch=%s, skip_batches=%s (optim_steps_per_epoch=%s)",
                start_epoch,
                initial_skip_batches,
                optim_steps_per_epoch,
            )

        for epoch in range(start_epoch, self.config.total_epochs):
            self.epoch = epoch
            skip_batches = initial_skip_batches if epoch == start_epoch else 0
            epoch_metrics = self.train_epoch(skip_batches=skip_batches)
            if self.is_logging:
                logger.info("Epoch %s metrics: %s", epoch + 1, epoch_metrics)
                if HAS_WANDB:
                    wandb.log(
                        {
                            "epoch/kl_loss": epoch_metrics["kl_loss"],
                            "epoch/student_perplexity": epoch_metrics["student_perplexity"],
                            "epoch/teacher_perplexity": epoch_metrics["teacher_perplexity"],
                            "epoch/index": epoch + 1,
                        }
                    )

        if self.global_step == 0:
            raise RuntimeError("Training finished without any optimizer step.")

        self._save_checkpoint()
        if self.config.save_merged_model or self.config.run_eval_after_training:
            if self.config.async_hf_export and not self.config.run_eval_after_training:
                self._launch_hf_export_async()
            else:
                self._export_hf_model()

        # Evaluation is launched from the shell driver after this process exits,
        # so all FSDP/training state is released before vLLM claims the GPUs.

    def run_evaluation(self):
        if not self.is_logging:
            return

        eval_model_path = os.path.join(self.config.model_save_dir, "hf_merged")
        if not os.path.isdir(eval_model_path):
            raise FileNotFoundError(
                f"Evaluation requires a merged HuggingFace checkpoint at {eval_model_path}."
            )

        eval_datasets = self.config.eval_datasets or ["aime24", "aime25", "math500"]
        dataset_paths = self.config.eval_dataset_paths

        base_model_name = self.config.base_model_name or self.config.student_model_path.split("/")[-1]
        eval_tag = build_eval_tag(self.config)
        model_name = build_eval_model_name(self.config)

        gen_results_dir = self.config.gen_results_dir or self.config.output_dir
        eval_output_dir = os.path.join(gen_results_dir, f"evaluate_{eval_tag}")
        os.makedirs(eval_output_dir, exist_ok=True)

        verl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        results_base_dir = os.path.join(verl_root, "results", base_model_name)
        results_file = os.path.join(results_base_dir, "results.json")

        selected_dataset_paths = {dataset: dataset_paths.get(dataset) for dataset in eval_datasets}
        eval_results = run_evaluation_suite(
            eval_model_path,
            selected_dataset_paths,
            eval_output_dir,
            pass_k=1,
            temperature=0.6,
            top_p=0.95,
            nnodes=self.config.nnodes,
            n_gpus_per_node=self.config.n_gpus_per_node,
            tensor_model_parallel_size=1,
            output_json_path=results_file,
            model_name=model_name,
        )

        for dataset, accuracy in eval_results.items():
            logger.info("Evaluation %s: %.2f%%", dataset, accuracy * 100)

        if not eval_results:
            raise RuntimeError(
                "Automatic evaluation was requested, but no valid evaluation datasets were found. "
                f"Checked datasets={eval_datasets} under {self.config.eval_datasets_dir}."
            )

        if HAS_WANDB and eval_results:
            wandb.log({f"eval/{dataset}": acc for dataset, acc in eval_results.items()})
