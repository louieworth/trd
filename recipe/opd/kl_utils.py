#!/usr/bin/env python3
#
# KL Divergence Computation Utilities
# Implements 4 variants of KL divergence for token-level training

import torch
import torch.nn.functional as F
from typing import Literal


def compute_reverse_kl_monte_carlo(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Compute Reverse KL divergence using Monte Carlo estimation.

    Formula: D_KL[P || Q] where P = student, Q = teacher
    KL = E_P[log(P/Q)] = E_P[log P - log Q]

    Monte Carlo surrogate evaluated on student-sampled tokens:
    KL = E_P[Q/P - 1 - log(Q/P)]
       = E_P[exp(log_q - log_p) - 1 - (log_q - log_p)]

    This is the most memory-efficient variant, only computing KL for sampled tokens.

    Args:
        student_logprobs: Log probabilities from student model [batch, seq_len]
        teacher_logprobs: Log probabilities from teacher model [batch, seq_len]
        mask: Attention mask [batch, seq_len]
        reduction: How to reduce the KL values

    Returns:
        KL divergence value(s)

    Reference:
        LMOps/minillm/minillm/utils.py:get_rev_kl()
    """
    log_ratio = (teacher_logprobs - student_logprobs) * mask
    # Symmetric clamp: the k3 estimator is `exp(log_ratio) - 1 - log_ratio`, so
    # the exp term and the linear term must share the same clamped input —
    # otherwise large log_ratio silently biases the estimator.
    log_ratio = log_ratio.clamp(max=20)
    kl = log_ratio.float().exp() - 1 - log_ratio

    if reduction == "mean":
        return (kl * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl * mask).sum()
    else:  # none
        return kl


def compute_forward_kl_monte_carlo(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    Forward KL MC in full form: kl = log q - log p (teacher-sampled).

    teacher_logprobs come from a forward_only teacher pass and are already
    detached from autograd, so keeping the teacher term does not change the
    gradient w.r.t. student params (-d log p_student / dtheta) but gives
    real per-token KL values for logging and per-token clipping.
    """
    kl = (teacher_logprobs - student_logprobs) * mask

    if reduction == "mean":
        return kl.sum() / mask.sum()
    elif reduction == "sum":
        return kl.sum()
    else:  # none
        return kl


def _reverse_kl_chunk(
    student_logits_chunk: torch.Tensor,
    teacher_logits_chunk: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position reverse KL for a chunk. Inputs: [chunk_len, vocab]."""
    student_logprobs = F.log_softmax(student_logits_chunk.float(), dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits_chunk.float(), dim=-1)
    student_probs = student_logprobs.exp()
    return (student_probs * (student_logprobs - teacher_logprobs)).sum(dim=-1)


def compute_reverse_kl_full_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Compute Reverse KL divergence over full vocabulary, chunked along the
    sequence dimension to control peak memory.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (default 1.0)
        chunk_size: Number of positions to process at once
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    batch_size, seq_len = mask.shape
    kl_per_position = mask.new_zeros(batch_size, seq_len)

    for b in range(batch_size):
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            kl_per_position[b, start:end] = _reverse_kl_chunk(
                student_logits[b, start:end],
                teacher_logits[b, start:end],
            )

    if reduction == "mean":
        return (kl_per_position * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl_per_position * mask).sum()
    else:
        return kl_per_position


def _forward_kl_chunk(
    teacher_logits_chunk: torch.Tensor,
    student_logits_chunk: torch.Tensor,
) -> torch.Tensor:
    """Compute per-position forward KL for a chunk. Inputs: [chunk_len, vocab]."""
    with torch.no_grad():
        teacher_logits = teacher_logits_chunk.float()
        teacher_log_z = torch.logsumexp(teacher_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_term = torch.einsum("lv,lv->l", teacher_probs, teacher_logits) - teacher_log_z
        del teacher_logits, teacher_log_z

    student_logits = student_logits_chunk.float()
    student_log_z = torch.logsumexp(student_logits, dim=-1)
    student_term = torch.einsum("lv,lv->l", teacher_probs, student_logits) - student_log_z
    return teacher_term - student_term


def _forward_kl_topk_chunk(
    teacher_logits_chunk: torch.Tensor,
    student_logits_chunk: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Forward KL on teacher's top-K local support.

    Paper: Fu et al. 2026, "Revisiting On-Policy Distillation". Eq. (6)–(8).
    The support set S(c) = TopK_q(c) is chosen by teacher logits at every
    position; both teacher and student are renormalized via a separate
    softmax over the top-K logits before the KL is taken — gradients flow
    only through the K selected logits on the student side.

    Forward direction (Σ_v p̂_t (log p̂_t − log p̂_s)) is the user-facing variant
    used here; the paper's Eq. (8) is reverse — see ``_reverse_kl_topk_chunk``.

    Inputs: ``[chunk_len, vocab]``. Returns ``[chunk_len]`` per-position KL.
    """
    _, V = teacher_logits_chunk.shape
    k = min(int(k), V)
    teacher = teacher_logits_chunk.float()
    student = student_logits_chunk.float()
    _, topk_idx = teacher.topk(k, dim=-1)              # [L, K]
    t_top = teacher.gather(-1, topk_idx)               # [L, K]
    s_top = student.gather(-1, topk_idx)               # [L, K] — gradient kept
    t_logp = F.log_softmax(t_top, dim=-1)              # renormalized teacher
    s_logp = F.log_softmax(s_top, dim=-1)              # renormalized student
    t_p = t_logp.exp()
    return (t_p * (t_logp - s_logp)).sum(dim=-1)


def _reverse_kl_topk_chunk(
    student_logits_chunk: torch.Tensor,
    teacher_logits_chunk: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Reverse KL on teacher's top-K local support — paper Eq. (8) verbatim.

    Same renormalization as :func:`_forward_kl_topk_chunk`, but the KL is
    student-leading: ``Σ_v π̂(v) (log π̂(v) − log q̂(v))``.

    Inputs: ``[chunk_len, vocab]``. Returns ``[chunk_len]`` per-position KL.
    """
    _, V = teacher_logits_chunk.shape
    k = min(int(k), V)
    teacher = teacher_logits_chunk.float()
    student = student_logits_chunk.float()
    _, topk_idx = teacher.topk(k, dim=-1)
    t_top = teacher.gather(-1, topk_idx)
    s_top = student.gather(-1, topk_idx)
    s_logp = F.log_softmax(s_top, dim=-1)
    t_logp = F.log_softmax(t_top, dim=-1)
    s_p = s_logp.exp()
    return (s_p * (s_logp - t_logp)).sum(dim=-1)


def compute_generalized_jsd_monte_carlo(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    beta: float = 0.0,
) -> torch.Tensor:
    """Generalized JSD via teacher-sampled Monte Carlo (offline SFT setup).

    Math: for y ~ teacher (our y_o/y_r data), we have `log q(y)` and
    `log p(y)`. The mixture at the sampled token is
        log M(y) = logsumexp([log p + log(1-β), log q + log(β)]).

    We combine
      * unbiased MC for β·KL(teacher || M):   β · (log q - log M)
      * k3 estimator for (1-β)·KL(student || M) using teacher samples:
          (1-β) · (exp(log_M - log_p) - 1 - (log_M - log_p))
        This is the same k3 form as our reverse-KL MC (see
        `compute_reverse_kl_monte_carlo`): non-negative, biased when the
        sampling distribution is not student, but smooth across β.

    Endpoint handling (matches OPSD's special cases at β=0/1 — the naive
    formula would give 0 there because the mixture collapses onto one of the
    source distributions):
      * β=0 → forward KL MC (KL(teacher || student))
      * β=1 → reverse KL MC (KL(student || teacher))

    Args:
        teacher_logprobs: log q(y_t) at sampled tokens [batch, seq_len]
        student_logprobs: log p(y_t) at sampled tokens [batch, seq_len]
        mask:             response-token mask [batch, seq_len]
        reduction:        "mean" | "sum" | "none"
        beta:             mixture coefficient in [0, 1]
    """
    import math

    if beta == 0:
        return compute_forward_kl_monte_carlo(teacher_logprobs, student_logprobs, mask, reduction)
    if beta == 1:
        return compute_reverse_kl_monte_carlo(student_logprobs, teacher_logprobs, mask, reduction)

    # log M(y_t) = logsumexp([log p + log(1-β), log q + log(β)])
    log_one_minus_beta = math.log(1.0 - beta)
    log_beta = math.log(beta)
    log_mix = torch.logsumexp(
        torch.stack([student_logprobs + log_one_minus_beta, teacher_logprobs + log_beta]),
        dim=0,
    )

    # Term 1: β·(log q - log M), unbiased for β·KL(teacher || M)
    term1 = beta * (teacher_logprobs - log_mix)

    # Term 2: (1-β)·k3(log M - log p), k3-style biased MC for (1-β)·KL(student || M)
    log_ratio = (log_mix - student_logprobs).clamp(max=20)
    term2 = (1.0 - beta) * (log_ratio.float().exp() - 1.0 - log_ratio)

    jsd = (term1 + term2) * mask

    if reduction == "mean":
        return (jsd * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (jsd * mask).sum()
    else:
        return jsd


def _generalized_jsd_chunk(
    student_logits_chunk: torch.Tensor,
    teacher_logits_chunk: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Per-position generalized JSD for a chunk. Inputs: [chunk_len, vocab].

    Faithful port of OPSD's `generalized_jsd_loss`
    (https://github.com/siyan-zhao/OPSD/blob/main/opsd_trainer.py#L376).

    beta=0 → forward KL(teacher || student)
    beta=1 → reverse KL(student || teacher)
    beta∈(0,1) → Generalized JSD: beta·KL(teacher||M) + (1-beta)·KL(student||M)
                 where M = (1-beta)·student + beta·teacher.
    Temperature is assumed to already be applied to logits by the caller.
    """
    student_log_probs = F.log_softmax(student_logits_chunk.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits_chunk.float(), dim=-1)

    if beta == 0:
        # forward KL(teacher || student) = sum_v P_t (log P_t - log P_s)
        return F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)
    if beta == 1:
        # reverse KL(student || teacher)
        return F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        ).sum(dim=-1)

    # Mixture M = (1-beta)·student + beta·teacher in log-space
    beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]
        ),
        dim=0,
    )
    kl_teacher = F.kl_div(
        mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)
    kl_student = F.kl_div(
        mixture_log_probs, student_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)
    return beta * kl_teacher + (1 - beta) * kl_student


def compute_generalized_jsd_full_vocab(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
    beta: float = 0.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Generalized JSD over full vocabulary, chunked along the sequence dim.

    Matches OPSD's `generalized_jsd_loss` signature, minus per-token clipping
    (clipping is applied by `kl_trainer.py` after this function returns, so
    it stays consistent with the other KL variants).
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    batch_size, seq_len = mask.shape
    kl_per_position = mask.new_zeros(batch_size, seq_len)
    for b in range(batch_size):
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            kl_per_position[b, start:end] = _generalized_jsd_chunk(
                student_logits[b, start:end],
                teacher_logits[b, start:end],
                beta=beta,
            )

    if reduction == "mean":
        return (kl_per_position * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl_per_position * mask).sum()
    else:
        return kl_per_position


def compute_forward_kl_full_vocab(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    mask: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Compute Forward KL divergence over full vocabulary, chunked along the
    sequence dimension to avoid materializing the entire [seq_len, vocab_size]
    fp32 tensor at once (which OOMs on A100-40GB for long sequences).

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len]
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (default 1.0)
        chunk_size: Number of positions to process at once (controls peak memory)
    """
    if temperature != 1.0:
        teacher_logits = teacher_logits / temperature
        student_logits = student_logits / temperature

    batch_size, seq_len = mask.shape
    kl_per_position = mask.new_zeros(batch_size, seq_len)

    for b in range(batch_size):
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            kl_per_position[b, start:end] = _forward_kl_chunk(
                teacher_logits[b, start:end],
                student_logits[b, start:end],
            )

    if reduction == "mean":
        return (kl_per_position * mask).sum() / mask.sum()
    elif reduction == "sum":
        return (kl_per_position * mask).sum()
    else:
        return kl_per_position


def compute_kl_divergence(
    student_logits_or_logprobs: torch.Tensor,
    teacher_logits_or_logprobs: torch.Tensor,
    mask: torch.Tensor,
    kl_type: Literal["reverse", "forward", "jsd"] = "reverse",
    kl_method: Literal["monte_carlo", "full_vocab"] = "monte_carlo",
    is_logprobs: bool = False,
    reduction: Literal["mean", "sum", "none"] = "mean",
    temperature: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Unified interface for computing KL divergence.

    Args:
        student_logits_or_logprobs: Student model outputs
        teacher_logits_or_logprobs: Teacher model outputs
        mask: Attention mask
        kl_type: "reverse", "forward", or "jsd" (generalized Jensen-Shannon, OPSD-style)
        kl_method: "monte_carlo" or "full_vocab" (jsd requires full_vocab)
        is_logprobs: Whether inputs are log probabilities (True) or logits (False)
        reduction: How to reduce the KL values
        temperature: Temperature for softmax (only used for full_vocab)
        beta: Mixture coefficient for generalized JSD (ignored unless kl_type="jsd")

    Returns:
        KL divergence value(s)
    """
    if kl_type == "jsd":
        if kl_method == "monte_carlo":
            if not is_logprobs:
                raise ValueError("Monte Carlo JSD requires log probabilities, not logits")
            return compute_generalized_jsd_monte_carlo(
                teacher_logits_or_logprobs,
                student_logits_or_logprobs,
                mask,
                reduction,
                beta,
            )
        # full_vocab
        if is_logprobs:
            raise ValueError("JSD full_vocab requires logits, not log probabilities")
        return compute_generalized_jsd_full_vocab(
            student_logits_or_logprobs,
            teacher_logits_or_logprobs,
            mask,
            reduction,
            temperature,
            beta,
        )

    if kl_method == "monte_carlo":
        if not is_logprobs:
            raise ValueError("Monte Carlo KL requires log probabilities, not logits")

        if kl_type == "reverse":
            return compute_reverse_kl_monte_carlo(
                student_logits_or_logprobs,
                teacher_logits_or_logprobs,
                mask,
                reduction,
            )
        else:  # forward
            return compute_forward_kl_monte_carlo(
                teacher_logits_or_logprobs,
                student_logits_or_logprobs,
                mask,
                reduction,
            )

    else:  # full_vocab
        if is_logprobs:
            raise ValueError("Full vocab KL requires logits, not log probabilities")

        if kl_type == "reverse":
            return compute_reverse_kl_full_vocab(
                student_logits_or_logprobs,
                teacher_logits_or_logprobs,
                mask,
                reduction,
                temperature,
            )
        else:  # forward
            return compute_forward_kl_full_vocab(
                teacher_logits_or_logprobs,
                student_logits_or_logprobs,
                mask,
                reduction,
                temperature,
            )
