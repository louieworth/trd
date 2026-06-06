"""
KL Divergence Training Module for Math Reasoning

This module implements token-level KL divergence training with 4 variants:
1. Reverse KL + Monte Carlo
2. Reverse KL + Full Vocabulary
3. Forward KL + Monte Carlo
4. Forward KL + Full Vocabulary
"""

from .kl_utils import (
    compute_reverse_kl_monte_carlo,
    compute_reverse_kl_full_vocab,
    compute_forward_kl_monte_carlo,
    compute_forward_kl_full_vocab,
)

__all__ = [
    'compute_reverse_kl_monte_carlo',
    'compute_reverse_kl_full_vocab',
    'compute_forward_kl_monte_carlo',
    'compute_forward_kl_full_vocab',
]
