"""
Diagnostics and Monitoring Utilities for MPS

Provides entropy calculations, statistics tracking, and observability tools.

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import math
from typing import Dict, List

import torch


def bond_entropy_from_S(S: torch.Tensor) -> float:
    """
    Compute entanglement entropy from singular values

    S = -Σ_i p_i log(p_i) where p_i = σ_i² / Σ_j σ_j²

    Args:
        S: Singular values (sorted descending)

    Returns:
        Entanglement entropy in bits (log base 2)
    """
    if len(S) == 0:
        return 0.0

    p = (S * S) / (S * S).sum()
    # Clamp to avoid log(0)
    entropy = -(p * torch.log2(torch.clamp(p, min=1e-30))).sum()
    return float(entropy.item())


def effective_rank(S: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Compute effective rank: smallest k where cumulative energy ≥ threshold

    Args:
        S: Singular values
        threshold: Energy retention threshold (default 99%)

    Returns:
        Effective rank
    """
    if len(S) == 0:
        return 0

    E = (S * S).cumsum(0)
    total = E[-1]
    k = int(torch.searchsorted(E, threshold * total).item()) + 1
    return min(k, len(S))


def spectral_gap(S: torch.Tensor, k: int) -> float:
    """
    Compute spectral gap σ_k / σ_{k+1}

    Large gaps indicate safe truncation points.

    Args:
        S: Singular values
        k: Truncation point

    Returns:
        Spectral gap ratio (or inf if k+1 doesn't exist or is zero)
    """
    if k >= len(S) or k < 1:
        return float("inf")
    if S[k] == 0:
        return float("inf")
    return float((S[k - 1] / S[k]).item())


class MPSStatistics:
    """
    Track and aggregate MPS operation statistics

    Maintains per-operation logs and rolling aggregates for:
    - Bond dimensions (χ)
    - Truncation errors (ε_local)
    - Entanglement entropies (S)
    - SVD driver usage
    - Computation times
    """

    def __init__(self):
        self.logs: Dict[str, List] = {
            "step": [],
            "bond": [],
            "k_star": [],
            "chi_before": [],
            "chi_after": [],
            "eps_local": [],
            "entropy": [],
            "svd_driver": [],
            "dtype": [],
            "ms_elapsed": [],
            "condS": [],
        }

    def record(self, **kwargs):
        """Record a single operation"""
        for key, value in kwargs.items():
            if key in self.logs:
                self.logs[key].append(value)

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics"""
        import numpy as np

        def safe_agg(key: str, fn):
            if not self.logs[key]:
                return 0.0
            try:
                return float(fn(np.array(self.logs[key])))
            except:
                return 0.0

        return {
            "total_operations": len(self.logs["step"]),
            "max_chi": safe_agg("chi_after", np.max),
            "mean_chi": safe_agg("chi_after", np.mean),
            "sum_eps2": safe_agg("eps_local", lambda x: (x**2).sum()),
            "max_eps": safe_agg("eps_local", np.max),
            "mean_entropy": safe_agg("entropy", np.mean),
            "p95_entropy": safe_agg("entropy", lambda x: np.percentile(x, 95)),
            "total_time_ms": safe_agg("ms_elapsed", np.sum),
            "cuda_svd_pct": self._driver_percentage("torch_cuda"),
            "cpu_fallback_pct": self._driver_percentage("torch_cpu"),
        }

    def _driver_percentage(self, driver_name: str) -> float:
        """Compute percentage of operations using specific driver"""
        if not self.logs["svd_driver"]:
            return 0.0
        count = sum(1 for d in self.logs["svd_driver"] if d == driver_name)
        return 100.0 * count / len(self.logs["svd_driver"])

    def global_error_bound(self) -> float:
        """Compute global error upper bound"""
        if not self.logs["eps_local"]:
            return 0.0
        return math.sqrt(sum(e**2 for e in self.logs["eps_local"]))

    def reset(self):
        """Clear all logs"""
        for key in self.logs:
            self.logs[key].clear()
