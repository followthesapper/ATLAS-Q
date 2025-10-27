"""
Robust Linear Algebra Operations with Fallback Cascade

Provides GPU-first SVD with automatic fallback to CPU and jitter-based
recovery for numerically unstable matrices.

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from typing import Tuple

import torch


def robust_svd(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Robust SVD with fallback cascade: CUDA → jitter → CPU

    Args:
        X: Input tensor to decompose

    Returns:
        U, S, Vh, driver_used

    Strategy:
    1. Try torch.linalg.svd on GPU (cuSOLVER backend)
    2. If fails, add small jitter and retry on GPU
    3. If still fails, fall back to CPU SVD
    4. Return which driver succeeded for diagnostics
    """
    device = X.device

    # Try 1: Direct CUDA SVD
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return U, S, Vh, "torch_cuda"
    except Exception:
        pass

    # Try 2: Add jitter for numerical stability
    try:
        Xj = X + (1e-12 * torch.randn_like(X))
        U, S, Vh = torch.linalg.svd(Xj, full_matrices=False)
        return U, S, Vh, "torch_cuda_jitter"
    except Exception:
        pass

    # Try 3: CPU fallback (always works, slower)
    U, S, Vh = torch.linalg.svd(X.cpu(), full_matrices=False)
    return U.to(device), S.to(device), Vh.to(device), "torch_cpu"


def robust_qr(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Robust QR decomposition with fallback

    Args:
        X: Input tensor to decompose

    Returns:
        Q, R, driver_used
    """
    device = X.device

    # Try 1: Direct CUDA QR
    try:
        Q, R = torch.linalg.qr(X)
        return Q, R, "torch_cuda"
    except Exception:
        pass

    # Try 2: CPU fallback
    Q, R = torch.linalg.qr(X.cpu())
    return Q.to(device), R.to(device), "torch_cpu"


def condition_number(S: torch.Tensor) -> float:
    """
    Compute condition number from singular values

    Args:
        S: Singular values (sorted descending)

    Returns:
        Condition number (σ_max / σ_min)
    """
    if len(S) == 0 or S[-1] == 0:
        return float("inf")
    return float((S[0] / S[-1]).item())
