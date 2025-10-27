"""
Adaptive Truncation Policy for MPS

Implements energy-based rank selection with per-bond caps and budget enforcement.

Mathematical foundation:
- Keep smallest k such that Σ_{i≤k} σ_i² ≥ (1-ε²) Σ_i σ_i²
- Local error: ε_local² = Σ_{i>k} σ_i²
- Entropy: S = -Σ_i p_i log(p_i) where p_i = σ_i² / Σ_j σ_j²

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from typing import Callable, Tuple

import torch


def choose_rank_from_sigma(
    S: torch.Tensor,
    eps_bond: float,
    chi_cap: int,
    budget_ok: Callable[[int], bool] = lambda k: True,
) -> Tuple[int, float, float, float]:
    """
    Adaptive rank selection from singular values

    Args:
        S: Singular values (sorted descending)
        eps_bond: Energy tolerance (truncation threshold)
        chi_cap: Maximum allowed rank for this bond
        budget_ok: Function that checks if rank k is within memory budget

    Returns:
        k: Selected rank
        eps_local: Local truncation error
        entropy: Entanglement entropy at this bond
        condS: Condition number (σ_max / σ_k)

    Strategy:
    1. Find k_tol from energy criterion: cumulative energy ≥ (1-ε²) * total
    2. Cap by chi_max_per_bond
    3. Reduce if budget violation
    4. Compute diagnostics (entropy, condition number, error)
    """
    if len(S) == 0:
        return 0, 0.0, 0.0, float("inf")

    # Energy criterion: keep singular values until (1-ε²) of energy retained
    E = (S * S).cumsum(0)
    total = E[-1]

    if total < 1e-30:  # Degenerate case
        return 1, 0.0, 0.0, float("inf")

    thresh = (1.0 - eps_bond**2) * total
    k_tol = int(torch.searchsorted(E, thresh).item()) + 1
    k_tol = min(k_tol, len(S))  # Can't exceed available singular values

    # Apply per-bond cap
    k = min(k_tol, chi_cap)

    # Apply budget constraint (greedy reduction)
    while k > 1 and not budget_ok(k):
        k -= 1

    # Compute local truncation error
    eps_local = torch.sqrt(torch.clamp(total - E[k - 1], min=0.0))

    # Compute entanglement entropy
    S_kept = S[:k]
    p = (S_kept * S_kept) / (S_kept * S_kept).sum()
    entropy = float(-(p * torch.log(torch.clamp(p, min=1e-30))).sum().item())

    # Compute condition number
    if k > 1 and S[k - 1] > 0:
        condS = float((S[0] / S[k - 1]).item())
    else:
        condS = float("inf")

    return k, float(eps_local.item()), entropy, condS


def compute_global_error_bound(local_errors: list) -> float:
    """
    Compute global error bound from local truncation errors

    Using simple Frobenius norm bound:
    ε_global ≤ sqrt(Σ_b ε²_local,b)

    Args:
        local_errors: List of local truncation errors from each bond

    Returns:
        Upper bound on global state error
    """
    import math

    return math.sqrt(sum(e**2 for e in local_errors))


def check_entropy_sanity(entropy: float, chi_left: int, chi_right: int) -> bool:
    """
    Verify entropy is within physical bounds

    For a bond with dimensions χ_L and χ_R, maximum entropy is:
    S_max = log₂(min(χ_L · 2, 2 · χ_R))

    Args:
        entropy: Measured entanglement entropy
        chi_left: Left bond dimension
        chi_right: Right bond dimension

    Returns:
        True if entropy is physically reasonable
    """
    import math

    max_entropy = math.log2(min(chi_left * 2, 2 * chi_right))
    return entropy <= max_entropy + 1e-6  # Allow small numerical error
