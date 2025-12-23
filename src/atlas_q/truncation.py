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


# =============================================================================
# IR-Enhanced Truncation (New in v0.7.0)
# =============================================================================

# IR constants
E2_THRESHOLD = 0.135  # e^-2 threshold for observability


def analyze_truncation_regime(
    S: torch.Tensor,
    site_index: int = 0,
    decoherence_rate: float = 0.05,
    initial_coherence: float = 1.0,
):
    """
    Analyze regime BEFORE truncation decision (IR-correct approach).

    This implements the key IR insight: regime determines representation,
    not the other way around. We diagnose first, then choose truncation.

    Args:
        S: Singular values (sorted descending)
        site_index: Distance from preparation (for L5 decoherence)
        decoherence_rate: Decay rate α in R̄(D) = R₀ e^(-αD)
        initial_coherence: Initial coherence R₀

    Returns:
        RegimeAnalysis from ir_enhanced.regime_analyzer
    """
    from .ir_enhanced.regime_analyzer import analyze_mps_bond_regime

    return analyze_mps_bond_regime(
        singular_values=S.cpu().numpy() if hasattr(S, 'cpu') else S,
        site_index=site_index,
        decoherence_rate=decoherence_rate,
        initial_coherence=initial_coherence,
    )


def choose_rank_with_regime(
    S: torch.Tensor,
    eps_bond: float,
    chi_cap: int,
    site_index: int = 0,
    decoherence_rate: float = 0.05,
    initial_coherence: float = 1.0,
    budget_ok: Callable[[int], bool] = lambda k: True,
) -> Tuple[int, float, float, float, "RegimeAnalysis"]:
    """
    IR-correct rank selection: Diagnose regime FIRST, then truncate.

    This is the proper IR approach:
    1. Analyze regime using L5 decoherence model
    2. Select epsilon based on regime (not the other way around)
    3. Truncate with regime-appropriate tolerance

    Args:
        S: Singular values (sorted descending)
        eps_bond: Base energy tolerance
        chi_cap: Maximum allowed rank
        site_index: Distance from preparation site
        decoherence_rate: Decay rate α
        initial_coherence: Initial coherence R₀
        budget_ok: Budget constraint function

    Returns:
        (k, eps_local, entropy, condS, regime_analysis)
    """
    from .ir_enhanced.regime_analyzer import ObservabilityRegime

    # Step 1: Diagnose regime BEFORE truncation
    regime_analysis = analyze_truncation_regime(
        S, site_index, decoherence_rate, initial_coherence
    )

    # Step 2: Select epsilon based on regime
    if regime_analysis.regime == ObservabilityRegime.IR:
        # Structure observable - preserve it with tight tolerance
        effective_eps = eps_bond * 0.5
    elif regime_analysis.regime == ObservabilityRegime.TRANSITION:
        # Near boundary - standard tolerance
        effective_eps = eps_bond
    else:  # AIR regime
        # Structure hidden - aggressive truncation is safe
        effective_eps = eps_bond * 2.0

    # Step 3: Standard energy-based truncation with regime-adjusted eps
    if len(S) == 0:
        return 0, 0.0, 0.0, float("inf"), regime_analysis

    E = (S * S).cumsum(0)
    total = E[-1]

    if total < 1e-30:
        return 1, 0.0, 0.0, float("inf"), regime_analysis

    thresh = (1.0 - effective_eps**2) * total
    k_tol = int(torch.searchsorted(E, thresh).item()) + 1
    k_tol = min(k_tol, len(S))

    k = min(k_tol, chi_cap)

    while k > 1 and not budget_ok(k):
        k -= 1

    # Compute diagnostics
    eps_local = torch.sqrt(torch.clamp(total - E[k - 1], min=0.0))

    S_kept = S[:k]
    p = (S_kept * S_kept) / (S_kept * S_kept).sum()
    entropy = float(-(p * torch.log(torch.clamp(p, min=1e-30))).sum().item())

    if k > 1 and S[k - 1] > 0:
        condS = float((S[0] / S[k - 1]).item())
    else:
        condS = float("inf")

    return k, float(eps_local.item()), entropy, condS, regime_analysis


def choose_rank_with_coherence(
    S: torch.Tensor,
    eps_bond: float,
    chi_cap: int,
    coherence: float,
    budget_ok: Callable[[int], bool] = lambda k: True,
    e2_threshold: float = E2_THRESHOLD,
) -> Tuple[int, float, float, float, float]:
    """
    IR-enhanced rank selection incorporating coherence metrics.

    Implements IR Law L5 (Exponential Decoherence) insight:
    - High coherence (R̄ > e^-2): Structure is observable, preserve it
    - Low coherence (R̄ < e^-2): Structure is hidden, aggressive truncation OK

    Args:
        S: Singular values (sorted descending)
        eps_bond: Energy tolerance (truncation threshold)
        chi_cap: Maximum allowed rank for this bond
        coherence: Current coherence metric R̄ (0-1)
        budget_ok: Function that checks if rank k is within memory budget
        e2_threshold: e^-2 threshold (default: 0.135)

    Returns:
        k: Selected rank
        eps_local: Local truncation error
        entropy: Entanglement entropy at this bond
        condS: Condition number (σ_max / σ_k)
        coherence_adjusted: Whether coherence influenced the decision

    Strategy:
    1. Start with standard energy criterion
    2. If coherence > e^-2: be conservative (preserve structure)
    3. If coherence < e^-2: allow aggressive truncation (structure hidden anyway)
    4. Apply caps and budget constraints
    """
    if len(S) == 0:
        return 0, 0.0, 0.0, float("inf"), 0.0

    # Energy criterion: keep singular values until (1-ε²) of energy retained
    E = (S * S).cumsum(0)
    total = E[-1]

    if total < 1e-30:  # Degenerate case
        return 1, 0.0, 0.0, float("inf"), 0.0

    # Adjust epsilon based on coherence regime
    if coherence > e2_threshold:
        # IR regime: structure observable, be conservative
        # Use tighter tolerance to preserve coherent structure
        effective_eps = eps_bond * 0.5
        coherence_adjusted = 1.0
    elif coherence > e2_threshold * 0.5:
        # Transition regime: moderate caution
        effective_eps = eps_bond
        coherence_adjusted = 0.5
    else:
        # AIR regime: structure hidden, aggressive truncation OK
        # Use looser tolerance (saves resources without losing information)
        effective_eps = eps_bond * 2.0
        coherence_adjusted = 0.0

    thresh = (1.0 - effective_eps**2) * total
    k_tol = int(torch.searchsorted(E, thresh).item()) + 1
    k_tol = min(k_tol, len(S))

    # Apply per-bond cap
    k = min(k_tol, chi_cap)

    # Apply budget constraint
    while k > 1 and not budget_ok(k):
        k -= 1

    # Compute diagnostics
    eps_local = torch.sqrt(torch.clamp(total - E[k - 1], min=0.0))

    S_kept = S[:k]
    p = (S_kept * S_kept) / (S_kept * S_kept).sum()
    entropy = float(-(p * torch.log(torch.clamp(p, min=1e-30))).sum().item())

    if k > 1 and S[k - 1] > 0:
        condS = float((S[0] / S[k - 1]).item())
    else:
        condS = float("inf")

    return k, float(eps_local.item()), entropy, condS, coherence_adjusted


def choose_rank_with_decoherence(
    S: torch.Tensor,
    eps_bond: float,
    chi_cap: int,
    decoherence_rate: float,
    site_index: int,
    initial_coherence: float = 1.0,
    budget_ok: Callable[[int], bool] = lambda k: True,
    e2_threshold: float = E2_THRESHOLD,
) -> Tuple[int, float, float, float, float]:
    """
    IR L5-based rank selection with exponential decoherence model.

    Implements IR Law L5: R̄(D) = R₀ e^(-αD)
    Coherence decays exponentially under perturbation/distance.

    Args:
        S: Singular values (sorted descending)
        eps_bond: Energy tolerance (truncation threshold)
        chi_cap: Maximum allowed rank for this bond
        decoherence_rate: Decay rate α (higher = faster decay)
        site_index: Distance from preparation site D
        initial_coherence: Initial coherence R₀ (default: 1.0)
        budget_ok: Function that checks if rank k is within memory budget
        e2_threshold: e^-2 threshold (default: 0.135)

    Returns:
        k: Selected rank
        eps_local: Local truncation error
        entropy: Entanglement entropy
        condS: Condition number
        estimated_coherence: Estimated coherence at this site

    References:
        - IR Paper Section 4.5: "L5: Exponential Decoherence Law"
        - Formula: R̄(D) = R₀ e^(-αD)
    """
    import math

    # Estimate coherence at this site using L5
    estimated_coherence = initial_coherence * math.exp(-decoherence_rate * site_index)

    # Use coherence-aware truncation
    return choose_rank_with_coherence(
        S=S,
        eps_bond=eps_bond,
        chi_cap=chi_cap,
        coherence=estimated_coherence,
        budget_ok=budget_ok,
        e2_threshold=e2_threshold
    )


def compute_spectral_coherence(S: torch.Tensor) -> float:
    """
    Compute spectral coherence from singular value distribution.

    High spectral coherence = power concentrated in dominant modes
    Low spectral coherence = power spread across many modes

    This can be used as a proxy for response field coherence when
    direct amplitude access is not available.

    Args:
        S: Singular values (sorted descending)

    Returns:
        Spectral coherence R̄ ∈ [0, 1]
    """
    if len(S) == 0:
        return 0.0

    # Normalize singular values to probabilities
    S2 = S * S
    total = S2.sum()

    if total < 1e-30:
        return 0.0

    p = S2 / total

    # Coherence = concentration in dominant mode
    # For perfectly coherent state: p = [1, 0, 0, ...] → R̄ = 1
    # For maximally mixed: p = [1/n, 1/n, ...] → R̄ = 1/n → 0 for large n
    coherence = float(p[0].item())

    return coherence


def adaptive_chi_from_coherence(
    coherence_history: list,
    current_chi: int,
    chi_min: int = 2,
    chi_max: int = 256,
    e2_threshold: float = E2_THRESHOLD,
    growth_rate: float = 1.2,
    shrink_rate: float = 0.8,
) -> int:
    """
    Four-layer hierarchy feedback: Observable → Structure adaptation.

    Adjusts bond dimension based on measured coherence to optimize
    the representation cost according to IR principles.

    IR insight:
    - Low coherence at observable layer → representation cost too high
    - Reduce χ to save resources (structure is hidden anyway)
    - High coherence → structure is valuable, allow χ growth

    Args:
        coherence_history: Recent coherence measurements (R̄ values)
        current_chi: Current bond dimension
        chi_min: Minimum allowed bond dimension
        chi_max: Maximum allowed bond dimension
        e2_threshold: e^-2 threshold
        growth_rate: Multiplier for chi growth (default: 1.2)
        shrink_rate: Multiplier for chi shrinkage (default: 0.8)

    Returns:
        Adjusted bond dimension

    References:
        - IR Paper Section 2: "The Four-Layer Framework"
        - Observable layer feeds back to Structure layer
    """
    if not coherence_history:
        return current_chi

    # Use recent average coherence
    avg_coherence = sum(coherence_history[-5:]) / len(coherence_history[-5:])

    if avg_coherence < e2_threshold:
        # AIR regime: structure hidden, reduce resources
        new_chi = int(current_chi * shrink_rate)
    elif avg_coherence > 0.5:
        # High coherence: structure valuable, allow growth
        new_chi = int(current_chi * growth_rate)
    else:
        # Moderate coherence: maintain current
        new_chi = current_chi

    # Apply bounds
    new_chi = max(chi_min, min(chi_max, new_chi))

    return new_chi


def coherence_aware_truncation_policy(
    site_coherences: list,
    base_eps: float = 1e-6,
    base_chi: int = 64,
    e2_threshold: float = E2_THRESHOLD,
) -> Tuple[list, list]:
    """
    Generate per-site truncation policy based on coherence distribution.

    Sites with high coherence get tighter truncation (preserve structure).
    Sites with low coherence get looser truncation (save resources).

    Args:
        site_coherences: Coherence R̄ at each site
        base_eps: Base truncation tolerance
        base_chi: Base bond dimension
        e2_threshold: e^-2 threshold

    Returns:
        eps_per_site: Truncation tolerance for each site
        chi_per_site: Bond dimension cap for each site
    """
    n_sites = len(site_coherences)
    eps_per_site = []
    chi_per_site = []

    for i, coherence in enumerate(site_coherences):
        if coherence > e2_threshold * 2:
            # High coherence: tight truncation, high chi
            eps_per_site.append(base_eps * 0.5)
            chi_per_site.append(int(base_chi * 1.5))
        elif coherence > e2_threshold:
            # Above threshold: standard
            eps_per_site.append(base_eps)
            chi_per_site.append(base_chi)
        elif coherence > e2_threshold * 0.5:
            # Near threshold: slightly loose
            eps_per_site.append(base_eps * 1.5)
            chi_per_site.append(int(base_chi * 0.8))
        else:
            # Low coherence: aggressive truncation
            eps_per_site.append(base_eps * 2.0)
            chi_per_site.append(int(base_chi * 0.5))

    return eps_per_site, chi_per_site
