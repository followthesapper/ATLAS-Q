# src/quantum_hybrid_system/sim_safety_checks.py
"""
Safety Checks for Quantum Simulator
====================================

Validates correctness of MPS operations: unitarity, norm preservation,
trace conditions, and truncation error budgets.

These checks ensure numerical stability and catch bugs early.

Author: ATLAS-Q Contributors
Date: October 24, 2025
"""

from typing import Dict, Optional

import torch


def check_unitary_gate(U: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-4) -> Dict[str, float]:
    """
    Verify that U is unitary: U @ U† = I.

    Args:
        U: [d, d] or [batch, d, d] gate matrix
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with 'is_unitary' (bool), 'max_error' (float)
    """
    if U.dim() == 2:
        U = U.unsqueeze(0)  # Add batch dim

    batch, d, _ = U.shape

    # Compute U @ U†
    UUdagger = torch.bmm(U, U.conj().transpose(-2, -1))

    # Expected: Identity
    eye = torch.eye(d, dtype=U.dtype, device=U.device).unsqueeze(0).expand(batch, -1, -1)

    # Error
    error = torch.abs(UUdagger - eye)
    max_error = error.max().item()

    is_unitary = bool(torch.allclose(UUdagger, eye, atol=atol, rtol=rtol))

    return {
        "is_unitary": is_unitary,
        "max_error": max_error,
    }


def check_norm_preservation(
    state_before: torch.Tensor,
    state_after: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, float]:
    """
    Check that norm is preserved after operation.

    Args:
        state_before: MPS state before operation
        state_after: MPS state after operation
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with 'norm_before', 'norm_after', 'is_preserved' (bool)
    """
    if isinstance(state_before, list):
        # Compute norm from cores
        norm_before = _compute_mps_norm(state_before)
        norm_after = _compute_mps_norm(state_after)
    else:
        # Direct tensor norm
        norm_before = torch.linalg.norm(state_before).item()
        norm_after = torch.linalg.norm(state_after).item()

    is_preserved = abs(norm_before - norm_after) < atol + rtol * norm_before

    return {
        "norm_before": norm_before,
        "norm_after": norm_after,
        "is_preserved": is_preserved,
    }


def check_rdm_properties(
    rdm: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Dict[str, any]:
    """
    Validate reduced density matrix (RDM) properties.

    Checks:
    1. Hermiticity: ρ = ρ†
    2. Trace = 1
    3. Positive semi-definite (all eigenvalues ≥ 0)

    Args:
        rdm: [d, d] reduced density matrix (complex)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Dict with 'is_hermitian', 'trace', 'trace_ok', 'min_eigenvalue', 'is_positive'
    """
    # Hermiticity check
    rdm_dagger = rdm.conj().T
    is_hermitian = bool(torch.allclose(rdm, rdm_dagger, atol=atol, rtol=rtol))

    # Trace check
    trace = torch.trace(rdm).real.item()
    trace_ok = abs(trace - 1.0) < atol + rtol

    # Eigenvalues (for positivity check)
    eigenvalues = torch.linalg.eigvalsh(rdm)  # Real eigenvalues for Hermitian
    min_eigenvalue = eigenvalues.min().item()
    is_positive = min_eigenvalue >= -atol

    return {
        "is_hermitian": is_hermitian,
        "trace": trace,
        "trace_ok": trace_ok,
        "min_eigenvalue": min_eigenvalue,
        "is_positive": is_positive,
    }


def check_truncation_error(
    singular_values: torch.Tensor,
    chi_kept: int,
    max_error: float = 1e-6,
) -> Dict[str, any]:
    """
    Validate truncation error is within budget.

    Truncation error: ε² = Σ_{i>chi} σᵢ²

    Args:
        singular_values: [chi_full] singular values (descending)
        chi_kept: Number of singular values kept
        max_error: Maximum allowed truncation error

    Returns:
        Dict with 'truncation_error', 'error_ok', 'weight_kept'
    """
    # Ensure sorted descending
    S = torch.sort(singular_values, descending=True)[0]

    # Total weight
    total_weight = (S**2).sum().item()

    # Kept weight
    kept_weight = (S[:chi_kept] ** 2).sum().item()

    # Discarded weight (truncation error²)
    if chi_kept < S.shape[0]:
        discarded_weight = (S[chi_kept:] ** 2).sum().item()
    else:
        discarded_weight = 0.0

    truncation_error = discarded_weight**0.5
    error_ok = truncation_error <= max_error

    weight_fraction = kept_weight / total_weight if total_weight > 0 else 1.0

    return {
        "truncation_error": truncation_error,
        "error_ok": error_ok,
        "weight_kept": weight_fraction,
        "discarded_weight": discarded_weight,
    }


def _compute_mps_norm(cores: list) -> float:
    """
    Compute norm of MPS from cores.

    Uses transfer matrix approach:
    ||ψ||² = ⟨ψ|ψ⟩ = product of transfer matrices

    Args:
        cores: List of [chi_l, d, chi_r] MPS tensors

    Returns:
        Norm (float)
    """
    device = cores[0].device
    dtype = cores[0].dtype

    # Start with left boundary
    chi_left = cores[0].shape[0]
    E = torch.zeros(chi_left, chi_left, dtype=dtype, device=device)
    E[0, 0] = 1.0

    # Contract through all sites
    for core in cores:
        chi_l, d, chi_r = core.shape

        # Transfer matrix: E_new[a', b'] = sum_{a,b,s} E[a,b] * core[a,s,a'] * core[b,s,b'].conj()
        E_new = torch.zeros(chi_r, chi_r, dtype=dtype, device=device)

        for s in range(d):
            # E_new += core[:, s, :].T @ E @ core[:, s, :].conj()
            temp = torch.mm(E, core[:, s, :].conj())
            E_new += torch.mm(core[:, s, :].T.conj(), temp)

        E = E_new

    # Final: should be [1, 1], extract scalar
    norm_sq = E[0, 0].real.item() if E.shape == (1, 1) else torch.trace(E).real.item()
    return norm_sq**0.5


def run_full_validation(
    cores_before: list,
    cores_after: list,
    gate: Optional[torch.Tensor] = None,
    singular_values: Optional[torch.Tensor] = None,
    chi_kept: Optional[int] = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_trunc_error: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Run all validation checks on MPS operation.

    Args:
        cores_before: MPS cores before operation
        cores_after: MPS cores after operation
        gate: Optional gate used (for unitarity check)
        singular_values: Optional SVD singular values (for truncation check)
        chi_kept: Number of singular values kept
        atol: Absolute tolerance
        rtol: Relative tolerance
        max_trunc_error: Max allowed truncation error
        verbose: Print results

    Returns:
        Dict with all check results and overall 'passed' (bool)
    """
    results = {}
    passed = True

    # 1. Norm preservation
    norm_check = check_norm_preservation(cores_before, cores_after, atol, rtol)
    results["norm"] = norm_check
    if not norm_check["is_preserved"]:
        passed = False
        if verbose:
            print(
                f"❌ Norm not preserved: {norm_check['norm_before']:.6f} → {norm_check['norm_after']:.6f}"
            )

    # 2. Gate unitarity (if provided)
    if gate is not None:
        unitary_check = check_unitary_gate(gate, atol, rtol)
        results["unitary"] = unitary_check
        if not unitary_check["is_unitary"]:
            passed = False
            if verbose:
                print(f"❌ Gate not unitary: max error = {unitary_check['max_error']:.2e}")

    # 3. Truncation error (if provided)
    if singular_values is not None and chi_kept is not None:
        trunc_check = check_truncation_error(singular_values, chi_kept, max_trunc_error)
        results["truncation"] = trunc_check
        if not trunc_check["error_ok"]:
            passed = False
            if verbose:
                print(
                    f"❌ Truncation error too large: {trunc_check['truncation_error']:.2e} > {max_trunc_error:.2e}"
                )

    results["passed"] = passed

    if verbose and passed:
        print("✅ All validation checks passed")

    return results


__all__ = [
    "check_unitary_gate",
    "check_norm_preservation",
    "check_rdm_properties",
    "check_truncation_error",
    "run_full_validation",
]
