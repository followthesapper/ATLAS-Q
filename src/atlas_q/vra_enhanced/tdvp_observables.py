"""
VRA-Enhanced TDVP Observable Grouping
======================================

Applies VRA coherence-based grouping to observable measurements during
time evolution with TDVP (Time-Dependent Variational Principle).

Key Insight:
-----------
TDVP simulates real-time dynamics: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩

At each time step, we measure multiple observables:
- Energy: ⟨H⟩
- Magnetization: ⟨Σ Zᵢ⟩
- Correlations: ⟨ZᵢZⱼ⟩
- Custom operators

Many observables commute → can be grouped for simultaneous measurement
VRA optimizes grouping and shot allocation

Target: 5-100× shot reduction per timestep

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TDVPObservableGroupingResult:
    """Result of TDVP observable grouping"""

    groups: List[List[int]]  # Groups of observable indices
    shots_per_group: np.ndarray  # Optimal shot allocation
    variance_reduction: float  # vs baseline
    method: str  # Grouping method
    n_observables: int  # Total observables
    n_groups: int  # Number of groups


def vra_tdvp_observable_grouping(
    observable_paulis: List[str],
    observable_coeffs: np.ndarray,
    total_shots: int = 10000,
    max_group_size: int = 10
) -> TDVPObservableGroupingResult:
    """
    VRA-enhanced grouping for TDVP observable measurements.

    Groups commuting observables and allocates shots optimally.

    Parameters
    ----------
    observable_paulis : List[str]
        Pauli strings for each observable
    observable_coeffs : np.ndarray
        Coefficients/weights for each observable
    total_shots : int
        Total measurement budget per timestep
    max_group_size : int
        Maximum observables per group

    Returns
    -------
    TDVPObservableGroupingResult
        Grouping strategy with shot allocation

    Examples
    --------
    >>> # Energy + correlation measurements
    >>> paulis = ["ZZ", "XX", "YY", "ZI", "IZ"]
    >>> coeffs = np.array([1.0, 0.5, 0.5, 0.3, 0.3])
    >>> result = vra_tdvp_observable_grouping(paulis, coeffs)
    >>> print(f"Variance reduction: {result.variance_reduction:.2f}×")
    """
    from .vqe_grouping import (
        allocate_shots_neyman,
        compute_variance_reduction,
        estimate_pauli_coherence_matrix,
        group_by_variance_minimization,
    )

    # Use VQE grouping infrastructure (same Pauli commutativ logic)
    Sigma = estimate_pauli_coherence_matrix(observable_coeffs, observable_paulis)

    groups = group_by_variance_minimization(
        Sigma, observable_coeffs,
        max_group_size=max_group_size,
        pauli_strings=observable_paulis,
        check_commutativity=True
    )

    shots_per_group = allocate_shots_neyman(Sigma, observable_coeffs, groups, total_shots)

    variance_reduction = compute_variance_reduction(Sigma, observable_coeffs, groups, total_shots)

    return TDVPObservableGroupingResult(
        groups=groups,
        shots_per_group=shots_per_group,
        variance_reduction=variance_reduction,
        method="vra_tdvp_observables",
        n_observables=len(observable_paulis),
        n_groups=len(groups)
    )
