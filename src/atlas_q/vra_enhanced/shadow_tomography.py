"""
VRA-Enhanced Shadow Tomography
===============================

Applies VRA coherence-informed sampling to classical shadows protocol
for efficient quantum state characterization.

Key Insight:
-----------
Classical shadows: Random Pauli measurements → Observable estimation

Standard: Uniform random sampling
VRA enhancement: Bias sampling toward high-coherence regions

Benefits:
- Fewer samples for same accuracy
- Better observable estimation
- Coherence-aware sample reuse

Target: 2-10× sample reduction

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ShadowSamplingResult:
    """Result of VRA shadow sampling strategy"""

    pauli_basis: List[str]  # Pauli strings to measure
    measurement_probs: np.ndarray  # Sampling probabilities
    n_samples: int  # Total samples
    expected_variance: float  # Expected measurement variance
    method: str  # Sampling method


def vra_shadow_sampling(
    target_observables: List[str],
    observable_coeffs: np.ndarray,
    n_samples: int = 1000,
    bias_strength: float = 0.5
) -> ShadowSamplingResult:
    """
    VRA-enhanced sampling strategy for classical shadows.

    Biases random Pauli sampling toward observables with high coherence.

    Parameters
    ----------
    target_observables : List[str]
        Pauli observables we want to estimate
    observable_coeffs : np.ndarray
        Importance weights for each observable
    n_samples : int
        Total measurement samples
    bias_strength : float
        0 = uniform (standard shadows), 1 = fully biased

    Returns
    -------
    ShadowSamplingResult
        Sampling strategy

    Examples
    --------
    >>> # Estimate energy observables
    >>> observables = ["ZZ", "XX", "YY", "ZI"]
    >>> coeffs = np.array([1.0, 0.5, 0.5, 0.3])
    >>> result = vra_shadow_sampling(observables, coeffs, n_samples=1000)
    >>> print(f"Sampling probs: {result.measurement_probs}")
    """
    from .vqe_grouping import estimate_pauli_coherence_matrix

    n_obs = len(target_observables)

    # Estimate coherence between observables
    Sigma = estimate_pauli_coherence_matrix(observable_coeffs, target_observables)

    # Compute importance scores based on coherence
    # Higher coherence with other observables → higher priority
    coherence_scores = np.sum(np.abs(Sigma), axis=1)
    weight_scores = np.abs(observable_coeffs)

    # Combined score: coherence + weight
    importance_scores = coherence_scores * weight_scores

    # Normalize to probabilities
    uniform_probs = np.ones(n_obs) / n_obs
    importance_probs = importance_scores / np.sum(importance_scores)

    # Blend uniform and importance-based sampling
    measurement_probs = (1 - bias_strength) * uniform_probs + bias_strength * importance_probs

    # Estimate expected variance (lower with biased sampling)
    # Variance ∝ 1 / (n_samples * probability)
    expected_variance = np.sum(weight_scores**2 / (n_samples * measurement_probs))

    return ShadowSamplingResult(
        pauli_basis=target_observables,
        measurement_probs=measurement_probs,
        n_samples=n_samples,
        expected_variance=expected_variance,
        method=f"vra_shadow_bias{bias_strength}"
    )
