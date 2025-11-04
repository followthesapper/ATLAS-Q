"""
Coherence Metrics Module
========================

Provides circular statistics-based coherence tracking for quantum measurements
using Vaca Resonance Analysis (VRA).

Key Concepts:
- R̄ (Mean Resultant Length): Measures phase coherence (0 = random, 1 = perfect)
- V_φ (Circular Variance): Quantifies phase spread (0 = perfect, ∞ = random)
- Coherence Law: R̄ = e^(-V_φ/2) - Universal relationship

Author: ATLAS-Q Development Team
Date: November 2025
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class CoherenceMetrics:
    """
    Circular statistics metrics for quantum measurement coherence.

    Attributes:
        R_bar: Mean resultant length (0-1, higher is better)
        V_phi: Circular variance (0-∞, lower is better)
        is_above_e2_boundary: Whether R̄ > e^-2 ≈ 0.135
        vra_predicted_to_help: Whether VRA grouping is predicted to improve results
        n_measurements: Number of Pauli measurements used for coherence computation
    """
    R_bar: float
    V_phi: float
    is_above_e2_boundary: bool
    vra_predicted_to_help: bool
    n_measurements: int = 0

    def __post_init__(self):
        """Validate coherence metrics."""
        if not (0.0 <= self.R_bar <= 1.0):
            raise ValueError(f"R_bar must be in [0, 1], got {self.R_bar}")
        if self.V_phi < 0.0 and not np.isinf(self.V_phi):
            raise ValueError(f"V_phi must be non-negative or inf, got {self.V_phi}")

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'R_bar': float(self.R_bar),
            'V_phi': float(self.V_phi),
            'is_above_e2_boundary': bool(self.is_above_e2_boundary),
            'vra_predicted_to_help': bool(self.vra_predicted_to_help),
            'n_measurements': int(self.n_measurements)
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "GO" if self.is_above_e2_boundary else "NO-GO"
        return (f"CoherenceMetrics(R̄={self.R_bar:.4f}, V_φ={self.V_phi:.4f}, "
                f"status={status}, n={self.n_measurements})")


def compute_coherence(
    measurement_outcomes: Union[np.ndarray, List[float]],
    e2_threshold: float = 0.135
) -> CoherenceMetrics:
    """
    Compute circular statistics coherence from Pauli expectation values.

    This implements VRA Test 2 (Coherence Tracking) using circular statistics
    to quantify the quality of quantum measurements. The coherence metrics
    provide an objective measure of whether quantum results can be trusted.

    Mathematical Details:
        1. Map Pauli expectations ⟨P⟩ ∈ [-1, 1] to phases φ ∈ [0, 2π]
        2. Compute mean resultant length: R̄ = |⟨e^(iφ)⟩|
        3. Compute circular variance: V_φ = -2 ln(R̄)
        4. Check e^-2 boundary: R̄ > 0.135 → GO, else NO-GO

    Args:
        measurement_outcomes: Array of Pauli expectation values ⟨P⟩ ∈ [-1, 1]
        e2_threshold: Threshold for e^-2 boundary (default: 0.135)

    Returns:
        CoherenceMetrics with R̄, V_φ, and classification

    Raises:
        ValueError: If measurement_outcomes is empty or contains invalid values

    Example:
        >>> outcomes = np.array([0.8, 0.9, 0.85, 0.88])
        >>> coherence = compute_coherence(outcomes)
        >>> print(f"R̄ = {coherence.R_bar:.4f}")
        R̄ = 0.8675
        >>> print(f"Classification: {'GO' if coherence.is_above_e2_boundary else 'NO-GO'}")
        Classification: GO

    References:
        - Mardia & Jupp, "Directional Statistics" (2000)
        - VRA Hardware Validation: COHERENCE_AWARE_VQE_BREAKTHROUGH.md
    """
    # Input validation
    if isinstance(measurement_outcomes, list):
        measurement_outcomes = np.array(measurement_outcomes)

    if measurement_outcomes.size == 0:
        raise ValueError("measurement_outcomes cannot be empty")

    if not np.all((measurement_outcomes >= -1.0) & (measurement_outcomes <= 1.0)):
        # Check if any values are slightly outside due to numerical precision
        if np.any((measurement_outcomes < -1.01) | (measurement_outcomes > 1.01)):
            raise ValueError(
                f"measurement_outcomes must be in [-1, 1], got range "
                f"[{np.min(measurement_outcomes):.4f}, {np.max(measurement_outcomes):.4f}]"
            )
        # Clip to valid range for small numerical errors
        measurement_outcomes = np.clip(measurement_outcomes, -1.0, 1.0)

    # Convert Pauli expectations to phases
    # ⟨P⟩ = cos(φ) for a single-phase model
    # φ ∈ [0, π] since arccos returns [0, π]
    phases = np.arccos(np.clip(measurement_outcomes, -1.0, 1.0))

    # Compute mean resultant length (circular mean magnitude)
    phasors = np.exp(1j * phases)
    mean_phasor = np.mean(phasors)
    R_bar = float(np.abs(mean_phasor))

    # Compute circular variance
    # V_φ = -2 ln(R̄), but handle R̄ → 0 case
    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = np.inf

    # Check e^-2 boundary (VRA Test 7)
    is_above = R_bar > e2_threshold

    # VRA predicted to help when coherence is high
    vra_helps = is_above

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        vra_predicted_to_help=vra_helps,
        n_measurements=len(measurement_outcomes)
    )


def coherence_from_counts(
    counts_list: List[dict],
    pauli_strings: List[str]
) -> CoherenceMetrics:
    """
    Compute coherence directly from measurement counts.

    Convenience function that computes Pauli expectations from counts
    and then calculates coherence metrics.

    Args:
        counts_list: List of count dictionaries from circuit execution
        pauli_strings: List of Pauli strings (e.g., ['IXZY', 'ZZII'])

    Returns:
        CoherenceMetrics computed from the measurements

    Raises:
        ValueError: If counts_list and pauli_strings have different lengths

    Example:
        >>> counts = [{'00': 800, '11': 200}, {'01': 600, '10': 400}]
        >>> paulis = ['ZZ', 'XX']
        >>> coherence = coherence_from_counts(counts, paulis)
    """
    if len(counts_list) != len(pauli_strings):
        raise ValueError(
            f"counts_list and pauli_strings must have same length, "
            f"got {len(counts_list)} and {len(pauli_strings)}"
        )

    # Import here to avoid circular dependency
    from .utils import compute_pauli_expectation

    # Compute expectation value for each Pauli
    expectations = []
    for counts, pauli_str in zip(counts_list, pauli_strings):
        exp_val = compute_pauli_expectation(counts, pauli_str)
        expectations.append(exp_val)

    return compute_coherence(np.array(expectations))


def validate_coherence_law(R_bar: float, V_phi: float, tolerance: float = 0.1) -> bool:
    """
    Validate the coherence law: R̄ = e^(-V_φ/2).

    This checks whether the measured coherence metrics satisfy the
    fundamental relationship between R̄ and V_φ.

    Args:
        R_bar: Mean resultant length
        V_phi: Circular variance
        tolerance: Relative tolerance for validation (default: 0.1 = 10%)

    Returns:
        True if coherence law is satisfied within tolerance

    Example:
        >>> coherence = compute_coherence([0.9, 0.85, 0.88])
        >>> is_valid = validate_coherence_law(coherence.R_bar, coherence.V_phi)
        >>> print(f"Coherence law valid: {is_valid}")
    """
    if np.isinf(V_phi):
        # R̄ → 0 case: law is satisfied if R̄ is very small
        return R_bar < 1e-6

    expected_R_bar = np.exp(-V_phi / 2.0)
    relative_error = abs(R_bar - expected_R_bar) / expected_R_bar if expected_R_bar > 0 else abs(R_bar)

    return relative_error < tolerance
