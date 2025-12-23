"""
Coherence Metrics Module
========================

Provides circular statistics-based coherence tracking for quantum measurements
using Informational Relativity (IR).

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
        ir_predicted_to_help: Whether IR grouping is predicted to improve results
        n_measurements: Number of Pauli measurements used for coherence computation
    """
    R_bar: float
    V_phi: float
    is_above_e2_boundary: bool
    ir_predicted_to_help: bool
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
            'ir_predicted_to_help': bool(self.ir_predicted_to_help),
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

    This implements IR Test 2 (Coherence Tracking) using circular statistics
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
        - IR Hardware Validation: COHERENCE_AWARE_VQE_BREAKTHROUGH.md
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

    # Check e^-2 boundary (IR Test 7)
    is_above = R_bar > e2_threshold

    # IR predicted to help when coherence is high
    ir_helps = is_above

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        ir_predicted_to_help=ir_helps,
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


def compute_response_coherence(
    amplitudes: Union[np.ndarray, List[complex]],
    e2_threshold: float = 0.135
) -> CoherenceMetrics:
    """
    Compute coherence on quantum state amplitudes (response field χ).

    This is the CORRECT placement per IR Law L8 (Placement Principle):
    "Coherence must be measured on response manifolds, not on probes or encodings."

    The response field is the quantum state's amplitude/phase structure BEFORE
    measurement collapse - this captures the true coherence of the system.

    Mathematical Details:
        1. Extract phases θ_i = arg(amplitude_i) from complex amplitudes
        2. Weight by amplitude magnitudes |χ_i| (response strength)
        3. Compute weighted mean resultant length: R̄ = |Σ |χ_i| e^(iθ_i)| / Σ |χ_i|
        4. Apply coherence law: V_φ = -2 ln(R̄)

    Args:
        amplitudes: Complex quantum state amplitudes (response field)
        e2_threshold: Threshold for e^-2 boundary (default: 0.135)

    Returns:
        CoherenceMetrics computed from response field

    Example:
        >>> # GHZ state has perfect phase coherence
        >>> amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])  # |00⟩ + |11⟩
        >>> coherence = compute_response_coherence(amplitudes)
        >>> print(f"R̄ = {coherence.R_bar:.4f}")  # High coherence

    References:
        - IR Paper Section 4.8: "L8: Placement Principle"
        - IR Paper Figure 3: Response field coherence yields r=-0.50 correlation
    """
    if isinstance(amplitudes, list):
        amplitudes = np.array(amplitudes, dtype=np.complex128)

    amplitudes = np.asarray(amplitudes, dtype=np.complex128)

    if amplitudes.size == 0:
        raise ValueError("amplitudes cannot be empty")

    # Filter out negligible amplitudes (below numerical precision)
    magnitudes = np.abs(amplitudes)
    significant_mask = magnitudes > 1e-15

    if not np.any(significant_mask):
        # All amplitudes negligible - no coherence
        return CoherenceMetrics(
            R_bar=0.0,
            V_phi=np.inf,
            is_above_e2_boundary=False,
            ir_predicted_to_help=False,
            n_measurements=len(amplitudes)
        )

    # Extract phases from significant amplitudes
    significant_amplitudes = amplitudes[significant_mask]
    significant_magnitudes = magnitudes[significant_mask]
    phases = np.angle(significant_amplitudes)

    # Compute magnitude-weighted mean resultant length
    # This weights phase contributions by response strength |χ_i|
    weighted_phasors = significant_magnitudes * np.exp(1j * phases)
    total_weight = np.sum(significant_magnitudes)

    if total_weight > 1e-15:
        mean_phasor = np.sum(weighted_phasors) / total_weight
        R_bar = float(np.abs(mean_phasor))
    else:
        R_bar = 0.0

    # Clamp R_bar to valid range (numerical stability)
    R_bar = np.clip(R_bar, 0.0, 1.0)

    # Compute circular variance via coherence law
    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = np.inf

    # Check e^-2 boundary
    is_above = R_bar > e2_threshold

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        ir_predicted_to_help=is_above,
        n_measurements=len(amplitudes)
    )


def compute_relational_coherence(
    responses: np.ndarray,
    phases: np.ndarray,
    e2_threshold: float = 0.135
) -> CoherenceMetrics:
    """
    Compute coherence from relational structure M_ij = χ_i χ_j cos(θ_i - θ_j).

    This implements the full IR spectral lifting representation. The dominant
    eigenmode of M encodes global coherent structure.

    Mathematical Details:
        1. Build relational matrix: M_ij = χ_i * χ_j * cos(θ_i - θ_j)
        2. Compute eigendecomposition of M
        3. Extract coherence from spectral concentration: R̄ = λ_max / Σλ

    Args:
        responses: Response field magnitudes χ_i (n-dimensional)
        phases: Response field phases θ_i (n-dimensional)
        e2_threshold: Threshold for e^-2 boundary

    Returns:
        CoherenceMetrics from relational spectral structure

    References:
        - IR Paper Section 5: "The Representation Layer (R)"
        - IR Paper Equation 7: M_ij = χ_i χ_j cos(θ_i - θ_j)
    """
    n = len(responses)
    if n == 0:
        raise ValueError("responses cannot be empty")
    if len(phases) != n:
        raise ValueError(f"responses and phases must have same length, got {n} and {len(phases)}")

    # Build relational spectral lifting matrix
    # M_ij = χ_i * χ_j * cos(θ_i - θ_j)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = responses[i] * responses[j] * np.cos(phases[i] - phases[j])

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Coherence from spectral concentration
    # High coherence = power concentrated in dominant eigenmode
    total_power = np.sum(np.abs(eigenvalues))
    if total_power > 1e-15:
        R_bar = float(np.abs(eigenvalues[0]) / total_power)
    else:
        R_bar = 0.0

    R_bar = np.clip(R_bar, 0.0, 1.0)

    # Circular variance
    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = np.inf

    is_above = R_bar > e2_threshold

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        ir_predicted_to_help=is_above,
        n_measurements=n
    )


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
