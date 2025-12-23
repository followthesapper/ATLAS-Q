"""
IR Regime Analyzer
==================

Pre-computation analysis to determine observability regime BEFORE running algorithms.

This is the CORE architectural fix for proper IR integration:
- IR says: Diagnose regime → Select representation → Then compute
- Old ATLAS-Q: Compute → Measure coherence → Classify result

Key Functions:
- analyze_hamiltonian_regime(): Diagnose structure observability before VQE
- analyze_state_regime(): Diagnose quantum state observability
- select_representation(): Choose computation strategy based on regime
- estimate_representation_cost(): Predict classical vs quantum cost

IR Laws Applied:
- L5: Exponential Decoherence - R̄(D) = R₀ e^(-αD)
- L8: Placement Principle - Coherence on response field, not probes
- C4: Representation Efficiency - Cost depends on regime

Author: ATLAS-Q Development Team
Date: December 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


# IR Constants
E2_THRESHOLD = 0.135  # e^-2 ≈ 0.1353 - observability boundary


class ObservabilityRegime(Enum):
    """IR observability regime classification."""
    IR = "ir"           # Structure directly observable (R̄ > e^-2)
    TRANSITION = "transition"  # Near boundary
    AIR = "air"         # Structure globally hidden (R̄ < e^-2)


class RepresentationType(Enum):
    """Representation strategy based on regime."""
    DIRECT = "direct"           # Direct measurement/computation
    SPECTRAL_LIFTING = "spectral_lifting"  # Use M_ij relational matrix
    EXPENSIVE = "expensive"     # Exponentially expensive representation needed


@dataclass
class RegimeAnalysis:
    """
    Complete IR regime analysis result.

    This should be computed BEFORE running any quantum algorithm.
    It determines whether the problem is tractable and which
    representation to use.

    Attributes:
        regime: IR/Transition/AIR classification
        coherence: Response field coherence R̄
        variance: Circular variance V_φ
        recommended_representation: Which R to use
        representation_cost: Estimated cost scaling
        is_classically_tractable: Whether classical simulation is feasible
        structure_observable: Whether structure can be observed
        warning: Any warnings about the analysis
    """
    regime: ObservabilityRegime
    coherence: float  # R̄
    variance: float   # V_φ
    recommended_representation: RepresentationType
    representation_cost: str  # "polynomial", "exponential", "unknown"
    is_classically_tractable: bool
    structure_observable: bool
    spectral_gap: Optional[float] = None
    dominant_modes: Optional[int] = None
    warning: Optional[str] = None

    def __str__(self) -> str:
        status = "OBSERVABLE" if self.structure_observable else "HIDDEN"
        return (
            f"RegimeAnalysis(\n"
            f"  regime={self.regime.value.upper()},\n"
            f"  coherence R̄={self.coherence:.4f},\n"
            f"  variance V_φ={self.variance:.4f},\n"
            f"  structure={status},\n"
            f"  representation={self.recommended_representation.value},\n"
            f"  cost={self.representation_cost},\n"
            f"  classically_tractable={self.is_classically_tractable}\n"
            f")"
        )


def compute_response_field_coherence(
    responses: np.ndarray,
    phases: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute coherence on the response field per IR L8 (Placement Principle).

    IR L8: "Coherence must be measured on response manifolds,
           not on probes or encodings."

    The response field χ consists of:
    - For VQE: Parameter gradients ∂E/∂θ
    - For quantum states: Amplitude magnitudes |ψ_i|
    - For Hamiltonians: Term susceptibilities

    Args:
        responses: Response magnitudes χ_i
        phases: Response phases θ_i

    Returns:
        (R_bar, V_phi): Coherence and circular variance
    """
    if len(responses) == 0:
        return 0.0, np.inf

    # Normalize responses
    responses = np.asarray(responses, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)

    total_weight = np.sum(responses)
    if total_weight < 1e-15:
        return 0.0, np.inf

    # Weighted mean phasor
    weighted_re = np.sum(responses * np.cos(phases))
    weighted_im = np.sum(responses * np.sin(phases))

    mean_phasor = complex(weighted_re, weighted_im) / total_weight
    R_bar = np.abs(mean_phasor)
    R_bar = np.clip(R_bar, 0.0, 1.0)

    # Circular variance via coherence law: V_φ = -2 ln(R̄)
    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = np.inf

    return float(R_bar), float(V_phi)


def compute_spectral_structure(
    responses: np.ndarray,
    phases: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Compute spectral structure from relational matrix M_ij.

    M_ij = χ_i χ_j cos(θ_i - θ_j)

    The dominant eigenmode encodes global coherent structure.

    Args:
        responses: Response magnitudes
        phases: Response phases

    Returns:
        (spectral_coherence, spectral_gap, n_dominant_modes)
    """
    n = len(responses)
    if n == 0:
        return 0.0, 1.0, 0

    # Build relational matrix
    phase_diff = np.subtract.outer(phases, phases)
    response_product = np.outer(responses, responses)
    M = response_product * np.cos(phase_diff)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]

    # Spectral coherence = concentration in dominant mode
    total_power = np.sum(eigenvalues)
    if total_power > 1e-15:
        spectral_coherence = eigenvalues[0] / total_power
    else:
        spectral_coherence = 0.0

    # Spectral gap = ratio of first to second eigenvalue
    if len(eigenvalues) > 1 and eigenvalues[1] > 1e-15:
        spectral_gap = eigenvalues[0] / eigenvalues[1]
    else:
        spectral_gap = np.inf

    # Count dominant modes (eigenvalues > 10% of max)
    threshold = 0.1 * eigenvalues[0] if eigenvalues[0] > 0 else 0
    n_dominant = int(np.sum(eigenvalues > threshold))

    return float(spectral_coherence), float(spectral_gap), n_dominant


def classify_regime(R_bar: float, e2_threshold: float = E2_THRESHOLD) -> ObservabilityRegime:
    """Classify observability regime from coherence."""
    if R_bar > e2_threshold:
        return ObservabilityRegime.IR
    elif R_bar > e2_threshold * 0.5:
        return ObservabilityRegime.TRANSITION
    else:
        return ObservabilityRegime.AIR


def select_representation(
    regime: ObservabilityRegime,
    spectral_gap: float,
    n_elements: int,
) -> Tuple[RepresentationType, str]:
    """
    Select representation strategy based on regime analysis.

    IR insight: Representation cost depends on observability regime.
    - IR regime: Polynomial cost, direct methods work
    - AIR regime: Exponential cost, need expensive representation

    Args:
        regime: Observability regime
        spectral_gap: Gap ratio from spectral analysis
        n_elements: Number of elements (qubits, terms, etc.)

    Returns:
        (representation_type, cost_estimate)
    """
    if regime == ObservabilityRegime.IR:
        # Structure observable - direct methods work
        if spectral_gap > 10:
            # Large gap = single dominant structure
            return RepresentationType.DIRECT, "polynomial"
        else:
            # Multiple structures - may need spectral lifting
            return RepresentationType.SPECTRAL_LIFTING, "polynomial"

    elif regime == ObservabilityRegime.TRANSITION:
        # Near boundary - spectral lifting recommended
        return RepresentationType.SPECTRAL_LIFTING, "polynomial"

    else:  # AIR regime
        # Structure globally hidden
        if n_elements > 20:
            # Large system - exponentially expensive
            return RepresentationType.EXPENSIVE, "exponential"
        else:
            # Small system - still tractable with expensive methods
            return RepresentationType.SPECTRAL_LIFTING, "exponential"


def analyze_hamiltonian_regime(
    coefficients: np.ndarray,
    pauli_strings: Optional[List[str]] = None,
    gradient_magnitudes: Optional[np.ndarray] = None,
    gradient_phases: Optional[np.ndarray] = None,
) -> RegimeAnalysis:
    """
    Analyze Hamiltonian observability regime BEFORE running VQE.

    This is the key function for pre-computation regime diagnosis.

    IR L8 (Placement Principle): We compute coherence on the RESPONSE FIELD,
    not on the Hamiltonian coefficients or measurement outcomes.

    For VQE, the response field is the parameter gradient landscape:
    - χ_i = |∂E/∂θ_i| (gradient magnitude)
    - θ_i = arg(∂E/∂θ_i) (gradient phase/direction)

    If gradients not available, we estimate from Hamiltonian structure.

    Args:
        coefficients: Hamiltonian term coefficients
        pauli_strings: Pauli strings for structure analysis
        gradient_magnitudes: Optional actual gradient magnitudes (preferred)
        gradient_phases: Optional actual gradient phases (preferred)

    Returns:
        RegimeAnalysis with complete regime classification

    Example:
        >>> coeffs = np.array([1.0, 0.5, -0.3, 0.2])
        >>> analysis = analyze_hamiltonian_regime(coeffs)
        >>> if analysis.regime == ObservabilityRegime.AIR:
        ...     print("Warning: Structure globally hidden")
        ...     print(f"Recommended: {analysis.recommended_representation.value}")
    """
    n_terms = len(coefficients)

    # Use actual gradients if provided (correct L8 placement)
    if gradient_magnitudes is not None and gradient_phases is not None:
        responses = gradient_magnitudes
        phases = gradient_phases
        warning = None
    else:
        # Estimate response field from Hamiltonian structure
        # This is a proxy - actual gradients are better
        responses = np.abs(coefficients)

        if pauli_strings is not None:
            # Estimate phases from Pauli structure
            phases = _estimate_phases_from_paulis(pauli_strings)
        else:
            # Use coefficient signs as phase proxy
            phases = np.where(coefficients >= 0, 0.0, np.pi)

        warning = "Using estimated response field (provide gradients for L8 compliance)"

    # Compute coherence on response field
    R_bar, V_phi = compute_response_field_coherence(responses, phases)

    # Compute spectral structure
    spectral_coh, spectral_gap, n_dominant = compute_spectral_structure(responses, phases)

    # Classify regime
    regime = classify_regime(R_bar)

    # Select representation
    rep_type, cost = select_representation(regime, spectral_gap, n_terms)

    # Determine tractability
    is_tractable = (cost == "polynomial") or (n_terms < 50)
    structure_observable = (regime == ObservabilityRegime.IR)

    return RegimeAnalysis(
        regime=regime,
        coherence=R_bar,
        variance=V_phi,
        recommended_representation=rep_type,
        representation_cost=cost,
        is_classically_tractable=is_tractable,
        structure_observable=structure_observable,
        spectral_gap=spectral_gap,
        dominant_modes=n_dominant,
        warning=warning,
    )


def analyze_state_regime(
    amplitudes: np.ndarray,
) -> RegimeAnalysis:
    """
    Analyze quantum state observability regime.

    For quantum states, the response field IS the amplitude structure:
    - χ_i = |ψ_i| (amplitude magnitude)
    - θ_i = arg(ψ_i) (amplitude phase)

    Args:
        amplitudes: Complex quantum state amplitudes

    Returns:
        RegimeAnalysis for the quantum state
    """
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)

    responses = np.abs(amplitudes)
    phases = np.angle(amplitudes)

    # Filter negligible amplitudes
    significant_mask = responses > 1e-15
    if not np.any(significant_mask):
        return RegimeAnalysis(
            regime=ObservabilityRegime.AIR,
            coherence=0.0,
            variance=np.inf,
            recommended_representation=RepresentationType.EXPENSIVE,
            representation_cost="undefined",
            is_classically_tractable=False,
            structure_observable=False,
            warning="All amplitudes negligible",
        )

    responses = responses[significant_mask]
    phases = phases[significant_mask]

    R_bar, V_phi = compute_response_field_coherence(responses, phases)
    spectral_coh, spectral_gap, n_dominant = compute_spectral_structure(responses, phases)

    regime = classify_regime(R_bar)
    rep_type, cost = select_representation(regime, spectral_gap, len(amplitudes))

    is_tractable = (cost == "polynomial")
    structure_observable = (regime == ObservabilityRegime.IR)

    return RegimeAnalysis(
        regime=regime,
        coherence=R_bar,
        variance=V_phi,
        recommended_representation=rep_type,
        representation_cost=cost,
        is_classically_tractable=is_tractable,
        structure_observable=structure_observable,
        spectral_gap=spectral_gap,
        dominant_modes=n_dominant,
    )


def analyze_mps_bond_regime(
    singular_values: np.ndarray,
    site_index: int = 0,
    decoherence_rate: float = 0.05,
    initial_coherence: float = 1.0,
) -> RegimeAnalysis:
    """
    Analyze MPS bond observability regime for truncation decisions.

    Uses IR L5 (Exponential Decoherence): R̄(D) = R₀ e^(-αD)

    Args:
        singular_values: SVD singular values at this bond
        site_index: Distance from preparation (D in L5)
        decoherence_rate: Decay rate α
        initial_coherence: Initial coherence R₀

    Returns:
        RegimeAnalysis for truncation decision
    """
    import math

    # Estimate coherence using L5
    estimated_R_bar = initial_coherence * math.exp(-decoherence_rate * site_index)

    # Also compute spectral coherence from singular value distribution
    S = np.asarray(singular_values)
    if len(S) == 0:
        return RegimeAnalysis(
            regime=ObservabilityRegime.AIR,
            coherence=0.0,
            variance=np.inf,
            recommended_representation=RepresentationType.DIRECT,
            representation_cost="polynomial",
            is_classically_tractable=True,
            structure_observable=False,
        )

    # Singular values as response field (magnitude = value, phase = 0)
    responses = S / np.max(S) if np.max(S) > 0 else S
    phases = np.zeros_like(responses)

    spectral_coh, spectral_gap, n_dominant = compute_spectral_structure(responses, phases)

    # Use estimated coherence for regime classification
    R_bar = estimated_R_bar
    V_phi = -2.0 * np.log(R_bar) if R_bar > 1e-10 else np.inf

    regime = classify_regime(R_bar)

    return RegimeAnalysis(
        regime=regime,
        coherence=R_bar,
        variance=V_phi,
        recommended_representation=RepresentationType.DIRECT,
        representation_cost="polynomial",
        is_classically_tractable=True,
        structure_observable=(regime == ObservabilityRegime.IR),
        spectral_gap=spectral_gap,
        dominant_modes=n_dominant,
    )


def _estimate_phases_from_paulis(pauli_strings: List[str]) -> np.ndarray:
    """
    Estimate response phases from Pauli string structure.

    This is a heuristic when actual gradient phases aren't available.
    Uses Pauli structure to estimate phase relationships.
    """
    n = len(pauli_strings)
    if n == 0:
        return np.array([])

    # Simple heuristic: phase based on Pauli content
    phases = np.zeros(n)
    for i, pauli in enumerate(pauli_strings):
        # Count each Pauli type
        n_x = pauli.count('X')
        n_y = pauli.count('Y')
        n_z = pauli.count('Z')

        # Heuristic phase assignment
        # X-heavy terms: phase ~ 0
        # Y-heavy terms: phase ~ π/2
        # Z-heavy terms: phase ~ π
        total = n_x + n_y + n_z
        if total > 0:
            phases[i] = (n_y * np.pi/2 + n_z * np.pi) / total

    return phases


def should_use_ir_grouping(analysis: RegimeAnalysis) -> Tuple[bool, str]:
    """
    Determine if IR measurement grouping will help based on regime analysis.

    Args:
        analysis: Pre-computed regime analysis

    Returns:
        (should_use, reason): Whether to use IR grouping and why
    """
    if analysis.regime == ObservabilityRegime.IR:
        return True, f"IR regime (R̄={analysis.coherence:.3f} > e^-2): grouping effective"

    elif analysis.regime == ObservabilityRegime.TRANSITION:
        return True, f"Transition regime (R̄={analysis.coherence:.3f}): grouping may help"

    else:  # AIR regime
        return False, f"AIR regime (R̄={analysis.coherence:.3f} < e^-2): grouping ineffective, structure hidden"


def predict_quantum_advantage(
    analysis: RegimeAnalysis,
    n_qubits: int,
    classical_hardware_factor: float = 1e9,  # Classical ops/second
) -> Tuple[bool, str]:
    """
    Predict whether quantum advantage exists for this problem.

    Uses IR insight: If structure is in AIR regime and problem is large,
    quantum may have advantage. If in IR regime, classical simulation works.

    Args:
        analysis: Pre-computed regime analysis
        n_qubits: Problem size
        classical_hardware_factor: Classical computation speed

    Returns:
        (quantum_advantage_exists, explanation)
    """
    if analysis.regime == ObservabilityRegime.IR:
        return False, (
            f"No quantum advantage: Structure observable (R̄={analysis.coherence:.3f}). "
            f"Classical simulation via ATLAS-Q is sufficient."
        )

    elif analysis.regime == ObservabilityRegime.TRANSITION:
        if n_qubits > 30:
            return True, (
                f"Possible quantum advantage: Near boundary (R̄={analysis.coherence:.3f}), "
                f"large system ({n_qubits} qubits). Consider quantum hardware."
            )
        else:
            return False, (
                f"No quantum advantage: Near boundary but small system ({n_qubits} qubits). "
                f"Classical methods still tractable."
            )

    else:  # AIR regime
        if n_qubits > 20:
            return True, (
                f"Quantum advantage likely: Structure hidden (R̄={analysis.coherence:.3f}), "
                f"large system ({n_qubits} qubits). Classical cost exponential."
            )
        else:
            return False, (
                f"No quantum advantage: Structure hidden but small system ({n_qubits} qubits). "
                f"Brute force still feasible."
            )
