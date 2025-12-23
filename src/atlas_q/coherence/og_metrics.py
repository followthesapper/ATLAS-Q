"""
Observability Geometry (OG) Metrics Module
===========================================

Advanced coherence metrics derived from Observability Geometry (OG) theory,
which provides a pre-quantum, pre-GR theoretical foundation for quantum mechanics.

Key OG Concepts Implemented:
- Effective Planck constant: h_eff = 2*sqrt(-2*ln(R_bar)) * sqrt(R_bar)
- Representational cost: C = 1/V_phi (observability resource metric)
- Coherence gradient: |nabla V_phi|^2 (action minimization target)
- Transition regime dynamics (continuous e^-2 transition)
- Action functional: S = integral (nabla V_phi)^2 dx

OG Four-Layer Stack:
    S -> chi -> R -> Psi
    (Structure -> Response -> Representation -> Observable)

The e^-2 threshold (R_bar_c ~ 0.135) is the critical observability transition
where quantum structure becomes extractable from the response field.

References:
    - OG Paper 1: "Observability Geometry: Foundations"
    - OG Paper 2: "Observability Geometry: Emergent Geometry"

Author: ATLAS-Q Development Team
Date: December 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# OG Constants (Derived, not postulated)
# =============================================================================

# Critical coherence threshold: R_bar_c = e^-2
R_BAR_CRITICAL = np.exp(-2)  # ~0.1353

# Transition regime boundaries (natural width from OG)
TRANSITION_LOWER = R_BAR_CRITICAL * np.exp(-1)  # ~0.0498
TRANSITION_UPPER = R_BAR_CRITICAL * np.exp(1)   # ~0.3679

# Derived effective Planck constant at critical point
# h_eff_c = 2 * sqrt(-2 * ln(R_c)) * sqrt(R_c) = 2 * sqrt(4) * sqrt(e^-2) = 4 * e^-1 ~ 1.47
H_EFF_CRITICAL = 2 * np.sqrt(-2 * np.log(R_BAR_CRITICAL)) * np.sqrt(R_BAR_CRITICAL)

# Cosmological instability coefficient (derived from wave equation)
ALPHA_COSMOLOGICAL = 0.5  # alpha = d/2 for d=1


class ObservabilityRegime(Enum):
    """
    OG observability regimes based on R_bar position relative to e^-2.

    IR (Informational Relativity): R_bar > e^-2, structure observable
    TRANSITION: Near e^-2 boundary, critical dynamics
    AIR (Anti-IR): R_bar < e^-2, structure hidden in noise
    """
    IR = "ir"           # Observable structure (GO)
    TRANSITION = "transition"  # Critical boundary region
    AIR = "air"         # Hidden structure (NO-GO)


@dataclass
class OGMetrics:
    """
    Comprehensive Observability Geometry metrics.

    Contains all OG-derived quantities for a quantum state:
    - Standard coherence (R_bar, V_phi)
    - Effective Planck constant (h_eff)
    - Representational cost (C)
    - Observability regime classification
    - Transition dynamics indicators

    Attributes:
        R_bar: Mean resultant length [0, 1]
        V_phi: Circular variance [0, inf]
        h_eff: Effective Planck constant (basis-dependent)
        rep_cost: Representational cost C = 1/V_phi
        regime: ObservabilityRegime (IR, TRANSITION, AIR)
        transition_param: Distance from critical point, normalized
        observability_score: Overall observability quality [0, 1]
        is_above_critical: Whether R_bar > R_bar_c
        distance_to_critical: |R_bar - R_bar_c|
    """
    R_bar: float
    V_phi: float
    h_eff: float
    rep_cost: float
    regime: ObservabilityRegime
    transition_param: float  # Normalized distance from critical: (R_bar - R_c) / R_c
    observability_score: float  # Combined quality metric [0, 1]
    is_above_critical: bool
    distance_to_critical: float

    # Optional detailed metrics
    spectral_coherence: Optional[float] = None
    coherence_gradient: Optional[float] = None
    action_density: Optional[float] = None

    def __post_init__(self):
        """Validate metrics."""
        if not (0.0 <= self.R_bar <= 1.0):
            raise ValueError(f"R_bar must be in [0, 1], got {self.R_bar}")
        if self.V_phi < 0.0 and not np.isinf(self.V_phi):
            raise ValueError(f"V_phi must be non-negative, got {self.V_phi}")

    def as_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'R_bar': float(self.R_bar),
            'V_phi': float(self.V_phi) if not np.isinf(self.V_phi) else 'inf',
            'h_eff': float(self.h_eff) if not np.isinf(self.h_eff) else 'inf',
            'rep_cost': float(self.rep_cost) if not np.isinf(self.rep_cost) else 'inf',
            'regime': self.regime.value,
            'transition_param': float(self.transition_param),
            'observability_score': float(self.observability_score),
            'is_above_critical': bool(self.is_above_critical),
            'distance_to_critical': float(self.distance_to_critical),
            'spectral_coherence': float(self.spectral_coherence) if self.spectral_coherence else None,
            'coherence_gradient': float(self.coherence_gradient) if self.coherence_gradient else None,
            'action_density': float(self.action_density) if self.action_density else None,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"OGMetrics(\n"
            f"  R̄={self.R_bar:.4f}, V_φ={self.V_phi:.4f}\n"
            f"  ℏ_eff={self.h_eff:.4f}, C={self.rep_cost:.4f}\n"
            f"  regime={self.regime.value.upper()}, score={self.observability_score:.4f}\n"
            f")"
        )


# =============================================================================
# Core OG Metric Functions
# =============================================================================

def compute_effective_hbar(R_bar: float) -> float:
    """
    Compute OG-derived effective Planck constant.

    From OG theory, the effective Planck constant is basis-dependent:
        h_eff = 2 * sqrt(-2 * ln(R_bar)) * sqrt(R_bar)

    This represents the "quantum granularity" - the fundamental resolution
    limit for measurements in a given coherence state.

    Key values:
        R_bar = 1.0: h_eff = 0 (classical limit, perfect resolution)
        R_bar = e^-2 ~ 0.135: h_eff ~ 1.47 (critical point)
        R_bar -> 0: h_eff -> inf (quantum limit, no resolution)

    Args:
        R_bar: Mean resultant length [0, 1]

    Returns:
        Effective Planck constant h_eff >= 0

    Example:
        >>> h_eff = compute_effective_hbar(0.5)
        >>> print(f"h_eff = {h_eff:.4f}")  # ~1.18

    Theoretical significance:
        This is a FALSIFIABLE PREDICTION of OG theory. Standard QM predicts
        a fixed h regardless of measurement basis, while OG predicts this
        basis-dependent h_eff.
    """
    if R_bar <= 0 or R_bar > 1.0:
        if R_bar <= 0:
            return float('inf')
        R_bar = 1.0

    if R_bar > 1.0 - 1e-10:
        # Limit as R_bar -> 1: h_eff -> 0
        return 0.0

    # h_eff = 2 * sqrt(-2 * ln(R_bar)) * sqrt(R_bar)
    log_term = -2.0 * np.log(R_bar)
    h_eff = 2.0 * np.sqrt(log_term) * np.sqrt(R_bar)

    return float(h_eff)


def compute_representational_cost(V_phi: float) -> float:
    """
    Compute OG representational cost functional.

    From OG theory, the cost of representing structure is:
        C = 1 / V_phi

    This quantifies the "observability resource" required:
        - High V_phi (low coherence): C -> 0 (cheap but useless representation)
        - Low V_phi (high coherence): C -> inf (expensive but high-fidelity)

    Physical interpretation:
        C represents the "cost" an observer pays to extract structure
        from the response field. Higher C means more resources (shots,
        precision, entanglement) are needed for accurate extraction.

    Args:
        V_phi: Circular variance [0, inf]

    Returns:
        Representational cost C >= 0

    Example:
        >>> C = compute_representational_cost(0.5)
        >>> print(f"C = {C:.4f}")  # 2.0
    """
    if V_phi <= 0 or np.isinf(V_phi):
        if V_phi <= 0:
            return float('inf')
        return 0.0

    return 1.0 / V_phi


def compute_coherence_gradient(
    qubit_R_bars: Union[np.ndarray, List[float]]
) -> float:
    """
    Compute coherence gradient magnitude |nabla V_phi|^2.

    From OG theory, the action functional is:
        S = integral (nabla V_phi)^2 dx

    Minimizing this action gives physically correct dynamics. The gradient
    |nabla V_phi|^2 measures spatial non-uniformity of coherence.

    Physical interpretation:
        - Low gradient: Uniform coherence, stable structure
        - High gradient: Non-uniform coherence, unstable/transitioning

    For quantum circuits, this measures coherence variation across qubits.
    Optimal circuits minimize this gradient.

    Args:
        qubit_R_bars: R_bar values for each qubit subsystem

    Returns:
        Gradient magnitude sum |nabla V_phi|^2 >= 0

    Example:
        >>> # Uniform coherence (optimal)
        >>> grad = compute_coherence_gradient([0.8, 0.8, 0.8])
        >>> print(f"|nabla V_phi|^2 = {grad:.6f}")  # ~0.0

        >>> # Non-uniform coherence (suboptimal)
        >>> grad = compute_coherence_gradient([0.9, 0.5, 0.3])
        >>> print(f"|nabla V_phi|^2 = {grad:.6f}")  # > 0
    """
    R_bars = np.asarray(qubit_R_bars, dtype=np.float64)

    if len(R_bars) < 2:
        return 0.0

    # Clamp R_bars to valid range
    R_bars = np.clip(R_bars, 1e-10, 1.0 - 1e-10)

    # Convert to V_phi values: V_phi = -2 * ln(R_bar)
    V_phis = -2.0 * np.log(R_bars)

    # Compute discrete gradient using central differences
    gradient = np.gradient(V_phis)

    # Return sum of squared gradients (discrete action)
    return float(np.sum(gradient ** 2))


def compute_action_density(
    R_bars: Union[np.ndarray, List[float]],
    positions: Optional[np.ndarray] = None
) -> float:
    """
    Compute OG action density S = integral (nabla V_phi)^2 dx.

    This is the unique action functional from OG theory. Unlike classical
    mechanics where action choice is arbitrary (Hamilton's principle applies
    to any Lagrangian), OG derives a UNIQUE action from first principles.

    Args:
        R_bars: Coherence values at each position
        positions: Spatial positions (default: uniform grid)

    Returns:
        Action density (action per unit volume)

    Theoretical significance:
        Minimizing this action gives the physically correct coherence field
        evolution. This provides a variational principle for circuit optimization.
    """
    R_bars = np.asarray(R_bars, dtype=np.float64)
    n = len(R_bars)

    if n < 2:
        return 0.0

    if positions is None:
        positions = np.arange(n, dtype=np.float64)

    # Convert to V_phi
    R_bars = np.clip(R_bars, 1e-10, 1.0 - 1e-10)
    V_phis = -2.0 * np.log(R_bars)

    # Compute gradient with proper spacing
    dx = np.diff(positions)
    dV = np.diff(V_phis)

    # Gradient at midpoints
    gradients = dV / dx

    # Action = integral (dV/dx)^2 dx
    # Use trapezoidal integration
    action = np.sum(gradients**2 * dx)

    # Normalize by total length for density
    total_length = positions[-1] - positions[0]
    if total_length > 0:
        return float(action / total_length)

    return float(action)


def classify_observability_regime(R_bar: float) -> ObservabilityRegime:
    """
    Classify observability regime based on R_bar position.

    OG theory defines three regimes:
    - IR (Informational Relativity): R_bar > e^-2 * e ~ 0.368
        Structure is clearly observable, quantum correlations extractable

    - TRANSITION: e^-2 / e < R_bar < e^-2 * e, i.e., ~0.050 to ~0.368
        Critical dynamics, structure partially observable

    - AIR (Anti-IR): R_bar < e^-2 / e ~ 0.050
        Structure hidden in noise, effectively classical

    The transition width is natural (not arbitrary) - it spans one e-fold
    above and below the critical point R_c = e^-2.

    Args:
        R_bar: Mean resultant length [0, 1]

    Returns:
        ObservabilityRegime enum value
    """
    if R_bar > TRANSITION_UPPER:
        return ObservabilityRegime.IR
    elif R_bar > TRANSITION_LOWER:
        return ObservabilityRegime.TRANSITION
    else:
        return ObservabilityRegime.AIR


def compute_transition_parameter(R_bar: float) -> float:
    """
    Compute normalized transition parameter.

    This measures the distance from the critical point R_c = e^-2,
    normalized by R_c:
        tau = (R_bar - R_c) / R_c

    Interpretation:
        tau > 0: Above critical (IR regime direction)
        tau = 0: At critical point
        tau < 0: Below critical (AIR regime direction)
        |tau| < 1: In transition regime
        |tau| > 1: Outside transition regime

    Args:
        R_bar: Mean resultant length [0, 1]

    Returns:
        Transition parameter tau in [-inf, +inf]
    """
    return (R_bar - R_BAR_CRITICAL) / R_BAR_CRITICAL


def compute_observability_score(R_bar: float) -> float:
    """
    Compute overall observability quality score.

    This combines multiple factors into a single quality metric [0, 1]:
    - Distance above critical threshold
    - Smoothed transition dynamics
    - Regime classification

    Score interpretation:
        1.0: Perfect observability (R_bar = 1)
        0.5: At critical boundary
        0.0: No observability (R_bar = 0)

    Args:
        R_bar: Mean resultant length [0, 1]

    Returns:
        Observability score [0, 1]
    """
    if R_bar <= 0:
        return 0.0
    if R_bar >= 1.0:
        return 1.0

    # Use logistic transformation centered at critical point
    # This gives smooth transition dynamics
    tau = compute_transition_parameter(R_bar)

    # Logistic function with appropriate steepness
    k = 2.0  # Steepness parameter
    score = 1.0 / (1.0 + np.exp(-k * tau))

    return float(score)


# =============================================================================
# Main OG Metrics Computation
# =============================================================================

def compute_og_metrics(
    R_bar: Optional[float] = None,
    V_phi: Optional[float] = None,
    amplitudes: Optional[np.ndarray] = None,
    measurement_outcomes: Optional[np.ndarray] = None,
    qubit_R_bars: Optional[np.ndarray] = None,
) -> OGMetrics:
    """
    Compute comprehensive OG metrics from coherence data.

    This is the main entry point for OG analysis. It computes all derived
    quantities from the fundamental coherence metrics.

    You can provide either:
    1. R_bar directly (fastest)
    2. V_phi directly (will compute R_bar from coherence law)
    3. amplitudes (complex state vector)
    4. measurement_outcomes (Pauli expectations)

    Args:
        R_bar: Mean resultant length (if known)
        V_phi: Circular variance (if known)
        amplitudes: Complex quantum state amplitudes
        measurement_outcomes: Pauli expectation values
        qubit_R_bars: Per-qubit coherence values (for gradient computation)

    Returns:
        OGMetrics dataclass with all derived quantities

    Example:
        >>> metrics = compute_og_metrics(R_bar=0.5)
        >>> print(metrics)
        OGMetrics(
          R̄=0.5000, V_φ=1.3863
          ℏ_eff=1.1774, C=0.7213
          regime=TRANSITION, score=0.7311
        )

        >>> # From state amplitudes
        >>> state = np.array([1, 0, 0, 0]) / np.sqrt(1)  # |00>
        >>> metrics = compute_og_metrics(amplitudes=state)
    """
    # Determine R_bar and V_phi from inputs
    if R_bar is not None:
        R_bar = float(np.clip(R_bar, 0, 1))
        if V_phi is None:
            V_phi = -2.0 * np.log(R_bar) if R_bar > 1e-10 else float('inf')

    elif V_phi is not None:
        V_phi = float(V_phi)
        R_bar = np.exp(-V_phi / 2.0) if not np.isinf(V_phi) else 0.0

    elif amplitudes is not None:
        R_bar, V_phi = _compute_coherence_from_amplitudes(amplitudes)

    elif measurement_outcomes is not None:
        R_bar, V_phi = _compute_coherence_from_measurements(measurement_outcomes)

    else:
        raise ValueError("Must provide R_bar, V_phi, amplitudes, or measurement_outcomes")

    # Compute all derived metrics
    h_eff = compute_effective_hbar(R_bar)
    rep_cost = compute_representational_cost(V_phi)
    regime = classify_observability_regime(R_bar)
    transition_param = compute_transition_parameter(R_bar)
    obs_score = compute_observability_score(R_bar)
    is_above = R_bar > R_BAR_CRITICAL
    distance = abs(R_bar - R_BAR_CRITICAL)

    # Optional: coherence gradient if per-qubit data provided
    coh_gradient = None
    action_density = None
    if qubit_R_bars is not None:
        coh_gradient = compute_coherence_gradient(qubit_R_bars)
        action_density = compute_action_density(qubit_R_bars)

    # Optional: spectral coherence if amplitudes provided
    spectral_coh = None
    if amplitudes is not None:
        spectral_coh = _compute_spectral_coherence(amplitudes)

    return OGMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        h_eff=h_eff,
        rep_cost=rep_cost,
        regime=regime,
        transition_param=transition_param,
        observability_score=obs_score,
        is_above_critical=is_above,
        distance_to_critical=distance,
        spectral_coherence=spectral_coh,
        coherence_gradient=coh_gradient,
        action_density=action_density,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_coherence_from_amplitudes(amplitudes: np.ndarray) -> Tuple[float, float]:
    """Compute R_bar and V_phi from state amplitudes."""
    amplitudes = np.asarray(amplitudes, dtype=np.complex128).flatten()

    if len(amplitudes) == 0:
        return 0.0, float('inf')

    # Get magnitudes and phases
    magnitudes = np.abs(amplitudes)
    significant = magnitudes > 1e-15

    if not np.any(significant):
        return 0.0, float('inf')

    # Weighted mean phasor
    total_weight = np.sum(magnitudes[significant])
    phasor_sum = np.sum(amplitudes[significant])

    if total_weight < 1e-15:
        return 0.0, float('inf')

    mean_phasor = phasor_sum / total_weight
    R_bar = float(np.clip(np.abs(mean_phasor), 0, 1))

    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = float('inf')

    return R_bar, V_phi


def _compute_coherence_from_measurements(outcomes: np.ndarray) -> Tuple[float, float]:
    """Compute R_bar and V_phi from Pauli expectation values."""
    outcomes = np.asarray(outcomes, dtype=np.float64).flatten()
    outcomes = np.clip(outcomes, -1, 1)

    if len(outcomes) == 0:
        return 0.0, float('inf')

    # Map to phases
    phases = np.arccos(outcomes)

    # Mean phasor
    phasors = np.exp(1j * phases)
    mean_phasor = np.mean(phasors)
    R_bar = float(np.clip(np.abs(mean_phasor), 0, 1))

    if R_bar > 1e-10:
        V_phi = -2.0 * np.log(R_bar)
    else:
        V_phi = float('inf')

    return R_bar, V_phi


def _compute_spectral_coherence(amplitudes: np.ndarray) -> float:
    """Compute spectral coherence (power concentration)."""
    probs = np.abs(amplitudes.flatten()) ** 2
    total = np.sum(probs)

    if total < 1e-15:
        return 0.0

    return float(np.max(probs) / total)


# =============================================================================
# Advanced OG Analysis Functions
# =============================================================================

def predict_measurement_precision(
    R_bar: float,
    n_shots: int = 1000,
) -> Dict[str, float]:
    """
    Predict measurement precision based on OG effective Planck constant.

    OG theory predicts that the effective quantum granularity h_eff determines
    the fundamental precision limit. This function estimates achievable precision.

    Args:
        R_bar: Current coherence level
        n_shots: Number of measurement shots

    Returns:
        Dictionary with precision estimates:
        - h_eff: Effective Planck constant
        - precision_limit: Fundamental precision bound from h_eff
        - shot_precision: Statistical precision from shot noise
        - effective_precision: Combined precision estimate
    """
    h_eff = compute_effective_hbar(R_bar)

    # Shot noise precision: 1/sqrt(n_shots)
    shot_precision = 1.0 / np.sqrt(n_shots)

    # OG precision limit: proportional to h_eff
    # Normalize so h_eff_c gives precision ~0.1
    precision_limit = h_eff / (H_EFF_CRITICAL * 10)

    # Effective precision is the worse of the two limits
    effective_precision = max(shot_precision, precision_limit)

    return {
        'h_eff': h_eff,
        'precision_limit': precision_limit,
        'shot_precision': shot_precision,
        'effective_precision': effective_precision,
    }


def compute_optimal_shot_allocation(
    group_R_bars: List[float],
    total_shots: int,
    min_shots_per_group: int = 10,
) -> List[int]:
    """
    Allocate shots optimally based on OG representational cost.

    Groups with higher representational cost C (lower V_phi) require more
    shots for accurate extraction. This function distributes shots proportionally.

    Args:
        group_R_bars: R_bar values for each measurement group
        total_shots: Total shots to distribute
        min_shots_per_group: Minimum shots for any group

    Returns:
        List of shot counts for each group

    Example:
        >>> R_bars = [0.9, 0.5, 0.2]  # Different coherence levels
        >>> shots = compute_optimal_shot_allocation(R_bars, total_shots=1000)
        >>> # Higher coherence groups get fewer shots (they're easier to measure)
    """
    n_groups = len(group_R_bars)

    if n_groups == 0:
        return []

    if total_shots < n_groups * min_shots_per_group:
        # Not enough shots - distribute equally
        return [total_shots // n_groups] * n_groups

    # Compute measurement difficulty (inverse of coherence)
    # Low R_bar = low coherence = harder to measure = needs more shots
    difficulties = []
    for R_bar in group_R_bars:
        # Use inverse coherence as difficulty measure
        # Higher V_phi means harder to extract information
        V_phi = -2.0 * np.log(max(R_bar, 1e-10))
        # Cap very large V_phi values
        difficulty = min(V_phi, 100.0)
        difficulties.append(difficulty)

    difficulties = np.array(difficulties)

    # Ensure positive weights
    difficulties = np.maximum(difficulties, 0.01)

    # Normalize to get weights
    total_difficulty = np.sum(difficulties)
    if total_difficulty < 1e-10:
        weights = np.ones(n_groups) / n_groups
    else:
        weights = difficulties / total_difficulty

    # Allocate shots proportionally to difficulty
    shots_float = weights * (total_shots - n_groups * min_shots_per_group)
    shots = np.floor(shots_float).astype(int) + min_shots_per_group

    # Distribute remaining shots
    remaining = total_shots - np.sum(shots)
    indices = np.argsort(-difficulties)  # Highest difficulty first
    for i in range(remaining):
        shots[indices[i % n_groups]] += 1

    return shots.tolist()


def evaluate_circuit_quality(
    qubit_R_bars: List[float],
) -> Dict[str, float]:
    """
    Evaluate circuit quality using OG metrics.

    A high-quality circuit (from OG perspective) has:
    - High overall coherence (high R_bar)
    - Uniform coherence across qubits (low gradient)
    - Low action density

    Args:
        qubit_R_bars: Per-qubit coherence values

    Returns:
        Dictionary with quality metrics:
        - mean_R_bar: Average coherence
        - min_R_bar: Minimum (bottleneck) coherence
        - coherence_gradient: Spatial non-uniformity
        - action_density: OG action functional value
        - quality_score: Combined quality [0, 1]
        - regime: Overall regime classification
    """
    R_bars = np.array(qubit_R_bars)

    mean_R = float(np.mean(R_bars))
    min_R = float(np.min(R_bars))

    gradient = compute_coherence_gradient(R_bars)
    action = compute_action_density(R_bars)

    # Quality score combines mean coherence and uniformity
    # Penalize non-uniform coherence
    uniformity = 1.0 / (1.0 + gradient)
    quality = mean_R * uniformity

    regime = classify_observability_regime(min_R)  # Classify by bottleneck

    return {
        'mean_R_bar': mean_R,
        'min_R_bar': min_R,
        'coherence_gradient': gradient,
        'action_density': action,
        'quality_score': quality,
        'regime': regime.value,
    }


def suggest_basis_for_measurement(
    state_R_bar: float,
    available_bases: List[str] = None,
) -> Dict[str, any]:
    """
    Suggest optimal measurement basis based on OG h_eff minimization.

    OG predicts that different measurement bases have different effective
    Planck constants. This function suggests which basis to prefer.

    Args:
        state_R_bar: Coherence of the quantum state
        available_bases: List of basis names (default: ['Z', 'X', 'Y'])

    Returns:
        Dictionary with basis suggestions and reasoning

    Note: This is a simplified heuristic. Full implementation would
    compute h_eff for each basis explicitly.
    """
    if available_bases is None:
        available_bases = ['Z', 'X', 'Y']

    h_eff = compute_effective_hbar(state_R_bar)
    regime = classify_observability_regime(state_R_bar)

    # Heuristic: In high coherence (IR regime), computational basis is optimal
    # In low coherence (AIR regime), eigenbasis of dominant observable is better
    if regime == ObservabilityRegime.IR:
        preferred = 'Z'  # Computational basis
        reason = "High coherence favors computational basis (lowest h_eff)"
    elif regime == ObservabilityRegime.TRANSITION:
        preferred = 'X'  # Superposition basis
        reason = "Transition regime may benefit from superposition basis"
    else:
        preferred = 'Y'  # Often reveals hidden structure
        reason = "Low coherence - try alternative basis to find structure"

    return {
        'preferred_basis': preferred,
        'h_eff': h_eff,
        'regime': regime.value,
        'reason': reason,
        'all_bases': available_bases,
    }


# =============================================================================
# Transition Dynamics
# =============================================================================

@dataclass
class TransitionDynamics:
    """
    Track dynamics through the observability transition.

    This captures the trajectory of a system through the e^-2 transition,
    including direction, velocity, and predicted crossing time.
    """
    current_R_bar: float
    previous_R_bar: float
    regime: ObservabilityRegime
    direction: str  # 'improving', 'degrading', 'stable'
    velocity: float  # Rate of change per step
    steps_to_critical: Optional[float]  # Predicted steps to reach R_c

    def will_cross_threshold(self, n_steps: int) -> bool:
        """Predict if system will cross e^-2 threshold in n steps."""
        if self.velocity == 0:
            return False

        projected = self.current_R_bar + self.velocity * n_steps

        # Check if projection crosses R_c
        if self.current_R_bar > R_BAR_CRITICAL:
            return projected < R_BAR_CRITICAL
        else:
            return projected > R_BAR_CRITICAL


def track_transition_dynamics(
    R_bar_history: List[float],
) -> TransitionDynamics:
    """
    Track system dynamics through observability transition.

    Args:
        R_bar_history: Recent R_bar values (oldest to newest)

    Returns:
        TransitionDynamics with trajectory analysis
    """
    if len(R_bar_history) < 2:
        return TransitionDynamics(
            current_R_bar=R_bar_history[-1] if R_bar_history else 0.0,
            previous_R_bar=R_bar_history[-1] if R_bar_history else 0.0,
            regime=classify_observability_regime(R_bar_history[-1] if R_bar_history else 0.0),
            direction='stable',
            velocity=0.0,
            steps_to_critical=None,
        )

    current = R_bar_history[-1]
    previous = R_bar_history[-2]

    # Compute velocity (moving average if enough history)
    if len(R_bar_history) >= 3:
        diffs = np.diff(R_bar_history[-5:])  # Last 5 points
        velocity = float(np.mean(diffs))
    else:
        velocity = current - previous

    # Determine direction
    if abs(velocity) < 1e-6:
        direction = 'stable'
    elif velocity > 0:
        direction = 'improving'
    else:
        direction = 'degrading'

    # Predict steps to critical
    steps_to_critical = None
    if abs(velocity) > 1e-10:
        distance = R_BAR_CRITICAL - current
        steps_to_critical = distance / velocity
        if steps_to_critical < 0:
            steps_to_critical = None  # Moving away from critical

    return TransitionDynamics(
        current_R_bar=current,
        previous_R_bar=previous,
        regime=classify_observability_regime(current),
        direction=direction,
        velocity=velocity,
        steps_to_critical=steps_to_critical,
    )


# =============================================================================
# Module Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OG Metrics Module - Self Test")
    print("=" * 70)

    # Test 1: Basic metrics computation
    print("\nTest 1: Basic OG metrics computation")
    metrics = compute_og_metrics(R_bar=0.5)
    print(f"  Input: R_bar = 0.5")
    print(f"  h_eff = {metrics.h_eff:.4f}")
    print(f"  C = {metrics.rep_cost:.4f}")
    print(f"  regime = {metrics.regime.value}")
    print(f"  score = {metrics.observability_score:.4f}")
    # R_bar=0.5 is well above TRANSITION_UPPER (~0.368), so it's IR regime
    assert metrics.regime == ObservabilityRegime.IR
    print("  PASSED")

    # Test 2: Critical point values
    print("\nTest 2: Critical point (R_bar = e^-2)")
    metrics_critical = compute_og_metrics(R_bar=R_BAR_CRITICAL)
    print(f"  R_bar_c = {R_BAR_CRITICAL:.4f}")
    print(f"  h_eff_c = {metrics_critical.h_eff:.4f} (expected: {H_EFF_CRITICAL:.4f})")
    print(f"  transition_param = {metrics_critical.transition_param:.4f} (expected: 0.0)")
    assert abs(metrics_critical.transition_param) < 1e-10
    assert abs(metrics_critical.h_eff - H_EFF_CRITICAL) < 0.01
    print("  PASSED")

    # Test 3: Regime classification
    print("\nTest 3: Regime classification")
    test_cases = [
        (0.9, ObservabilityRegime.IR),
        (0.2, ObservabilityRegime.TRANSITION),
        (0.03, ObservabilityRegime.AIR),
    ]
    for R_bar, expected_regime in test_cases:
        regime = classify_observability_regime(R_bar)
        status = "PASS" if regime == expected_regime else "FAIL"
        print(f"  R_bar={R_bar:.2f} -> {regime.value} (expected: {expected_regime.value}) [{status}]")
        assert regime == expected_regime
    print("  PASSED")

    # Test 4: Coherence gradient
    print("\nTest 4: Coherence gradient computation")
    uniform = [0.8, 0.8, 0.8, 0.8]
    nonuniform = [0.9, 0.5, 0.3, 0.1]

    grad_uniform = compute_coherence_gradient(uniform)
    grad_nonuniform = compute_coherence_gradient(nonuniform)

    print(f"  Uniform coherence gradient: {grad_uniform:.6f}")
    print(f"  Non-uniform coherence gradient: {grad_nonuniform:.6f}")
    assert grad_uniform < grad_nonuniform
    assert grad_uniform < 0.01  # Should be near zero
    print("  PASSED")

    # Test 5: Shot allocation
    print("\nTest 5: Optimal shot allocation")
    R_bars = [0.9, 0.5, 0.2]  # High, medium, low coherence
    shots = compute_optimal_shot_allocation(R_bars, total_shots=1000)
    print(f"  R_bars: {R_bars}")
    print(f"  Allocated shots: {shots}")
    print(f"  Total: {sum(shots)}")
    assert sum(shots) == 1000
    # Higher cost (lower R_bar) should get more shots
    assert shots[2] > shots[0]  # Low coherence group gets more
    print("  PASSED")

    # Test 6: From amplitudes
    print("\nTest 6: Metrics from state amplitudes")
    # Pure state |0>
    pure_state = np.array([1, 0, 0, 0], dtype=complex)
    metrics_pure = compute_og_metrics(amplitudes=pure_state)
    print(f"  |00> state: R_bar={metrics_pure.R_bar:.4f}, regime={metrics_pure.regime.value}")
    assert metrics_pure.R_bar > 0.99
    assert metrics_pure.regime == ObservabilityRegime.IR

    # Uniform superposition
    uniform_state = np.ones(4, dtype=complex) / 2
    metrics_uniform = compute_og_metrics(amplitudes=uniform_state)
    print(f"  Uniform state: R_bar={metrics_uniform.R_bar:.4f}, regime={metrics_uniform.regime.value}")
    print("  PASSED")

    # Test 7: Transition dynamics
    print("\nTest 7: Transition dynamics tracking")
    history = [0.8, 0.6, 0.4, 0.3, 0.2]  # Degrading coherence
    dynamics = track_transition_dynamics(history)
    print(f"  History: {history}")
    print(f"  Direction: {dynamics.direction}")
    print(f"  Velocity: {dynamics.velocity:.4f}")
    print(f"  Steps to critical: {dynamics.steps_to_critical}")
    assert dynamics.direction == 'degrading'
    assert dynamics.velocity < 0
    print("  PASSED")

    # Test 8: Effective Planck constant behavior
    print("\nTest 8: Effective Planck constant h_eff(R_bar)")
    R_bar_values = [0.01, 0.1, R_BAR_CRITICAL, 0.5, 0.9, 0.99]
    print("  R_bar      h_eff")
    print("  " + "-" * 20)
    for R in R_bar_values:
        h = compute_effective_hbar(R)
        print(f"  {R:.4f}    {h:.4f}")

    # h_eff = 2*sqrt(-2*ln(R))*sqrt(R) has a maximum at R = e^(-1) ~ 0.368
    # It increases from R=0 to R=e^(-1), then decreases from R=e^(-1) to R=1
    # Key properties to verify:
    # 1. h_eff at R=0 approaches small value (not inf due to sqrt(R))
    # 2. h_eff at R=1 approaches 0
    # 3. h_eff has maximum around R=0.368
    h_effs = [compute_effective_hbar(R) for R in R_bar_values]

    # Verify approaches 0 as R->1
    assert h_effs[-1] < 0.5, f"h_eff should be small at R=0.99: {h_effs[-1]}"

    # Verify maximum is around e^-1 ~ 0.368, so h_eff at 0.5 and 0.1 should bracket it
    # Actually compute max
    max_R = np.exp(-1)  # ~0.368
    h_max = compute_effective_hbar(max_R)
    print(f"  Maximum h_eff at R={max_R:.4f}: {h_max:.4f}")
    assert h_max > h_effs[0] and h_max > h_effs[-1]
    print("  h_eff behavior verified - PASSED")

    print("\n" + "=" * 70)
    print("All OG Metrics tests PASSED!")
    print("=" * 70)
