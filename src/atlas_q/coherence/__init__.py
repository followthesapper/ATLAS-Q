"""
ATLAS-Q Coherence Module
========================

Coherence-aware quantum computing framework based on Informational Relativity (IR).

This module provides:
- Circular statistics-based coherence tracking (R̄, V_φ)
- GO/NO-GO classification using e^-2 boundary
- Adaptive IR decision logic
- Measurement grouping utilities

Key Components:
    - CoherenceMetrics: Circular statistics metrics
    - compute_coherence(): Calculate R̄ and V_φ from measurements
    - classify_go_no_go(): e^-2 boundary classification
    - adaptive_ir_decision(): Adaptive grouping control

Example:
    >>> from atlas_q.coherence import compute_coherence, classify_go_no_go
    >>> import numpy as np
    >>>
    >>> # Measure Pauli expectations
    >>> outcomes = np.array([0.9, 0.85, 0.88, 0.87])
    >>>
    >>> # Compute coherence
    >>> coherence = compute_coherence(outcomes)
    >>> print(f"R̄ = {coherence.R_bar:.4f}")
    R̄ = 0.8750
    >>>
    >>> # Classify quality
    >>> classification = classify_go_no_go(coherence)
    >>> print(classification)
    [GO] Coherence above e^-2 boundary (R̄=0.875 > 0.135) (confidence: 0.98)

Author: ATLAS-Q Development Team
Date: November 2025
"""

# Classification
from .classification import (
    CoherenceClassification,
    adaptive_ir_decision,
    classify_go_no_go,
    classify_with_history,
)

# Core metrics
from .metrics import (
    CoherenceMetrics,
    coherence_from_counts,
    compute_coherence,
    compute_relational_coherence,
    compute_response_coherence,
    validate_coherence_law,
)

# Utilities
from .utils import (
    compute_pauli_expectation,
    group_paulis_qwc,
    pauli_commute,
    qubit_wise_commute,
)

# OG (Observability Geometry) metrics - Pre-quantum theoretical foundation
from .og_metrics import (
    # Core OG metrics
    OGMetrics,
    ObservabilityRegime,
    compute_og_metrics,
    compute_effective_hbar,
    compute_representational_cost,
    compute_coherence_gradient,
    compute_action_density,
    # Regime classification
    classify_observability_regime,
    compute_transition_parameter,
    compute_observability_score,
    # Advanced analysis
    predict_measurement_precision,
    compute_optimal_shot_allocation,
    evaluate_circuit_quality,
    suggest_basis_for_measurement,
    # Transition dynamics
    TransitionDynamics,
    track_transition_dynamics,
    # Constants
    R_BAR_CRITICAL,
    TRANSITION_LOWER,
    TRANSITION_UPPER,
    H_EFF_CRITICAL,
)

# Version info
__version__ = "0.8.0"  # Updated for OG integration

# Public API
__all__ = [
    # Metrics
    'CoherenceMetrics',
    'compute_coherence',
    'compute_response_coherence',  # Correct L8 placement
    'compute_relational_coherence',  # Full spectral lifting
    'coherence_from_counts',
    'validate_coherence_law',

    # Classification
    'CoherenceClassification',
    'classify_go_no_go',
    'classify_with_history',
    'adaptive_ir_decision',

    # Utilities
    'compute_pauli_expectation',
    'pauli_commute',
    'qubit_wise_commute',
    'group_paulis_qwc',

    # OG (Observability Geometry) - Pre-quantum foundation
    'OGMetrics',
    'ObservabilityRegime',
    'compute_og_metrics',
    'compute_effective_hbar',
    'compute_representational_cost',
    'compute_coherence_gradient',
    'compute_action_density',
    'classify_observability_regime',
    'compute_transition_parameter',
    'compute_observability_score',
    'predict_measurement_precision',
    'compute_optimal_shot_allocation',
    'evaluate_circuit_quality',
    'suggest_basis_for_measurement',
    'TransitionDynamics',
    'track_transition_dynamics',
    'R_BAR_CRITICAL',
    'TRANSITION_LOWER',
    'TRANSITION_UPPER',
    'H_EFF_CRITICAL',
]


# Module-level constants
E2_BOUNDARY = 0.135  # e^-2 threshold for GO/NO-GO classification
COHERENCE_LAW_TOLERANCE = 0.1  # Tolerance for validating R̄ = e^(-V_φ/2)


def get_version():
    """Get module version."""
    return __version__


def get_e2_boundary():
    """Get e^-2 boundary threshold."""
    return E2_BOUNDARY
