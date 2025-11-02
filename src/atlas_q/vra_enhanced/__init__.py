"""
VRA Enhanced Module
===================

Integration of Vaca Resonance Analysis (VRA) with ATLAS-Q quantum simulation.

VRA is a coherence-based spectral framework that can reduce quantum measurement
requirements through classical preprocessing.

Key Features:
- Period finding with 29-42% shot reduction (validated)
- VQE Hamiltonian grouping with 2-60× variance reduction (proof-of-concept)
- Coherence-based correlation analysis for optimal measurement strategies

References:
- VRA Project: https://github.com/followthesapper/VRA
- Coherence Law: C = exp(-V_φ/2), threshold at e^-2 ≈ 0.135
"""

from .core import (
    multiplicative_order,
    compute_averaged_spectrum,
    find_period_candidates,
)

from .qpe_bridge import (
    vra_enhanced_period_finding,
    vra_preprocess_period,
    estimate_shot_reduction,
)

from .vqe_grouping import (
    vra_hamiltonian_grouping,
    estimate_pauli_coherence_matrix,
    compute_Q_GLS,
    GroupingResult,
    pauli_commutes,
    check_group_commutativity,
    group_by_variance_minimization,
    allocate_shots_neyman,
)

__all__ = [
    # Period finding
    'multiplicative_order',
    'compute_averaged_spectrum',
    'find_period_candidates',
    'vra_enhanced_period_finding',
    'vra_preprocess_period',
    'estimate_shot_reduction',
    # VQE grouping
    'vra_hamiltonian_grouping',
    'estimate_pauli_coherence_matrix',
    'compute_Q_GLS',
    'GroupingResult',
    'group_by_variance_minimization',
    'allocate_shots_neyman',
    # Commutativity utilities
    'pauli_commutes',
    'check_group_commutativity',
]

__version__ = '0.3.0'
