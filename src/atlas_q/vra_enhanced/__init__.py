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
    compute_averaged_spectrum,
    find_period_candidates,
    multiplicative_order,
)
from .gradient_grouping import (
    GradientGroupingResult,
    estimate_gradient_coherence_matrix,
    group_parameters_by_variance,
    parameter_shift_gradient_vra,
    vra_gradient_grouping,
)
from .qaoa_grouping import (
    QAOAGroupingResult,
    check_group_commutativity_edges,
    edges_commute,
    estimate_edge_coherence_matrix,
    group_edges_by_commutativity,
    vra_qaoa_grouping,
)
from .qpe_bridge import (
    estimate_shot_reduction,
    vra_enhanced_period_finding,
    vra_preprocess_period,
)
from .shadow_tomography import (
    ShadowSamplingResult,
    vra_shadow_sampling,
)
from .state_tomography import (
    TomographyStrategy,
    generate_pauli_basis,
    tomography_measurement_groups,
    vra_state_tomography,
)
from .tdvp_observables import (
    TDVPObservableGroupingResult,
    vra_tdvp_observable_grouping,
)
from .vqe_grouping import (
    GroupingResult,
    allocate_shots_neyman,
    check_group_commutativity,
    compute_Q_GLS,
    estimate_pauli_coherence_matrix,
    group_by_variance_minimization,
    pauli_commutes,
    vra_hamiltonian_grouping,
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
    # QAOA grouping
    'vra_qaoa_grouping',
    'QAOAGroupingResult',
    'edges_commute',
    'check_group_commutativity_edges',
    'estimate_edge_coherence_matrix',
    'group_edges_by_commutativity',
    # Gradient grouping
    'vra_gradient_grouping',
    'GradientGroupingResult',
    'parameter_shift_gradient_vra',
    'estimate_gradient_coherence_matrix',
    'group_parameters_by_variance',
    # TDVP observables
    'vra_tdvp_observable_grouping',
    'TDVPObservableGroupingResult',
    # Shadow tomography
    'vra_shadow_sampling',
    'ShadowSamplingResult',
    # State tomography
    'vra_state_tomography',
    'TomographyStrategy',
    'generate_pauli_basis',
    'tomography_measurement_groups',
]

__version__ = '1.0.0'
