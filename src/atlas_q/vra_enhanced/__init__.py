"""
VRA Enhanced Module
===================

Integration of Vaca Resonance Analysis (VRA) with ATLAS-Q quantum simulation.

VRA is a coherence-based spectral framework that can reduce quantum measurement
requirements through classical preprocessing.

Key Features:
- Period finding with 29-42% shot reduction (validated)
- VQE Hamiltonian grouping with 2350× variance reduction
- Coherence-based MPS truncation guidance

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

__all__ = [
    'multiplicative_order',
    'compute_averaged_spectrum',
    'find_period_candidates',
    'vra_enhanced_period_finding',
    'vra_preprocess_period',
    'estimate_shot_reduction',
]

__version__ = '0.1.0'
