"""
Triton GPU Kernels for Quantum Hybrid System

Custom optimized kernels for:
- Modular exponentiation (period finding)
- MPS tensor contractions
- IR coherence computation (NEW: v0.2.0)
- OG (Observability Geometry) metrics (NEW: v0.3.0)

Author: ATLAS-Q Development Team
Date: December 2025
"""

from .modpow import (
    batched_modpow_check_triton,
    batched_modpow_triton,
    benchmark_modpow_implementations,
)

# IR-Enhanced Coherence Kernels (v0.2.0)
from .ir_coherence import (
    E2_THRESHOLD,
    coherence_aware_truncation_triton,
    compute_relational_matrix_triton,
    compute_response_coherence_triton,
    compute_spectral_coherence_triton,
    ir_coherence_metrics,
)

# OG (Observability Geometry) Kernels (v0.3.0)
from .og_kernels import (
    # Core OG computations
    compute_effective_hbar_triton,
    compute_representational_cost_triton,
    classify_regime_triton,
    compute_coherence_gradient_triton,
    compute_observability_score_triton,
    # Fused metrics kernel
    compute_og_metrics_from_amplitudes_triton,
    # Action density
    compute_action_density_triton,
    # Constants
    E_NEG_2,
    E_NEG_1,
)

__all__ = [
    # Modpow kernels
    'batched_modpow_check_triton',
    'batched_modpow_triton',
    'benchmark_modpow_implementations',
    # IR coherence kernels
    'E2_THRESHOLD',
    'compute_response_coherence_triton',
    'compute_spectral_coherence_triton',
    'compute_relational_matrix_triton',
    'coherence_aware_truncation_triton',
    'ir_coherence_metrics',
    # OG metrics kernels (NEW v0.3.0)
    'compute_effective_hbar_triton',
    'compute_representational_cost_triton',
    'classify_regime_triton',
    'compute_coherence_gradient_triton',
    'compute_observability_score_triton',
    'compute_og_metrics_from_amplitudes_triton',
    'compute_action_density_triton',
    'E_NEG_2',
    'E_NEG_1',
]

__version__ = '0.3.0'
