"""
Triton GPU Kernels for Quantum Hybrid System

Custom optimized kernels for:
- Modular exponentiation (period finding)
- MPS tensor contractions
- IR coherence computation (NEW: v0.2.0)

Author: ATLAS-Q Development Team
Date: October 2025
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

__all__ = [
    # Modpow kernels
    'batched_modpow_check_triton',
    'batched_modpow_triton',
    'benchmark_modpow_implementations',
    # IR coherence kernels (NEW)
    'E2_THRESHOLD',
    'compute_response_coherence_triton',
    'compute_spectral_coherence_triton',
    'compute_relational_matrix_triton',
    'coherence_aware_truncation_triton',
    'ir_coherence_metrics',
]

__version__ = '0.2.0'
