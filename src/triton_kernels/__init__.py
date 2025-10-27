"""
Triton GPU Kernels for Quantum Hybrid System

Custom optimized kernels for:
- Modular exponentiation (period finding)
- MPS tensor contractions (future)

Author: Claude Code
Date: October 2025
"""

from .modpow import (
    batched_modpow_check_triton,
    batched_modpow_triton,
    benchmark_modpow_implementations,
)

__all__ = [
    'batched_modpow_check_triton',
    'batched_modpow_triton',
    'benchmark_modpow_implementations',
]

__version__ = '0.1.0'
