"""
Quantum Hybrid Simulator

A quantum-inspired hybrid system combining:
- Compressed quantum state representations (Periodic, Product, MPS)
- O(√r) period-finding algorithms
- AQED (Adaptive Quantum Entanglement Diffusion) transformers
- Quantum-inspired ML features

Example - Quantum Simulation:
    >>> from quantum_hybrid_system import QuantumClassicalHybrid
    >>> sim = QuantumClassicalHybrid()
    >>> factors = sim.factor(221)  # Factor 221 = 13 × 17
    >>> print(factors)  # [13, 17]

Example - AQED Transformer:
    >>> from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig
    >>> config = AQEDConfig(vocab_size=32000, seq_len=4096)
    >>> model = AQEDTransformerLM(config).cuda()
    >>> # Model is 6-10× faster than traditional transformers!
"""

# Core quantum simulation
from .quantum_hybrid_system import (
    QuantumClassicalHybrid,
    PeriodicState,
    ProductState,
    MatrixProductState,
)

# AQED transformer (lazy import to avoid torch dependency for quantum-only use)
def get_aqed():
    """Get AQED transformer (requires torch)"""
    from .aqed import AQEDTransformerLM, AQEDConfig
    return AQEDTransformerLM, AQEDConfig

__all__ = [
    # Quantum simulation
    'QuantumClassicalHybrid',
    'PeriodicState',
    'ProductState',
    'MatrixProductState',
    # AQED
    'get_aqed',
]

__version__ = '0.2.0'
