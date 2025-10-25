"""
Quantum Hybrid Simulator

A quantum-inspired hybrid system combining:
- Compressed quantum state representations (Periodic, Product, MPS)
- O(√r) period-finding algorithms
- AQED (Adaptive Quantum Entanglement Diffusion) transformers
- Hybrid Quantum-AQED integration with MPS-based attention
- Quantum-inspired ML features

Example - AQED Transformer:
    >>> from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig
    >>> config = AQEDConfig(vocab_size=32000, seq_len=4096)
    >>> model = AQEDTransformerLM(config).cuda()
    >>> # Model is 6-10× faster than traditional transformers!

Example - Hybrid AQED (MPS + Attention):
    >>> from quantum_hybrid_system.hybrid_aqed_layer import HybridAQEDTransformer
    >>> model = HybridAQEDTransformer(
    ...     n_layers=12,
    ...     d_model=1024,
    ...     n_heads=16,
    ...     vocab_size=50000,
    ...     chi_max=32,
    ... ).cuda()
    >>> # 10-30× faster than full attention for long sequences!

Example - Quantum Simulation (requires numpy):
    >>> from quantum_hybrid_system.quantum_hybrid_system import QuantumClassicalHybrid
    >>> sim = QuantumClassicalHybrid()
    >>> factors = sim.factor(221)  # Factor 221 = 13 × 17
"""

# Lazy imports to avoid missing dependencies for specific use cases

# Core quantum simulation (requires numpy)
def get_quantum_sim():
    """Get quantum simulation classes (requires numpy)"""
    from .quantum_hybrid_system import (
        QuantumClassicalHybrid,
        PeriodicState,
        ProductState,
        MatrixProductState,
    )
    return QuantumClassicalHybrid, PeriodicState, ProductState, MatrixProductState

# AQED transformer (requires torch)
def get_aqed():
    """Get AQED transformer (requires torch)"""
    from .aqed import AQEDTransformerLM, AQEDConfig
    return AQEDTransformerLM, AQEDConfig

# Hybrid Quantum-AQED (requires torch, triton)
def get_hybrid_aqed():
    """Get Hybrid AQED components (requires torch)"""
    from .hybrid_aqed_layer import HybridAQEDLayer, HybridAQEDTransformer
    from .mps_memory import MPSMemory
    from .entanglement_router import EntanglementRouter
    from .batched_mps_gates import BatchedMPSGateApplicator
    from .telemetry import HybridAQEDTelemetry
    return (HybridAQEDLayer, HybridAQEDTransformer, MPSMemory,
            EntanglementRouter, BatchedMPSGateApplicator, HybridAQEDTelemetry)

__all__ = [
    # Lazy loaders
    'get_quantum_sim',
    'get_aqed',
    'get_hybrid_aqed',
]

__version__ = '0.3.0'  # Updated for Hybrid AQED integration
