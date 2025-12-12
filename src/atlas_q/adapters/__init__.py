"""
ATLAS-Q Adapters for Qiskit and Cirq

Provides seamless integration with popular quantum frameworks, automatically
leveraging ATLAS-Q's advanced features:
- IR: 5× measurement reduction via automatic observable grouping
- MPS: 626,000× memory compression for large circuits
- GPU: 1.5-3× speedup via Triton kernels
- Stabilizer: 20× speedup for Clifford circuits
- Coherence: Automatic quality validation for VQE results
"""

__all__ = []

try:
    from .qiskit_adapter import ATLASQBackend, ATLASQProvider
    __all__.extend(['ATLASQBackend', 'ATLASQProvider'])
except ImportError:
    pass

try:
    from .cirq_adapter import ATLASQSimulator
    __all__.extend(['ATLASQSimulator'])
except ImportError:
    pass
