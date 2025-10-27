"""
Quantum Hybrid Simulator

A quantum-inspired system for period-finding and factorization:
- Compressed quantum state representations (Periodic, Product, MPS)
- O(√r) period-finding algorithms
- Quantum-inspired ML features
- GPU acceleration via CuPy and Triton
- Adaptive MPS for moderate-to-high entanglement simulation

Example - Quantum Simulation:
    >>> from atlas_q.quantum_hybrid_system import QuantumClassicalHybrid
    >>> sim = QuantumClassicalHybrid()
    >>> factors = sim.factor(221)  # Factor 221 = 13 × 17

Example - Basic MPS for quantum states:
    >>> from atlas_q.mps_pytorch import MatrixProductStatePyTorch
    >>> mps = MatrixProductStatePyTorch(num_qubits=50, bond_dim=16, device='cuda')
    >>> # Simulate 50 qubits with GPU acceleration!

Example - Adaptive MPS for moderate entanglement:
    >>> adaptive = get_adaptive_mps()
    >>> mps = adaptive['AdaptiveMPS'](16, bond_dim=8, eps_bond=1e-6, chi_max_per_bond=64)
    >>> # Apply quantum gates with automatic adaptive truncation!
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

# GPU-accelerated MPS (requires torch)
def get_mps_pytorch():
    """Get PyTorch-based MPS for GPU acceleration (requires torch)"""
    from .mps_pytorch import MatrixProductStatePyTorch
    return MatrixProductStatePyTorch

# Adaptive MPS for moderate-to-high entanglement (requires torch)
def get_adaptive_mps():
    """Get Adaptive MPS for moderate-to-high entanglement (requires torch)"""
    from .adaptive_mps import AdaptiveMPS, DTypePolicy
    from .linalg_robust import robust_svd, robust_qr, condition_number
    from .truncation import choose_rank_from_sigma, compute_global_error_bound, check_entropy_sanity
    from .diagnostics import MPSStatistics, bond_entropy_from_S, effective_rank, spectral_gap
    return {
        'AdaptiveMPS': AdaptiveMPS,
        'DTypePolicy': DTypePolicy,
        'robust_svd': robust_svd,
        'robust_qr': robust_qr,
        'condition_number': condition_number,
        'choose_rank_from_sigma': choose_rank_from_sigma,
        'compute_global_error_bound': compute_global_error_bound,
        'check_entropy_sanity': check_entropy_sanity,
        'MPSStatistics': MPSStatistics,
        'bond_entropy_from_S': bond_entropy_from_S,
        'effective_rank': effective_rank,
        'spectral_gap': spectral_gap,
    }

# Noise models (requires torch, numpy)
def get_noise_models():
    """Get NISQ noise models and channels"""
    from .noise_models import (
        NoiseModel, NoiseChannel, StochasticNoiseApplicator,
        kraus_to_choi, choi_to_kraus
    )
    return {
        'NoiseModel': NoiseModel,
        'NoiseChannel': NoiseChannel,
        'StochasticNoiseApplicator': StochasticNoiseApplicator,
        'kraus_to_choi': kraus_to_choi,
        'choi_to_kraus': choi_to_kraus,
    }

# Stabilizer backend (requires torch, numpy)
def get_stabilizer():
    """Get Clifford/stabilizer fast path simulator"""
    from .stabilizer_backend import (
        StabilizerSimulator, StabilizerState, HybridSimulator,
        is_clifford_gate
    )
    return {
        'StabilizerSimulator': StabilizerSimulator,
        'StabilizerState': StabilizerState,
        'HybridSimulator': HybridSimulator,
        'is_clifford_gate': is_clifford_gate,
    }

# MPO operations (requires torch)
def get_mpo_ops():
    """Get Matrix Product Operator operations"""
    from .mpo_ops import (
        MPO, MPOBuilder, apply_mpo_to_mps, expectation_value,
        correlation_function
    )
    return {
        'MPO': MPO,
        'MPOBuilder': MPOBuilder,
        'apply_mpo_to_mps': apply_mpo_to_mps,
        'expectation_value': expectation_value,
        'correlation_function': correlation_function,
    }

# TDVP time evolution (requires torch)
def get_tdvp():
    """Get Time-Dependent Variational Principle time evolution"""
    from .tdvp import (
        TDVP1Site, TDVP2Site, TDVPConfig, run_tdvp
    )
    return {
        'TDVP1Site': TDVP1Site,
        'TDVP2Site': TDVP2Site,
        'TDVPConfig': TDVPConfig,
        'run_tdvp': run_tdvp,
    }

# VQE/QAOA (requires torch, scipy)
def get_vqe_qaoa():
    """Get Variational Quantum Eigensolver and QAOA"""
    from .vqe_qaoa import (
        VQE, QAOA, VQEConfig, HardwareEfficientAnsatz, QAOAAnsatz,
        build_molecular_hamiltonian
    )
    return {
        'VQE': VQE,
        'QAOA': QAOA,
        'VQEConfig': VQEConfig,
        'HardwareEfficientAnsatz': HardwareEfficientAnsatz,
        'QAOAAnsatz': QAOAAnsatz,
        'build_molecular_hamiltonian': build_molecular_hamiltonian,
    }

# cuQuantum backend (requires cuquantum, optional)
def get_cuquantum():
    """Get cuQuantum acceleration backend (optional)"""
    from .cuquantum_backend import (
        CuQuantumBackend, CuStateVecBackend, CuQuantumConfig,
        is_cuquantum_available, get_cuquantum_version, benchmark_backend
    )
    return {
        'CuQuantumBackend': CuQuantumBackend,
        'CuStateVecBackend': CuStateVecBackend,
        'CuQuantumConfig': CuQuantumConfig,
        'is_cuquantum_available': is_cuquantum_available,
        'get_cuquantum_version': get_cuquantum_version,
        'benchmark_backend': benchmark_backend,
    }

# Circuit cutting (requires torch, numpy)
def get_circuit_cutting():
    """Get circuit cutting and entanglement forging tools"""
    from .circuit_cutting import (
        CircuitCutter, CouplingGraph, MinCutPartitioner, CuttingConfig,
        CutPoint, CircuitPartition, visualize_entanglement_heatmap
    )
    return {
        'CircuitCutter': CircuitCutter,
        'CouplingGraph': CouplingGraph,
        'MinCutPartitioner': MinCutPartitioner,
        'CuttingConfig': CuttingConfig,
        'CutPoint': CutPoint,
        'CircuitPartition': CircuitPartition,
        'visualize_entanglement_heatmap': visualize_entanglement_heatmap,
    }

# 2D/Planar circuits (requires torch, numpy)
def get_planar_2d():
    """Get 2D/planar circuit support"""
    from .planar_2d import (
        Planar2DCircuit, SnakeMapper, SWAPSynthesizer, ChiScheduler,
        Layout2D, Topology, MappingConfig
    )
    return {
        'Planar2DCircuit': Planar2DCircuit,
        'SnakeMapper': SnakeMapper,
        'SWAPSynthesizer': SWAPSynthesizer,
        'ChiScheduler': ChiScheduler,
        'Layout2D': Layout2D,
        'Topology': Topology,
        'MappingConfig': MappingConfig,
    }

# Distributed MPS (requires torch.distributed)
def get_distributed_mps():
    """Get distributed multi-GPU MPS simulator"""
    from .distributed_mps import (
        DistributedMPS, DistributedConfig, DistMode, MPSPartition,
        launch_distributed_simulation
    )
    return {
        'DistributedMPS': DistributedMPS,
        'DistributedConfig': DistributedConfig,
        'DistMode': DistMode,
        'MPSPartition': MPSPartition,
        'launch_distributed_simulation': launch_distributed_simulation,
    }

# PEPS (requires torch)
def get_peps():
    """Get PEPS (Projected Entangled Pair States) 2D tensor networks"""
    from .peps import (
        PEPS, PatchPEPS, PEPSConfig, PEPSTensor, ContractionStrategy,
        benchmark_peps_vs_mps
    )
    return {
        'PEPS': PEPS,
        'PatchPEPS': PatchPEPS,
        'PEPSConfig': PEPSConfig,
        'PEPSTensor': PEPSTensor,
        'ContractionStrategy': ContractionStrategy,
        'benchmark_peps_vs_mps': benchmark_peps_vs_mps,
    }

# Quantum-inspired ML tools (requires numpy, optional torch)
def get_qih_tools():
    """Get quantum-inspired ML tools"""
    from . import tools_qih
    return tools_qih

__all__ = [
    # Lazy loaders - Core
    'get_quantum_sim',
    'get_mps_pytorch',
    'get_adaptive_mps',
    'get_qih_tools',
    # Lazy loaders - New features (v0.5.0)
    'get_noise_models',
    'get_stabilizer',
    'get_mpo_ops',
    'get_tdvp',
    'get_vqe_qaoa',
    'get_cuquantum',
    'get_circuit_cutting',
    'get_planar_2d',
    'get_distributed_mps',
    'get_peps',
]

__version__ = '0.6.1'  # Import fixes for PyPI users (Oct 2025)
