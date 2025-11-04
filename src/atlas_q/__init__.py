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
        MatrixProductState,
        PeriodicState,
        ProductState,
        QuantumClassicalHybrid,
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
    from .diagnostics import MPSStatistics, bond_entropy_from_S, effective_rank, spectral_gap
    from .linalg_robust import condition_number, robust_qr, robust_svd
    from .truncation import check_entropy_sanity, choose_rank_from_sigma, compute_global_error_bound
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
        NoiseChannel,
        NoiseModel,
        StochasticNoiseApplicator,
        choi_to_kraus,
        kraus_to_choi,
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
        HybridSimulator,
        StabilizerSimulator,
        StabilizerState,
        is_clifford_gate,
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
    from .mpo_ops import MPO, MPOBuilder, apply_mpo_to_mps, correlation_function, expectation_value
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
    from .tdvp import TDVP1Site, TDVP2Site, TDVPConfig, run_tdvp
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
        QAOA,
        VQE,
        HardwareEfficientAnsatz,
        QAOAAnsatz,
        VQEConfig,
        build_molecular_hamiltonian,
    )
    return {
        'VQE': VQE,
        'QAOA': QAOA,
        'VQEConfig': VQEConfig,
        'HardwareEfficientAnsatz': HardwareEfficientAnsatz,
        'QAOAAnsatz': QAOAAnsatz,
        'build_molecular_hamiltonian': build_molecular_hamiltonian,
    }

# Grover's algorithm (requires torch)
def get_grover():
    """Get Grover's quantum search algorithm"""
    from .grover import (
        BitmapOracle,
        DiffusionOperator,
        FunctionOracle,
        GroverConfig,
        GroverSearch,
        OracleBase,
        calculate_grover_iterations,
        grover_search,
    )
    return {
        'GroverSearch': GroverSearch,
        'GroverConfig': GroverConfig,
        'OracleBase': OracleBase,
        'FunctionOracle': FunctionOracle,
        'BitmapOracle': BitmapOracle,
        'DiffusionOperator': DiffusionOperator,
        'grover_search': grover_search,
        'calculate_grover_iterations': calculate_grover_iterations,
    }

# cuQuantum backend (requires cuquantum, optional)
def get_cuquantum():
    """Get cuQuantum acceleration backend (optional)"""
    from .cuquantum_backend import (
        CuQuantumBackend,
        CuQuantumConfig,
        CuStateVecBackend,
        benchmark_backend,
        get_cuquantum_version,
        is_cuquantum_available,
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
        CircuitCutter,
        CircuitPartition,
        CouplingGraph,
        CutPoint,
        CuttingConfig,
        MinCutPartitioner,
        visualize_entanglement_heatmap,
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
        ChiScheduler,
        Layout2D,
        MappingConfig,
        Planar2DCircuit,
        SnakeMapper,
        SWAPSynthesizer,
        Topology,
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
        DistMode,
        DistributedConfig,
        DistributedMPS,
        MPSPartition,
        launch_distributed_simulation,
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
        PEPS,
        ContractionStrategy,
        PatchPEPS,
        PEPSConfig,
        PEPSTensor,
        benchmark_peps_vs_mps,
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

# Coherence module (requires torch, numpy)
def get_coherence():
    """Get coherence-aware quantum computing tools"""
    from .coherence import (
        CoherenceClassification,
        CoherenceMetrics,
        adaptive_vra_decision,
        classify_go_no_go,
        classify_with_history,
        coherence_from_counts,
        compute_coherence,
        compute_pauli_expectation,
        group_paulis_qwc,
    )
    from .coherence_aware_vqe import (
        CoherenceAwareVQE,
        CoherenceAwareVQEResult,
        coherence_aware_vqe,
    )
    return {
        # Metrics
        'CoherenceMetrics': CoherenceMetrics,
        'CoherenceClassification': CoherenceClassification,
        'compute_coherence': compute_coherence,
        'coherence_from_counts': coherence_from_counts,
        # Classification
        'classify_go_no_go': classify_go_no_go,
        'classify_with_history': classify_with_history,
        'adaptive_vra_decision': adaptive_vra_decision,
        # Utilities
        'compute_pauli_expectation': compute_pauli_expectation,
        'group_paulis_qwc': group_paulis_qwc,
        # VQE
        'CoherenceAwareVQE': CoherenceAwareVQE,
        'CoherenceAwareVQEResult': CoherenceAwareVQEResult,
        'coherence_aware_vqe': coherence_aware_vqe,
    }

# Direct module access (preferred, simpler API)
# These are lazily loaded when first accessed
from . import (
    adaptive_mps,
    circuit_cutting,
    coherence,
    distributed_mps,
    grover,
    mpo_ops,
    mps_pytorch,
    noise_models,
    peps,
    planar_2d,
    stabilizer_backend,
    tdvp,
    vqe_qaoa,
)

# Direct imports for coherence-aware VQE (new feature)
try:
    from .coherence import CoherenceMetrics, classify_go_no_go, compute_coherence
    from .coherence_aware_vqe import CoherenceAwareVQE, CoherenceAwareVQEResult, coherence_aware_vqe
except ImportError:
    CoherenceAwareVQE = None
    CoherenceAwareVQEResult = None
    coherence_aware_vqe = None
    CoherenceMetrics = None
    compute_coherence = None
    classify_go_no_go = None

__all__ = [
    # Direct module imports (PREFERRED - use these!)
    'mpo_ops',
    'tdvp',
    'vqe_qaoa',
    'grover',
    'adaptive_mps',
    'mps_pytorch',
    'noise_models',
    'stabilizer_backend',
    'circuit_cutting',
    'planar_2d',
    'distributed_mps',
    'peps',
    'coherence',  # NEW: Coherence-aware quantum computing
    # Direct class imports (for backwards compatibility)
    'QuantumClassicalHybrid',
    # Coherence-aware VQE (NEW)
    'CoherenceAwareVQE',
    'CoherenceAwareVQEResult',
    'coherence_aware_vqe',
    'CoherenceMetrics',
    'compute_coherence',
    'classify_go_no_go',
    # Lazy loaders (legacy compatibility - getters return dicts)
    'get_quantum_sim',
    'get_mps_pytorch',
    'get_adaptive_mps',
    'get_qih_tools',
    'get_noise_models',
    'get_stabilizer',
    'get_mpo_ops',
    'get_tdvp',
    'get_vqe_qaoa',
    'get_grover',
    'get_cuquantum',
    'get_circuit_cutting',
    'get_planar_2d',
    'get_distributed_mps',
    'get_peps',
    'get_coherence',  # NEW
]

# Direct imports for backwards compatibility
try:
    from .quantum_hybrid_system import (
        GPUAccelerator,
        MatrixProductState,
        PeriodicState,
        ProductState,
        QuantumClassicalHybrid,
    )
except ImportError:
    QuantumClassicalHybrid = None
    PeriodicState = None
    ProductState = None
    MatrixProductState = None
    GPUAccelerator = None

__version__ = '0.6.3'  # Coherence-Aware VQE + VRA Integration (Nov 2025)
