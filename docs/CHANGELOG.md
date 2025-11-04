# Changelog

All notable changes to ATLAS-Q will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.3] - 2025-11-04

### Added

#### Qiskit & Cirq Adapters - Drop-in Replacement for Popular Frameworks
- **Qiskit Adapter** (`src/atlas_q/adapters/qiskit_adapter.py`): Zero-code-change replacement for Qiskit Aer
  - `ATLASQBackend`: Implements Qiskit BackendV2 interface
  - `ATLASQProvider`: Provider interface for backend discovery
  - Automatic backend selection: Clifford → Stabilizer, >25 qubits → MPS, else → Statevector
  - VRA observable grouping: 5× measurement reduction for VQE Hamiltonians
  - Coherence metrics: Automatic R̄ computation for VQE patterns
  - GPU acceleration: Transparent Triton kernel usage for MPS operations
  - Full Qiskit compatibility: Drop-in replacement for `Aer.get_backend('qasm_simulator')`
- **Cirq Adapter** (`src/atlas_q/adapters/cirq_adapter.py`): Zero-code-change replacement for Cirq simulators
  - `ATLASQSimulator`: Implements Cirq `SimulatesSamples` and `SimulatesExpectationValues`
  - Same automatic optimizations as Qiskit adapter (VRA, MPS, stabilizer, GPU)
  - Parameter sweep support via `run_sweep()`
  - Expectation value computation with VRA grouping
  - Full Cirq compatibility: Drop-in replacement for `cirq.Simulator()`
- **Comprehensive Tests** (`tests/integration/test_qiskit_adapter.py`, `test_cirq_adapter.py`):
  - Bell state, GHZ state, parametric circuits
  - Clifford detection and stabilizer backend activation
  - MPS threshold testing (>25 qubits)
  - VRA measurement compression verification
  - Coherence metric validation for VQE patterns
  - Multi-circuit execution
- **Performance Benchmarks** (`benchmarks/adapter_comparison_benchmark.py`):
  - Qiskit Aer vs ATLAS-Q: Bell states, Clifford circuits, VQE with VRA, large MPS circuits
  - Cirq simulator vs ATLAS-Q: Same comprehensive comparison suite
  - Demonstrates 5× VRA reduction, 20× stabilizer speedup, 626,000× MPS memory efficiency

### Changed
- **Optional Dependencies** (`pyproject.toml`):
  - Added `[qiskit]`: `qiskit>=0.44.0`
  - Added `[cirq]`: `cirq>=1.2.0`
  - Added `[adapters]`: Meta-package for both Qiskit and Cirq
  - Updated `[all]` to include adapters
- **README**: Added prominent "Drop-in Qiskit/Cirq Adapters" section with usage examples
- **Documentation**: Installation instructions for `pip install atlas-quantum[adapters]`

### Performance
- **5× measurement reduction**: Automatic VRA grouping for VQE observables (Qiskit/Cirq)
- **20× Clifford speedup**: Automatic stabilizer backend (Qiskit/Cirq)
- **626,000× memory efficiency**: Automatic MPS for >25 qubits (Qiskit/Cirq)
- **1.5-3× GPU speedup**: Transparent Triton kernels for MPS ops (Qiskit/Cirq)

## [0.6.2] - 2025-11-04

### Added

#### BREAKTHROUGH: Coherence-Aware Quantum Computing Framework
- **World's First Coherence-Aware VQE** (`benchmarks/vra_coherence_aware_hardware_benchmark.py`): Self-diagnostic quantum algorithms
 - Real-time coherence tracking (R̄, V_φ) based on Vaca Resonance Analysis (VRA)
 - Universal GO/NO-GO classifier using e^-2 boundary (R̄ ≈ 0.135)
 - Hardware-validated on IBM Brisbane: H2O achieved R̄=0.988 (near-ideal)
 - VRA grouping reduces measurement overhead by 5× (1086 terms → 219 groups)
 - Circular statistics and Random Matrix Theory integration
 - Critical bug fix: Proper per-term Pauli measurement (3× energy accuracy improvement)
 - Production-scale testing: H2 (4q), LiH (12q), H2O (14q) on real quantum hardware
 - See `COHERENCE_AWARE_VQE_BREAKTHROUGH.md` for complete technical details
- **VRA Enhanced Modules** (`src/atlas_q/vra_enhanced/`): Integration across ATLAS-Q ecosystem
 - `gradient_grouping.py` - Parameter shift rules with VRA measurement compression
 - `qaoa_grouping.py` - Coherence-aware QAOA for combinatorial optimization
 - `shadow_tomography.py` - Adaptive classical shadows with quality monitoring
 - `state_tomography.py` - Full state reconstruction with VRA grouping
 - `tdvp_observables.py` - Time evolution with real-time coherence tracking
- **Comprehensive Documentation**: Full Sphinx documentation for coherence-aware computing
 - User guide: `docs/user_guide/coherence_aware_vqe.rst` (425 lines, production-ready)
 - Tutorial: `docs/user_guide/tutorials/coherence_aware_vqe.rst`
 - Working example: `examples/coherence_aware_vqe_example.py` (350+ lines)
 - Hardware validation summaries with job IDs and provenance
 - Integration examples and best practices
 - Updated README.md with prominent coherence-aware VQE section

#### UCCSD Ansatz for Molecular VQE
- **UCCSD (Unitary Coupled-Cluster Singles and Doubles)** (`ansatz_uccsd.py`): Chemistry-aware variational ansatz
 - OpenFermion integration for fermionic operator generation
 - MPS-compatible implementation (no exponential memory)
 - Pauli string decomposition with `apply_pauli_exp_to_mps()`
 - Hartree-Fock reference state initialization
 - Compatible with VQE for ground state chemistry calculations

#### Quantum Chemistry & Optimization Hamiltonians
- **Molecular Hamiltonian Builder** (`mpo_ops.py`): PySCF integration for quantum chemistry
 - `molecular_hamiltonian_from_specs()` - Build electronic structure Hamiltonians
 - Support for H2, LiH, H2O, and custom geometry strings
 - Jordan-Wigner fermion-to-qubit transformation
 - Compatible with VQE for ground state energy calculations
 - 4/4 tests passing in `test_molecular_hamiltonians.py`
- **MaxCut Hamiltonian Builder** (`mpo_ops.py`): QAOA graph optimization
 - `maxcut_hamiltonian()` - Build MaxCut problem Hamiltonians
 - Weighted and unweighted graph support
 - Automatic edge normalization for undirected graphs
 - Compatible with QAOA for combinatorial optimization
 - 4/4 tests passing in `test_maxcut.py`

#### Advanced Tensor Network Features
- **Circuit Cutting** (`circuit_cutting.py`): Partition large circuits for simulation
 - Coupling graph analysis and entanglement heatmaps
 - Min-cut and spectral partitioning algorithms
 - Classical stitching with variance reduction
 - 7/7 tests passing in `test_circuit_cutting.py`
- **PEPS (Projected Entangled Pair States)** (`peps.py`): 2D tensor networks
 - True 2D representation for shallow quantum circuits
 - Boundary-MPS contraction strategy
 - PatchPEPS for 4×4 and 5×5 grids
 - Single and two-site gate application
 - 10/10 tests passing in `test_peps.py`
- **Distributed MPS** (`distributed_mps.py`): Multi-GPU scaling
 - Bond-wise domain decomposition across GPUs
 - Overlapped communication and computation
 - Checkpoint/restart for long simulations
 - 10/10 tests passing in `test_distributed_mps.py`
- **cuQuantum Backend** (`cuquantum_backend.py`): Optional NVIDIA acceleration
 - cuTensorNet 25.x integration for tensor operations
 - Automatic fallback to PyTorch if unavailable
 - 2-10× speedup on compatible NVIDIA GPUs (requires `cuquantum-python`)
 - 11/11 tests passing in `test_cuquantum.py` (tested with cuQuantum 25.09.1)
 - **Install:** `pip install cuquantum-python` (optional, ~320MB)

### Changed

#### Improved Import System (Better UX)
- **Direct module imports now supported** (`__init__.py`): Pythonic import pattern
 - **New (recommended)**: `from atlas_q import mpo_ops, tdvp, vqe_qaoa`
 - **Legacy (still works)**: `atlas_q.get_mpo_ops()` returns dict
 - Enables IDE autocomplete and type hints
 - Matches standard Python package conventions (like NumPy, PyTorch)
 - Backwards compatible - old getter pattern still supported

## [0.5.0] - 2025-10-26

### Added

#### GPU Acceleration & Triton Kernels
- Custom Triton kernels for MPS gate operations (1.5-3× speedup)
- GPU-optimized tensor contractions with cuBLAS tensor cores
- Modular exponentiation kernels for period-finding
- 77,000+ ops/sec gate throughput achieved

#### Tensor Network Features
- **Noise Models** (`noise_models.py`): Full Kraus operator framework for NISQ simulation
 - Depolarizing, dephasing, amplitude damping, Pauli noise channels
 - Stochastic noise applicator with reproducible seeds
- **Stabilizer Backend** (`stabilizer_backend.py`): Clifford circuit fast path
 - O(n²) complexity via Gottesman-Knill theorem
 - 20× speedup over generic MPS
 - Automatic handoff to MPS for non-Clifford gates
- **MPO Operations** (`mpo_ops.py`): Hamiltonian and observable framework
 - Pre-built Hamiltonians: Ising, Heisenberg, custom spin chains
 - Expectation values and correlation functions
- **TDVP** (`tdvp.py`): Time evolution for Hamiltonian dynamics
 - 1-site (conserves bond dimension) and 2-site (adaptive) variants
 - Krylov subspace methods for efficient evolution
- **VQE/QAOA** (`vqe_qaoa.py`): Variational quantum algorithms
 - Ground state finding and combinatorial optimization
 - Hardware-efficient ansätze with classical optimizer integration

#### Adaptive MPS Implementation
- Energy-based adaptive truncation with error bounds
- Per-bond dimension caps and global memory budgets
- Mixed precision support (complex64/complex128) with auto-promotion
- Comprehensive statistics tracking and diagnostics
- Canonical forms (left, right, mixed) with robust QR/SVD

#### Documentation & Testing
- Complete whitepaper documenting architecture and algorithms
- Research paper with mathematical foundations
- 75+ unit tests across unit, integration, and performance suites
- All 7/7 benchmark suites passing

### Performance
- 626,454× memory compression (30 qubits)
- 20.4× Clifford circuit speedup
- GPU-accelerated operations throughout
- Demonstrated capacity: 100,000+ qubits (χ=64, moderate entanglement)

### Changed
- Updated lazy import structure in `__init__.py`
- Enhanced numerical stability with multi-driver SVD fallback
- Improved error tracking and diagnostics

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Fixed`: Bug fixes
- `Performance`: Performance improvements
