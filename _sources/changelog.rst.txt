Changelog
=========

All notable changes to ATLAS-Q are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

0.7.0 - 2025-12-16 (IR v1.1.0)
------------------------------

IR Integration - COMPLETE
^^^^^^^^^^^^^^^^^^^^^^^^^

This release completes the Informational Relativity (IR) integration with pre-computation regime diagnosis.

Pre-Computation Diagnosis (NEW)
"""""""""""""""""""""""""""""""

- **Regime Analyzer** (``ir_enhanced/regime_analyzer.py``): Diagnose quantum problems BEFORE running

  - IR (Informational Relativity) regime: High coherence, quantum advantage likely
  - Transition regime: Marginal coherence, needs careful tuning
  - AIR (Anti-IR) regime: Low coherence, classical methods may be better
  - GO/NO-GO classification using physics-derived e^-2 threshold (0.135)
  - ``analyze_state_regime()``, ``analyze_hamiltonian_regime()``, ``analyze_mps_bond_regime()``
  - ``predict_quantum_advantage()`` for resource planning
  - ``should_use_ir_grouping()`` for adaptive algorithm selection

Spectral Lifting (NEW)
""""""""""""""""""""""

- **Full M_ij Relational Matrix Analysis** (``ir_enhanced/spectral_lifting.py``):

  - ``compute_relational_matrix()`` - Build coherence correlation matrices
  - ``extract_structure_modes()`` - Find dominant eigenmodes
  - ``spectral_grouping()`` - Group observables by spectral similarity
  - ``coherent_structure_score()`` - Quantify structure detection quality

Performance Improvements (VALIDATED)
""""""""""""""""""""""""""""""""""""

+---------------------+------------------------+------------+
| Feature             | Improvement            | Status     |
+=====================+========================+============+
| VQE Grouping        | 4× circuit reduction   | Production |
+---------------------+------------------------+------------+
| Period Finding      | 42% shot reduction     | Production |
+---------------------+------------------------+------------+
| Variance Reduction  | 2-60× (VQE)            | Production |
+---------------------+------------------------+------------+
| MPS Scalability     | 100 qubits in 1.56s    | Validated  |
+---------------------+------------------------+------------+
| Memory Compression  | 10^25× (100 qubits)    | Validated  |
+---------------------+------------------------+------------+

New IR Modules
""""""""""""""

- ``gradient_grouping.py`` - Parameter shift optimization with coherence
- ``qaoa_grouping.py`` - Edge-based cost operator grouping
- ``shadow_tomography.py`` - Adaptive classical shadows with IR
- ``state_tomography.py`` - IR-enhanced state reconstruction
- ``tdvp_observables.py`` - Real-time coherence tracking for TDVP

Added
^^^^^

GPU CUDA Backend
""""""""""""""""

- Direct CUDA Driver API integration via ctypes
- Pre-compiled PTX kernels (version-independent)
- Supports single-qubit gates: H, X, Y, Z, RX, RY, RZ
- Supports two-qubit gates: CNOT, CZ, SWAP
- f64 precision for numerical stability
- 2-13× faster than CPU for 15+ qubits

Triton IR Coherence Kernels
"""""""""""""""""""""""""""

- GPU-accelerated coherence computation
- ``compute_response_coherence_triton()`` for R-bar, V_phi metrics
- GO/NO-GO classification at GPU speed

Changed
^^^^^^^

- Improved exception handling in GPU backend
- Updated ``ir_enhanced/__init__.py`` exports for regime analyzer
- Coherence-aware VQE now includes pre-computation regime analysis

Fixed
^^^^^

- Fixed Triton IR Coherence benchmark (amplitudes vs responses/phases)
- Fixed UCCSD Ansatz benchmark imports
- Fixed MANIFEST.in to include compiled Rust extensions

Benchmark Results
"""""""""""""""""

- **30/31 benchmarks passing** (cuQuantum optional skip)
- **7/7 IR Enhanced tests passing**

0.6.x Series
------------

0.6.4 - 2025-11-04
^^^^^^^^^^^^^^^^^^

Added
"""""

UCCSD Ansatz for Molecular VQE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **UCCSD (Unitary Coupled-Cluster Singles and Doubles)** (``ansatz_uccsd.py``): Chemistry-aware variational ansatz

  - OpenFermion integration for fermionic operator generation
  - MPS-compatible implementation (no exponential memory)
  - Pauli string decomposition with ``apply_pauli_exp_to_mps()``
  - Hartree-Fock reference state initialization
  - Compatible with VQE for ground state chemistry calculations

Quantum Chemistry & Optimization Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Molecular Hamiltonian Builder** (``mpo_ops.py``): PySCF integration for quantum chemistry

  - ``molecular_hamiltonian_from_specs()`` - Build electronic structure Hamiltonians
  - Support for H2, LiH, H2O, and custom geometry strings
  - Jordan-Wigner fermion-to-qubit transformation
  - Compatible with VQE for ground state energy calculations
  - 4/4 tests passing in ``test_molecular_hamiltonians.py``

- **MaxCut Hamiltonian Builder** (``mpo_ops.py``): QAOA graph optimization

  - ``maxcut_hamiltonian()`` - Build MaxCut problem Hamiltonians
  - Weighted and unweighted graph support
  - Automatic edge normalization for undirected graphs
  - Compatible with QAOA for combinatorial optimization
  - 4/4 tests passing in ``test_maxcut.py``

Advanced Tensor Network Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Circuit Cutting** (``circuit_cutting.py``): Partition large circuits for simulation

  - Coupling graph analysis and entanglement heatmaps
  - Min-cut and spectral partitioning algorithms
  - Classical stitching with variance reduction
  - 7/7 tests passing in ``test_circuit_cutting.py``

- **PEPS (Projected Entangled Pair States)** (``peps.py``): 2D tensor networks

  - True 2D representation for shallow quantum circuits
  - Boundary-MPS contraction strategy
  - PatchPEPS for 4×4 and 5×5 grids
  - Single and two-site gate application
  - 10/10 tests passing in ``test_peps.py``

- **Distributed MPS** (``distributed_mps.py``): Multi-GPU scaling

  - Bond-wise domain decomposition across GPUs
  - Overlapped communication and computation
  - Checkpoint/restart for long simulations
  - 10/10 tests passing in ``test_distributed_mps.py``

- **cuQuantum Backend** (``cuquantum_backend.py``): Optional NVIDIA acceleration

  - cuTensorNet 25.x integration for tensor operations
  - Automatic fallback to PyTorch if unavailable
  - 2-10× speedup on compatible NVIDIA GPUs (requires ``cuquantum-python``)
  - 11/11 tests passing in ``test_cuquantum.py`` (tested with cuQuantum 25.09.1)
  - **Install:** ``pip install cuquantum-python`` (optional, ~320MB)

Changed
"""""""

Improved Import System (Better UX)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Direct module imports now supported** (``__init__.py``): Pythonic import pattern

  - **New (recommended)**: ``from atlas_q import mpo_ops, tdvp, vqe_qaoa``
  - **Legacy (still works)**: ``atlas_q.get_mpo_ops()`` returns dict
  - Enables IDE autocomplete and type hints
  - Matches standard Python package conventions (like NumPy, PyTorch)
  - Backwards compatible - old getter pattern still supported

0.5.0 - 2025-10-26
------------------

Added
^^^^^

GPU Acceleration & Triton Kernels
""""""""""""""""""""""""""""""""""

- Custom Triton kernels for MPS gate operations (1.5-3× speedup)
- GPU-optimized tensor contractions with cuBLAS tensor cores
- Modular exponentiation kernels for period-finding
- 77,000+ ops/sec gate throughput achieved

Tensor Network Features
"""""""""""""""""""""""

- **Noise Models** (``noise_models.py``): Full Kraus operator framework for NISQ simulation

  - Depolarizing, dephasing, amplitude damping, Pauli noise channels
  - Stochastic noise applicator with reproducible seeds

- **Stabilizer Backend** (``stabilizer_backend.py``): Clifford circuit fast path

  - O(n²) complexity via Gottesman-Knill theorem
  - 20× speedup over generic MPS
  - Automatic handoff to MPS for non-Clifford gates

- **MPO Operations** (``mpo_ops.py``): Hamiltonian and observable framework

  - Pre-built Hamiltonians: Ising, Heisenberg, custom spin chains
  - Expectation values and correlation functions

- **TDVP** (``tdvp.py``): Time evolution for Hamiltonian dynamics

  - 1-site (conserves bond dimension) and 2-site (adaptive) variants
  - Krylov subspace methods for efficient evolution

- **VQE/QAOA** (``vqe_qaoa.py``): Variational quantum algorithms

  - Ground state finding and combinatorial optimization
  - Hardware-efficient ansätze with classical optimizer integration

Adaptive MPS Implementation
""""""""""""""""""""""""""""

- Energy-based adaptive truncation with error bounds
- Per-bond dimension caps and global memory budgets
- Mixed precision support (complex64/complex128) with auto-promotion
- Comprehensive statistics tracking and diagnostics
- Canonical forms (left, right, mixed) with robust QR/SVD

Documentation & Testing
"""""""""""""""""""""""

- Complete whitepaper documenting architecture and algorithms
- Research paper with mathematical foundations
- 75+ unit tests across unit, integration, and performance suites
- All 7/7 benchmark suites passing

Performance
^^^^^^^^^^^

- 626,454× memory compression (30 qubits)
- 20.4× Clifford circuit speedup
- GPU-accelerated operations throughout
- Demonstrated capacity: 100,000+ qubits (χ=64, moderate entanglement)

Changed
^^^^^^^

- Updated lazy import structure in ``__init__.py``
- Enhanced numerical stability with multi-driver SVD fallback
- Improved error tracking and diagnostics

Legend
------

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Fixed**: Bug fixes
- **Performance**: Performance improvements
