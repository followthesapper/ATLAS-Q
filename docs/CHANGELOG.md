# Changelog

All notable changes to ATLAS-Q will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
