# ATLAS-Q Quantum Simulator - Complete Feature List

**Version:** 0.7.0 (IR v1.1.0)
**Last Updated:** December 2025

---

## Core Quantum Simulation

| Feature | Description | Status |
|---------|-------------|--------|
| **Matrix Product State (MPS)** | GPU-accelerated tensor network with PyTorch, 1.5-2x faster than NumPy | Production |
| **Adaptive MPS** | Per-bond adaptive dimensions, mixed precision, memory budget enforcement | Experimental |
| **MPO Operations** | Matrix Product Operators for Hamiltonians with Jordan-Wigner encoding | Production |
| **PEPS** | 2D tensor networks for shallow circuits (4x4, 5x5 grids) | Experimental |
| **Statevector** | Full state simulation for small circuits | Production |

## Quantum Algorithms

| Feature | Description | Status |
|---------|-------------|--------|
| **VQE** | Variational Quantum Eigensolver with multiple optimizers | Production |
| **QAOA** | Quantum Approximate Optimization for graph problems | Production |
| **Coherence-Aware VQE** | VQE with IR coherence tracking and GO/NO-GO classification | Experimental |
| **Grover's Search** | Complete quantum search with oracle abstraction | Production |
| **Shor's Algorithm** | Period finding and integer factorization via quantum hybrid | Production |
| **TDVP** | Time-Dependent Variational Principle for real-time dynamics | Production |

## GPU Acceleration

| Feature | Speedup | Status |
|---------|---------|--------|
| **GPU Statevector** | 2-13x vs CPU (15-30 qubits) | Production |
| **Triton Modpow Kernels** | 2-3x for period finding | Production |
| **Triton IR Coherence Kernels** | GPU coherence computation | Experimental |
| **cuQuantum Backend** | 3-5x additional (optional) | Optional |
| **Stabilizer Backend** | 20x for Clifford circuits | Production |

## Informational Relativity (IR) Integration

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Regime Analyzer** | Pre-computation diagnosis (IR/Transition/AIR) | Skip hopeless problems |
| **VQE Grouping** | Coherence-based measurement grouping | 2-60x variance reduction |
| **QAOA Grouping** | Edge grouping for cost operators | Shot reduction |
| **Gradient Grouping** | Parameter shift rule optimization | Fewer gradient evals |
| **Spectral Lifting** | Full M_ij relational matrix analysis | Structure detection |
| **Period Finding Enhancement** | IR-preprocessed QPE | 29-42% shot reduction |
| **Quantum Advantage Prediction** | Regime-based Q vs C guidance | Resource planning |

## Framework Integrations

| Adapter | Features | Status |
|---------|----------|--------|
| **Qiskit Backend** | Drop-in replacement, auto-leverages all ATLAS-Q features | Production |
| **Cirq Simulator** | Full Cirq API compatibility | Production |
| **IBM Quantum** | Hardware deployment scripts, coherence validation | Experimental |

## Advanced Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Circuit Cutting** | Entanglement forging, distributed simulation | Production |
| **Distributed MPS** | Multi-GPU with partition-based load balancing | Experimental |
| **Planar 2D Circuits** | SWAP synthesis, chi scheduling for 2D topologies | Production |
| **Noise Models** | Depolarizing, dephasing, amplitude damping channels | Production |
| **UCCSD Ansatz** | Chemistry-grade ansatz from PySCF | Production |

## Coherence & Measurement

| Feature | Description |
|---------|-------------|
| **Coherence Metrics** | R-bar (mean resultant length), V_phi (circular variance) |
| **GO/NO-GO Classification** | e^-2 boundary at 0.135 |
| **Adaptive IR Decision** | Dynamic grouping control based on coherence |
| **Pauli Grouping** | Qubit-wise commutativity optimization |

## CLI Commands

```bash
# Factor integers
python -m atlas_q factor 15

# Run benchmarks
python -m atlas_q benchmark

# System info
python -m atlas_q info

# Interactive demo
python -m atlas_q demo
```

## Key Performance Numbers

| Metric | Value |
|--------|-------|
| GPU Speedup | 2-13x vs CPU |
| Memory Compression | 626,000x (MPS vs statevector) |
| IR Shot Reduction | 29-42% (period finding) |
| IR Variance Reduction | 2-60x (VQE grouping) |
| Stabilizer Speedup | 20x for Clifford |
| Supported Qubits | 100+ (MPS), 30 (statevector) |

## What Makes ATLAS-Q Different

1. **IR-First Architecture**: Diagnose regime -> Select representation -> Then compute (not the reverse)

2. **Automatic Backend Selection**: Stabilizer for Clifford, MPS for entangled, GPU for speed

3. **Pre-computation Diagnosis**: Know if VQE will work BEFORE running it

4. **Drop-in Compatibility**: Works as Qiskit/Cirq backend with zero code changes

5. **Hybrid Classical-Quantum**: Tensor networks + IR coherence + GPU = efficient simulation

---

## Module Summary

- **Total Modules:** 68 Python files
- **Total Lines:** ~23,000 (src/atlas_q)
- **Largest Modules:**
  - quantum_hybrid_system.py (1565 LOC)
  - mpo_ops.py (1358 LOC)
  - qiskit_adapter.py (861 LOC)

## Dependencies

**Core Requirements:**
- numpy, torch, scipy, triton

**Optional:**
- cuQuantum (3-5x additional speedup)
- PySCF + OpenFermion (molecular chemistry)
- Qiskit (backend integration)
- Cirq (Google framework)
- IBM Quantum API (hardware access)
