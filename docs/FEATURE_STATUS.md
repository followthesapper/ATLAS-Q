# ATLAS-Q Feature Status
**What's Actually Implemented vs Documented**

**Last Updated:** October 2025

---

## See Also

- **[Complete Guide](COMPLETE_GUIDE.md)** - Full API reference and tutorials
- **[ Interactive Notebook](../ATLAS_Q_Demo.ipynb)** - Try all features interactively
- **[Whitepaper](WHITEPAPER.md)** - Technical architecture details
- **[Research Paper](RESEARCH_PAPER.md)** - Algorithm implementations

---

## Current Development Status (Dev Branch)

**Latest Update:** 2025-10-27 - Commit d82c9fc

### Recently Completed (Priority 1)
- **Molecular Hamiltonian Builder** - Fully implemented and tested
 - Function: `MPOBuilder.molecular_hamiltonian_from_specs()`
 - Integration: PySCF with Jordan-Wigner transformation
 - Tests: 4/4 passing in `test_molecular_hamiltonians.py`
 - Supports: H2, LiH, H2O, custom geometries

- **MaxCut Hamiltonian Builder** - Fully implemented and tested
 - Function: `MPOBuilder.maxcut_hamiltonian()`
 - QAOA graph optimization problems
 - Tests: 4/4 passing in `test_maxcut.py`
 - Supports: Weighted/unweighted graphs, edge normalization

### Completed (Priority 2)
- **Circuit Cutting Integration** - Fully tested and working
 - 7/7 tests passing in `test_circuit_cutting.py`
 - Min-cut partitioning, coupling graph analysis, entanglement heatmaps
- **PEPS (Projected Entangled Pair States)** - Fully tested and working
 - 10/10 tests passing in `test_peps.py`
 - 2D tensor networks, boundary-MPS contraction, PatchPEPS
- **Distributed MPS** - Tested in single-GPU mode
 - 10/10 tests passing in `test_distributed_mps.py`
 - Bond-parallel decomposition, multi-GPU ready (requires NCCL)
- **cuQuantum Backend** - Tested with cuQuantum 25.09.1 (OPTIONAL)
 - 11/11 tests passing in `test_cuquantum.py`
 - Auto-detection, PyTorch fallback verified
 - **Tested with:** cuQuantum 25.09.1, cuTensorNet API
 - **Install:** `pip install cuquantum-python` (optional, ~320MB)

**Branch:** Dev
**Latest Work:** Priority 1 + 2 complete (46 tests, all passing)
**Status:** Ready for merge to main

---

## Fully Implemented & Tested

These features pass benchmarks and are production-ready:

### 1. Period-Finding & Factorization
- **Module:** `quantum_hybrid_system.py`
- **Access:** `get_quantum_sim()`
- **Status:** Verified against canonical benchmarks (N=15, 21, 143)
- **Example:**
 ```python
 from atlas_q import get_quantum_sim
 QCH, _, _, _ = get_quantum_sim()
 sim = QCH()
 factors = sim.factor_number(221) # [13, 17]
 ```

### 2. Adaptive MPS
- **Module:** `adaptive_mps.py`
- **Access:** `get_adaptive_mps()`
- **Status:** GPU-accelerated with Triton kernels
- **Example:**
 ```python
 from atlas_q import get_adaptive_mps
 modules = get_adaptive_mps()
 mps = modules['AdaptiveMPS'](10, bond_dim=8, device='cuda')
 ```

### 3. Noise Models
- **Module:** `noise_models.py`
- **Access:** `get_noise_models()`
- **Status:** Kraus completeness verified, fidelity tracking works
- **Example:**
 ```python
 from atlas_q import get_noise_models
 noise = get_noise_models()
 model = noise['NoiseModel'].depolarizing(p1q=0.001, device='cuda')
 ```

### 4. Stabilizer Backend
- **Module:** `stabilizer_backend.py`
- **Access:** `get_stabilizer()`
- **Status:** 20× speedup vs MPS on Clifford circuits
- **Example:**
 ```python
 from atlas_q import get_stabilizer
 stab = get_stabilizer()
 sim = stab['StabilizerSimulator'](n_qubits=50, device='cuda')
 sim.h(0)
 sim.cnot(0, 1)
 ```

### 5. MPO Operations
- **Module:** `mpo_ops.py`
- **Access:** `get_mpo_ops()`
- **Status:** Ising, Heisenberg, Molecular, MaxCut Hamiltonians tested
- **Limitations:** None
- **Example:**
 ```python
 from atlas_q import get_mpo_ops
 mpo = get_mpo_ops()
 H = mpo['MPOBuilder'].ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')
 energy = mpo['expectation_value'](mps, H)
 ```

### 6. TDVP Time Evolution
- **Module:** `tdvp.py`
- **Access:** `get_tdvp()`
- **Status:** Energy conservation verified
- **Example:**
 ```python
 from atlas_q import get_tdvp
 tdvp_mod = get_tdvp()
 config = tdvp_mod['TDVPConfig'](dt=0.01, t_final=1.0)
 tdvp = tdvp_mod['TDVP1Site'](H, mps, config)
 times, energies = tdvp.run()
 ```

### 7. VQE/QAOA
- **Module:** `vqe_qaoa.py`
- **Access:** `get_vqe_qaoa()`
- **Status:** Convergence to known ground states tested
- **Limitations:** No high-level molecular Hamiltonian builder
- **Example:**
 ```python
 from atlas_q import get_vqe_qaoa, get_mpo_ops
 vqe_mod = get_vqe_qaoa()
 mpo_mod = get_mpo_ops()

 H = mpo_mod['MPOBuilder'].heisenberg_hamiltonian(6, device='cuda')
 config = vqe_mod['VQEConfig'](n_layers=3, max_iter=50)
 vqe = vqe_mod['VQE'](H, config)
 energy, params = vqe.run()
 ```

### 8. 2D Circuits
- **Module:** `planar_2d.py`
- **Access:** `get_planar_2d()`
- **Status:** SWAP insertion verified
- **Example:**
 ```python
 from atlas_q import get_planar_2d
 planar = get_planar_2d()
 layout = planar['Layout2D'](rows=4, cols=4, topology='grid')
 circuit = planar['Planar2DCircuit'](layout, device='cuda')
 ```

### 9. Circuit Cutting
- **Module:** `circuit_cutting.py`
- **Access:** `get_circuit_cutting()`
- **Status:** Fully tested (7/7 tests)
- **Features:** Min-cut partitioning, coupling graph analysis, entanglement heatmaps
- **Example:**
 ```python
 from atlas_q import get_circuit_cutting

 cutting = get_circuit_cutting()
 config = cutting['CuttingConfig'](max_partition_size=4)
 cutter = cutting['CircuitCutter'](config)

 # Analyze circuit
 gates = [('H', [0], []), ('CNOT', [0, 1], [])]
 graph = cutter.analyze_circuit(gates)
 partitions = cutter.partition_circuit(gates, n_partitions=2)
 ```

### 10. PEPS (2D Tensor Networks)
- **Module:** `peps.py`
- **Access:** `get_peps()`
- **Status:** Fully tested (10/10 tests)
- **Features:** Boundary-MPS contraction, PatchPEPS for shallow circuits
- **Example:**
 ```python
 from atlas_q import get_peps

 peps_mod = get_peps()
 patch = peps_mod['PatchPEPS'](patch_size=4, device='cuda')

 # Apply shallow 2D circuit
 gates = [('H', [(0, 0)], []), ('CZ', [(0, 0), (0, 1)], [])]
 patch.apply_shallow_circuit(gates)
 norm = patch.peps.compute_norm()
 ```

---

## Advanced Features (Experimental)

These features are fully tested but may require special setup:

### 11. Distributed MPS
- **Module:** `distributed_mps.py`
- **Access:** `get_distributed_mps()`
- **Status:** Tested in single-GPU mode (10/10 tests)
- **Limitation:** Multi-GPU requires NCCL backend and multiple GPUs
- **Example:**
 ```python
 from atlas_q import get_distributed_mps

 dmps_mod = get_distributed_mps()
 config = dmps_mod['DistributedConfig'](mode=dmps_mod['DistMode'].BOND_PARALLEL)
 dmps = dmps_mod['DistributedMPS'](num_qubits=100, bond_dim=16, config=config)
 ```

### 12. cuQuantum Backend
- **Module:** `cuquantum_backend.py`
- **Access:** `get_cuquantum()`
- **Status:** Tested with cuQuantum 25.09.1 (11/11 tests)
- **Limitation:** Requires `pip install cuquantum-python` (~320MB, optional)
- **Example:**
 ```python
 from atlas_q import get_cuquantum

 cuq = get_cuquantum()
 backend = cuq['CuQuantumBackend']()

 # Automatically uses cuQuantum if available, falls back to PyTorch
 U, S, Vt = backend.svd(tensor, chi_max=16)
 ```

---

## Recently Implemented Features (v0.6.0)

### 1. Molecular Hamiltonian from Specs
**Function:** `MPOBuilder.molecular_hamiltonian_from_specs(molecule='H2', basis='sto-3g', ...)`
- **Status:** IMPLEMENTED & TESTED
- **Integration:** PySCF for quantum chemistry calculations
- **Mapping:** Jordan-Wigner transformation
- **Supported molecules:** H2, LiH, H2O, or custom geometry strings
- **Tests:** 4/4 passing in `tests/integration/test_molecular_hamiltonians.py`
- **Example:**
 ```python
 from atlas_q import get_mpo_ops
 mpo = get_mpo_ops()

 # Build H2 Hamiltonian
 H = mpo['MPOBuilder'].molecular_hamiltonian_from_specs(
 molecule='H2',
 basis='sto-3g',
 charge=0,
 spin=0,
 device='cuda'
 )

 # Use with VQE for ground state energy
 from atlas_q import get_vqe_qaoa
 vqe_mod = get_vqe_qaoa()
 vqe = vqe_mod['VQE'](H, ansatz_depth=3, device='cuda')
 energy, params = vqe.optimize(max_iter=100)
 ```

### 2. MaxCut Hamiltonian Builder
**Function:** `MPOBuilder.maxcut_hamiltonian(edges, weights, ...)`
- **Status:** IMPLEMENTED & TESTED
- **Description:** Build QAOA Hamiltonian for graph MaxCut problems
- **Formulation:** H = Σ_{(i,j)∈E} w_{ij} (1 - Z_i Z_j) / 2
- **Tests:** 4/4 passing in `tests/integration/test_maxcut.py`
- **Example:**
 ```python
 from atlas_q import get_mpo_ops, get_vqe_qaoa
 mpo = get_mpo_ops()

 # Define graph edges (triangle)
 edges = [(0, 1), (1, 2), (0, 2)]
 weights = [1.0, 1.0, 1.0]

 # Build MaxCut Hamiltonian
 H = mpo['MPOBuilder'].maxcut_hamiltonian(
 edges=edges,
 weights=weights,
 device='cuda'
 )

 # Solve with QAOA
 qaoa_mod = get_vqe_qaoa()
 qaoa = qaoa_mod['QAOA'](H, depth=3, device='cuda')
 energy, params = qaoa.optimize(max_iter=100)
 ```

---

## Import Patterns

** WRONG (Won't Work):**
```python
from atlas_q import AdaptiveMPS # Not exported
from atlas_q import VQE # Not exported
from atlas_q.vqe_qaoa import VQE # Works but not recommended
```

** CORRECT (Recommended):**
```python
# Use lazy loaders
from atlas_q import get_adaptive_mps, get_vqe_qaoa

mps_modules = get_adaptive_mps()
AdaptiveMPS = mps_modules['AdaptiveMPS']

vqe_modules = get_vqe_qaoa()
VQE = vqe_modules['VQE']
```

---

## Verification

Run benchmarks to verify features:

```bash
# Comprehensive feature validation
python scripts/benchmarks/validate_all_features.py

# Expected output:
# Benchmark 1: Noise Models - 3/3 passing
# Benchmark 2: Stabilizer Backend - 3/3 passing
# Benchmark 3: MPO Operations - 3/3 passing
# Benchmark 4: TDVP Time Evolution - 2/2 passing
# Benchmark 5: VQE/QAOA - 2/2 passing
# Benchmark 6: 2D Circuits - 2/2 passing
# Benchmark 7: Integration Tests - 2/2 passing
```

---

## Honest Assessment

**What ATLAS-Q Does Well:**
- Period-finding for specific semiprimes
- GPU-accelerated tensor networks with adaptive truncation
- Stabilizer optimization for Clifford circuits
- Basic NISQ algorithm implementations (VQE/QAOA)
- Memory-efficient state representation

**What ATLAS-Q Does NOT Do:**
- High-level quantum chemistry (use PySCF + this)
- Arbitrary circuit execution (use Qiskit/Cirq for that)
- Full statevector simulation of highly entangled states
- Production-grade error correction

**Best Use Cases:**
- Research on tensor network methods
- NISQ algorithm prototyping
- GPU acceleration experiments
- Shor's algorithm demonstrations

---

## Documentation Accuracy

When you see examples in docs, verify they actually work:

1. **README.md** - Honest, high-level claims ( accurate)
2. **USAGE_GUIDE.md** - **Needs fixing** (has wrong imports)
3. **API_GUIDE.md** - **Needs fixing** (documents non-existent features)
4. **Benchmarks** - Shows what actually works

---

**Use this document as the source of truth for what's implemented.**
