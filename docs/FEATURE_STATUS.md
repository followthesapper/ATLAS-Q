# ATLAS-Q Feature Status
**What's Actually Implemented vs Documented**

**Last Updated:** October 2025

---

## 📚 See Also

- **[Complete Guide](COMPLETE_GUIDE.md)** - Full API reference and tutorials
- **[📓 Interactive Notebook](../ATLAS_Q_Demo.ipynb)** - Try all features interactively
- **[Whitepaper](WHITEPAPER.md)** - Technical architecture details
- **[Research Paper](RESEARCH_PAPER.md)** - Algorithm implementations

---

## ✅ Fully Implemented & Tested

These features pass benchmarks and are production-ready:

### 1. Period-Finding & Factorization
- **Module:** `quantum_hybrid_system.py`
- **Access:** `get_quantum_sim()`
- **Status:** ✅ Verified against canonical benchmarks (N=15, 21, 143)
- **Example:**
  ```python
  from atlas_q import get_quantum_sim
  QCH, _, _, _ = get_quantum_sim()
  sim = QCH()
  factors = sim.factor_number(221)  # [13, 17]
  ```

### 2. Adaptive MPS
- **Module:** `adaptive_mps.py`
- **Access:** `get_adaptive_mps()`
- **Status:** ✅ GPU-accelerated with Triton kernels
- **Example:**
  ```python
  from atlas_q import get_adaptive_mps
  modules = get_adaptive_mps()
  mps = modules['AdaptiveMPS'](10, bond_dim=8, device='cuda')
  ```

### 3. Noise Models
- **Module:** `noise_models.py`
- **Access:** `get_noise_models()`
- **Status:** ✅ Kraus completeness verified, fidelity tracking works
- **Example:**
  ```python
  from atlas_q import get_noise_models
  noise = get_noise_models()
  model = noise['NoiseModel'].depolarizing(p1q=0.001, device='cuda')
  ```

### 4. Stabilizer Backend
- **Module:** `stabilizer_backend.py`
- **Access:** `get_stabilizer()`
- **Status:** ✅ 20× speedup vs MPS on Clifford circuits
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
- **Status:** ✅ Ising, Heisenberg Hamiltonians tested
- **Limitations:** Molecular Hamiltonians are PLACEHOLDERS
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
- **Status:** ✅ Energy conservation verified
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
- **Status:** ✅ Convergence to known ground states tested
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
- **Status:** ✅ SWAP insertion verified
- **Example:**
  ```python
  from atlas_q import get_planar_2d
  planar = get_planar_2d()
  layout = planar['Layout2D'](rows=4, cols=4, topology='grid')
  circuit = planar['Planar2DCircuit'](layout, device='cuda')
  ```

---

## ⚠️ Partially Implemented

These features exist but have limitations:

### 9. Circuit Cutting
- **Module:** `circuit_cutting.py`
- **Access:** `get_circuit_cutting()`
- **Status:** ⚠️ Classes exist, needs integration testing
- **Limitation:** May not be fully integrated with MPS workflows

### 10. PEPS
- **Module:** `peps.py`
- **Access:** `get_peps()`
- **Status:** ⚠️ Basic implementation, not extensively tested
- **Limitation:** Contraction algorithms may be incomplete

### 11. Distributed MPS
- **Module:** `distributed_mps.py`
- **Access:** `get_distributed_mps()`
- **Status:** ⚠️ Basic implementation, needs multi-GPU testing
- **Limitation:** May not scale efficiently in practice

### 12. cuQuantum Backend
- **Module:** `cuquantum_backend.py`
- **Access:** `get_cuquantum()`
- **Status:** ⚠️ Wrapper exists, requires cuquantum-python installed
- **Limitation:** Optional dependency, not tested in CI

---

## ❌ Not Implemented (Placeholders)

These functions exist but are not functional:

### 1. Molecular Hamiltonian from Specs
**Function:** `build_molecular_hamiltonian(molecule='H2', basis='sto-3g', ...)`
- **Status:** ❌ Does NOT exist
- **What exists:** `build_molecular_hamiltonian(h1, h2, mapping, device)` returns identity (placeholder)
- **Workaround:** Use external tools (PySCF, OpenFermion) to get h1/h2 matrices

### 2. MaxCut Hamiltonian Builder
**Function:** `build_maxcut_hamiltonian(edges, ...)`
- **Status:** ❌ Does NOT exist
- **Workaround:** Build Ising Hamiltonian manually with appropriate J coefficients

---

## 🔧 Import Patterns

**❌ WRONG (Won't Work):**
```python
from atlas_q import AdaptiveMPS  # ❌ Not exported
from atlas_q import VQE          # ❌ Not exported
from atlas_q.vqe_qaoa import VQE # ✅ Works but not recommended
```

**✅ CORRECT (Recommended):**
```python
# Use lazy loaders
from atlas_q import get_adaptive_mps, get_vqe_qaoa

mps_modules = get_adaptive_mps()
AdaptiveMPS = mps_modules['AdaptiveMPS']

vqe_modules = get_vqe_qaoa()
VQE = vqe_modules['VQE']
```

---

## 📊 Verification

Run benchmarks to verify features:

```bash
# Comprehensive feature validation
python scripts/benchmarks/validate_all_features.py

# Expected output:
# ✅ Benchmark 1: Noise Models          - 3/3 passing
# ✅ Benchmark 2: Stabilizer Backend    - 3/3 passing
# ✅ Benchmark 3: MPO Operations        - 3/3 passing
# ✅ Benchmark 4: TDVP Time Evolution   - 2/2 passing
# ✅ Benchmark 5: VQE/QAOA             - 2/2 passing
# ✅ Benchmark 6: 2D Circuits          - 2/2 passing
# ✅ Benchmark 7: Integration Tests    - 2/2 passing
```

---

## 🎯 Honest Assessment

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

## 📖 Documentation Accuracy

When you see examples in docs, verify they actually work:

1. **README.md** - Honest, high-level claims (✅ accurate)
2. **USAGE_GUIDE.md** - **Needs fixing** (has wrong imports)
3. **API_GUIDE.md** - **Needs fixing** (documents non-existent features)
4. **Benchmarks** - ✅ Shows what actually works

---

**Use this document as the source of truth for what's implemented.**
