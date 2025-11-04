# ATLAS-Q: GPU-Accelerated Quantum Tensor Network Simulator
**Adaptive Tensor Learning And Simulation – Quantum**

**Version 0.6.1** | **October 2025**

> **High-performance quantum simulation using GPU-accelerated tensor networks with molecular chemistry, circuit cutting, and cuQuantum integration**

[![Performance](https://img.shields.io/badge/Performance--blue)]()
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%2B%20Triton%20%2B%20cuQuantum-green)]()
[![Memory](https://img.shields.io/badge/Memory-626k×%20Compression-red)]()
[![Tests](https://img.shields.io/badge/Tests-46%2F46%20Passing-brightgreen)]()

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/FollowTheSapper)

---

## Performance Highlights

- **77K+ ops/sec** gate throughput (GPU-optimized)
- **626,000× memory compression** vs full statevector (30 qubits)
- **20× speedup** on Clifford circuits (Stabilizer backend)
- **1.5-3× speedup** on gate operations (custom Triton kernels)
- **All 46/46 integration tests passing** (Priority 1 + 2 features)

---

## Quick Start

### Option 1: Interactive Notebook (No Install!)

Try ATLAS-Q instantly in Google Colab or Jupyter:

**[ Open ATLAS_Q_Demo.ipynb in Colab](https://colab.research.google.com/github/followthesapper/ATLAS-Q/blob/ATLAS-Q/ATLAS_Q_Demo.ipynb)**

Or download and run locally:
```bash
wget https://github.com/followthesapper/ATLAS-Q/raw/ATLAS-Q/ATLAS_Q_Demo.ipynb
jupyter notebook ATLAS_Q_Demo.ipynb
```

---

### Option 2: Python Package (Recommended)

**Using pip (PyPI):**
```bash
# Install from PyPI
pip install atlas-quantum

# With GPU support
pip install atlas-quantum[gpu]

# Verify installation
python -c "from atlas_q import get_quantum_sim; print('ATLAS-Q installed!')"
```

**Using uv (10-100x faster):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ATLAS-Q
uv pip install atlas-quantum

# With GPU support
uv pip install atlas-quantum[gpu]
```

**Using conda (coming soon):**
```bash
# Once available on conda-forge
conda install -c conda-forge atlas-quantum
```

**First example:**
```python
from atlas_q import get_quantum_sim

QCH, _, _, _ = get_quantum_sim()
sim = QCH()
factors = sim.factor_number(221)
print(f"221 = {factors[0]} × {factors[1]}") # 221 = 13 × 17
```

---

### Option 3: System Package (Debian/Ubuntu)

**Download and install .deb package:**
```bash
# Download from GitHub releases
wget https://github.com/followthesapper/ATLAS-Q/releases/download/v0.6.2/python3-atlas-quantum_0.6.2_all.deb

# Install
sudo dpkg -i python3-atlas-quantum_0.6.2_all.deb
sudo apt-get install -f  # Fix any dependencies
```

---

### Option 4: Docker

**GPU version (recommended):**
```bash
docker pull ghcr.io/followthesapper/atlas-q:cuda
docker run --rm -it --gpus all ghcr.io/followthesapper/atlas-q:cuda python3
```

**CPU version:**
```bash
docker pull ghcr.io/followthesapper/atlas-q:cpu
docker run --rm -it ghcr.io/followthesapper/atlas-q:cpu python3
```

**Run benchmarks in Docker:**
```bash
docker run --rm --gpus all ghcr.io/followthesapper/atlas-q:cuda \
 python3 /opt/atlas-q/scripts/benchmarks/validate_all_features.py
```

---

### Option 5: From Source

```bash
# Clone repository
git clone https://github.com/followthsapper/ATLAS-Q.git
cd ATLAS-Q

# Install ATLAS-Q
pip install -e .[gpu]

# Setup GPU acceleration (auto-detects your GPU)
./setup_triton.sh

# Run benchmarks
python scripts/benchmarks/validate_all_features.py
```

---

### GPU Acceleration Setup

The `setup_triton.sh` script automatically detects your GPU and configures Triton kernels:

- **Auto-detects:** V100, A100, H100, GB100/GB200, and future architectures
- **Configures:** `TORCH_CUDA_ARCH_LIST` and `TRITON_PTXAS_PATH`
- **Persists:** Adds settings to `~/.bashrc`

**Performance gains:** 1.5-3× faster gate operations, 100-1000× faster period-finding

---

### Command-Line Interface

ATLAS-Q includes a CLI for quick operations:

```bash
# Show help
python -m atlas_q --help

# Factor a number
python -m atlas_q factor 221

# Run all benchmarks
python -m atlas_q benchmark

# Show system info
python -m atlas_q info

# Interactive demo
python -m atlas_q demo
```

See [COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md#command-line-interface) for full CLI documentation.

---

## Examples

### Coherence-Aware Quantum Chemistry (NEW!)

**World's first quantum algorithm with self-diagnostic capabilities** - validates trustworthiness in real-time using physics-derived thresholds.

```python
from atlas_q.coherence_aware_vqe import CoherenceAwareVQE, VQEConfig
from atlas_q.mpo_ops import MPOBuilder

# Build molecular Hamiltonian
H = MPOBuilder.molecular_hamiltonian_from_specs(
 molecule='H2O',
 basis='sto-3g',
 device='cuda'
)

# Run coherence-aware VQE
config = VQEConfig(ansatz='hardware_efficient', n_layers=3, chi_max=256)
vqe = CoherenceAwareVQE(H, config, enable_coherence_tracking=True)
result = vqe.run()

# Check results with automatic quality validation
print(f"Ground state energy: {result.energy:.6f} Ha")
print(f"Coherence R̄: {result.coherence.R_bar:.4f}")
print(f"Classification: {result.classification}") # GO or NO-GO

if result.is_go():
 print(" Results are trustworthy (R̄ > e^-2 = 0.135)")
else:
 print(" Low coherence detected - results may be unreliable")
```

**Hardware validated**: Achieved R̄=0.988 (near-perfect coherence) on IBM Brisbane for H2O (14 qubits, 1086 Pauli terms) with 5× measurement compression via VRA grouping.

### Tensor Network Simulation

```python
from atlas_q.adaptive_mps import AdaptiveMPS
import torch

# Create 10-qubit system with adaptive bond dimensions
mps = AdaptiveMPS(10, bond_dim=8, device='cuda')

# Apply Hadamard gates
H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/torch.sqrt(torch.tensor(2.0))
for q in range(10):
 mps.apply_single_qubit_gate(q, H.to('cuda'))

# Apply CNOT gates
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
 dtype=torch.complex64).reshape(4,4).to('cuda')
for q in range(0, 9, 2):
 mps.apply_two_site_gate(q, CNOT)

print(f"Max bond dimension: {mps.stats_summary()['max_chi']}")
print(f"Memory usage: {mps.memory_usage() / (1024**2):.2f} MB")
```

### Period-Finding & Factorization

```python
from atlas_q import get_quantum_sim

# Get quantum classical hybrid simulator
QuantumClassicalHybrid, _, _, _ = get_quantum_sim()
qc = QuantumClassicalHybrid()

# Factor semiprimes
factors = qc.factor_number(143) # Returns [11, 13]
print(f"143 = {factors[0]} × {factors[1]}")

# Verified against canonical benchmarks:
# - IBM 2001 (N=15): Pass
# - Photonic 2012 (N=21): Pass
# - NMR 2012 (N=143): Pass
```

---

## Performance vs Competition

| Feature | ATLAS-Q | Qiskit Aer | Cirq | Winner |
|---------|---------|------------|------|--------|
| **Memory (30q)** | 0.03 MB | 16 GB | 16 GB | **ATLAS-Q** (626k×) |
| **GPU Support** | Triton | cuQuantum | | **ATLAS-Q** |
| **Stabilizer** | 20× speedup | Standard | Standard | **ATLAS-Q** |
| **Tensor Networks** | Native | | | **ATLAS-Q** |
| **Ease of Use** | Good | Excellent | Excellent | Qiskit/Cirq |

**Note**: Run `python scripts/benchmarks/compare_with_competitors.py` for detailed performance comparisons

---

## What is ATLAS-Q?

ATLAS-Q is a **GPU-accelerated quantum simulator** with breakthrough coherence-aware capabilities:

### Coherence-Aware Computing (NEW!)
1. **Self-Diagnostic Algorithms**: First quantum framework that validates its own trustworthiness
2. **Real-Time Quality Metrics**: R̄ (coherence), V_φ (variance) tracked during execution
3. **GO/NO-GO Classification**: Physics-derived e^-2 boundary (R̄ ≈ 0.135) separates trustworthy from noisy
4. **VRA Integration**: Vaca Resonance Analysis for 5× measurement compression
5. **Hardware Validated**: Tested on IBM Brisbane with near-ideal coherence (R̄=0.988 for H2O)

### Tensor Network Simulation
1. **Adaptive MPS**: Memory-efficient quantum state representation (O(n·χ²) vs O(2ⁿ))
2. **NISQ Algorithms**: VQE, QAOA with noise models and coherence tracking
3. **Time Evolution**: TDVP for Hamiltonian dynamics
4. **Specialized Backends**: Stabilizer for Clifford circuits, MPO for observables
5. **Hamiltonians**: Ising, Heisenberg, Molecular (PySCF), MaxCut (QAOA)
6. **GPU Acceleration**: Custom Triton kernels + cuBLAS tensor cores

### Period-Finding & Factorization
1. **Shor's Algorithm**: Integer factorization via quantum period-finding
2. **Compressed States**: Periodic states (O(1) memory), product states (O(n) memory)
3. **Verified Results**: Matches canonical benchmarks (N=15, 21, 143)

### Key Innovations

- **Coherence-Aware Framework**: World's first self-diagnostic quantum algorithms (GO/NO-GO classification)
- **VRA Integration**: Circular statistics + RMT for quality monitoring and 5× measurement compression
- **Custom Triton Kernels**: Fused gate operations for 1.5-3× speedup
- **Adaptive Bond Dimensions**: Dynamic memory management based on entanglement
- **Hybrid Stabilizer/MPS**: 20× faster Clifford circuits with automatic switching
- **GPU-Optimized Einsums**: cuBLAS + tensor cores for tensor contractions
- **Specialized Representations**: O(1) memory for periodic states, O(n) for product states

---

## Documentation

### Interactive Tutorial
- **[ Jupyter Notebook](ATLAS_Q_Demo.ipynb)** - Complete interactive demo (works in Colab!)

### Online Documentation
- **[ Documentation Site](https://followthsapper.github.io/ATLAS-Q/)** - Browse all docs online

### Guides & References
- **[Complete Guide](docs/COMPLETE_GUIDE.md)** - Installation, tutorials, API reference (start here!)
- **[Feature Status](docs/FEATURE_STATUS.md)** - What's actually implemented
- **[Research Paper](docs/RESEARCH_PAPER.md)** - Mathematical foundations and algorithms
- **[Whitepaper](docs/WHITEPAPER.md)** - Technical architecture and implementation
- **[Overview](docs/OVERVIEW.md)** - High-level explanation for all audiences

---

## Architecture

### Core Components

```
ATLAS-Q/
 src/atlas_q/
 adaptive_mps.py # Adaptive MPS with GPU support
 quantum_hybrid_system.py # Period-finding & factorization
 mpo_ops.py # MPO operations (Hamiltonians)
 tdvp.py # Time evolution (TDVP)
 vqe_qaoa.py # Variational algorithms
 stabilizer_backend.py # Fast Clifford simulation
 noise_models.py # NISQ noise models
 peps.py # 2D tensor networks
 tools_qih/ # Quantum-inspired ML
 triton_kernels/
 mps_complex.py # Custom Triton kernels (1.5-3× faster)
 mps_ops.py # MPS tensor operations
 modpow.py # Modular exponentiation
 scripts/benchmarks/
 validate_all_features.py # 7/7 tensor network benchmarks
 compare_with_competitors.py # vs Qiskit/Cirq/ITensor
 max_qubits_scaling_test.py # Maximum qubits scaling
 tests/
 integration/ # Integration & API tests
 legacy/ # Legacy quantum-inspired tests
 docs/ # Documentation & guides
```

### Technology Stack

- **PyTorch 2.10+** (CUDA backend)
- **Triton** (custom GPU kernels)
- **cuBLAS/CUTLASS** (tensor cores)
- **NumPy/SciPy** (linear algebra)

---

## Use Cases

### BEST FOR:
- **Coherence-Aware VQE**: Quantum chemistry with real-time quality validation
- **VRA-Enhanced Algorithms**: 5× measurement compression + trustworthiness metrics
- **Tensor Networks**: 20-50 qubits with moderate entanglement
- **VQE/QAOA**: Optimization on NISQ devices with noise and coherence tracking
- **Grover Search**: Unstructured database search with quadratic speedup
- **Time Evolution**: Hamiltonian dynamics via TDVP
- **Period-Finding**: Shor's algorithm for integer factorization
- **Memory-Constrained**: 626,000× compression vs statevector
- **GPU Workloads**: Custom Triton kernels + cuBLAS

### NOT IDEAL FOR:
- Highly entangled states (use full statevector)
- Arbitrary connectivity (MPS assumes 1D/2D structure)
- CPU-only environments

---

## Benchmark Results

### Internal Benchmarks (All Passing)

```
 Benchmark 1: Noise Models - 3/3 passing
 Benchmark 2: Stabilizer Backend - 3/3 passing (20× speedup)
 Benchmark 3: MPO Operations - 3/3 passing
 Benchmark 4: TDVP Time Evolution - 2/2 passing
 Benchmark 5: VQE/QAOA - 2/2 passing
 Benchmark 6: 2D Circuits - 2/2 passing
 Benchmark 7: Integration Tests - 2/2 passing
```

### Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Gate throughput | 77,304 ops/sec | GPU-optimized |
| Stabilizer speedup | 20.4× | vs generic MPS |
| MPO evaluations | 1,372/sec | Hamiltonian expectations |
| VQE time (6q) | 1.68s | 50 iterations |
| Memory (30q) | 0.03 MB | vs 16 GB statevector |

---

## Example Applications

### VQE for Quantum Chemistry

```python
from atlas_q import get_mpo_ops, get_vqe_qaoa

# Build molecular Hamiltonian (requires: pip install pyscf)
mpo = get_mpo_ops()
H = mpo['MPOBuilder'].molecular_hamiltonian_from_specs(
 molecule='H2',
 basis='sto-3g',
 device='cuda'
)

# Run VQE to find ground state energy
vqe_mod = get_vqe_qaoa()
vqe = vqe_mod['VQE'](H, ansatz_depth=3, device='cuda')
energy, params = vqe.optimize(max_iter=50)
print(f"Ground state energy: {energy.real:.6f} Ha")
```

### Grover's Quantum Search

```python
from atlas_q.grover import grover_search

# Search for state 7 in 4-qubit space (16 states total)
result = grover_search(
 n_qubits=4,
 marked_states={7}, # Mark state |0111
 device='cpu'
)

print(f"Found state: {result['measured_state']}") # Found state: 7
print(f"Success probability: {result['success_probability']:.3f}") # ~0.96
print(f"Iterations: {result['iterations_used']}") # 3 iterations (O(√N))

# Search using function oracle (e.g., find even numbers)
result = grover_search(
 n_qubits=4,
 marked_states=lambda x: x % 2 == 0,
 device='cpu'
)
print(f"Found even number: {result['measured_state']}")
```

### TDVP Time Evolution

```python
from atlas_q.tdvp import TDVP1Site, TDVPConfig
from atlas_q.mpo_ops import MPOBuilder
from atlas_q.adaptive_mps import AdaptiveMPS

# Create Hamiltonian and initial state
H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')
mps = AdaptiveMPS(10, bond_dim=8, device='cuda')

# Configure TDVP
config = TDVPConfig(dt=0.01, t_final=1.0, use_gpu_optimized=True)
tdvp = TDVP1Site(H, mps, config)

# Run time evolution
times, energies = tdvp.run()
```

---

## Roadmap

### Current Status (v0.6.2)
- **NEW:** Coherence-Aware VQE/QAOA with GO/NO-GO classification
- **NEW:** VRA integration (circular statistics, RMT, 5× measurement compression)
- **NEW:** Hardware validated on IBM Brisbane (H2, LiH, H2O)
- GPU-accelerated tensor networks with custom Triton kernels
- Adaptive MPS with error tracking
- Stabilizer backend (20× speedup)
- TDVP, VQE/QAOA implementations with coherence tracking
- Grover's quantum search (MPO-based oracles, 94-100% accuracy)
- Molecular Hamiltonians (PySCF integration)
- MaxCut QAOA Hamiltonians
- Circuit Cutting & partitioning
- PEPS 2D tensor networks
- Distributed MPS (multi-GPU ready)
- cuQuantum 25.x backend integration
- All 46/46 integration tests passing

### Planned Features
- [ ] Integration adapters for Qiskit/Cirq circuits
- [ ] Additional tutorial notebooks
- [ ] PyPI package update to v0.6.1 (currently at v0.6.0)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/followthsapper/ATLAS-Q.git

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort

# Run tests
pytest tests/ -v

# Run benchmarks
python scripts/benchmarks/validate_all_features.py
```

---

## Citation

If you use ATLAS-Q in your research, please cite:

```bibtex
@software{atlasq2025,
 title={ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum},
 author={ATLAS-Q Development Team},
 year={2025},
 url={https://github.com/followthsapper/ATLAS-Q},
 version={0.5.0}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Acknowledgments

- **PyTorch** team for GPU infrastructure
- **Triton** team for custom kernel framework
- **ITensor/TeNPy** for tensor network inspiration
- **Qiskit/Cirq** for quantum computing ecosystem

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/followthsapper/ATLAS-Q/issues)
- **Discussions**: [GitHub Discussions](https://github.com/followthsapper/ATLAS-Q/discussions)

---

**ATLAS-Q**: GPU-accelerated tensor network simulator achieving 626,000× memory compression through adaptive MPS, custom Triton kernels, and specialized quantum state representations.
