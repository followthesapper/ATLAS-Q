# ATLAS-Q Documentation
**Adaptive Tensor Learning And Simulation – Quantum**

<div style="text-align: center; margin: 2rem 0;">
 <span class="badge">Version 0.5.0</span>
 <span class="badge">77K+ ops/sec</span>
 <span class="badge">626,000× compression</span>
 <span class="badge">7/7 tests passing</span>
</div>

---

## Welcome to ATLAS-Q

ATLAS-Q is a GPU-accelerated quantum tensor network simulator featuring:

- **77K+ ops/sec** gate throughput (GPU-optimized)
- **626,000× memory compression** vs full statevector (30 qubits)
- **20× speedup** on Clifford circuits (Stabilizer backend)
- **Custom Triton kernels** for 1.5-3× gate operation speedup

---

## Documentation

### Getting Started
- **[Complete Guide](COMPLETE_GUIDE)** - Installation, tutorials, API reference (start here!)
- **[Feature Status](FEATURE_STATUS)** - What's actually implemented

### Technical Documentation
- **[Overview](OVERVIEW)** - High-level explanation for all audiences
- **[Whitepaper](WHITEPAPER)** - Technical architecture and implementation
- **[Research Paper](RESEARCH_PAPER)** - Mathematical foundations and algorithms

---

## Quick Start

### Installation

```bash
# From PyPI
pip install atlas-quantum[gpu]

# From source
git clone https://github.com/followthsapper/ATLAS-Q.git
cd ATLAS-Q
pip install -e .[gpu]
```

### First Example

```python
from atlas_q import get_quantum_sim

# Factor a number using Shor's algorithm
QCH, _, _, _ = get_quantum_sim()
sim = QCH()
factors = sim.factor_number(221)
print(f"221 = {factors[0]} × {factors[1]}") # 221 = 13 × 17
```

---

## Key Features

### 1. Adaptive Matrix Product States (MPS)
Memory-efficient quantum state representation with automatic bond dimension adaptation.

**Use case:** Simulate 20-50 qubits with moderate entanglement

### 2. Period-Finding & Factorization
Integer factorization using Shor's algorithm with compressed quantum states.

**Use case:** Factor semiprimes, quantum period-finding

### 3. NISQ Noise Models
Realistic noise simulation with Kraus operators for near-term quantum devices.

**Use case:** Test algorithms on noisy quantum hardware

### 4. Stabilizer Backend
Fast Clifford circuit simulation with 20× speedup vs generic MPS.

**Use case:** Clifford circuits, quantum error correction codes

### 5. VQE/QAOA
Variational quantum algorithms for optimization problems.

**Use case:** Ground state estimation, combinatorial optimization

### 6. Time Evolution (TDVP)
Simulate quantum dynamics using Time-Dependent Variational Principle.

**Use case:** Hamiltonian time evolution, quantum dynamics

### 7. 2D Circuit Support
Automatic SWAP insertion for grid-based quantum processors.

**Use case:** Match connectivity of real quantum hardware

---

## Performance Highlights

| Metric | ATLAS-Q | Qiskit Aer | Cirq |
|--------|---------|------------|------|
| **Memory (30q)** | 0.03 MB | 16 GB | 16 GB |
| **GPU Support** | Triton | cuQuantum | |
| **Stabilizer Speedup** | 20× | Standard | Standard |
| **Tensor Networks** | Native | | |

---

## External Links

- **[GitHub Repository](https://github.com/followthsapper/ATLAS-Q)** - Source code
- **[Issues](https://github.com/followthsapper/ATLAS-Q/issues)** - Bug reports & feature requests
- **[Discussions](https://github.com/followthsapper/ATLAS-Q/discussions)** - Community Q&A
- **[Demo Notebook](https://github.com/followthsapper/ATLAS-Q/blob/main/ATLAS_Q_Demo.ipynb)** - Interactive Jupyter demo

---

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/followthsapper/ATLAS-Q/blob/main/CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

---

## License

ATLAS-Q is released under the [MIT License](https://github.com/followthsapper/ATLAS-Q/blob/main/LICENSE).

---

## Support

- **Bug Reports:** [GitHub Issues](https://github.com/followthsapper/ATLAS-Q/issues)
- **Questions:** [GitHub Discussions](https://github.com/followthsapper/ATLAS-Q/discussions)
- **Email:** Check repository for contact information

---

**Making quantum simulation accessible through honest, working code.**

*Last updated: October 2025*
