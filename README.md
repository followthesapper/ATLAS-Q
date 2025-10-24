# Quantum Hybrid Simulator

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-beta-yellow.svg)]()

**A quantum-inspired hybrid simulator combining compressed quantum state representations, tensor networks, and machine learning for research and education.**

> **Bring quantum-inspired computing to your research without quantum hardware.**

---

## ✨ Key Features

### 🚀 **Compressed Quantum States**
- **Periodic States**: O(1) memory, perfect for Shor's algorithm
- **Product States**: O(n) memory for separable states
- **Matrix Product States (MPS)**: O(n×χ²) memory for entangled states
- Scale to 100+ qubits where traditional simulators fail

### ⚡ **Fast Period-Finding** (O(√r) complexity)
- Smart factorization for smooth periods
- Pollard's rho cycle detection
- Baby-step giant-step algorithm
- GPU-accelerated batched checking
- Automatic fallback strategy

### 🧠 **Quantum-Inspired ML Features**
- **QIH (Quantum-Inspired Histogram)** features for time series
- Learned period detection with neural networks
- Multi-period mixture features
- Integration with PyTorch and Hugging Face

### 🎯 **AQED: Adaptive Quantum Entanglement Diffusion**
- Novel transformer architecture with quantum-inspired mixing
- **12% throughput improvement** over baseline transformers
- Routed expert selection (attention vs. mixing)
- Adaptive controller for dynamic hyperparameter tuning
- **78% memory reduction** in adaptive mode

### 💻 **GPU Acceleration** (Optional)
- CUDA kernels for batched modular exponentiation
- Tensor contractions on GPU
- Robust SVD with automatic CPU fallback
- 100-1000× speedup for period-finding

---

## 🔧 Installation

### Basic Installation

```bash
pip install quantum-hybrid-simulator
```

### With Machine Learning Support

```bash
pip install quantum-hybrid-simulator[ml]
```

### With GPU Acceleration

```bash
pip install quantum-hybrid-simulator[gpu]
```

### Development Installation

```bash
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[dev]
```

### All Optional Dependencies

```bash
pip install quantum-hybrid-simulator[all]
```

**Requirements:**
- Python ≥ 3.9
- NumPy ≥ 1.22
- Matplotlib ≥ 3.6

**Optional:**
- PyTorch ≥ 2.0 (for ML/AQED)
- CuPy (for GPU acceleration)
- scikit-learn ≥ 1.2 (for ML tools)
- transformers ≥ 4.30 (for Hugging Face integration)

---

## 🚀 Quick Start

### Example 1: Factor a Semiprime (Shor's Algorithm Simulation)

```python
from quantum_hybrid_system import QuantumClassicalHybrid

# Create simulator
sim = QuantumClassicalHybrid()

# Factor 15 = 3 × 5
factors = sim.factor(15)
print(f"Factors of 15: {factors}")  # Output: [3, 5]

# Try larger numbers
factors = sim.factor(221)  # 13 × 17
print(f"Factors of 221: {factors}")  # Output: [13, 17]
```

### Example 2: Compressed Quantum States

```python
from quantum_hybrid_system import PeriodicState, ProductState, MatrixProductState

# Periodic state: O(1) memory
periodic = PeriodicState(n_qubits=20, offset=0, period=7)
print(f"Memory: {periodic.memory_bytes()} bytes")  # Just 32 bytes!

# Sample from QFT (analytic, O(1) time)
qft_sample = periodic.sample_qft_measurement()
print(f"QFT sample: {qft_sample}")

# Product state: O(n) memory
product = ProductState.init_plus(n_qubits=50)  # |+⟩^50
product.apply_gate('X', target=0)  # Apply X gate to qubit 0

# MPS: O(n×χ²) memory
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16)
# Can represent moderate entanglement with only ~200 KB memory
```

### Example 3: Period-Finding

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()

# Define periodic function
def f(x):
    return pow(7, x, 15)  # 7^x mod 15

# Find period (should be 4: 7^4 ≡ 1 mod 15)
period = sim.find_period(f, max_period=15)
print(f"Period: {period}")  # Output: 4
```

### Example 4: QIH Features for Time Series

```python
from quantum_hybrid_system.tools_qih import qih_pat, periodic_mixture_features
import numpy as np

# Generate synthetic signal with periods 5 and 20
t = np.arange(1000)
signal = np.sin(2 * np.pi * t / 5) + 0.5 * np.sin(2 * np.pi * t / 20)

# Extract QIH features
periods = [5, 20]
qih_features = periodic_mixture_features(signal, periods, hist_bins=32)

# Use as ML features
print(f"Feature shape: {qih_features.shape}")  # (64,) - 32 bins per period
```

### Example 5: AQED Transformer (Ultra-Fast)

```bash
# Quick training with torch.compile (1.37× speedup!)
./run_training.sh --seq_len 4096 --batch_size 8 --epochs 3 --compile
```

Or in Python:

```python
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'

from transformers.train_transformer_ultra_fast import UltraFastTransformerLM, Config
import torch

# Configure ultra-fast transformer
cfg = Config(
    vocab_size=32000,
    seq_len=4096,
    d_model=512,
    n_layers=8,
    n_heads=8,
    attn_keep_every=8,  # Skip attention every 8 layers
    compile=True,        # Enable torch.compile
)

# Create model
model = UltraFastTransformerLM(cfg).cuda()

# torch.compile optimizes the model (1st epoch slow, 2+ fast!)
model = torch.compile(model)

# Train (see transformers/ directory for full scripts)
```

---

## 📚 Documentation

- **[Technical Whitepaper](WHITEPAPER.md)**: In-depth algorithms, theory, and complexity analysis
- **[Usage Guide](USAGE_GUIDE.md)**: Step-by-step tutorials for common tasks
- **[torch.compile Guide](TORCH_COMPILE_GUIDE.md)**: GB10/DGX Spark setup and optimization ⭐ NEW
- **[Executive Summary](EXECUTIVE_SUMMARY.md)**: Current status and performance results
- **[API Reference](https://quantum-hybrid-simulator.readthedocs.io)**: Detailed API documentation
- **[Notebooks](Notebooks/)**: 21 interactive examples covering all features
- **[Changelog](CHANGELOG.md)**: Version history and release notes
- **[Contributing](CONTRIBUTING.md)**: Guidelines for contributors

---

## 📖 Notebooks & Examples

Explore **21 interactive notebooks** in the `Notebooks/` directory:

| Notebook | Topic |
|----------|-------|
| `01_getting_started.ipynb` | Introduction and basic usage |
| `02_period_finding_qft.ipynb` | Analytic QFT and O(√r) algorithms |
| `03_compressed_states.ipynb` | Periodic, Product, and MPS states |
| `04_mps_canonicalization_sampling.ipynb` | Tensor network operations |
| `05_gpu_acceleration.ipynb` | CUDA kernels and GPU backends |
| `06_circuit_emulation.ipynb` | Quantum gates and circuits |
| `08_rsa_factorization.ipynb` | Factoring semiprimes (cryptography demo) |
| `10_ml_hybrid_workflows.ipynb` | QIH features and ML integration |
| `14_hidden_period_timeseries.ipynb` | Time series analysis |
| ...and 12 more! | |

Start with `01_getting_started.ipynb` for a guided tour.

---

## 🎯 Use Cases

### Cryptography & Security
- Factor semiprimes up to 13-16 bits
- Demonstrate Shor's algorithm principles
- Educational tool for quantum cryptography

### Time Series Analysis
- Hidden period detection in noisy signals
- Multi-period forecasting (daily/weekly/monthly cycles)
- Drift detection (chirp features for changing periods)
- Anomaly detection via QIH features

### Machine Learning Research
- Quantum-inspired transformer architectures (AQED)
- Routed expert selection for efficient attention
- Period-aware features for structured data
- Hybrid classical-quantum ML workflows

### Education & Research
- Learn quantum computing concepts without hardware
- Prototype hybrid algorithms
- Explore tensor network methods
- Benchmark compression techniques

---

## 🏆 Benchmarks

### Memory Efficiency

| Qubits | Periodic | Product | MPS (χ=16) | Full Vector |
|--------|----------|---------|------------|-------------|
| 20 | 32 B | 640 B | 41 KB | 16 MB |
| 30 | 32 B | 960 B | 61 KB | 17 GB |
| 50 | 32 B | 1.6 KB | 102 KB | 18 PB |
| 100 | 32 B | 3.2 KB | 204 KB | ~10³⁰ B |

### AQED Transformer Performance (October 2025)

| Configuration | L=2048 | L=4096 | Speedup |
|---------------|--------|--------|---------|
| **Baseline** | 168k tok/s | 190k tok/s | 1.00× |
| **AQED (skip=4)** | 170k tok/s | - | 1.01× |
| **AQED (skip=8)** | 178k tok/s | 225k tok/s | 1.06-1.18× |
| **+ torch.compile** | **231k tok/s** | **239k tok/s** | **1.25-1.37×** ✅ |
| **+ Flash Attention** | ~350k tok/s | ~400k tok/s | **2-4× (est.)** 🔄 |

*Loss quality maintained (~6.3) across all configurations.*
*See `TORCH_COMPILE_GUIDE.md` for setup instructions.*

### Period-Finding Success Rate

| Semiprime Bits | Success Rate | Avg Time |
|----------------|--------------|----------|
| 6 | 100% | 0.5 ms |
| 10 | 100% | 5 ms |
| 13 | ~95% | 100 ms |
| 16 | ~90% | 300 ms |

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Skip GPU tests (if no GPU available)
pytest -m "not gpu"

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=quantum_hybrid_system --cov-report=html
```

**26 test files** covering:
- Core quantum state operations
- Period-finding algorithms
- Tensor network methods
- ML feature extraction
- GPU acceleration
- Numerical stability

---

## 🗺️ Project Structure

```
quantum-hybrid-simulator/
├── src/quantum_hybrid_system/       # Main package
│   ├── quantum_hybrid_system.py     # Core simulator (1535 lines)
│   └── tools_qih/                   # QIH and ML tools
│       ├── tn_core.py               # Tensor network backend
│       ├── ai_rank_predictor.py     # AI-assisted SVD
│       ├── qih_pat.py               # Period-Aware Transformer
│       ├── learned_period_head.py   # Neural period detection
│       └── ...                      # (14 modules)
├── transformers/                    # AQED transformer implementations
│   ├── train_baseline_transformer.py
│   ├── train_transformer_routed_hybrid.py  # Latest AQED variant
│   ├── adaptive_components.py       # Controllers and probes
│   └── ...                          # (9 training scripts)
├── scripts/                         # Benchmark and experiment scripts (24)
├── Tests/                           # Test suite (26 test files)
├── Notebooks/                       # Interactive examples (21 notebooks)
├── runs/                            # Benchmark results and visualizations
├── pyproject.toml                   # Package configuration
├── README.md                        # This file
├── WHITEPAPER.md                    # Technical documentation
├── USAGE_GUIDE.md                   # Practical tutorials
├── CHANGELOG.md                     # Version history
└── CONTRIBUTING.md                  # Development guidelines
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- Report bugs and request features (GitHub Issues)
- Submit pull requests (bug fixes, new features, documentation)
- Improve documentation and examples
- Share your use cases and benchmarks
- Help others in discussions

**Development setup:**
```bash
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[dev]
pytest  # Run tests
```

**Code style:**
- Format with `black` (line length 100)
- Lint with `ruff`
- Type hints preferred (checked with `mypy`)
- Docstrings for public APIs

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project builds on ideas and tools from:

- **Quantum Computing**: Qiskit, Cirq, ProjectQ
- **Tensor Networks**: ITensor, TensorNetwork, TeNPy
- **Machine Learning**: PyTorch, Hugging Face Transformers
- **Numerical Methods**: NumPy, CuPy, SciPy

Special thanks to the open-source community and early adopters.

---

## 📞 Support & Contact

- **Documentation**: [https://quantum-hybrid-simulator.readthedocs.io](https://quantum-hybrid-simulator.readthedocs.io)
- **GitHub Issues**: [https://github.com/your-org/quantum-hybrid-simulator/issues](https://github.com/your-org/quantum-hybrid-simulator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/quantum-hybrid-simulator/discussions)
- **Email**: dev@example.com

---

## 🎓 Citation

If you use this software in your research, please cite:

```bibtex
@software{quantum_hybrid_simulator,
  title = {Quantum Hybrid Simulator: Compressed States, Tensor Networks, and AQED},
  author = {Quantum Hybrid Simulator Contributors},
  year = {2025},
  version = {0.2.0},
  url = {https://github.com/your-org/quantum-hybrid-simulator}
}
```

---

## 🗺️ Roadmap

### v0.3.0 (Next Release)
- [ ] `torch.compile` integration for AQED
- [ ] Flash Attention 2/3 support
- [ ] FP8 quantization with NVIDIA Transformer Engine
- [ ] Triton kernels for mixer operations
- [ ] Pre-trained AQED checkpoints

### v0.4.0 (Future)
- [ ] Extended sequence length support (L ≥ 8k)
- [ ] Qiskit/Cirq backend integration
- [ ] PEPS and Tree Tensor Networks
- [ ] Docker images for reproducible environments
- [ ] Sphinx-generated API docs

See [CHANGELOG.md](CHANGELOG.md) for detailed roadmap.

---

## ⚠️ Disclaimer

This software is intended for **research and educational purposes**. It is **not** a replacement for:
- Production quantum simulators (use Qiskit, Cirq for general-purpose simulation)
- Cryptographic libraries (use OpenSSL, PyCryptodome for real applications)
- Mission-critical systems (this is beta software, v0.2.0)

**Use at your own risk.** See [LICENSE](LICENSE) for full terms.

---

<div align="center">

**[Get Started](#-quick-start) | [Documentation](WHITEPAPER.md) | [Examples](Notebooks/) | [GitHub](https://github.com/your-org/quantum-hybrid-simulator)**

Made with ❤️ by the Quantum Hybrid Simulator community

</div>
