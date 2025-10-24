# Changelog

All notable changes to the Quantum Hybrid Simulator project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-24

### Added
- **Comprehensive package metadata** with classifiers for PyPI
- **AQED (Adaptive Quantum Entanglement Diffusion)** mixer for transformers
  - Routed hybrid attention/mixer architecture
  - Adaptive controller with loss-aware knobs
  - Vectorized implementation with torch.compile support
- **AI-assisted SVD compression** with learned rank prediction
- **Extended QIH features**:
  - Periodic mixture features for multi-period signals
  - 2D texture/spectrogram features
  - Chirp features for drift tracking
  - NTT (Number Theoretic Transform) features
- **Comprehensive documentation**:
  - Technical whitepaper (`WHITEPAPER.md`)
  - Usage guide (`USAGE_GUIDE.md`)
  - API reference documentation
  - 21 example notebooks
- **Enhanced testing** with 26 test files covering all major components
- **Transformer training pipelines**:
  - Baseline transformer
  - AQED mixer variants (fixed, adaptive, hybrid)
  - Routed expert selection
  - Comprehensive benchmarking scripts
- **Packaging improvements**:
  - Optional dependencies for ml, gpu, hf, docs, notebooks
  - MANIFEST.in for proper source distribution
  - Development tools configuration (black, ruff, mypy)

### Changed
- **Updated pyproject.toml** to version 0.2.0 with comprehensive metadata
- **Improved tensor network core** with robust multi-driver SVD
- **Enhanced MPS operations** with adaptive truncation
- **Optimized GPU acceleration** with automatic CPU fallback

### Fixed
- **SVD numerical stability** with jitter injection and fallback drivers
- **MPS canonicalization** for accurate sampling
- **GPU memory management** in tensor contractions
- **Batch dimension handling** in transformer training

## [0.1.0] - 2024-10-22

### Added
- **Core quantum state representations**:
  - `PeriodicState` - O(1) memory periodic states
  - `ProductState` - O(n) memory separable states
  - `MatrixProductState` - O(n×χ²) memory entangled states
- **Period-finding algorithms** with O(√r) complexity:
  - Smart factorization
  - Parallel candidate search
  - Pollard's rho
  - Collision detection
  - Baby-step giant-step
  - GPU-accelerated batched checking
- **Analytic QFT sampling** from periodic states
- **Quantum circuit emulation** with tensor contractions
- **GPU acceleration** via CuPy (optional)
- **QIH (Quantum-Inspired Histogram) features** for ML
- **Hybrid quantum-classical system** for factorization
- **Comprehensive test suite** (26 tests)
- **Example notebooks** (21 notebooks) covering:
  - Getting started
  - Period finding with QFT
  - Compressed states
  - MPS canonicalization
  - GPU acceleration
  - Circuit emulation
  - RSA factorization
  - ML workflows
  - Time series analysis
- **Basic documentation**:
  - README with quick start
  - CONTRIBUTING guidelines
  - MIT LICENSE

### Dependencies
- numpy >= 1.22
- matplotlib >= 3.6
- Python >= 3.9

### Optional Dependencies
- PyTorch for ML components
- scikit-learn for ML tools
- CuPy for GPU acceleration
- Hugging Face transformers for LLM integration

---

## Release Notes

### Version 0.2.0 Highlights

This release represents a major step forward in both functionality and documentation:

1. **AQED Innovation**: The new Adaptive Quantum Entanglement Diffusion mixer brings quantum-inspired dynamics to transformer architectures, showing up to 12% throughput improvements with comparable loss in benchmarks.

2. **Production-Ready Packaging**: Comprehensive metadata, classifiers, and optional dependencies make the package ready for PyPI distribution and easy integration into research workflows.

3. **Documentation Overhaul**: The addition of a technical whitepaper, usage guide, and API documentation makes the theory and practice accessible to researchers and developers.

4. **ML Integration**: Enhanced transformer training pipelines with adaptive controllers, routed expert selection, and comprehensive benchmarking demonstrate the practical value of quantum-inspired techniques.

5. **Robustness Improvements**: Multi-driver SVD with automatic fallbacks, jitter injection for numerical stability, and comprehensive testing ensure reliability across different hardware and use cases.

### Upgrade Guide

If upgrading from 0.1.0:

- **Breaking changes**: None. All 0.1.0 APIs remain compatible.
- **New features**: Explore AQED mixer in `transformers/` directory
- **Dependencies**: Optionally install `[ml]` extras for transformer components
- **Documentation**: See `WHITEPAPER.md` and `USAGE_GUIDE.md` for detailed guides

### Installation

```bash
# Basic installation
pip install quantum-hybrid-simulator

# With machine learning components
pip install quantum-hybrid-simulator[ml]

# With GPU acceleration
pip install quantum-hybrid-simulator[gpu]

# Everything (development)
pip install quantum-hybrid-simulator[all]

# From source (development mode)
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[dev]
```

### Known Issues

- AQED controller adaptive mode shows mixed accuracy results in some configurations (see benchmark results in `runs/`)
- GPU acceleration requires CUDA 12.x; older CUDA versions not tested
- Some transformer configurations at L=512 show overhead that only pays off at longer sequences (L≥1024)
- CuPy dependency resolution can be tricky on ARM platforms (aarch64)

### Roadmap for 0.3.0

Planned features for the next release:

- [ ] Torch.compile integration for AQED components
- [ ] Flash Attention 2/3 integration
- [ ] FP8 quantization support
- [ ] Triton kernels for mixer operations
- [ ] Extended sequence length support (L≥8k)
- [ ] Pre-trained AQED checkpoints
- [ ] Sphinx-generated API documentation
- [ ] ReadTheDocs integration
- [ ] Performance profiling guide
- [ ] Docker images for reproducible environments

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
