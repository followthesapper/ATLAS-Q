# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-25

### Added
- **Documentation Overhaul:** Separated AQED and Quantum Simulator into dedicated whitepapers
- `AQED_WHITEPAPER.md`: Comprehensive technical documentation for transformer optimization
- `QUANTUM_SIMULATOR_WHITEPAPER.md`: Detailed quantum simulation algorithms documentation
- `AQED_USAGE_GUIDE.md`: Practical training tutorials
- `QUANTUM_SIMULATOR_USAGE_GUIDE.md`: Hands-on quantum simulation examples
- Unified `README.md` covering both major innovations
- `CONTRIBUTING.md`: Contribution guidelines
- `LICENSE`: MIT license
- `CHANGELOG.md`: This file

### Changed
- Reorganized documentation structure for clarity
- Archived legacy status/summary documents to `docs/archive/` and `archive/`
- Cleaned up root directory (moved logs, profiling data, old scripts to archive)

### Removed
- Redundant status documents from root (moved to archive)
- Old combined whitepaper (moved to archive)
- Old usage guide (moved to archive)

## [0.2.0] - 2025-10-24

### Added
- **AQED (Adaptive Quantum Entanglement Diffusion)** transformer architecture
- Proven 6-10× speedup over traditional transformers
- torch.compile integration for GB10 (NVIDIA DGX Spark)
- PyTorch SDPA (Scaled Dot-Product Attention) for memory efficiency
- Comprehensive benchmarking suite (L=2048, 4096, 8192)
- Low-rank AQED variant with Linformer projection
- Hybrid AQED layer with MPS-based attention

### Changed
- Updated to PyTorch 2.10 nightly
- Improved MPS SVD stability with multi-driver fallback
- Enhanced period-finding with GPU acceleration

### Performance
- L=4096: 6.18× speedup (211k vs 34k tokens/sec)
- L=8192: 10.59× speedup (197k vs 19k tokens/sec)
- Loss quality maintained (Δloss ≤ 0.02)

## [0.1.0] - 2024-10-01

### Added
- Initial release
- Quantum state compression (Periodic, Product, MPS)
- O(√r) period-finding algorithms
- Shor's algorithm simulation
- Matrix Product State (MPS) backend
- GPU acceleration with CuPy
- Quantum-inspired ML features (QIH)
- 21 interactive Jupyter notebooks
- Comprehensive test suite (26 test files)

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Security fixes
- `Performance`: Performance improvements

### Changed (v0.3.0 - continued)
- **Consolidated test suite:** Merged `Tests/` and `tests/` into single `tests/` directory
- **Cleaned runs/ folder:** Reduced from 46MB to 232KB by archiving experimental data
- **Archived old diagrams:** Moved `Diagrams/` → `archive/diagrams/` (pre-v0.3.0 visualizations)
- Archived old `Tests/` directory → `archive/Tests_old/`
- Archived experimental runs → `archive/runs_experiments/` (ftdata, svd_logs, alpha sweeps, etc.)
- Added READMEs to both `triton_kernels/` directories explaining their different purposes

### Documentation
- Added `tests/README.md` - Comprehensive test suite documentation
- Added `triton_kernels/README.md` - Quantum simulator kernels documentation
- Added `transformers/triton_kernels/README.md` - AQED kernels documentation
- Added `runs/README.md` - Benchmark results documentation
- Added `archive/diagrams/README.md` - Archived diagrams documentation
- Added `archive/runs_experiments/README.md` - Experimental data documentation
- Added `docs/figures/README.md` - Diagram documentation
- Updated `PROJECT_STRUCTURE.md` - Reflects cleaned structure and docs/ location

### Visual Documentation (New)
- Added `docs/figures/aqed_architecture_comparison.svg` - AQED vs Traditional vs LowRank architecture comparison
- Added `docs/figures/aqed_performance_comparison.svg` - Tests 1-6 benchmark results visualization
- Added `docs/figures/quantum_simulator_comparison.svg` - Memory/complexity comparison with traditional simulators
- Updated whitepapers and README with diagram references
