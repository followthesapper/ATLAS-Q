# ATLAS-Q Setup Summary

**Date**: October 26, 2025

## Quick Reference

### Installation
```bash
# Clone and install
git clone https://github.com/yourusername/ATLAS-Q.git
cd ATLAS-Q
pip install -r requirements.txt
pip install -e .

# Setup GPU acceleration (recommended)
./setup_triton.sh
```

### Development
```bash
make help              # Show all available commands
make test              # Run all tests
make test-unit         # Run unit tests
make bench             # Run benchmarks
make demo              # Run demonstrations
```

---

## Files Updated (October 26, 2025)

### 1. **requirements.txt** ✅
**Added:**
- `triton>=2.0.0` (critical - was missing!)
- `scikit-learn>=1.0.0` (for AI model training scripts)

### 2. **MANIFEST.in** ✅
**Fixed:**
- Updated `Tests` → `tests` (correct case)
- Removed references to archived Notebooks/
- Removed old AQED transformers/
- Added `benchmarks/`, `triton_kernels/`, `models/`
- Added docs/, README files

### 3. **Makefile** ✅
**Improved:**
- Added `make help` with all commands
- Split tests: `test-unit`, `test-integration`, `test-performance`
- Added `make install` and `make dev-install`
- Added `make clean` for build artifacts
- Fixed outdated notebook references

### 4. **setup_triton.sh** (NEW) ✅
**Created automated setup script:**
- Checks CUDA installation
- Sets required environment variables:
  - `TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"`
  - `TORCH_CUDA_ARCH_LIST="12.0"`
- Adds to ~/.bashrc for persistence
- Tests Triton kernel compilation
- Provides clear success/failure messages

**Usage:**
```bash
./setup_triton.sh
```

### 5. **README.md** ✅
**Added:**
- GPU Setup section with Triton environment variables
- `pip install -e .` to installation instructions
- Reference to `setup_triton.sh` script
- Note about GPU architecture configuration

### 6. **README files created/updated:**
- `scripts/README.md` - Documents all 25 working scripts
- `models/README.md` - Documents 3 AI models and training workflow
- `tests/README.md` - Updated for new 4-directory structure
- `triton_kernels/README.md` - Removed AQED references

---

## Package Structure

### Clean Package Layout
```
ATLAS-Q/
├── src/atlas_q/              # Main package (renamed from quantum_hybrid_system)
├── triton_kernels/           # Custom GPU kernels
├── models/                   # 3 trained AI models (2.6 MB)
├── scripts/                  # 25 working scripts (demos, benchmarks, AI tools)
├── tests/                    # Organized into 4 subdirectories
│   ├── unit/                 # 9 core component tests
│   ├── integration/          # 6 multi-component tests
│   ├── performance/          # 2 GPU/performance tests
│   └── legacy/               # 15 old tests
├── benchmarks/               # Comprehensive benchmark suite
├── docs/                     # Documentation
└── archive/                  # Archived old content
```

---

## Triton GPU Acceleration

### Why These Environment Variables?

**TRITON_PTXAS_PATH**: Points to CUDA assembler for compiling Triton kernels
**TORCH_CUDA_ARCH_LIST**: Forces compilation for CUDA 12.0 (compatible with GB10/H100)

Without these, you'll see:
```
ptxas fatal: Value 'sm_121a' is not defined for option 'gpu-name'
```

### Performance Impact

With Triton kernels enabled:
- **1.5-3× speedup** on MPS operations
- **100-1000× speedup** on modular exponentiation (period-finding)
- **10-20× speedup** on tensor contractions (bond dimension χ ≥ 32)

### Verification

Test if Triton is working:
```bash
python3 -c "from triton_kernels.modpow import batched_modpow_triton; \
batched_modpow_triton(7, [1,2,3], 899, 'cuda'); print('✅ Triton working!')"
```

---

## Archived Content

Moved to `archive/` directory:
- `archive/notebooks_old/` - 16 notebooks (broken imports)
- `archive/demos/` - real_world_tests (marketing demos)
- `archive/old_scripts/` - 4 broken AQED benchmark scripts

---

## Testing

### Run Tests
```bash
# All tests
pytest

# By category
pytest tests/unit/              # Fast, isolated tests
pytest tests/integration/        # Multi-component tests
pytest tests/performance/        # GPU benchmarks

# Skip GPU tests
pytest -m "not gpu"

# With coverage
pytest --cov=atlas_q --cov-report=html
```

### Run Benchmarks
```bash
# Comprehensive benchmark suite
python benchmarks/comprehensive_benchmark.py

# Specific benchmarks
python scripts/benchmark_simulator.py
python scripts/qubit_capacity_probe.py
```

---

## AI Models

Three trained models in `models/`:
1. **rank_predictor.pt** (1.8 MB) - Base model
2. **rank_predictor_ft.pt** (407 KB) - Fine-tuned
3. **rank_predictor_ft_compiled.pt** (415 KB) - Compiled for production

Used for AI-guided MPS truncation (40% memory savings, 1.3× speedup)

See `models/README.md` for training instructions.

---

## Common Issues

### Issue: "triton_kernels not found"
**Solution**: Run `pip install triton>=2.0.0`

### Issue: "ptxas fatal: Value 'sm_121a' is not defined"
**Solution**: Run `./setup_triton.sh` or set environment variables manually

### Issue: Tests failing with old imports
**Solution**: Package was renamed. Use:
```python
from atlas_q import get_quantum_sim  # Correct
# NOT: from quantum_hybrid_system import ...  # Old
```

### Issue: "module 'atlas_q' has no attribute 'X'"
**Solution**: Reinstall package: `pip install -e .`

---

## What Changed vs Previous Session

### Package Rename
- `quantum_hybrid_system` → `atlas_q`
- 197 files updated with new imports
- All tests passing with new package name

### Documentation Unified
- Period-finding (original) + Tensor networks (v0.5.0) = Unified ATLAS-Q
- Updated whitepaper, research paper, usage guide
- All docs now in `docs/` directory

### File Organization
- Tests organized into subdirectories
- Scripts cleaned (25 working, 4 archived)
- Created READMEs for all major directories

### Critical Fixes
- Added missing `triton` to requirements.txt
- Fixed MANIFEST.in with correct paths
- Improved Makefile with better targets
- Created automated Triton setup script

---

## Next Steps

1. **For new users**: Follow Quick Start in README.md
2. **For developers**: Run `make dev-install` then `make test`
3. **For GPU users**: Run `./setup_triton.sh` for optimal performance
4. **For contributors**: See `archive/CONTRIBUTING.md`

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: `docs/` directory
- **Examples**: `scripts/` directory (25 working scripts)
- **Tests**: `tests/` directory (organized by type)

---

**Status**: ✅ All configuration files updated and tested
**Version**: 0.5.0
**Last Updated**: October 26, 2025
