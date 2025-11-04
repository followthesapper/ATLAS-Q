# ATLAS-Q Installation Methods

This document describes all the ways to install ATLAS-Q v0.6.2.

## Quick Start (Recommended)

### Option 1: PyPI with pip

```bash
# Basic installation
pip install atlas-quantum

# With GPU support
pip install atlas-quantum[gpu]

# With all features
pip install atlas-quantum[all]
```

### Option 2: uv (Faster Alternative)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer written in Rust:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ATLAS-Q with uv
uv pip install atlas-quantum

# With GPU support
uv pip install atlas-quantum[gpu]
```

Benefits of uv:
- 10-100x faster than pip
- Better dependency resolution
- Automatically handles virtual environments

## Advanced Installation Methods

### From Source with pip

```bash
git clone https://github.com/followthesapper/ATLAS-Q.git
cd ATLAS-Q
pip install -e .[gpu,chemistry,dev]
```

### From Source with uv

```bash
git clone https://github.com/followthesapper/ATLAS-Q.git
cd ATLAS-Q
uv pip install -e .[gpu,chemistry,dev]
```

### Docker

#### GPU Version

```bash
docker pull ghcr.io/followthesapper/atlas-q:cuda
docker run --rm -it --gpus all ghcr.io/followthesapper/atlas-q:cuda
```

#### CPU Version

```bash
docker pull ghcr.io/followthesapper/atlas-q:cpu
docker run --rm -it ghcr.io/followthesapper/atlas-q:cpu
```

### Conda-forge (Coming Soon)

```bash
# NOTE: Submission pending approval
conda install -c conda-forge atlas-quantum
```

Status: Package submitted to conda-forge, awaiting review. Check:
https://github.com/conda-forge/staged-recipes/pulls

### System Package Manager (APT) - Manual Setup

For Debian/Ubuntu systems, you can create a local package:

#### Step 1: Install dependencies

```bash
sudo apt update
sudo apt install python3-pip python3-numpy python3-scipy python3-matplotlib
```

#### Step 2: Install ATLAS-Q via pip

```bash
pip3 install atlas-quantum --user
```

#### Step 3: Make it system-wide (optional)

```bash
sudo pip3 install atlas-quantum
```

**Note**: We are working on official .deb packages for Ubuntu/Debian. Until then, use pip with --user flag or in a virtual environment.

## Installation Options

### Core Dependencies Only

```bash
pip install atlas-quantum
```

Includes:
- numpy >= 1.22
- scipy >= 1.10.0
- matplotlib >= 3.6
- torch >= 2.0.0

### GPU Acceleration

```bash
pip install atlas-quantum[gpu]
```

Adds:
- triton >= 2.0.0 (custom GPU kernels, 1.5-3x speedup)

### cuQuantum Backend

```bash
pip install atlas-quantum[cuquantum]
```

Adds:
- cuquantum-python >= 23.0.0 (2-10x speedup on NVIDIA GPUs)

### Quantum Chemistry

```bash
pip install atlas-quantum[chemistry]
```

Adds:
- pyscf >= 2.0.0 (molecular Hamiltonians)
- openfermion >= 1.5.0 (fermion-qubit mappings)
- openfermionpyscf >= 0.5.0 (PySCF integration)

### Machine Learning Tools

```bash
pip install atlas-quantum[ml]
```

Adds:
- scikit-learn >= 1.0.0 (rank prediction, AI-guided truncation)

### Development Tools

```bash
pip install atlas-quantum[dev]
```

Adds:
- pytest >= 7.3
- pytest-cov >= 4.0
- black >= 23.0
- ruff >= 0.0.260

### All Features

```bash
pip install atlas-quantum[all]
```

Includes all optional dependencies.

## Verifying Installation

```python
import atlas_q
print(f"ATLAS-Q version: {atlas_q.__version__}")

# Test basic functionality
from atlas_q import get_quantum_sim
QCH, _, _, _ = get_quantum_sim()
sim = QCH()
factors = sim.factor_number(221)
print(f"221 = {factors[0]} × {factors[1]}")  # Should print: 221 = 13 × 17
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies first
sudo apt install python3-dev libopenblas-dev

# Then install ATLAS-Q
pip install atlas-quantum[gpu]
```

### macOS

```bash
# Install via Homebrew (if needed)
brew install python@3.12

# Install ATLAS-Q
pip3 install atlas-quantum
```

Note: GPU acceleration (Triton/cuQuantum) requires NVIDIA GPU, not available on macOS.

### Windows

```bash
# Install via conda or pip
pip install atlas-quantum

# For GPU support, ensure CUDA is installed
pip install atlas-quantum[gpu]
```

## Troubleshooting

### Import Error: "No module named 'atlas_q'"

Solution: Ensure you activated the correct environment:

```bash
# With pip
python -m pip list | grep atlas

# With uv
uv pip list | grep atlas
```

### CUDA Not Found

Solution: Install CUDA toolkit:

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or use conda
conda install cudatoolkit=11.8
```

### Triton Kernel Compilation Errors

Solution: Set CUDA architecture:

```bash
export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
```

## Getting Help

- **Documentation**: https://github.com/followthesapper/ATLAS-Q/tree/main/docs
- **Issues**: https://github.com/followthesapper/ATLAS-Q/issues
- **Discussions**: https://github.com/followthesapper/ATLAS-Q/discussions

## Next Steps

After installation, see:
- `examples/coherence_aware_vqe_example.py` for usage examples
- `docs/COMPLETE_GUIDE.md` for full tutorial
- `docs/user_guide/coherence_aware_vqe.rst` for coherence-aware VQE
