# Triton Kernels (Quantum Simulator)

Custom Triton/CUDA kernels for quantum simulation and tensor network operations.

## Kernels

### `modpow.py` - Modular Exponentiation
Batched modular exponentiation for period-finding algorithms.

**Used by:** Shor's algorithm, period-finding
**Speedup:** 100-1000× vs CPU for large batches

```python
from triton_kernels.modpow import batched_modpow_kernel
results = batched_modpow_kernel(base, exponents, modulus)
```

### `mps_ops.py` - MPS Tensor Operations
Tensor contractions and operations for Matrix Product States.

**Used by:** MPS backend, tensor network simulations
**Speedup:** 10-20× vs CPU for χ ≥ 32

```python
from triton_kernels.mps_ops import contract_tensors
result = contract_tensors(A, B, mode='left')
```

### `mps_complex.py` - Complex MPS Operations
Complex-valued tensor operations for quantum states.

**Used by:** Quantum state evolution, complex amplitudes
**Features:** Native complex arithmetic, fused operations

### `linproj_bmm.py` - Low-Rank Projection
Linformer-style low-rank projection for attention.

**Used by:** AQED LowRank variant (lowrank_aqed_layer.py)
**Speedup:** 2× vs sequential K/V projection

```python
from triton_kernels.linproj_bmm import linproj_bmm_fused
K_proj, V_proj = linproj_bmm_fused(E_matrix, K, V)
```

## Requirements

- Triton ≥ 2.0
- PyTorch ≥ 2.0
- CUDA-capable GPU

## Setup for GB10 (DGX Spark)

```bash
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
```

## Note

For **AQED transformer kernels** (routing, mixer, packed attention), see:
`transformers/triton_kernels/README.md`
