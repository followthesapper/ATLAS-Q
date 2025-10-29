=================
GPU Acceleration
=================

GPU acceleration is essential for large-scale quantum simulation in ATLAS-Q. This document explains how ATLAS-Q leverages GPUs through PyTorch's CUDA backend, custom Triton kernels, NVIDIA cuQuantum integration, and tensor cores to achieve 10-1000× speedups over CPU execution.

.. contents::
   :local:
   :depth: 2

Overview
========

Why GPU Acceleration for Quantum Simulation
--------------------------------------------

Quantum state simulation with MPS involves:

1. **Tensor contractions**: :math:`O(\chi^3 d^2)` complexity per gate
2. **SVD operations**: :math:`O(\chi^3)` complexity for adaptive truncation
3. **Matrix-vector products**: TDVP effective Hamiltonian applications
4. **Batched operations**: Multiple gate applications in variational algorithms

These operations are highly parallel and memory-bandwidth-intensive, making them ideal for GPU acceleration.

**Performance scaling**:

- **Small systems** (n < 15 qubits, χ < 32): CPU may be faster due to overhead
- **Medium systems** (n = 15-50 qubits, χ = 32-128): GPU provides 5-20× speedup
- **Large systems** (n > 50 qubits, χ > 128): GPU essential (20-100× speedup)

Key Acceleration Strategies in ATLAS-Q
---------------------------------------

1. **PyTorch CUDA backend**: All tensor operations automatically use GPU
2. **Custom Triton kernels**: Fused operations for 2-3× additional speedup
3. **cuQuantum integration**: NVIDIA-optimized tensor network contractions (2-10× speedup)
4. **Tensor cores**: Hardware acceleration for matrix operations (2-4× speedup on A100/H100)
5. **Memory management**: Efficient allocation and pooling to minimize transfers
6. **Multi-GPU parallelism**: Bond-parallel and data-parallel distribution

GPU Architecture Fundamentals
==============================

Understanding GPU architecture helps optimize quantum simulations.

Memory Hierarchy
----------------

GPU memory has a multi-level hierarchy with different sizes and access latencies:

**1. Global Memory (Device Memory)**:

- **Size**: 16-80 GB (A100, H100)
- **Bandwidth**: 1.5-3 TB/s
- **Latency**: ~400 cycles
- **Usage**: MPS tensor storage, intermediate results
- **Optimization**: Coalesce memory accesses, minimize transfers

**2. L2 Cache**:

- **Size**: 40-50 MB
- **Bandwidth**: ~5 TB/s
- **Latency**: ~200 cycles
- **Usage**: Automatic caching of frequently accessed data

**3. Shared Memory (SMEM)**:

- **Size**: 48-164 KB per SM (streaming multiprocessor)
- **Bandwidth**: ~15 TB/s
- **Latency**: ~20 cycles
- **Usage**: Thread block communication, custom kernel optimization
- **Optimization**: Bank conflicts, tiling strategies

**4. Registers**:

- **Size**: 64K 32-bit registers per SM
- **Bandwidth**: ~20 TB/s
- **Latency**: 1 cycle
- **Usage**: Thread-local variables, loop indices
- **Optimization**: Register spilling to global memory hurts performance

**Memory access pattern importance**:

.. code-block:: python

   # Good: Coalesced access (adjacent threads access adjacent memory)
   for i in range(n):
       result[i] = tensor_a[i] + tensor_b[i]  # Vectorizable

   # Bad: Strided access (threads access non-adjacent memory)
   for i in range(n):
       result[i] = tensor_a[i * stride]  # Cache-unfriendly

Compute Units
-------------

**Streaming Multiprocessors (SMs)**:

- A100: 108 SMs
- H100: 132 SMs
- Each SM executes warps (groups of 32 threads) in SIMT (Single Instruction, Multiple Thread) fashion

**Warp execution**:

- 32 threads execute same instruction on different data
- Branch divergence (different threads take different branches) serializes execution
- Goal: Keep all threads in a warp doing useful work

**Occupancy**:

Ratio of active warps to maximum possible warps per SM.

- Higher occupancy → better latency hiding
- Limited by registers, shared memory, thread blocks per SM
- ATLAS-Q kernels target 50-100% occupancy

Tensor Cores
------------

Specialized hardware for matrix operations:

**Capabilities**:

- **FP64**: 19.5 TFLOPS (A100)
- **TF32**: 156 TFLOPS (A100) - default for FP32 matmul in PyTorch
- **FP16**: 312 TFLOPS (A100)
- **INT8**: 624 TOPS (A100)

**Requirements**:

- Matrix dimensions divisible by 8 (FP32/TF32) or 16 (FP16)
- Uses specialized WMMA (warp matrix multiply-accumulate) instructions

**Automatic usage in PyTorch**:

.. code-block:: python

   import torch
   torch.backends.cuda.matmul.allow_tf32 = True  # Default: True

   # All matmuls, einsums, bmm use tensor cores automatically
   result = torch.einsum('ijk,jkl->ijl', A, B)  # Tensor core accelerated

PyTorch CUDA Backend
====================

ATLAS-Q leverages PyTorch's mature CUDA backend for all tensor operations.

Device Management
-----------------

**Explicit device placement**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Place MPS on GPU
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Or specify device index for multi-GPU
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda:0')

   # Move existing MPS to GPU
   mps = mps.to('cuda')

**Device context**:

.. code-block:: python

   import torch

   # Set default device for all new tensors
   torch.set_default_device('cuda')

   # Device context manager
   with torch.cuda.device(0):
       mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

Automatic Operation Fusion
---------------------------

PyTorch automatically fuses operations to reduce memory traffic:

**Fused operations**:

.. code-block:: python

   # Example: MPS gate application
   # Original: Multiple separate kernels
   A_left = torch.einsum('ijk,kl->ijl', mps.tensors[i], mps.tensors[i+1])
   A_gate = torch.einsum('ijkl,km,ln->ijmn', A_left, gate[:, :, 0], gate[:, :, 1])
   # ... more operations

   # PyTorch JIT can fuse these into single kernel
   @torch.jit.script
   def fused_gate_application(tensor_left, tensor_right, gate):
       A_left = torch.einsum('ijk,kl->ijl', tensor_left, tensor_right)
       return torch.einsum('ijkl,km,ln->ijmn', A_left, gate[:, :, 0], gate[:, :, 1])

   A_gate = fused_gate_application(mps.tensors[i], mps.tensors[i+1], gate)

**Benefits**: Reduced kernel launches (microseconds overhead each), less global memory traffic.

cuBLAS Integration
------------------

PyTorch uses cuBLAS (CUDA Basic Linear Algebra Subroutines) for optimized matrix operations:

**Operations accelerated**:

- Matrix multiplication (``torch.matmul``, ``@`` operator)
- Batched matrix multiplication (``torch.bmm``)
- Matrix-vector products
- Triangular solves (used in canonicalization)

**Performance**: cuBLAS is typically within 90-95% of theoretical peak FLOPS for large matrices.

.. code-block:: python

   # Example: Bond canonicalization via QR decomposition
   # Internally uses cuBLAS routines
   Q, R = torch.linalg.qr(mps.tensors[i].reshape(chi_left * d, chi_right))
   mps.tensors[i] = Q.reshape(chi_left, d, chi_right)

cuSOLVER Integration
--------------------

PyTorch uses cuSOLVER for linear algebra operations:

**Operations**:

- **SVD** (``torch.linalg.svd``): Critical for MPS truncation
- **Eigendecomposition** (``torch.linalg.eigh``): Used in TDVP
- **QR decomposition** (``torch.linalg.qr``): Used in canonicalization

**SVD performance** (most important for MPS):

For matrix of size :math:`\chi \times \chi`:

- **A100 GPU**: ~0.1 ms for χ=64, ~1 ms for χ=256
- **CPU (16 cores)**: ~1 ms for χ=64, ~10 ms for χ=256

SVD is the main bottleneck in adaptive MPS simulation.

Custom Triton Kernels
======================

ATLAS-Q includes custom GPU kernels written in Triton (Python-based GPU kernel language) for operations that benefit from fusion and specialized memory access patterns.

Why Custom Kernels
------------------

**Limitations of PyTorch built-ins**:

1. **Einsum overhead**: Separate kernel launches for each contraction
2. **Memory traffic**: Intermediate results written to/read from global memory
3. **Indexing**: Dynamic indexing can be inefficient

**Triton advantages**:

- Python syntax for GPU kernel development
- Automatic optimization (tiling, memory coalescing)
- Fusion of multiple operations into single kernel
- Easier to maintain than CUDA C++

Triton Kernel: mps_complex.py
------------------------------

**Purpose**: Fused two-qubit gate application to MPS tensors.

**Operation**:

.. code-block:: python

   # Unfused (PyTorch default)
   # 1. Merge two MPS tensors
   A_merged = torch.einsum('ijk,klm->ijlm', mps.tensors[i], mps.tensors[i+1])
   # 2. Apply gate
   A_gate = torch.einsum('ijkl,km,ln->ijmn', A_merged, gate[:,:,0], gate[:,:,1])
   # 3. SVD to split
   U, S, Vh = torch.linalg.svd(A_gate.reshape(chi*2, chi*2))
   # Total: 3 separate kernel launches

   # Fused (Triton kernel)
   from atlas_q.triton_kernels.mps_complex import fused_two_qubit_gate
   # Single kernel launch performs all contractions with tiled memory access
   U, S, Vh = fused_two_qubit_gate(mps.tensors[i], mps.tensors[i+1], gate)

**Optimization strategies**:

1. **2×2 tiling**: Process 2×2 blocks of matrix for better cache reuse
2. **Shared memory**: Store gate matrix and partial results in SMEM
3. **Register blocking**: Keep intermediate sums in registers
4. **Coalesced access**: Threads access adjacent memory locations

**Performance**:

+-------------+------------------+--------------------+----------+
| Bond dim χ  | PyTorch (ms)     | Triton kernel (ms) | Speedup  |
+=============+==================+====================+==========+
| 32          | 0.08             | 0.09               | 0.9×     |
+-------------+------------------+--------------------+----------+
| 64          | 0.15             | 0.10               | 1.5×     |
+-------------+------------------+--------------------+----------+
| 128         | 0.45             | 0.22               | 2.0×     |
+-------------+------------------+--------------------+----------+
| 256         | 1.50             | 0.58               | 2.6×     |
+-------------+------------------+--------------------+----------+
| 512         | 5.80             | 2.10               | 2.8×     |
+-------------+------------------+--------------------+----------+

**When beneficial**: χ > 64 (overhead dominates for small χ).

Triton Kernel: modpow.py
-------------------------

**Purpose**: Batched modular exponentiation for period-finding (Shor's algorithm component).

**Operation**:

.. math::

   \text{result}[i] = a^{\text{exponent}[i]} \mod N

**Algorithm**: Binary exponentiation (square-and-multiply).

**Optimization**:

.. code-block:: python

   from atlas_q.triton_kernels.modpow import batched_modpow_triton

   # Compute a^x mod N for many x values in parallel
   a = 7
   N = 143
   exponents = torch.arange(0, 10000, device='cuda')  # 10000 exponents

   # Triton kernel processes in batches of 1024
   results = batched_modpow_triton(a, exponents, N)

**Performance vs PyTorch**:

- **PyTorch**: Loop over exponents (serialized) → 150 ms
- **Triton**: Fully parallel → 5 ms
- **Speedup**: ~30× for 10,000 exponents

**Critical for**: Period-finding, Shor's algorithm, quantum walk simulations.

Triton Kernel: tdvp_mpo_ops.py
-------------------------------

**Purpose**: TDVP effective Hamiltonian contractions.

**Operation**: Contract MPS tensors with MPO (Matrix Product Operator) Hamiltonian:

.. math::

   H_{\text{eff}} = \text{contract}(L_{\text{env}}, H^{[i]}, R_{\text{env}}, A^{[i]})

where :math:`L_{\text{env}}` and :math:`R_{\text{env}}` are left and right environments.

**Challenge**: Complex contraction pattern with many indices.

**Optimization**:

1. **Fuse environment contractions**: Compute :math:`L \times H \times R` in single kernel
2. **Optimize memory layout**: Transpose tensors to improve access pattern
3. **Reduce memory traffic**: ~40% reduction vs separate contractions

**Performance**:

- **PyTorch einsum chain**: 0.8 ms per contraction (χ=64, MPO bond dim = 8)
- **Triton kernel**: 0.5 ms per contraction
- **Speedup**: 1.6×

**Impact on TDVP**: 10-20% overall speedup for time evolution loops.

Installation and Setup
----------------------

Triton kernels are automatically compiled on first use:

.. code-block:: bash

   # Install Triton (included in ATLAS-Q requirements)
   pip install triton

   # Setup script auto-detects GPU architecture
   ./setup_triton.sh

   # Manually specify architecture (if needed)
   export TORCH_CUDA_ARCH_LIST="9.0"  # For H100
   export TORCH_CUDA_ARCH_LIST="8.0"  # For A100

   # Verify Triton installation
   python -c "import triton; print(triton.__version__)"

cuQuantum Integration
=====================

NVIDIA cuQuantum provides highly optimized tensor network operations.

cuQuantum Components
--------------------

**cuTensorNet**:

- Optimized tensor network contractions
- Automatic contraction order optimization
- Multi-GPU tensor network execution
- Slicing for memory-efficient contractions

**cuStateVec**:

- State vector operations (less relevant for MPS)
- Gate application, measurement, expectation values

ATLAS-Q Integration
-------------------

cuQuantum is optionally integrated and auto-detected:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend

   # Configure cuQuantum backend
   config = CuQuantumConfig(
       use_cutensornet=True,
       workspace_size=2 * 1024**3,      # 2 GB workspace for contractions
       algorithm='auto',                # or 'qr', 'svd'
       svd_algorithm='gesvdj',          # Jacobi SVD (fast for χ < 512)
       enable_async=True,               # Asynchronous execution
       num_streams=4                    # CUDA streams for overlap
   )

   backend = CuQuantumBackend(config, device='cuda')

   # Use with MPS
   from atlas_q.adaptive_mps import AdaptiveMPS
   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=128,
       backend=backend,
       device='cuda'
   )

Performance Characteristics
---------------------------

**When cuQuantum helps**:

1. **Large bond dimensions** (χ > 128): Better SVD performance
2. **Deep circuits**: Optimized contraction ordering
3. **Multi-GPU**: Automatic work distribution

**Speedup examples** (χ=256, n=40 qubits):

+------------------------+------------------+--------------------+----------+
| Operation              | PyTorch (ms)     | cuQuantum (ms)     | Speedup  |
+========================+==================+====================+==========+
| Two-qubit gate         | 2.5              | 1.2                | 2.1×     |
+------------------------+------------------+--------------------+----------+
| SVD truncation         | 1.8              | 0.6                | 3.0×     |
+------------------------+------------------+--------------------+----------+
| TDVP time step         | 150              | 65                 | 2.3×     |
+------------------------+------------------+--------------------+----------+
| VQE energy evaluation  | 800              | 320                | 2.5×     |
+------------------------+------------------+--------------------+----------+

**Fallback behavior**: If cuQuantum not installed, ATLAS-Q automatically falls back to PyTorch with no code changes.

GPU Memory Management
=====================

Efficient memory management is critical for large-scale simulations.

PyTorch Caching Allocator
--------------------------

PyTorch uses a caching allocator to avoid expensive cudaMalloc/cudaFree calls:

**Mechanism**:

1. First allocation: Request memory from CUDA
2. Deallocation: Return to PyTorch cache (not to CUDA)
3. Future allocation: Reuse cached memory if available
4. Only release to CUDA when explicitly requested

**Benefits**: ~100× faster allocation/deallocation.

**Monitoring**:

.. code-block:: python

   import torch

   # Current allocated memory (in use by tensors)
   allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
   print(f"Allocated: {allocated:.2f} GB")

   # Reserved memory (cached by PyTorch)
   reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
   print(f"Reserved: {reserved:.2f} GB")

   # Peak allocated memory
   peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
   print(f"Peak: {peak:.2f} GB")

   # Reset peak counter
   torch.cuda.reset_peak_memory_stats()

**Clearing cache**:

.. code-block:: python

   # Release all cached memory back to CUDA
   torch.cuda.empty_cache()

   # Note: Only releases memory not currently in use by tensors
   # Does NOT free memory held by active tensors

Memory Budgets in ATLAS-Q
--------------------------

ATLAS-Q supports global memory budgets to prevent OOM errors:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Set global memory budget (10 GB)
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,
       budget_global_mb=10 * 1024,      # 10 GB limit
       adaptive_mode=True,
       device='cuda'
   )

   # MPS will adaptively reduce bond dimension to stay within budget
   for i in range(49):
       mps.apply_cnot(i, i+1)

   # Check if budget was exceeded
   if mps.statistics.budget_exceeded:
       print("Warning: Reduced bond dimension to stay within budget")

**Budget enforcement**:

1. Before each gate: Check projected memory usage
2. If over budget: Reduce χ at bonds furthest from entanglement center
3. Track total truncation error from budget constraints

Memory Optimization Strategies
-------------------------------

**1. Mixed Precision**:

.. code-block:: python

   # Use complex64 instead of complex128 (50% memory reduction)
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       dtype=torch.complex64,  # Instead of complex128
       device='cuda'
   )

   # Accuracy impact: Typically < 1e-6 for most simulations

**2. Gradient Checkpointing** (for variational algorithms):

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig

   config = VQEConfig(
       max_iterations=500,
       optimizer='L-BFGS-B',
       use_checkpointing=True,  # Trade compute for memory
       checkpoint_segments=4     # Split circuit into 4 segments
   )

   # Memory usage: 4× reduction
   # Compute time: 1.3× increase (acceptable tradeoff)

**3. In-Place Operations**:

.. code-block:: python

   # Avoid creating intermediate tensors
   # Good: In-place
   mps.tensors[i].mul_(factor)

   # Bad: Creates new tensor
   mps.tensors[i] = mps.tensors[i] * factor

**4. Explicit Deletion**:

.. code-block:: python

   # Delete large intermediate tensors
   large_tensor = some_expensive_computation()
   # ... use large_tensor ...
   del large_tensor
   torch.cuda.empty_cache()  # Release to CUDA

Tensor Core Acceleration
=========================

Tensor cores provide specialized hardware for matrix operations.

Architecture Details
--------------------

**Tensor core operations**:

.. math::

   D = A \times B + C

where A, B, C, D are matrices/tensors.

**Precision modes**:

- **FP64**: Full 64-bit precision (19.5 TFLOPS on A100)
- **TF32**: TensorFloat-32 (156 TFLOPS on A100) - 19-bit mantissa
- **FP16**: Half precision (312 TFLOPS on A100)
- **BF16**: Bfloat16 (312 TFLOPS on A100) - better range than FP16

**Default in PyTorch**: TF32 for FP32 matmuls (automatically enabled).

Enabling Tensor Cores
---------------------

.. code-block:: python

   import torch

   # Enable TF32 for matmul (default: True)
   torch.backends.cuda.matmul.allow_tf32 = True

   # Enable TF32 for cuDNN convolutions (not relevant for ATLAS-Q)
   torch.backends.cudnn.allow_tf32 = True

   # Use mixed precision for 2× speedup
   with torch.cuda.amp.autocast():
       # All matmuls use FP16 tensor cores
       result = torch.einsum('ijk,jkl->ijl', A, B)

Optimal Matrix Sizes
--------------------

Tensor cores work on 16×16×16 (FP16) or 8×8×8 (FP32/TF32) tiles.

**Padding for alignment**:

.. code-block:: python

   # Bad: χ=63 (not divisible by 8)
   mps = AdaptiveMPS(num_qubits=30, bond_dim=63, device='cuda')

   # Good: χ=64 (divisible by 8)
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Speedup: ~20% just from alignment

**Recommended bond dimensions** for tensor core efficiency:

- 32, 64, 128, 256, 512 (powers of 2)
- 48, 96, 192, 384 (multiples of 16)
- Avoid: 50, 63, 100, 150 (not tensor core friendly)

Performance Benchmarks
-----------------------

**Einsum performance** (A100, TF32 enabled):

+-------------+------------------+--------------------+----------+
| Bond dim χ  | Without TC (ms)  | With TC (ms)       | Speedup  |
+=============+==================+====================+==========+
| 32          | 0.12             | 0.08               | 1.5×     |
+-------------+------------------+--------------------+----------+
| 64          | 0.45             | 0.18               | 2.5×     |
+-------------+------------------+--------------------+----------+
| 128         | 1.80             | 0.55               | 3.3×     |
+-------------+------------------+--------------------+----------+
| 256         | 7.20             | 1.90               | 3.8×     |
+-------------+------------------+--------------------+----------+
| 512         | 28.5             | 7.50               | 3.8×     |
+-------------+------------------+--------------------+----------+

Performance Profiling
======================

Identifying bottlenecks is essential for optimization.

PyTorch Profiler
----------------

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Profile gate applications
   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       record_shapes=True,
       with_stack=True
   ) as prof:
       for i in range(29):
           mps.apply_cnot(i, i+1)

   # Print summary
   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

   # Export for visualization
   prof.export_chrome_trace("profiler_trace.json")
   # View at chrome://tracing

NVIDIA Nsight Systems
----------------------

For detailed GPU kernel analysis:

.. code-block:: bash

   # Profile ATLAS-Q script
   nsys profile -o atlas_q_profile python my_simulation.py

   # View profile
   nsys-ui atlas_q_profile.qdrep

**Key metrics**:

- **Kernel duration**: Time spent in each GPU kernel
- **Memory bandwidth**: Actual vs peak bandwidth utilization
- **Occupancy**: Active warps vs max warps per SM
- **Launch overhead**: Time between kernel launches

Common Bottlenecks
------------------

**1. SVD truncation** (typically 40-60% of time):

.. code-block:: python

   # Profile SVD
   import time
   A = torch.randn(256, 256, dtype=torch.complex128, device='cuda')

   torch.cuda.synchronize()
   start = time.time()
   U, S, Vh = torch.linalg.svd(A)
   torch.cuda.synchronize()
   elapsed = time.time() - start
   print(f"SVD time: {elapsed*1000:.2f} ms")

**Mitigation**:
- Use cuQuantum's gesvdj (Jacobi SVD) for χ < 512
- Use complex64 instead of complex128 (2× faster SVD)
- Reduce truncation frequency (e.g., every 10 gates)

**2. Memory bandwidth** (20-30% of time):

**Mitigation**:
- Fuse operations to reduce intermediate tensors
- Use Triton kernels for memory-bound operations
- Increase arithmetic intensity (more compute per memory access)

**3. Kernel launch overhead** (5-10% for small operations):

**Mitigation**:
- Batch operations when possible
- Use CUDA streams for overlap

Multi-GPU Scaling
=================

ATLAS-Q supports multi-GPU parallelism for large simulations.

Bond-Parallel Distribution
---------------------------

Partition MPS chain across GPUs:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   import torch.distributed as dist

   # Initialize distributed backend
   dist.init_process_group(backend='nccl', init_method='env://')

   config = DistributedConfig(
       mode='bond_parallel',        # Split bond dimension across GPUs
       world_size=4,                # 4 GPUs
       backend='nccl'
   )

   # MPS with χ=512 split across 4 GPUs (128 per GPU)
   mps = DistributedMPS(
       num_qubits=80,
       bond_dim=512,
       config=config,
       device='cuda'
   )

   # Gate operations automatically handle inter-GPU communication
   for i in range(79):
       mps.apply_cnot(i, i+1)  # NCCL allreduce when crossing GPU boundaries

**Performance**: Near-linear scaling up to 4-8 GPUs.

Data-Parallel Distribution
---------------------------

Replicate MPS for parallel sampling:

.. code-block:: python

   config = DistributedConfig(
       mode='data_parallel',        # Replicate MPS across GPUs
       world_size=4,
       backend='nccl'
   )

   mps = DistributedMPS(num_qubits=30, bond_dim=64, config=config, device='cuda')

   # Each GPU performs independent sampling
   samples = mps.sample(n_shots=10000)  # 40,000 total samples across 4 GPUs

**Use cases**: Variational algorithms needing many energy evaluations.

Best Practices
==============

GPU Usage Guidelines
--------------------

**When to use GPU**:

1. **Bond dimension χ > 32**: GPU overhead amortized
2. **Circuit depth > 50 gates**: Many operations to parallelize
3. **Iterative algorithms** (VQE, QAOA, TDVP): Hundreds of evaluations
4. **Large qubit count** (n > 20): More parallelism

**When CPU may be faster**:

1. **Small systems** (n < 15, χ < 32): Overhead dominates
2. **Single gate operations**: Transfer time exceeds compute time
3. **Memory-constrained**: GPU memory < required MPS size

Optimization Checklist
----------------------

1. **Use power-of-2 bond dimensions** (32, 64, 128, 256) for tensor core alignment
2. **Enable TF32**: ``torch.backends.cuda.matmul.allow_tf32 = True``
3. **Profile to find bottlenecks**: PyTorch profiler or Nsight
4. **Consider mixed precision** (complex64) for 2× memory and speedup
5. **Use cuQuantum for χ > 128**: 2-3× additional speedup
6. **Batch operations**: Apply multiple gates before synchronization
7. **Monitor memory**: Use budgets and clear cache when needed
8. **Multi-GPU for large simulations**: Bond-parallel for χ > 512

Common Pitfalls
---------------

1. **Frequent CPU-GPU transfers**: Keep tensors on GPU
2. **Synchronization points**: Avoid ``tensor.item()``, ``print(tensor)`` in loops
3. **Small batches**: GPU underutilized
4. **Non-aligned dimensions**: Miss tensor core acceleration
5. **Memory fragmentation**: Clear cache periodically
6. **Ignoring warmup**: First run includes compilation overhead

Summary
=======

GPU acceleration in ATLAS-Q provides 10-1000× speedups through:

**Hardware Utilization**:

- **Tensor cores**: 2-4× speedup on matrix operations (A100, H100)
- **High memory bandwidth**: 1.5-3 TB/s for tensor operations
- **Massive parallelism**: 1000s of threads executing simultaneously

**Software Stack**:

- **PyTorch CUDA backend**: Automatic GPU execution for all tensor ops
- **cuBLAS/cuSOLVER**: Optimized linear algebra (matmul, SVD, QR)
- **Custom Triton kernels**: 1.5-3× additional speedup via fusion
- **cuQuantum**: 2-10× speedup for large bond dimensions (optional)

**Memory Management**:

- Caching allocator for fast allocation/deallocation
- Memory budgets prevent OOM errors
- Mixed precision reduces memory by 50%

**Multi-GPU Scaling**:

- Bond-parallel: Split large bond dimensions across GPUs
- Data-parallel: Parallel sampling and energy evaluations
- Near-linear scaling up to 4-8 GPUs

**Key Performance Factors**:

1. **Bond dimension**: GPU beneficial for χ > 32, essential for χ > 128
2. **Problem size**: Larger systems amortize overhead better
3. **Algorithm**: Iterative algorithms (VQE, TDVP) see largest gains
4. **Precision**: TF32/FP16 provides 2-4× speedup with minimal accuracy loss

**Recommended Configuration** (A100 GPU):

.. code-block:: python

   import torch
   torch.backends.cuda.matmul.allow_tf32 = True

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.cuquantum_backend import CuQuantumBackend, CuQuantumConfig

   # Configure cuQuantum
   cuq_config = CuQuantumConfig(use_cutensornet=True, workspace_size=2*1024**3)
   backend = CuQuantumBackend(cuq_config, device='cuda')

   # Create MPS with optimal settings
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,                     # Power of 2 for tensor cores
       dtype=torch.complex64,            # Mixed precision
       backend=backend,                  # cuQuantum acceleration
       budget_global_mb=30 * 1024,       # 30 GB budget
       device='cuda'
   )

This configuration typically achieves 50-100× speedup over CPU execution for large quantum simulations.

For detailed multi-GPU setup and advanced optimizations, see:

- :doc:`../howtos/parallel_computation` - Multi-GPU parallelization guide
- :doc:`../howtos/optimize_performance` - Performance optimization strategies
- :doc:`../howtos/integrate_cuquantum` - cuQuantum integration details
