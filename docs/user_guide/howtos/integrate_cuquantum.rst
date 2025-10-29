========================================
Integrate cuQuantum
========================================

Problem
=======

NVIDIA cuQuantum provides GPU-accelerated tensor network operations that can significantly
speed up MPS simulations. Key benefits:

- **SVD acceleration**: 2-5× faster than PyTorch for large bond dimensions (χ > 64)
- **Tensor contractions**: 2-10× faster for multi-dimensional contractions
- **Memory efficiency**: Optimized workspace management for large tensors
- **Hardware utilization**: Better use of modern GPU features (Tensor Cores, async ops)
- **Production-ready**: Battle-tested on NVIDIA's HPC systems

This guide covers cuQuantum installation, configuration, and integration with ATLAS-Q
to maximize MPS simulation performance on NVIDIA GPUs.

.. seealso::
   :doc:`../explanations/gpu_acceleration` for GPU optimization theory,
   :doc:`optimize_performance` for general performance tuning,
   :doc:`handle_large_systems` for large-scale distributed simulations with cuQuantum.

Prerequisites
=============

You need:

- **NVIDIA GPU**: Volta (V100), Ampere (A100), or Hopper (H100) recommended
- **CUDA Toolkit**: Version 11.8 or later (12.x recommended)
- **Python**: 3.8-3.12
- **Storage**: ~500 MB for cuQuantum libraries
- **System permissions**: Ability to install CUDA-dependent packages

Strategies
==========

Strategy 1: Installation and Setup
-----------------------------------

Install cuQuantum for GPU-accelerated tensor operations.

**Standard installation via pip**:

.. code-block:: bash

   # Install cuQuantum Python bindings
   pip install cuquantum-python

   # Verify installation
   python -c "import cuquantum; print(f'cuQuantum version: {cuquantum.__version__}')"

**Expected output**:

.. code-block:: text

   cuQuantum version: 24.03.0

**Installation with specific CUDA version**:

.. code-block:: bash

   # If you have CUDA 11.8
   pip install cuquantum-python-cu11

   # If you have CUDA 12.x
   pip install cuquantum-python-cu12

**Verify GPU and CUDA compatibility**:

.. code-block:: python

   import torch
   import cuquantum

   # Check CUDA availability
   print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version (PyTorch): {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")

   # Check cuQuantum version
   print(f"cuQuantum version: {cuquantum.__version__}")

   # Test basic cuQuantum functionality
   from cuquantum import cutensornet as cutn
   handle = cutn.create()
   print("cuQuantum successfully initialized")
   cutn.destroy(handle)

**Expected output**:

.. code-block:: text

   PyTorch CUDA available: True
   CUDA version (PyTorch): 12.1
   GPU: NVIDIA A100-SXM4-80GB
   cuQuantum version: 24.03.0
   cuQuantum successfully initialized

Strategy 2: Automatic cuQuantum Detection
------------------------------------------

ATLAS-Q automatically detects and uses cuQuantum when available.

**Check cuQuantum availability**:

.. code-block:: python

   from atlas_q import get_cuquantum

   # Get cuQuantum info
   cuq = get_cuquantum()

   if cuq['is_cuquantum_available']():
       version = cuq['get_cuquantum_version']()
       print(f"cuQuantum {version} available")
       print("ATLAS-Q will automatically use cuQuantum for:")
       print("  - SVD operations (bond dimension truncation)")
       print("  - Large tensor contractions")
       print("  - QR decompositions")
   else:
       print("cuQuantum not available")
       print("ATLAS-Q will use PyTorch fallback implementations")
       print("Performance will be ~2-5× slower for large bond dimensions")

**Test cuQuantum with simple MPS**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import time

   # Create MPS - will automatically use cuQuantum if available
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=128,
       device='cuda',
       use_cuquantum=True  # Explicitly enable (default is auto-detect)
   )

   # Apply gates - SVD uses cuQuantum
   start = time.time()
   for i in range(29):
       mps.apply_cnot(i, i+1)
   elapsed = time.time() - start

   print(f"Applied 29 CNOTs in {elapsed:.3f}s")
   print(f"Backend: {'cuQuantum' if mps._using_cuquantum else 'PyTorch'}")

Strategy 3: Manual Backend Configuration
-----------------------------------------

Manually configure cuQuantum backend for fine-grained control.

**Configure cuQuantum backend**:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend, CuQuantumConfig

   # Configure cuQuantum with custom settings
   config = CuQuantumConfig(
       use_cutensornet=True,         # Use cuTensorNet for tensor networks
       workspace_size=2 * 1024**3,   # 2 GB workspace (adjust based on GPU memory)
       algorithm='auto',             # Auto-select best algorithm
       svd_algorithm='gesvdj',       # Jacobi SVD (fast, good for χ < 512)
       # svd_algorithm='gesvd',      # Standard SVD (more stable for large χ)
       enable_async=True,            # Asynchronous operations
       num_streams=4                 # CUDA streams for parallelism
   )

   # Create backend
   backend = CuQuantumBackend(config, device='cuda')

   print(f"cuQuantum backend initialized")
   print(f"  Workspace size: {config.workspace_size / 1024**3:.1f} GB")
   print(f"  SVD algorithm: {config.svd_algorithm}")
   print(f"  Async enabled: {config.enable_async}")

**Use cuQuantum backend explicitly**:

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Create MPS with explicit cuQuantum backend
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=256,
       device='cuda',
       backend=backend  # Use custom-configured backend
   )

   # All tensor operations use cuQuantum
   for i in range(49):
       mps.apply_cnot(i, i+1)

   print(f"MPS using cuQuantum backend: {mps._backend_name}")

Strategy 4: Selective Backend Usage
------------------------------------

Choose when to use cuQuantum vs PyTorch based on operation characteristics.

**Hybrid backend strategy**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Small bond dimension: PyTorch is sufficient and may be faster
   # due to kernel launch overhead
   mps_small = AdaptiveMPS(
       num_qubits=30,
       bond_dim=32,
       device='cuda',
       use_cuquantum=False  # Disable cuQuantum for small χ
   )

   # Large bond dimension: cuQuantum provides significant speedup
   mps_large = AdaptiveMPS(
       num_qubits=50,
       bond_dim=256,
       device='cuda',
       use_cuquantum=True  # Enable cuQuantum for large χ
   )

   # Benchmark: Small bond dimension
   import time

   start = time.time()
   for i in range(29):
       mps_small.apply_cnot(i, i+1)
   small_time = time.time() - start

   # Benchmark: Large bond dimension
   start = time.time()
   for i in range(49):
       mps_large.apply_cnot(i, i+1)
   large_time = time.time() - start

   print(f"Small χ (32): {small_time:.3f}s (PyTorch)")
   print(f"Large χ (256): {large_time:.3f}s (cuQuantum)")
   print(f"Speedup for large χ: {small_time / large_time:.2f}×")

**Adaptive backend selection**:

.. code-block:: python

   def create_mps_with_optimal_backend(num_qubits, bond_dim, device='cuda'):
       """
       Create MPS with optimal backend based on bond dimension.

       cuQuantum beneficial for χ >= 64, especially for χ >= 128.
       """
       use_cuquantum = bond_dim >= 64

       mps = AdaptiveMPS(
           num_qubits=num_qubits,
           bond_dim=bond_dim,
           device=device,
           use_cuquantum=use_cuquantum
       )

       backend_name = 'cuQuantum' if use_cuquantum else 'PyTorch'
       print(f"Created MPS with χ={bond_dim} using {backend_name} backend")

       return mps

   # Usage
   mps1 = create_mps_with_optimal_backend(30, 32)   # Uses PyTorch
   mps2 = create_mps_with_optimal_backend(50, 128)  # Uses cuQuantum
   mps3 = create_mps_with_optimal_backend(80, 512)  # Uses cuQuantum

Strategy 5: Performance Tuning
-------------------------------

Optimize cuQuantum performance by tuning workspace size and algorithm choices.

**Workspace size optimization**:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend
   import torch

   # Check available GPU memory
   total_memory = torch.cuda.get_device_properties(0).total_memory
   available_memory = total_memory - torch.cuda.memory_allocated(0)

   print(f"Total GPU memory: {total_memory / 1024**3:.1f} GB")
   print(f"Available memory: {available_memory / 1024**3:.1f} GB")

   # Allocate workspace: Use ~25-50% of available memory
   # More workspace → faster operations, but less room for MPS tensors
   workspace_size = int(0.3 * available_memory)

   config = CuQuantumConfig(
       use_cutensornet=True,
       workspace_size=workspace_size,
       algorithm='auto'
   )

   backend = CuQuantumBackend(config, device='cuda')

   print(f"cuQuantum workspace: {workspace_size / 1024**3:.2f} GB")

**SVD algorithm selection**:

.. code-block:: python

   # Different SVD algorithms have different performance characteristics:

   # 1. gesvdj (Jacobi SVD) - Fast for χ < 512, good accuracy
   config_jacobi = CuQuantumConfig(
       svd_algorithm='gesvdj',
       workspace_size=1 * 1024**3
   )

   # 2. gesvd (Standard SVD) - More stable for large χ, slower
   config_standard = CuQuantumConfig(
       svd_algorithm='gesvd',
       workspace_size=1 * 1024**3
   )

   # 3. gesvda (Approximate SVD) - Fastest, slight accuracy trade-off
   config_approx = CuQuantumConfig(
       svd_algorithm='gesvda',
       workspace_size=1 * 1024**3
   )

   # Recommendation:
   # - χ <= 128: gesvdj (fast and accurate)
   # - 128 < χ <= 512: gesvd (stable)
   # - χ > 512: gesvda (only if accuracy permits)

   # Example: TDVP with Jacobi SVD
   from atlas_q.tdvp import TDVP
   from atlas_q.adaptive_mps import AdaptiveMPS

   backend_jacobi = CuQuantumBackend(config_jacobi, device='cuda')

   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=128,
       device='cuda',
       backend=backend_jacobi
   )

   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

   # TDVP will use Jacobi SVD via cuQuantum
   for step in range(1000):
       E = tdvp.evolve_step()
       if step % 100 == 0:
           print(f"[Step {step}] E={E:.8f}")

Strategy 6: Benchmarking cuQuantum vs PyTorch
----------------------------------------------

Compare cuQuantum and PyTorch performance for your specific workload.

**Comprehensive benchmark**:

.. code-block:: python

   import time
   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   def benchmark_mps_gates(num_qubits, bond_dim, num_gates, use_cuquantum):
       """
       Benchmark MPS gate application.

       Returns
       -------
       float
           Time in seconds
       """
       mps = AdaptiveMPS(
           num_qubits=num_qubits,
           bond_dim=bond_dim,
           device='cuda',
           use_cuquantum=use_cuquantum
       )

       torch.cuda.synchronize()
       start = time.time()

       for i in range(num_gates):
           qubit = i % (num_qubits - 1)
           mps.apply_cnot(qubit, qubit + 1)

       torch.cuda.synchronize()
       elapsed = time.time() - start

       return elapsed

   # Benchmark different bond dimensions
   bond_dims = [32, 64, 128, 256, 512]
   num_qubits = 50
   num_gates = 100

   print(f"Benchmarking {num_gates} CNOT gates on {num_qubits} qubits")
   print(f"{'χ':<10} {'PyTorch (s)':<15} {'cuQuantum (s)':<15} {'Speedup':<10}")
   print("-" * 50)

   for chi in bond_dims:
       time_pytorch = benchmark_mps_gates(num_qubits, chi, num_gates, use_cuquantum=False)
       time_cuquantum = benchmark_mps_gates(num_qubits, chi, num_gates, use_cuquantum=True)
       speedup = time_pytorch / time_cuquantum

       print(f"{chi:<10} {time_pytorch:<15.3f} {time_cuquantum:<15.3f} {speedup:<10.2f}×")

   # Expected output (A100 GPU):
   # χ          PyTorch (s)     cuQuantum (s)   Speedup
   # --------------------------------------------------
   # 32         0.245           0.198           1.24×
   # 64         0.512           0.301           1.70×
   # 128        1.234           0.487           2.53×
   # 256        3.567           0.892           4.00×
   # 512        9.123           1.987           4.59×

Strategy 7: Distributed cuQuantum
----------------------------------

Use cuQuantum with distributed MPS across multiple GPUs.

**Multi-GPU cuQuantum configuration**:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend
   import torch.distributed as dist

   # Configure distributed environment
   dist_config = DistributedConfig(
       mode='bond_parallel',
       world_size=4,
       backend='nccl',
       device_ids=[0, 1, 2, 3]
   )

   # Each GPU gets its own cuQuantum backend
   rank = dist.get_rank()
   cuq_config = CuQuantumConfig(
       use_cutensornet=True,
       workspace_size=2 * 1024**3,  # 2 GB per GPU
       algorithm='auto',
       svd_algorithm='gesvdj'
   )
   backend = CuQuantumBackend(cuq_config, device=f'cuda:{rank}')

   # Create distributed MPS with cuQuantum
   mps = DistributedMPS(
       num_qubits=100,
       bond_dim=512,  # Split across 4 GPUs = 128 per GPU
       config=dist_config,
       backend=backend
   )

   print(f"Rank {rank}: Distributed MPS with cuQuantum backend")

   # Operations use cuQuantum on each GPU
   for i in range(99):
       mps.apply_cnot(i, i+1)

   dist.barrier()
   print(f"Rank {rank}: Completed gate sequence")

Troubleshooting
===============

cuQuantum Import Fails
-----------------------

**Problem**: ``ImportError: No module named 'cuquantum'``.

**Solution**: Install cuquantum-python package.

.. code-block:: bash

   # Install
   pip install cuquantum-python

   # If specific CUDA version needed
   pip install cuquantum-python-cu12  # For CUDA 12.x

   # Verify
   python -c "import cuquantum; print(cuquantum.__version__)"

CUDA Compatibility Error
-------------------------

**Problem**: ``RuntimeError: cuQuantum requires CUDA 11.8 or later``.

**Solution**: Update CUDA Toolkit or use compatible cuQuantum version.

.. code-block:: bash

   # Check CUDA version
   nvcc --version

   # If CUDA < 11.8, upgrade CUDA Toolkit
   # Or use older cuQuantum version
   pip install cuquantum-python==23.03.0  # Compatible with CUDA 11.0+

Workspace Size Too Large
-------------------------

**Problem**: ``RuntimeError: Failed to allocate cuQuantum workspace: out of memory``.

**Solution**: Reduce workspace size.

.. code-block:: python

   import torch

   # Check available memory
   available = torch.cuda.mem_get_info()[0]
   print(f"Available GPU memory: {available / 1024**3:.1f} GB")

   # Reduce workspace size
   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend

   config = CuQuantumConfig(
       use_cutensornet=True,
       workspace_size=512 * 1024**2,  # 512 MB instead of 2 GB
       algorithm='auto'
   )

   backend = CuQuantumBackend(config, device='cuda')

cuQuantum Slower Than PyTorch
------------------------------

**Problem**: cuQuantum backend slower than PyTorch for small problems.

**Solution**: Use cuQuantum only for large bond dimensions (χ ≥ 64).

.. code-block:: python

   # Adaptive backend selection
   def optimal_backend(bond_dim):
       """Return optimal backend based on bond dimension."""
       return bond_dim >= 64

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=32,  # Small χ
       device='cuda',
       use_cuquantum=optimal_backend(32)  # False, uses PyTorch
   )

   mps_large = AdaptiveMPS(
       num_qubits=50,
       bond_dim=256,  # Large χ
       device='cuda',
       use_cuquantum=optimal_backend(256)  # True, uses cuQuantum
   )

SVD Convergence Issues with cuQuantum
--------------------------------------

**Problem**: ``RuntimeError: cuQuantum SVD did not converge``.

**Solution**: Switch to more stable SVD algorithm or increase tolerance.

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend

   # Use standard SVD (more stable)
   config = CuQuantumConfig(
       use_cutensornet=True,
       svd_algorithm='gesvd',  # Was 'gesvdj'
       workspace_size=1 * 1024**3
   )

   backend = CuQuantumBackend(config, device='cuda')

   # Or increase truncation threshold
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=256,
       truncation_threshold=1e-7,  # Was 1e-10
       device='cuda',
       backend=backend
   )

Summary
=======

cuQuantum integration strategies for ATLAS-Q:

1. **Installation**: ``pip install cuquantum-python`` (requires CUDA 11.8+)
2. **Automatic detection**: ATLAS-Q auto-detects and uses cuQuantum when available
3. **Manual configuration**: Fine-tune workspace size and SVD algorithms
4. **Selective usage**: Use cuQuantum for χ ≥ 64, PyTorch for smaller bond dimensions
5. **Performance tuning**: Workspace size ~30% of GPU memory, choose SVD algorithm by χ
6. **Benchmarking**: Compare PyTorch vs cuQuantum for your specific workload
7. **Distributed cuQuantum**: Each GPU gets cuQuantum backend for multi-GPU systems

Performance expectations:

- **SVD operations**: 2-5× speedup (depends on χ and GPU)
- **Tensor contractions**: 2-10× speedup (larger speedup for complex contractions)
- **Overall simulation**: 1.5-3× speedup (depends on algorithm and problem)
- **Best speedup**: Large bond dimensions (χ ≥ 128) on modern GPUs (A100, H100)

cuQuantum is most beneficial for:

- Large bond dimensions (χ ≥ 64)
- Long simulations with many SVD operations (TDVP, VQE)
- Production workloads where performance is critical
- Modern NVIDIA GPUs (Ampere A100, Hopper H100)

See Also
========

- :doc:`../explanations/gpu_acceleration`: GPU optimization theory and best practices
- :doc:`optimize_performance`: General performance tuning strategies
- :doc:`handle_large_systems`: Distributed MPS with cuQuantum on multiple GPUs
- :doc:`configure_precision`: Precision considerations with cuQuantum
- :doc:`benchmark_comparison`: Benchmarking ATLAS-Q with different backends
