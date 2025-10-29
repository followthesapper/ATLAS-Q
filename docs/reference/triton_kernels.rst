Triton GPU Kernels
==================

Overview
--------

ATLAS-Q includes custom Triton GPU kernels for accelerated tensor operations. Triton integration is embedded within core modules rather than exposed as a standalone ``triton_kernels`` module. Key features include:

- Custom kernels for MPS gate operations (1.5-3× speedup)
- GPU-optimized tensor contractions with Tensor Core utilization
- Modular exponentiation kernels for period-finding
- Automatic fallback to PyTorch when Triton is unavailable

Performance improvements are most significant for bond dimensions χ > 32 and moderate to large qubit counts (20+).

Architecture
------------

Triton kernels are integrated into:

- ``adaptive_mps`` - MPS gate application kernels
- ``quantum_hybrid_system`` - Modular exponentiation for factorization
- ``mpo_ops`` - MPO-MPS contraction kernels

The integration is transparent: operations automatically use Triton kernels when available, falling back to standard PyTorch operations otherwise.

Installation
------------

Triton support requires:

.. code-block:: bash

   pip install triton

Or install ATLAS-Q with GPU support:

.. code-block:: bash

   pip install atlas-quantum[gpu]

Triton works with NVIDIA GPUs (compute capability 7.0+) and requires CUDA toolkit.

Kernel Types
------------

Gate Application Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^

Accelerated single-qubit and two-qubit gate operations:

- Single-qubit gates: Fused tensor reshape and gate application
- Two-qubit gates: Optimized two-site tensor contraction
- Batched gate operations for multiple gates

Typical speedup: 1.5-2× over PyTorch for χ > 32.

Tensor Contraction Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optimized Einstein summation for MPS operations:

- MPO-MPS contractions
- Bond merging and SVD preparation
- Multi-bond operations

Utilizes Tensor Cores on Ampere+ GPUs for additional acceleration.

Typical speedup: 2-3× over PyTorch for large contractions.

Modular Exponentiation Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fast modular arithmetic for period-finding:

- Batch modular exponentiation: a^x mod N
- Montgomery multiplication
- GPU-parallel evaluation

Typical speedup: 100-1000× over CPU for large batches.

Usage
-----

Automatic (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

Triton kernels are used automatically when available:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Triton kernels automatically used if available
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
   H = H.to('cuda')

   # This uses Triton-accelerated gate application
   for q in range(30):
       mps.apply_single_qubit_gate(q, H)

Manual Control
^^^^^^^^^^^^^^

Disable Triton kernels for benchmarking:

.. code-block:: python

   import os

   # Disable Triton before importing ATLAS-Q
   os.environ['ATLAS_Q_USE_TRITON'] = '0'

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Now uses standard PyTorch operations
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

Verification
^^^^^^^^^^^^

Check if Triton is being used:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Check statistics for kernel usage
   stats = mps.stats_summary()

   # Triton usage reflected in performance metrics
   print(f"Average operation time: {stats['total_time_ms'] / stats['total_operations']:.2f} ms")

Performance Characteristics
---------------------------

Speedup by System Size
^^^^^^^^^^^^^^^^^^^^^^

Expected speedup factors:

- Small systems (χ ≤ 16): Minimal (overhead may dominate)
- Medium systems (χ = 32-128): 1.5-2.5×
- Large systems (χ ≥ 256): 2-3×

Speedup increases with:

- Larger bond dimensions
- More qubits
- Repeated operations (kernel compilation amortized)

GPU Requirements
^^^^^^^^^^^^^^^^

Optimal performance requires:

- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Hopper)
- CUDA 11.0+
- Recommended: A100, H100, RTX 4090 for Tensor Core utilization

Memory Overhead
^^^^^^^^^^^^^^^

Triton kernels require minimal additional memory:

- Kernel cache: ~10-50 MB
- Intermediate buffers: Proportional to operation size

Total overhead typically < 100 MB.

Compilation
-----------

First Invocation
^^^^^^^^^^^^^^^^

Triton kernels are JIT-compiled on first use, adding latency (typically 1-5 seconds):

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import time

   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / torch.sqrt(torch.tensor(2.0))

   # First call: includes compilation
   start = time.time()
   mps.apply_single_qubit_gate(0, H)
   first_time = time.time() - start

   # Subsequent calls: use cached kernel
   start = time.time()
   mps.apply_single_qubit_gate(1, H)
   cached_time = time.time() - start

   print(f"First call: {first_time*1000:.1f} ms (includes compilation)")
   print(f"Cached call: {cached_time*1000:.1f} ms")

Persistent Caching
^^^^^^^^^^^^^^^^^^

Compiled kernels are cached across sessions in ``~/.triton/cache/``. Cache can be cleared if issues arise:

.. code-block:: bash

   rm -rf ~/.triton/cache

Troubleshooting
---------------

Triton Not Found
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install triton

Compilation Errors
^^^^^^^^^^^^^^^^^^

Update Triton to latest version:

.. code-block:: bash

   pip install --upgrade triton

Slower Than Expected
^^^^^^^^^^^^^^^^^^^^

- Ensure GPU has compute capability 7.0+
- Check that operations are large enough to benefit (χ > 32)
- Verify CUDA drivers are up to date

Debugging
^^^^^^^^^

Enable Triton debug output:

.. code-block:: python

   import os
   os.environ['TRITON_DEBUG'] = '1'

Examples
--------

Benchmarking Triton vs PyTorch:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import time
   import os

   # Benchmark with Triton
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / torch.sqrt(torch.tensor(2.0))

   # Warmup
   for q in range(5):
       mps.apply_single_qubit_gate(q, H)

   # Benchmark
   start = time.time()
   for q in range(100):
       mps.apply_single_qubit_gate(q % 30, H)
   triton_time = time.time() - start

   # Benchmark without Triton (restart Python or use different process)
   # os.environ['ATLAS_Q_USE_TRITON'] = '0'
   # ... repeat benchmark ...

   print(f"Triton: {triton_time:.3f}s for 100 operations")
   print(f"Throughput: {100/triton_time:.1f} ops/sec")

Best Practices
--------------

**Enabling Triton**

Set environment variable before importing:

.. code-block:: bash

   export ATLAS_Q_USE_TRITON=1
   python your_script.py

**When to Use Triton Kernels**

- χ ≥ 32: Measurable speedup
- χ ≥ 64: Significant speedup (1.5-2.5×)
- Batch operations: Element-wise ops on many tensors
- Custom operations: Write domain-specific kernels

**Performance Tips**

1. Use power-of-2 bond dimensions (32, 64, 128) for optimal memory coalescing
2. Batch tensor operations when possible
3. Profile with ``torch.profiler`` to identify bottlenecks
4. Consider Triton for custom gates not in standard library

Use Cases
---------

**Ideal For**

- Production MPS simulations with χ > 32
- Custom quantum gate implementations
- Element-wise tensor operations on GPU
- Research requiring maximum GPU utilization

**Not Needed For**

- Small bond dimensions (χ < 32)
- CPU-only systems
- Standard operations where PyTorch is sufficient

See Also
--------

- :doc:`adaptive_mps` - MPS module using Triton kernels
- :doc:`cuquantum_backend` - Alternative GPU acceleration
- :doc:`quantum_hybrid_system` - Period-finding with GPU acceleration
- :doc:`../user_guide/howtos/optimize_performance` - Performance optimization guide
- :doc:`../user_guide/explanations/gpu_acceleration` - GPU acceleration details

References
----------

.. [Triton] OpenAI Triton, https://github.com/openai/triton

.. [Tillet19] P. Tillet et al., "Triton: An intermediate language and compiler for tiled neural network computations," *MAPL 2019* (2019).

.. [CUDA] NVIDIA CUDA Programming Guide, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
