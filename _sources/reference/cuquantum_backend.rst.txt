atlas_q.cuquantum_backend
==========================

.. automodule:: atlas_q.cuquantum_backend
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``cuquantum_backend`` module provides optional NVIDIA cuQuantum acceleration for tensor operations in ATLAS-Q. Key features include:

- cuTensorNet integration for accelerated tensor contractions and SVD
- cuStateVec support for state-vector operations
- Automatic fallback to PyTorch if cuQuantum is unavailable
- Version compatibility handling
- 2-10× speedup on compatible NVIDIA GPUs

This module is optional and ATLAS-Q functions normally without it, using PyTorch as the backend.

Installation
------------

To enable cuQuantum acceleration:

.. code-block:: bash

   pip install cuquantum-python

Requires NVIDIA GPU with CUDA support and cuQuantum library (typically ~320MB download).

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CuQuantumConfig
   CuQuantumBackend

CuQuantumConfig
---------------

.. autoclass:: CuQuantumConfig
   :members:
   :undoc-members:

   Configuration dataclass for cuQuantum backend with the following options:

   .. attribute:: use_cutensornet
      :type: bool

      Enable cuTensorNet for tensor contractions (default: True)

   .. attribute:: use_custatevec
      :type: bool

      Enable cuStateVec for state vector operations (default: True)

   .. attribute:: workspace_size
      :type: int

      GPU workspace memory in bytes (default: 1GB)

   .. attribute:: algorithm
      :type: str

      SVD algorithm selection: 'auto', 'gesvd', 'gesvdj', 'gesvdp' (default: 'auto')

   .. attribute:: device
      :type: str

      Compute device (default: 'cuda')

CuQuantumBackend
----------------

.. autoclass:: CuQuantumBackend
   :members:
   :undoc-members:
   :show-inheritance:

   Primary interface for cuQuantum-accelerated operations.

   .. rubric:: Methods

   .. autosummary::

      ~CuQuantumBackend.__init__
      ~CuQuantumBackend.svd
      ~CuQuantumBackend.contract

   Automatically detects cuQuantum availability and falls back to PyTorch if:

   - cuQuantum is not installed
   - Initialization fails
   - Individual operations fail

Examples
--------

Basic usage with automatic detection:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend
   import torch

   # Backend automatically detects cuQuantum
   backend = CuQuantumBackend()

   if backend.available:
       print(f"cuQuantum {backend.version} detected")
   else:
       print("Using PyTorch backend")

   # Perform SVD (uses cuQuantum if available, else PyTorch)
   tensor = torch.randn(100, 50, dtype=torch.complex64, device='cuda')
   U, S, Vt = backend.svd(tensor, chi_max=32)

   print(f"Truncated to {len(S)} singular values")

Custom configuration:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend, CuQuantumConfig

   # Configure cuQuantum parameters
   config = CuQuantumConfig(
       use_cutensornet=True,
       use_custatevec=True,
       workspace_size=2 * 1024**3,  # 2GB workspace
       algorithm='gesvdj',  # Jacobi SVD
       device='cuda:0'
   )

   backend = CuQuantumBackend(config)

   # SVD with custom config
   tensor = torch.randn(200, 100, dtype=torch.complex64, device='cuda:0')
   U, S, Vt = backend.svd(tensor, chi_max=64, cutoff=1e-12)

Integration with AdaptiveMPS:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.cuquantum_backend import CuQuantumBackend

   # Initialize cuQuantum backend
   cu_backend = CuQuantumBackend()

   # Create MPS (automatically uses cuQuantum if available)
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=16,
       device='cuda'
   )

   # AdaptiveMPS will use cuQuantum backend internally if available
   # Apply gates - operations accelerated by cuQuantum
   import torch
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
   H = H.to('cuda')

   for q in range(30):
       mps.apply_single_qubit_gate(q, H)

Tensor contraction:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend
   import torch

   backend = CuQuantumBackend()

   # Create tensors
   A = torch.randn(10, 20, 30, dtype=torch.complex64, device='cuda')
   B = torch.randn(30, 40, 50, dtype=torch.complex64, device='cuda')

   # Contract using Einstein notation
   C = backend.contract([A, B], 'ijk,klm->ijlm', optimize='auto')

   print(f"Contraction result shape: {C.shape}")

Checking cuQuantum availability:

.. code-block:: python

   from atlas_q.cuquantum_backend import CUQUANTUM_AVAILABLE, CUQUANTUM_VERSION

   if CUQUANTUM_AVAILABLE:
       print(f"cuQuantum version {CUQUANTUM_VERSION} is available")
       print("GPU-accelerated operations enabled")
   else:
       print("cuQuantum not available")
       print("Install with: pip install cuquantum-python")

Performance Considerations
--------------------------

Speedup Factors
^^^^^^^^^^^^^^^

Expected speedup with cuQuantum:

- SVD operations: 2-5× for χ > 64
- Tensor contractions: 3-10× for large tensors
- Overall MPS operations: 1.5-3× average

GPU Requirements
^^^^^^^^^^^^^^^^

- NVIDIA GPU with CUDA compute capability 7.0+ (Volta, Turing, Ampere, Hopper)
- CUDA Toolkit 11.0+
- Recommended: A100, H100, or RTX 4090

Memory Usage
^^^^^^^^^^^^

cuQuantum requires additional GPU workspace memory (configurable via ``workspace_size``). Default is 1GB, but larger workspaces can improve performance for large tensors.

Fallback Behavior
-----------------

The backend gracefully handles failures:

1. If cuQuantum is not installed: All operations use PyTorch
2. If initialization fails: Falls back to PyTorch with warning
3. If individual operations fail: Automatic PyTorch fallback for that operation

Warnings are printed to help diagnose issues, but simulation continues.

Compatibility
-------------

Tested with:

- cuQuantum 23.x - 25.x
- CUDA 11.8+
- PyTorch 2.0+
- NVIDIA A100, H100, RTX 4090

Version-specific features are auto-detected and handled.

Troubleshooting
---------------

cuQuantum not detected
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.cuquantum_backend import CUQUANTUM_AVAILABLE

   if not CUQUANTUM_AVAILABLE:
       # Try installing
       # pip install cuquantum-python
       pass

Out of memory errors
^^^^^^^^^^^^^^^^^^^^

Reduce workspace size:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend

   config = CuQuantumConfig(workspace_size=512 * 1024**2)  # 512MB
   backend = CuQuantumBackend(config)

Slower than PyTorch
^^^^^^^^^^^^^^^^^^^

cuQuantum overhead is only worthwhile for larger tensors. For small systems (χ < 32), PyTorch may be faster. Consider disabling cuQuantum for small simulations.

Best Practices
--------------

**When to Use cuQuantum**

- χ > 32: Significant speedup (1.5-3×)
- χ > 64: Major speedup (2-5×)
- Long-running simulations: Reduced wall-clock time
- Multi-GPU: cuQuantum's distributed capabilities

**When to Use PyTorch**

- χ < 32: Overhead dominates
- Rapid prototyping: Simpler setup
- CPU-only systems: cuQuantum requires NVIDIA GPU

**Optimization Tips**

1. Increase workspace_size for better performance (2-4GB recommended)
2. Use 'gesvdj' algorithm for moderate χ (32-128)
3. Enable both cuTensorNet and cuStateVec for best results
4. Monitor fallback rate - should be < 1%

Use Cases
---------

**Ideal Applications**

- Large-scale tensor network simulations (χ > 64)
- Production systems requiring maximum performance
- Multi-hour simulations where 2× speedup matters
- Research requiring state-of-the-art performance

**Not Recommended**

- Small systems (N < 20, χ < 32)
- Educational/tutorial code (unnecessary complexity)
- Systems without NVIDIA GPUs

See Also
--------

- :doc:`adaptive_mps` - MPS operations accelerated by cuQuantum
- :doc:`linalg_robust` - Alternative robust linear algebra
- :doc:`triton_kernels` - Custom GPU kernels for specific operations
- :doc:`../user_guide/howtos/integrate_cuquantum` - Integration guide
- :doc:`../user_guide/howtos/optimize_performance` - Performance optimization

References
----------

.. [cuQuantum] NVIDIA cuQuantum SDK, https://developer.nvidia.com/cuquantum-sdk

.. [cuTensorNet] cuTensorNet Documentation, https://docs.nvidia.com/cuda/cuquantum/cutensornet/index.html

.. [Lykov22] D. Lykov et al., "Tensor network quantum simulator with step-dependent parallelization," *arXiv:2212.14703* (2022).
