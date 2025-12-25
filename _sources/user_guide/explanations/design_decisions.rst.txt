================
Design Decisions
================

ATLAS-Q's design reflects deliberate choices balancing performance, usability, extensibility, and numerical stability. This document explains the key architectural decisions, their rationale, and alternative approaches considered.

.. contents::
   :local:
   :depth: 2

Overview
========

Design Philosophy
-----------------

ATLAS-Q follows these core principles:

1. **Performance by default**: GPU acceleration and optimized operations without configuration
2. **Fail safely**: Automatic numerical stability measures prevent silent errors
3. **Extensibility**: Modular architecture allows adding new algorithms and backends
4. **User control**: Override defaults when needed for advanced use cases
5. **Research-friendly**: Comprehensive diagnostics and statistics for debugging

**Key challenges addressed**:

- **Numerical instability**: SVD failures, truncation errors, ill-conditioning
- **Memory constraints**: Large bond dimensions exceed GPU memory
- **Performance bottlenecks**: SVD dominates runtime for χ>128
- **Usability**: Quantum computing experts may not be system programmers
- **Heterogeneous hardware**: CPUs, NVIDIA GPUs, future AMD/Intel GPUs

Alternative Approaches Considered
----------------------------------

**Option 1: NumPy-based implementation**

- Pros: Simple, pure Python, widely understood
- Cons: Poor GPU support, manual batching, slow for production
- **Decision**: Rejected due to GPU performance critical for >30 qubits

**Option 2: Custom CUDA kernels**

- Pros: Maximum performance control
- Cons: High development cost, maintenance burden, NVIDIA-only
- **Decision**: Hybrid approach - PyTorch default, Triton for hot paths

**Option 3: JAX-based**

- Pros: Excellent for automatic differentiation
- Cons: Less mature cuBLAS/cuSOLVER integration than PyTorch
- **Decision**: PyTorch chosen for better GPU libraries, more stable

Core Architecture
=================

MPS Representation
------------------

**Decision**: Store MPS as list of PyTorch tensors, not unified tensor

.. code-block:: python

   class AdaptiveMPS:
       def __init__(self, num_qubits, bond_dim, device='cuda'):
           self.tensors = [
               torch.zeros(chi_left, d, chi_right, dtype=torch.complex64, device=device)
               for i in range(num_qubits)
           ]

**Rationale**:

1. **Heterogeneous bond dimensions**: Each bond can have different χ
2. **Memory efficiency**: Only allocate needed dimensions
3. **Parallelization**: Operations on different sites can be independent
4. **Compatibility**: Standard format used by other MPS libraries

**Alternative considered**: Unified tensor with padding

- Pros: Single memory allocation, potential BLAS optimization
- Cons: Wasted memory for variable χ, complex indexing
- **Rejected**: Memory waste significant for adaptive bond dimensions

Adaptive vs Fixed Bond Dimension
---------------------------------

**Decision**: Adaptive bond dimension as default

**AdaptiveMPS** (default):

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,           # Initial χ
       chi_max_per_bond=256,  # Maximum allowed
       adaptive_mode=True     # Enable adaptation
   )

Automatically adjusts χ based on truncation error.

**MatrixProductStatePyTorch** (fixed):

.. code-block:: python

   from atlas_q.mps_pytorch import MatrixProductStatePyTorch
   mps = MatrixProductStatePyTorch(num_qubits=50, bond_dim=64)

Keeps χ=64 throughout simulation.

**Rationale**:

- **Adaptive**: Better accuracy/performance tradeoff for most use cases
- **Fixed**: Predictable memory usage, easier to reason about
- **Both**: Different algorithms have different needs (QAOA→fixed, TDVP→adaptive)

Statistics and Diagnostics
---------------------------

**Decision**: Track statistics by default with minimal overhead

.. code-block:: python

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply gates
   for i in range(100):
       mps.apply_cnot(i % 29, (i % 29) + 1)

   # Automatic statistics
   stats = mps.statistics
   print(f"Truncations: {stats.num_truncations}")
   print(f"Global error: {stats.total_truncation_error:.2e}")
   print(f"Max bond dimension: {max(s.shape[1] for s in mps.tensors)}")

**Rationale**:

1. **Observability**: Users should know simulation accuracy
2. **Research**: Essential for debugging and understanding algorithms
3. **Performance**: <1% overhead using lightweight counters
4. **Optional**: Can disable via ``track_statistics=False`` if needed

**Alternative**: No default tracking, require explicit requests

- Pros: Absolute minimum overhead
- Cons: Users unaware of errors, poor research experience
- **Rejected**: Observability too important for quantum simulation

API Design
==========

Device Agnostic Interface
--------------------------

**Decision**: Single API works on CPU and GPU

.. code-block:: python

   # Same code works on both devices
   mps_gpu = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')
   mps_cpu = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cpu')

   # Automatic fallback if CUDA unavailable
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda:0')
   # Falls back to CPU if no GPU

**Rationale**:

1. **Portability**: Same code runs on laptops (CPU) and servers (GPU)
2. **Testing**: CI/CD can test on CPU without GPU runners
3. **Development**: Prototyping on CPU before scaling to GPU
4. **Hybrid**: Some operations better on CPU (e.g., small χ)

Lazy vs Eager Imports
----------------------

**Decision**: Support both for compatibility

**Direct imports** (recommended):

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.vqe_qaoa import VQE
   from atlas_q.tdvp import TDVP

**Lazy imports** (legacy):

.. code-block:: python

   from atlas_q import get_adaptive_mps, get_vqe

   AdaptiveMPS = get_adaptive_mps()['AdaptiveMPS']
   VQE = get_vqe()['VQE']

**Rationale**:

- Direct imports fail if dependencies missing (e.g., PySCF not installed)
- Lazy imports allow partial functionality (core MPS without chemistry)
- Both supported for backwards compatibility with existing code

**Modern approach**: Optional dependencies properly declared in setup.py

.. code-block:: python

   pip install atlas-q              # Core only
   pip install atlas-q[chemistry]   # With PySCF
   pip install atlas-q[all]         # Everything

Automatic vs Manual Canonicalization
-------------------------------------

**Decision**: Automatic canonicalization before sensitive operations

.. code-block:: python

   # TDVP automatically canonicalizes
   tdvp = TDVP(mps=mps, hamiltonian=H, dt=0.05)
   tdvp.evolve_step()  # Automatic canonicalization ensures stability

   # Manual override if needed
   mps.canonicalize(center=15)  # Explicit control

**Rationale**:

1. **Correctness**: TDVP requires canonical form for numerical stability
2. **User-friendly**: Don't require quantum algorithm knowledge
3. **Performance**: Only when needed (not after every gate)
4. **Control**: Advanced users can manage manually

**Alternative**: Always manual, never automatic

- Pros: Explicit control, no hidden operations
- Cons: Easy to forget, leads to numerical errors
- **Rejected**: Too error-prone for non-expert users

Backend Selection
=================

PyTorch as Foundation
---------------------

**Decision**: PyTorch for tensor operations and GPU support

**Why PyTorch over alternatives**:

+-------------+-------------------+-------------------+-------------------+
| Feature     | PyTorch           | NumPy             | JAX               |
+=============+===================+===================+===================+
| GPU Support | Excellent         | Poor (CuPy)       | Good              |
+-------------+-------------------+-------------------+-------------------+
| cuBLAS      | Native            | Via CuPy          | Via XLA           |
+-------------+-------------------+-------------------+-------------------+
| SVD         | cuSOLVER          | Slow              | Good              |
+-------------+-------------------+-------------------+-------------------+
| Ecosystem   | Large             | Largest           | Growing           |
+-------------+-------------------+-------------------+-------------------+
| Maturity    | Very mature       | Most mature       | Maturing          |
+-------------+-------------------+-------------------+-------------------+
| Triton      | Compatible        | No                | Compatible        |
+-------------+-------------------+-------------------+-------------------+

**Rationale**:

1. **cuBLAS/cuSOLVER**: PyTorch has best integration
2. **Ecosystem**: Largest deep learning ecosystem, many tools
3. **GPU memory**: Excellent allocator, memory pooling
4. **JIT**: TorchScript for optimization (not yet used, future)
5. **Community**: Large user base, extensive documentation

**JAX consideration**:

JAX was seriously considered due to automatic differentiation for gradients. However:

- PyTorch autodiff sufficient for parameter-shift rule
- PyTorch's cuSOLVER integration more mature (critical for SVD)
- Could add JAX backend in future if needed (modular design allows this)

Triton for Custom Kernels
--------------------------

**Decision**: Triton for performance-critical kernels, PyTorch fallback

.. code-block:: python

   # Automatically uses Triton if available and beneficial (χ>64)
   from atlas_q.triton_kernels.mps_complex import fused_two_qubit_gate

   # Falls back to PyTorch if:
   # - Triton not installed
   # - χ too small (overhead dominates)
   # - Operation not implemented in Triton

**Rationale**:

1. **Performance**: 1.5-3× speedup for χ>64 via fusion
2. **Maintainability**: Python syntax easier than CUDA C++
3. **Portability**: Triton compiles for different GPUs
4. **Optional**: Not required, provides bonus performance

**CUDA C++ consideration**:

Could achieve 10-20% more performance, but:

- Development time 5-10× longer
- Maintenance burden much higher
- NVIDIA-only (Triton compiles for AMD ROCm potentially)

cuQuantum Integration
---------------------

**Decision**: Optional cuQuantum backend for >2× SVD speedup

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumConfig, CuQuantumBackend

   # Optional: Use cuQuantum for better performance
   config = CuQuantumConfig(svd_algorithm='gesvdj')
   backend = CuQuantumBackend(config, device='cuda')

   mps = AdaptiveMPS(num_qubits=40, bond_dim=256, backend=backend)

**Rationale**:

1. **Performance**: 2-3× faster SVD for χ>128 (dominant bottleneck)
2. **Optional**: Works without cuQuantum (PyTorch fallback)
3. **NVIDIA-specific**: Leverages cutting-edge GPU tensor network research
4. **Future-proof**: Can add other backends (AMD, Intel) similarly

**Alternative**: Require cuQuantum

- Pros: Always maximum performance
- Cons: NVIDIA-only, complicates installation
- **Rejected**: Portability and ease-of-install important

Numerical Stability
===================

Robust Linear Algebra
----------------------

**Decision**: Automatic fallbacks for ill-conditioned operations

.. code-block:: python

   from atlas_q.linalg_robust import robust_svd

   # Automatic fallback sequence:
   # 1. Try torch.linalg.svd on GPU
   # 2. Add Tikhonov regularization if ill-conditioned
   # 3. Fall back to CPU if GPU fails
   # 4. Try different LAPACK drivers

   U, S, Vh = robust_svd(tensor, threshold=1e-14)

**Rationale**:

1. **Reliability**: SVD failures crash simulations
2. **Automatic**: Users shouldn't debug linear algebra
3. **Performance**: Only use fallbacks when necessary
4. **Transparency**: Log warnings when using fallbacks

**Alternative**: Let SVD fail, user handles

- Pros: Explicit, no hidden behavior
- Cons: Most users can't debug cuSOLVER failures
- **Rejected**: Too difficult for typical users

Mixed Precision Strategy
------------------------

**Decision**: complex64 default, complex128 for stability

.. code-block:: python

   # Default: complex64 (faster, sufficient for most)
   mps = AdaptiveMPS(num_qubits=50, bond_dim=64, dtype=torch.complex64)

   # High precision when needed
   mps_precise = AdaptiveMPS(num_qubits=30, bond_dim=128, dtype=torch.complex128)

   # Automatic promotion policy
   from atlas_q.adaptive_mps import DTypePolicy
   policy = DTypePolicy(
       default=torch.complex64,
       promote_if_cond_gt=1e6  # Auto-upgrade on ill-conditioning
   )

**Rationale**:

1. **Performance**: complex64 is 2× faster and uses half memory
2. **Sufficient**: Most simulations don't need complex128 precision
3. **Safety**: Auto-promotion prevents silent errors
4. **Control**: Users can force complex128 if needed

Canonical Form Management
--------------------------

**Decision**: Maintain canonical form lazily, enforce before TDVP

.. code-block:: python

   # Gates don't canonicalize (fast)
   for i in range(100):
       mps.apply_cnot(i % 29, (i % 29) + 1)

   # Algorithms canonicalize when needed
   tdvp = TDVP(mps=mps, hamiltonian=H, dt=0.05)  # Canonicalizes here

   # Manual if needed
   mps.canonicalize(center=15)

**Rationale**:

1. **Performance**: Canonicalization is :math:`O(n \chi^3)` - expensive
2. **Not always needed**: Gate application doesn't require it
3. **Automatic for algorithms**: TDVP, measurements require it
4. **User control**: Can force when needed

Memory Management
=================

Adaptive Memory Budgets
-----------------------

**Decision**: Global memory budgets with automatic χ reduction

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=100,
       bond_dim=64,
       budget_global_mb=10 * 1024,  # 10 GB limit
       adaptive_mode=True
   )

   # Automatically reduces χ if budget exceeded
   # Tracks truncation error from budget constraints

**Rationale**:

1. **Prevent OOM**: GPU out-of-memory crashes entire process
2. **Graceful degradation**: Reduce χ rather than crash
3. **Transparency**: Track error from budget reduction
4. **Research**: Can disable for maximum accuracy

**Alternative**: Crash on OOM

- Pros: Explicit failure
- Cons: Lose entire computation
- **Rejected**: Graceful degradation better for long-running simulations

PyTorch Allocator Strategy
---------------------------

**Decision**: Use PyTorch's caching allocator, clear cache strategically

.. code-block:: python

   # PyTorch caches freed memory (fast)
   # Manually clear when switching tasks
   import torch
   torch.cuda.empty_cache()  # Return memory to CUDA

**Rationale**:

1. **Performance**: Allocation ~100× faster with cache
2. **Fragmentation**: Large simulations fragment memory
3. **Control**: User can clear when needed
4. **Automatic**: PyTorch manages cache size

In-Place Operations
-------------------

**Decision**: Use in-place operations for large tensors when safe

.. code-block:: python

   # In-place (good for memory)
   mps.tensors[i].mul_(factor)

   # Out-of-place (creates copy)
   mps.tensors[i] = mps.tensors[i] * factor

**Rationale**:

1. **Memory**: Avoid temporary copies for large tensors
2. **Safety**: Only when semantics allow (no dependencies)
3. **Performance**: Reduces memory traffic

Modularity and Extension
=========================

Backend Interface
-----------------

**Decision**: Abstract backend interface for different implementations

.. code-block:: python

   class TensorBackend:
       def svd(self, tensor): ...
       def qr(self, tensor): ...
       def contract(self, a, b): ...

   # Implementations
   class PyTorchBackend(TensorBackend): ...
   class CuQuantumBackend(TensorBackend): ...
   # Future: JAXBackend, NumPyBackend

**Rationale**:

1. **Extensibility**: Add new backends (AMD, Intel) without touching core
2. **Testing**: Mock backend for unit tests
3. **Comparison**: Benchmark different backends easily
4. **Future-proof**: Quantum hardware may have custom backends

Algorithm Plugin Architecture
------------------------------

**Decision**: Algorithms are separate classes, not methods on MPS

.. code-block:: python

   # Good: Algorithm as separate class
   from atlas_q.tdvp import TDVP
   tdvp = TDVP(mps=mps, hamiltonian=H, dt=0.05)
   tdvp.evolve_step()

   # Bad: Algorithm as MPS method
   # mps.tdvp_evolve(hamiltonian=H, dt=0.05)  # Not scalable

**Rationale**:

1. **Separation of concerns**: MPS is data structure, algorithms use it
2. **Extensibility**: Add algorithms without modifying MPS class
3. **Testing**: Test algorithms independently
4. **State**: Algorithms maintain their own state (environment tensors, etc.)

**Alternative**: Algorithms as MPS methods

- Pros: Simpler API, fewer imports
- Cons: MPS class becomes huge, hard to extend
- **Rejected**: Poor separation of concerns

Flexible Gate Interface
-----------------------

**Decision**: Support multiple gate specification formats

.. code-block:: python

   # Matrix format
   gate = np.array([[1, 0], [0, -1]])
   mps.apply_one_site_gate(gate, site=5)

   # String format (convenience)
   mps.apply_h(5)     # Hadamard
   mps.apply_cnot(5, 6)  # CNOT

   # Batch format
   mps.apply_batch_single_gates('H', sites=[0, 2, 4, 6])

**Rationale**:

1. **Convenience**: String format for common gates
2. **Flexibility**: Matrix for custom gates
3. **Performance**: Batch for many gates
4. **Compatibility**: Different use cases prefer different formats

Error Handling
==============

Fail-Fast Philosophy
--------------------

**Decision**: Detect errors early, fail with clear messages

.. code-block:: python

   mps = AdaptiveMPS(num_qubits=50, bond_dim=64)

   # Fail fast on invalid input
   try:
       mps.apply_cnot(49, 50)  # Out of bounds
   except ValueError as e:
       print(e)  # "qubit_b=50 out of range [0, 49]"

**Rationale**:

1. **Debugging**: Catch errors at source, not later
2. **Clear messages**: Include values and valid ranges
3. **Type safety**: Validate inputs before expensive operations

Warnings for Numerical Issues
------------------------------

**Decision**: Warn on numerical instability, don't crash

.. code-block:: python

   import warnings

   # Warn on high condition number
   if condition_number > 1e6:
       warnings.warn(f"High condition number {cond:.2e}, consider complex128")

   # Warn on large truncation error
   if global_error > user_threshold:
       warnings.warn(f"Global error {global_error:.2e} exceeds threshold")

**Rationale**:

1. **Don't crash**: Simulations can continue
2. **Inform user**: They can decide if error acceptable
3. **Actionable**: Message suggests fix (use complex128, increase χ)

Graceful Degradation
--------------------

**Decision**: Fall back to slower but correct operations

.. code-block:: python

   # Try fast path
   try:
       U, S, Vh = torch.linalg.svd(tensor)
   except RuntimeError:
       # Fall back to CPU
       tensor_cpu = tensor.cpu()
       U, S, Vh = torch.linalg.svd(tensor_cpu)
       U, S, Vh = U.cuda(), S.cuda(), Vh.cuda()
       warnings.warn("SVD failed on GPU, fell back to CPU")

**Rationale**:

1. **Reliability**: Simulation completes even if GPU SVD fails
2. **Transparency**: User informed of performance degradation
3. **Research**: Don't lose hours of computation to rare GPU glitch

Future-Proofing
===============

Extensibility Points
--------------------

**Designed for extension**:

1. **Backends**: Add JAX, NumPy, vendor-specific backends
2. **Algorithms**: Plugin new algorithms (MERA, PEPS evolution)
3. **Gates**: Custom gates via matrix interface
4. **Noise models**: Extensible noise model framework
5. **Hamiltonians**: Custom MPO construction

Version Compatibility
---------------------

**Decision**: Maintain backwards compatibility for 1.x releases

.. code-block:: python

   # Legacy APIs kept for compatibility
   from atlas_q import get_adaptive_mps  # Old style
   from atlas_q.adaptive_mps import AdaptiveMPS  # New style
   # Both work

**Rationale**:

1. **User trust**: Breaking changes hurt adoption
2. **Deprecation path**: Warnings for 2 releases before removal
3. **Documentation**: Clear migration guides
4. **Semantic versioning**: Follow semver strictly

Configuration System
--------------------

**Decision**: Dataclasses for configuration, not dicts

.. code-block:: python

   from dataclasses import dataclass

   @dataclass
   class VQEConfig:
       max_iterations: int = 500
       optimizer: str = 'L-BFGS-B'
       tol: float = 1e-6
       bond_dim: int = 64

   # Type-checked, autocomplete in IDEs
   config = VQEConfig(max_iterations=1000, bond_dim=128)

**Rationale**:

1. **Type safety**: Catch typos at development time
2. **Documentation**: Self-documenting via type annotations
3. **IDE support**: Autocomplete, type checking
4. **Validation**: Can add validators to dataclass

Lessons Learned
===============

What Worked Well
----------------

1. **PyTorch choice**: GPU support and ecosystem exceeded expectations
2. **Adaptive bond dimension**: Default for 90% of use cases
3. **Statistics tracking**: Essential for research, <1% overhead
4. **Modular backends**: Made cuQuantum integration trivial
5. **Device-agnostic API**: Smooth development→production transition

What We'd Change
----------------

1. **Earlier cuQuantum integration**: Should have been day-one priority
2. **Dataclasses from start**: Would have caught more bugs early
3. **More aggressive in-place**: Could save 10-20% memory
4. **Benchmark suite**: Should have built comprehensive benchmarks earlier
5. **JAX exploration**: Should pilot JAX backend for comparison

Trade-Offs We Accept
--------------------

1. **PyTorch dependency**: Heavy, but worth it for GPU support
2. **Statistics overhead**: <1% performance for essential observability
3. **Automatic canonicalization**: Hidden operation, but prevents errors
4. **Memory for speed**: Cache allocations rather than minimize memory
5. **Complexity for performance**: Triton kernels add complexity for 2-3× speedup

Summary
=======

ATLAS-Q's design prioritizes:

**Core Decisions**:

1. **PyTorch foundation**: Best GPU support and ecosystem
2. **Adaptive by default**: Better accuracy/performance balance
3. **Fail safely**: Automatic stability measures, graceful degradation
4. **Modular architecture**: Clean separation, extensible backends
5. **Observable**: Comprehensive statistics and diagnostics

**Key Innovations**:

- Adaptive bond dimensions with memory budgets
- Robust linear algebra with automatic fallbacks
- Mixed precision with automatic promotion
- Modular backend system for future extensibility
- Comprehensive statistics with minimal overhead

**Trade-Offs**:

- Heavy PyTorch dependency for GPU support
- Some hidden operations (canonicalization) for stability
- Memory overhead for performance (caching, statistics)

**Design Principles**:

1. **Performance by default**: Fast without configuration
2. **Fail safely**: Prevent silent numerical errors
3. **User control**: Override defaults when needed
4. **Extensibility**: Add new backends and algorithms easily
5. **Research-friendly**: Observe simulation internals

These decisions enable ATLAS-Q to:

- Simulate 100+ qubits on single GPU
- Achieve 10-100× CPU speedup
- Maintain numerical stability automatically
- Support research with comprehensive diagnostics
- Extend to new hardware and algorithms

For implementation details of specific design decisions, see:

- :doc:`gpu_acceleration` - GPU backend implementation
- :doc:`numerical_stability` - Stability mechanisms
- :doc:`performance_model` - Performance implications of design choices
- :doc:`comparisons` - How ATLAS-Q compares to alternative designs
