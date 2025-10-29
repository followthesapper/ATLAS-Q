API Reference
=============

Complete API documentation for ATLAS-Q modules, classes, and functions.

.. toctree::
   :maxdepth: 2

   adaptive_mps
   mps_pytorch
   mpo_ops
   tdvp
   vqe_qaoa
   grover
   stabilizer_backend
   noise_models
   peps
   circuit_cutting
   planar_2d
   distributed_mps
   cuquantum_backend
   quantum_hybrid_system
   triton_kernels
   diagnostics
   linalg_robust
   truncation

Module Overview
---------------

Core Simulation
^^^^^^^^^^^^^^^

:doc:`adaptive_mps`
   Adaptive Matrix Product States with per-bond dimension control and error tracking.

:doc:`mps_pytorch`
   Basic PyTorch-based MPS implementation.

:doc:`mpo_ops`
   Matrix Product Operators for Hamiltonians and observables.

Variational Algorithms
^^^^^^^^^^^^^^^^^^^^^^

:doc:`vqe_qaoa`
   Variational Quantum Eigensolver and Quantum Approximate Optimization Algorithm.

:doc:`tdvp`
   Time-Dependent Variational Principle for quantum dynamics.

:doc:`grover`
   Grover's quantum search algorithm for unstructured database search with quadratic speedup.

Advanced Backends
^^^^^^^^^^^^^^^^^

:doc:`stabilizer_backend`
   Efficient simulation of Clifford circuits via stabilizer formalism.

:doc:`peps`
   Projected Entangled Pair States for 2D tensor networks.

:doc:`cuquantum_backend`
   NVIDIA cuQuantum integration for GPU acceleration.

Specialized Features
^^^^^^^^^^^^^^^^^^^^

:doc:`circuit_cutting`
   Circuit partitioning and entanglement forging.

:doc:`planar_2d`
   2D qubit layouts and SWAP synthesis.

:doc:`distributed_mps`
   Multi-GPU distributed simulation.

:doc:`noise_models`
   NISQ noise channels and error models.

Period-Finding
^^^^^^^^^^^^^^

:doc:`quantum_hybrid_system`
   Compressed quantum states and period-finding for Shor's algorithm.

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

:doc:`triton_kernels`
   Custom GPU kernels for tensor operations.

Utilities
^^^^^^^^^

:doc:`diagnostics`
   Monitoring, statistics, and entropy calculations.

:doc:`linalg_robust`
   Robust linear algebra with automatic fallbacks.

:doc:`truncation`
   Truncation strategies and error bounds.

Quick Access
------------

Common classes:

- :class:`atlas_q.adaptive_mps.AdaptiveMPS` - Adaptive MPS
- :class:`atlas_q.mpo_ops.MPO` - Matrix Product Operator
- :class:`atlas_q.mpo_ops.MPOBuilder` - Hamiltonian builder
- :class:`atlas_q.vqe_qaoa.VQE` - Variational Quantum Eigensolver
- :class:`atlas_q.vqe_qaoa.QAOA` - Quantum Approximate Optimization Algorithm
- :class:`atlas_q.grover.GroverSearch` - Grover's quantum search
- :class:`atlas_q.grover.GroverConfig` - Grover configuration
- :class:`atlas_q.tdvp.TDVP1Site` - 1-site TDVP evolution
- :class:`atlas_q.tdvp.TDVP2Site` - 2-site TDVP evolution
- :class:`atlas_q.stabilizer_backend.StabilizerSimulator` - Stabilizer simulator
- :class:`atlas_q.peps.PEPS` - PEPS tensor network
- :class:`atlas_q.quantum_hybrid_system.QuantumClassicalHybrid` - Period-finding

Common functions:

- :func:`atlas_q.get_quantum_sim` - Get period-finding classes
- :func:`atlas_q.get_adaptive_mps` - Get adaptive MPS classes
- :func:`atlas_q.get_mpo_ops` - Get MPO operations
- :func:`atlas_q.get_tdvp` - Get TDVP classes
- :func:`atlas_q.get_vqe_qaoa` - Get VQE/QAOA classes
- :func:`atlas_q.get_stabilizer` - Get stabilizer simulator
- :func:`atlas_q.get_peps` - Get PEPS classes
- :func:`atlas_q.grover.grover_search` - Convenience function for Grover search
- :func:`atlas_q.grover.calculate_grover_iterations` - Calculate optimal iterations
- :func:`atlas_q.mpo_ops.expectation_value` - Compute expectation values
- :func:`atlas_q.mpo_ops.apply_mpo_to_mps` - Apply MPO to MPS

Module Import Patterns
----------------------

Direct imports (recommended):

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

Lazy imports (legacy compatibility):

.. code-block:: python

   from atlas_q import get_adaptive_mps, get_mpo_ops, get_vqe_qaoa

   mps_mod = get_adaptive_mps()
   AdaptiveMPS = mps_mod['AdaptiveMPS']

   mpo_mod = get_mpo_ops()
   MPOBuilder = mpo_mod['MPOBuilder']

   vqe_mod = get_vqe_qaoa()
   VQE = vqe_mod['VQE']

Module imports:

.. code-block:: python

   from atlas_q import adaptive_mps, mpo_ops, vqe_qaoa

   mps = adaptive_mps.AdaptiveMPS(10, bond_dim=8)
   H = mpo_ops.MPOBuilder.ising_hamiltonian(10)
   vqe = vqe_qaoa.VQE(H, vqe_qaoa.VQEConfig())

Type Annotations
----------------

ATLAS-Q uses type hints throughout. Common types:

- ``torch.Tensor`` - PyTorch tensors
- ``torch.dtype`` - Data types (``torch.complex64``, ``torch.complex128``)
- ``torch.device`` - Compute devices (``'cuda'``, ``'cpu'``)
- ``np.ndarray`` - NumPy arrays
- ``int`` - Integers
- ``float`` - Floating-point numbers
- ``complex`` - Complex numbers
- ``str`` - Strings
- ``Optional[T]`` - Optional values
- ``List[T]`` - Lists
- ``Dict[K, V]`` - Dictionaries
- ``Tuple[T, ...]`` - Tuples

Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
