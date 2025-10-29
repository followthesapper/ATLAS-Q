How-To Guides
=============

Problem-oriented guides for specific tasks. Each guide addresses a particular challenge and provides practical solutions.

.. toctree::
   :maxdepth: 2

   optimize_performance
   handle_large_systems
   configure_precision
   integrate_cuquantum
   debug_simulations
   save_load_state
   custom_hamiltonians
   parallel_computation
   benchmark_comparison

Guide Overview
--------------

:doc:`optimize_performance`
   Maximize simulation speed using custom Triton kernels, GPU optimization, and efficient tensor operations.

:doc:`handle_large_systems`
   Simulate systems beyond single-GPU memory limits using adaptive truncation, distributed MPS, and memory budgets.

:doc:`configure_precision`
   Choose appropriate numerical precision (complex32/64/128) and configure mixed-precision policies.

:doc:`integrate_cuquantum`
   Enable NVIDIA cuQuantum acceleration for 2-10× speedup on supported operations.

:doc:`debug_simulations`
   Diagnose numerical issues, track error propagation, and validate simulation correctness.

:doc:`save_load_state`
   Checkpoint MPS states, save optimization progress, and resume long-running simulations.

:doc:`custom_hamiltonians`
   Build custom Hamiltonians using MPO operations, including non-local interactions and time-dependent terms.

:doc:`parallel_computation`
   Leverage multi-GPU parallelism with distributed MPS and data-parallel measurement sampling.

:doc:`benchmark_comparison`
   Compare ATLAS-Q performance against Qiskit, Cirq, and ITensor for specific use cases.

Prerequisites
-------------

Guides assume familiarity with basic ATLAS-Q usage. If you are new to ATLAS-Q, complete the :doc:`../tutorials/index` first.

Quick Navigation
----------------

Common tasks:

- **Slow simulations?** See :doc:`optimize_performance`
- **Out of memory?** See :doc:`handle_large_systems`
- **Numerical instability?** See :doc:`configure_precision` and :doc:`debug_simulations`
- **Multi-GPU setup?** See :doc:`parallel_computation`
- **Custom Hamiltonians?** See :doc:`custom_hamiltonians`
