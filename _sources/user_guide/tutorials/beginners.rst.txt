Beginner's Tutorial
===================

Welcome to ATLAS-Q! This tutorial introduces the fundamentals of quantum simulation using tensor networks. By the end, you will understand how to create quantum states, apply gates, and run basic simulations.

Prerequisites
-------------

This tutorial assumes:

- Basic Python knowledge (variables, functions, loops)
- Familiarity with NumPy arrays
- Understanding of quantum mechanics basics (qubits, gates, measurements)
- ATLAS-Q installed (see :doc:`../../installation`)

If you are new to quantum computing, consider reviewing introductory materials on qubits, quantum gates, and measurement first.

Learning Objectives
-------------------

After completing this tutorial, you will be able to:

1. Create Matrix Product States (MPS) for quantum simulation
2. Apply single-qubit and two-qubit gates
3. Measure quantum states and interpret results
4. Factor numbers using period-finding
5. Understand bond dimensions and truncation
6. Monitor simulation statistics and memory usage

Installation Verification
--------------------------

First, verify ATLAS-Q is installed correctly:

.. code-block:: python

   import atlas_q
   print(f"ATLAS-Q version: {atlas_q.__version__}")

   # Check for GPU support
   import torch
   if torch.cuda.is_available():
       print(f"GPU available: {torch.cuda.get_device_name(0)}")
   else:
       print("Running on CPU (GPU recommended for larger simulations)")

Part 1: Understanding Matrix Product States
--------------------------------------------

What is an MPS?
^^^^^^^^^^^^^^^

A Matrix Product State (MPS) is a compressed representation of quantum states. Instead of storing all 2ⁿ amplitudes for n qubits, MPS stores only O(n·χ²) parameters, where χ is the bond dimension.

For example:

- 30 qubits full state: 2³⁰ = 1,073,741,824 complex numbers (~16 GB)
- 30 qubits MPS (χ=64): 30 × 64² = 122,880 complex numbers (~0.002 GB)

This is a 626,000× memory reduction!

Creating Your First MPS
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create a 10-qubit MPS on GPU with bond dimension 8
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Check memory usage
   memory_mb = mps.memory_usage() / (1024**2)
   print(f"MPS memory: {memory_mb:.2f} MB")

   # Get statistics
   stats = mps.stats_summary()
   print(f"Bond dimensions: {stats}")

Bond Dimension (χ)
^^^^^^^^^^^^^^^^^^

The bond dimension controls the accuracy vs efficiency tradeoff:

- χ=1: Product states only (no entanglement)
- χ=8-32: Weakly entangled states
- χ=64-256: Moderately entangled states
- χ≥512: Highly entangled states

Start with small χ and increase if accuracy is insufficient.

Part 2: Quantum Gates
----------------------

Single-Qubit Gates
^^^^^^^^^^^^^^^^^^

Apply gates to individual qubits:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   mps = AdaptiveMPS(num_qubits=5, bond_dim=8, device='cuda')

   # Define Hadamard gate: H = (1/√2) * [[1, 1], [1, -1]]
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
   H = H.to('cuda')

   # Apply Hadamard to qubit 0 (creates superposition)
   mps.apply_single_qubit_gate(0, H)

   print("Applied Hadamard gate to qubit 0")

Common single-qubit gates:

.. code-block:: python

   # Pauli X (NOT gate)
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64).to('cuda')

   # Pauli Y
   Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64).to('cuda')

   # Pauli Z
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64).to('cuda')

   # Phase gate S
   S = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64).to('cuda')

   # T gate
   T = torch.tensor([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=torch.complex64).to('cuda')

   # Apply gates
   mps.apply_single_qubit_gate(0, X)
   mps.apply_single_qubit_gate(1, S)
   mps.apply_single_qubit_gate(2, T)

Two-Qubit Gates
^^^^^^^^^^^^^^^

Apply gates between pairs of qubits:

.. code-block:: python

   # CNOT gate (controlled-NOT)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   # Apply CNOT between qubits 0 and 1
   mps.apply_two_site_gate(0, CNOT)

   # CZ gate (controlled-Z)
   CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64)).to('cuda')

   # Apply CZ between qubits 2 and 3
   mps.apply_two_site_gate(2, CZ)

   # SWAP gate
   SWAP = torch.tensor([
       [1, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   mps.apply_two_site_gate(1, SWAP)

Creating Bell States
^^^^^^^^^^^^^^^^^^^^

Let's create maximally entangled Bell states:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   # Create 2-qubit MPS
   mps = AdaptiveMPS(num_qubits=2, bond_dim=4, device='cuda')

   # Define gates
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
   mps.apply_single_qubit_gate(0, H)  # Create superposition on qubit 0
   mps.apply_two_site_gate(0, CNOT)   # Entangle qubits 0 and 1

   print("Created Bell state |Φ+⟩")

   # Check bond dimension (should be 2 for Bell state)
   stats = mps.stats_summary()
   print(f"Bond dimensions: {stats['max_chi']}")

Part 3: Measurement and Sampling
---------------------------------

Basic Measurement
^^^^^^^^^^^^^^^^^

Measure the quantum state:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   # Create GHZ state: (|000⟩ + |111⟩)/√2
   mps = AdaptiveMPS(num_qubits=3, bond_dim=4, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   mps.apply_single_qubit_gate(0, H)
   mps.apply_two_site_gate(0, CNOT)
   mps.apply_two_site_gate(1, CNOT)

   # Sample measurements
   samples = mps.sample(num_shots=1000)

   # Count outcomes
   from collections import Counter
   counts = Counter(samples)

   # Display results
   for state, count in sorted(counts.items()):
       binary = format(state, '03b')
       percentage = 100 * count / 1000
       print(f"|{binary}⟩: {count} ({percentage:.1f}%)")

Expected output shows ~50% |000⟩ and ~50% |111⟩, confirming the GHZ state.

Interpreting Results
^^^^^^^^^^^^^^^^^^^^

Measurement outcomes are integers representing basis states:

- 0 = |000⟩ (all qubits in state |0⟩)
- 7 = |111⟩ (all qubits in state |1⟩)
- 5 = |101⟩ (qubits 0 and 2 in |1⟩, qubit 1 in |0⟩)

Convert to binary for visualization:

.. code-block:: python

   outcome = 5
   n_qubits = 3
   binary = format(outcome, f'0{n_qubits}b')
   print(f"Outcome {outcome} = |{binary}⟩")

Part 4: Building Quantum Circuits
----------------------------------

Sequential Gate Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build complex circuits by applying gates in sequence:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   # Create 5-qubit circuit
   mps = AdaptiveMPS(num_qubits=5, bond_dim=16, device='cuda')

   # Define gates
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')
   CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64)).to('cuda')

   # Layer 1: Hadamards on all qubits
   for q in range(5):
       mps.apply_single_qubit_gate(q, H)

   # Layer 2: CNOT chain
   for q in range(4):
       mps.apply_two_site_gate(q, CNOT)

   # Layer 3: CZ gates on alternating pairs
   for q in range(0, 4, 2):
       mps.apply_two_site_gate(q, CZ)

   # Layer 4: Final Hadamards
   for q in range(5):
       mps.apply_single_qubit_gate(q, H)

   print("Circuit complete")

   # Check final statistics
   stats = mps.stats_summary()
   print(f"Max bond dimension: {stats['max_chi']}")
   print(f"Memory: {mps.memory_usage() / (1024**2):.2f} MB")

Monitoring Circuit Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Track how the MPS evolves during circuit execution:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   mps = AdaptiveMPS(num_qubits=10, bond_dim=32, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   # Apply circuit and monitor
   for q in range(10):
       mps.apply_single_qubit_gate(q, H)

   stats_after_hadamards = mps.stats_summary()

   for q in range(9):
       mps.apply_two_site_gate(q, CNOT)

   stats_after_cnots = mps.stats_summary()

   # Compare statistics
   print(f"After Hadamards: max χ = {stats_after_hadamards['max_chi']}")
   print(f"After CNOTs: max χ = {stats_after_cnots['max_chi']}")
   print(f"Global error bound: {mps.global_error_bound():.2e}")

Part 5: Period-Finding and Factorization
-----------------------------------------

Period-Finding Basics
^^^^^^^^^^^^^^^^^^^^^

ATLAS-Q includes specialized algorithms for period-finding (Shor's algorithm):

.. code-block:: python

   from atlas_q import get_quantum_sim

   # Get the quantum-classical hybrid simulator
   QCH, _, _, _ = get_quantum_sim()

   # Create simulator instance
   sim = QCH()

   # Factor a small number
   factors = sim.factor_number(15)

   if factors:
       p, q = factors
       print(f"15 = {p} × {q}")
       assert p * q == 15
   else:
       print("Factorization failed (retry may succeed)")

Factoring Larger Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q import get_quantum_sim

   QCH, _, _, _ = get_quantum_sim()
   sim = QCH()

   # Factor larger semiprimes
   numbers_to_factor = [21, 35, 77, 143, 221, 323]

   for N in numbers_to_factor:
       print(f"\nFactoring {N}...")
       factors = sim.factor_number(N)

       if factors:
           p, q = factors
           print(f"  {N} = {p} × {q}")
           assert p * q == N
       else:
           print(f"  Failed (may need retry)")

How Period-Finding Works
^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithm uses compressed quantum states that require only O(1) memory:

.. code-block:: python

   from atlas_q.quantum_hybrid_system import PeriodicState

   # Create periodic state with period=7
   state = PeriodicState(
       num_qubits=10,
       period=7,
       offset=0
   )

   # This state has period 7, meaning amplitudes repeat every 7 basis states
   # Memory usage: O(1) regardless of qubit count!

   print(f"Amplitude at |0⟩: {state.get_amplitude(0)}")
   print(f"Amplitude at |7⟩: {state.get_amplitude(7)}")
   print(f"Amplitude at |14⟩: {state.get_amplitude(14)}")

   # Sample to extract period
   samples = state.measure(num_shots=100)
   print(f"Sample outcomes: {samples[:10]}")

Part 6: Understanding Truncation
---------------------------------

What is Truncation?
^^^^^^^^^^^^^^^^^^^

When bond dimensions grow too large, ATLAS-Q truncates them to maintain efficiency. This introduces small errors.

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   # Create MPS with truncation parameters
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       eps_bond=1e-6,         # Truncation tolerance
       chi_max_per_bond=64,   # Maximum χ per bond
       device='cuda'
   )

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   # Apply circuit
   for q in range(20):
       mps.apply_single_qubit_gate(q, H)

   for q in range(19):
       mps.apply_two_site_gate(q, CNOT)

   # Check truncation error
   global_error = mps.global_error_bound()
   print(f"Global truncation error: {global_error:.2e}")

   stats = mps.stats_summary()
   print(f"Max bond dimension reached: {stats['max_chi']}")

Controlling Accuracy
^^^^^^^^^^^^^^^^^^^^

Adjust truncation parameters for accuracy vs speed tradeoff:

.. code-block:: python

   # High accuracy (slower, more memory)
   mps_accurate = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       eps_bond=1e-10,        # Tight tolerance
       chi_max_per_bond=256,  # Large max χ
       device='cuda'
   )

   # Fast (less accurate, less memory)
   mps_fast = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       eps_bond=1e-4,         # Loose tolerance
       chi_max_per_bond=32,   # Small max χ
       device='cuda'
   )

Part 7: Memory Management
--------------------------

Memory Budgets
^^^^^^^^^^^^^^

Limit total memory usage:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Enforce 4GB memory limit
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=16,
       budget_global_mb=4096,  # 4GB limit
       device='cuda'
   )

   # ATLAS-Q will automatically reduce bond dimensions to stay within budget

Checking Memory Usage
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   memory_bytes = mps.memory_usage()
   memory_mb = memory_bytes / (1024**2)
   memory_gb = memory_bytes / (1024**3)

   print(f"MPS memory usage:")
   print(f"  {memory_bytes:,} bytes")
   print(f"  {memory_mb:.2f} MB")
   print(f"  {memory_gb:.3f} GB")

   # Check against budget
   if hasattr(mps, 'budget_global_mb') and mps.budget_global_mb:
       utilization = 100 * memory_mb / mps.budget_global_mb
       print(f"  {utilization:.1f}% of budget")

Next Steps
----------

Congratulations! You have completed the beginner's tutorial. You now know:

- How to create and manipulate MPS
- Apply quantum gates and build circuits
- Measure quantum states
- Factor numbers using period-finding
- Control truncation and memory usage

Where to Go Next
^^^^^^^^^^^^^^^^

- :doc:`mps_basics` - Deeper dive into Matrix Product States
- :doc:`vqe_tutorial` - Variational algorithms for ground states
- :doc:`tdvp_tutorial` - Time evolution with TDVP
- :doc:`molecular_vqe` - Quantum chemistry applications
- :doc:`../howtos/index` - Task-specific guides
- :doc:`../explanations/index` - Conceptual explanations

Practice Exercises
^^^^^^^^^^^^^^^^^^

1. Create a 3-qubit W state: (|100⟩ + |010⟩ + |001⟩)/√3
2. Build a quantum Fourier transform circuit for 4 qubits
3. Factor the number 35 and verify the result
4. Create an entangled state and measure the bond dimension growth
5. Compare truncation errors for different ``eps_bond`` values

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

Out of Memory
"""""""""""""

Reduce bond dimensions or enable memory budgets:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=16,
       chi_max_per_bond=64,
       budget_global_mb=2048,
       device='cuda'
   )

Slow Performance
""""""""""""""""

- Ensure GPU is available (``torch.cuda.is_available()``)
- Reduce bond dimensions for faster simulation
- Install Triton for GPU acceleration: ``pip install triton``

Numerical Instability
"""""""""""""""""""""

Use higher precision:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       dtype=torch.complex128,  # Higher precision
       device='cuda'
   )

Getting Help
^^^^^^^^^^^^

- Documentation: https://followthesapper.github.io/ATLAS-Q/
- GitHub Issues: https://github.com/followthesapper/ATLAS-Q/issues
- GitHub Discussions: https://github.com/followthesapper/ATLAS-Q/discussions

See Also
--------

- :doc:`../../installation` - Installation instructions
- :doc:`../../quickstart` - Quick start guide
- :doc:`index` - All tutorials
- :doc:`../../reference/index` - API reference
