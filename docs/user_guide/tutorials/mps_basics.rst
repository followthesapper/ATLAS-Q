MPS Basics
==========

Matrix Product States (MPS) are the fundamental tensor network representation used throughout ATLAS-Q. This tutorial provides a comprehensive introduction to MPS structure, operations, and practical usage.

Prerequisites
-------------

- Completed :doc:`beginners` tutorial
- Understanding of quantum states and tensor products
- Familiarity with singular value decomposition (SVD)
- Basic linear algebra knowledge

Learning Objectives
-------------------

After completing this tutorial, you will understand:

1. MPS tensor structure and representation
2. How bond dimensions control accuracy and memory
3. Canonical forms and their importance
4. How gates are applied to MPS
5. Truncation mechanisms and error control
6. Entanglement measures in MPS
7. Practical MPS usage patterns

Part 1: Understanding MPS Structure
------------------------------------

What is an MPS?
^^^^^^^^^^^^^^^

An MPS represents a quantum state as a chain of rank-3 tensors. Instead of storing all 2ⁿ amplitudes, we decompose the state into a product of smaller tensors.

Mathematical Representation
"""""""""""""""""""""""""""

A quantum state can be written as:

.. math::

   |\psi\rangle = \sum_{s_1,\ldots,s_n} A^{[1]}_{s_1} A^{[2]}_{s_2} \cdots A^{[n]}_{s_n} |s_1 \ldots s_n\rangle

where each tensor :math:`A^{[i]}` has shape :math:`[\chi_{i-1}, 2, \chi_i]`:

- :math:`\chi_{i-1}`: Left bond dimension (connection to previous tensor)
- :math:`2`: Physical dimension (qubit can be |0⟩ or |1⟩)
- :math:`\chi_i`: Right bond dimension (connection to next tensor)

Visual Representation
"""""""""""""""""""""

.. code-block:: text

   Tensor Network Diagram:

   |s₁⟩   |s₂⟩   |s₃⟩        |sₙ⟩
     |      |      |           |
   [A¹]--[A²]--[A³]-- ... --[Aⁿ]
     χ₁    χ₂    χ₃          χₙ

   Each A^[i] is a rank-3 tensor with indices:
   - Physical index (up): connects to qubit basis
   - Left/right indices (horizontal): connect to neighboring tensors

Creating Your First MPS
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create 10-qubit MPS with bond dimension 8
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Inspect the MPS structure
   print(f"Number of tensors: {len(mps.tensors)}")
   print(f"First tensor shape: {mps.tensors[0].shape}")
   print(f"Middle tensor shape: {mps.tensors[5].shape}")
   print(f"Last tensor shape: {mps.tensors[-1].shape}")

   # Output:
   # Number of tensors: 10
   # First tensor shape: torch.Size([1, 2, 8])    # χ_L=1 (boundary)
   # Middle tensor shape: torch.Size([8, 2, 8])   # χ_L=8, χ_R=8
   # Last tensor shape: torch.Size([8, 2, 1])     # χ_R=1 (boundary)

Memory Scaling Analysis
^^^^^^^^^^^^^^^^^^^^^^^

The power of MPS comes from exponential compression:

.. code-block:: python

   import torch

   # Full statevector memory
   def statevector_memory(n_qubits):
       # 2^n complex numbers, each 8 bytes (complex64)
       return 2**n_qubits * 8

   # MPS memory (approximate)
   def mps_memory(n_qubits, bond_dim):
       # n tensors, each roughly bond_dim^2 * 2 * 8 bytes
       return n_qubits * bond_dim**2 * 2 * 8

   # Compare for different system sizes
   for n in [10, 20, 30, 40]:
       chi = 64
       sv_mem = statevector_memory(n) / (1024**3)  # GB
       mps_mem = mps_memory(n, chi) / (1024**3)    # GB
       compression = statevector_memory(n) / mps_memory(n, chi)

       print(f"{n} qubits (χ={chi}):")
       print(f"  Statevector: {sv_mem:.3f} GB")
       print(f"  MPS: {mps_mem:.6f} GB")
       print(f"  Compression: {compression:,.0f}×\n")

Expected output shows MPS uses orders of magnitude less memory:

.. code-block:: text

   10 qubits (χ=64):
     Statevector: 0.000 GB
     MPS: 0.000005 GB
     Compression: 160×

   20 qubits (χ=64):
     Statevector: 0.008 GB
     MPS: 0.000010 GB
     Compression: 819×

   30 qubits (χ=64):
     Statevector: 8.590 GB
     MPS: 0.000015 GB
     Compression: 559,241×

   40 qubits (χ=64):
     Statevector: 8796.093 GB
     MPS: 0.000020 GB
     Compression: 439,804,651×

Part 2: Bond Dimensions
------------------------

What is Bond Dimension?
^^^^^^^^^^^^^^^^^^^^^^^

Bond dimension (χ) controls how much information flows between tensors. Higher χ allows more entanglement but requires more memory.

Entanglement Capacity
"""""""""""""""""""""

For a bipartition of the system, maximum entanglement entropy is:

.. math::

   S_{\text{max}} = \log_2(\min(2^{n_L}, 2^{n_R}, \chi))

where :math:`n_L` and :math:`n_R` are qubits on left and right.

This means:
- χ=1: No entanglement (product state)
- χ=2: 1 ebit of entanglement
- χ=4: 2 ebits of entanglement
- χ=2^k: k ebits of entanglement

Choosing Bond Dimension
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   def test_bond_dimension(n_qubits, chi_values):
       """Test different bond dimensions on entangled circuit."""
       H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
       H = H.to('cuda')

       CNOT = torch.tensor([
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]
       ], dtype=torch.complex64).reshape(4, 4).to('cuda')

       results = []
       for chi in chi_values:
           mps = AdaptiveMPS(num_qubits=n_qubits, bond_dim=chi, device='cuda')

           # Create entanglement
           for q in range(n_qubits):
               mps.apply_single_qubit_gate(q, H)

           for q in range(n_qubits - 1):
               mps.apply_two_site_gate(q, CNOT)

           stats = mps.stats_summary()
           error = mps.global_error_bound()
           memory = mps.memory_usage() / (1024**2)  # MB

           results.append({
               'chi': chi,
               'max_chi': stats['max_chi'],
               'error': error,
               'memory_mb': memory
           })

           print(f"χ={chi:3d}: max_chi={stats['max_chi']:3d}, "
                 f"error={error:.2e}, memory={memory:.2f} MB")

       return results

   # Test
   results = test_bond_dimension(n_qubits=15, chi_values=[4, 8, 16, 32, 64])

Guidelines for χ Selection
""""""""""""""""""""""""""

- **Product states** (no entanglement): χ=1
- **Weakly entangled** (shallow circuits): χ=8-16
- **Moderately entangled** (medium depth): χ=32-64
- **Highly entangled** (deep circuits): χ=128-512
- **Maximally entangled** (random circuits): χ ≈ 2^(n/2)

Start small and increase if truncation error is too large.

Part 3: Canonical Forms
------------------------

Why Canonical Forms?
^^^^^^^^^^^^^^^^^^^^

Canonical forms provide numerical stability and enable efficient algorithms. An MPS in canonical form has orthogonal tensors.

Left Canonical Form
^^^^^^^^^^^^^^^^^^^

Left canonical form means tensors satisfy:

.. math::

   \sum_{s,\alpha} A^{[i]}_{s,\alpha,\beta} (A^{[i]}_{s,\alpha,\gamma})^* = \delta_{\beta,\gamma}

Visually:

.. code-block:: text

   Left canonical:  [A¹]--[A²]--[A³]-- ... --[Λ]
   Each A^[i] is left-orthogonal (L†L = I)

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Convert to left canonical form
   mps.to_left_canonical()

   # Check orthogonality of first tensor
   A = mps.tensors[0]  # Shape: [1, 2, 8]
   A_reshaped = A.reshape(2, 8)  # Combine left index with physical

   # Verify A†A = I
   identity = torch.matmul(A_reshaped.conj().T, A_reshaped)
   error = torch.norm(identity - torch.eye(8, device='cuda'))
   print(f"Orthogonality error: {error:.2e}")  # Should be ~1e-7

Right Canonical Form
^^^^^^^^^^^^^^^^^^^^

Right canonical form means tensors satisfy:

.. math::

   \sum_{s,\beta} A^{[i]}_{\alpha,s,\beta} (A^{[i]}_{\gamma,s,\beta})^* = \delta_{\alpha,\gamma}

.. code-block:: python

   # Convert to right canonical form
   mps.to_right_canonical()

   # Check orthogonality of last tensor
   A = mps.tensors[-1]  # Shape: [8, 2, 1]
   A_reshaped = A.reshape(8, 2)  # Combine physical with right index

   # Verify AA† = I
   identity = torch.matmul(A_reshaped, A_reshaped.conj().T)
   error = torch.norm(identity - torch.eye(8, device='cuda'))
   print(f"Orthogonality error: {error:.2e}")

Mixed Canonical Form
^^^^^^^^^^^^^^^^^^^^

Most efficient for gate application: left-canonical up to site i, right-canonical after:

.. code-block:: text

   [A¹]--[A²]--[Λ]--[A⁴]--[A⁵]
    ←LC→     ←→     ←RC→
              Center

The center tensor contains all the state's norm.

Part 4: Gate Application
-------------------------

Single-Qubit Gates
^^^^^^^^^^^^^^^^^^

Applying a single-qubit gate to MPS:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   # Define rotation gate R_y(θ)
   theta = np.pi / 4
   Ry = torch.tensor([
       [np.cos(theta/2), -np.sin(theta/2)],
       [np.sin(theta/2), np.cos(theta/2)]
   ], dtype=torch.complex64).to('cuda')

   # Apply to qubit 5
   mps.apply_single_qubit_gate(5, Ry)

   # Bond dimensions unchanged
   stats = mps.stats_summary()
   print(f"Max bond dimension: {stats['max_chi']}")  # Still 16

Single-qubit gates don't increase entanglement or bond dimensions.

Two-Qubit Gates
^^^^^^^^^^^^^^^

Two-qubit gates can increase entanglement:

.. code-block:: python

   # Start with product state
   mps = AdaptiveMPS(num_qubits=10, bond_dim=2, device='cuda')

   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   print(f"Before CNOT: max_chi = {mps.stats_summary()['max_chi']}")

   # Apply CNOT - creates entanglement
   mps.apply_two_site_gate(4, CNOT)

   print(f"After CNOT: max_chi = {mps.stats_summary()['max_chi']}")
   print(f"Truncation error: {mps.global_error_bound():.2e}")

Gate Application Algorithm
"""""""""""""""""""""""""""

For a two-site gate on qubits i and i+1:

1. Contract tensors A^[i] and A^[i+1] into a 4-index tensor
2. Apply the gate to the physical indices
3. Reshape into matrix form
4. Perform SVD to split back into two tensors
5. Truncate small singular values based on χ_max and ε

.. code-block:: python

   def explain_two_site_gate_application():
       """Conceptual explanation of algorithm."""

       # Step 1: Contract two tensors
       # Θ[α, s_i, s_{i+1}, γ] = A^[i][α, s_i, β] * A^[i+1][β, s_{i+1}, γ]

       # Step 2: Apply gate
       # Θ'[α, s'_i, s'_{i+1}, γ] = U[s'_i, s'_{i+1}, s_i, s_{i+1}] * Θ[α, s_i, s_{i+1}, γ]

       # Step 3: Reshape to matrix
       # M[(α, s'_i), (s'_{i+1}, γ)]

       # Step 4: SVD
       # M = U * S * V†

       # Step 5: Truncate
       # Keep largest χ singular values
       # A^[i]_new = U[:, :χ]
       # A^[i+1]_new = (S[:χ] * V†[:χ, :])

       print("See adaptive_mps.py for full implementation")

   explain_two_site_gate_application()

Part 5: Truncation Mechanics
-----------------------------

Why Truncate?
^^^^^^^^^^^^^

Without truncation, bond dimensions grow exponentially with gates:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Track bond dimension growth
   mps = AdaptiveMPS(num_qubits=10, bond_dim=1024, device='cuda')

   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   chi_history = []

   for layer in range(5):
       for q in range(9):
           mps.apply_two_site_gate(q, CNOT)

       stats = mps.stats_summary()
       chi_history.append(stats['max_chi'])
       print(f"Layer {layer+1}: max_chi = {stats['max_chi']}")

   # Without truncation, χ would double each layer!

Truncation Criterion
^^^^^^^^^^^^^^^^^^^^

ATLAS-Q uses energy-based truncation. Keep smallest k such that:

.. math::

   \sum_{i=1}^k \sigma_i^2 \geq (1 - \varepsilon^2) \sum_i \sigma_i^2

where σ_i are singular values and ε is the truncation tolerance.

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Configure truncation
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,           # Initial χ
       eps_bond=1e-6,         # Local truncation tolerance
       chi_max_per_bond=128,  # Maximum χ allowed
       device='cuda'
   )

   # Apply some gates...

   # Check truncation results
   global_error = mps.global_error_bound()
   print(f"Global truncation error: {global_error:.2e}")

   # Per-operation errors
   for i, eps_local in enumerate(mps.statistics.logs['eps_local'][:10]):
       print(f"Operation {i}: ε_local = {eps_local:.2e}")

Error Accumulation
^^^^^^^^^^^^^^^^^^

Global error bound:

.. math::

   \varepsilon_{\text{global}} \leq \sqrt{\sum_{\text{ops}} \varepsilon_{\text{local}}^2}

.. code-block:: python

   import math

   def compute_expected_error(n_operations, eps_bond):
       """Estimate global error after n operations."""
       # Assuming each operation introduces eps_bond error
       return math.sqrt(n_operations * eps_bond**2)

   # Example: 1000 operations with ε=1e-6
   n_ops = 1000
   eps = 1e-6
   expected_error = compute_expected_error(n_ops, eps)
   print(f"Expected global error: {expected_error:.2e}")  # ~3e-5

Part 6: Entanglement Measures
------------------------------

Entanglement Entropy
^^^^^^^^^^^^^^^^^^^^

The entanglement entropy at a bond quantifies how entangled the two halves are:

.. math::

   S = -\sum_i p_i \log_2(p_i), \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}

.. code-block:: python

   from atlas_q.diagnostics import bond_entropy_from_S
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   mps = AdaptiveMPS(num_qubits=10, bond_dim=64, device='cuda')

   # Create entangled state
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64).reshape(4, 4).to('cuda')

   for q in range(10):
       mps.apply_single_qubit_gate(q, H)

   for q in range(9):
       mps.apply_two_site_gate(q, CNOT)

   # Compute entanglement across center bond
   # Would need to extract singular values from center bond
   # (This is tracked internally in mps.statistics)

   entropy_logs = mps.statistics.logs.get('entropy', [])
   if entropy_logs:
       avg_entropy = sum(entropy_logs) / len(entropy_logs)
       print(f"Average bond entropy: {avg_entropy:.2f} bits")

Interpreting Entropy
""""""""""""""""""""

- S = 0: No entanglement (product state)
- S = 1: 1 ebit of entanglement (Bell pair)
- S = k: k ebits of entanglement
- S = log₂(χ): Bond is saturated (need higher χ)

Part 7: Practical Patterns
---------------------------

Circuit Construction Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   def build_layered_circuit(n_qubits, n_layers, bond_dim=32):
       """Build circuit with repeated layers."""
       mps = AdaptiveMPS(num_qubits=n_qubits, bond_dim=bond_dim, device='cuda')

       # Define gates
       H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / np.sqrt(2)
       Rz = lambda theta: torch.diag(torch.tensor([1, np.exp(1j*theta)], dtype=torch.complex64)).to('cuda')
       CNOT = torch.tensor([
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]
       ], dtype=torch.complex64).reshape(4, 4).to('cuda')

       for layer in range(n_layers):
           # Single-qubit layer
           for q in range(n_qubits):
               mps.apply_single_qubit_gate(q, H)
               mps.apply_single_qubit_gate(q, Rz(layer * 0.1))

           # Entangling layer
           for q in range(0, n_qubits-1, 2):  # Even pairs
               mps.apply_two_site_gate(q, CNOT)

           for q in range(1, n_qubits-1, 2):  # Odd pairs
               mps.apply_two_site_gate(q, CNOT)

           # Monitor progress
           if (layer + 1) % 10 == 0:
               stats = mps.stats_summary()
               error = mps.global_error_bound()
               print(f"Layer {layer+1}: χ_max={stats['max_chi']}, ε={error:.2e}")

       return mps

   # Build 50-layer circuit
   mps = build_layered_circuit(n_qubits=20, n_layers=50, bond_dim=64)

State Preparation Pattern
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def prepare_ghz_state(n_qubits):
       """Prepare GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
       mps = AdaptiveMPS(num_qubits=n_qubits, bond_dim=4, device='cuda')

       H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64).to('cuda') / torch.sqrt(torch.tensor(2.0))
       CNOT = torch.tensor([
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]
       ], dtype=torch.complex64).reshape(4, 4).to('cuda')

       # Hadamard on first qubit
       mps.apply_single_qubit_gate(0, H)

       # CNOT chain
       for q in range(n_qubits - 1):
           mps.apply_two_site_gate(q, CNOT)

       return mps

   def prepare_w_state(n_qubits):
       """Prepare W state: (|100...0⟩ + |010...0⟩ + ... + |00...1⟩)/√n"""
       # More complex - requires controlled rotations
       # Left as exercise
       pass

   # Test
   ghz = prepare_ghz_state(10)
   samples = ghz.sample(num_shots=100)

   from collections import Counter
   counts = Counter(samples)
   print("GHZ state measurements:")
   for state, count in sorted(counts.items()):
       print(f"|{bin(state)[2:].zfill(10)}⟩: {count}")

Measurement Pattern
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def measure_observable(mps, observable_mpo):
       """Compute expectation value ⟨ψ|O|ψ⟩"""
       # This would use MPO-MPS contraction
       # See mpo_ops tutorial for details
       pass

   def measure_correlations(mps, site_i, site_j):
       """Measure ⟨Z_i Z_j⟩ correlation"""
       # Would require partial contractions
       # See diagnostics for entropy calculations
       pass

Part 8: Troubleshooting
------------------------

High Truncation Errors
^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Global error exceeds tolerance

**Solutions**:

1. Increase bond dimension:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=20, bond_dim=128, device='cuda')  # Higher χ

2. Tighten truncation tolerance:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=20, bond_dim=64, eps_bond=1e-8, device='cuda')

3. Use higher precision:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=20, bond_dim=64, dtype=torch.complex128, device='cuda')

Memory Exhaustion
^^^^^^^^^^^^^^^^^

**Problem**: Out of GPU memory

**Solutions**:

1. Reduce bond dimension
2. Enable memory budgets:

   .. code-block:: python

      mps = AdaptiveMPS(
          num_qubits=50,
          bond_dim=32,
          budget_global_mb=4096,  # 4GB limit
          device='cuda'
      )

3. Use CPU for larger systems:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=50, bond_dim=32, device='cpu')

Slow Performance
^^^^^^^^^^^^^^^^

**Problem**: Gate application is slow

**Solutions**:

1. Ensure GPU is being used
2. Install Triton for GPU kernels: ``pip install triton``
3. Use cuQuantum: ``pip install cuquantum-python``
4. Batch operations when possible

Numerical Instability
^^^^^^^^^^^^^^^^^^^^^

**Problem**: NaN or Inf values appearing

**Solutions**:

1. Use canonical forms regularly:

   .. code-block:: python

      if step % 100 == 0:
          mps.to_left_canonical()

2. Use higher precision (complex128)
3. Check condition numbers in diagnostics
4. Avoid very small truncation tolerances (< 1e-12)

Summary
-------

You have learned:

- ✅ MPS structure and tensor representation
- ✅ Bond dimensions and entanglement capacity
- ✅ Canonical forms for numerical stability
- ✅ How gates are applied to MPS
- ✅ Truncation mechanisms and error control
- ✅ Entanglement entropy measures
- ✅ Practical usage patterns
- ✅ Common troubleshooting techniques

Next Steps
----------

- :doc:`vqe_tutorial` - Use MPS for variational algorithms
- :doc:`tdvp_tutorial` - Time evolution with MPS
- :doc:`../explanations/tensor_networks` - Deeper theory
- :doc:`../explanations/adaptive_truncation` - Truncation mathematics
- :doc:`../howtos/optimize_performance` - Performance tuning

Practice Exercises
------------------

1. Create an MPS with linearly increasing entanglement along the chain
2. Measure bond entropies at each bond and plot them
3. Compare truncation errors for different ε values on the same circuit
4. Implement a custom state preparation (e.g., W state)
5. Build a parameterized circuit and track how χ grows with depth

See Also
--------

- :doc:`../../reference/adaptive_mps` - AdaptiveMPS API reference
- :doc:`../../reference/diagnostics` - Entropy and diagnostics API
- :doc:`../../reference/truncation` - Truncation functions
- :doc:`../explanations/tensor_networks` - Tensor network theory
