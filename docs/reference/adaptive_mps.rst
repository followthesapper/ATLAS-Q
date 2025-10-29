atlas_q.adaptive_mps
====================

.. automodule:: atlas_q.adaptive_mps
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Overview
--------

The ``adaptive_mps`` module provides **adaptive Matrix Product State simulation** with intelligent resource management. Unlike fixed bond dimension approaches, adaptive MPS dynamically adjusts χ based on entanglement structure, memory constraints, and accuracy requirements.

Key Features
~~~~~~~~~~~~

- **Per-bond adaptation**: Individual χ limits for each bond
- **Global memory budgets**: Automatic χ reduction to fit memory constraints
- **Mixed precision**: Automatic promotion to float64 when condition numbers are high
- **Error tracking**: Comprehensive statistics on truncation errors and resource usage
- **GPU-optimized**: Native CUDA support with efficient memory management
- **Canonicalization**: Left, right, and mixed canonical forms for numerical stability

Why Adaptive MPS?
~~~~~~~~~~~~~~~~~

**Fixed bond dimension** wastes resources:

- High entanglement regions need large χ
- Low entanglement regions can use small χ
- Fixed χ either wastes memory or loses accuracy

**Adaptive bond dimension** optimizes dynamically:

.. math::

   \chi_i = \min\left(\chi_{\text{max},i}, \,\left\lceil \frac{\text{Budget}_{\text{global}}}{n \cdot d^2} \right\rceil, \,\chi_{\text{SVD}}(\epsilon)\right)

where:
  - :math:`\chi_{\text{max},i}` is the per-bond maximum
  - Budget is the global memory limit
  - :math:`\chi_{\text{SVD}}(\epsilon)` is determined by truncation threshold ε

Mathematical Background
-----------------------

Matrix Product State Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An n-qubit quantum state is represented as:

.. math::

   |\psi\rangle = \sum_{s_1,\ldots,s_n} A^{[1]}_{s_1} A^{[2]}_{s_2} \cdots A^{[n]}_{s_n} |s_1 s_2 \ldots s_n\rangle

where each tensor :math:`A^{[i]}_{s_i}` has shape :math:`[\chi_{i-1}, d, \chi_i]` with:
  - :math:`d = 2` for qubits
  - :math:`\chi_i` is the bond dimension between sites i and i+1

**Memory scaling**: :math:`O(n \chi^2 d)` vs. :math:`O(d^n)` for full statevector.

Adaptive Truncation
~~~~~~~~~~~~~~~~~~~

After applying a two-qubit gate, the bond dimension can grow. SVD is used to truncate:

.. math::

   M = U \Sigma V^\dagger \approx U_k \Sigma_k V_k^\dagger

where k singular values are kept such that:

.. math::

   \sum_{i=1}^k \sigma_i^2 \geq (1 - \epsilon^2) \sum_{i=1}^r \sigma_i^2

**Adaptive strategy**: Choose k differently for each bond based on:

1. **Error threshold** ε: User-specified truncation tolerance
2. **Per-bond cap** :math:`\chi_{\text{max},i}`: Maximum allowed at bond i
3. **Memory budget**: Global constraint across all bonds

Canonical Forms
~~~~~~~~~~~~~~~

MPS can be brought into canonical forms for numerical stability:

**Left-canonical**: :math:`\sum_{s,\alpha} |A^{[i]}_{s,\alpha,\beta}|^2 = \delta_{\alpha,\beta}`

**Right-canonical**: :math:`\sum_{s,\beta} |A^{[i]}_{s,\alpha,\beta}|^2 = \delta_{\alpha,\beta}`

**Mixed-canonical** (centered at site c):
  - Sites 1 to c-1: left-canonical
  - Site c: general tensor
  - Sites c+1 to n: right-canonical

**Advantage**: Simplifies expectation value calculations and improves numerical stability.

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AdaptiveMPS
   DTypePolicy

AdaptiveMPS
-----------

.. autoclass:: AdaptiveMPS
   :members:
   :undoc-members:
   :show-inheritance:

   Main class for adaptive Matrix Product State simulation.

   Provides dynamic bond dimension management, error tracking, and efficient gate application. Supports both single-site and two-site updates with automatic truncation.

   **Constructor**:

   .. code-block:: python

      mps = AdaptiveMPS(
          num_qubits=20,
          bond_dim=32,                    # Initial χ
          chi_max_per_bond=256,           # Per-bond maximum
          budget_global_mb=4096,          # 4GB memory budget
          eps_bond=1e-8,                  # Truncation threshold
          dtype_policy=None,              # Mixed precision policy
          device='cuda'
      )

   **Parameters**:

   - ``num_qubits`` (int): Number of qubits in the system
   - ``bond_dim`` (int): Initial bond dimension χ (default: 16)
   - ``chi_max_per_bond`` (int or list): Maximum χ per bond. If int, applied uniformly. If list, per-bond limits.
   - ``budget_global_mb`` (float): Global memory budget in megabytes
   - ``eps_bond`` (float): Truncation tolerance for SVD (default: 1e-8)
   - ``dtype_policy`` (DTypePolicy): Mixed precision configuration
   - ``device`` (str): 'cpu' or 'cuda'

   **Storage**: Approximately :math:`16 n \chi^2` bytes for complex64

   .. rubric:: Methods

   .. autosummary::

      ~AdaptiveMPS.__init__
      ~AdaptiveMPS.apply_single_qubit_gate
      ~AdaptiveMPS.apply_two_site_gate
      ~AdaptiveMPS.expectation_value
      ~AdaptiveMPS.inner_product
      ~AdaptiveMPS.stats_summary
      ~AdaptiveMPS.global_error_bound
      ~AdaptiveMPS.reset_stats
      ~AdaptiveMPS.memory_usage
      ~AdaptiveMPS.to_left_canonical
      ~AdaptiveMPS.to_right_canonical
      ~AdaptiveMPS.to_statevector
      ~AdaptiveMPS.sample
      ~AdaptiveMPS.measure
      ~AdaptiveMPS.normalize

Key Methods
~~~~~~~~~~~

apply_single_qubit_gate
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   mps.apply_single_qubit_gate(qubit, gate_matrix)

Apply a single-qubit gate without changing bond dimensions.

**Parameters**:
   - ``qubit`` (int): Target qubit index
   - ``gate_matrix`` (torch.Tensor): 2×2 unitary matrix

**Complexity**: O(χ²) where χ is bond dimension at qubit

**Example**:

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   # Hadamard gate
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)
   mps.apply_single_qubit_gate(0, H)

   # Pauli-X gate
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
   mps.apply_single_qubit_gate(5, X)

apply_two_site_gate
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   mps.apply_two_site_gate(site, gate_matrix)

Apply a two-qubit gate with adaptive truncation.

**Parameters**:
   - ``site`` (int): Index of first qubit (gate acts on site and site+1)
   - ``gate_matrix`` (torch.Tensor): 4×4 unitary matrix

**Complexity**: O(χ³) for SVD truncation

**Example**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, chi_max_per_bond=128, device='cuda')

   # CNOT gate
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64, device='cuda')

   for i in range(19):
       mps.apply_two_site_gate(i, CNOT)

   print(f"Max bond dimension: {max(mps.bond_dimensions)}")
   print(f"Memory usage: {mps.memory_usage() / 1024**2:.2f} MB")

expectation_value
^^^^^^^^^^^^^^^^^

.. code-block:: python

   energy = mps.expectation_value(operator_mpo)

Compute expectation value :math:`\langle\psi|\hat{O}|\psi\rangle`.

**Parameters**:
   - ``operator_mpo`` (MPO): Operator represented as Matrix Product Operator

**Returns**:
   - ``complex``: Expectation value

**Complexity**: O(n χ² D²) where D is MPO bond dimension

stats_summary
^^^^^^^^^^^^^

.. code-block:: python

   stats = mps.stats_summary()

Get comprehensive statistics on MPS state.

**Returns**:
   Dictionary with:
   - ``max_chi``: Maximum bond dimension
   - ``avg_chi``: Average bond dimension
   - ``total_params``: Total number of MPS parameters
   - ``memory_mb``: Memory usage in megabytes
   - ``num_truncations``: Number of truncation operations performed
   - ``max_local_error``: Maximum local truncation error
   - ``sum_squared_errors``: Sum of squared truncation errors

**Example**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # ... apply gates ...

   stats = mps.stats_summary()
   print(f"χ_max = {stats['max_chi']}, χ_avg = {stats['avg_chi']:.1f}")
   print(f"Memory: {stats['memory_mb']:.2f} MB")
   print(f"Truncations: {stats['num_truncations']}")
   print(f"Max error: {stats['max_local_error']:.2e}")

global_error_bound
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   error = mps.global_error_bound()

Compute rigorous upper bound on accumulated truncation error.

**Returns**:
   - ``float``: Error bound δ such that :math:`\||\psi_{\text{true}}\rangle - |\psi_{\text{MPS}}\rangle\| \leq \delta`

**Formula**:

.. math::

   \delta \leq \sqrt{\sum_{i=1}^{N_{\text{trunc}}} \epsilon_i^2}

where :math:`\epsilon_i` are local truncation errors.

to_left_canonical / to_right_canonical
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   mps.to_left_canonical()
   mps.to_right_canonical()

Convert MPS to canonical form for numerical stability.

**Use cases**:
   - Improve condition numbers before SVD operations
   - Simplify expectation value calculations
   - Prepare for TDVP time evolution

**Complexity**: O(n χ³)

DTypePolicy
-----------

.. autoclass:: DTypePolicy
   :members:
   :undoc-members:

   Configuration for mixed-precision simulation with automatic promotion.

   **Constructor**:

   .. code-block:: python

      from atlas_q.adaptive_mps import DTypePolicy
      import torch

      policy = DTypePolicy(
          default=torch.complex64,
          promote_if_cond_gt=1e6
      )

   **Parameters**:

   - ``default`` (torch.dtype): Default data type (torch.complex64 or torch.complex128)
   - ``promote_if_cond_gt`` (float): Threshold for automatic promotion. If condition number exceeds this value, promote to higher precision.

   **Strategy**:

   1. Use ``default`` dtype (e.g., complex64) for most operations (2× memory savings)
   2. Monitor condition numbers during SVD
   3. If cond(M) > ``promote_if_cond_gt``, promote to complex128 for that operation
   4. Convert result back to default dtype

   **Example**:

   .. code-block:: python

      from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy
      import torch

      # Use complex64 by default, promote if condition number > 10^6
      policy = DTypePolicy(default=torch.complex64, promote_if_cond_gt=1e6)

      mps = AdaptiveMPS(
          num_qubits=30,
          bond_dim=64,
          dtype_policy=policy,
          device='cuda'
      )

      # MPS will automatically:
      # - Use complex64 for well-conditioned operations (2× faster, 2× less memory)
      # - Promote to complex128 when ill-conditioned (maintains accuracy)

   **Performance**:

   +------------------------+----------------+----------------+----------------+
   | Metric                 | complex64      | complex128     | Adaptive       |
   +========================+================+================+================+
   | Memory                 | 1.0×           | 2.0×           | ~1.1× (best)   |
   +------------------------+----------------+----------------+----------------+
   | Speed                  | 1.0× (fastest) | 0.5×           | ~0.9×          |
   +------------------------+----------------+----------------+----------------+
   | Accuracy (well-cond.)  | Good           | Excellent      | Good           |
   +------------------------+----------------+----------------+----------------+
   | Accuracy (ill-cond.)   | Poor           | Excellent      | Excellent      |
   +------------------------+----------------+----------------+----------------+

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+----------------------+----------------------+
| Operation               | Fixed χ MPS          | Adaptive MPS         |
+=========================+======================+======================+
| Single-qubit gate       | O(χ²)                | O(χ²)                |
+-------------------------+----------------------+----------------------+
| Two-qubit gate          | O(χ³)                | O(χ³) + O(χ) check   |
+-------------------------+----------------------+----------------------+
| Canonicalization        | O(n χ³)              | O(n χ³)              |
+-------------------------+----------------------+----------------------+
| Expectation value       | O(n χ² D²)           | O(n χ² D²)           |
+-------------------------+----------------------+----------------------+
| Memory                  | O(n χ²)              | O(n χ_avg²) (better) |
+-------------------------+----------------------+----------------------+

where D is MPO bond dimension.

Memory Savings
~~~~~~~~~~~~~~

Adaptive MPS reduces memory by using smaller χ where possible:

.. code-block:: python

   # Example: 30-qubit system with varying entanglement
   Fixed χ=128: 30 × 128² × 8 bytes = 39.3 MB
   Adaptive χ: χ ∈ [16, 32, 64, 128] → avg 50 → 6.0 MB

   Memory savings: 6.5× with minimal accuracy loss

Benchmark Results
~~~~~~~~~~~~~~~~~

From ``scripts/benchmarks/validate_all_features.py``:

.. code-block:: python

   # 50-qubit quantum chemistry VQE
   Fixed χ=128:  Memory=156 MB, Time=2.5 sec, Error=1e-7
   Adaptive:     Memory= 45 MB, Time=2.2 sec, Error=1e-7
   Savings: 3.5× memory, 12% faster (less data movement)

   # 100-qubit random circuit (depth=50)
   Fixed χ=64:   Memory= 31 MB, Time=5.0 sec, χ insufficient → Error=1e-3
   Adaptive:     Memory= 48 MB, Time=5.8 sec, Error=1e-7
   Result: Higher accuracy with 55% more memory (still practical)

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create MPS
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Apply gates
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)
   for q in range(10):
       mps.apply_single_qubit_gate(q, H)

   # Check statistics
   stats = mps.stats_summary()
   print(f"Max χ: {stats['max_chi']}")
   print(f"Global error: {mps.global_error_bound():.2e}")

Adaptive Truncation with Memory Budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Limit memory to 2 GB
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=32,                    # Initial χ
       chi_max_per_bond=256,           # Per-bond max
       budget_global_mb=2048,          # 2 GB limit
       eps_bond=1e-8,
       device='cuda'
   )

   # Apply gates - χ will adapt automatically
   CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                       dtype=torch.complex64, device='cuda')

   for i in range(49):
       mps.apply_two_site_gate(i, CNOT)

   print(f"Bond dimensions: {mps.bond_dimensions}")
   print(f"Memory usage: {mps.memory_usage() / 1024**2:.2f} MB")
   print(f"Within budget: {mps.memory_usage() / 1024**2 <= 2048}")

Mixed Precision for Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy
   import torch

   # Mixed precision with automatic promotion
   policy = DTypePolicy(default=torch.complex64, promote_if_cond_gt=1e6)

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       dtype_policy=policy,
       device='cuda'
   )

   # Apply ill-conditioned gates
   # MPS will automatically promote to complex128 when needed
   for i in range(29):
       # Some gates create ill-conditioned matrices
       U = random_unitary(4, device='cuda')
       mps.apply_two_site_gate(i, U)

   # Check how many promotions occurred
   # (This info would be in debug logs)

Per-Bond Adaptation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Different χ limits for different regions
   # High entanglement in center, low at edges
   chi_per_bond = [32] * 10 + [128] * 20 + [32] * 9  # 40 qubits

   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=16,                    # Initial
       chi_max_per_bond=chi_per_bond,  # Per-bond limits
       eps_bond=1e-8,
       device='cuda'
   )

   # Center bonds can grow to 128, edges limited to 32
   # Automatically optimizes memory usage based on entanglement structure

Measurement and Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

   # ... prepare state ...

   # Single qubit measurement
   outcome, probability = mps.measure(qubit=5, collapse=True)
   print(f"Measured {outcome} with P={probability:.4f}")

   # Sample multiple times
   outcomes = [mps.sample() for _ in range(1000)]
   from collections import Counter
   histogram = Counter(outcomes)
   print(f"Histogram: {histogram}")

Canonicalization for Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # After many operations, numerical errors accumulate
   # ... 1000s of gate applications ...

   # Restore numerical stability
   mps.to_left_canonical()
   mps.normalize()

   # Check norm
   norm = torch.sqrt(mps.inner_product(mps).real)
   print(f"Norm: {norm:.10f}")  # Should be ~1.0

Use Cases
---------

When to Use Adaptive MPS
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Unknown entanglement structure**: Don't know a priori which regions need high χ
2. **Memory constraints**: Limited GPU memory but need to simulate as many qubits as possible
3. **Variable entanglement**: System has regions of high and low entanglement
4. **Long simulations**: Numerical stability important over many operations
5. **Production systems**: Need guaranteed memory bounds

When to Use Fixed χ
~~~~~~~~~~~~~~~~~~~

1. **Predictable entanglement**: Know exact χ requirements beforehand
2. **Performance critical**: Need absolute maximum speed (no adaptive overhead)
3. **Small systems**: χ requirements low enough that adaptation unnecessary
4. **Benchmarking**: Comparing against fixed-χ results in literature

Cross-References
----------------

See Also
~~~~~~~~

- :doc:`../user_guide/explanations/adaptive_truncation` - Theory of adaptive truncation
- :doc:`../user_guide/howtos/handle_large_systems` - Practical tips for large-scale simulation
- :doc:`mps_pytorch` - Basic MPS implementation
- :doc:`diagnostics` - Statistics and monitoring
- :doc:`linalg_robust` - Robust linear algebra with automatic fallbacks
- :doc:`truncation` - Truncation strategies and error bounds

References
~~~~~~~~~~

Key papers:

1. **Schollwöck, U.** (2011). "The density-matrix renormalization group in the age of matrix product states." Annals of Physics, 326(1), 96-192.
2. **Paeckel, S. et al.** (2019). "Time-evolution methods for matrix-product states." Annals of Physics, 411, 167998.
3. **Hubig, C. et al.** (2015). "Strictly single-site DMRG algorithm with subspace expansion." Physical Review B, 91(15), 155115.
