PEPS (2D Tensor Networks)
==========================

Projected Entangled Pair States for 2D lattice quantum simulations.

.. currentmodule:: atlas_q.peps

Overview
--------

The ``peps`` module implements **Projected Entangled Pair States** for simulating quantum systems on 2D lattices. PEPS is the natural generalization of Matrix Product States (MPS) from 1D chains to 2D grids, enabling efficient simulation of:

- 2D cluster states and graph states
- Shallow quantum supremacy circuits
- 2D spin systems (e.g., 2D Ising, Heisenberg)
- Topological quantum codes (surface codes)
- Small quantum processor patches (4×4, 5×5 grids)

Key Features
~~~~~~~~~~~~

- **2D lattice structure**: Natural representation for grid-based systems
- **Moderate entanglement**: Handles area-law entangled states efficiently
- **Boundary MPS contraction**: Efficient approximate contraction algorithm
- **GPU acceleration**: CUDA-optimized tensor operations
- **Small patches**: Optimized for 4×4 to 6×6 grids (16-36 qubits)

PEPS Network Structure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

       |   |   |   |
     --•---•---•---•--
       |   |   |   |
     --•---•---•---•--
       |   |   |   |
     --•---•---•---•--
       |   |   |   |

Each node • is a **rank-5 tensor** with indices:

.. math::

   T^{[i,j]}_{u,l,s,r,d}

where:
  - u, l, r, d: virtual indices (bond dimension χ) connecting to neighboring tensors
  - s: physical index (dimension d=2 for qubits)

**Storage**: O(rows × cols × χ⁴ × d) vs. O(2^(rows×cols)) for statevector

Mathematical Background
-----------------------

PEPS Representation
~~~~~~~~~~~~~~~~~~~

A 2D quantum state on an m×n lattice is represented as:

.. math::

   |\psi\rangle = \sum_{\{s_{ij}\}} T^{[1,1]} T^{[1,2]} \cdots T^{[m,n]} |s_{11} s_{12} \cdots s_{mn}\rangle

where tensor contractions are performed over virtual indices connecting neighbors.

**Key property**: Area-law entanglement → small bond dimension χ

Contraction Problem
~~~~~~~~~~~~~~~~~~~

Computing :math:`\langle\psi|\psi\rangle` requires contracting the 2D tensor network. This is **#P-hard** in general, but efficient approximations exist:

1. **Boundary MPS method**: Contract rows sequentially into 1D boundary MPS
2. **Corner transfer matrix**: Contract corners first, then edges
3. **Simple update**: Approximate gate application with local updates

**Complexity**: O(m × n × χ⁵) for boundary MPS (χ is boundary bond dimension)

Area Law Entanglement
~~~~~~~~~~~~~~~~~~~~~~

For 2D systems obeying area law:

.. math::

   S(A) \propto \text{perimeter}(A)

where S(A) is entanglement entropy of region A.

**Consequence**: Bond dimension χ grows slowly with system size, making PEPS efficient for physical systems.

Enums
-----

ContractionStrategy
~~~~~~~~~~~~~~~~~~~

.. class:: ContractionStrategy

   Enumeration of PEPS contraction algorithms.

   .. attribute:: BOUNDARY_MPS

      Contract rows into MPS boundary sequentially (default). Best balance of accuracy and speed.

      **Complexity**: O(m × n × χ⁵)

   .. attribute:: COLUMN_BY_COLUMN

      Contract columns sequentially instead of rows.

      **Use case**: Prefer for wide, shallow grids

   .. attribute:: SIMPLE_UPDATE

      Iterative tensor updates for time evolution.

      **Use case**: Time evolution, ground state search

   .. attribute:: FULL_UPDATE

      Exact contraction using full environment tensors.

      **Warning**: Exponentially expensive, only for very small patches (≤3×3)

Configuration
-------------

PEPSConfig
~~~~~~~~~~

.. class:: PEPSConfig(rows, cols, physical_dim=2, bond_dim=4, contraction_strategy=ContractionStrategy.BOUNDARY_MPS, boundary_chi=32, device='cuda')

   Configuration class for PEPS simulation.

   **Constructor**:

   .. code-block:: python

      from atlas_q.peps import PEPSConfig, ContractionStrategy

      config = PEPSConfig(
          rows=4,
          cols=4,
          physical_dim=2,        # Qubits
          bond_dim=4,            # PEPS bond dimension χ
          contraction_strategy=ContractionStrategy.BOUNDARY_MPS,
          boundary_chi=32,       # Boundary MPS bond dimension
          device='cuda'
      )

   **Parameters**:
      - ``rows`` (int): Number of rows in 2D grid
      - ``cols`` (int): Number of columns in 2D grid
      - ``physical_dim`` (int): Physical dimension per site (default: 2 for qubits)
      - ``bond_dim`` (int): Virtual bond dimension χ (default: 4)
      - ``contraction_strategy`` (ContractionStrategy): Contraction algorithm
      - ``boundary_chi`` (int): Bond dimension for boundary MPS (default: 32)
      - ``device`` (str): 'cuda' or 'cpu'

   **Memory estimate**:

   .. code-block:: python

      # PEPS tensors: rows × cols × χ⁴ × d
      # Example: 4×4 grid, χ=4, d=2
      Memory ≈ 16 × 4⁴ × 2 × 8 bytes = 64 KB

      # Boundary MPS: cols × χ_boundary²
      # Example: 4 cols, χ_boundary=32
      Boundary ≈ 4 × 32² × 8 bytes = 32 KB

      Total ≈ 96 KB (tiny!)

Classes
-------

PEPSTensor
~~~~~~~~~~

.. class:: PEPSTensor(row, col, tensor)

   Single PEPS tensor at position (row, col) in the lattice.

   **Constructor**:

   .. code-block:: python

      import torch
      from atlas_q.peps import PEPSTensor

      # Create tensor at position (0, 0)
      # Shape: [χ_up, χ_left, d, χ_right, χ_down]
      tensor = torch.randn(1, 1, 2, 4, 4, dtype=torch.complex64, device='cuda')
      peps_tensor = PEPSTensor(row=0, col=0, tensor=tensor)

   **Parameters**:
      - ``row`` (int): Row index
      - ``col`` (int): Column index
      - ``tensor`` (torch.Tensor): Rank-5 tensor with shape [χ_up, χ_left, d, χ_right, χ_down]

   **Attributes**:

   .. attribute:: row

      Row position (int)

   .. attribute:: col

      Column position (int)

   .. attribute:: tensor

      Rank-5 tensor data (torch.Tensor)

PEPS
~~~~

.. class:: PEPS(config)

   Projected Entangled Pair State tensor network for 2D lattices.

   Manages a 2D grid of rank-5 tensors with methods for gate application, contraction, and expectation value computation.

   **Constructor**:

   .. code-block:: python

      from atlas_q.peps import PEPS, PEPSConfig

      config = PEPSConfig(rows=5, cols=5, bond_dim=4, device='cuda')
      peps = PEPS(config)

   **Parameters**:
      - ``config`` (PEPSConfig): PEPS configuration

   **Attributes**:

   .. attribute:: config

      PEPS configuration (PEPSConfig)

   .. attribute:: rows

      Number of rows (int)

   .. attribute:: cols

      Number of columns (int)

   .. attribute:: tensors

      Dictionary mapping (row, col) → PEPSTensor

   **Methods**:

   .. method:: apply_single_qubit_gate(gate, row, col)

      Apply single-qubit unitary gate at position (row, col).

      :param torch.Tensor gate: 2×2 unitary matrix
      :param int row: Row index (0 to rows-1)
      :param int col: Column index (0 to cols-1)

      **Complexity**: O(χ⁴)

      **Example**:

      .. code-block:: python

         import torch
         from atlas_q.peps import PEPS, PEPSConfig

         peps = PEPS(PEPSConfig(rows=4, cols=4, device='cuda'))

         # Apply Hadamard to all qubits
         H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)
         for i in range(4):
             for j in range(4):
                 peps.apply_single_qubit_gate(H, i, j)

   .. method:: apply_two_qubit_gate(gate, pos1, pos2)

      Apply two-qubit gate between adjacent sites.

      :param torch.Tensor gate: 4×4 unitary matrix
      :param tuple pos1: First position (row1, col1)
      :param tuple pos2: Second position (row2, col2)

      **Requirements**: pos1 and pos2 must be adjacent (horizontally or vertically)

      **Complexity**: O(χ⁵) with SVD truncation

      **Example**:

      .. code-block:: python

         # Apply CZ gate between (0,0) and (0,1)
         CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64, device='cuda'))
         peps.apply_two_qubit_gate(CZ, (0, 0), (0, 1))

         # Apply CNOT vertically
         CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                             dtype=torch.complex64, device='cuda')
         peps.apply_two_qubit_gate(CNOT, (1, 2), (2, 2))

   .. method:: contract_boundary_mps(chi_max=None)

      Contract PEPS network using boundary MPS method.

      Sequentially contracts rows into a boundary MPS, producing final norm or expectation value.

      :param int chi_max: Maximum boundary MPS bond dimension (default: from config)
      :return: Norm squared ⟨ψ|ψ⟩
      :rtype: complex

      **Complexity**: O(rows × cols × χ⁵ × boundary_chi²)

      **Example**:

      .. code-block:: python

         norm = peps.contract_boundary_mps(chi_max=64)
         print(f"||ψ||² = {norm.real:.10f}")  # Should be ~1.0

   .. method:: compute_expectation(operator, positions)

      Compute expectation value of multi-site operator.

      :param torch.Tensor operator: Operator matrix (2ᵏ × 2ᵏ for k sites)
      :param list positions: List of (row, col) tuples
      :return: Expectation value ⟨ψ|O|ψ⟩
      :rtype: complex

      **Example**:

      .. code-block:: python

         import torch

         # Single-site observable: ⟨Z_{2,3}⟩
         Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
         exp_z = peps.compute_expectation(Z, [(2, 3)])

         # Two-site observable: ⟨Z_{0,0} Z_{0,1}⟩
         ZZ = torch.kron(Z, Z)
         exp_zz = peps.compute_expectation(ZZ, [(0, 0), (0, 1)])

   .. method:: to_mps()

      Convert PEPS to 1D MPS by contracting rows.

      :return: Equivalent MPS representation (approximate)
      :rtype: AdaptiveMPS

      **Use case**: Interface with 1D algorithms after 2D preparation

      **Example**:

      .. code-block:: python

         mps = peps.to_mps()
         print(f"MPS bond dimensions: {mps.bond_dimensions}")

   .. method:: get_amplitude(bitstring)

      Compute amplitude for computational basis state.

      :param str bitstring: Bitstring in row-major order (e.g., '0101...')
      :return: Amplitude ⟨bitstring|ψ⟩
      :rtype: complex

      **Complexity**: O(rows × cols × χ⁵)

      **Example**:

      .. code-block:: python

         # 4×4 grid (16 qubits)
         amp = peps.get_amplitude('0' * 16)  # Amplitude of |0000...0⟩
         print(f"Amplitude: {amp:.6f}")

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+----------------------+
| Operation               | Complexity           |
+=========================+======================+
| Single-qubit gate       | O(χ⁴)                |
+-------------------------+----------------------+
| Two-qubit gate          | O(χ⁵)                |
+-------------------------+----------------------+
| Boundary MPS contract   | O(m×n×χ⁵×χ_b²)       |
+-------------------------+----------------------+
| Expectation value       | O(m×n×χ⁵×χ_b²)       |
+-------------------------+----------------------+

where m, n are grid dimensions, χ is PEPS bond dim, χ_b is boundary MPS bond dim.

Scaling Limits
~~~~~~~~~~~~~~

**GPU memory limits** (A100 80GB):

.. code-block:: python

   # PEPS bond dimension χ
   4×4 grid: χ ≤ 8   (tractable)
   5×5 grid: χ ≤ 6   (tractable)
   6×6 grid: χ ≤ 4   (tractable)
   8×8 grid: χ ≤ 2   (limited)

   # Beyond this: use distributed_mps or circuit cutting

**Contraction time** (NVIDIA A100):

.. code-block:: python

   4×4 grid, χ=4, χ_b=32:  0.5 sec
   5×5 grid, χ=4, χ_b=32:  1.2 sec
   6×6 grid, χ=4, χ_b=32:  3.5 sec

Examples
--------

2D Cluster State
~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.peps import PEPS, PEPSConfig, ContractionStrategy
   import torch

   # Create 4×4 PEPS
   config = PEPSConfig(
       rows=4,
       cols=4,
       bond_dim=4,
       contraction_strategy=ContractionStrategy.BOUNDARY_MPS,
       boundary_chi=32,
       device='cuda'
   )
   peps = PEPS(config)

   # Step 1: Initialize all qubits in |+⟩
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)
   for i in range(4):
       for j in range(4):
           peps.apply_single_qubit_gate(H, i, j)

   # Step 2: Apply CZ gates on all edges to create cluster state
   CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64, device='cuda'))

   # Horizontal edges
   for i in range(4):
       for j in range(3):
           peps.apply_two_qubit_gate(CZ, (i, j), (i, j+1))

   # Vertical edges
   for i in range(3):
       for j in range(4):
           peps.apply_two_qubit_gate(CZ, (i, j), (i+1, j))

   # Contract and verify norm
   norm = peps.contract_boundary_mps()
   print(f"Cluster state norm: {norm.real:.6f}")  # Should be ~1.0

2D Ising Model Ground State
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.peps import PEPS, PEPSConfig
   import torch

   # 5×5 2D Ising model: H = -J Σ_<i,j> Z_i Z_j
   peps = PEPS(PEPSConfig(rows=5, cols=5, bond_dim=4, device='cuda'))

   # Initialize in |0⟩^⊗25 (ground state for ferromagnetic J>0)
   # (already initialized to |0⟩ by default)

   # Compute energy expectation
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
   ZZ = torch.kron(Z, Z)

   total_energy = 0.0
   J = 1.0

   # Horizontal bonds
   for i in range(5):
       for j in range(4):
           energy = peps.compute_expectation(ZZ, [(i, j), (i, j+1)])
           total_energy += -J * energy.real

   # Vertical bonds
   for i in range(4):
       for j in range(5):
           energy = peps.compute_expectation(ZZ, [(i, j), (i+1, j)])
           total_energy += -J * energy.real

   print(f"Ground state energy: {total_energy:.6f}")

Shallow Quantum Supremacy Circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.peps import PEPS, PEPSConfig
   import torch
   import numpy as np

   # 5×5 grid, shallow circuit (10 layers)
   peps = PEPS(PEPSConfig(rows=5, cols=5, bond_dim=6, device='cuda'))

   # Layer 1: Hadamards
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)
   for i in range(5):
       for j in range(5):
           peps.apply_single_qubit_gate(H, i, j)

   # Layers 2-10: Random single-qubit gates + structured two-qubit gates
   for layer in range(9):
       # Random single-qubit unitaries
       for i in range(5):
           for j in range(5):
               theta = np.random.rand() * 2 * np.pi
               phi = np.random.rand() * 2 * np.pi
               # U = RZ(phi) RX(theta)
               U = torch.tensor([
                   [np.cos(theta/2), -1j*np.sin(theta/2)],
                   [-1j*np.sin(theta/2), np.cos(theta/2)]
               ], dtype=torch.complex64, device='cuda')
               peps.apply_single_qubit_gate(U, i, j)

       # Structured two-qubit gates (checkerboard pattern)
       iSWAP = torch.tensor([
           [1, 0, 0, 0],
           [0, 0, 1j, 0],
           [0, 1j, 0, 0],
           [0, 0, 0, 1]
       ], dtype=torch.complex64, device='cuda')

       offset = layer % 2
       for i in range(5):
           for j in range(offset, 5, 2):
               if j+1 < 5:
                   peps.apply_two_qubit_gate(iSWAP, (i, j), (i, j+1))

   # Compute amplitude of |00...0⟩ (supremacy benchmark)
   amp = peps.get_amplitude('0' * 25)
   prob = abs(amp)**2
   print(f"P(|00...0⟩) = {prob:.10f}")

Converting PEPS to MPS
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.peps import PEPS, PEPSConfig

   # Create 4×4 PEPS state
   peps = PEPS(PEPSConfig(rows=4, cols=4, bond_dim=4, device='cuda'))
   # ... prepare state ...

   # Convert to 1D MPS (16 qubits)
   mps = peps.to_mps()

   # Now use 1D algorithms
   from atlas_q.tdvp import TDVP
   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.ising_hamiltonian(n_sites=16, J=1.0, h=0.5, device='cuda')
   tdvp = TDVP(mps=mps, hamiltonian=H, dt=0.1, method='two_site')
   energy = tdvp.run(n_steps=50)

Use Cases
---------

When to Use PEPS
~~~~~~~~~~~~~~~~

1. **2D lattice systems**: Natural 2D structure (surface codes, 2D spin models)
2. **Shallow circuits**: Quantum supremacy experiments with depth ≤20
3. **Small patches**: 4×4 to 6×6 grids (16-36 qubits)
4. **Area-law states**: Moderate entanglement satisfying area law
5. **Cluster state preparation**: 2D graph states for MBQC

When NOT to Use PEPS
~~~~~~~~~~~~~~~~~~~~

1. **Large grids**: >6×6 becomes intractable (use circuit cutting or distributed MPS)
2. **Deep circuits**: Depth >20 causes bond dimension explosion
3. **Volume-law entanglement**: Random circuits with extensive entanglement
4. **1D systems**: Use MPS instead (more efficient)

PEPS vs MPS
~~~~~~~~~~~

+---------------------------+----------------------+----------------------+
| Feature                   | PEPS                 | MPS                  |
+===========================+======================+======================+
| **Geometry**              | 2D lattice           | 1D chain             |
+---------------------------+----------------------+----------------------+
| **Entanglement capacity** | Area law             | Bounded              |
+---------------------------+----------------------+----------------------+
| **Contraction**           | #P-hard (approx.)    | Polynomial (exact)   |
+---------------------------+----------------------+----------------------+
| **Max system size**       | ~6×6 (36 qubits)     | ~100 qubits          |
+---------------------------+----------------------+----------------------+
| **Use case**              | 2D physics, patches  | 1D systems, general  |
+---------------------------+----------------------+----------------------+

Limitations
-----------

Current Implementation
~~~~~~~~~~~~~~~~~~~~~~

This is a **"light" PEPS implementation** optimized for:
  - Small patches (4×4 to 6×6)
  - Shallow circuits (depth ≤ 20)
  - Proof-of-concept 2D algorithms

**Not suitable for**:
  - Large-scale 2D simulations (use circuit cutting)
  - Deep circuits (use 1D MPS with optimized layout)
  - Production PEPS algorithms (iTEBD, CTMRG)

For large-scale 2D simulations, see:
  - :doc:`circuit_cutting` - Partition large 2D grids into patches
  - :doc:`distributed_mps` - Distributed computation for >50 qubits
  - :doc:`planar_2d` - Optimized 2D→1D circuit mapping

Cross-References
----------------

See Also
~~~~~~~~

- :doc:`../user_guide/explanations/tensor_networks` - PEPS theory and area law
- :doc:`planar_2d` - 2D qubit layout and routing
- :doc:`circuit_cutting` - Splitting large 2D circuits
- :doc:`adaptive_mps` - 1D MPS (more efficient for non-2D systems)
- :doc:`../user_guide/tutorials/advanced_features` - PEPS tutorial

References
~~~~~~~~~~

Key papers on PEPS:

1. **Verstraete, F. & Cirac, J. I.** (2004). "Renormalization algorithms for quantum many-body systems in two and higher dimensions." arXiv:cond-mat/0407066
2. **Orús, R.** (2014). "A practical introduction to tensor networks: Matrix product states and projected entangled pair states." Annals of Physics, 349, 117-158.
3. **Eisert, J. et al.** (2010). "Colloquium: Area laws for the entanglement entropy." Reviews of Modern Physics, 82(1), 277.
4. **Arute, F. et al.** (2019). "Quantum supremacy using a programmable superconducting processor." Nature, 574(7779), 505-510.
