atlas_q.mpo_ops
===============

.. automodule:: atlas_q.mpo_ops
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``mpo_ops`` module provides **Matrix Product Operator (MPO)** operations for representing quantum operators as tensor networks. MPOs enable efficient computation of expectation values, operator application, and correlation functions for systems where operators have local structure.

Key Features
~~~~~~~~~~~~

- **Efficient operator representation**: O(n D²) vs. O(4ⁿ) for dense matrices
- **Hamiltonian builders**: Ising, Heisenberg, molecular chemistry, MaxCut
- **Expectation values**: :math:`\langle\psi|\hat{O}|\psi\rangle` in O(n χ² D²) time
- **Operator application**: :math:`|\psi'\rangle = \hat{O}|\psi\rangle`
- **Correlation functions**: Two-point and multi-point correlators
- **GPU acceleration**: CUDA-enabled tensor contractions

Why MPOs?
~~~~~~~~~

Dense operator matrices scale as :math:`d^{2n}` (16 GB for 15 qubits). MPOs exploit locality:

**Local Hamiltonians**: Many-body Hamiltonians with nearest-neighbor or short-range interactions can be represented exactly with small bond dimension D (typically D ≤ 10).

**Examples**:
  - Ising model: D = 3
  - Heisenberg model: D = 5
  - Molecular Hamiltonians: D = O(N_orbitals)

Mathematical Background
-----------------------

Matrix Product Operator Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An n-site operator :math:`\hat{O}` is decomposed as:

.. math::

   \hat{O} = \sum_{s_1,s_1',\ldots,s_n,s_n'} W^{[1]}_{s_1 s_1'} W^{[2]}_{s_2 s_2'} \cdots W^{[n]}_{s_n s_n'}
             |s_1 \ldots s_n\rangle \langle s_1' \ldots s_n'|

where each tensor :math:`W^{[i]}` has shape :math:`[\chi_{i-1}, d, d, \chi_i]`:
  - :math:`d = 2` for qubits
  - :math:`\chi_i` is the MPO bond dimension (controls accuracy)

**Storage**: O(n D² d²) vs. O(d^{2n}) for dense operator

Hamiltonian as MPO
~~~~~~~~~~~~~~~~~~

Consider the transverse-field Ising model:

.. math::

   \hat{H} = -J \sum_{i=1}^{n-1} Z_i Z_{i+1} - h \sum_{i=1}^n X_i

This can be exactly represented as an MPO with bond dimension D = 3:

.. math::

   W^{[i]} = \begin{pmatrix}
   I & 0 & 0 \\
   Z & 0 & 0 \\
   -h X & -J Z & I
   \end{pmatrix}

for internal sites, with boundary conditions for first and last sites.

Expectation Value
~~~~~~~~~~~~~~~~~

Computing :math:`\langle\psi|\hat{O}|\psi\rangle` for MPS :math:`|\psi\rangle` and MPO :math:`\hat{O}`:

.. math::

   \langle\psi|\hat{O}|\psi\rangle = \text{Contract}(A^*, W, A)

**Complexity**: O(n χ² D²) where χ is MPS bond dimension, D is MPO bond dimension

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MPO
   MPOBuilder

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   expectation_value
   apply_mpo_to_mps
   correlation_function

MPO
---

.. autoclass:: MPO
   :members:
   :undoc-members:
   :show-inheritance:

   Matrix Product Operator representation for quantum operators.

   Stores operator as chain of 4-tensors with efficient contraction algorithms. Supports operator addition, scaling, and composition.

   **Constructor**:

   .. code-block:: python

      from atlas_q.mpo_ops import MPO
      import torch

      # Create MPO from tensor list
      tensors = [...]  # List of shape [chi_L, d, d, chi_R]
      mpo = MPO(tensors, device='cuda')

   **Parameters**:
      - ``tensors`` (list[torch.Tensor]): List of MPO tensors
      - ``device`` (str): 'cuda' or 'cpu'

   **Storage**: Approximately :math:`16 n D^2 d^2` bytes for complex64

   .. rubric:: Methods

   .. autosummary::

      ~MPO.identity
      ~MPO.from_local_ops
      ~MPO.from_operators
      ~MPO.__add__
      ~MPO.__mul__
      ~MPO.bond_dimensions

   **Class Methods**:

   .. classmethod:: identity(n_sites, device='cuda')

      Create identity operator MPO.

      :param int n_sites: Number of sites
      :param str device: Device
      :return: Identity MPO
      :rtype: MPO

      **Example**:

      .. code-block:: python

         from atlas_q.mpo_ops import MPO

         I = MPO.identity(n_sites=10, device='cuda')

   .. classmethod:: from_local_ops(operators, device='cuda')

      Create MPO from list of local (single-site) operators.

      :param list operators: List of 2×2 operator matrices
      :param str device: Device
      :return: MPO representing tensor product
      :rtype: MPO

      **Example**:

      .. code-block:: python

         import torch
         from atlas_q.mpo_ops import MPO

         X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
         Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
         I = torch.eye(2, dtype=torch.complex64)

         # Create X ⊗ Z ⊗ I
         mpo = MPO.from_local_ops([X, Z, I], device='cuda')

   .. classmethod:: from_operators(op_strings, coeffs, n_sites, device='cuda')

      Create MPO from sum of operator strings.

      :param list op_strings: List of operator strings (e.g., ['XXI', 'ZZI'])
      :param list coeffs: Corresponding coefficients
      :param int n_sites: Number of sites
      :param str device: Device
      :return: MPO
      :rtype: MPO

      **Example**:

      .. code-block:: python

         # H = 0.5 X₀X₁ + 0.3 Z₀Z₁
         mpo = MPO.from_operators(
             op_strings=['XXI', 'ZZI', 'IIX'],
             coeffs=[0.5, 0.3, 0.2],
             n_sites=3,
             device='cuda'
         )

   **Operator Arithmetic**:

   .. method:: __add__(other)

      Add two MPOs: :math:`\hat{O}_1 + \hat{O}_2`

      :param MPO other: Another MPO
      :return: Sum MPO
      :rtype: MPO

      **Bond dimension grows**: :math:`D_{\text{sum}} = D_1 + D_2`

      **Example**:

      .. code-block:: python

         H1 = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0)
         H2 = MPOBuilder.ising_hamiltonian(10, J=0.0, h=0.5)
         H_total = H1 + H2  # Combined Hamiltonian

   .. method:: __mul__(scalar)

      Scalar multiplication: :math:`c \hat{O}`

      :param float scalar: Scalar coefficient
      :return: Scaled MPO
      :rtype: MPO

MPOBuilder
----------

.. autoclass:: MPOBuilder
   :members:
   :undoc-members:
   :show-inheritance:

   Factory class for constructing common quantum Hamiltonians as MPOs.

   Provides pre-built Hamiltonians for:
     - Spin models (Ising, Heisenberg, XY)
     - Molecular chemistry (via PySCF)
     - Combinatorial optimization (MaxCut, QAOA)

   .. rubric:: Methods

   .. autosummary::

      ~MPOBuilder.identity_mpo
      ~MPOBuilder.ising_hamiltonian
      ~MPOBuilder.heisenberg_hamiltonian
      ~MPOBuilder.xy_hamiltonian
      ~MPOBuilder.molecular_hamiltonian_from_specs
      ~MPOBuilder.maxcut_hamiltonian

   .. staticmethod:: identity_mpo(n_sites, device='cuda')

      Create identity operator.

      :param int n_sites: Number of sites
      :param str device: Device
      :return: Identity MPO
      :rtype: MPO

   .. staticmethod:: ising_hamiltonian(n_sites, J=1.0, h=0.5, periodic=False, device='cuda')

      Create transverse-field Ising Hamiltonian.

      .. math::

         \hat{H} = -J \sum_{i=1}^{n-1} Z_i Z_{i+1} - h \sum_{i=1}^n X_i

      :param int n_sites: Number of sites
      :param float J: Coupling strength
      :param float h: Transverse field strength
      :param bool periodic: Periodic boundary conditions
      :param str device: Device
      :return: Ising MPO (bond dimension D=3)
      :rtype: MPO

      **Example**:

      .. code-block:: python

         from atlas_q.mpo_ops import MPOBuilder

         # Ferromagnetic (J > 0)
         H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')

         # Antiferromagnetic (J < 0)
         H = MPOBuilder.ising_hamiltonian(n_sites=10, J=-1.0, h=0.5, device='cuda')

   .. staticmethod:: heisenberg_hamiltonian(n_sites, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

      Create Heisenberg Hamiltonian.

      .. math::

         \hat{H} = \sum_{i=1}^{n-1} (J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1})

      :param int n_sites: Number of sites
      :param float Jx: X coupling
      :param float Jy: Y coupling
      :param float Jz: Z coupling
      :param str device: Device
      :return: Heisenberg MPO (bond dimension D=5)
      :rtype: MPO

      **Special cases**:
         - Jx = Jy = Jz: Isotropic Heisenberg
         - Jx = Jy, Jz = 0: XY model
         - Jx = Jy = 0: Ising model

      **Example**:

      .. code-block:: python

         # Isotropic Heisenberg (SU(2) symmetric)
         H = MPOBuilder.heisenberg_hamiltonian(n_sites=10, Jx=1.0, Jy=1.0, Jz=1.0)

         # XXZ model
         H = MPOBuilder.heisenberg_hamiltonian(n_sites=10, Jx=1.0, Jy=1.0, Jz=2.0)

   .. staticmethod:: xy_hamiltonian(n_sites, Jx=1.0, Jy=1.0, device='cuda')

      Create XY model Hamiltonian.

      .. math::

         \hat{H} = \sum_{i=1}^{n-1} (J_x X_i X_{i+1} + J_y Y_i Y_{i+1})

      :param int n_sites: Number of sites
      :param float Jx: X coupling
      :param float Jy: Y coupling
      :param str device: Device
      :return: XY MPO (bond dimension D=3)
      :rtype: MPO

   .. staticmethod:: molecular_hamiltonian_from_specs(molecule, basis='sto-3g', geometry=None, device='cuda')

      Create molecular Hamiltonian using PySCF.

      :param str molecule: Molecule name ('H2', 'LiH', 'H2O', etc.)
      :param str basis: Basis set ('sto-3g', '6-31g', etc.)
      :param list geometry: Optional custom geometry as [(atom, coords), ...]
      :param str device: Device
      :return: Molecular Hamiltonian MPO
      :rtype: MPO

      **Requires**: PySCF installed (``pip install pyscf``)

      **Example**:

      .. code-block:: python

         # H2 molecule at equilibrium
         H = MPOBuilder.molecular_hamiltonian_from_specs(
             molecule='H2',
             basis='sto-3g',
             device='cuda'
         )

         # LiH with custom geometry
         H = MPOBuilder.molecular_hamiltonian_from_specs(
             molecule='LiH',
             basis='6-31g',
             geometry=[('Li', (0, 0, 0)), ('H', (1.6, 0, 0))],  # Angstroms
             device='cuda'
         )

   .. staticmethod:: maxcut_hamiltonian(graph, device='cuda')

      Create MaxCut Hamiltonian for graph.

      .. math::

         \hat{H} = -\frac{1}{2} \sum_{(i,j) \in E} (I - Z_i Z_j)

      :param networkx.Graph graph: Graph for MaxCut problem
      :param str device: Device
      :return: MaxCut MPO
      :rtype: MPO

      **Example**:

      .. code-block:: python

         import networkx as nx
         from atlas_q.mpo_ops import MPOBuilder

         # 4-node cycle graph
         G = nx.Graph()
         G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

         H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

         # Use with QAOA
         from atlas_q.vqe_qaoa import QAOA
         qaoa = QAOA(hamiltonian=H, p=3)
         result = qaoa.run()

Functions
---------

expectation_value
^^^^^^^^^^^^^^^^^

.. autofunction:: expectation_value

Compute expectation value :math:`\langle\psi|\hat{O}|\psi\rangle`.

**Signature**:

.. code-block:: python

   expectation_value(mps, mpo) -> complex

**Parameters**:
   - ``mps`` (AdaptiveMPS): State as Matrix Product State
   - ``mpo`` (MPO): Operator as Matrix Product Operator

**Returns**:
   - ``complex``: Expectation value

**Complexity**: O(n χ² D²) where n is number of sites, χ is MPS bond dimension, D is MPO bond dimension

**Example**:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder, expectation_value
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Create Hamiltonian
   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')

   # Create state
   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')
   # ... prepare state ...

   # Compute energy
   energy = expectation_value(mps, H)
   print(f"Energy: {energy.real:.6f} Ha")

apply_mpo_to_mps
^^^^^^^^^^^^^^^^

.. autofunction:: apply_mpo_to_mps

Apply MPO to MPS: :math:`|\psi'\rangle = \hat{O}|\psi\rangle`.

**Signature**:

.. code-block:: python

   apply_mpo_to_mps(mps, mpo, chi_max=None) -> AdaptiveMPS

**Parameters**:
   - ``mps`` (AdaptiveMPS): Input state
   - ``mpo`` (MPO): Operator to apply
   - ``chi_max`` (int, optional): Maximum bond dimension for result

**Returns**:
   - ``AdaptiveMPS``: Resulting state

**Complexity**: O(n χ² D²) with bond dimension growth χ' ≤ χ·D

**Example**:

.. code-block:: python

   from atlas_q.mpo_ops import MPO, apply_mpo_to_mps
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Create state |ψ⟩
   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   # Create operator (e.g., Hadamard on all qubits)
   H_tensor = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / (2**0.5)
   H_mpo = MPO.from_local_ops([H_tensor] * 10, device='cuda')

   # Apply: |ψ'⟩ = H|ψ⟩
   mps_prime = apply_mpo_to_mps(mps, H_mpo, chi_max=64)

correlation_function
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: correlation_function

Compute two-point correlation function :math:`\langle\psi|\hat{O}_i \hat{O}_j|\psi\rangle`.

**Signature**:

.. code-block:: python

   correlation_function(mps, op_i, op_j, site_i, site_j) -> complex

**Parameters**:
   - ``mps`` (AdaptiveMPS): State
   - ``op_i`` (torch.Tensor): Operator at site i (2×2 matrix)
   - ``op_j`` (torch.Tensor): Operator at site j (2×2 matrix)
   - ``site_i`` (int): First site index
   - ``site_j`` (int): Second site index

**Returns**:
   - ``complex``: Correlation value

**Complexity**: O(|j-i| χ²)

**Example**:

.. code-block:: python

   from atlas_q.mpo_ops import correlation_function
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device='cuda')
   # ... prepare state ...

   # Compute <Z_5 Z_10>
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
   corr = correlation_function(mps, Z, Z, site_i=5, site_j=10)
   print(f"<Z_5 Z_10> = {corr.real:.6f}")

   # Spin-spin correlation function
   for j in range(20):
       corr_j = correlation_function(mps, Z, Z, site_i=0, site_j=j)
       print(f"<Z_0 Z_{j}> = {corr_j.real:.4f}")

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+----------------------+
| Operation               | Complexity           |
+=========================+======================+
| Expectation value       | O(n χ² D²)           |
+-------------------------+----------------------+
| Apply MPO to MPS        | O(n χ² D²)           |
+-------------------------+----------------------+
| Correlation function    | O(|i-j| χ²)          |
+-------------------------+----------------------+
| MPO addition            | O(n D₁ D₂)           |
+-------------------------+----------------------+

where:
  - n: number of sites
  - χ: MPS bond dimension
  - D: MPO bond dimension

Memory Usage
~~~~~~~~~~~~

+-------------------------+----------------------+
| Structure               | Memory               |
+=========================+======================+
| MPO storage             | O(n D² d²)           |
+-------------------------+----------------------+
| MPS storage             | O(n χ² d)            |
+-------------------------+----------------------+
| Expectation value work  | O(χ² D)              |
+-------------------------+----------------------+

**Example**: 50 sites, χ=128, D=5, d=2

.. code-block:: python

   MPO: 50 × 5² × 2² × 8 bytes = 40 KB
   MPS: 50 × 128² × 2 × 8 bytes = 13 MB
   Working memory: 128² × 5 × 8 bytes = 0.6 MB

   Total: ~14 MB (dominated by MPS)

Benchmark Results
~~~~~~~~~~~~~~~~~

From ``scripts/benchmarks/validate_all_features.py``:

.. code-block:: python

   # Expectation value: 50 sites, χ=128, D=3 (Ising)
   Time: 0.08 sec (GPU), 0.35 sec (CPU)
   Speedup: 4.4×

   # Apply MPO: 50 sites, χ=64, D=3
   Time: 0.15 sec (GPU), 0.62 sec (CPU)
   Speedup: 4.1×

   # Correlation function: 50 sites, distance=25, χ=128
   Time: 0.05 sec (GPU), 0.18 sec (CPU)
   Speedup: 3.6×

Examples
--------

Ising Hamiltonian
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder, expectation_value
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Build Hamiltonian: H = -J Σ Z_i Z_{i+1} - h Σ X_i
   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')

   # Create ground state (all |0⟩ for J > 0, h = 0)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Compute energy
   energy = expectation_value(mps, H)
   print(f"Energy: {energy.real:.6f}")

Heisenberg Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # H = Σ (Jx X_i X_{i+1} + Jy Y_i Y_{i+1} + Jz Z_i Z_{i+1})
   H = MPOBuilder.heisenberg_hamiltonian(
       n_sites=10,
       Jx=1.0,
       Jy=1.0,
       Jz=1.0,
       device='cuda'
   )

   # Use with TDVP for ground state
   from atlas_q.tdvp import TDVP
   mps = AdaptiveMPS(num_qubits=10, bond_dim=32, device='cuda')
   tdvp = TDVP(mps=mps, hamiltonian=H, dt=0.1j, method='two_site')
   energy = tdvp.run(n_steps=100)

Molecular Hamiltonian (H2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Build H2 Hamiltonian
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       device='cuda'
   )

   # VQE for ground state energy
   from atlas_q.vqe_qaoa import VQE
   mps = AdaptiveMPS(num_qubits=4, bond_dim=16, device='cuda')  # H2 has 4 qubits
   vqe = VQE(mps=mps, hamiltonian=H, optimizer='COBYLA')
   result = vqe.run()
   print(f"Ground state energy: {result['energy']:.6f} Ha")

MaxCut Hamiltonian
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import networkx as nx
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA

   # Create graph
   G = nx.cycle_graph(4)  # 4-node cycle

   # Build MaxCut Hamiltonian
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Solve with QAOA
   from atlas_q.adaptive_mps import AdaptiveMPS
   mps = AdaptiveMPS(num_qubits=4, bond_dim=8, device='cuda')
   qaoa = QAOA(mps=mps, hamiltonian=H, p=3)
   result = qaoa.run()

   print(f"MaxCut value: {result['max_cut']}")
   print(f"Best partition: {result['partition']}")

Custom Hamiltonian
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from atlas_q.mpo_ops import MPO

   # Define local operators
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
   Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device='cuda')
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
   I = torch.eye(2, dtype=torch.complex64, device='cuda')

   # Build H = 0.5 X₀X₁ + 0.3 Y₀Y₁ + 0.2 Z₀Z₁
   H = MPO.from_operators(
       op_strings=['XXI', 'YYI', 'ZZI'],
       coeffs=[0.5, 0.3, 0.2],
       n_sites=3,
       device='cuda'
   )

MPO Addition
~~~~~~~~~~~~

.. code-block:: python

   # H_total = H_kinetic + H_potential
   H_kinetic = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0)
   H_potential = MPOBuilder.ising_hamiltonian(10, J=0.0, h=0.5)
   H_total = H_kinetic + H_potential

   # Bond dimension increases: D_total = D_kin + D_pot = 3 + 3 = 6

Correlation Function Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.mpo_ops import correlation_function
   import matplotlib.pyplot as plt

   n_sites = 50
   mps = AdaptiveMPS(num_qubits=n_sites, bond_dim=32, device='cuda')
   # ... prepare state (e.g., ground state of Heisenberg) ...

   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')

   # Compute spin-spin correlation
   correlations = []
   for j in range(n_sites):
       corr = correlation_function(mps, Z, Z, site_i=0, site_j=j)
       correlations.append(corr.real)

   # Plot
   plt.plot(range(n_sites), correlations)
   plt.xlabel('Distance')
   plt.ylabel('<Z_0 Z_j>')
   plt.title('Spin-Spin Correlation Function')
   plt.show()

Use Cases
---------

When to Use MPOs
~~~~~~~~~~~~~~~~

1. **Local Hamiltonians**: Nearest-neighbor or short-range interactions
2. **Expectation values**: Need frequent energy calculations (VQE, TDVP)
3. **Large systems**: >20 qubits where dense matrices infeasible
4. **Quantum chemistry**: Molecular Hamiltonians with PySCF
5. **Combinatorial optimization**: MaxCut, graph problems with QAOA

MPO Bond Dimensions
~~~~~~~~~~~~~~~~~~~

Common models and their bond dimensions:

+---------------------------+----------------+
| Hamiltonian               | Bond Dimension |
+===========================+================+
| Ising (transverse field)  | D = 3          |
+---------------------------+----------------+
| Heisenberg (isotropic)    | D = 5          |
+---------------------------+----------------+
| XY model                  | D = 3          |
+---------------------------+----------------+
| Molecular (N orbitals)    | D ~ 4N         |
+---------------------------+----------------+
| MaxCut (graph)            | D ~ max_degree |
+---------------------------+----------------+

**Rule of thumb**: Sparse graphs and local interactions → small D

Cross-References
----------------

See Also
~~~~~~~~

- :doc:`adaptive_mps` - MPS state representation
- :doc:`vqe_qaoa` - Variational algorithms using MPOs
- :doc:`tdvp` - Time evolution with MPO Hamiltonians
- :doc:`../user_guide/tutorials/vqe_tutorial` - VQE with molecular Hamiltonians
- :doc:`../user_guide/explanations/tensor_networks` - Tensor network theory

References
~~~~~~~~~~

Key papers on MPOs:

1. **Schollwöck, U.** (2011). "The density-matrix renormalization group in the age of matrix product states." Annals of Physics, 326(1), 96-192.
2. **McCulloch, I. P.** (2007). "From density-matrix renormalization group to matrix product states." Journal of Statistical Mechanics: Theory and Experiment, 2007(10), P10014.
3. **Crosswhite, G. M. & Bacon, D.** (2008). "Finite automata for caching in matrix product algorithms." Physical Review A, 78(1), 012356.
