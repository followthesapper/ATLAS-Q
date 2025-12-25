How to Build Custom Hamiltonians
==================================

This guide shows how to construct custom Hamiltonians for physics models, optimization problems, or molecular systems beyond the built-in options in ATLAS-Q.

Problem
-------

You need to:

- Implement a physics model not provided by MPOBuilder
- Build Hamiltonians with custom interaction patterns
- Create problem Hamiltonians for QAOA on custom graphs
- Construct time-dependent Hamiltonians for driven systems
- Combine multiple Hamiltonian terms with different coefficients

Prerequisites
-------------

- Understanding of MPO (Matrix Product Operator) representation
- Knowledge of Pauli operators and quantum mechanics notation
- Familiarity with torch tensors
- See :doc:`../tutorials/mps_basics` for MPS/MPO fundamentals

Build from Local Operators
---------------------------

Create Hamiltonians from single-site operators.

Simple Local Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^

Sum of non-interacting single-site terms:

.. code-block:: python

   import torch
   from atlas_q.mpo_ops import MPO

   # Pauli operators
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
   Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device='cuda')
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
   I = torch.eye(2, dtype=torch.complex64, device='cuda')

   # H = X₀ + Z₁ + X₂ + Z₃ + ...
   n_sites = 10
   ops = [X if i % 2 == 0 else Z for i in range(n_sites)]

   H = MPO.from_local_ops(ops, device='cuda')

   # Verify
   from atlas_q.adaptive_mps import AdaptiveMPS
   mps = AdaptiveMPS(num_qubits=n_sites, bond_dim=2, device='cuda')
   energy = mps.expectation(H)
   print(f"Energy: {energy:.6f}")

Weighted Local Terms
^^^^^^^^^^^^^^^^^^^^^

Apply different coefficients to each site:

.. code-block:: python

   # H = h₀X₀ + h₁Z₁ + h₂X₂ + ...
   coefficients = [1.0, 0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1.0]

   ops_weighted = [coeff * (X if i % 2 == 0 else Z)
                   for i, coeff in enumerate(coefficients)]

   H_weighted = MPO.from_local_ops(ops_weighted, device='cuda')

Build Nearest-Neighbor Interactions
------------------------------------

Two-Site Terms
^^^^^^^^^^^^^^

Construct Hamiltonians with nearest-neighbor interactions:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   # Heisenberg model: H = Σᵢ (JₓXᵢXᵢ₊₁ + JᵧYᵢYᵢ₊₁ + JᵤZᵢZᵢ₊₁)
   H_heisenberg = MPOBuilder.heisenberg_hamiltonian(
       n_sites=12,
       Jx=1.0,
       Jy=1.0,
       Jz=1.0,
       device='cuda'
   )

   # Ising model: H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ
   H_ising = MPOBuilder.ising_hamiltonian(
       n_sites=12,
       J=1.0,
       h=0.5,
       device='cuda'
   )

   # XY model: H = Σᵢ (JXᵢXᵢ₊₁ + JYᵢYᵢ₊₁)
   H_xy = MPOBuilder.xy_hamiltonian(
       n_sites=12,
       J=1.0,
       device='cuda'
   )

Custom Two-Site Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build arbitrary nearest-neighbor terms:

.. code-block:: python

   # Custom interaction: H = Σᵢ (XᵢZᵢ₊₁ + ZᵢXᵢ₊₁)
   def build_custom_nn_hamiltonian(n_sites, coupling, device='cuda'):
       """Build custom nearest-neighbor Hamiltonian."""
       X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
       Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
       I = torch.eye(2, dtype=torch.complex64, device=device)

       # Build MPO tensors
       W_first = torch.zeros(1, 2, 2, 3, dtype=torch.complex64, device=device)
       W_first[0, :, :, 0] = I
       W_first[0, :, :, 1] = coupling * X
       W_first[0, :, :, 2] = coupling * Z

       W_middle = torch.zeros(3, 2, 2, 3, dtype=torch.complex64, device=device)
       W_middle[0, :, :, 0] = I
       W_middle[1, :, :, 2] = Z
       W_middle[2, :, :, 2] = X
       W_middle[0, :, :, 1] = coupling * X
       W_middle[0, :, :, 2] = coupling * Z

       W_last = torch.zeros(3, 2, 2, 1, dtype=torch.complex64, device=device)
       W_last[0, :, :, 0] = I
       W_last[1, :, :, 0] = Z
       W_last[2, :, :, 0] = X

       tensors = [W_first] + [W_middle.clone() for _ in range(n_sites-2)] + [W_last]
       return MPO(tensors)

   H_custom = build_custom_nn_hamiltonian(n_sites=10, coupling=1.5, device='cuda')

Sum and Combine Hamiltonians
-----------------------------

Linear Combination
^^^^^^^^^^^^^^^^^^

Add multiple Hamiltonian terms:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   # H = H_ising + λ H_transverse
   H_ising = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0, device='cuda')
   H_transverse = MPOBuilder.ising_hamiltonian(10, J=0.0, h=1.0, device='cuda')

   lambda_coupling = 0.3
   H_total = H_ising + lambda_coupling * H_transverse

   # More complex combinations
   H1 = MPOBuilder.heisenberg_hamiltonian(10, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')
   H2 = MPOBuilder.ising_hamiltonian(10, J=0.5, h=0.0, device='cuda')
   H3 = MPOBuilder.xy_hamiltonian(10, J=0.2, device='cuda')

   H_combined = 1.0 * H1 + 0.5 * H2 + 0.2 * H3

Weighted Sum
^^^^^^^^^^^^

Build Hamiltonians with many terms:

.. code-block:: python

   # H = Σᵢ cᵢ Hᵢ
   terms = [
       (1.0, MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0, device='cuda')),
       (0.5, MPOBuilder.ising_hamiltonian(10, J=0.0, h=1.0, device='cuda')),
       (0.3, MPOBuilder.xy_hamiltonian(10, J=1.0, device='cuda')),
       (0.1, MPOBuilder.heisenberg_hamiltonian(10, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda'))
   ]

   H_total = sum(coeff * H for coeff, H in terms)

Build Long-Range Interactions
------------------------------

Power-Law Interactions
^^^^^^^^^^^^^^^^^^^^^^

Interactions decaying with distance:

.. code-block:: python

   # H = Σᵢ<ⱼ (1/|i-j|^α) ZᵢZⱼ
   def build_long_range_hamiltonian(n_sites, alpha=3.0, device='cuda'):
       """Build long-range interaction Hamiltonian."""
       Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

       # Bond dimension grows with range
       max_bond_dim = min(2**(n_sites//2), 64)  # Cap to avoid explosion

       tensors = []
       for i in range(n_sites):
           if i == 0:
               bond_L, bond_R = 1, max_bond_dim
           elif i == n_sites - 1:
               bond_L, bond_R = max_bond_dim, 1
           else:
               bond_L, bond_R = max_bond_dim, max_bond_dim

           W = torch.zeros(bond_L, 2, 2, bond_R, dtype=torch.complex64, device=device)

           # Implement interaction pattern
           # (Simplified - full implementation requires careful MPO construction)
           for j in range(i+1, n_sites):
               coupling = 1.0 / abs(i - j)**alpha
               # Add term to W encoding Zᵢ Zⱼ interaction
               # ...

           tensors.append(W)

       return MPO(tensors)

   H_long_range = build_long_range_hamiltonian(n_sites=12, alpha=2.0, device='cuda')

All-to-All Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^

Every qubit interacts with every other:

.. code-block:: python

   # Dense coupling matrix
   import numpy as np

   n_sites = 8
   coupling_matrix = np.random.randn(n_sites, n_sites)
   coupling_matrix = (coupling_matrix + coupling_matrix.T) / 2  # Symmetrize

   # H = Σᵢ<ⱼ Jᵢⱼ ZᵢZⱼ
   def build_all_to_all_hamiltonian(coupling_matrix, device='cuda'):
       """Build all-to-all coupling Hamiltonian."""
       n_sites = len(coupling_matrix)
       Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

       # Build as sum of two-qubit terms
       H_total = None
       for i in range(n_sites):
           for j in range(i+1, n_sites):
               J_ij = coupling_matrix[i, j]

               # Create single term Hᵢⱼ = Jᵢⱼ ZᵢZⱼ
               ops = [torch.eye(2, dtype=torch.complex64, device=device) for _ in range(n_sites)]
               ops[i] = Z
               ops[j] = Z

               H_ij = J_ij * MPO.from_local_ops(ops, device=device)

               if H_total is None:
                   H_total = H_ij
               else:
                   H_total = H_total + H_ij

       return H_total

   H_all_to_all = build_all_to_all_hamiltonian(coupling_matrix, device='cuda')

Build Graph Hamiltonians for QAOA
----------------------------------

MaxCut Hamiltonian
^^^^^^^^^^^^^^^^^^

Encode MaxCut problem as Hamiltonian:

.. code-block:: python

   import networkx as nx
   from atlas_q.mpo_ops import MPOBuilder

   # Create graph
   G = nx.Graph()
   G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

   # Build MaxCut Hamiltonian: H = -½ Σ_{(i,j) ∈ E} (I - ZᵢZⱼ)
   H_maxcut = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Verify: ground state maximizes cut
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   config = VQEConfig(device='cuda')
   qaoa = QAOA(H_maxcut, p=2, config=config)
   energy, params = qaoa.optimize()

   print(f"MaxCut value (approx): {-energy:.2f}")

Custom Graph Problem
^^^^^^^^^^^^^^^^^^^^

General graph optimization Hamiltonians:

.. code-block:: python

   def build_vertex_cover_hamiltonian(G, penalty_weight=5.0, device='cuda'):
       """
       Minimize vertex cover.
       H = Σᵢ Zᵢ + penalty × Σ_{(i,j) ∈ E} (1-Zᵢ)(1-Zⱼ)/4
       """
       n_qubits = G.number_of_nodes()
       Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
       I = torch.eye(2, dtype=torch.complex64, device=device)

       # Cost: sum of vertices
       ops_cost = [Z for _ in range(n_qubits)]
       H_cost = MPO.from_local_ops(ops_cost, device=device)

       # Penalty: edges not covered
       H_penalty = None
       for i, j in G.edges():
           # (1-Zᵢ)(1-Zⱼ)/4 = (I - Zᵢ - Zⱼ + ZᵢZⱼ)/4
           ops = [I for _ in range(n_qubits)]
           H_term = MPO.from_local_ops(ops, device=device)  # I⊗I⊗...

           ops_zi = ops.copy()
           ops_zi[i] = Z
           H_term = H_term - MPO.from_local_ops(ops_zi, device=device)

           ops_zj = ops.copy()
           ops_zj[j] = Z
           H_term = H_term - MPO.from_local_ops(ops_zj, device=device)

           ops_zizj = ops.copy()
           ops_zizj[i] = Z
           ops_zizj[j] = Z
           H_term = H_term + MPO.from_local_ops(ops_zizj, device=device)

           H_term = H_term * 0.25

           if H_penalty is None:
               H_penalty = H_term
           else:
               H_penalty = H_penalty + H_term

       return H_cost + penalty_weight * H_penalty

   # Example usage
   G_vc = nx.cycle_graph(6)
   H_vertex_cover = build_vertex_cover_hamiltonian(G_vc, penalty_weight=10.0, device='cuda')

Build Time-Dependent Hamiltonians
----------------------------------

Periodic Driving
^^^^^^^^^^^^^^^^

Hamiltonians with sinusoidal time dependence:

.. code-block:: python

   import numpy as np
   from atlas_q.tdvp import TDVP1Site, TDVPConfig

   def time_dependent_H(t, H_static, H_drive, omega, amplitude):
       """
       H(t) = H_static + A sin(ωt) H_drive
       """
       return H_static + amplitude * np.sin(omega * t) * H_drive

   # Static and drive Hamiltonians
   H_static = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0, device='cuda')
   H_drive = MPOBuilder.ising_hamiltonian(10, J=0.0, h=1.0, device='cuda')

   # Time evolution with periodic driving
   mps = AdaptiveMPS(num_qubits=10, bond_dim=32, device='cuda')

   omega = 2.0 * np.pi  # Angular frequency
   amplitude = 0.5
   dt = 0.01
   t_final = 5.0

   config = TDVPConfig(dt=dt, t_final=t_final, device='cuda')

   for t in np.arange(0, t_final, dt):
       H_t = time_dependent_H(t, H_static, H_drive, omega, amplitude)

       tdvp = TDVP1Site(H_t, mps, TDVPConfig(dt=dt, t_final=dt, device='cuda'))
       tdvp.evolve()

       if int(t / dt) % 10 == 0:
           energy = mps.expectation(H_t)
           print(f"t = {t:.2f}, E(t) = {energy:.6f}")

Quench Dynamics
^^^^^^^^^^^^^^^

Sudden change in Hamiltonian:

.. code-block:: python

   # Initial Hamiltonian (h = 0)
   H_initial = MPOBuilder.ising_hamiltonian(12, J=1.0, h=0.0, device='cuda')

   # Prepare ground state
   mps = AdaptiveMPS(num_qubits=12, bond_dim=64, device='cuda')
   config_prep = TDVPConfig(dt=0.05, t_final=10.0, imaginary_time=True, device='cuda')
   tdvp_prep = TDVP1Site(H_initial, mps, config_prep)
   tdvp_prep.evolve()

   E_initial = mps.expectation(H_initial)
   print(f"Initial ground state energy: {E_initial:.6f}")

   # Quench: suddenly change h=0 → h=2
   H_final = MPOBuilder.ising_hamiltonian(12, J=1.0, h=2.0, device='cuda')

   # Real-time evolution after quench
   config_quench = TDVPConfig(dt=0.01, t_final=5.0, imaginary_time=False, device='cuda')
   tdvp_quench = TDVP1Site(H_final, mps, config_quench)

   times = []
   energies = []
   for step in range(500):
       tdvp_quench.step()
       t = step * config_quench.dt
       E = mps.expectation(H_final)
       times.append(t)
       energies.append(E)

   # Plot energy oscillations
   import matplotlib.pyplot as plt
   plt.plot(times, energies)
   plt.xlabel('Time')
   plt.ylabel('Energy')
   plt.title('Energy after quench')
   plt.savefig('quench_dynamics.png')

Build Molecular Hamiltonians
-----------------------------

Custom Molecular Geometries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond built-in molecules:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   # Custom molecule geometry
   geometry = [
       ('C', (0.0, 0.0, 0.0)),
       ('O', (1.2, 0.0, 0.0)),
       ('H', (-0.5, 0.9, 0.0)),
       ('H', (-0.5, -0.9, 0.0))
   ]

   H_molecule = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='formaldehyde',  # Name for caching
       basis='sto-3g',
       geometry=geometry,
       charge=0,
       spin=0,
       active_space=(4, 4),  # 4 electrons in 4 orbitals
       device='cuda'
   )

Custom Electronic Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct Hamiltonian construction from integrals:

.. code-block:: python

   # Fermionic Hamiltonian from one- and two-electron integrals
   def build_fermionic_hamiltonian(h_pq, h_pqrs, n_qubits, device='cuda'):
       """
       Build electronic Hamiltonian: H = Σ_{pq} h_{pq} a†_p a_q + ½ Σ_{pqrs} h_{pqrs} a†_p a†_q a_r a_s

       Args:
           h_pq: One-electron integrals (n_orbitals × n_orbitals)
           h_pqrs: Two-electron integrals (n_orbitals^4)
           n_qubits: Number of qubits (spin-orbitals)
       """
       # Convert fermionic operators to Pauli via Jordan-Wigner
       # (Simplified - use openfermion for full implementation)

       from openfermion import FermionOperator, jordan_wigner
       import openfermion

       # Build fermionic operator
       ferm_op = FermionOperator()
       n_orbitals = n_qubits // 2

       for p in range(n_orbitals):
           for q in range(n_orbitals):
               for spin in [0, 1]:
                   p_spin = 2*p + spin
                   q_spin = 2*q + spin
                   ferm_op += FermionOperator(f'{p_spin}^ {q_spin}', h_pq[p, q])

       # Add two-electron terms...
       # (Full implementation omitted for brevity)

       # Jordan-Wigner transformation
       qubit_op = jordan_wigner(ferm_op)

       # Convert to MPO
       # ... (implementation details)

       return H_mpo

Verification and Testing
-------------------------

Validate Custom Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Check Hamiltonian properties:

.. code-block:: python

   def validate_hamiltonian(H, mps):
       """Verify Hamiltonian is valid."""
       # 1. Hermiticity check (H† = H)
       is_hermitian = H.is_hermitian()
       print(f"Hermitian: {is_hermitian}")

       # 2. Real eigenvalues check
       energy = mps.expectation(H)
       print(f"Energy (should be real): {energy}")
       assert abs(energy.imag) < 1e-10, "Energy has imaginary component!"

       # 3. Size check
       print(f"Number of sites: {H.num_sites}")
       print(f"Bond dimensions: {H.bond_dimensions}")

       # 4. Norm check
       norm = H.norm()
       print(f"Hamiltonian norm: {norm:.6f}")

       return is_hermitian and abs(energy.imag) < 1e-10

   # Test custom Hamiltonian
   H_custom = build_custom_nn_hamiltonian(n_sites=10, coupling=1.0, device='cuda')
   mps_test = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')
   valid = validate_hamiltonian(H_custom, mps_test)
   print(f"Validation passed: {valid}")

Compare to Known Results
^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmark against exact or known solutions:

.. code-block:: python

   # Compare custom Hamiltonian to built-in
   H_custom = build_custom_nn_hamiltonian(n_sites=8, coupling=1.0, device='cuda')
   H_builtin = MPOBuilder.ising_hamiltonian(8, J=1.0, h=0.0, device='cuda')

   mps = AdaptiveMPS(num_qubits=8, bond_dim=16, device='cuda')

   E_custom = mps.expectation(H_custom)
   E_builtin = mps.expectation(H_builtin)

   print(f"Custom H energy: {E_custom:.8f}")
   print(f"Built-in H energy: {E_builtin:.8f}")
   print(f"Difference: {abs(E_custom - E_builtin):.2e}")

   assert abs(E_custom - E_builtin) < 1e-6, "Hamiltonians differ!"

Summary
-------

To build custom Hamiltonians:

1. **Local operators**: Use ``MPO.from_local_ops`` for single-site terms
2. **Nearest-neighbor**: Extend built-in models or construct MPO tensors manually
3. **Long-range**: Build MPO with larger bond dimensions encoding non-local interactions
4. **Combinations**: Sum MPOs with ``H_total = H1 + coeff * H2``
5. **Graph problems**: Convert graph structure to Pauli operators (MaxCut, vertex cover)
6. **Time-dependent**: Rebuild Hamiltonian at each timestep in TDVP
7. **Molecular**: Use PySCF integration or build from electronic structure integrals
8. **Validation**: Check Hermiticity, real eigenvalues, and compare to known results

Custom Hamiltonians enable simulating arbitrary physics and optimization problems with ATLAS-Q.

See Also
--------

- :doc:`../reference/mpo_ops` - MPO and MPOBuilder API
- :doc:`../tutorials/vqe_tutorial` - VQE with custom Hamiltonians
- :doc:`../tutorials/qaoa_tutorial` - Graph problem Hamiltonians
- :doc:`../tutorials/molecular_vqe` - Molecular Hamiltonian construction
- :doc:`../explanations/tensor_networks` - MPO structure and properties
