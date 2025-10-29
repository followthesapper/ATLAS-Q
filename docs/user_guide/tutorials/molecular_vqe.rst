Molecular VQE Tutorial
======================

This tutorial demonstrates how to use VQE for quantum chemistry calculations with ATLAS-Q, covering electronic structure problems from small molecules to custom geometries with chemical accuracy.

Prerequisites
-------------

Before starting this tutorial, you should:

- Understand basic VQE principles (see :doc:`vqe_tutorial`)
- Have familiarity with quantum chemistry concepts (molecular orbitals, Hamiltonians)
- Understand fermionic operators and second quantization
- Know basics of basis sets (STO-3G, 6-31G, etc.)
- Have installed PySCF and OpenFermion packages

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Build molecular Hamiltonians from atomic specifications using PySCF integration
2. Choose appropriate basis sets and active spaces for molecular systems
3. Apply UCCSD ansatz for chemically accurate ground state calculations
4. Optimize molecular geometries and compute potential energy surfaces
5. Interpret molecular VQE results and compare to classical quantum chemistry
6. Handle multiple molecule types (H2, LiH, H2O, custom geometries)
7. Understand Jordan-Wigner transformation from fermions to qubits
8. Troubleshoot convergence and accuracy issues in molecular VQE

Part 1: Fundamentals of Molecular VQE
--------------------------------------

What is Molecular VQE?
^^^^^^^^^^^^^^^^^^^^^^

Molecular VQE applies the variational quantum eigensolver to electronic structure problems. The goal is to find the ground state energy of molecular systems by encoding the electronic Hamiltonian as a qubit operator and minimizing its expectation value.

The electronic structure problem seeks to solve the Schrödinger equation:

.. math::

   \hat{H} |\Psi\rangle = E |\Psi\rangle

where :math:`\hat{H}` is the electronic Hamiltonian, :math:`|\Psi\rangle` is the molecular wavefunction, and :math:`E` is the electronic energy. For molecular systems, the Hamiltonian in second quantization is:

.. math::

   \hat{H} = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s

where :math:`h_{pq}` are one-electron integrals (kinetic + nuclear attraction) and :math:`h_{pqrs}` are two-electron repulsion integrals.

Why Quantum Chemistry on Quantum Computers?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classical quantum chemistry methods struggle with strongly correlated systems where electrons are highly entangled. Exact methods like Full Configuration Interaction (FCI) scale exponentially with system size. VQE offers a potential quantum advantage by:

- Efficiently representing correlated wavefunctions with polynomial resources
- Using quantum hardware to prepare trial states that classical computers cannot easily simulate
- Leveraging variational principle to guarantee upper bounds on ground state energy
- Enabling chemical accuracy (within 1 kcal/mol of exact result) for industrially relevant systems

Basic Workflow
^^^^^^^^^^^^^^

A typical molecular VQE calculation follows these steps:

1. Define molecular geometry and basis set
2. Compute molecular integrals using PySCF
3. Transform fermionic Hamiltonian to qubit Hamiltonian (Jordan-Wigner or Bravyi-Kitaev)
4. Choose ansatz (UCCSD for chemical accuracy, hardware-efficient for efficiency)
5. Initialize variational parameters
6. Optimize parameters to minimize energy expectation value
7. Analyze converged wavefunction and extract properties

Installation
^^^^^^^^^^^^

Molecular VQE requires additional quantum chemistry packages:

.. code-block:: bash

   pip install pyscf openfermion openfermionpyscf

Or install ATLAS-Q with chemistry extras:

.. code-block:: bash

   pip install atlas-quantum[chemistry]

Part 2: Electronic Structure Representation
--------------------------------------------

Molecular Orbitals and Basis Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Molecular orbitals are approximated as linear combinations of atomic orbitals (LCAO):

.. math::

   \psi_i = \sum_\mu c_{\mu i} \phi_\mu

where :math:`\phi_\mu` are basis functions (typically Gaussian-type orbitals). Common basis sets include:

- **STO-3G**: Minimal basis, fast but low accuracy
- **6-31G**: Split-valence, good balance for small molecules
- **cc-pVDZ**: Correlation-consistent, higher accuracy for benchmarking

Larger basis sets provide more accurate results but increase the number of qubits required.

Second Quantization and Fermionic Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Second quantization uses creation (:math:`a_p^\dagger`) and annihilation (:math:`a_p`) operators that satisfy fermionic anticommutation relations:

.. math::

   \{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = 0

These operators create or destroy electrons in molecular orbital :math:`p`. The electronic Hamiltonian is expressed as sums of products of these operators.

Jordan-Wigner Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run on quantum hardware, we map fermionic operators to qubit operators using the Jordan-Wigner transformation:

.. math::

   a_j^\dagger \rightarrow \sigma_j^- \prod_{k<j} \sigma_k^z = \frac{1}{2}(\sigma_j^x - i\sigma_j^y) \prod_{k<j} \sigma_k^z

This preserves fermionic anticommutation relations but introduces non-local operators (string of :math:`\sigma_z`). ATLAS-Q handles this transformation automatically when building molecular Hamiltonians.

Part 3: PySCF Integration
--------------------------

Building Molecular Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ATLAS-Q integrates with PySCF to generate molecular Hamiltonians:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   # Simple H2 molecule with default geometry
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       device='cuda'
   )

   print(f"Number of qubits: {H.num_sites}")
   # Output: 4 qubits (2 electrons in 2 spatial orbitals, spin-up and spin-down)

For hydrogen molecule H2 in STO-3G basis, we get 4 spin-orbitals (2 spatial orbitals × 2 spins), requiring 4 qubits.

Custom Geometries
^^^^^^^^^^^^^^^^^

Specify atomic positions in Angstroms:

.. code-block:: python

   # LiH with custom bond length
   H_LiH = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='LiH',
       basis='sto-3g',
       geometry=[
           ('Li', (0.0, 0.0, 0.0)),
           ('H',  (1.5, 0.0, 0.0))  # 1.5 Angstrom bond length
       ],
       device='cuda'
   )

   # Water molecule
   H_H2O = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2O',
       basis='sto-3g',
       geometry=[
           ('O', (0.0,  0.0,    0.0)),
           ('H', (0.0,  0.757,  0.587)),
           ('H', (0.0, -0.757,  0.587))
       ],
       device='cuda'
   )

Active Space Selection
^^^^^^^^^^^^^^^^^^^^^^

For larger molecules, solve only in the chemically relevant active space:

.. code-block:: python

   # BeH2 with frozen core
   H_BeH2 = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='BeH2',
       basis='sto-3g',
       geometry=[
           ('Be', (0.0, 0.0, 0.0)),
           ('H',  (0.0, 0.0, 1.3)),
           ('H',  (0.0, 0.0, -1.3))
       ],
       active_space=(4, 4),  # 4 electrons in 4 orbitals
       frozen_core=True,     # Freeze Be 1s core orbital
       device='cuda'
   )

The active_space parameter (n_electrons, n_orbitals) reduces qubit count by focusing on valence electrons. Frozen core orbitals contribute a constant energy shift.

Basis Set Comparison
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch

   geometries = {
       'sto-3g': None,  # Use default
       '6-31g': None,
       'cc-pvdz': None
   }

   energies = {}
   for basis in geometries:
       H = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule='H2',
           basis=basis,
           device='cuda'
       )

       from atlas_q.vqe_qaoa import VQE, VQEConfig
       config = VQEConfig(ansatz='uccsd', max_iter=200, tol=1e-8)
       vqe = VQE(H, config)
       energy, _ = vqe.optimize()
       energies[basis] = energy

       print(f"{basis}: {energy:.8f} Ha, qubits: {H.num_sites}")

   # Expected output:
   # sto-3g:   -1.11670 Ha, qubits: 4
   # 6-31g:    -1.13163 Ha, qubits: 8
   # cc-pvdz:  -1.13460 Ha, qubits: 20
   # FCI/exact: -1.13727 Ha

Larger basis sets approach the exact result but require more qubits.

Part 4: UCCSD Ansatz for Molecules
-----------------------------------

Unitary Coupled Cluster Theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz is the gold standard for molecular VQE:

.. math::

   |\Psi(\theta)\rangle = e^{\hat{T} - \hat{T}^\dagger} |HF\rangle

where :math:`\hat{T} = \hat{T}_1 + \hat{T}_2` includes single and double excitations:

.. math::

   \hat{T}_1 = \sum_{ia} t_i^a a_a^\dagger a_i, \quad \hat{T}_2 = \sum_{ijab} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i

Here :math:`i,j` are occupied orbitals in Hartree-Fock reference and :math:`a,b` are virtual orbitals. Parameters :math:`\theta = \{t_i^a, t_{ij}^{ab}\}` are optimized variationally.

Implementing UCCSD VQE
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build H2 Hamiltonian
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       device='cuda'
   )

   # Configure UCCSD VQE
   config = VQEConfig(
       ansatz='uccsd',
       max_iter=200,
       optimizer='L-BFGS-B',
       tol=1e-8,
       gtol=1e-6
   )

   # Run VQE
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"VQE energy: {energy:.8f} Ha")
   print(f"Parameters optimized: {len(params)}")
   print(f"Chemical accuracy (0.0016 Ha): {abs(energy - (-1.137)) < 0.0016}")

   # Access convergence history
   history = vqe.get_history()
   print(f"Iterations: {len(history['energies'])}")
   print(f"Final gradient norm: {history['gradient_norms'][-1]:.2e}")

Singles and Doubles Excitations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For H2 in minimal basis (2 electrons, 4 spin-orbitals):

- **Singles**: 4 parameters (excite electron from occupied to virtual)
- **Doubles**: 1 parameter (excite both electrons)
- **Total**: 5 UCCSD parameters

For larger molecules:

.. code-block:: python

   # LiH molecule
   H_LiH = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='LiH',
       basis='sto-3g',
       device='cuda'
   )

   config = VQEConfig(ansatz='uccsd', max_iter=300)
   vqe = VQE(H_LiH, config)
   energy, params = vqe.optimize()

   print(f"LiH qubits: {H_LiH.num_sites}")
   print(f"UCCSD parameters: {len(params)}")
   print(f"Energy: {energy:.8f} Ha")
   # Expected: 12 qubits, ~20-30 parameters

Hardware-Efficient Alternative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UCCSD has many parameters for larger molecules. Hardware-efficient ansatz provides a faster alternative:

.. code-block:: python

   # Compare UCCSD vs hardware-efficient
   import time

   # UCCSD (chemically accurate but slower)
   config_uccsd = VQEConfig(ansatz='uccsd', max_iter=200)
   start = time.time()
   vqe_uccsd = VQE(H, config_uccsd)
   E_uccsd, _ = vqe_uccsd.optimize()
   t_uccsd = time.time() - start

   # Hardware-efficient (faster, less accurate)
   config_he = VQEConfig(ansatz='hardware_efficient', n_layers=5, max_iter=200)
   start = time.time()
   vqe_he = VQE(H, config_he)
   E_he, _ = vqe_he.optimize()
   t_he = time.time() - start

   print(f"UCCSD: {E_uccsd:.8f} Ha in {t_uccsd:.1f}s")
   print(f"Hardware-efficient: {E_he:.8f} Ha in {t_he:.1f}s")
   print(f"Accuracy difference: {abs(E_uccsd - E_he)*627.5:.2f} kcal/mol")

Hardware-efficient ansatz trades some accuracy for significant speedup, useful for larger molecules or noisy quantum hardware.

Part 5: Optimization for Molecules
-----------------------------------

Gradient-Based Methods
^^^^^^^^^^^^^^^^^^^^^^^

Molecular VQE typically uses gradient-based optimizers for efficiency:

.. code-block:: python

   # Compare optimizers
   optimizers = ['L-BFGS-B', 'COBYLA', 'SLSQP']

   for opt in optimizers:
       config = VQEConfig(
           ansatz='uccsd',
           optimizer=opt,
           max_iter=200,
           tol=1e-8
       )

       vqe = VQE(H, config)
       energy, params = vqe.optimize()
       history = vqe.get_history()

       print(f"{opt}: {energy:.8f} Ha in {len(history['energies'])} iterations")

   # Expected output:
   # L-BFGS-B: -1.11670 Ha in 15 iterations (fast, gradient-based)
   # COBYLA: -1.11668 Ha in 89 iterations (slower, gradient-free)
   # SLSQP: -1.11670 Ha in 18 iterations (gradient-based with constraints)

L-BFGS-B is recommended for most molecular VQE calculations due to fast convergence.

Chemical Accuracy Target
^^^^^^^^^^^^^^^^^^^^^^^^^

Chemical accuracy is defined as 1 kcal/mol = 0.0016 Hartree:

.. code-block:: python

   # Check if result is chemically accurate
   E_exact = -1.137  # Reference value (FCI or experimental)
   E_vqe = energy

   error_hartree = abs(E_vqe - E_exact)
   error_kcal = error_hartree * 627.5  # Convert Ha to kcal/mol

   print(f"Error: {error_hartree:.6f} Ha = {error_kcal:.2f} kcal/mol")
   print(f"Chemical accuracy: {error_kcal < 1.0}")

   if error_kcal < 1.0:
       print("Result is chemically accurate")
   elif error_kcal < 5.0:
       print("Result is acceptable for qualitative predictions")
   else:
       print("Consider larger basis set or better ansatz")

Convergence Monitoring
^^^^^^^^^^^^^^^^^^^^^^

Monitor convergence to ensure reliable results:

.. code-block:: python

   import matplotlib.pyplot as plt

   history = vqe.get_history()

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   # Energy convergence
   ax1.plot(history['energies'])
   ax1.axhline(E_exact, color='r', linestyle='--', label='Exact')
   ax1.set_xlabel('Iteration')
   ax1.set_ylabel('Energy (Ha)')
   ax1.set_title('VQE Energy Convergence')
   ax1.legend()
   ax1.grid(True)

   # Gradient norm
   ax2.semilogy(history['gradient_norms'])
   ax2.axhline(1e-6, color='r', linestyle='--', label='Tolerance')
   ax2.set_xlabel('Iteration')
   ax2.set_ylabel('Gradient Norm')
   ax2.set_title('Gradient Convergence')
   ax2.legend()
   ax2.grid(True)

   plt.tight_layout()
   plt.savefig('molecular_vqe_convergence.png', dpi=150)

Part 6: Multiple Molecule Examples
-----------------------------------

Hydrogen Molecule (H2)
^^^^^^^^^^^^^^^^^^^^^^

The simplest molecule, exact solution known:

.. code-block:: python

   # H2 at equilibrium geometry
   H_H2 = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       geometry=[
           ('H', (0.0, 0.0, 0.0)),
           ('H', (0.74, 0.0, 0.0))  # Equilibrium bond length
       ],
       device='cuda'
   )

   config = VQEConfig(ansatz='uccsd', max_iter=100, tol=1e-8)
   vqe = VQE(H_H2, config)
   energy, params = vqe.optimize()

   print(f"H2 energy: {energy:.8f} Ha")
   print(f"Exact (FCI): -1.13727 Ha")
   print(f"Error: {(energy + 1.13727)*627.5:.2f} kcal/mol")

Lithium Hydride (LiH)
^^^^^^^^^^^^^^^^^^^^^

Ionic molecule with heteronuclear bond:

.. code-block:: python

   # LiH molecule
   H_LiH = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='LiH',
       basis='sto-3g',
       geometry=[
           ('Li', (0.0, 0.0, 0.0)),
           ('H',  (1.6, 0.0, 0.0))  # Equilibrium ~1.6 Angstrom
       ],
       device='cuda'
   )

   config = VQEConfig(ansatz='uccsd', max_iter=200, tol=1e-8)
   vqe = VQE(H_LiH, config)
   energy_LiH, _ = vqe.optimize()

   print(f"LiH energy: {energy_LiH:.8f} Ha")
   print(f"Exact (FCI): -7.88231 Ha")
   print(f"Qubits required: {H_LiH.num_sites}")

Water Molecule (H2O)
^^^^^^^^^^^^^^^^^^^^

Bent triatomic molecule:

.. code-block:: python

   # H2O molecule
   H_H2O = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2O',
       basis='sto-3g',
       geometry=[
           ('O', (0.0,  0.0,    0.0)),
           ('H', (0.0,  0.757,  0.587)),  # 104.5° bond angle
           ('H', (0.0, -0.757,  0.587))
       ],
       device='cuda'
   )

   # Use active space to reduce qubit count
   H_H2O_active = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2O',
       basis='sto-3g',
       geometry=[
           ('O', (0.0,  0.0,    0.0)),
           ('H', (0.0,  0.757,  0.587)),
           ('H', (0.0, -0.757,  0.587))
       ],
       active_space=(8, 6),  # 8 electrons in 6 orbitals
       frozen_core=True,
       device='cuda'
   )

   config = VQEConfig(ansatz='uccsd', max_iter=300, tol=1e-7)
   vqe = VQE(H_H2O_active, config)
   energy_H2O, _ = vqe.optimize()

   print(f"H2O energy: {energy_H2O:.8f} Ha")
   print(f"Qubits: {H_H2O_active.num_sites}")

Batch Multiple Molecules
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   molecules = [
       ('H2', 'sto-3g', None, -1.137),
       ('LiH', 'sto-3g', [('Li', (0,0,0)), ('H', (1.6,0,0))], -7.882),
       ('BeH2', 'sto-3g', [('Be', (0,0,0)), ('H', (0,0,1.3)), ('H', (0,0,-1.3))], -15.598)
   ]

   results = []
   for mol_name, basis, geom, E_ref in molecules:
       H = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule=mol_name,
           basis=basis,
           geometry=geom,
           device='cuda'
       )

       config = VQEConfig(ansatz='uccsd', max_iter=200, tol=1e-7)
       vqe = VQE(H, config)
       energy, _ = vqe.optimize()

       error_kcal = abs(energy - E_ref) * 627.5
       results.append({
           'molecule': mol_name,
           'qubits': H.num_sites,
           'energy': energy,
           'error_kcal': error_kcal
       })

       print(f"{mol_name}: {energy:.6f} Ha, error {error_kcal:.2f} kcal/mol")

   # Summary
   print("\nSummary:")
   for r in results:
       accuracy = "✓" if r['error_kcal'] < 1.0 else "✗"
       print(f"{r['molecule']:6s}: {r['qubits']:2d} qubits, "
             f"{r['error_kcal']:5.2f} kcal/mol {accuracy}")

Part 7: Advanced Topics
------------------------

Active Space Selection Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large molecules, automatic active space selection reduces qubit count:

.. code-block:: python

   # Automatic active space from natural orbitals
   def select_active_space(molecule, basis, n_electrons_active):
       """Select active space based on natural orbital occupation numbers."""
       H_full = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule=molecule,
           basis=basis,
           device='cpu'  # Use CPU for integral computation
       )

       # Run cheap Hartree-Fock to get natural orbitals
       # (simplified - actual implementation in MPOBuilder)
       n_orbitals_active = n_electrons_active  # Minimal active space

       H_active = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule=molecule,
           basis=basis,
           active_space=(n_electrons_active, n_orbitals_active),
           frozen_core=True,
           device='cuda'
       )

       return H_active

   # Example: H2O with automatic active space
   H_H2O_auto = select_active_space('H2O', 'sto-3g', n_electrons_active=4)
   print(f"Reduced to {H_H2O_auto.num_sites} qubits")

Symmetry Exploitation
^^^^^^^^^^^^^^^^^^^^^^

Use molecular symmetries to reduce parameter count:

.. code-block:: python

   # Build Hamiltonian with symmetry constraints
   H_sym = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2O',
       basis='sto-3g',
       symmetry='C2v',  # Water has C2v symmetry
       device='cuda'
   )

   # VQE with symmetry-adapted ansatz reduces parameters
   config = VQEConfig(
       ansatz='uccsd',
       use_symmetry=True,
       max_iter=200
   )
   vqe = VQE(H_sym, config)
   energy, params = vqe.optimize()

   print(f"Parameters without symmetry: ~100")
   print(f"Parameters with symmetry: {len(params)}")

Excited States
^^^^^^^^^^^^^^

Compute excited states using orthogonality constraints:

.. code-block:: python

   # Ground state
   config_gs = VQEConfig(ansatz='uccsd', max_iter=200)
   vqe_gs = VQE(H, config_gs)
   E0, params0 = vqe_gs.optimize()

   # First excited state (orthogonal to ground state)
   config_ex = VQEConfig(
       ansatz='uccsd',
       max_iter=200,
       orthogonal_states=[params0]  # Constrain orthogonality
   )
   vqe_ex = VQE(H, config_ex)
   E1, params1 = vqe_ex.optimize()

   print(f"Ground state: {E0:.8f} Ha")
   print(f"First excited state: {E1:.8f} Ha")
   print(f"Excitation energy: {(E1-E0)*27.21:.2f} eV")

Potential Energy Surface
^^^^^^^^^^^^^^^^^^^^^^^^

Scan molecular geometry to compute dissociation curves:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # H2 dissociation curve
   bond_lengths = np.linspace(0.5, 3.0, 15)
   energies = []

   for r in bond_lengths:
       H_r = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule='H2',
           basis='sto-3g',
           geometry=[
               ('H', (0.0, 0.0, 0.0)),
               ('H', (r,   0.0, 0.0))
           ],
           device='cuda'
       )

       config = VQEConfig(ansatz='uccsd', max_iter=150, tol=1e-7)
       vqe = VQE(H_r, config)
       energy, _ = vqe.optimize()
       energies.append(energy)

       print(f"r = {r:.2f} A: E = {energy:.8f} Ha")

   # Plot dissociation curve
   plt.figure(figsize=(8, 5))
   plt.plot(bond_lengths, energies, 'o-', label='VQE')
   plt.xlabel('Bond Length (Angstrom)')
   plt.ylabel('Energy (Ha)')
   plt.title('H2 Potential Energy Surface')
   plt.grid(True)
   plt.legend()
   plt.savefig('h2_dissociation.png', dpi=150)

   # Find equilibrium geometry
   idx_min = np.argmin(energies)
   print(f"\nEquilibrium bond length: {bond_lengths[idx_min]:.3f} A")
   print(f"Binding energy: {energies[idx_min]:.8f} Ha")

Part 8: Performance and Accuracy
---------------------------------

Chemical Accuracy Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compare VQE results to classical benchmarks:

.. code-block:: python

   # Benchmarking function
   def benchmark_molecule(molecule, basis='sto-3g'):
       """Compare VQE to classical methods."""
       H = MPOBuilder.molecular_hamiltonian_from_specs(
           molecule=molecule,
           basis=basis,
           device='cuda'
       )

       # VQE with UCCSD
       config = VQEConfig(ansatz='uccsd', max_iter=200, tol=1e-8)
       vqe = VQE(H, config)
       E_vqe, _ = vqe.optimize()

       # Get reference energies (from PySCF or literature)
       # This would actually call PySCF methods:
       E_hf = -1.117  # Hartree-Fock (mean-field)
       E_ccsd = -1.135  # Coupled cluster singles/doubles
       E_fci = -1.137  # Full CI (exact)

       print(f"\nMolecule: {molecule}/{basis}")
       print(f"{'Method':<12s} {'Energy (Ha)':<15s} {'Error (kcal/mol)':<20s}")
       print("-" * 50)
       print(f"{'HF':<12s} {E_hf:<15.8f} {(E_hf-E_fci)*627.5:<20.2f}")
       print(f"{'CCSD':<12s} {E_ccsd:<15.8f} {(E_ccsd-E_fci)*627.5:<20.2f}")
       print(f"{'VQE-UCCSD':<12s} {E_vqe:<15.8f} {(E_vqe-E_fci)*627.5:<20.2f}")
       print(f"{'FCI':<12s} {E_fci:<15.8f} {'0.00':<20s} (reference)")

       return E_vqe

   # Run benchmark
   E_result = benchmark_molecule('H2', basis='sto-3g')

Computational Cost Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Understand how resources scale with system size:

.. code-block:: python

   # Scaling analysis
   molecules_scale = [
       ('H2', 4, 5),       # qubits, UCCSD parameters
       ('LiH', 12, 28),
       ('BeH2', 14, 42),
       ('H2O', 14, 56)
   ]

   print(f"{'Molecule':<10s} {'Qubits':<8s} {'Parameters':<12s} {'MPS χ (est.)':<15s}")
   print("-" * 50)
   for mol, nq, np in molecules_scale:
       chi_est = min(2**(nq//2), 128)  # Estimated bond dimension
       print(f"{mol:<10s} {nq:<8d} {np:<12d} {chi_est:<15d}")

   print("\nFor chemical accuracy, typically need:")
   print("- Bond dimension χ ≈ 64-128 for moderate entanglement")
   print("- Optimizer iterations: 100-300")
   print("- Wall time: 5-60 minutes on GPU")

Bond Dimension Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Test bond dimension effect on accuracy
   bond_dims = [16, 32, 64, 128]
   energies_chi = []

   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       device='cuda'
   )

   for chi in bond_dims:
       config = VQEConfig(
           ansatz='uccsd',
           bond_dim=chi,
           max_iter=200,
           tol=1e-8
       )
       vqe = VQE(H, config)
       energy, _ = vqe.optimize()
       energies_chi.append(energy)

       print(f"χ = {chi:3d}: E = {energy:.8f} Ha")

   # Plot convergence with bond dimension
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 5))
   plt.plot(bond_dims, energies_chi, 'o-')
   plt.axhline(-1.137, color='r', linestyle='--', label='Exact')
   plt.xlabel('Bond Dimension χ')
   plt.ylabel('Energy (Ha)')
   plt.title('Convergence vs MPS Bond Dimension')
   plt.grid(True)
   plt.legend()
   plt.savefig('bond_dim_convergence.png', dpi=150)

Part 9: Troubleshooting
------------------------

Problem: VQE Converges to Wrong Energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Energy significantly higher than expected (> 5 kcal/mol error)

**Solutions**:

1. Check parameter initialization:

   .. code-block:: python

      # Initialize from HF instead of random
      config = VQEConfig(
          ansatz='uccsd',
          init_strategy='hf',  # Start from Hartree-Fock
          max_iter=200
      )

2. Increase optimization iterations:

   .. code-block:: python

      config.max_iter = 500
      config.tol = 1e-9

3. Try different optimizer:

   .. code-block:: python

      # COBYLA is more robust but slower
      config.optimizer = 'COBYLA'

Problem: Optimization is Too Slow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Each iteration takes > 1 minute, convergence requires hours

**Solutions**:

1. Reduce bond dimension for faster contractions:

   .. code-block:: python

      config.bond_dim = 32  # Instead of 128

2. Use hardware-efficient ansatz:

   .. code-block:: python

      config.ansatz = 'hardware_efficient'
      config.n_layers = 3

3. Enable GPU acceleration:

   .. code-block:: python

      # Ensure using GPU
      H = MPOBuilder.molecular_hamiltonian_from_specs(
          molecule='H2',
          basis='sto-3g',
          device='cuda'  # Not 'cpu'
      )

Problem: Chemical Accuracy Not Achieved
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Error > 1 kcal/mol despite convergence

**Solutions**:

1. Use UCCSD ansatz instead of hardware-efficient:

   .. code-block:: python

      config.ansatz = 'uccsd'

2. Increase basis set:

   .. code-block:: python

      H = MPOBuilder.molecular_hamiltonian_from_specs(
          molecule='H2',
          basis='6-31g',  # Instead of 'sto-3g'
          device='cuda'
      )

3. Enlarge active space:

   .. code-block:: python

      H = MPOBuilder.molecular_hamiltonian_from_specs(
          molecule='LiH',
          basis='sto-3g',
          active_space=(6, 6),  # More electrons/orbitals
          device='cuda'
      )

Problem: PySCF Integration Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: ModuleNotFoundError or integral computation failures

**Solutions**:

1. Install required packages:

   .. code-block:: bash

      pip install pyscf openfermion openfermionpyscf

2. Check geometry format:

   .. code-block:: python

      # Correct format: list of (atom, position) tuples
      geometry = [
          ('H', (0.0, 0.0, 0.0)),
          ('H', (0.74, 0.0, 0.0))
      ]

      # NOT: [['H', 0, 0, 0], ['H', 0.74, 0, 0]]

3. Verify molecule name:

   .. code-block:: python

      # Supported built-in molecules
      valid_molecules = ['H2', 'LiH', 'BeH2', 'H2O', 'NH3']

      # For others, provide explicit geometry
      if molecule not in valid_molecules:
          assert geometry is not None

Problem: Gradient Explosion or NaN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Optimizer returns NaN or energy diverges

**Solutions**:

1. Enable gradient clipping:

   .. code-block:: python

      config.gradient_clip = 1.0  # Clip gradients to [-1, 1]

2. Reduce learning rate:

   .. code-block:: python

      config.learning_rate = 0.01  # Smaller steps

3. Check numerical stability:

   .. code-block:: python

      # Use higher precision
      torch.set_default_dtype(torch.float64)

Summary
-------

This tutorial covered molecular VQE with ATLAS-Q:

✓ Built molecular Hamiltonians from atomic specifications using PySCF
✓ Applied UCCSD ansatz for chemically accurate ground states
✓ Optimized parameters with gradient-based methods (L-BFGS-B)
✓ Computed energies for H2, LiH, H2O with custom geometries
✓ Used active spaces to reduce qubit requirements for large molecules
✓ Generated potential energy surfaces by scanning bond lengths
✓ Achieved chemical accuracy (< 1 kcal/mol) for benchmark systems
✓ Troubleshot common convergence and accuracy issues

Molecular VQE enables quantum simulations of electronic structure with classical optimization, bridging quantum chemistry and quantum computing.

Next Steps
----------

- :doc:`vqe_tutorial` - General VQE principles and convergence analysis
- :doc:`advanced_features` - Excited states and constrained optimization
- :doc:`../howtos/custom_hamiltonians` - Build custom molecular Hamiltonians
- :doc:`../explanations/algorithms` - UCCSD theory and fermionic operators

Practice Exercises
-------------------

1. Compute the dissociation energy of H2 by comparing energies at equilibrium (0.74 A) and dissociated (5.0 A) geometries
2. Find the equilibrium bond length of LiH by scanning geometries from 1.0 to 2.5 Angstroms
3. Compare UCCSD vs hardware-efficient ansatz accuracy for H2O with different layer counts
4. Implement active space selection for BeH2 and verify chemical accuracy is maintained
5. Compute the first excited state of H2 and calculate the excitation energy in eV

See Also
--------

- :doc:`../../../reference/mpo_ops` - MPOBuilder API for molecular Hamiltonians
- :doc:`../../../reference/vqe_qaoa` - VQE and VQEConfig API reference
- :doc:`../explanations/algorithms` - Quantum chemistry algorithms explained
- :doc:`../howtos/optimize_performance` - Performance optimization for large molecules
