Quick Start
===========

This guide demonstrates basic ATLAS-Q usage. For installation instructions, see :doc:`installation`.

Verification
------------

After installation, verify ATLAS-Q is working:

.. code-block:: python

   import atlas_q
   print(atlas_q.__version__)  # Should print: 0.6.1

   from atlas_q import get_quantum_sim
   print("Installation verified")

Basic Usage
-----------

Factorization with Period-Finding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q import get_quantum_sim

   QCH, _, _, _ = get_quantum_sim()
   sim = QCH()
   factors = sim.factor_number(221)
   print(f"221 = {factors[0]} × {factors[1]}")  # Output: 221 = 13 × 17

Tensor Network Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an MPS and apply quantum gates:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create 10-qubit MPS on GPU
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Define Hadamard gate
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
   H = H.to('cuda')

   # Apply Hadamard to all qubits
   for q in range(10):
       mps.apply_single_qubit_gate(q, H)

   # Define CZ gate
   CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64)).to('cuda')

   # Apply CZ to adjacent pairs
   for q in range(0, 9, 2):
       mps.apply_two_site_gate(q, CZ)

   # Check statistics
   stats = mps.stats_summary()
   print(f"Maximum bond dimension: {stats['max_chi']}")
   print(f"Memory usage: {mps.memory_usage() / (1024**2):.2f} MB")

VQE for Ground State Finding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build Hamiltonian (6-site Ising model)
   H = MPOBuilder.ising_hamiltonian(n_sites=6, J=1.0, h=0.5, device='cuda')

   # Configure VQE
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       max_iter=50,
       device='cuda'
   )

   # Run optimization
   vqe = VQE(H, config)
   energy, params = vqe.optimize()
   print(f"Ground state energy: {energy:.6f}")

TDVP Time Evolution
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.tdvp import TDVP1Site, TDVPConfig

   # Create Hamiltonian and initial state
   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Configure time evolution
   config = TDVPConfig(dt=0.01, t_final=1.0, use_gpu_optimized=True)
   tdvp = TDVP1Site(H, mps, config)

   # Run evolution
   times, energies = tdvp.run()
   print(f"Final energy: {energies[-1]:.6f}")

Stabilizer Simulation (Clifford Circuits)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # Create 100-qubit simulator (efficient for Clifford gates)
   sim = StabilizerSimulator(n_qubits=100)

   # Apply Clifford gates
   sim.h(0)
   sim.cnot(0, 1)
   sim.s(1)
   sim.cz(1, 2)

   # Measure qubit
   outcome = sim.measure(0)
   print(f"Measurement outcome: {outcome}")

Molecular VQE with PySCF
^^^^^^^^^^^^^^^^^^^^^^^^^

Requires PySCF installation: ``pip install pyscf openfermion openfermionpyscf``

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build molecular Hamiltonian for H2
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       geometry=None,  # Uses default H2 geometry
       device='cuda'
   )

   # Run VQE
   config = VQEConfig(ansatz='hardware_efficient', n_layers=3, max_iter=100)
   vqe = VQE(H, config)
   energy, params = vqe.optimize()
   print(f"H2 ground state energy: {energy:.6f} Ha")

QAOA for MaxCut
^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx

   # Create graph
   G = nx.Graph()
   G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

   # Build MaxCut Hamiltonian
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Run QAOA
   config = VQEConfig(ansatz='hardware_efficient', n_layers=2, device='cuda')
   qaoa = QAOA(H, p=2, config=config)
   energy, params = qaoa.optimize()
   print(f"MaxCut value: {-energy:.2f}")

Circuit Cutting
^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig

   # Configure circuit cutting
   config = CuttingConfig(max_partition_size=4, max_cuts=2)
   cutter = CircuitCutter(config)

   # Define circuit as list of gates
   gates = [
       ('H', [0], []),
       ('H', [1], []),
       ('CNOT', [0, 1], []),
       ('CNOT', [1, 2], []),
       ('CNOT', [2, 3], []),
   ]

   # Analyze and partition
   graph = cutter.analyze_circuit(gates)
   partitions = cutter.partition_circuit(gates, n_partitions=2)
   print(f"Created {len(partitions)} partitions")

PEPS for 2D Circuits
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.peps import PatchPEPS

   # Create 4×4 PEPS patch
   patch = PatchPEPS(patch_size=4, device='cuda')

   # Define 2D circuit (row, col coordinates)
   gates = [
       ('H', [(0, 0)], []),
       ('H', [(0, 1)], []),
       ('CZ', [(0, 0), (0, 1)], []),
       ('CZ', [(0, 1), (1, 1)], []),
   ]

   # Apply circuit
   patch.apply_shallow_circuit(gates)
   norm = patch.peps.compute_norm()
   print(f"State norm: {norm:.6f}")

Command-Line Interface
----------------------

ATLAS-Q provides a CLI for common operations:

.. code-block:: bash

   # Show help
   python -m atlas_q --help

   # Show version
   python -m atlas_q --version

   # Factor a number
   python -m atlas_q factor 221

   # Run benchmarks
   python -m atlas_q benchmark

   # Show system info
   python -m atlas_q info

   # Interactive demo
   python -m atlas_q demo

Configuration
-------------

Device Selection
^^^^^^^^^^^^^^^^

Specify computation device:

.. code-block:: python

   # Use GPU (default if available)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Use CPU
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cpu')

   # Specify GPU device
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda:1')

Precision Control
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy

   # Use complex64 (default, faster)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, dtype=torch.complex64)

   # Use complex128 (higher precision)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, dtype=torch.complex128)

   # Mixed precision with automatic promotion
   policy = DTypePolicy(default=torch.complex64, promote_if_cond_gt=1e6)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, dtype_policy=policy)

Memory Budgets
^^^^^^^^^^^^^^

.. code-block:: python

   # Enforce global memory limit (in MB)
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       budget_global_mb=4096,  # 4GB limit
       device='cuda'
   )

   # Per-bond dimension caps
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       chi_max_per_bond=128,  # Maximum χ=128 per bond
       device='cuda'
   )

Error Tolerance
^^^^^^^^^^^^^^^

.. code-block:: python

   # Set truncation tolerance
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       eps_bond=1e-8,  # Tighter tolerance (higher accuracy)
       device='cuda'
   )

   # Check global error bound
   global_error = mps.global_error_bound()
   print(f"Global truncation error: {global_error:.2e}")

Next Steps
----------

- :doc:`user_guide/tutorials/index` - Comprehensive tutorials
- :doc:`user_guide/howtos/index` - Task-oriented guides
- :doc:`reference/index` - Complete API reference
- :doc:`examples/index` - Example gallery

For detailed explanations of concepts, see :doc:`user_guide/explanations/index`.
