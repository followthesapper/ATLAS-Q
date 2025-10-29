Examples
========

Code examples demonstrating ATLAS-Q capabilities. All examples are tested and runnable.

.. toctree::
   :maxdepth: 2

Example Categories
------------------

Basic Usage
^^^^^^^^^^^

- **MPS Basics**: Creating and manipulating Matrix Product States
- **Gate Application**: Single-qubit and two-qubit gate operations
- **Measurement**: Sampling from quantum states

Algorithms
^^^^^^^^^^

- **VQE for Ground States**: Finding ground state energies using variational optimization
- **QAOA for MaxCut**: Solving graph optimization problems
- **TDVP Time Evolution**: Real-time quantum dynamics
- **Imaginary Time Evolution**: Ground state preparation

Quantum Chemistry
^^^^^^^^^^^^^^^^^

- **Molecular Hamiltonians**: Building Hamiltonians from molecular geometries
- **VQE for H2**: Computing H2 ground state energy
- **UCCSD Ansatz**: Unitary coupled-cluster ansatz for chemistry
- **Active Space Calculations**: Reducing problem size for large molecules

Advanced Features
^^^^^^^^^^^^^^^^^

- **Circuit Cutting**: Partitioning large circuits
- **PEPS Simulation**: 2D tensor network methods
- **Stabilizer Circuits**: Efficient Clifford simulation
- **Distributed MPS**: Multi-GPU parallelization

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

- **GPU Acceleration**: Using Triton kernels and cuQuantum
- **Memory Management**: Bond dimension control and budgets
- **Mixed Precision**: Balancing accuracy and performance
- **Adaptive Truncation**: Dynamic bond dimension selection

Complete Examples
-----------------

Interactive Notebook
^^^^^^^^^^^^^^^^^^^^

The comprehensive Jupyter notebook demonstrating all major features:

`ATLAS_Q_Demo.ipynb <https://github.com/followthesapper/ATLAS-Q/blob/ATLAS-Q/ATLAS_Q_Demo.ipynb>`_

Run in Google Colab:

`Open in Colab <https://colab.research.google.com/github/followthesapper/ATLAS-Q/blob/ATLAS-Q/ATLAS_Q_Demo.ipynb>`_

Benchmark Scripts
^^^^^^^^^^^^^^^^^

Located in ``scripts/benchmarks/``:

- ``validate_all_features.py`` - Comprehensive feature validation (46 tests)
- ``compare_with_competitors.py`` - Performance comparison with Qiskit, Cirq
- ``max_qubits_scaling_test.py`` - Maximum qubit capacity testing

Run all benchmarks:

.. code-block:: bash

   python scripts/benchmarks/validate_all_features.py

Example: MPS Basics
-------------------

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create MPS
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Apply Hadamard to all qubits
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
   for q in range(10):
       mps.apply_single_qubit_gate(q, H.to('cuda'))

   # Apply CNOT chain
   CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                        dtype=torch.complex64).reshape(4,4).to('cuda')
   for q in range(9):
       mps.apply_two_site_gate(q, CNOT)

   # Sample measurements
   samples = mps.sample(num_shots=1000)
   print(f"Sampled {len(samples)} measurements")

   # Check statistics
   stats = mps.stats_summary()
   print(f"Max bond dimension: {stats['max_chi']}")
   print(f"Memory: {mps.memory_usage() / (1024**2):.2f} MB")

Example: VQE for Ising Model
-----------------------------

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build Hamiltonian
   H = MPOBuilder.ising_hamiltonian(n_sites=8, J=1.0, h=0.5, device='cuda')

   # Run VQE
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       optimizer='L-BFGS-B',
       max_iter=100,
       device='cuda'
   )
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"Ground state energy: {energy:.6f}")
   print(f"Converged in {len(vqe.energies)} iterations")

Example: TDVP Quantum Quench
-----------------------------

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.tdvp import TDVP1Site, TDVPConfig

   # Initial Hamiltonian
   H_initial = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0, device='cuda')

   # Prepare ground state (assume prepared via VQE or imaginary time)
   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   # Quench to new Hamiltonian
   H_final = MPOBuilder.ising_hamiltonian(10, J=1.0, h=2.0, device='cuda')

   # Evolve
   config = TDVPConfig(dt=0.01, t_final=5.0)
   tdvp = TDVP1Site(H_final, mps, config)
   times, energies = tdvp.run()

   # Plot energy dynamics
   import matplotlib.pyplot as plt
   plt.plot(times, energies.real)
   plt.xlabel('Time')
   plt.ylabel('Energy')
   plt.title('Quantum Quench Dynamics')
   plt.savefig('quench_dynamics.png')

Example: Molecular VQE for H2
------------------------------

Requires PySCF: ``pip install pyscf openfermion openfermionpyscf``

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build H2 Hamiltonian
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       geometry=None,  # Uses default geometry
       device='cuda'
   )

   # Run VQE with hardware-efficient ansatz
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       max_iter=100,
       device='cuda'
   )
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"H2 ground state energy: {energy:.6f} Ha")

   # Experimental value: -1.137 Ha

Example: QAOA for MaxCut
-------------------------

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx

   # Create graph
   G = nx.Graph()
   G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

   # Build Hamiltonian
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Run QAOA
   config = VQEConfig(device='cuda')
   qaoa = QAOA(H, p=2, config=config)
   energy, params = qaoa.optimize()

   print(f"MaxCut value: {-energy:.2f}")

Example: Stabilizer Simulation
-------------------------------

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # Simulate 100-qubit Clifford circuit (efficient!)
   sim = StabilizerSimulator(n_qubits=100)

   # Create Bell pairs
   for i in range(0, 100, 2):
       sim.h(i)
       sim.cnot(i, i+1)

   # Measure
   outcomes = [sim.measure(i) for i in range(100)]
   print(f"Measurement outcomes: {outcomes[:10]}...")  # First 10

Example: Circuit Cutting
-------------------------

.. code-block:: python

   from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig

   # Define circuit
   gates = [
       ('H', [0], []),
       ('H', [1], []),
       ('CNOT', [0, 1], []),
       ('CNOT', [1, 2], []),
       ('CNOT', [2, 3], []),
       ('CNOT', [3, 4], []),
   ]

   # Configure cutting
   config = CuttingConfig(max_partition_size=3, max_cuts=2)
   cutter = CircuitCutter(config)

   # Partition circuit
   partitions = cutter.partition_circuit(gates, n_partitions=2)

   print(f"Created {len(partitions)} partitions")
   for i, partition in enumerate(partitions):
       print(f"Partition {i}: {len(partition.qubits)} qubits, {len(partition.gates)} gates")

More Examples
-------------

Additional examples are in the repository:

- ``scripts/demos/`` - Demonstration scripts
- ``scripts/ai_tools/`` - AI-guided truncation examples
- ``tests/integration/`` - Integration test examples

For interactive examples, see the Jupyter notebook.

Contributing Examples
---------------------

To contribute examples:

1. Add example to appropriate category directory
2. Include docstring with description and expected output
3. Test example works on both CPU and GPU
4. Add to this index with brief description
5. Submit pull request

See :doc:`../developer/contributing` for contribution guidelines.
