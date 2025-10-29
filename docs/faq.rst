Frequently Asked Questions
==========================

Installation and Setup
----------------------

How do I install ATLAS-Q?
^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended method is via PyPI:

.. code-block:: bash

   pip install atlas-quantum[gpu]

See :doc:`installation` for other methods (Docker, source, Conda).

Does ATLAS-Q require a GPU?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. ATLAS-Q works on CPU, but GPU acceleration provides significant speedup (1.5-3× for tensor operations, 100-1000× for period-finding). CPU fallback is automatic.

Which GPUs are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA GPUs with CUDA support (compute capability 7.0+):

- Tested: V100, A100, H100, RTX 4090
- Recommended: A100 or newer

AMD GPUs are not currently supported.

How do I verify my installation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run:

.. code-block:: python

   from atlas_q import get_quantum_sim
   QCH, _, _, _ = get_quantum_sim()
   sim = QCH()
   factors = sim.factor_number(21)
   assert factors[0] * factors[1] == 21
   print("Installation verified")

Usage and Configuration
-----------------------

What is bond dimension and how do I choose it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bond dimension (χ) controls MPS expressiveness and memory usage. Higher χ captures more entanglement but requires more memory and computation.

Guidelines:

- Product states: χ=1
- Weakly entangled: χ=8-32
- Moderately entangled: χ=64-256
- Highly entangled: χ=512+

Start small (χ=16) and increase if accuracy is insufficient. ATLAS-Q's adaptive truncation automatically manages χ within bounds.

How many qubits can ATLAS-Q simulate?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depends on entanglement structure:

- Clifford circuits (stabilizer backend): 1000+ qubits
- Product states: 1000+ qubits
- MPS (moderate entanglement, χ=64): 50-100 qubits per GPU
- MPS (high entanglement, χ=256): 20-50 qubits per GPU
- Multi-GPU: 100s-1000s of qubits

Memory usage scales as O(n·χ²) for n qubits.

When should I use ATLAS-Q vs Qiskit?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ATLAS-Q when:

- Simulating 20+ qubits with moderate entanglement
- Memory is limited
- GPU acceleration is available
- Need tensor network methods (TDVP, VQE, QAOA)

Use Qiskit when:

- Need circuit visualization
- Integrating with IBM quantum hardware
- Require extensive gate library
- Prefer mature ecosystem

ATLAS-Q and Qiskit can be used together.

Algorithms and Methods
----------------------

What is the difference between VQE and QAOA?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQE (Variational Quantum Eigensolver):

- Finds ground states of Hamiltonians
- Applications: quantum chemistry, condensed matter physics
- Ansatz: problem-specific (UCCSD, hardware-efficient)

QAOA (Quantum Approximate Optimization Algorithm):

- Solves combinatorial optimization problems
- Applications: MaxCut, TSP, scheduling
- Ansatz: alternating problem/mixer Hamiltonians

Both use variational optimization but target different problem classes.

When should I use 1-site vs 2-site TDVP?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1-site TDVP:

- Faster (no SVD)
- Conserves bond dimension
- Use when: short evolution times, limited entanglement growth

2-site TDVP:

- More accurate
- Allows bond dimension growth
- Use when: long evolution times, significant entanglement growth

Should I use hardware-efficient or UCCSD ansatz for VQE?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hardware-Efficient:

- Fewer parameters
- Faster optimization
- Use when: approximate ground state sufficient, large systems

UCCSD (Unitary Coupled-Cluster):

- More parameters
- Higher accuracy for molecular systems
- Use when: chemical accuracy required, small molecules (≤10 orbitals)

Performance and Optimization
-----------------------------

Why is my simulation slow?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Common causes:

1. **High bond dimension**: Reduce χ or use adaptive truncation
2. **CPU-only**: Install CUDA and Triton for GPU acceleration
3. **Large system**: Consider distributed MPS or circuit cutting
4. **Tight tolerances**: Relax ``eps_bond`` parameter
5. **Dense Hamiltonians**: Use sparse MPO representation

See :doc:`user_guide/howtos/optimize_performance` for optimization strategies.

How do I reduce memory usage?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Strategies:

1. Reduce bond dimension: ``chi_max_per_bond=64``
2. Enable memory budgets: ``budget_global_mb=4096``
3. Use mixed precision: ``dtype=torch.complex64``
4. Apply adaptive truncation: ``eps_bond=1e-6``
5. Use stabilizer backend for Clifford circuits

See :doc:`user_guide/howtos/handle_large_systems`.

Can ATLAS-Q use multiple GPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Use ``distributed_mps`` module:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig

   config = DistributedConfig(mode='bond_parallel', world_size=4)
   mps = DistributedMPS(num_qubits=100, bond_dim=16, config=config)

Requires NCCL and ``torch.distributed``.

Errors and Debugging
--------------------

What does "SVD did not converge" mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This indicates numerical instability in singular value decomposition. Solutions:

1. Use robust SVD: ATLAS-Q does this automatically
2. Increase precision: ``dtype=torch.complex128``
3. Add regularization: enabled by default in ``linalg_robust``
4. Reduce condition number: check ``diagnostics.condition_number``

ATLAS-Q handles this via automatic CPU fallback and Tikhonov regularization.

Why are my energies not conserved in TDVP?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Causes:

1. **Time step too large**: Reduce ``dt``
2. **Truncation error**: Increase ``chi_max`` or tighten ``eps_bond``
3. **Non-Hermitian Hamiltonian**: Verify Hamiltonian construction
4. **Numerical drift**: Use higher precision (``torch.complex128``)

Check energy conservation:

.. code-block:: python

   times, energies = tdvp.run()
   energy_drift = (energies[-1] - energies[0]) / energies[0]
   print(f"Relative energy drift: {energy_drift.real:.2e}")

How do I debug VQE convergence issues?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Monitor optimization**:

   .. code-block:: python

      vqe = VQE(H, config)
      energy, params = vqe.optimize()
      import matplotlib.pyplot as plt
      plt.plot(vqe.energies)
      plt.savefig('convergence.png')

2. **Check gradients**: Use ``gradient_method='per_pauli'``

3. **Try different optimizers**: L-BFGS-B, COBYLA, BFGS

4. **Increase ansatz depth**: More layers may be needed

5. **Verify Hamiltonian**: Check energy bounds are reasonable

Molecular Chemistry
-------------------

How do I compute molecular energies?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Requires PySCF:

.. code-block:: bash

   pip install pyscf openfermion openfermionpyscf

Then:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       device='cuda'
   )
   config = VQEConfig(ansatz='hardware_efficient', n_layers=3)
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

What molecules can ATLAS-Q simulate?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any molecule supported by PySCF. Practical limits:

- Small molecules (H2, LiH, H2O): Up to ~20 qubits (10 orbitals)
- Medium molecules: Limited by MPS bond dimension and entanglement
- Active space approximations recommended for large systems

See :doc:`user_guide/tutorials/molecular_vqe` for examples.

Advanced Features
-----------------

What is circuit cutting and when should I use it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Circuit cutting partitions large circuits into smaller subcircuits that are simulated independently and classically stitched together.

Use when:

- Circuit exceeds connectivity limits of MPS (highly non-local gates)
- Need to simulate beyond single-GPU memory
- Circuit has natural partition structure

Trade-off: Increases classical sampling overhead.

See :doc:`user_guide/howtos/integrate_cuquantum` and :doc:`reference/circuit_cutting`.

How do I use cuQuantum acceleration?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install cuQuantum:

.. code-block:: bash

   pip install cuquantum-python

ATLAS-Q automatically detects and uses cuQuantum if available. Manual configuration:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend, CuQuantumConfig

   config = CuQuantumConfig(use_cutensornet=True, workspace_size=1024**3)
   backend = CuQuantumBackend(config)
   U, S, Vdagger = backend.svd(tensor, chi_max=128)

Provides 2-10× speedup on compatible operations.

General Questions
-----------------

Is ATLAS-Q production-ready?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Version 0.6.1 has 46/46 integration tests passing, comprehensive error handling, and has been used in research. Suitable for academic and industrial applications.

How do I cite ATLAS-Q?
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @software{atlasq2025,
     title={ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum},
     author={ATLAS-Q Development Team},
     year={2025},
     url={https://github.com/followthsapper/ATLAS-Q},
     version={0.6.1}
   }

See :doc:`citing` for other formats.

Where can I get help?
^^^^^^^^^^^^^^^^^^^^^

- GitHub Issues: https://github.com/followthesapper/ATLAS-Q/issues
- GitHub Discussions: https://github.com/followthesapper/ATLAS-Q/discussions
- Documentation: https://followthsapper.github.io/ATLAS-Q/

For bug reports, include:

1. ATLAS-Q version
2. Python version
3. PyTorch version
4. GPU model (if applicable)
5. Minimal reproduction code

Can I contribute to ATLAS-Q?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. Contributions are welcome. See :doc:`developer/contributing` for guidelines.

Areas needing contribution:

- Additional examples and tutorials
- Integration with other frameworks (Qiskit, Cirq)
- Performance optimization
- Documentation improvements

Further Reading
---------------

- :doc:`quickstart` - Quick start guide
- :doc:`user_guide/tutorials/index` - Tutorials
- :doc:`user_guide/howtos/index` - How-to guides
- :doc:`reference/index` - API reference
