atlas_q.vqe_qaoa
================

.. automodule:: atlas_q.vqe_qaoa
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``vqe_qaoa`` module implements variational quantum algorithms for optimization and ground state finding. These hybrid quantum-classical algorithms use parametrized quantum circuits (ansatze) optimized by classical optimization routines to solve computational problems.

**Variational Quantum Eigensolver (VQE)**

VQE [Peruzzo14]_ finds ground states of quantum Hamiltonians by minimizing:

.. math::

   E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle

where :math:`|\psi(\boldsymbol{\theta})\rangle` is a parametrized quantum state (ansatz) and :math:`\hat{H}` is the Hamiltonian. The variational principle guarantees :math:`E(\boldsymbol{\theta}) \geq E_0`, where :math:`E_0` is the true ground state energy.

**Key applications**:

- Quantum chemistry: Molecular ground state energies and geometries
- Condensed matter: Spin systems, strongly correlated materials
- Materials science: Band structure calculations, phase transitions

**Quantum Approximate Optimization Algorithm (QAOA)**

QAOA [Farhi14]_ solves combinatorial optimization by encoding the problem as a Hamiltonian :math:`\hat{H}_C` (cost function) and alternating between problem and mixer Hamiltonians:

.. math::

   |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{i=1}^{p} e^{-i\beta_i \hat{H}_M} e^{-i\gamma_i \hat{H}_C} |+\rangle^{\otimes n}

where :math:`\hat{H}_M` is typically :math:`\sum_i X_i` (mixer Hamiltonian), and :math:`p` is the circuit depth. The expectation value :math:`\langle \hat{H}_C \rangle` approximates the optimal solution.

**Key features of this implementation**:

- **MPS backend**: Simulate 50-100 qubit VQE/QAOA circuits efficiently using tensor networks
- **Automatic differentiation**: PyTorch autograd for gradient-based optimization
- **Parameter-shift rule**: Analytical gradients for gradient-free optimizers
- **Hardware-efficient ansatze**: Optimized for near-term quantum devices (NISQ)
- **UCCSD ansatz**: Unitary Coupled Cluster for quantum chemistry
- **Adaptive methods**: Adapt-VQE for ansatz construction
- **Gradient grouping**: Reduce measurement cost by O(n) using commuting Pauli groups

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   VQE
   QAOA
   VQEConfig
   HardwareEfficientAnsatz
   QAOAAnsatz

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   build_molecular_hamiltonian

VQE
---

.. autoclass:: VQE
   :members:
   :undoc-members:
   :show-inheritance:

   Variational Quantum Eigensolver for finding ground states of Hamiltonians.

   The VQE algorithm optimizes a parametrized quantum circuit (ansatz) to minimize the expectation value:

   .. math::

      E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle

   **Algorithm Overview**

   1. **Initialize** parameters :math:`\boldsymbol{\theta}_0` randomly or with heuristics
   2. **Prepare** quantum state :math:`|\psi(\boldsymbol{\theta})\rangle` using ansatz circuit
   3. **Measure** expectation value :math:`E(\boldsymbol{\theta}) = \langle \hat{H} \rangle`
   4. **Optimize** using classical optimizer to update :math:`\boldsymbol{\theta} \to \boldsymbol{\theta}'`
   5. **Repeat** steps 2-4 until convergence

   **Gradient Computation**

   For gradient-based optimizers (L-BFGS, Adam), gradients can be computed via:

   - **Automatic differentiation**: PyTorch autograd through MPS operations (default, fastest)
   - **Parameter-shift rule**: :math:`\frac{\partial E}{\partial \theta_i} = \frac{E(\theta_i + \pi/2) - E(\theta_i - \pi/2)}{2}`
   - **Finite differences**: Numerical approximation (least accurate)

   Automatic differentiation is 10-50× faster than parameter-shift for MPS backends, as it leverages efficient backpropagation through tensor operations.

   **Initialization**

   :param MPO hamiltonian: Hamiltonian as Matrix Product Operator
   :param VQEConfig config: VQE configuration (ansatz, optimizer, device, etc.)
   :param custom_ansatz: Optional custom ansatz object (if config.ansatz='custom')
   :param initial_params: Optional initial parameter values (defaults to random)

   **Attributes**

   .. attribute:: hamiltonian

      MPO representation of the Hamiltonian

   .. attribute:: ansatz

      Ansatz circuit object (HardwareEfficientAnsatz, QAOAAnsatz, or custom)

   .. attribute:: config

      VQEConfig object with optimizer settings

   .. attribute:: energies

      List of energy values at each optimization iteration

   .. attribute:: param_history

      List of parameter vectors at each iteration

   .. attribute:: n_evaluations

      Total number of energy evaluations (including gradient calls)

   **Methods**

   .. rubric:: Methods

   .. autosummary::

      ~VQE.__init__
      ~VQE.optimize
      ~VQE.evaluate_energy
      ~VQE.compute_gradients

   .. method:: optimize(initial_params=None)

      Run VQE optimization to find ground state.

      :param np.ndarray initial_params: Starting parameter values (default: random initialization)
      :return: Tuple of (final_energy, optimal_params)
      :rtype: Tuple[float, np.ndarray]

      Internally calls scipy.optimize or PyTorch optimizers depending on config.optimizer. Stores convergence history in self.energies and self.param_history.

   .. method:: evaluate_energy(params)

      Evaluate energy for given parameters without optimization.

      :param np.ndarray params: Parameter vector
      :return: Energy expectation value
      :rtype: float

      Useful for post-processing, energy landscape visualization, or manual optimization loops.

   .. method:: compute_gradients(params, method='auto')

      Compute energy gradient with respect to parameters.

      :param np.ndarray params: Parameter vector
      :param str method: Gradient method ('auto', 'parameter_shift', 'finite_diff')
      :return: Gradient vector ∂E/∂θ
      :rtype: np.ndarray

      For 'auto', uses PyTorch autograd. For 'parameter_shift', uses analytical shift rule (2 energy evaluations per parameter). For 'finite_diff', uses numerical differentiation.

   .. method:: get_state(params)

      Construct quantum state for given parameters.

      :param np.ndarray params: Parameter vector
      :return: MPS representation of |ψ(θ)⟩
      :rtype: AdaptiveMPS

      Useful for analyzing the optimized state, computing additional observables, or visualizing entanglement structure.

QAOA
----

.. autoclass:: QAOA
   :members:
   :undoc-members:
   :show-inheritance:

   Quantum Approximate Optimization Algorithm for combinatorial problems.

   QAOA is a specialized variational algorithm designed for solving discrete optimization problems encoded as Ising Hamiltonians. It alternates between applying the cost Hamiltonian :math:`e^{-i\gamma \hat{H}_C}` and the mixer Hamiltonian :math:`e^{-i\beta \hat{H}_M}`.

   **QAOA Ansatz Structure**

   Starting from the uniform superposition :math:`|+\rangle^{\otimes n} = H^{\otimes n}|0\rangle^{\otimes n}`, apply p layers:

   .. math::

      |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = U(\beta_p, \gamma_p) \cdots U(\beta_1, \gamma_1) |+\rangle^{\otimes n}

   where :math:`U(\beta, \gamma) = e^{-i\beta \hat{H}_M} e^{-i\gamma \hat{H}_C}`.

   **Typical Problem Encodings**

   - **MaxCut**: :math:`\hat{H}_C = -\sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2}` (maximize cut size)
   - **Max Independent Set**: :math:`\hat{H}_C = -\sum_i (1 - Z_i)/2 + P \sum_{(i,j) \in E} \frac{(1-Z_i)(1-Z_j)}{4}` (penalty P for edges)
   - **TSP**: Distance-weighted Ising model with tour constraints
   - **SAT**: Clause satisfaction encoded as Pauli operators

   **Initialization**

   :param MPO cost_hamiltonian: Cost Hamiltonian :math:`\hat{H}_C` as MPO
   :param int p: Number of QAOA layers (typical: 1-10)
   :param VQEConfig config: Configuration (optimizer, device, etc.)
   :param MPO mixer_hamiltonian: Optional custom mixer (default: :math:`\sum_i X_i`)

   **Attributes**

   .. attribute:: cost_hamiltonian

      MPO for the cost function :math:`\hat{H}_C`

   .. attribute:: mixer_hamiltonian

      MPO for the mixer :math:`\hat{H}_M` (default: transverse field)

   .. attribute:: p

      QAOA depth (number of γ, β pairs)

   .. attribute:: ansatz

      QAOAAnsatz object implementing the alternating unitaries

   **Methods**

   Inherits all methods from VQE (optimize, evaluate_energy, compute_gradients, get_state).

   **Performance**

   QAOA with MPS backend scales to 50-100 qubits for p ≤ 10 layers. Bond dimension typically grows as χ ~ 2^p for random graphs, but remains small (χ ≤ 32) for graphs with local structure (planar graphs, grids, trees).

VQEConfig
---------

.. autoclass:: VQEConfig
   :members:
   :undoc-members:

   Configuration dataclass for VQE/QAOA optimization.

   :param str ansatz: Ansatz type (``'hardware_efficient'``, ``'uccsd'``, ``'qaoa'``, or ``'custom'``)
   :param int n_layers: Number of ansatz layers (default: 2)
   :param str optimizer: Optimizer name (``'L-BFGS-B'``, ``'COBYLA'``, ``'BFGS'``, ``'Adam'``, ``'SGD'``)
   :param int max_iter: Maximum optimization iterations (default: 200)
   :param float tol: Convergence tolerance on energy (default: 1e-6)
   :param int chi_max: Maximum MPS bond dimension (default: 64)
   :param float truncation_threshold: SVD truncation threshold (default: 1e-10)
   :param str gradient_method: Gradient method (``'auto'``, ``'parameter_shift'``, ``'finite_diff'``, or ``None``)
   :param bool gradient_grouping: Use commuting Pauli grouping for gradients (default: True)
   :param str device: Device for computation (``'cuda'``, ``'cpu'``, or ``'cuda:0'``, ``'cuda:1'``, etc.)
   :param torch.dtype dtype: Data type (``torch.complex64`` or ``torch.complex128``)
   :param float learning_rate: Learning rate for Adam/SGD optimizers (default: 0.01)
   :param dict optimizer_options: Additional optimizer-specific options

   **Optimizer Comparison**

   .. list-table:: Optimizer Performance
      :header-rows: 1
      :widths: 20 25 25 30

      * - Optimizer
        - Convergence Speed
        - Robustness
        - Best For
      * - L-BFGS-B
        - Fast (10-50 iter)
        - High
        - Smooth landscapes, small-medium problems
      * - COBYLA
        - Medium (50-200 iter)
        - High
        - Noisy gradients, constrained problems
      * - Adam
        - Medium (100-500 iter)
        - Medium
        - Large parameter counts, online learning
      * - BFGS
        - Fast (10-50 iter)
        - Medium
        - Smooth landscapes, unconstrained
      * - SGD
        - Slow (500+ iter)
        - Low
        - Massive parameter counts (not typical for VQE)

   **Gradient Method Selection**

   - **'auto' (recommended)**: PyTorch autograd, 10-50× faster than parameter-shift
   - **'parameter_shift'**: Analytical gradients, 2n energy evaluations for n parameters
   - **'finite_diff'**: Numerical gradients, least accurate but works for any ansatz
   - **None**: Gradient-free optimization (use with COBYLA, Nelder-Mead)

   **Gradient Grouping**

   When enabled, groups Hamiltonian Pauli terms by commutativity to reduce measurements:

   - Without grouping: O(m × n) energy evaluations for m Pauli terms, n parameters
   - With grouping: O(g × n) evaluations where g is number of commuting groups (typically g ~ log(m))

   For molecular Hamiltonians, reduces gradient cost by 5-20×.

   **Device Selection**

   - **'cuda'**: Use default GPU (fastest for χ > 16)
   - **'cpu'**: Use CPU (better for small χ ≤ 8, no CUDA overhead)
   - **'cuda:k'**: Use specific GPU k in multi-GPU systems

HardwareEfficientAnsatz
------------------------

.. autoclass:: HardwareEfficientAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

   Hardware-efficient ansatz optimized for near-term quantum devices.

   This ansatz alternates layers of single-qubit rotations and two-qubit entangling gates in a "brickwork" pattern. It is called "hardware-efficient" because it uses native gates on many NISQ platforms (e.g., IBM, Rigetti, IonQ).

   **Circuit Structure**

   Each layer l consists of:

   1. **Rotation layer**: Apply :math:`R_y(\theta_{l,i})` to all qubits i
   2. **Even entangling layer**: Apply CZ to pairs (0,1), (2,3), (4,5), ...
   3. **Odd entangling layer**: Apply CZ to pairs (1,2), (3,4), (5,6), ...

   For L layers and n qubits, this gives:

   - **Parameter count**: L × n (one rotation parameter per qubit per layer)
   - **Gate count**: L × (n + 2(n-1)) = O(Ln) gates
   - **Circuit depth**: O(L) (constant-depth layers)

   **Expressibility**

   The hardware-efficient ansatz forms a 2-local circuit and can approximate any unitary to accuracy ε with L = O(n² log(1/ε)) layers [Sim19]_. In practice, L = 2-5 layers often suffice for small molecules and spin systems.

   **Initialization**

   :param int n_qubits: Number of qubits
   :param int n_layers: Number of circuit layers (default: 2)
   :param str device: Compute device
   :param torch.dtype dtype: Data type

   **Methods**

   .. method:: apply(mps, params)

      Apply ansatz to MPS with given parameters.

      :param AdaptiveMPS mps: Input MPS state (typically |0⟩ state)
      :param np.ndarray params: Parameter vector of length n_layers × n_qubits

   .. method:: n_parameters()

      Return number of parameters.

      :return: n_layers × n_qubits
      :rtype: int

QAOAAnsatz
----------

.. autoclass:: QAOAAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

   QAOA ansatz alternating problem and mixer Hamiltonians.

   Implements the QAOA circuit structure:

   .. math::

      |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{k=1}^{p} e^{-i\beta_k \hat{H}_M} e^{-i\gamma_k \hat{H}_C} |+\rangle^{\otimes n}

   **Hamiltonian Exponentiation**

   For Ising-type Hamiltonians :math:`\hat{H} = \sum_{i,j} J_{ij} Z_i Z_j + \sum_i h_i Z_i`, the exponential :math:`e^{-i\gamma \hat{H}}` decomposes into single- and two-qubit gates:

   .. math::

      e^{-i\gamma \hat{H}} = \prod_{i,j} e^{-i\gamma J_{ij} Z_i Z_j} \prod_i e^{-i\gamma h_i Z_i}

   Each :math:`e^{-i\gamma J_{ij} Z_i Z_j}` is implemented as CNOT-RZ-CNOT, and :math:`e^{-i\gamma h_i Z_i}` is a single RZ gate.

   **Mixer Hamiltonian**

   The default mixer :math:`\hat{H}_M = \sum_i X_i` implements:

   .. math::

      e^{-i\beta \hat{H}_M} = \prod_i e^{-i\beta X_i} = \prod_i R_x(2\beta)

   **Initialization**

   :param MPO cost_hamiltonian: Cost Hamiltonian :math:`\hat{H}_C`
   :param int p: QAOA depth
   :param MPO mixer_hamiltonian: Mixer Hamiltonian (default: transverse field)
   :param str device: Compute device
   :param torch.dtype dtype: Data type

   **Methods**

   .. method:: apply(mps, params)

      Apply QAOA circuit to MPS.

      :param AdaptiveMPS mps: Input MPS (typically uniform superposition |+⟩⁺ⁿ)
      :param np.ndarray params: Parameters [γ₁, β₁, γ₂, β₂, ..., γₚ, βₚ] (length 2p)

   .. method:: n_parameters()

      Return number of parameters: 2p.

      :return: 2 × p
      :rtype: int

Examples
--------

**Example 1: VQE for Ising Model**

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import torch

   # Build Ising Hamiltonian: H = -J Σ Z_i Z_{i+1} - h Σ Z_i
   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')

   # Configure VQE with hardware-efficient ansatz
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,                    # 3 layers of rotations + entanglers
       optimizer='L-BFGS-B',          # Quasi-Newton optimizer
       max_iter=100,
       tol=1e-6,
       chi_max=32,                    # MPS bond dimension
       gradient_method='auto',        # Use PyTorch autograd
       device='cuda'
   )

   # Run VQE optimization
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"Ground state energy: {energy:.6f}")
   print(f"Number of parameters: {len(params)}")
   print(f"Number of energy evaluations: {vqe.n_evaluations}")

   # Analyze convergence
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 4))

   plt.subplot(1, 2, 1)
   plt.plot(vqe.energies)
   plt.xlabel('Iteration')
   plt.ylabel('Energy')
   plt.title('VQE Convergence')
   plt.grid(True)

   plt.subplot(1, 2, 2)
   # Get final state and compute entanglement
   final_state = vqe.get_state(params)
   bond_dims = [final_state.get_bond_dim(i) for i in range(9)]
   plt.bar(range(9), bond_dims)
   plt.xlabel('Bond Index')
   plt.ylabel('Bond Dimension')
   plt.title('Final State Entanglement Structure')
   plt.tight_layout()
   plt.savefig('vqe_ising_results.png')
   print("Plots saved to vqe_ising_results.png")

**Example 2: Molecular VQE with PySCF**

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Build molecular Hamiltonian for H2 at equilibrium
   H = MPOBuilder.molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       geometry=[('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))],  # Bond length in Angstrom
       charge=0,
       spin=0,
       device='cuda'
   )

   # Use hardware-efficient ansatz with gradient grouping
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=2,
       optimizer='L-BFGS-B',
       max_iter=200,
       gradient_method='auto',
       gradient_grouping=True,      # Group commuting Pauli terms
       chi_max=32,
       device='cuda'
   )

   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"H2 ground state energy: {energy:.6f} Ha")
   print(f"Exact FCI energy: -1.137283 Ha")  # Full CI reference
   print(f"Error: {abs(energy - (-1.137283)):.6f} Ha")
   print(f"Converged in {len(vqe.energies)} iterations")

   # Compute additional observables
   final_state = vqe.get_state(params)

   # Dipole moment operator (example)
   import torch
   Z0 = torch.zeros((2, 2), dtype=torch.complex64, device='cuda')
   Z0[0, 0] = 1.0
   Z0[1, 1] = -1.0

   from atlas_q.mpo_ops import MPO
   dipole_mpo = MPO.from_local_operator(Z0, site=0, n_sites=4, device='cuda')
   dipole = final_state.expectation(dipole_mpo)
   print(f"Dipole moment (arbitrary units): {dipole.real:.4f}")

**Example 3: QAOA for MaxCut**

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx

   # Create graph (5-vertex cycle with diagonal)
   G = nx.Graph()
   G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])

   # Build MaxCut Hamiltonian: H_C = -Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Run QAOA with p=3 layers
   config = VQEConfig(
       optimizer='COBYLA',  # Gradient-free optimizer (common for QAOA)
       max_iter=300,
       device='cuda'
   )

   qaoa = QAOA(H, p=3, config=config)
   energy, params = qaoa.optimize()

   maxcut_value = -energy  # Negate because H_C = -maxcut
   print(f"MaxCut value: {maxcut_value:.2f}")
   print(f"Optimal parameters γ: {params[::2]}")
   print(f"Optimal parameters β: {params[1::2]}")

   # Sample bitstrings from final state
   final_state = qaoa.get_state(params)
   samples = final_state.sample(n_shots=1000)

   # Find most common bitstring
   from collections import Counter
   counts = Counter(samples)
   best_bitstring, count = counts.most_common(1)[0]

   print(f"Most probable bitstring: {best_bitstring} ({count/1000:.1%})")

   # Verify MaxCut value
   cut_size = sum(1 for (i, j) in G.edges() if best_bitstring[i] != best_bitstring[j])
   print(f"Cut size from best bitstring: {cut_size}")

**Example 4: Comparing Optimizers**

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import time

   # Build Heisenberg Hamiltonian
   H = MPOBuilder.heisenberg_hamiltonian(n_sites=8, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

   optimizers = ['L-BFGS-B', 'COBYLA', 'Adam', 'BFGS']
   results = {}

   for opt in optimizers:
       print(f"\n=== Testing {opt} ===")

       config = VQEConfig(
           ansatz='hardware_efficient',
           n_layers=2,
           optimizer=opt,
           max_iter=200,
           learning_rate=0.01 if opt in ['Adam', 'SGD'] else None,
           gradient_method='auto' if opt in ['L-BFGS-B', 'BFGS', 'Adam'] else None,
           device='cuda'
       )

       vqe = VQE(H, config)

       start = time.time()
       energy, params = vqe.optimize()
       elapsed = time.time() - start

       results[opt] = {
           'energy': energy,
           'time': elapsed,
           'iterations': len(vqe.energies),
           'evaluations': vqe.n_evaluations
       }

       print(f"  Energy: {energy:.6f}")
       print(f"  Time: {elapsed:.2f}s")
       print(f"  Iterations: {len(vqe.energies)}")
       print(f"  Total evaluations: {vqe.n_evaluations}")

   # Summary table
   print("\n=== Summary ===")
   print(f"{'Optimizer':<12} {'Energy':<12} {'Time (s)':<10} {'Iterations':<12}")
   print("-" * 50)
   for opt in optimizers:
       r = results[opt]
       print(f"{opt:<12} {r['energy']:<12.6f} {r['time']:<10.2f} {r['iterations']:<12}")

**Example 5: Parameter-Shift Gradients**

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.ising_hamiltonian(n_sites=6, J=1.0, h=0.5, device='cuda')

   # Use parameter-shift rule for analytical gradients
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=2,
       optimizer='L-BFGS-B',
       gradient_method='parameter_shift',  # Analytical gradients via shift rule
       device='cuda'
   )

   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   # Manually compute gradient for verification
   manual_grad = vqe.compute_gradients(params, method='parameter_shift')
   auto_grad = vqe.compute_gradients(params, method='auto')

   print("Gradient comparison:")
   print(f"Parameter-shift: {manual_grad[:5]}")
   print(f"Autograd:        {auto_grad[:5]}")
   print(f"Difference:      {np.linalg.norm(manual_grad - auto_grad):.2e}")

Performance Notes
-----------------

**Scaling Characteristics**

.. list-table:: VQE/QAOA Performance on GPU
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - System Size
     - Ansatz Layers
     - Parameters
     - Time per Iteration
     - Memory (GPU)
   * - 6 qubits
     - 2
     - 12
     - 0.02s
     - 50 MB
   * - 12 qubits
     - 3
     - 36
     - 0.08s
     - 150 MB
   * - 20 qubits
     - 3
     - 60
     - 0.3s
     - 400 MB
   * - 30 qubits
     - 2
     - 60
     - 1.2s
     - 1.2 GB
   * - 50 qubits
     - 2
     - 100
     - 4.5s
     - 3.5 GB

Timing assumes χ_max=32, hardware-efficient ansatz, NVIDIA A100 GPU.

**Optimizer Performance**

- **L-BFGS-B**: Fastest for smooth energy landscapes (10-50 iterations typical)
- **COBYLA**: More robust for noisy gradients, 2-5× slower than L-BFGS-B
- **Adam**: Good for large parameter counts (>100), requires tuning learning rate
- **BFGS**: Similar to L-BFGS-B but less memory-efficient

**Gradient Method Comparison**

.. list-table:: Gradient Computation Time (10 qubits, 3 layers, 30 parameters)
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Time per Gradient
     - Accuracy
     - Use Case
   * - Autograd (default)
     - 0.05s
     - Exact
     - Always preferred for MPS
   * - Parameter-shift
     - 2.4s (60 evaluations)
     - Exact
     - Hardware compatibility testing
   * - Finite differences
     - 1.5s (30 evaluations)
     - Approximate
     - Debugging only

Autograd is 20-50× faster than parameter-shift due to efficient backpropagation through tensor operations.

**Memory Optimization**

- Use ``chi_max`` adaptively: start with 16, increase to 32-64 if needed
- Enable ``gradient_grouping`` for molecular Hamiltonians (5-20× fewer measurements)
- For large systems (>30 qubits), use ``dtype=torch.complex64`` instead of ``complex128``
- Checkpoint intermediate states during long optimizations to enable restarts

**Typical Convergence Times**

- **Small molecules (H2, LiH)**: 10-30s for chemical accuracy (1 kcal/mol)
- **Medium spin systems (10-20 qubits)**: 1-5 minutes for ground state
- **QAOA MaxCut (20-node graph, p=3)**: 2-10 minutes depending on graph structure

Best Practices
--------------

**Ansatz Selection**

1. **Hardware-efficient**: General-purpose, good expressibility, works for most problems
2. **UCCSD**: Best for molecular systems, chemistry-inspired structure
3. **QAOA**: Specifically designed for combinatorial optimization
4. **Custom**: Use when problem structure suggests specific gate patterns

**Initialization Strategies**

- **Random**: Default, sample parameters from uniform(-π, π)
- **Zero initialization**: For QAOA, often starts from reasonable state
- **Problem-specific**: Use classical heuristics (e.g., MP2 amplitudes for UCCSD)
- **Transfer learning**: Initialize from solution of similar problem

**Avoiding Local Minima**

- Run multiple optimizations with different random seeds
- Use global optimizers (Basin-hopping, Differential Evolution) for rugged landscapes
- Gradually increase ansatz depth (start with 1-2 layers, add more if needed)
- Monitor variance: high variance in repeated runs indicates local minima issues

**Convergence Monitoring**

.. code-block:: python

   # Check convergence quality
   if vqe.energies[-1] - vqe.energies[-10] > 1e-5:
       print("Warning: May not have converged")

   # Check for overfitting (ansatz too flexible)
   if len(params) > 2 * n_qubits:
       print("Warning: Large parameter count may cause optimization difficulties")

   # Inspect gradient norm
   final_grad = vqe.compute_gradients(params)
   if np.linalg.norm(final_grad) > 0.01:
       print("Warning: Large gradient at convergence, may need more iterations")

**Troubleshooting**

- **Optimization stuck**: Try different optimizer (e.g., switch from L-BFGS-B to COBYLA)
- **Energy increases**: Reduce learning rate (for Adam/SGD) or check gradient implementation
- **Memory errors**: Reduce ``chi_max`` or use smaller bond dimensions
- **Slow convergence**: Enable gradient grouping, use autograd instead of parameter-shift
- **Poor final energy**: Increase ``n_layers``, try different initialization, or check Hamiltonian

Limitations
-----------

**Barren Plateaus**

For deep circuits (>10 layers) on large systems (>20 qubits), gradients vanish exponentially [McClean18]_. Symptoms:

- Gradient norm < 10⁻⁶ early in optimization
- Energy barely changes over many iterations
- Random parameter changes have no effect

**Mitigation**: Use shallow circuits, problem-inspired ansatze (UCCSD, QAOA), or local cost functions.

**Expressibility vs. Trainability**

More expressive ansatze (more layers, parameters) can represent more states but are harder to optimize:

.. list-table:: Ansatz Trade-offs
   :header-rows: 1
   :widths: 25 25 25 25

   * - Layers
     - Expressibility
     - Trainability
     - Recommendation
   * - 1-2
     - Low
     - Excellent
     - Small molecules, quick tests
   * - 3-5
     - Medium
     - Good
     - Production use
   * - 6-10
     - High
     - Poor (barren plateaus)
     - Avoid unless necessary
   * - >10
     - Very High
     - Very Poor
     - Not practical

**System Size Limits**

- **MPS backend**: 50-100 qubits with χ ≤ 32 (highly entangled circuits may require χ > 64)
- **Memory**: 8GB GPU handles ~40 qubits with χ=32
- **Time**: 100-qubit VQE with 2 layers, 200 iterations: ~1-2 hours on A100

See Also
--------

- :doc:`mpo_ops` - Hamiltonian construction and MPO operations
- :doc:`adaptive_mps` - Adaptive MPS for state representation
- :doc:`mps_pytorch` - PyTorch MPS backend
- :doc:`../user_guide/tutorials/vqe_tutorial` - Step-by-step VQE tutorial
- :doc:`../user_guide/tutorials/qaoa_tutorial` - QAOA implementation guide
- :doc:`../user_guide/tutorials/molecular_vqe` - Quantum chemistry applications

References
----------

.. [Peruzzo14] A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor," *Nature Communications* 5, 4213 (2014).

.. [Farhi14] E. Farhi, J. Goldstone, & S. Gutmann, "A quantum approximate optimization algorithm," *arXiv:1411.4028* (2014).

.. [McClean18] J. R. McClean et al., "Barren plateaus in quantum neural network training landscapes," *Nature Communications* 9, 4812 (2018).

.. [Sim19] S. Sim et al., "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms," *Advanced Quantum Technologies* 2, 1900070 (2019).

.. [Grimsley19] H. R. Grimsley et al., "An adaptive variational algorithm for exact molecular simulations on a quantum computer," *Nature Communications* 10, 3007 (2019).

.. [Cerezo21] M. Cerezo et al., "Variational quantum algorithms," *Nature Reviews Physics* 3, 625 (2021).
