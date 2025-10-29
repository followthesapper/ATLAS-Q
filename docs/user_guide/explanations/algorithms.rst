=========
Algorithms
=========

This document explains the core algorithms implemented in ATLAS-Q for quantum state simulation, ground state finding, optimization, and time evolution. Understanding these algorithms helps you choose the right method for your quantum computing task.

.. contents::
   :local:
   :depth: 2

Overview
========

ATLAS-Q implements several classes of algorithms:

**Time Evolution**:
- **TDVP** (Time-Dependent Variational Principle): Evolves MPS states in time via projection onto tangent space
- **Trotter decomposition**: Exact time evolution for short-depth circuits

**Ground State Finding**:
- **DMRG** (Density Matrix Renormalization Group): Sweeping algorithm for ground states
- **Imaginary-time TDVP**: Projection to lower energy states

**Variational Methods**:
- **VQE** (Variational Quantum Eigensolver): Hybrid quantum-classical eigenvalue finding
- **QAOA** (Quantum Approximate Optimization Algorithm): Variational approach for combinatorial problems

**Specialized Methods**:
- **Stabilizer formalism**: Efficient simulation of Clifford circuits via tableau
- **Period-finding**: Compressed representation for Shor's algorithm components

This guide covers the mathematical foundations, implementation details, and practical considerations for each algorithm.

Time-Dependent Variational Principle (TDVP)
============================================

Mathematical Foundation
-----------------------

The Time-Dependent Variational Principle simulates quantum dynamics by evolving the MPS state within the manifold of fixed bond dimension states. Given the Schrödinger equation:

.. math::

   i\hbar \frac{d|\psi(t)\rangle}{dt} = \hat{H} |\psi(t)\rangle

TDVP approximates this evolution by projecting onto the tangent space of the MPS manifold:

.. math::

   i\hbar \frac{dA^{[i]}}{dt} = P_{\mathcal{M}} \hat{H} |\psi\rangle

where :math:`P_{\mathcal{M}}` is the projector onto the tangent space at the current MPS state.

**Key properties**:

1. **Manifold preservation**: Evolution stays on MPS manifold (no arbitrary bond growth)
2. **Energy conservation**: For Hamiltonian evolution, total energy is conserved
3. **Gauge freedom**: Requires canonical gauge (left or right) for numerical stability

1-Site TDVP
-----------

The 1-site TDVP algorithm evolves a single tensor at a time while keeping bond dimension fixed:

**Algorithm**:

1. **Sweep right** (sites i = 1, 2, ..., n-1):

   a. Bring MPS to left-canonical form up to site i
   b. Compute effective Hamiltonian :math:`H_{\text{eff}}^{[i]}` acting on site i
   c. Evolve :math:`A^{[i]}` for half time step: :math:`A^{[i]}(t + \delta t/2) = \exp(-i H_{\text{eff}}^{[i]} \delta t / 2) A^{[i]}(t)`
   d. Update bond between i and i+1 by evolving backward in time (projection step)
   e. Move to next site

2. **Sweep left** (sites i = n, n-1, ..., 2):

   Repeat process moving left through chain

**Complexity**: :math:`O(n \chi^3 d^2)` per time step for nearest-neighbor Hamiltonians

**Advantages**:
- Conserves bond dimension exactly
- No truncation error beyond variational approximation
- Efficient for moderate bond dimensions

**Disadvantages**:
- Cannot capture rapid entanglement growth
- May violate local conservation laws slightly
- Requires small time steps for accuracy

**ATLAS-Q implementation**:

.. code-block:: python

   from atlas_q.tdvp import TDVP
   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import build_heisenberg_hamiltonian

   # Create initial state
   n_qubits = 20
   mps = AdaptiveMPS(num_qubits=n_qubits, bond_dim=64, device='cuda')

   # Apply initial state preparation (e.g., Néel state)
   for i in range(0, n_qubits, 2):
       mps.apply_x(i)

   # Build Heisenberg Hamiltonian
   J = [1.0, 1.0, 1.0]  # [Jx, Jy, Jz]
   h = 0.5              # External field
   hamiltonian_mpo = build_heisenberg_hamiltonian(n_qubits, J=J, h=h)

   # Initialize TDVP with 1-site algorithm
   tdvp = TDVP(
       mps=mps,
       hamiltonian=hamiltonian_mpo,
       dt=0.05,                    # Time step
       method='one_site',          # 1-site TDVP
       normalize=True,
       track_energy=True
   )

   # Time evolution
   times = []
   energies = []
   for step in range(200):
       E = tdvp.evolve_step()
       times.append(step * tdvp.dt)
       energies.append(E)

       if step % 20 == 0:
           print(f"t = {times[-1]:.2f}, E = {E:.6f}")

**Numerical stability**:

1. Use canonical gauges (left or right)
2. Renormalize after each sweep
3. Monitor energy conservation (dE/dt ≈ 0 for Hamiltonian evolution)
4. Reduce time step if energy drift exceeds tolerance

2-Site TDVP
-----------

The 2-site TDVP evolves a two-site tensor, allowing bond dimension to grow adaptively via SVD truncation:

**Algorithm**:

1. **Sweep right** (bonds i = 1, 2, ..., n-2):

   a. Merge tensors at sites i and i+1 into two-site tensor Θ[i:i+1]
   b. Compute effective Hamiltonian acting on two sites
   c. Evolve Θ for time step: :math:`\Theta(t + \delta t) = \exp(-i H_{\text{eff}} \delta t) \Theta(t)`
   d. SVD Θ back into two tensors with truncation to bond dimension χ
   e. Backward evolution on bond (projection step)
   f. Move to next bond

2. **Sweep left**: Repeat moving left through chain

**Complexity**: :math:`O(n \chi^3 d^3 + n \chi^3)` per time step

**Advantages**:
- Captures entanglement growth naturally
- More accurate than 1-site for rapid dynamics
- Conserves local quantities (charge, magnetization)
- Adapts bond dimension based on truncation error

**Disadvantages**:
- Higher computational cost (extra factor of d)
- Requires SVD truncation (introduces error)
- Bond dimension may grow unbounded

**ATLAS-Q implementation with adaptive truncation**:

.. code-block:: python

   from atlas_q.tdvp import TDVP
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Initial state with adaptive bond growth
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=32,                    # Initial χ
       chi_max_per_bond=256,           # Maximum allowed χ
       truncation_threshold=1e-8,      # SVD cutoff
       adaptive_mode=True,
       device='cuda'
   )

   # Initialize TDVP with 2-site algorithm
   tdvp = TDVP(
       mps=mps,
       hamiltonian=hamiltonian_mpo,
       dt=0.1,
       method='two_site',              # 2-site TDVP
       chi_max=256,                    # Maximum bond during evolution
       truncation_threshold=1e-8,
       normalize=True
   )

   # Evolve and monitor bond dimension growth
   for step in range(100):
       E = tdvp.evolve_step()
       max_chi = max(t.shape[1] if i > 0 else 1
                     for i, t in enumerate(mps.tensors))

       if step % 10 == 0:
           print(f"Step {step}: E = {E:.6f}, max χ = {max_chi}")

**When to use 2-site vs 1-site**:

- **Use 1-site** when:
  - Bond dimension is already large enough to represent state
  - Long-time evolution where entanglement saturates
  - Memory constraints require fixed χ
  - Hamiltonian is time-independent and weakly entangling

- **Use 2-site** when:
  - Initial χ may be insufficient
  - Rapid entanglement growth expected (e.g., quenches, deep circuits)
  - Need high accuracy for short-time dynamics
  - Have computational resources for larger χ

Density Matrix Renormalization Group (DMRG)
============================================

Mathematical Foundation
-----------------------

DMRG finds the ground state of a Hamiltonian by minimizing energy over the MPS manifold:

.. math::

   E_0 = \min_{|\psi\rangle \in \text{MPS}} \langle\psi|\hat{H}|\psi\rangle

The algorithm performs a variational sweep through the chain, locally minimizing energy at each site.

Sweeping Algorithm
------------------

**Single-site DMRG**:

1. **Initialize**: Random or product state MPS with bond dimension χ
2. **Sweep right** (i = 1, 2, ..., n-1):

   a. Compute effective Hamiltonian :math:`H_{\text{eff}}^{[i]}`
   b. Find ground state of :math:`H_{\text{eff}}^{[i]}` (dominant eigenvector)
   c. Update site tensor :math:`A^{[i]}` with ground state
   d. Orthogonalize and move to next site

3. **Sweep left** (i = n, n-1, ..., 2): Repeat
4. **Convergence**: Iterate until energy change < tolerance

**Two-site DMRG** (more common):

Optimizes two adjacent sites simultaneously, then splits via SVD. Allows bond dimension growth.

**Complexity**: :math:`O(N_{\text{sweep}} \cdot n \chi^3 d^2)` for convergence

**ATLAS-Q implementation** (via imaginary-time TDVP):

.. code-block:: python

   from atlas_q.tdvp import TDVP
   import torch

   # Ground state search via imaginary-time evolution
   # Use imaginary time: τ = it

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

   # TDVP with imaginary time
   tdvp = TDVP(
       mps=mps,
       hamiltonian=hamiltonian_mpo,
       dt=0.1j,                        # Imaginary time step (complex)
       method='two_site',
       chi_max=128,
       truncation_threshold=1e-10,
       normalize=True                  # Essential for imaginary time
   )

   # Evolve until convergence
   prev_energy = float('inf')
   for step in range(500):
       E = tdvp.evolve_step().real     # Energy is real

       if step % 10 == 0:
           dE = abs(E - prev_energy)
           print(f"Step {step}: E = {E:.8f}, dE = {dE:.2e}")

           if dE < 1e-8:
               print(f"Converged to ground state: E_0 = {E:.8f}")
               break

           prev_energy = E

**Advantages**:
- Finds global ground state reliably
- Efficient for 1D systems with area-law entanglement
- Can target excited states with modifications
- Provides accurate energy and observables

**Disadvantages**:
- Requires many sweeps for convergence (10-100+)
- Struggles with critical systems (diverging entanglement)
- Bond dimension requirements grow near phase transitions
- Imaginary-time evolution requires careful normalization

Variational Quantum Eigensolver (VQE)
======================================

Mathematical Foundation
-----------------------

VQE is a hybrid quantum-classical algorithm that minimizes the energy expectation value:

.. math::

   E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|\hat{H}|\psi(\boldsymbol{\theta})\rangle = \langle 0 | U^\dagger(\boldsymbol{\theta}) \hat{H} U(\boldsymbol{\theta}) | 0 \rangle

over parametrized quantum states (ansatz):

.. math::

   |\psi(\boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) |0\rangle

**Algorithm**:

1. Prepare ansatz state :math:`|\psi(\boldsymbol{\theta})\rangle`
2. Measure energy :math:`E(\boldsymbol{\theta}) = \langle \psi | H | \psi \rangle`
3. Classical optimizer updates parameters: :math:`\boldsymbol{\theta} \to \boldsymbol{\theta} - \alpha \nabla E`
4. Repeat until convergence

**Key to success**: Ansatz design balancing expressibility and trainability.

Ansatz Construction
-------------------

**Hardware-Efficient Ansatz (HEA)**:

Shallow circuit matching hardware connectivity:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.adaptive_mps import AdaptiveMPS

   n_qubits = 12

   # Define hardware-efficient ansatz
   def hardware_efficient_ansatz(mps, params):
       """
       Hardware-efficient ansatz with layers of single-qubit rotations + CZ.

       Circuit: (RY ⊗ RZ) - CZ - (RY ⊗ RZ) - CZ - ...
       """
       n_layers = len(params) // (2 * mps.num_qubits)
       idx = 0

       for layer in range(n_layers):
           # Single-qubit rotations
           for i in range(mps.num_qubits):
               mps.apply_ry(i, params[idx])
               idx += 1
           for i in range(mps.num_qubits):
               mps.apply_rz(i, params[idx])
               idx += 1

           # Entangling layer (CZ on nearest neighbors)
           for i in range(0, mps.num_qubits - 1, 2):
               mps.apply_cz(i, i + 1)
           for i in range(1, mps.num_qubits - 1, 2):
               mps.apply_cz(i, i + 1)

       return mps

   # Build Hamiltonian (e.g., Heisenberg model)
   from atlas_q.mpo_ops import build_heisenberg_hamiltonian
   hamiltonian = build_heisenberg_hamiltonian(n_qubits, J=[1.0, 1.0, 1.0], h=0.5)

   # VQE configuration
   config = VQEConfig(
       max_iterations=500,
       optimizer='L-BFGS-B',
       tol=1e-6,
       bond_dim=64,
       gradient_method='finite_diff',   # or 'parameter_shift'
       gradient_epsilon=1e-4
   )

   # Initialize VQE
   vqe = VQE(
       num_qubits=n_qubits,
       hamiltonian=hamiltonian,
       ansatz_fn=hardware_efficient_ansatz,
       config=config,
       device='cuda'
   )

   # Optimize
   n_params = 4 * n_qubits  # 2 layers × 2 rotations/qubit
   initial_params = torch.rand(n_params) * 2 * 3.14159
   result = vqe.optimize(initial_params)

   print(f"Ground state energy: {result['energy']:.8f}")
   print(f"Converged in {result['n_iterations']} iterations")

**Unitary Coupled-Cluster (UCCSD)**:

Chemically-inspired ansatz for molecular systems:

.. math::

   U(\boldsymbol{\theta}) = \prod_i \exp(\theta_i (T_i - T_i^\dagger))

where :math:`T_i` are fermionic excitation operators (singles, doubles).

.. code-block:: python

   from atlas_q.ansatz_uccsd import build_uccsd_ansatz
   from atlas_q.vqe_qaoa import VQE

   # Molecular Hamiltonian (from PySCF, OpenFermion, etc.)
   # Assume we have molecular_hamiltonian as MPO

   n_qubits = 8   # For 4-electron, 8-orbital system
   n_electrons = 4

   # UCCSD ansatz
   ansatz_fn, n_params = build_uccsd_ansatz(
       n_qubits=n_qubits,
       n_electrons=n_electrons,
       excitation_type='SD'  # Singles and doubles
   )

   config = VQEConfig(
       max_iterations=1000,
       optimizer='COBYLA',  # Gradient-free for noisy problems
       tol=1e-6,
       bond_dim=128         # Higher χ for chemistry
   )

   vqe = VQE(
       num_qubits=n_qubits,
       hamiltonian=molecular_hamiltonian,
       ansatz_fn=ansatz_fn,
       config=config,
       device='cuda'
   )

   # Optimize from Hartree-Fock initial state
   initial_params = torch.zeros(n_params)
   result = vqe.optimize(initial_params)

   print(f"Molecular ground state: {result['energy']:.8f} Ha")

Gradient Computation
--------------------

**Finite Differences**:

.. math::

   \frac{\partial E}{\partial \theta_i} \approx \frac{E(\theta_i + \epsilon) - E(\theta_i - \epsilon)}{2\epsilon}

Simple but requires 2m energy evaluations for m parameters.

**Parameter-Shift Rule**:

For gates :math:`G(\theta) = \exp(-i\theta P)` where :math:`P^2 = I`:

.. math::

   \frac{\partial \langle H \rangle}{\partial \theta} = \frac{1}{2} \left( \langle H \rangle_{\theta + \pi/4} - \langle H \rangle_{\theta - \pi/4} \right)

Exact gradient from two shifted energy evaluations.

**ATLAS-Q gradient implementations**:

.. code-block:: python

   config = VQEConfig(
       optimizer='L-BFGS-B',
       gradient_method='parameter_shift',  # Exact gradients
       gradient_epsilon=None                # Not used for parameter_shift
   )

   # Or use finite differences
   config = VQEConfig(
       optimizer='L-BFGS-B',
       gradient_method='finite_diff',
       gradient_epsilon=1e-4               # Step size for finite diff
   )

**Optimization convergence**:

Monitor energy, gradient norm, and parameter updates:

.. code-block:: python

   result = vqe.optimize(initial_params)

   # Convergence diagnostics
   print(f"Final energy: {result['energy']:.8f}")
   print(f"Gradient norm: {result['gradient_norm']:.2e}")
   print(f"Iterations: {result['n_iterations']}")
   print(f"Function evaluations: {result['n_function_evals']}")

   # Visualize optimization trajectory
   import matplotlib.pyplot as plt
   plt.plot(result['energy_history'])
   plt.xlabel('Iteration')
   plt.ylabel('Energy')
   plt.title('VQE Convergence')
   plt.yscale('log')  # Log scale shows convergence rate

Quantum Approximate Optimization Algorithm (QAOA)
==================================================

Mathematical Foundation
-----------------------

QAOA solves combinatorial optimization problems by encoding the cost function as a quantum Hamiltonian :math:`\hat{H}_C` and using a parametrized ansatz:

.. math::

   |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}

where:

- :math:`U_C(\gamma) = e^{-i\gamma \hat{H}_C}` encodes the problem
- :math:`U_B(\beta) = e^{-i\beta \hat{H}_B}` is the mixer (typically :math:`\hat{H}_B = \sum_i \hat{X}_i`)
- :math:`p` is the number of QAOA layers (depth)

**Goal**: Minimize :math:`\langle \psi | \hat{H}_C | \psi \rangle` to find approximate solution.

MaxCut Problem
--------------

Find a graph partition maximizing edges between partitions.

**Cost Hamiltonian**:

.. math::

   \hat{H}_C = \sum_{(i,j) \in E} \frac{1 - \hat{Z}_i \hat{Z}_j}{2}

**ATLAS-Q implementation**:

.. code-block:: python

   from atlas_q.vqe_qaoa import QAOA, QAOAConfig
   import networkx as nx

   # Define graph (MaxCut problem)
   G = nx.erdos_renyi_graph(n=12, p=0.4, seed=42)

   # Build cost Hamiltonian from graph
   from atlas_q.mpo_ops import build_maxcut_hamiltonian
   cost_hamiltonian = build_maxcut_hamiltonian(G)

   # QAOA configuration
   config = QAOAConfig(
       p=3,                        # QAOA depth (layers)
       optimizer='COBYLA',
       max_iterations=300,
       bond_dim=32,                # Usually small for QAOA
       init_strategy='random'      # or 'linear_ramp'
   )

   # Initialize QAOA
   qaoa = QAOA(
       graph=G,
       hamiltonian=cost_hamiltonian,
       config=config,
       device='cuda'
   )

   # Optimize QAOA parameters
   result = qaoa.optimize()

   print(f"Best cost: {result['cost']:.6f}")
   print(f"Optimal parameters γ: {result['gamma']}")
   print(f"Optimal parameters β: {result['beta']}")

   # Extract solution (classical bitstring with highest probability)
   optimal_state = result['optimal_state']
   bitstring = qaoa.sample_bitstring(optimal_state, n_samples=1000)
   print(f"Best partition: {bitstring}")

**Initialization strategies**:

1. **Random**: :math:`\gamma_i, \beta_i \sim U(0, 2\pi)`
2. **Linear ramp**: :math:`\gamma_i = i \cdot \gamma_{\text{max}} / p`, :math:`\beta_i = (p - i) \cdot \beta_{\text{max}} / p`
3. **Warm start**: Use p-1 solution to initialize p-layer QAOA

**Performance vs depth**:

- p=1: Often 0.6-0.7 approximation ratio
- p=3-5: Typically 0.8-0.9 approximation ratio
- p→∞: Approaches optimal solution

Higher p increases expressibility but also optimization difficulty (more parameters, more local minima).

Stabilizer Formalism
=====================

Mathematical Foundation
-----------------------

The **Gottesman-Knill theorem** states that Clifford circuits (H, S, CNOT, CZ) on stabilizer states can be simulated efficiently in :math:`O(n^2)` time and :math:`O(n^2)` space, compared to :math:`O(2^n)` for generic statevector simulation.

**Stabilizer state**: Eigenstate of a set of commuting Pauli operators (stabilizers):

.. math::

   \hat{S}_i |\psi\rangle = |\psi\rangle \quad \forall i

Represented by a **tableau**: :math:`n \times 2n` binary matrix encoding n independent stabilizer generators.

Tableau Representation
-----------------------

Each stabilizer :math:`\hat{S}_i = \pm i^r \hat{X}^{x_1} \hat{Z}^{z_1} \otimes \cdots \otimes \hat{X}^{x_n} \hat{Z}^{z_n}` is encoded as:

.. math::

   [ x_1, x_2, \ldots, x_n | z_1, z_2, \ldots, z_n | r ]

**Gate updates** modify tableau rows via Gaussian elimination over :math:`\mathbb{F}_2`.

**ATLAS-Q implementation**:

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerState

   n_qubits = 50
   state = StabilizerState(n_qubits)

   # Clifford gates update tableau in O(n²)
   state.h(0)               # Hadamard
   state.s(0)               # Phase gate
   state.cnot(0, 1)
   state.cz(2, 3)

   # Measurement collapses state (O(n²))
   result = state.measure(0)
   print(f"Measured qubit 0: {result}")

   # Check expectation values efficiently
   pauli_string = 'ZZIIZ'  # Z₀Z₁I₂I₃Z₄
   expectation = state.expectation_value(pauli_string)
   print(f"⟨{pauli_string}⟩ = {expectation}")

**Performance comparison**:

For Clifford circuits on n=50 qubits with depth d=1000:

- **Stabilizer**: ~0.5 seconds, 20 MB memory
- **MPS** (χ=64): ~5 seconds, 400 MB memory
- **Statevector**: Impossible (10^15 TB memory)

**Use cases**:

- Benchmarking non-Clifford gate error rates
- Simulating syndrome extraction circuits (error correction)
- Randomized benchmarking
- Gottesman-Knill subcircuits in hybrid algorithms

Period-Finding Algorithm
=========================

Quantum period-finding is the core subroutine of Shor's factoring algorithm. Given a function :math:`f(x) = a^x \mod N`, find the period :math:`r` where :math:`f(x + r) = f(x)`.

Compressed Representation
--------------------------

ATLAS-Q uses a specialized **PeriodicState** class representing periodic superpositions in :math:`O(1)` memory:

.. math::

   |\psi_{\text{period}}\rangle = \frac{1}{\sqrt{M}} \sum_{k=0}^{M-1} |kr \mod 2^n\rangle

where :math:`M = \lfloor 2^n / r \rfloor`.

**Quantum Fourier Transform** on periodic states yields analytical peak positions without full :math:`O(2^n)` statevector.

**ATLAS-Q implementation**:

.. code-block:: python

   from atlas_q.period_finding import PeriodicState, find_period_quantum

   # Factor N = 15 (Shor's algorithm example)
   N = 15
   a = 7  # Coprime to 15

   # Period-finding quantum subroutine
   n_qubits = 8  # Precision qubits
   periodic_state = PeriodicState(n_qubits=n_qubits, period_candidate=4)

   # QFT + measurement (analytical for periodic state)
   measurement_outcomes = periodic_state.qft_measure(n_shots=10)

   # Classical post-processing: continued fractions
   from atlas_q.period_finding import extract_period_from_measurements
   r = extract_period_from_measurements(measurement_outcomes, n_qubits)

   print(f"Found period r = {r}")

   # Verify: a^r mod N = 1
   assert pow(a, r, N) == 1, f"Period verification failed"

   # Factor N using period
   from math import gcd
   factor1 = gcd(pow(a, r//2, N) - 1, N)
   factor2 = gcd(pow(a, r//2, N) + 1, N)

   if factor1 > 1 and factor1 < N:
       print(f"Non-trivial factor: {factor1}")
   if factor2 > 1 and factor2 < N:
       print(f"Non-trivial factor: {factor2}")

**Complexity**:

- **Quantum part**: :math:`O(n^2)` for QFT on n qubits (efficiently on periodic state)
- **Classical part**: :math:`O(\log^2 N)` for continued fractions
- **Memory**: :math:`O(1)` for periodic state vs :math:`O(2^n)` for generic state

**Benchmarks verified**: N = 15, 21, 35, 143 (canonical Shor's algorithm test cases).

Algorithm Selection Guide
==========================

Choosing the right algorithm depends on your problem, system size, and computational resources:

**Time Evolution** (simulate quantum dynamics):

- **Short-depth circuits** (< 100 gates): Apply gates directly to MPS
- **Long-time evolution, moderate entanglement**: 1-site TDVP (fixed χ)
- **Rapid entanglement growth**: 2-site TDVP (adaptive χ)
- **Very long evolution, saturated entanglement**: 1-site TDVP with large fixed χ

**Ground State Finding**:

- **1D systems, small χ**: Imaginary-time TDVP (DMRG-like)
- **Chemistry, small molecules**: VQE with UCCSD ansatz
- **Need excited states**: Multi-target DMRG or penalty methods

**Optimization Problems**:

- **Combinatorial optimization** (MaxCut, TSP, etc.): QAOA
- **Structured problems with good ansatz**: VQE with problem-specific ansatz
- **Classical preprocessing available**: Warm-start QAOA

**Special Cases**:

- **Clifford-only circuits**: Stabilizer formalism (20-50× speedup)
- **Period-finding, Shor's algorithm**: PeriodicState compression
- **Sampling tasks**: MPS-based sampling more efficient than statevector

**Scaling considerations**:

+------------------+-------------------+-------------------+------------------------+
| Algorithm        | Time Complexity   | Space Complexity  | Max Qubits (practical) |
+==================+===================+===================+========================+
| 1-site TDVP      | O(n χ³ d²)        | O(n χ² d)         | 100+ (χ=64)            |
+------------------+-------------------+-------------------+------------------------+
| 2-site TDVP      | O(n χ³ d³)        | O(n χ² d)         | 80+ (χ=128)            |
+------------------+-------------------+-------------------+------------------------+
| VQE              | O(iter · T · χ³)  | O(n χ² d)         | 50+ (depends on ansatz)|
+------------------+-------------------+-------------------+------------------------+
| QAOA (p layers)  | O(iter · p · χ³)  | O(n χ² d)         | 30+ (χ typically low)  |
+------------------+-------------------+-------------------+------------------------+
| Stabilizer       | O(d · n²)         | O(n²)             | 1000+ (Clifford only)  |
+------------------+-------------------+-------------------+------------------------+
| DMRG             | O(sweeps·n·χ³·d²) | O(n χ² d)         | 100+ (1D systems)      |
+------------------+-------------------+-------------------+------------------------+

Where:
- n = number of qubits
- χ = bond dimension
- d = local dimension (2 for qubits)
- T = circuit depth
- iter = optimization iterations

Summary
=======

ATLAS-Q implements a comprehensive suite of quantum algorithms leveraging MPS efficiency:

**Time Evolution**:
- TDVP (1-site and 2-site) for accurate quantum dynamics with controlled bond growth
- Trotter decomposition for exact short-depth simulation

**Ground State Finding**:
- DMRG via imaginary-time evolution for ground states of 1D systems
- Converges reliably with O(n χ³) complexity per sweep

**Variational Methods**:
- VQE for hybrid quantum-classical optimization with customizable ansätze
- QAOA for combinatorial optimization problems with provable approximation guarantees
- Parameter-shift rule for exact gradients

**Specialized Algorithms**:
- Stabilizer formalism for 20-50× speedup on Clifford circuits
- Period-finding with O(1) memory for Shor's algorithm

**Key Insights**:

1. **1-site vs 2-site TDVP**: Use 1-site for efficiency when χ is sufficient; use 2-site when entanglement grows rapidly
2. **VQE ansatz design**: Hardware-efficient for NISQ devices; UCCSD for chemistry; problem-specific for structured problems
3. **QAOA depth**: Start with p=1-3 for quick approximations; increase p if higher accuracy needed
4. **Algorithm switching**: Begin with stabilizer for Clifford parts, switch to MPS for non-Clifford gates
5. **Convergence monitoring**: Track energy, truncation error, and bond dimensions to ensure accuracy

For detailed usage examples, see:

- :doc:`../tutorials/tdvp_tutorial` - TDVP time evolution walkthrough
- :doc:`../tutorials/vqe_tutorial` - VQE ground state finding
- :doc:`../tutorials/qaoa_tutorial` - QAOA for MaxCut
- :doc:`../howtos/optimize_performance` - Performance tuning for all algorithms

All algorithms in ATLAS-Q benefit from GPU acceleration, adaptive truncation, and hybrid MPS-stabilizer execution for optimal performance across problem scales.
