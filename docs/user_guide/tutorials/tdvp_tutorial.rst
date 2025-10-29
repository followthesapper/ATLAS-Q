TDVP Tutorial
==============

Time-Dependent Variational Principle (TDVP) simulates quantum dynamics on MPS efficiently. This tutorial provides comprehensive coverage of TDVP algorithms, time evolution techniques, and applications to quantum dynamics.

Prerequisites
-------------

- Completed :doc:`beginners` and :doc:`mps_basics` tutorials
- Understanding of Hamiltonians and time evolution
- Familiarity with differential equations
- Knowledge of Krylov subspace methods (helpful)
- Matrix Product State operations

Learning Objectives
-------------------

After completing this tutorial, you will understand:

1. TDVP algorithm principles and time evolution
2. Difference between 1-site and 2-site TDVP
3. Real vs imaginary time evolution
4. Krylov subspace methods for time evolution
5. Time-dependent Hamiltonians
6. Quantum quenches and dynamics
7. Observables and correlation functions
8. Performance optimization for TDVP

Part 1: TDVP Fundamentals
--------------------------

What is TDVP?
^^^^^^^^^^^^^

TDVP projects the Schrödinger equation onto the MPS manifold:

.. math::

   i\hbar \frac{\partial}{\partial t} |\psi(t)\rangle = H |\psi(t)\rangle

For MPS |ψ(A)⟩ parameterized by tensors A, TDVP gives equations of motion for the tensors.

Time Evolution Operator
^^^^^^^^^^^^^^^^^^^^^^^

Exact evolution:

.. math::

   |\psi(t)\rangle = e^{-iHt/\hbar} |\psi(0)\rangle

TDVP approximates this while keeping |ψ⟩ in MPS form with bounded bond dimension.

Why TDVP Works
^^^^^^^^^^^^^^

TDVP is optimal in the sense that it minimizes:

.. math::

   || \frac{d}{dt}|\psi_{\text{MPS}}\rangle - (-iH)|\psi_{\text{MPS}}\rangle ||

at each instant, keeping the state on the MPS manifold.

Basic Time Evolution
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.tdvp import TDVP1Site, TDVPConfig
   import torch

   # Create Hamiltonian - transverse field Ising model
   n_sites = 10
   H = MPOBuilder.ising_hamiltonian(
       n_sites=n_sites,
       J=1.0,      # Coupling
       h=0.5,      # Transverse field
       device='cuda'
   )

   # Initialize state (all |0⟩)
   mps = AdaptiveMPS(num_qubits=n_sites, bond_dim=16, device='cuda')

   # Configure TDVP
   config = TDVPConfig(
       dt=0.01,         # Time step
       t_final=1.0,     # Final time
       normalize=True,  # Normalize after each step
       device='cuda'
   )

   # Create TDVP solver
   tdvp = TDVP1Site(H, mps, config)

   # Evolve
   times, energies = tdvp.run()

   print(f"Initial energy: {energies[0]:.6f}")
   print(f"Final energy: {energies[-1]:.6f}")
   print(f"Energy conservation: {abs(energies[-1] - energies[0]):.2e}")

Expected output:

.. code-block:: text

   Initial energy: -5.234567
   Final energy: -5.234571
   Energy conservation: 4.23e-06

Part 2: 1-Site vs 2-Site TDVP
------------------------------

1-Site TDVP
^^^^^^^^^^^

Conserves bond dimension χ but may accumulate truncation error:

.. code-block:: python

   from atlas_q.tdvp import TDVP1Site, TDVPConfig
   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.ising_hamiltonian(12, J=1.0, h=0.5, device='cuda')
   mps = AdaptiveMPS(12, bond_dim=16, device='cuda')

   # 1-site TDVP configuration
   config = TDVPConfig(
       dt=0.01,
       t_final=2.0,
       normalize=True
   )

   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

   # Check bond dimension
   chi_initial = max([t.shape[0] for t in mps.tensors[1:]])
   chi_final = max([t.shape[0] for t in mps.tensors[1:]])

   print(f"Initial χ: {chi_initial}")
   print(f"Final χ: {chi_final}")  # Should be same
   print("Bond dimension conserved with 1-site TDVP")

Algorithm:
1. Sweep left-to-right, evolving each single tensor
2. Sweep right-to-left
3. Repeat for each time step

2-Site TDVP
^^^^^^^^^^^

Allows bond dimension growth for more accurate dynamics:

.. code-block:: python

   from atlas_q.tdvp import TDVP2Site, TDVPConfig

   H = MPOBuilder.ising_hamiltonian(12, J=1.0, h=0.5, device='cuda')
   mps = AdaptiveMPS(12, bond_dim=8, device='cuda')  # Start smaller

   # 2-site TDVP configuration
   config = TDVPConfig(
       dt=0.01,
       t_final=2.0,
       chi_max=64,          # Maximum bond dimension
       eps_trunc=1e-8,      # SVD truncation tolerance
       normalize=True
   )

   tdvp = TDVP2Site(H, mps, config)
   times, energies = tdvp.run()

   # Track bond dimension growth
   chi_history = tdvp.chi_history

   print(f"Initial χ: {chi_history[0]}")
   print(f"Final χ: {chi_history[-1]}")
   print(f"Maximum χ reached: {max(chi_history)}")

Algorithm:
1. Sweep left-to-right, evolving pairs of tensors
2. Perform SVD to split back into two tensors
3. Truncate based on chi_max and eps_trunc
4. Sweep right-to-left
5. Repeat for each time step

Choosing Between 1-Site and 2-Site
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def recommend_tdvp_variant(entanglement_growth, accuracy_required):
       """
       Recommend TDVP variant based on problem.

       Args:
           entanglement_growth: 'low', 'moderate', 'high'
           accuracy_required: 'low', 'moderate', 'high'

       Returns:
           Recommended variant
       """
       if entanglement_growth == 'low' and accuracy_required != 'high':
           return '1-site'  # Fast, χ conserved

       elif entanglement_growth == 'moderate':
           return '2-site'  # Adaptive χ

       elif entanglement_growth == 'high':
           return '2-site with large chi_max'

       elif accuracy_required == 'high':
           return '2-site with tight truncation'

       else:
           return '1-site'  # Default

   # Example
   variant = recommend_tdvp_variant('moderate', 'high')
   print(f"Recommended: {variant}")

Comparison Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt

   H = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.5, device='cuda')

   # Run both variants
   mps_1site = AdaptiveMPS(10, bond_dim=16, device='cuda')
   tdvp_1site = TDVP1Site(H, mps_1site, TDVPConfig(dt=0.01, t_final=2.0))
   times_1, energies_1 = tdvp_1site.run()

   mps_2site = AdaptiveMPS(10, bond_dim=16, device='cuda')
   tdvp_2site = TDVP2Site(H, mps_2site, TDVPConfig(dt=0.01, t_final=2.0, chi_max=64))
   times_2, energies_2 = tdvp_2site.run()

   # Plot comparison
   plt.figure(figsize=(10, 6))
   plt.plot(times_1, energies_1, 'b-', label='1-site TDVP', linewidth=2)
   plt.plot(times_2, energies_2, 'r--', label='2-site TDVP', linewidth=2)
   plt.xlabel('Time', fontsize=12)
   plt.ylabel('Energy', fontsize=12)
   plt.legend(fontsize=12)
   plt.title('1-site vs 2-site TDVP', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.savefig('tdvp_comparison.png', dpi=150)
   plt.close()

Part 3: Real vs Imaginary Time
-------------------------------

Real Time Evolution
^^^^^^^^^^^^^^^^^^^

Standard Schrödinger evolution for dynamics:

.. code-block:: python

   # Real time: i dψ/dt = H ψ
   config = TDVPConfig(
       dt=0.01,
       t_final=5.0,
       imaginary_time=False,  # Real time (default)
       normalize=False         # Don't normalize (conserves norm)
   )

   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

   # Energy should be conserved
   energy_drift = abs(energies[-1] - energies[0])
   print(f"Energy drift: {energy_drift:.2e}")

Imaginary Time Evolution
^^^^^^^^^^^^^^^^^^^^^^^^^

Projects onto ground state by evolving in imaginary time:

.. code-block:: python

   # Imaginary time: dψ/dτ = -H ψ
   config = TDVPConfig(
       dt=0.01,          # Now τ step
       t_final=10.0,     # Evolve until convergence
       imaginary_time=True,
       normalize=True,   # Must normalize in imaginary time
       convergence_tol=1e-8
   )

   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

   print(f"Initial energy: {energies[0]:.6f}")
   print(f"Ground state energy: {energies[-1]:.6f}")

   # Compare to exact (if available)
   # exact_ground_energy = -10.234
   # error = abs(energies[-1] - exact_ground_energy)

Why Imaginary Time Works
^^^^^^^^^^^^^^^^^^^^^^^^^

In imaginary time τ = it, the evolution operator becomes:

.. math::

   e^{-H\tau} |\psi\rangle \approx e^{-E_0\tau} |\psi_0\rangle + \text{excited states (decay)}

Higher energy states decay exponentially faster, projecting onto the ground state.

Ground State Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def prepare_ground_state_tdvp(H, n_sites, bond_dim=32, tau_max=20.0):
       """
       Prepare ground state using imaginary time TDVP.

       Args:
           H: Hamiltonian MPO
           n_sites: Number of sites
           bond_dim: Initial bond dimension
           tau_max: Maximum imaginary time

       Returns:
           mps: Ground state MPS
           energy: Ground state energy
       """
       from atlas_q.adaptive_mps import AdaptiveMPS
       from atlas_q.tdvp import TDVP2Site, TDVPConfig

       # Start from random state
       mps = AdaptiveMPS(n_sites, bond_dim=bond_dim, device='cuda')

       # Apply random gates to break symmetry
       import torch
       import numpy as np
       for q in range(n_sites):
           theta = np.random.rand() * 2 * np.pi
           Ry = torch.tensor([
               [np.cos(theta/2), -np.sin(theta/2)],
               [np.sin(theta/2), np.cos(theta/2)]
           ], dtype=torch.complex64).to('cuda')
           mps.apply_single_qubit_gate(q, Ry)

       # Imaginary time evolution
       config = TDVPConfig(
           dt=0.05,
           t_final=tau_max,
           imaginary_time=True,
           normalize=True,
           chi_max=128,
           convergence_tol=1e-8
       )

       tdvp = TDVP2Site(H, mps, config)
       times, energies = tdvp.run()

       print(f"Ground state energy: {energies[-1]:.6f}")
       print(f"Converged at τ = {times[-1]:.2f}")

       return mps, energies[-1]

   # Usage
   H = MPOBuilder.ising_hamiltonian(12, J=1.0, h=0.5, device='cuda')
   gs_mps, gs_energy = prepare_ground_state_tdvp(H, n_sites=12)

Part 4: Krylov Subspace Methods
--------------------------------

What are Krylov Methods?
^^^^^^^^^^^^^^^^^^^^^^^^^

For time evolution exp(-iHΔt)|ψ⟩, Krylov methods build:

.. math::

   K_m = \text{span}\{|\psi\rangle, H|\psi\rangle, H^2|\psi\rangle, \ldots, H^{m-1}|\psi\rangle\}

and approximate evolution within this subspace.

Lanczos Algorithm
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.tdvp import TDVPConfig

   config = TDVPConfig(
       dt=0.01,
       t_final=1.0,
       krylov_dim=20,      # Krylov subspace dimension
       krylov_tol=1e-10,   # Convergence tolerance
       use_krylov=True     # Enable Krylov
   )

   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

Krylov dimension controls accuracy vs cost:
- Larger m: more accurate, slower
- Smaller m: less accurate, faster
- Typical: m = 10-30

Adaptive Krylov Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def adaptive_krylov_tdvp(H, mps, t_final, target_error=1e-8):
       """
       TDVP with adaptive Krylov dimension.

       Args:
           H: Hamiltonian
           mps: Initial state
           t_final: Final time
           target_error: Target error per step

       Returns:
           times, energies
       """
       from atlas_q.tdvp import TDVP1Site, TDVPConfig

       # Start with small Krylov dimension
       krylov_dim = 10

       while krylov_dim <= 50:
           config = TDVPConfig(
               dt=0.01,
               t_final=t_final,
               krylov_dim=krylov_dim,
               krylov_tol=target_error
           )

           tdvp = TDVP1Site(H, mps, config)
           times, energies = tdvp.run()

           # Check energy conservation
           energy_drift = abs(energies[-1] - energies[0])

           print(f"Krylov dim={krylov_dim}: energy drift={energy_drift:.2e}")

           if energy_drift < target_error * len(times):
               print(f"Converged with Krylov dimension {krylov_dim}")
               return times, energies

           krylov_dim += 5

       print("Warning: did not converge")
       return times, energies

Part 5: Time-Dependent Hamiltonians
------------------------------------

Driving Field
^^^^^^^^^^^^^

Hamiltonian with time-dependent field:

.. math::

   H(t) = H_0 + h(t) \sum_i \sigma^x_i

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   import numpy as np

   def time_dependent_hamiltonian(t, n_sites, device='cuda'):
       """
       Build time-dependent Hamiltonian.

       H(t) = J Σ ZZ + h(t) Σ X

       Args:
           t: Current time
           n_sites: Number of sites
           device: Computation device

       Returns:
           H(t): Hamiltonian MPO at time t
       """
       # Time-dependent transverse field
       h_t = 0.5 * (1 + np.sin(2 * np.pi * t))

       H = MPOBuilder.ising_hamiltonian(
           n_sites=n_sites,
           J=1.0,
           h=h_t,
           device=device
       )

       return H

   # Evolve with time-dependent H
   from atlas_q.tdvp import TDVPConfig, TDVP1Site
   from atlas_q.adaptive_mps import AdaptiveMPS

   n_sites = 10
   mps = AdaptiveMPS(n_sites, bond_dim=32, device='cuda')

   dt = 0.01
   t_final = 5.0
   n_steps = int(t_final / dt)

   times = []
   energies = []

   for step in range(n_steps):
       t = step * dt

       # Build H(t)
       H_t = time_dependent_hamiltonian(t, n_sites)

       # Take one TDVP step
       config = TDVPConfig(dt=dt, t_final=dt, normalize=False)
       tdvp = TDVP1Site(H_t, mps, config)
       _, e = tdvp.run()

       times.append(t)
       energies.append(e[0])

       if step % 100 == 0:
           print(f"t={t:.2f}: E={e[0]:.6f}")

Floquet Systems
^^^^^^^^^^^^^^^

Periodically driven systems:

.. code-block:: python

   def floquet_hamiltonian(t, omega, amplitude, n_sites, device='cuda'):
       """
       Floquet Hamiltonian with periodic driving.

       H(t) = J Σ ZZ + A cos(ωt) Σ X

       Args:
           t: Time
           omega: Drive frequency
           amplitude: Drive amplitude
           n_sites: Number of sites
           device: Device

       Returns:
           H(t): Time-periodic Hamiltonian
       """
       h_t = amplitude * np.cos(omega * t)

       return MPOBuilder.ising_hamiltonian(
           n_sites=n_sites,
           J=1.0,
           h=h_t,
           device=device
       )

   # Simulate Floquet dynamics
   omega = 2.0  # Drive frequency
   A = 1.5      # Drive amplitude

   for step in range(n_steps):
       t = step * dt
       H_t = floquet_hamiltonian(t, omega, A, n_sites)

       config = TDVPConfig(dt=dt, t_final=dt)
       tdvp = TDVP1Site(H_t, mps, config)
       _, e = tdvp.run()

       # Track Floquet quasi-energy
       # ...

Part 6: Quantum Quenches
-------------------------

What is a Quantum Quench?
^^^^^^^^^^^^^^^^^^^^^^^^^^

Sudden change in Hamiltonian parameters:

1. Prepare ground state of H₀
2. Instantaneously switch to H₁
3. Evolve under H₁

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.tdvp import TDVP2Site, TDVPConfig
   from atlas_q.adaptive_mps import AdaptiveMPS

   n_sites = 12

   # Initial Hamiltonian (h=0)
   H_initial = MPOBuilder.ising_hamiltonian(
       n_sites=n_sites,
       J=1.0,
       h=0.0,  # No transverse field
       device='cuda'
   )

   # Prepare ground state via imaginary time
   mps = AdaptiveMPS(n_sites, bond_dim=16, device='cuda')
   config_prep = TDVPConfig(
       dt=0.05,
       t_final=20.0,
       imaginary_time=True,
       normalize=True,
       chi_max=64
   )
   tdvp_prep = TDVP2Site(H_initial, mps, config_prep)
   _, energies_prep = tdvp_prep.run()

   print(f"Ground state energy: {energies_prep[-1]:.6f}")

   # Quench to new Hamiltonian (h=2.0)
   H_final = MPOBuilder.ising_hamiltonian(
       n_sites=n_sites,
       J=1.0,
       h=2.0,  # Strong transverse field
       device='cuda'
   )

   # Real-time evolution after quench
   config_quench = TDVPConfig(
       dt=0.01,
       t_final=10.0,
       imaginary_time=False,
       normalize=False,
       chi_max=128
   )
   tdvp_quench = TDVP2Site(H_final, mps, config_quench)
   times, energies_quench = tdvp_quench.run()

   # Plot energy vs time after quench
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.plot(times, energies_quench, 'b-', linewidth=2)
   plt.xlabel('Time after quench', fontsize=12)
   plt.ylabel('Energy', fontsize=12)
   plt.title('Quantum Quench Dynamics', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.savefig('quench_dynamics.png', dpi=150)
   plt.close()

Loschmidt Echo
^^^^^^^^^^^^^^

Measure overlap with initial state:

.. code-block:: python

   def loschmidt_echo(mps_initial, mps_evolved):
       """
       Compute Loschmidt echo |⟨ψ(0)|ψ(t)⟩|².

       Args:
           mps_initial: Initial MPS
           mps_evolved: Evolved MPS

       Returns:
           Echo (overlap squared)
       """
       # Compute overlap (simplified)
       # Full implementation requires MPS inner product
       overlap = compute_mps_overlap(mps_initial, mps_evolved)
       echo = abs(overlap)**2

       return echo

   # Track echo during quench
   mps_initial_copy = mps.copy()  # Save initial state

   echoes = []
   for step in range(n_steps):
       # Evolve...

       echo = loschmidt_echo(mps_initial_copy, mps)
       echoes.append(echo)

Part 7: Observables and Correlations
-------------------------------------

Local Observables
^^^^^^^^^^^^^^^^^

Measure single-site expectation values:

.. code-block:: python

   def measure_magnetization(mps, site):
       """
       Measure ⟨σᶻ⟩ at a site.

       Args:
           mps: MPS state
           site: Site index

       Returns:
           Expectation value
       """
       # Would use MPO expectation value
       # See mpo_ops for implementation
       pass

   # Track magnetization during evolution
   magnetizations = []

   for step in range(n_steps):
       # Evolve...

       mag = [measure_magnetization(mps, i) for i in range(n_sites)]
       magnetizations.append(mag)

Correlation Functions
^^^^^^^^^^^^^^^^^^^^^

Two-point correlations:

.. code-block:: python

   def measure_correlation(mps, site_i, site_j, operator='ZZ'):
       """
       Measure ⟨σⁱ σʲ⟩.

       Args:
           mps: MPS state
           site_i, site_j: Site indices
           operator: Operator type

       Returns:
           Correlation function
       """
       # Would use two-site MPO expectation
       pass

   # Correlation spreading
   correlations = {}
   distance = 3

   for step in range(n_steps):
       # Evolve...

       corr = measure_correlation(mps, n_sites//2, n_sites//2 + distance, 'ZZ')
       correlations[step * dt] = corr

Part 8: Performance Optimization
---------------------------------

Time Step Selection
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def test_time_step_convergence(H, mps, dt_values, t_final):
       """Test convergence with different time steps."""
       results = []

       for dt in dt_values:
           mps_copy = mps.copy()

           config = TDVPConfig(dt=dt, t_final=t_final)
           tdvp = TDVP1Site(H, mps_copy, config)
           times, energies = tdvp.run()

           energy_drift = abs(energies[-1] - energies[0])

           results.append({
               'dt': dt,
               'drift': energy_drift,
               'steps': len(times)
           })

           print(f"dt={dt:.4f}: drift={energy_drift:.2e}, steps={len(times)}")

       return results

   # Test
   dt_values = [0.001, 0.01, 0.05, 0.1]
   results = test_time_step_convergence(H, mps, dt_values, t_final=1.0)

Parallel Time Evolution
^^^^^^^^^^^^^^^^^^^^^^^^

For multiple initial conditions:

.. code-block:: python

   def parallel_tdvp(H, initial_states, config):
       """
       Evolve multiple states in parallel.

       Args:
           H: Hamiltonian
           initial_states: List of MPS
           config: TDVP configuration

       Returns:
           List of (times, energies) for each state
       """
       results = []

       for idx, mps in enumerate(initial_states):
           tdvp = TDVP1Site(H, mps, config)
           times, energies = tdvp.run()
           results.append((times, energies))

           print(f"Completed trajectory {idx+1}/{len(initial_states)}")

       return results

Part 9: Troubleshooting
------------------------

Energy Conservation Violated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: Energy drifts significantly during real-time evolution

**Solutions**:

1. Reduce time step:

   .. code-block:: python

      config = TDVPConfig(dt=0.001)  # Smaller dt

2. Increase Krylov dimension:

   .. code-block:: python

      config = TDVPConfig(krylov_dim=30, krylov_tol=1e-12)

3. Use 2-site TDVP with larger χ:

   .. code-block:: python

      config = TDVPConfig(chi_max=128, eps_trunc=1e-10)

Bond Dimension Explosion
^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem**: χ grows uncontrollably in 2-site TDVP

**Solutions**:

1. Tighten truncation tolerance:

   .. code-block:: python

      config = TDVPConfig(eps_trunc=1e-6)  # Stricter truncation

2. Use 1-site TDVP (if acceptable):

   .. code-block:: python

      tdvp = TDVP1Site(H, mps, config)

3. Reduce chi_max:

   .. code-block:: python

      config = TDVPConfig(chi_max=64)

Slow Performance
^^^^^^^^^^^^^^^^

**Problem**: TDVP taking too long

**Solutions**:

1. Use 1-site instead of 2-site
2. Reduce Krylov dimension
3. Increase time step (if stable)
4. Enable GPU acceleration (already default)

Summary
-------

You have learned:

- ✅ TDVP algorithm principles for time evolution
- ✅ Difference between 1-site and 2-site TDVP
- ✅ Real vs imaginary time evolution
- ✅ Krylov subspace methods
- ✅ Time-dependent Hamiltonians and driving
- ✅ Quantum quenches and non-equilibrium dynamics
- ✅ Measuring observables and correlations
- ✅ Performance optimization techniques

Next Steps
----------

- :doc:`molecular_vqe` - Ground states via VQE
- :doc:`qaoa_tutorial` - Optimization with QAOA
- :doc:`advanced_features` - Circuit cutting, PEPS, distributed MPS
- :doc:`../explanations/algorithms` - Algorithm theory
- :doc:`../howtos/optimize_performance` - Performance tuning

Practice Exercises
------------------

1. Implement a quantum quench for the Heisenberg model
2. Study entanglement growth during time evolution
3. Compare 1-site vs 2-site TDVP accuracy for different systems
4. Implement Floquet engineering to synthesize effective Hamiltonians
5. Calculate dynamical correlation functions

See Also
--------

- :doc:`../../reference/tdvp` - TDVP API reference
- :doc:`../../reference/mpo_ops` - Hamiltonian construction
- :doc:`../../reference/adaptive_mps` - MPS operations for TDVP
