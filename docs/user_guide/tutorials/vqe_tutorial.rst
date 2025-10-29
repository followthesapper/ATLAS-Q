VQE Tutorial
=============

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm for finding ground state energies of quantum systems. This tutorial provides comprehensive coverage of VQE implementation, ansatz design, optimization strategies, and practical applications.

Prerequisites
-------------

- Completed :doc:`beginners` and :doc:`mps_basics` tutorials
- Understanding of quantum gates and circuits
- Familiarity with Hamiltonians and expectation values
- Basic knowledge of optimization algorithms
- Linear algebra background

Learning Objectives
-------------------

After completing this tutorial, you will understand:

1. VQE algorithm principles and workflow
2. Ansatz design and selection criteria
3. Classical optimization methods for VQE
4. Gradient computation techniques
5. Convergence monitoring and analysis
6. Noise-aware VQE strategies
7. Hardware-efficient circuit design
8. Practical VQE applications

Part 1: VQE Fundamentals
-------------------------

What is VQE?
^^^^^^^^^^^^

VQE finds the ground state energy of a Hamiltonian H by optimizing a parameterized quantum state:

.. math::

   E_0 \approx \min_{\vec{\theta}} \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle

The algorithm alternates between:

1. **Quantum**: Prepare state |ψ(θ)⟩ and measure ⟨H⟩
2. **Classical**: Update parameters θ to minimize ⟨H⟩

Algorithm Workflow
^^^^^^^^^^^^^^^^^^

.. code-block:: text

   1. Initialize parameters θ
   2. Prepare ansatz state |ψ(θ)⟩
   3. Measure energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
   4. Update θ using classical optimizer
   5. Repeat until convergence

Why VQE Works
^^^^^^^^^^^^^

The variational principle guarantees:

.. math::

   E(\vec{\theta}) = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle \geq E_0

Any parameterized state gives an upper bound on the ground state energy. By minimizing E(θ), we approach E₀.

Basic VQE Implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import torch

   # Build Hamiltonian - transverse field Ising model
   n_sites = 8
   H = MPOBuilder.ising_hamiltonian(
       n_sites=n_sites,
       J=1.0,      # Coupling strength
       h=0.5,      # Transverse field
       device='cuda'
   )

   # Configure VQE
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       optimizer='L-BFGS-B',
       max_iter=100,
       tol=1e-6
   )

   # Run optimization
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"Ground state energy: {energy:.6f}")
   print(f"Number of parameters: {len(params)}")
   print(f"Optimization iterations: {len(vqe.energies)}")

Expected output:

.. code-block:: text

   Ground state energy: -9.234567
   Number of parameters: 48
   Optimization iterations: 23

Part 2: Ansatz Design
----------------------

What is an Ansatz?
^^^^^^^^^^^^^^^^^^

An ansatz is a parameterized quantum circuit that prepares trial states. Good ansätze balance:

- **Expressivity**: Can represent states close to ground state
- **Trainability**: Gradients don't vanish
- **Efficiency**: Shallow enough for NISQ devices

Hardware-Efficient Ansatz
^^^^^^^^^^^^^^^^^^^^^^^^^^

Designed for near-term hardware with limited gate fidelity:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.mpo_ops import MPOBuilder

   # Hardware-efficient ansatz structure:
   # Layer = [R_y(θ) on all qubits] + [CNOT ladder] + [R_z(θ) on all qubits]

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       entangling_gate='cnot',  # Options: 'cnot', 'cz', 'swap'
       device='cuda'
   )

   H = MPOBuilder.heisenberg_hamiltonian(n_sites=10, device='cuda')
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

Circuit structure visualization:

.. code-block:: text

   Layer 1:
   q0: ─Ry(θ₀)─●───────Rz(φ₀)─
   q1: ─Ry(θ₁)─┼───●───Rz(φ₁)─
   q2: ─Ry(θ₂)─X───┼───Rz(φ₂)─
   q3: ─Ry(θ₃)─────X───Rz(φ₃)─

   Repeat for n_layers

Number of parameters: n_sites × 2 × n_layers

UCCSD Ansatz
^^^^^^^^^^^^

Unitary Coupled-Cluster Singles and Doubles - chemistry-aware ansatz:

.. code-block:: python

   # UCCSD ansatz for molecular systems
   config = VQEConfig(
       ansatz='uccsd',
       n_electrons=4,      # Number of electrons
       n_orbitals=8,       # Number of spatial orbitals
       device='cuda'
   )

   # For H2 molecule
   from atlas_q.mpo_ops import molecular_hamiltonian_from_specs

   H_h2 = molecular_hamiltonian_from_specs(
       molecule='H2',
       basis='sto-3g',
       bond_length=0.74,  # Angstroms
       device='cuda'
   )

   vqe = VQE(H_h2, config)
   energy, params = vqe.optimize()

   print(f"H2 ground state energy: {energy:.6f} Hartree")

UCCSD provides chemical accuracy for small molecules but requires O(n⁴) parameters for n orbitals.

Custom Ansatz
^^^^^^^^^^^^^

Define custom circuit structure:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import numpy as np

   def custom_ansatz(n_qubits, params, device='cuda'):
       """
       Custom ansatz with alternating RX and RZ rotations.

       Args:
           n_qubits: Number of qubits
           params: Parameter array (length = 2*n_qubits + n_qubits-1)
           device: Computation device

       Returns:
           AdaptiveMPS with ansatz applied
       """
       mps = AdaptiveMPS(num_qubits=n_qubits, bond_dim=32, device=device)

       idx = 0

       # Layer 1: RX rotations
       for q in range(n_qubits):
           theta = params[idx]
           RX = torch.tensor([
               [np.cos(theta/2), -1j*np.sin(theta/2)],
               [-1j*np.sin(theta/2), np.cos(theta/2)]
           ], dtype=torch.complex64).to(device)
           mps.apply_single_qubit_gate(q, RX)
           idx += 1

       # Layer 2: CNOT ladder
       CNOT = torch.tensor([
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]
       ], dtype=torch.complex64).reshape(4, 4).to(device)

       for q in range(n_qubits - 1):
           mps.apply_two_site_gate(q, CNOT)

       # Layer 3: RZ rotations
       for q in range(n_qubits):
           phi = params[idx]
           RZ = torch.diag(torch.tensor(
               [1, np.exp(1j*phi)],
               dtype=torch.complex64
           )).to(device)
           mps.apply_single_qubit_gate(q, RZ)
           idx += 1

       return mps

   # Use custom ansatz with VQE
   # (Would need to wrap in VQE-compatible interface)

Ansatz Selection Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose ansatz based on problem:

.. code-block:: python

   def recommend_ansatz(problem_type, n_qubits):
       """Recommend ansatz based on problem characteristics."""

       if problem_type == 'chemistry':
           # Chemical systems - use UCCSD
           return 'uccsd'

       elif problem_type == 'optimization':
           # Combinatorial problems - use QAOA
           return 'qaoa'

       elif problem_type == 'general' and n_qubits <= 15:
           # Small systems - use hardware-efficient
           return 'hardware_efficient'

       elif problem_type == 'general' and n_qubits > 15:
           # Large systems - use shallow ansatz
           return 'hardware_efficient_shallow'

       else:
           return 'hardware_efficient'

   # Example
   ansatz = recommend_ansatz('chemistry', n_qubits=12)
   print(f"Recommended: {ansatz}")

Part 3: Classical Optimization
-------------------------------

Optimizer Selection
^^^^^^^^^^^^^^^^^^^

Different optimizers have different characteristics:

.. code-block:: python

   # Gradient-free optimizers
   config_nelder = VQEConfig(
       optimizer='Nelder-Mead',
       max_iter=500
   )

   config_cobyla = VQEConfig(
       optimizer='COBYLA',
       max_iter=500
   )

   # Gradient-based optimizers
   config_lbfgs = VQEConfig(
       optimizer='L-BFGS-B',
       max_iter=100
   )

   config_adam = VQEConfig(
       optimizer='Adam',
       learning_rate=0.01,
       max_iter=1000
   )

Optimizer comparison:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import time

   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')

   optimizers = ['Nelder-Mead', 'COBYLA', 'L-BFGS-B', 'Adam']
   results = []

   for opt_name in optimizers:
       config = VQEConfig(
           ansatz='hardware_efficient',
           n_layers=2,
           optimizer=opt_name,
           max_iter=100
       )

       vqe = VQE(H, config)
       start = time.time()
       energy, params = vqe.optimize()
       elapsed = time.time() - start

       results.append({
           'optimizer': opt_name,
           'energy': energy,
           'iterations': len(vqe.energies),
           'time': elapsed
       })

       print(f"{opt_name:15s}: E={energy:.6f}, "
             f"iter={len(vqe.energies)}, time={elapsed:.2f}s")

Expected output:

.. code-block:: text

   Nelder-Mead    : E=-13.456789, iter=87, time=12.34s
   COBYLA         : E=-13.456234, iter=93, time=13.21s
   L-BFGS-B       : E=-13.456891, iter=23, time=3.45s
   Adam           : E=-13.456543, iter=100, time=8.76s

Gradient Computation
^^^^^^^^^^^^^^^^^^^^

For gradient-based optimizers, compute gradients via parameter shift:

.. code-block:: python

   def parameter_shift_gradient(vqe, params, param_idx, shift=np.pi/2):
       """
       Compute gradient via parameter shift rule.

       ∂E/∂θᵢ = (E(θᵢ + π/2) - E(θᵢ - π/2)) / 2

       Args:
           vqe: VQE instance
           params: Current parameters
           param_idx: Index of parameter to differentiate
           shift: Shift amount (default π/2)

       Returns:
           Gradient for parameter param_idx
       """
       params_plus = params.copy()
       params_plus[param_idx] += shift

       params_minus = params.copy()
       params_minus[param_idx] -= shift

       energy_plus = vqe.compute_energy(params_plus)
       energy_minus = vqe.compute_energy(params_minus)

       gradient = (energy_plus - energy_minus) / 2

       return gradient

   # Compute full gradient
   def compute_full_gradient(vqe, params):
       """Compute gradient for all parameters."""
       gradient = np.zeros_like(params)

       for i in range(len(params)):
           gradient[i] = parameter_shift_gradient(vqe, params, i)

       return gradient

   # Example usage
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.ising_hamiltonian(n_sites=6, device='cuda')
   config = VQEConfig(ansatz='hardware_efficient', n_layers=2)
   vqe = VQE(H, config)

   # Initial parameters
   params = np.random.randn(24) * 0.1

   # Compute gradient
   grad = compute_full_gradient(vqe, params)
   print(f"Gradient norm: {np.linalg.norm(grad):.4f}")

Advanced: Natural Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Natural gradient accounts for parameter space geometry:

.. math::

   \Delta \vec{\theta} = -\eta F^{-1} \nabla E

where F is the quantum Fisher information matrix.

.. code-block:: python

   def natural_gradient_step(vqe, params, learning_rate=0.01):
       """
       Natural gradient descent step.

       Args:
           vqe: VQE instance
           params: Current parameters
           learning_rate: Step size

       Returns:
           Updated parameters
       """
       # Compute gradient
       grad = compute_full_gradient(vqe, params)

       # Compute Fisher information matrix (simplified)
       # Full implementation requires quantum Fisher information
       # Here we use identity as approximation
       F_inv = np.eye(len(params))

       # Natural gradient update
       natural_grad = F_inv @ grad
       params_new = params - learning_rate * natural_grad

       return params_new

Part 4: Convergence Monitoring
-------------------------------

Basic Monitoring
^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.ising_hamiltonian(n_sites=10, device='cuda')
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       optimizer='L-BFGS-B',
       max_iter=100,
       callback=True  # Enable iteration callbacks
   )

   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   # Access convergence history
   print(f"Initial energy: {vqe.energies[0]:.6f}")
   print(f"Final energy: {vqe.energies[-1]:.6f}")
   print(f"Energy improvement: {vqe.energies[0] - vqe.energies[-1]:.6f}")

   # Plot convergence
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.plot(vqe.energies, 'b-', linewidth=2)
   plt.xlabel('Iteration', fontsize=12)
   plt.ylabel('Energy', fontsize=12)
   plt.title('VQE Convergence', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.savefig('vqe_convergence.png', dpi=150)
   plt.close()

Advanced Monitoring
^^^^^^^^^^^^^^^^^^^

Track multiple metrics:

.. code-block:: python

   class VQEMonitor:
       """Monitor VQE optimization progress."""

       def __init__(self):
           self.energies = []
           self.grad_norms = []
           self.param_norms = []
           self.walltime = []
           self.start_time = None

       def callback(self, params, energy, gradient=None):
           """Called after each optimization step."""
           import time

           if self.start_time is None:
               self.start_time = time.time()

           self.energies.append(energy)
           self.param_norms.append(np.linalg.norm(params))
           self.walltime.append(time.time() - self.start_time)

           if gradient is not None:
               self.grad_norms.append(np.linalg.norm(gradient))

           # Print progress
           if len(self.energies) % 10 == 0:
               print(f"Iteration {len(self.energies)}: E={energy:.6f}, "
                     f"time={self.walltime[-1]:.2f}s")

       def plot_diagnostics(self, filename='vqe_diagnostics.png'):
           """Plot comprehensive diagnostics."""
           import matplotlib.pyplot as plt

           fig, axes = plt.subplots(2, 2, figsize=(12, 10))

           # Energy convergence
           axes[0, 0].plot(self.energies, 'b-')
           axes[0, 0].set_xlabel('Iteration')
           axes[0, 0].set_ylabel('Energy')
           axes[0, 0].set_title('Energy Convergence')
           axes[0, 0].grid(True, alpha=0.3)

           # Gradient norm
           if self.grad_norms:
               axes[0, 1].semilogy(self.grad_norms, 'r-')
               axes[0, 1].set_xlabel('Iteration')
               axes[0, 1].set_ylabel('Gradient Norm')
               axes[0, 1].set_title('Gradient Magnitude')
               axes[0, 1].grid(True, alpha=0.3)

           # Parameter norm
           axes[1, 0].plot(self.param_norms, 'g-')
           axes[1, 0].set_xlabel('Iteration')
           axes[1, 0].set_ylabel('Parameter Norm')
           axes[1, 0].set_title('Parameter Evolution')
           axes[1, 0].grid(True, alpha=0.3)

           # Wall time
           axes[1, 1].plot(self.walltime, self.energies, 'k-')
           axes[1, 1].set_xlabel('Time (s)')
           axes[1, 1].set_ylabel('Energy')
           axes[1, 1].set_title('Time vs Energy')
           axes[1, 1].grid(True, alpha=0.3)

           plt.tight_layout()
           plt.savefig(filename, dpi=150)
           plt.close()

   # Usage
   monitor = VQEMonitor()

   # Integrate with VQE
   # (Would need custom callback in VQE implementation)

Convergence Criteria
^^^^^^^^^^^^^^^^^^^^

Determine when to stop optimization:

.. code-block:: python

   def check_convergence(energies, tol=1e-6, window=5):
       """
       Check if VQE has converged.

       Criteria:
       1. Energy change < tol over window iterations
       2. No improvement for window iterations

       Args:
           energies: List of energies
           tol: Tolerance for energy change
           window: Number of iterations to check

       Returns:
           True if converged
       """
       if len(energies) < window + 1:
           return False

       # Check energy change
       recent_energies = energies[-window-1:]
       energy_changes = [abs(recent_energies[i+1] - recent_energies[i])
                        for i in range(window)]

       max_change = max(energy_changes)

       return max_change < tol

   # Example
   converged = check_convergence(vqe.energies, tol=1e-6, window=5)
   if converged:
       print("VQE has converged!")
   else:
       print("VQE has not converged, continue optimization")

Part 5: Noise-Aware VQE
------------------------

Error Mitigation
^^^^^^^^^^^^^^^^

Mitigate measurement errors:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Enable error mitigation
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=2,
       error_mitigation='zero_noise_extrapolation',
       noise_factors=[1.0, 1.5, 2.0],  # Noise scaling factors
       device='cuda'
   )

   H = MPOBuilder.ising_hamiltonian(n_sites=8, device='cuda')
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"Mitigated energy: {energy:.6f}")

Noisy Simulation
^^^^^^^^^^^^^^^^

Simulate VQE under noise:

.. code-block:: python

   from atlas_q.noise_models import DepolarizingNoise
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Create noise model
   noise = DepolarizingNoise(
       single_qubit_error=0.001,  # 0.1% error per gate
       two_qubit_error=0.01        # 1% error per two-qubit gate
   )

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=2,
       noise_model=noise,
       device='cuda'
   )

   H = MPOBuilder.ising_hamiltonian(n_sites=8, device='cuda')

   # Compare noiseless vs noisy
   vqe_ideal = VQE(H, VQEConfig(ansatz='hardware_efficient', n_layers=2))
   energy_ideal, _ = vqe_ideal.optimize()

   vqe_noisy = VQE(H, config)
   energy_noisy, _ = vqe_noisy.optimize()

   print(f"Ideal energy: {energy_ideal:.6f}")
   print(f"Noisy energy: {energy_noisy:.6f}")
   print(f"Error: {abs(energy_noisy - energy_ideal):.6f}")

Part 6: Practical Applications
-------------------------------

Ising Model Ground State
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # 1D Ising chain with periodic boundary conditions
   n_sites = 12
   H_ising = MPOBuilder.ising_hamiltonian(
       n_sites=n_sites,
       J=-1.0,      # Ferromagnetic coupling
       h=0.5,       # Transverse field
       periodic=True,
       device='cuda'
   )

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=4,
       optimizer='L-BFGS-B',
       max_iter=200
   )

   vqe = VQE(H_ising, config)
   energy, params = vqe.optimize()

   # Compare to exact (for small systems)
   # exact_energy = compute_exact_ground_state(H_ising)
   # error = abs(energy - exact_energy)

   print(f"VQE ground state energy: {energy:.6f}")

Heisenberg Model
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Heisenberg XXZ model
   H_heisenberg = MPOBuilder.heisenberg_hamiltonian(
       n_sites=10,
       Jx=1.0,
       Jy=1.0,
       Jz=1.5,  # Anisotropic
       device='cuda'
   )

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=5,
       optimizer='L-BFGS-B'
   )

   vqe = VQE(H_heisenberg, config)
   energy, params = vqe.optimize()

   print(f"Heisenberg ground state energy: {energy:.6f}")

Custom Hamiltonian
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   # Build custom spin chain
   builder = MPOBuilder(n_sites=8, device='cuda')

   # Add terms manually
   for i in range(7):
       # ZZ interaction
       builder.add_two_site_term(i, i+1, 'ZZ', coefficient=-1.0)

       # X field
       builder.add_single_site_term(i, 'X', coefficient=0.5)

   # Add final site X field
   builder.add_single_site_term(7, 'X', coefficient=0.5)

   H_custom = builder.build()

   vqe = VQE(H_custom, VQEConfig(ansatz='hardware_efficient', n_layers=3))
   energy, params = vqe.optimize()

   print(f"Custom Hamiltonian ground state: {energy:.6f}")

Part 7: Troubleshooting
------------------------

Convergence Issues
^^^^^^^^^^^^^^^^^^

**Problem**: VQE not converging or stuck at local minimum

**Solutions**:

1. Try different random initializations:

   .. code-block:: python

      best_energy = float('inf')
      best_params = None

      for trial in range(10):
          vqe = VQE(H, config)
          energy, params = vqe.optimize()

          if energy < best_energy:
              best_energy = energy
              best_params = params

      print(f"Best energy over 10 trials: {best_energy:.6f}")

2. Increase ansatz depth:

   .. code-block:: python

      config = VQEConfig(ansatz='hardware_efficient', n_layers=5)  # More layers

3. Use different optimizer:

   .. code-block:: python

      config = VQEConfig(optimizer='Nelder-Mead')  # Gradient-free

Barren Plateaus
^^^^^^^^^^^^^^^

**Problem**: Gradients vanish with deep circuits

**Solutions**:

1. Use hardware-efficient ansatz with shallow depth
2. Initialize parameters near identity (small values)
3. Use layer-wise training

Poor Accuracy
^^^^^^^^^^^^^

**Problem**: VQE energy far from true ground state

**Solutions**:

1. Increase bond dimension for MPS simulation:

   .. code-block:: python

      # Higher χ for more accurate MPS
      mps = AdaptiveMPS(num_qubits=n, bond_dim=128, device='cuda')

2. Tighten convergence criteria:

   .. code-block:: python

      config = VQEConfig(tol=1e-8, max_iter=500)

3. Use chemistry-aware ansatz for molecules (UCCSD)

Summary
-------

You have learned:

- ✅ VQE algorithm principles and variational principle
- ✅ Ansatz design (hardware-efficient, UCCSD, custom)
- ✅ Classical optimizer selection and configuration
- ✅ Gradient computation via parameter shift
- ✅ Convergence monitoring and diagnostics
- ✅ Noise-aware VQE and error mitigation
- ✅ Practical applications to spin models
- ✅ Troubleshooting common issues

Next Steps
----------

- :doc:`molecular_vqe` - Apply VQE to quantum chemistry
- :doc:`qaoa_tutorial` - Combinatorial optimization with QAOA
- :doc:`tdvp_tutorial` - Time evolution methods
- :doc:`../howtos/optimize_performance` - Performance optimization
- :doc:`../explanations/algorithms` - Algorithm theory

Practice Exercises
------------------

1. Implement VQE for the XY model and compare to Heisenberg
2. Test different optimizers and compare convergence rates
3. Implement natural gradient descent
4. Study how ansatz depth affects accuracy vs convergence
5. Compare VQE to exact diagonalization for small systems

See Also
--------

- :doc:`../../reference/vqe_qaoa` - VQE/QAOA API reference
- :doc:`../../reference/mpo_ops` - Hamiltonian construction
- :doc:`../../reference/adaptive_mps` - MPS for VQE state preparation
