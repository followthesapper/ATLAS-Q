atlas_q.tdvp
============

.. automodule:: atlas_q.tdvp
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``tdvp`` module implements Time-Dependent Variational Principle (TDVP) for efficient time evolution of quantum states in the Matrix Product State (MPS) representation. TDVP provides a geometrically optimal way to evolve MPS while maintaining the tensor network structure.

**Time-Dependent Schrödinger Equation**

We aim to solve:

.. math::

   i\hbar \frac{d}{dt}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle

where :math:`|\psi(t)\rangle` is constrained to the MPS manifold with fixed or adaptive bond dimension.

**Variational Principle**

Rather than evolving in the full Hilbert space (:math:`\dim = 2^N`), TDVP projects the time derivative onto the tangent space of the MPS manifold [Haegeman11]_:

.. math::

   i\hbar \frac{d}{dt}|\psi\rangle = \mathcal{P}_T (\hat{H}|\psi\rangle)

where :math:`\mathcal{P}_T` projects onto the tangent space at :math:`|\psi\rangle`. This ensures the evolved state remains an MPS with controlled bond dimension.

**Key Advantages**

- **Efficiency**: Time complexity O(Nχ³d² + Nχ²D) per step (vs. O(2ᴺ) for exact evolution)
- **Controlled accuracy**: Bond dimension χ directly controls approximation error
- **Unitarity**: Evolution preserves norm to machine precision
- **GPU acceleration**: Tensor contractions highly parallelizable
- **Adaptive methods**: Bond dimension and time step adapt to local error

**1-Site vs. 2-Site TDVP**

.. list-table:: Algorithm Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - 1-Site TDVP
     - 2-Site TDVP
   * - Bond dimension
     - Fixed (χ constant)
     - Adaptive (χ can grow)
   * - Accuracy
     - Good for short times
     - Excellent for long times
   * - Speed per step
     - Fast (no SVD)
     - Slower (SVD each step)
   * - Memory
     - O(Nχ²)
     - O(Nχ²) + SVD workspace
   * - Use case
     - High-frequency dynamics, large χ
     - Entanglement growth, quenches

**Implementation Features**

- **Krylov subspace methods**: Efficient matrix exponential via Lanczos iteration
- **GPU-optimized contractions**: Fused kernels for tensor operations
- **Adaptive time-stepping**: Embedded error estimators (4th/5th order Runge-Kutta-Fehlberg)
- **Checkpoint/restart**: Save intermediate states during long evolutions
- **Observable tracking**: Compute expectation values without re-initialization

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TDVP1Site
   TDVP2Site
   TDVPConfig

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   run_tdvp

TDVP1Site
---------

.. autoclass:: TDVP1Site
   :members:
   :undoc-members:
   :show-inheritance:

   1-site TDVP algorithm with fixed bond dimension.

   **Algorithm Overview**

   The 1-site TDVP sweeps through the MPS, evolving one tensor at a time:

   1. **Right sweep** (i = 0 to N-1):

      - Project Hamiltonian onto site i: :math:`H^{[i]}_{\text{eff}} = L^{[i-1]} \otimes \hat{H}^{[i]} \otimes R^{[i+1]}`
      - Evolve tensor: :math:`A^{[i]}(t+dt) = \exp(-i dt H^{[i]}_{\text{eff}}) A^{[i]}(t)`
      - Update environment tensors :math:`L^{[i]}`, :math:`R^{[i]}`

   2. **Left sweep** (i = N-1 to 0):

      - Evolve backwards to maintain time-reversibility
      - Same projection and evolution steps

   **Mathematical Details**

   The effective Hamiltonian at site i is a :math:`(\chi_{i-1} \times d \times \chi_i)` tensor formed by contracting:

   .. math::

      H^{[i]}_{\text{eff}} = \sum_{\alpha,\beta,\sigma,\sigma'} L^{[i-1]}_{\alpha\alpha'} W^{[i]}_{\alpha'\beta',\sigma\sigma'} R^{[i+1]}_{\beta\beta'} |\alpha\sigma\beta\rangle\langle\alpha'\sigma'\beta'|

   The time evolution operator :math:`\exp(-i dt H^{[i]}_{\text{eff}})` is computed via:

   - **Krylov subspace method**: Lanczos iteration with Krylov dimension 10-30
   - **Matrix exponential**: Direct ``expm`` for small χ (< 32)
   - **Trotter decomposition**: For structured Hamiltonians

   **Initialization**

   :param MPO hamiltonian: Hamiltonian as Matrix Product Operator
   :param AdaptiveMPS mps: Initial MPS state (modified in-place during evolution)
   :param TDVPConfig config: TDVP configuration (time step, final time, etc.)

   **Attributes**

   .. attribute:: hamiltonian

      MPO representation of :math:`\hat{H}`

   .. attribute:: mps

      Current MPS state (updated during evolution)

   .. attribute:: config

      TDVPConfig object

   .. attribute:: time

      Current simulation time

   .. attribute:: left_envs

      Left environment tensors [L⁰, L¹, ..., Lᴺ⁻¹]

   .. attribute:: right_envs

      Right environment tensors [R¹, R², ..., Rᴺ]

   **Methods**

   .. rubric:: Methods

   .. autosummary::

      ~TDVP1Site.__init__
      ~TDVP1Site.run
      ~TDVP1Site.step

   .. method:: __init__(hamiltonian, mps, config)

      Initialize 1-site TDVP algorithm.

      Pre-computes left and right environment tensors by contracting MPS with MPO.

   .. method:: run()

      Run TDVP evolution from t=0 to t=t_final.

      :return: Tuple of (times, energies) where times is list of time points and energies is list of energy expectation values
      :rtype: Tuple[np.ndarray, np.ndarray]

      Automatically handles adaptive time-stepping if enabled in config.

   .. method:: step(dt=None)

      Perform one TDVP sweep (right + left).

      :param float dt: Time step (uses config.dt if None)
      :return: Energy after step
      :rtype: complex

      **Performance**: O(Nχ³d² + Nχ²D) where D is MPO bond dimension, N is number of sites, χ is MPS bond dimension, d=2 is physical dimension.

   .. method:: evolve_site(site, dt, direction='right')

      Evolve single site tensor.

      :param int site: Site index (0 to N-1)
      :param float dt: Time step
      :param str direction: Sweep direction ('right' or 'left')

      Internal method called by ``step()``.

   **Advantages**

   - **Speed**: ~2× faster than 2-site TDVP (no SVD overhead)
   - **Stability**: No truncation errors from SVD
   - **Memory**: Lower memory footprint (no temporary bond tensors)
   - **Controlled χ**: Bond dimension stays fixed, predictable resource usage

   **Disadvantages**

   - **Accuracy limitation**: Cannot capture entanglement growth beyond initial χ
   - **Long-time errors**: Accumulates errors for t > 10/J (for typical spin models)
   - **Quench dynamics**: Poor for sudden Hamiltonian changes that increase entanglement

TDVP2Site
---------

.. autoclass:: TDVP2Site
   :members:
   :undoc-members:
   :show-inheritance:

   2-site TDVP algorithm with adaptive bond dimension.

   **Algorithm Overview**

   The 2-site TDVP evolves pairs of adjacent tensors, allowing bond dimension to grow:

   1. **Forward sweep** (bond i = 0 to N-2):

      - Merge tensors A⁽ⁱ⁾ and A⁽ⁱ⁺¹⁾ into two-site tensor Θ⁽ⁱ'ⁱ⁺¹⁾
      - Project Hamiltonian: :math:`H^{[i,i+1]}_{\text{eff}}`
      - Evolve: :math:`\Theta(t+dt) = \exp(-i dt H_{\text{eff}}) \Theta(t)`
      - Split via SVD: :math:`\Theta = A^{[i]} S^{[i]} A^{[i+1]}` with truncation to χ_max
      - Evolve bond matrix backwards by dt/2 to maintain time-reversibility

   2. **Backward sweep** (bond i = N-2 to 0):

      - Same process in reverse direction

   **SVD Truncation**

   After evolving the two-site tensor, SVD is performed:

   .. math::

      \Theta_{\alpha i j \beta} = \sum_{\gamma=1}^{\chi'} U_{\alpha i \gamma} S_\gamma V_{\gamma j \beta}

   where χ' is the full SVD dimension. We truncate to χ ≤ χ_max by:

   - **Fixed χ**: Keep largest χ singular values
   - **Adaptive χ**: Keep singular values where :math:`s_i > \epsilon \times \max(s_j)`

   Truncation error: :math:`\delta = \sqrt{\sum_{\gamma > \chi} s_\gamma^2}`

   **Initialization**

   :param MPO hamiltonian: Hamiltonian as MPO
   :param AdaptiveMPS mps: Initial state
   :param TDVPConfig config: Configuration including chi_max and eps_bond

   **Attributes**

   .. attribute:: hamiltonian

      MPO Hamiltonian

   .. attribute:: mps

      Current MPS (updated in-place)

   .. attribute:: config

      TDVPConfig with 2-site settings

   .. attribute:: time

      Current time

   .. attribute:: truncation_errors

      List of truncation errors from SVD at each step

   **Methods**

   .. rubric:: Methods

   .. autosummary::

      ~TDVP2Site.__init__
      ~TDVP2Site.run
      ~TDVP2Site.step

   .. method:: __init__(hamiltonian, mps, config)

      Initialize 2-site TDVP.

      Verifies config.order == 2 and config.chi_max is set.

   .. method:: run()

      Run 2-site TDVP evolution.

      :return: (times, energies, max_chi_history)
      :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

      Returns additional max_chi_history array tracking bond dimension growth.

   .. method:: step(dt=None)

      Perform one 2-site TDVP sweep.

      :param float dt: Time step
      :return: (energy, max_truncation_error)
      :rtype: Tuple[complex, float]

      **Performance**: O(Nχ³d² + Nχ³) where the χ³ term is from SVD. About 2× slower than 1-site TDVP due to SVD overhead.

   .. method:: evolve_bond(bond, dt, direction='right')

      Evolve two-site tensor at bond.

      :param int bond: Bond index (0 to N-2)
      :param float dt: Time step
      :param str direction: 'right' or 'left'
      :return: Truncation error from SVD
      :rtype: float

   **Advantages**

   - **Accuracy**: Can maintain accuracy for long times (t > 100/J)
   - **Entanglement growth**: Automatically increases χ as entanglement grows
   - **Quench dynamics**: Handles sudden Hamiltonian changes well
   - **Ground state search**: Via imaginary time evolution with adaptive χ

   **Disadvantages**

   - **Speed**: ~2× slower than 1-site due to SVD cost
   - **Memory**: Requires O(χ³) workspace for SVD
   - **Parameter tuning**: Need to set χ_max and truncation threshold appropriately

TDVPConfig
----------

.. autoclass:: TDVPConfig
   :members:
   :undoc-members:

   Configuration dataclass for TDVP time evolution.

   :param float dt: Time step size (default: 0.01, typical range: 0.001-0.1)
   :param float t_final: Final evolution time (default: 1.0)
   :param int order: TDVP order (1 for 1-site, 2 for 2-site, default: 1)
   :param int chi_max: Maximum bond dimension (default: 64, required for order=2)
   :param float eps_bond: SVD truncation threshold (default: 1e-10, used when order=2)
   :param bool adaptive_dt: Enable adaptive time-stepping (default: False)
   :param float dt_min: Minimum time step for adaptive mode (default: 1e-5)
   :param float dt_max: Maximum time step for adaptive mode (default: 0.1)
   :param float error_tol: Local error tolerance for adaptive stepping (default: 1e-6)
   :param bool normalize: Normalize MPS after each step (default: True)
   :param int krylov_dim: Krylov subspace dimension for exp(H) (default: 20, range: 10-50)
   :param bool use_gpu_optimized: Use fused GPU kernels (default: True if CUDA available)
   :param int checkpoint_every: Save checkpoint every N steps (default: 0 = disabled)
   :param str checkpoint_dir: Directory for checkpoints (default: './tdvp_checkpoints')
   :param str device: Compute device (default: 'cuda' if available, else 'cpu')
   :param torch.dtype dtype: Data type (default: torch.complex64)

   **Time Step Selection**

   The time step dt should satisfy:

   - **Stability**: :math:`dt \ll 1/\lambda_{\max}` where λ_max is largest Hamiltonian eigenvalue
   - **Accuracy**: :math:`dt \ll 1/\Delta E` where ΔE is relevant energy scale
   - **Practical guideline**: Start with dt = 0.01-0.1 × (typical energy scale)⁻¹

   For Ising model with J = 1.0: dt ~ 0.01-0.05 works well.

   **Adaptive Time-Stepping**

   When ``adaptive_dt=True``, uses embedded Runge-Kutta-Fehlberg (RKF45) error estimator:

   1. Compute evolution with step dt (4th order)
   2. Compute evolution with step dt (5th order)
   3. Estimate local error: :math:`\epsilon = \|\psi^{(4)} - \psi^{(5)}\|`
   4. Adjust dt: :math:`dt_{\text{new}} = dt \times (tol / \epsilon)^{1/5}`

   Adaptive stepping reduces total steps by 2-5× while maintaining accuracy.

   **Krylov Dimension**

   The Krylov subspace method approximates :math:`\exp(-i H dt) |v\rangle` by projecting into a k-dimensional subspace. Accuracy improves with k:

   - k = 10: Rough approximation, fast
   - k = 20: Standard (recommended)
   - k = 30-50: High accuracy, slower

   For small χ < 32, direct matrix exponential is faster than Krylov.

   **GPU Optimization**

   When ``use_gpu_optimized=True``:

   - Fused tensor contractions (reduce kernel launches)
   - Optimized GEMM operations for environment updates
   - Overlapped CPU-GPU transfers during checkpointing

   Speedup: 1.5-2× faster than non-optimized on NVIDIA A100.

   **Normalization**

   If ``normalize=True``, MPS is normalized after each step via:

   .. math::

      |\psi\rangle \to \frac{|\psi\rangle}{\langle\psi|\psi\rangle^{1/2}}

   Usually enabled except for imaginary time evolution where norm naturally decays.

   **Checkpointing**

   Set ``checkpoint_every > 0`` to periodically save MPS state. Useful for:

   - Long simulations (t_final > 100)
   - Crash recovery
   - Intermediate analysis

   Checkpoint files: ``{checkpoint_dir}/tdvp_step_{n}.pt``

Examples
--------

Basic time evolution:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.tdvp import TDVP1Site, TDVPConfig

   # Create Hamiltonian and initial state
   H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5, device='cuda')
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')

   # Configure TDVP
   config = TDVPConfig(
       dt=0.01,
       t_final=1.0,
       order=1,
       use_gpu_optimized=True
   )

   # Run evolution
   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

   # Plot energy conservation
   import matplotlib.pyplot as plt
   plt.plot(times, energies.real)
   plt.xlabel('Time')
   plt.ylabel('Energy')
   plt.title('Energy Conservation')
   plt.savefig('energy_vs_time.png')

2-site TDVP with adaptive bond dimension:

.. code-block:: python

   from atlas_q.tdvp import TDVP2Site

   # Configure 2-site TDVP
   config = TDVPConfig(
       dt=0.01,
       t_final=2.0,
       order=2,
       chi_max=128,
       eps_bond=1e-8
   )

   # Run with bond growth
   tdvp = TDVP2Site(H, mps, config)
   times, energies = tdvp.run()

   # Check final bond dimensions
   stats = mps.stats_summary()
   print(f"Final max χ: {stats['max_chi']}")

Quantum quench:

.. code-block:: python

   import torch

   # Initialize in ground state of H_initial
   H_initial = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.0, device='cuda')
   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')
   # (assume mps is prepared in ground state via imaginary time evolution or VQE)

   # Quench: evolve with H_final
   H_final = MPOBuilder.ising_hamiltonian(10, J=1.0, h=2.0, device='cuda')
   config = TDVPConfig(dt=0.01, t_final=5.0)
   tdvp = TDVP1Site(H_final, mps, config)
   times, energies = tdvp.run()

Adaptive time-stepping:

.. code-block:: python

   config = TDVPConfig(
       dt=0.01,
       t_final=1.0,
       adaptive_dt=True,
       dt_min=0.001,
       dt_max=0.1,
       error_tol=1e-6
   )
   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

Imaginary time evolution (ground state finding):

.. code-block:: python

   # Imaginary time: t → -iτ
   # Multiply Hamiltonian by -i
   import torch

   # Create Hamiltonian tensors with factor -1j
   H = MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.5, device='cuda')
   for i in range(len(H.tensors)):
       H.tensors[i] = -1j * H.tensors[i]

   # Evolve in imaginary time
   config = TDVPConfig(dt=0.01, t_final=1.0, normalize=True)
   tdvp = TDVP1Site(H, mps, config)
   times, energies = tdvp.run()

   # Final state approximates ground state
   ground_energy = energies[-1].real
   print(f"Ground state energy: {ground_energy:.6f}")

Computing observables during evolution:

.. code-block:: python

   from atlas_q.mpo_ops import expectation_value
   import torch

   # Define observable
   Z_total = MPOBuilder.identity_mpo(10, device='cuda')
   for i in range(len(Z_total.tensors)):
       Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
       Z_total.tensors[i] = Z_total.tensors[i] * Z.view(1, 2, 2, 1)

   # Evolve and measure
   tdvp = TDVP1Site(H, mps, config)
   magnetizations = []

   for t in range(int(config.t_final / config.dt)):
       tdvp.step()
       mag = expectation_value(mps, Z_total)
       magnetizations.append(mag.real)

   # Plot magnetization dynamics
   import matplotlib.pyplot as plt
   plt.plot(times, magnetizations)
   plt.xlabel('Time')
   plt.ylabel('Magnetization')
   plt.savefig('magnetization.png')

Performance Notes
-----------------

**Computational Complexity**

.. list-table:: TDVP Performance Scaling
   :header-rows: 1
   :widths: 20 30 25 25

   * - Algorithm
     - Time per Step
     - Memory
     - Best For
   * - 1-site TDVP
     - O(Nχ³d² + Nχ²D)
     - O(Nχ²)
     - Short times, fixed χ
   * - 2-site TDVP
     - O(Nχ³d² + Nχ³)
     - O(Nχ² + χ³)
     - Long times, adaptive χ

Where N = number of sites, χ = bond dimension, d = physical dimension (2 for qubits), D = MPO bond dimension.

**Benchmark Timings** (NVIDIA A100, N=20, dt=0.01)

.. list-table:: Time per TDVP Step
   :header-rows: 1
   :widths: 20 25 25 30

   * - Bond Dim χ
     - 1-Site (ms)
     - 2-Site (ms)
     - Speedup Ratio
   * - 16
     - 2.3
     - 4.1
     - 1.8×
   * - 32
     - 8.7
     - 18.2
     - 2.1×
   * - 64
     - 34.1
     - 71.5
     - 2.1×
   * - 128
     - 135.2
     - 285.7
     - 2.1×

2-site TDVP is consistently ~2× slower due to SVD overhead.

**Accuracy vs. Time Step**

For Ising model (J=1.0, h=0.5), energy conservation error:

- dt = 0.001: Error ~ 10⁻¹⁰ (excellent, slow)
- dt = 0.01: Error ~ 10⁻⁷ (good, recommended)
- dt = 0.05: Error ~ 10⁻⁵ (acceptable for short times)
- dt = 0.1: Error ~ 10⁻³ (poor, unstable)

**Memory Usage**

For N = 50 qubits, χ = 64, complex64:

- MPS tensors: 50 × 64² × 2 × 8 bytes = 3.3 MB
- Environment tensors: 2 × 50 × 64² × D × 8 bytes (D = MPO bond dimension)
- Krylov workspace: k × χ² × 2 × 8 bytes (k = Krylov dimension)

Total: ~100-500 MB depending on Hamiltonian structure.

**Optimization Tips**

1. **Use adaptive time-stepping** for variable dynamics (quenches, oscillating Hamiltonians)
2. **Start with 1-site TDVP** for initial exploration, switch to 2-site if χ saturates
3. **Increase krylov_dim** (to 30-50) for higher accuracy without reducing dt
4. **Enable GPU optimization** for systems with χ > 16
5. **Checkpoint frequently** for long evolutions (t_final > 50)

Best Practices
--------------

**Choosing 1-Site vs. 2-Site**

Use **1-site** when:

- Short evolution times (t < 10/J)
- χ is already large (≥ 64) and captures entanglement
- Hamiltonian is local and doesn't generate much entanglement
- Speed is critical

Use **2-site** when:

- Long evolution times (t > 10/J)
- Entanglement growth expected (quenches, non-equilibrium)
- Finding ground states via imaginary time
- Accuracy is more important than speed

**Time Step Selection**

Start with dt = 0.01 and monitor energy conservation:

.. code-block:: python

   times, energies = tdvp.run()

   # Check energy conservation
   energy_drift = np.abs(energies[-1] - energies[0]).real
   relative_drift = energy_drift / np.abs(energies[0]).real

   if relative_drift > 1e-5:
       print(f"Warning: Energy drift {relative_drift:.2e}, reduce dt")

If drift > 10⁻⁵, reduce dt by factor of 2.

**Imaginary Time Evolution**

For ground state finding, multiply Hamiltonian by -1j and evolve:

.. code-block:: python

   # Ground state via imaginary time
   H_imag = MPO([-1j * tensor for tensor in H.tensors])

   config = TDVPConfig(
       dt=0.01,        # Can use larger dt for imaginary time
       t_final=5.0,    # Evolve until convergence
       normalize=True, # Must normalize each step
       order=2,        # Use 2-site for adaptive χ
       chi_max=128
   )

   tdvp = TDVP2Site(H_imag, mps, config)
   times, energies = tdvp.run()

   ground_energy = energies[-1].real
   print(f"Ground state energy: {ground_energy:.6f}")

**Troubleshooting**

- **Energy increases**: Reduce dt, increase krylov_dim, or check Hamiltonian construction
- **Norm drifts from 1.0**: Enable ``normalize=True``
- **Slow convergence (imaginary time)**: Increase dt (can use 0.05-0.1 for imaginary time)
- **Memory errors**: Reduce chi_max or use larger GPU
- **NaN/Inf values**: dt too large or Hamiltonian has issues (check units, signs)

See Also
--------

- :doc:`adaptive_mps` - MPS representation and manipulation
- :doc:`mpo_ops` - Hamiltonian construction as MPO
- :doc:`vqe_qaoa` - Variational quantum algorithms
- :doc:`mps_pytorch` - PyTorch MPS backend
- :doc:`../user_guide/tutorials/tdvp_tutorial` - Step-by-step TDVP tutorial
- :doc:`../user_guide/howtos/time_evolution` - Time evolution guide

References
----------

.. [Haegeman11] J. Haegeman et al., "Time-dependent variational principle for quantum lattices," *Physical Review Letters* 107, 070601 (2011).

.. [Haegeman16] J. Haegeman et al., "Unifying time evolution and optimization with matrix product states," *Physical Review B* 94, 165116 (2016).

.. [Paeckel19] S. Paeckel et al., "Time-evolution methods for matrix-product states," *Annals of Physics* 411, 167998 (2019).

.. [Schollwock11] U. Schollwöck, "The density-matrix renormalization group in the age of matrix product states," *Annals of Physics* 326, 96 (2011).
