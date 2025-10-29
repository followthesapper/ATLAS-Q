Noise Models
============

NISQ-era quantum noise simulation with Kraus operators and density matrix evolution.

.. currentmodule:: atlas_q.noise_models

Overview
--------

The ``noise_models`` module implements realistic quantum noise channels for **NISQ (Noisy Intermediate-Scale Quantum)** device simulation. Noise is essential for understanding the behavior of real quantum hardware, which suffers from decoherence, gate errors, and measurement errors.

Key Features
~~~~~~~~~~~~

- **Physical noise models**: T1 relaxation, T2 dephasing, thermal noise
- **Gate error models**: Depolarizing, bit flip, phase flip
- **Kraus representation**: Efficient superoperator formulation
- **MPO compatibility**: Noise channels as Matrix Product Operators
- **GPU acceleration**: CUDA-enabled noise application
- **Stochastic sampling**: Monte Carlo trajectory simulation

NISQ devices are characterized by:

1. **Limited coherence**: T1 (50-150 μs), T2 (30-100 μs)
2. **Gate errors**: 0.1-1% for single-qubit gates, 1-5% for two-qubit gates
3. **Measurement errors**: 1-5% readout error
4. **No error correction**: Too few qubits for QEC codes

ATLAS-Q simulates these effects using **quantum channels** represented as Kraus operators.

Mathematical Background
-----------------------

Quantum Channels
~~~~~~~~~~~~~~~~

A quantum channel Φ maps density matrices to density matrices:

.. math::

   \Phi: \rho \mapsto \sum_{i=1}^k K_i \rho K_i^\dagger

where :math:`\{K_i\}` are **Kraus operators** satisfying the completeness relation:

.. math::

   \sum_{i=1}^k K_i^\dagger K_i = I

This ensures trace preservation: :math:`\text{Tr}(\Phi(\rho)) = \text{Tr}(\rho) = 1`.

Depolarizing Channel
~~~~~~~~~~~~~~~~~~~~~

The depolarizing channel with probability p:

.. math::

   \Phi_{\text{depol}}(\rho) = (1-p)\rho + \frac{p}{d}I

For 1-qubit (d=2):

.. math::

   \Phi(\rho) = (1-p)\rho + \frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z + \rho)

**Kraus operators** (stochastic Pauli):

.. math::

   K_0 = \sqrt{1-\frac{3p}{4}}I, \quad K_1 = \sqrt{\frac{p}{4}}X, \quad K_2 = \sqrt{\frac{p}{4}}Y, \quad K_3 = \sqrt{\frac{p}{4}}Z

Amplitude Damping (T1)
~~~~~~~~~~~~~~~~~~~~~~

Models energy relaxation from |1⟩ to |0⟩ with rate γ = 1/T1:

.. math::

   K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad
   K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}

where γ = 1 - exp(-gate_time/T1).

Dephasing (T2)
~~~~~~~~~~~~~~

Models phase randomization without energy loss:

.. math::

   K_0 = \sqrt{1-\lambda} I, \quad K_1 = \sqrt{\lambda} Z

where λ = 1 - exp(-gate_time/T2).

Classes
-------

NoiseChannel
------------

.. class:: NoiseChannel(name, kraus_ops, num_qubits)

   A quantum noise channel represented as Kraus operators.

   Validates completeness relation and provides methods for applying noise to quantum states.

   **Constructor**:

   .. code-block:: python

      from atlas_q.noise_models import NoiseChannel
      import torch

      # Bit flip channel: p·X + (1-p)·I
      p = 0.01
      I = torch.eye(2, dtype=torch.complex64, device='cuda')
      X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')

      kraus_ops = [
          torch.sqrt(torch.tensor(1-p, device='cuda')) * I,
          torch.sqrt(torch.tensor(p, device='cuda')) * X
      ]

      channel = NoiseChannel(name='bit_flip', kraus_ops=kraus_ops, num_qubits=1)

   **Parameters**:
      - ``name`` (str): Human-readable channel name
      - ``kraus_ops`` (list[torch.Tensor]): List of Kraus operator matrices
      - ``num_qubits`` (int): Number of qubits (1 or 2)

   **Validation**: Automatically checks :math:`\sum_i K_i^\dagger K_i = I` (warns if violated)

   **Attributes**:

   .. attribute:: name

      Channel name (str)

   .. attribute:: kraus_ops

      List of Kraus operator tensors (list[torch.Tensor])

   .. attribute:: num_qubits

      Number of qubits affected (int)

   .. attribute:: is_valid

      Whether completeness relation is satisfied (bool)

NoiseModel
----------

.. class:: NoiseModel()

   Collection of noise channels applied to quantum circuits.

   Manages multiple noise channels for different gate types and provides factory methods for common NISQ noise models.

   **Constructor**:

   .. code-block:: python

      from atlas_q.noise_models import NoiseModel

      noise = NoiseModel()

   **Attributes**:

   .. attribute:: channels_1q

      Dictionary of 1-qubit noise channels: ``{name: NoiseChannel}``

   .. attribute:: channels_2q

      Dictionary of 2-qubit noise channels: ``{name: NoiseChannel}``

   .. attribute:: custom_channels

      Dictionary mapping qubit tuples to custom channels: ``{(q1, q2): NoiseChannel}``

   **Methods**:

   .. method:: set_seed(seed)

      Set random seed for stochastic Kraus operator sampling.

      :param int seed: Random seed

      **Example**:

      .. code-block:: python

         noise.set_seed(42)  # Reproducible noise

   .. method:: add_1q_channel(name, channel)

      Add a 1-qubit noise channel to the model.

      :param str name: Channel name
      :param NoiseChannel channel: Noise channel

      **Example**:

      .. code-block:: python

         noise.add_1q_channel('custom_depol', my_channel)

   .. method:: add_2q_channel(name, channel)

      Add a 2-qubit noise channel to the model.

      :param str name: Channel name
      :param NoiseChannel channel: Noise channel

   .. staticmethod:: depolarizing(p1q=0.001, p2q=0.01, device='cuda')

      Create depolarizing noise model.

      Applies uniform noise: ρ → (1-p)ρ + p·I/d

      :param float p1q: Single-qubit depolarizing probability (default: 0.001)
      :param float p2q: Two-qubit depolarizing probability (default: 0.01)
      :param str device: Torch device ('cuda' or 'cpu')
      :return: NoiseModel with depolarizing channels
      :rtype: NoiseModel

      **Typical values**:
         - IBM Quantum: p1q ≈ 0.001, p2q ≈ 0.01
         - Google Sycamore: p1q ≈ 0.0015, p2q ≈ 0.005
         - Ion traps: p1q ≈ 0.0001, p2q ≈ 0.001

      **Example**:

      .. code-block:: python

         from atlas_q.noise_models import NoiseModel

         # IBM quantum device noise
         noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01, device='cuda')

   .. staticmethod:: dephasing(T2, gate_time, device='cuda')

      Create dephasing noise model (T2 decoherence).

      Models pure dephasing without energy relaxation.

      :param float T2: Dephasing time (μs)
      :param float gate_time: Gate duration (μs)
      :param str device: Torch device
      :return: NoiseModel with dephasing channels
      :rtype: NoiseModel

      **Formula**: λ = 1 - exp(-gate_time/T2)

      **Example**:

      .. code-block:: python

         # T2 = 80 μs, 50 ns gates
         noise = NoiseModel.dephasing(T2=80.0, gate_time=0.05, device='cuda')

   .. staticmethod:: amplitude_damping(T1, gate_time, device='cuda')

      Create amplitude damping model (T1 relaxation).

      Models energy relaxation from |1⟩ to |0⟩.

      :param float T1: Relaxation time (μs)
      :param float gate_time: Gate duration (μs)
      :param str device: Torch device
      :return: NoiseModel with amplitude damping
      :rtype: NoiseModel

      **Formula**: γ = 1 - exp(-gate_time/T1)

      **Example**:

      .. code-block:: python

         # T1 = 100 μs, 50 ns gates
         noise = NoiseModel.amplitude_damping(T1=100.0, gate_time=0.05, device='cuda')

   .. staticmethod:: thermal_relaxation(T1, T2, gate_time, device='cuda')

      Combined T1 and T2 noise model.

      Realistic NISQ device noise combining amplitude damping and dephasing.

      :param float T1: Relaxation time (μs)
      :param float T2: Dephasing time (μs, must satisfy T2 ≤ 2T1)
      :param float gate_time: Gate duration (μs)
      :param str device: Torch device
      :return: NoiseModel combining T1 and T2 effects
      :rtype: NoiseModel

      **Example**:

      .. code-block:: python

         # IBM quantum device
         noise = NoiseModel.thermal_relaxation(T1=100.0, T2=80.0, gate_time=0.05)

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+----------------------+
| Operation               | Complexity           |
+=========================+======================+
| Apply 1-qubit channel   | O(k χ²)              |
+-------------------------+----------------------+
| Apply 2-qubit channel   | O(k χ³)              |
+-------------------------+----------------------+
| Stochastic sampling     | O(k)                 |
+-------------------------+----------------------+
| Completeness check      | O(k d²)              |
+-------------------------+----------------------+

where k is number of Kraus operators, χ is bond dimension, d is local dimension.

Memory Overhead
~~~~~~~~~~~~~~~

Noise simulation increases memory usage:

.. code-block:: python

   # Noiseless MPS: O(n χ²)
   # Noisy MPS: O(n χ² + k·d²) per channel

   # Example: 50 qubits, χ=128, 4 Kraus operators
   Noiseless: 78 MB
   Noisy: 78 MB + 0.001 MB (negligible)

**Result**: Kraus operator storage is negligible compared to MPS tensors.

Benchmark Results
~~~~~~~~~~~~~~~~~

From ``scripts/benchmarks/validate_all_features.py``:

.. code-block:: python

   # 50-qubit circuit with 1000 gates
   Noiseless: 2.5 sec
   Depolarizing (p1q=0.001): 2.8 sec (12% overhead)
   Thermal (T1=100, T2=80): 3.1 sec (24% overhead)

   # Memory usage
   Noiseless: 78 MB
   Noisy: 78.2 MB (<1% increase)

**Conclusion**: Noise simulation adds 10-30% computational overhead with minimal memory impact.

Examples
--------

Depolarizing Noise
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.noise_models import NoiseModel
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create noise model (IBM quantum device parameters)
   noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01, device='cuda')

   # Create MPS
   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   # Apply gates with noise
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)

   for i in range(10):
       # Apply Hadamard gate
       mps.apply_single_qubit_gate(i, H)

       # Apply noise after gate
       mps.apply_noise_channel(noise.channels_1q['depolarizing_1q'], i)

   # Apply two-qubit gates with noise
   CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                       dtype=torch.complex64, device='cuda')

   for i in range(9):
       mps.apply_two_qubit_gate(i, CNOT)
       mps.apply_noise_channel(noise.channels_2q['depolarizing_2q'], i)

Thermal Relaxation (T1 and T2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.noise_models import NoiseModel
   from atlas_q.adaptive_mps import AdaptiveMPS

   # IBM quantum device parameters (approximate)
   T1 = 100.0       # Relaxation time (μs)
   T2 = 80.0        # Dephasing time (μs)
   gate_time = 0.05 # Gate duration (50 ns)

   noise = NoiseModel.thermal_relaxation(T1, T2, gate_time, device='cuda')

   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device='cuda')

   # Simulate noisy circuit
   # ... apply gates with noise ...

   # Compare fidelity with noiseless simulation
   fidelity = abs(mps_noisy.inner_product(mps_ideal))**2
   print(f"Fidelity: {fidelity:.4f}")

Custom Pauli Channel
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from atlas_q.noise_models import NoiseChannel, NoiseModel

   # Create custom Pauli channel: p_x X + p_y Y + p_z Z
   p_x, p_y, p_z = 0.01, 0.005, 0.005

   I = torch.eye(2, dtype=torch.complex64, device='cuda')
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
   Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device='cuda')
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')

   kraus_ops = [
       torch.sqrt(torch.tensor(1 - p_x - p_y - p_z, device='cuda')) * I,
       torch.sqrt(torch.tensor(p_x, device='cuda')) * X,
       torch.sqrt(torch.sqrt(p_y, device='cuda')) * Y,
       torch.sqrt(torch.tensor(p_z, device='cuda')) * Z
   ]

   channel = NoiseChannel(name='custom_pauli', kraus_ops=kraus_ops, num_qubits=1)

   model = NoiseModel()
   model.add_1q_channel('custom', channel)

Bit Flip and Phase Flip
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from atlas_q.noise_models import NoiseChannel, NoiseModel

   # Bit flip channel: p·X + (1-p)·I
   p_bit = 0.01
   I = torch.eye(2, dtype=torch.complex64, device='cuda')
   X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')

   bit_flip = NoiseChannel(
       name='bit_flip',
       kraus_ops=[
           torch.sqrt(torch.tensor(1-p_bit, device='cuda')) * I,
           torch.sqrt(torch.tensor(p_bit, device='cuda')) * X
       ],
       num_qubits=1
   )

   # Phase flip channel: p·Z + (1-p)·I
   p_phase = 0.005
   Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')

   phase_flip = NoiseChannel(
       name='phase_flip',
       kraus_ops=[
           torch.sqrt(torch.tensor(1-p_phase, device='cuda')) * I,
           torch.sqrt(torch.tensor(p_phase, device='cuda')) * Z
       ],
       num_qubits=1
   )

   # Combine into noise model
   noise = NoiseModel()
   noise.add_1q_channel('bit_flip', bit_flip)
   noise.add_1q_channel('phase_flip', phase_flip)

Comparing Noisy vs. Noiseless
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.noise_models import NoiseModel
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Create two MPS: noiseless and noisy
   mps_ideal = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')
   mps_noisy = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

   noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / (2**0.5)

   # Apply same gates to both
   for i in range(10):
       # Noiseless
       mps_ideal.apply_single_qubit_gate(i, H)

       # Noisy
       mps_noisy.apply_single_qubit_gate(i, H)
       mps_noisy.apply_noise_channel(noise.channels_1q['depolarizing_1q'], i)

   # Compute fidelity
   fidelity = abs(mps_noisy.inner_product(mps_ideal))**2
   print(f"Fidelity after 10 noisy gates: {fidelity:.6f}")

   # Expected: ~0.99 for p=0.001, 10 gates

Stochastic Trajectory Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.noise_models import NoiseModel
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Monte Carlo: sample Kraus operators stochastically
   noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01, device='cuda')
   noise.set_seed(42)  # Reproducible

   n_trajectories = 100
   outcomes = []

   for traj in range(n_trajectories):
       mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')

       # Apply gates with stochastic noise
       for i in range(10):
           # Gate
           mps.apply_single_qubit_gate(i, H)

           # Stochastically sample Kraus operator
           mps.apply_stochastic_noise(noise.channels_1q['depolarizing_1q'], i)

       # Measure
       outcome = mps.measure_all()
       outcomes.append(outcome)

   # Analyze distribution
   from collections import Counter
   histogram = Counter(outcomes)
   print(f"Top 10 outcomes: {histogram.most_common(10)}")

Use Cases
---------

When to Use Noise Models
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **NISQ algorithm validation**: Test VQE, QAOA on realistic hardware
2. **Error mitigation research**: Study effect of noise on quantum algorithms
3. **Hardware comparison**: Compare IBM, Google, IonQ noise characteristics
4. **Variational training**: Include noise during VQE optimization for robustness
5. **Quantum error correction**: Simulate logical qubit noise rates

NISQ Device Parameters
~~~~~~~~~~~~~~~~~~~~~~

**IBM Quantum** (superconducting qubits):

.. code-block:: python

   noise_ibm = NoiseModel.thermal_relaxation(
       T1=100.0,      # 100 μs
       T2=80.0,       # 80 μs
       gate_time=0.05 # 50 ns
   )

**Google Sycamore** (superconducting qubits):

.. code-block:: python

   noise_google = NoiseModel.thermal_relaxation(
       T1=20.0,       # 20 μs (shorter than IBM)
       T2=15.0,       # 15 μs
       gate_time=0.025 # 25 ns (faster gates)
   )

**IonQ** (trapped ions):

.. code-block:: python

   noise_ionq = NoiseModel.thermal_relaxation(
       T1=100000.0,   # ~100 ms (much longer!)
       T2=50000.0,    # ~50 ms
       gate_time=0.1  # 100 ns (slower gates)
   )

**Comparison**:

+------------------+----------+----------+----------+
| Platform         | T1 (μs)  | T2 (μs)  | p1q      |
+==================+==========+==========+==========+
| IBM Quantum      | 100      | 80       | 0.001    |
+------------------+----------+----------+----------+
| Google Sycamore  | 20       | 15       | 0.0015   |
+------------------+----------+----------+----------+
| IonQ             | 100,000  | 50,000   | 0.0001   |
+------------------+----------+----------+----------+

Cross-References
----------------

See Also
~~~~~~~~

- :doc:`adaptive_mps` - Applying noise channels to MPS
- :doc:`../user_guide/howtos/debug_simulations` - Debugging noisy simulations
- :doc:`../user_guide/explanations/algorithms` - Noise channel theory
- :doc:`vqe_qaoa` - Variational algorithms with noise

References
~~~~~~~~~~

Key papers on quantum noise:

1. **Preskill, J.** (2018). "Quantum Computing in the NISQ era and beyond." Quantum, 2, 79.
2. **Nielsen, M. A. & Chuang, I. L.** (2010). "Quantum Computation and Quantum Information." Chapter 8: Quantum Noise.
3. **Kraus, K.** (1983). "States, Effects, and Operations: Fundamental Notions of Quantum Theory." Lecture Notes in Physics.
4. **Choi, M.** (1975). "Completely positive linear maps on complex matrices." Linear Algebra and its Applications, 10(3), 285-290.
