Advanced Features Tutorial
==========================

This tutorial covers advanced simulation techniques in ATLAS-Q including circuit cutting, PEPS for 2D systems, Clifford circuit simulation, distributed computing, noise modeling, and custom optimization strategies for research and production use.

Prerequisites
-------------

Before starting this tutorial, you should:

- Have completed basic MPS tutorials (see :doc:`mps_basics`)
- Understand variational algorithms (see :doc:`vqe_tutorial`, :doc:`qaoa_tutorial`)
- Be familiar with multi-GPU or multi-node computing concepts
- Know tensor network basics and entanglement structure
- Have experience with Python profiling and performance analysis

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Use circuit cutting to simulate large circuits with limited resources
2. Apply PEPS to efficiently simulate 2D quantum systems
3. Leverage stabilizer backend for fast Clifford circuit simulation
4. Deploy distributed MPS across multiple GPUs or compute nodes
5. Model realistic quantum noise with custom channels
6. Implement constrained and multi-objective optimization
7. Profile simulations to identify performance bottlenecks
8. Troubleshoot advanced simulation issues

Part 1: Circuit Cutting and Distribution
-----------------------------------------

What is Circuit Cutting?
^^^^^^^^^^^^^^^^^^^^^^^^^

Circuit cutting decomposes large quantum circuits into smaller subcircuits that fit on available hardware or simulation resources. This trades circuit depth for classical post-processing: instead of simulating one large circuit, we simulate many small circuits and combine results.

For a circuit with :math:`n` qubits cut into :math:`k` partitions, the overhead is exponential in the number of cuts :math:`c`: typically :math:`O(4^c)` subcircuits. This is worthwhile when :math:`n` is too large but :math:`c` is small.

Why Cut Circuits?
^^^^^^^^^^^^^^^^^

Circuit cutting is valuable when:

- Full circuit exceeds memory or qubit capacity
- Circuit has sparse connectivity (natural cutting points)
- Simulating on distributed hardware without direct qubit communication
- Prototyping algorithms before full-scale hardware is available

The technique is particularly effective for modular circuits where subsystems have limited entanglement.

Implementing Circuit Cutting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig
   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Define a large circuit (30 qubits)
   def build_large_circuit():
       """Circuit too large for direct simulation."""
       gates = []

       # Layer 1: Local Hadamards
       for i in range(30):
           gates.append(('H', [i], []))

       # Layer 2: Nearest-neighbor CNOTs (creates entanglement)
       for i in range(29):
           gates.append(('CNOT', [i, i+1], []))

       # Layer 3: More local rotations
       for i in range(30):
           gates.append(('RZ', [i], [0.5]))

       # Sparse long-range gates (cutting targets)
       gates.append(('CNOT', [0, 15], []))
       gates.append(('CNOT', [14, 29], []))

       return gates

   # Configure circuit cutter
   config = CuttingConfig(
       max_partition_size=10,  # Each partition ≤ 10 qubits
       max_cuts=2,             # Allow up to 2 cuts
       optimization='min_cuts' # Minimize number of cuts
   )

   cutter = CircuitCutter(config)

   # Partition the circuit
   gates = build_large_circuit()
   partitions, cut_locations = cutter.partition_circuit(gates, n_partitions=3)

   print(f"Original circuit: 30 qubits, {len(gates)} gates")
   print(f"Partitioned into {len(partitions)} subcircuits with {len(cut_locations)} cuts")

   for i, part in enumerate(partitions):
       print(f"  Partition {i}: {part.n_qubits} qubits, {len(part.gates)} gates")

   # Simulate each partition independently
   results = []
   for i, partition in enumerate(partitions):
       mps = AdaptiveMPS(num_qubits=partition.n_qubits, bond_dim=32, device='cuda')

       for gate_name, qubits, params in partition.gates:
           if gate_name == 'H':
               H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / torch.sqrt(torch.tensor(2.0))
               mps.apply_single_qubit_gate(qubits[0], H)
           elif gate_name == 'CNOT':
               mps.apply_cnot(qubits[0], qubits[1])
           # ... other gates ...

       results.append(mps)

   # Combine results (classical post-processing)
   final_result = cutter.reconstruct(results, cut_locations)
   print(f"Final expectation value: {final_result:.8f}")

Adaptive Cutting Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Different cutting strategies for different circuit structures
   strategies = ['min_cuts', 'balanced_size', 'min_entanglement']

   for strategy in strategies:
       config = CuttingConfig(
           max_partition_size=12,
           max_cuts=3,
           optimization=strategy
       )
       cutter = CircuitCutter(config)
       partitions, cuts = cutter.partition_circuit(gates, n_partitions=3)

       overhead = 4 ** len(cuts)  # Classical overhead
       print(f"{strategy:20s}: {len(cuts)} cuts, overhead = {overhead}x")

Part 2: PEPS for 2D Systems
----------------------------

Projected Entangled Pair States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PEPS generalize MPS to 2D lattices. Instead of a 1D chain of tensors, PEPS arrange tensors on a 2D grid where each tensor connects to 4 neighbors (up, down, left, right). This captures area-law entanglement scaling natural to 2D systems.

PEPS are ideal for:

- 2D materials and spin models
- Quantum error correction codes (surface code)
- Shallow circuits on 2D architectures

Implementing 2D Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.peps import PatchPEPS

   # Simulate 4×4 qubit grid
   patch = PatchPEPS(
       patch_size=4,
       bond_dim=8,
       device='cuda'
   )

   # Apply 2D circuit
   # Coordinates are (row, col)
   gates_2d = [
       # Layer 1: Hadamards on all qubits
       *[('H', [(i, j)], []) for i in range(4) for j in range(4)],

       # Layer 2: Horizontal CNOTs
       *[('CZ', [(i, j), (i, j+1)], []) for i in range(4) for j in range(3)],

       # Layer 3: Vertical CNOTs
       *[('CZ', [(i, j), (i+1, j)], []) for i in range(3) for j in range(4)]
   ]

   # Apply gates
   for gate_name, qubits, params in gates_2d:
       if gate_name == 'H':
           patch.apply_single_site_gate(qubits[0], 'H')
       elif gate_name == 'CZ':
           patch.apply_two_site_gate(qubits[0], qubits[1], 'CZ')

   # Measure observable
   magnetization = patch.measure_magnetization()
   print(f"Total magnetization: {magnetization:.6f}")

   # Sample final state
   bitstring_2d = patch.sample()
   print(f"Sample configuration:\n{bitstring_2d}")

Surface Code Simulation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simulate distance-3 surface code (5×5 qubits)
   def surface_code_stabilizers(distance=3):
       """Generate stabilizer measurements for surface code."""
       size = 2 * distance - 1
       patch = PatchPEPS(patch_size=size, bond_dim=16, device='cuda')

       # Initialize |+⟩ state
       for i in range(size):
           for j in range(size):
               patch.apply_single_site_gate((i, j), 'H')

       # Apply stabilizer measurements
       # X-stabilizers (plaquettes)
       for i in range(0, size-1, 2):
           for j in range(1, size-1, 2):
               qubits = [(i, j), (i+1, j), (i, j-1), (i, j+1)]
               syndrome_x = patch.measure_stabilizer(qubits, pauli_type='X')

       # Z-stabilizers (vertices)
       for i in range(1, size-1, 2):
           for j in range(0, size-1, 2):
               qubits = [(i, j), (i, j+1), (i-1, j), (i+1, j)]
               syndrome_z = patch.measure_stabilizer(qubits, pauli_type='Z')

       return patch

   # Run surface code
   patch_sc = surface_code_stabilizers(distance=3)
   error_rate = patch_sc.estimate_logical_error_rate()
   print(f"Logical error rate: {error_rate:.4e}")

Part 3: Stabilizer Backend
---------------------------

Clifford Circuits and Stabilizers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The stabilizer formalism provides exponentially faster simulation for Clifford circuits (gates: H, S, CNOT, CZ). The Gottesman-Knill theorem states Clifford circuits can be simulated efficiently classically using stabilizer tableaux.

Stabilizer simulation is :math:`O(n^2)` per gate for :math:`n` qubits, compared to :math:`O(2^n)` for full statevector simulation. This enables simulation of 1000+ qubit Clifford circuits.

Fast Clifford Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # Simulate large Clifford circuit
   sim = StabilizerSimulator(n_qubits=200)

   # Apply Clifford gates
   for i in range(200):
       sim.h(i)

   for i in range(199):
       sim.cnot(i, i+1)

   for i in range(0, 200, 2):
       sim.s(i)

   # Measure in Z basis
   outcomes = []
   for i in range(200):
       outcome = sim.measure(i)
       outcomes.append(outcome)

   print(f"Measurement outcomes (first 20): {outcomes[:20]}")

   # Check stabilizers
   stabilizers = sim.get_stabilizers()
   print(f"Number of independent stabilizers: {len(stabilizers)}")

Use Cases
^^^^^^^^^

Stabilizer backend is ideal for:

1. **Quantum error correction**: Simulate encoding/decoding circuits
2. **Circuit verification**: Check if circuit is Clifford
3. **Magic state distillation**: Clifford part of fault-tolerant protocols
4. **Debugging**: Fast validation before expensive full simulation

.. code-block:: python

   # Verify circuit is Clifford before full simulation
   def is_clifford_circuit(gates):
       """Check if gate sequence contains only Clifford gates."""
       clifford_gates = {'H', 'S', 'CNOT', 'CZ', 'X', 'Y', 'Z'}
       return all(gate[0] in clifford_gates for gate in gates)

   # Example workflow
   gates = [('H', [0], []), ('CNOT', [0, 1], []), ('S', [1], [])]

   if is_clifford_circuit(gates):
       # Use fast stabilizer simulation
       sim = StabilizerSimulator(n_qubits=10)
       for gate_name, qubits, _ in gates:
           if gate_name == 'H':
               sim.h(qubits[0])
           elif gate_name == 'CNOT':
               sim.cnot(qubits[0], qubits[1])
           elif gate_name == 'S':
               sim.s(qubits[0])
       print("Used fast Clifford simulation")
   else:
       # Use full MPS simulation
       mps = AdaptiveMPS(num_qubits=10, bond_dim=32, device='cuda')
       # ... apply gates ...
       print("Used full MPS simulation")

Part 4: Distributed MPS
------------------------

Multi-GPU and Multi-Node Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed MPS partitions tensors across multiple devices to overcome single-device memory limits and accelerate computation through parallelism.

Two distribution strategies:

- **Bond-parallel**: Split along bond dimension, useful for large χ
- **Qubit-parallel**: Split along qubit index, useful for many qubits

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig

   # Bond-parallel distribution (4 GPUs)
   config_bond = DistributedConfig(
       mode='bond_parallel',
       world_size=4,           # Use 4 devices
       backend='nccl',         # NCCL for GPU communication
       compression=True        # Compress communications
   )

   mps_bond = DistributedMPS(
       num_qubits=50,
       bond_dim=256,          # 256 split across 4 GPUs = 64 per GPU
       config=config_bond
   )

   print(f"Bond dimension per device: {256 // 4}")

   # Qubit-parallel distribution
   config_qubit = DistributedConfig(
       mode='qubit_parallel',
       world_size=4
   )

   mps_qubit = DistributedMPS(
       num_qubits=200,        # 200 qubits split across 4 GPUs = 50 per GPU
       bond_dim=64,
       config=config_qubit
   )

   print(f"Qubits per device: {200 // 4}")

Communication Patterns
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import time

   # Apply gates and measure communication overhead
   mps = DistributedMPS(num_qubits=100, bond_dim=128, config=config_bond)

   # Local gates (no communication)
   start = time.time()
   for i in range(100):
       mps.apply_local_gate(i, 'H')
   t_local = time.time() - start

   # Two-qubit gates (require communication if qubits on different devices)
   start = time.time()
   for i in range(99):
       mps.apply_cnot(i, i+1)
   t_two_qubit = time.time() - start

   # Expectation values (global communication for reduction)
   start = time.time()
   energy = mps.expectation(H)
   t_expectation = time.time() - start

   print(f"Local gates: {t_local:.3f}s")
   print(f"Two-qubit gates: {t_two_qubit:.3f}s (includes communication)")
   print(f"Expectation value: {t_expectation:.3f}s (includes reduction)")

Scaling Analysis
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Benchmark scaling with number of devices
   world_sizes = [1, 2, 4, 8]
   times = []

   for ws in world_sizes:
       config = DistributedConfig(mode='bond_parallel', world_size=ws)
       mps = DistributedMPS(num_qubits=100, bond_dim=256, config=config)

       # Benchmark gate application
       start = time.time()
       for _ in range(1000):
           mps.apply_cnot(50, 51)  # Central qubits
       elapsed = time.time() - start

       times.append(elapsed)
       speedup = times[0] / elapsed if ws > 1 else 1.0
       efficiency = speedup / ws

       print(f"World size {ws}: {elapsed:.2f}s, speedup {speedup:.2f}x, efficiency {efficiency:.2%}")

   # Expected output:
   # World size 1: 5.23s, speedup 1.00x, efficiency 100%
   # World size 2: 2.87s, speedup 1.82x, efficiency 91%
   # World size 4: 1.65s, speedup 3.17x, efficiency 79%
   # World size 8: 1.12s, speedup 4.67x, efficiency 58%

Part 5: Noise Models
---------------------

Depolarizing Noise
^^^^^^^^^^^^^^^^^^

Depolarizing channels model uniform errors: with probability :math:`p`, apply random Pauli (X, Y, Z):

.. math::

   \mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)

.. code-block:: python

   from atlas_q.noise_models import NoiseModel

   # Depolarizing noise for single and two-qubit gates
   noise = NoiseModel.depolarizing(
       p1q=0.001,   # 0.1% error rate for single-qubit gates
       p2q=0.01,    # 1% error rate for two-qubit gates
       device='cuda'
   )

   # Apply noisy gates to MPS
   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')
   mps.set_noise_model(noise)

   # Gates are now automatically noisy
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / torch.sqrt(torch.tensor(2.0))
   mps.apply_single_qubit_gate(0, H)  # Applied with 0.1% depolarizing noise
   mps.apply_cnot(0, 1)               # Applied with 1% depolarizing noise

   # Measure fidelity degradation
   ideal_mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')
   ideal_mps.apply_single_qubit_gate(0, H)
   ideal_mps.apply_cnot(0, 1)

   fidelity = mps.fidelity(ideal_mps)
   print(f"State fidelity after noisy gates: {fidelity:.6f}")

Amplitude and Phase Damping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model T1 and T2 decoherence:

.. code-block:: python

   # Amplitude damping (T1 decay, energy relaxation)
   noise_t1 = NoiseModel.amplitude_damping(
       gamma=0.01,  # Decay rate
       device='cuda'
   )

   # Phase damping (T2 decay, dephasing)
   noise_t2 = NoiseModel.phase_damping(
       lambda_=0.005,  # Dephasing rate
       device='cuda'
   )

   # Combined T1 and T2
   noise_t1t2 = NoiseModel.t1_t2_noise(
       t1=50e-6,   # T1 time (seconds)
       t2=30e-6,   # T2 time (seconds)
       gate_time=50e-9,  # Gate duration (nanoseconds)
       device='cuda'
   )

   # Apply to simulation
   mps.set_noise_model(noise_t1t2)

Custom Noise Channels
^^^^^^^^^^^^^^^^^^^^^^

Define arbitrary noise channels:

.. code-block:: python

   import torch

   # Custom noise: preferential X errors
   def x_biased_noise(p_x=0.01, p_y=0.001, p_z=0.001):
       """Noise biased toward X errors."""
       def channel(rho):
           X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
           Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
           Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

           p_total = p_x + p_y + p_z
           return (1 - p_total) * rho + p_x * (X @ rho @ X) + p_y * (Y @ rho @ Y) + p_z * (Z @ rho @ Z)

       return NoiseModel.from_kraus_operators(
           [torch.sqrt(torch.tensor(1 - p_x - p_y - p_z)) * torch.eye(2),
            torch.sqrt(torch.tensor(p_x)) * X,
            torch.sqrt(torch.tensor(p_y)) * Y,
            torch.sqrt(torch.tensor(p_z)) * Z],
           device='cuda'
       )

   noise_biased = x_biased_noise(p_x=0.02, p_y=0.001, p_z=0.001)
   mps.set_noise_model(noise_biased)

Part 6: Advanced Optimization Techniques
-----------------------------------------

Constrained Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Optimize VQE subject to constraints:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   from atlas_q.mpo_ops import MPOBuilder

   # Problem Hamiltonian
   H = MPOBuilder.heisenberg_hamiltonian(n_sites=12, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

   # Constraint: total spin Sz must be zero
   Sz_total = MPOBuilder.total_spin_z(n_sites=12, device='cuda')

   # Configure constrained VQE
   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=4,
       max_iter=200,
       optimizer='SLSQP',  # Supports constraints
       constraints=[
           {'type': 'eq', 'fun': lambda params: Sz_total.expectation(params)}
       ],
       device='cuda'
   )

   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   print(f"Constrained energy: {energy:.8f}")
   print(f"Sz constraint violation: {abs(Sz_total.expectation(params)):.2e}")

Multi-Objective VQE
^^^^^^^^^^^^^^^^^^^

Optimize multiple objectives simultaneously:

.. code-block:: python

   # Objectives: minimize energy AND maximize entanglement
   H1 = MPOBuilder.heisenberg_hamiltonian(n_sites=10, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')
   H2_entanglement = MPOBuilder.entanglement_hamiltonian(n_sites=10, device='cuda')

   def multi_objective_loss(params, weights=[0.8, 0.2]):
       """Weighted combination of objectives."""
       E1 = vqe_obj1.energy(params)
       E2 = -vqe_obj2.entanglement(params)  # Maximize entanglement = minimize negative
       return weights[0] * E1 + weights[1] * E2

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       max_iter=200,
       custom_loss=multi_objective_loss,
       device='cuda'
   )

   vqe_multi = VQE(H1, config)
   energy, params = vqe_multi.optimize()

   print(f"Energy: {H1.expectation(params):.6f}")
   print(f"Entanglement: {-H2_entanglement.expectation(params):.6f}")

Adaptive Circuits
^^^^^^^^^^^^^^^^^

Dynamically adjust circuit structure during optimization:

.. code-block:: python

   # Start with shallow circuit, add layers if not converged
   for n_layers in range(1, 10):
       config = VQEConfig(
           ansatz='hardware_efficient',
           n_layers=n_layers,
           max_iter=100,
           tol=1e-6,
           device='cuda'
       )

       vqe = VQE(H, config)
       energy, params = vqe.optimize()

       # Check convergence
       gradient_norm = vqe.get_history()['gradient_norms'][-1]

       print(f"Layers {n_layers}: E = {energy:.8f}, grad_norm = {gradient_norm:.2e}")

       if gradient_norm < 1e-6:
           print(f"Converged with {n_layers} layers")
           break

Part 7: Performance Profiling
------------------------------

Timing Analysis
^^^^^^^^^^^^^^^

Identify computational bottlenecks:

.. code-block:: python

   import time
   from atlas_q.diagnostics import MPSStatistics

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')
   stats = MPSStatistics()

   # Time gate operations
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / torch.sqrt(torch.tensor(2.0))

   start = time.time()
   for i in range(30):
       mps.apply_single_qubit_gate(i, H)
   t_single = time.time() - start

   start = time.time()
   for i in range(29):
       mps.apply_cnot(i, i+1)
   t_two_qubit = time.time() - start

   start = time.time()
   for i in range(30):
       mps.canonicalize(center=i)
   t_canonicalize = time.time() - start

   print(f"Single-qubit gates: {t_single*1000:.1f} ms ({t_single/30*1000:.2f} ms per gate)")
   print(f"Two-qubit gates: {t_two_qubit*1000:.1f} ms ({t_two_qubit/29*1000:.2f} ms per gate)")
   print(f"Canonicalization: {t_canonicalize*1000:.1f} ms ({t_canonicalize/30*1000:.2f} ms per site)")

   # Use stats summary
   summary = mps.stats_summary()
   print(f"\nTotal operations: {summary['total_operations']}")
   print(f"Total time: {summary['total_time_ms']:.1f} ms")
   print(f"Average operation time: {summary['total_time_ms']/summary['total_operations']:.3f} ms")

Memory Profiling
^^^^^^^^^^^^^^^^

Track memory usage:

.. code-block:: python

   import torch.cuda

   # Monitor GPU memory
   def profile_memory(func, *args, **kwargs):
       """Profile GPU memory usage of function."""
       torch.cuda.reset_peak_memory_stats()
       torch.cuda.synchronize()

       initial_mem = torch.cuda.memory_allocated() / 1024**3  # GB

       result = func(*args, **kwargs)

       torch.cuda.synchronize()
       final_mem = torch.cuda.memory_allocated() / 1024**3
       peak_mem = torch.cuda.max_memory_allocated() / 1024**3

       print(f"Initial memory: {initial_mem:.2f} GB")
       print(f"Final memory: {final_mem:.2f} GB")
       print(f"Peak memory: {peak_mem:.2f} GB")
       print(f"Delta: {final_mem - initial_mem:.2f} GB")

       return result

   # Profile VQE optimization
   def run_vqe():
       H = MPOBuilder.heisenberg_hamiltonian(n_sites=20, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')
       config = VQEConfig(ansatz='hardware_efficient', n_layers=3, max_iter=50, device='cuda')
       vqe = VQE(H, config)
       return vqe.optimize()

   energy, params = profile_memory(run_vqe)

Bottleneck Identification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use profiling to find slow operations:

.. code-block:: python

   import cProfile
   import pstats

   # Profile VQE run
   profiler = cProfile.Profile()
   profiler.enable()

   H = MPOBuilder.heisenberg_hamiltonian(n_sites=15, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')
   config = VQEConfig(ansatz='hardware_efficient', n_layers=3, max_iter=30, device='cuda')
   vqe = VQE(H, config)
   energy, params = vqe.optimize()

   profiler.disable()

   # Print stats
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions

   # Expected bottlenecks:
   # - apply_two_site_gate (most time in entangling gates)
   # - svd (truncation overhead)
   # - expectation (Hamiltonian evaluation)

Part 8: Troubleshooting
------------------------

Problem: Out of Memory Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: CUDA out of memory, allocation failures

**Solutions**:

1. Reduce bond dimension:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=30, bond_dim=32, device='cuda')  # Instead of 128

2. Use CPU for large systems:

   .. code-block:: python

      mps = AdaptiveMPS(num_qubits=50, bond_dim=64, device='cpu')

3. Enable distributed mode:

   .. code-block:: python

      config = DistributedConfig(mode='bond_parallel', world_size=2)
      mps = DistributedMPS(num_qubits=50, bond_dim=128, config=config)

Problem: Distributed Simulation Hangs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Process stuck, no progress, timeouts

**Solutions**:

1. Check communication backend:

   .. code-block:: python

      config = DistributedConfig(
          mode='qubit_parallel',
          world_size=4,
          backend='nccl',  # Try 'gloo' if NCCL fails
          timeout=300      # Increase timeout
      )

2. Verify device placement:

   .. code-block:: python

      # Ensure each process uses different GPU
      import os
      local_rank = int(os.environ.get('LOCAL_RANK', 0))
      torch.cuda.set_device(local_rank)

3. Enable debug logging:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.DEBUG)
      # Watch for communication deadlocks

Problem: Noise Simulation is Too Slow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Noisy simulation 100× slower than ideal

**Solutions**:

1. Increase truncation tolerance for noisy states:

   .. code-block:: python

      mps = AdaptiveMPS(
          num_qubits=20,
          bond_dim=64,
          truncation_threshold=1e-4,  # Less strict for noisy states
          device='cuda'
      )

2. Use approximate noise models:

   .. code-block:: python

      # Depolarizing instead of full Kraus
      noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01, device='cuda')

3. Apply noise only to critical gates:

   .. code-block:: python

      # Apply noise selectively
      mps.apply_cnot(0, 1, noise=True)   # Noisy two-qubit gate
      mps.apply_single_qubit_gate(0, H, noise=False)  # Ideal single-qubit gate

Problem: Circuit Cutting Overhead Too High
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: 1000× overhead, impractical runtimes

**Solutions**:

1. Minimize cuts:

   .. code-block:: python

      config = CuttingConfig(
          max_cuts=2,  # Strict limit
          optimization='min_cuts'
      )

2. Restructure circuit to reduce entanglement:

   .. code-block:: python

      # Reorder gates to localize interactions
      # Use SWAP networks to bring interacting qubits together

3. Use hybrid simulation:

   .. code-block:: python

      # Cut only hard-to-simulate parts
      # Simulate easy parts with full MPS

Summary
-------

This tutorial covered advanced ATLAS-Q features:

✓ Circuit cutting to partition large circuits with controlled overhead
✓ PEPS for efficient simulation of 2D quantum systems
✓ Stabilizer backend for fast Clifford circuit simulation
✓ Distributed MPS for multi-GPU and multi-node scaling
✓ Noise models including depolarizing, amplitude damping, and custom channels
✓ Constrained and multi-objective optimization for VQE
✓ Performance profiling to identify and resolve bottlenecks
✓ Troubleshooting techniques for advanced simulation challenges

These techniques enable research-grade simulations of complex quantum systems beyond basic MPS capabilities.

Next Steps
----------

- :doc:`mps_basics` - Review MPS fundamentals
- :doc:`vqe_tutorial` - Apply advanced techniques to VQE
- :doc:`../howtos/optimize_performance` - Deep dive into performance tuning
- :doc:`../explanations/design_decisions` - Understand architectural choices

Practice Exercises
------------------

1. Cut a 50-qubit circuit into 5 partitions and measure overhead vs accuracy trade-off
2. Simulate a 6×6 surface code using PEPS and estimate logical error rate
3. Benchmark stabilizer vs full MPS simulation for random Clifford circuits up to 500 qubits
4. Implement distributed VQE across 4 GPUs and measure scaling efficiency
5. Design a custom noise model for specific hardware and validate against experimental data

See Also
--------

- :doc:`../../../reference/circuit_cutting` - Circuit cutting API
- :doc:`../../../reference/peps` - PEPS API reference
- :doc:`../../../reference/stabilizer_backend` - Stabilizer simulator API
- :doc:`../../../reference/distributed_mps` - Distributed computing API
- :doc:`../../../reference/noise_models` - Noise modeling API
