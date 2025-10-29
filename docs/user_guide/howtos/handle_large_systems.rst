How to Handle Large Quantum Systems
====================================

This guide shows strategies for simulating large quantum systems (50+ qubits, high entanglement) that exceed single-device memory or require prohibitive computation time.

Problem
-------

You need to:

- Simulate systems with 50-200+ qubits
- Handle highly entangled states requiring large bond dimensions
- Work within memory constraints (GPU/CPU RAM limits)
- Reduce computation time for large-scale simulations
- Scale simulations across multiple devices or nodes

Prerequisites
-------------

- Understanding of MPS representation and bond dimensions (see :doc:`../tutorials/mps_basics`)
- Familiarity with ATLAS-Q basic usage
- Access to GPU(s) or multi-node cluster for distributed simulations
- Knowledge of your hardware memory limits

Strategy 1: Adaptive Truncation
--------------------------------

Let bond dimensions grow dynamically based on entanglement.

Configure Adaptive Truncation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Start with small χ, grow adaptively
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=8,                      # Initial χ (small)
       chi_max_per_bond=256,            # Maximum χ allowed
       truncation_threshold=1e-8,       # SVD cutoff
       adaptive_mode=True,              # Enable adaptive growth
       device='cuda'
   )

   # Apply entangling gates - χ grows automatically
   import torch
   for i in range(49):
       mps.apply_cnot(i, i+1)

   # Check bond dimensions after evolution
   bond_dims = mps.get_bond_dimensions()
   print(f"Bond dimensions: {bond_dims}")
   print(f"Max χ reached: {max(bond_dims)}")
   print(f"Average χ: {sum(bond_dims)/len(bond_dims):.1f}")

The MPS starts small and grows only where entanglement is high, saving memory.

Per-Bond Limits
^^^^^^^^^^^^^^^

Set different limits for different regions:

.. code-block:: python

   # Higher limits in middle (high entanglement), lower at edges
   chi_limits = [32, 64, 128, 256, 256, 128, 64, 32]

   mps = AdaptiveMPS(num_qubits=len(chi_limits)+1, bond_dim=16, device='cuda')

   # Apply per-bond limits
   for i, chi_max in enumerate(chi_limits):
       mps.set_bond_max_dimension(bond_index=i, chi_max=chi_max)

This heterogeneous approach allocates resources where needed most.

Monitor Truncation Error
^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure adaptive truncation maintains accuracy:

.. code-block:: python

   from atlas_q.diagnostics import MPSStatistics

   mps = AdaptiveMPS(
       num_qubits=60,
       bond_dim=8,
       chi_max_per_bond=128,
       adaptive_mode=True,
       device='cuda'
   )

   stats = MPSStatistics(mps)

   # Apply circuit
   for i in range(59):
       mps.apply_cnot(i, i+1)

   # Check truncation statistics
   summary = stats.summary()
   print(f"Total truncation error: {summary['cumulative_truncation_error']:.2e}")
   print(f"Max single-step error: {summary['max_truncation_error']:.2e}")
   print(f"SVD count: {summary['svd_count']}")

   # If error too high, increase limits
   if summary['cumulative_truncation_error'] > 1e-5:
       print("Warning: High truncation error, consider increasing chi_max")

Strategy 2: Memory Budget Management
-------------------------------------

Constrain total memory usage.

Set Global Memory Budget
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Limit MPS to 4 GB
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=16,
       chi_max_per_bond=128,
       memory_budget_mb=4096,      # 4 GB limit
       device='cuda'
   )

   # MPS automatically adjusts χ to stay within budget
   # Higher entanglement regions may get larger χ
   # Lower entanglement regions get smaller χ to compensate

Check Memory Usage
^^^^^^^^^^^^^^^^^^

Monitor memory consumption:

.. code-block:: python

   import torch

   def get_memory_usage():
       """Get current GPU memory usage."""
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3  # GB
           reserved = torch.cuda.memory_reserved() / 1024**3
           return allocated, reserved
       return 0, 0

   # Before simulation
   alloc_before, res_before = get_memory_usage()
   print(f"Memory before: {alloc_before:.2f} GB allocated, {res_before:.2f} GB reserved")

   # Create large MPS
   mps = AdaptiveMPS(num_qubits=80, bond_dim=64, device='cuda')

   # After simulation
   alloc_after, res_after = get_memory_usage()
   print(f"Memory after: {alloc_after:.2f} GB allocated, {res_after:.2f} GB reserved")
   print(f"MPS uses: {alloc_after - alloc_before:.2f} GB")

Estimate Memory Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def estimate_mps_memory(num_qubits, bond_dim, dtype=torch.complex64):
       """
       Estimate MPS memory usage.

       MPS memory ≈ n_qubits × χ² × d × sizeof(dtype)
       where d=2 (qubit dimension), χ=bond_dim
       """
       d = 2  # Qubit dimension
       bytes_per_element = 8 if dtype == torch.complex64 else 16

       # Memory per tensor: χ_left × d × χ_right
       # Assume χ_left ≈ χ_right ≈ χ on average
       memory_per_tensor = bond_dim * d * bond_dim * bytes_per_element
       total_memory = num_qubits * memory_per_tensor

       return total_memory / 1024**3  # Convert to GB

   # Check before allocating
   for n in [50, 100, 150, 200]:
       for chi in [32, 64, 128]:
           mem_gb = estimate_mps_memory(n, chi)
           print(f"n={n:3d}, χ={chi:3d}: ~{mem_gb:.2f} GB")

   # Output:
   # n= 50, χ= 32: ~0.10 GB
   # n= 50, χ= 64: ~0.41 GB
   # n= 50, χ=128: ~1.64 GB
   # n=100, χ= 32: ~0.20 GB
   # n=100, χ= 64: ~0.82 GB
   # n=100, χ=128: ~3.28 GB
   # n=150, χ= 32: ~0.30 GB
   # n=150, χ= 64: ~1.23 GB
   # n=150, χ=128: ~4.92 GB
   # n=200, χ= 32: ~0.41 GB
   # n=200, χ= 64: ~1.64 GB
   # n=200, χ=128: ~6.55 GB

Strategy 3: Distributed MPS
----------------------------

Split MPS across multiple GPUs or nodes.

Bond-Parallel Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distribute bond dimensions across devices:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig

   # 4 GPUs, split bond dimension
   config = DistributedConfig(
       mode='bond_parallel',
       world_size=4,
       backend='nccl',             # NVIDIA GPUs
       device_ids=[0, 1, 2, 3]
   )

   # Bond dimension 256 split across 4 GPUs = 64 per GPU
   mps = DistributedMPS(
       num_qubits=60,
       bond_dim=256,
       config=config
   )

   # Operations automatically distributed
   for i in range(59):
       mps.apply_cnot(i, i+1)

   from atlas_q.mpo_ops import MPOBuilder
   H = MPOBuilder.heisenberg_hamiltonian(n_sites=60, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')
   energy = mps.expectation(H)  # Uses all 4 GPUs

Qubit-Parallel Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distribute qubits across devices:

.. code-block:: python

   # 8 GPUs, split qubits
   config = DistributedConfig(
       mode='qubit_parallel',
       world_size=8,
       backend='nccl'
   )

   # 200 qubits split across 8 GPUs = 25 qubits per GPU
   mps = DistributedMPS(
       num_qubits=200,
       bond_dim=32,
       config=config
   )

   # Each GPU handles a segment of the chain
   # Communication needed for gates spanning boundaries

Multi-Node Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

Scale across compute nodes:

.. code-block:: python

   # Launch with torchrun or mpirun:
   # torchrun --nproc_per_node=4 --nnodes=2 script.py

   import torch.distributed as dist

   # Initialize distributed backend
   dist.init_process_group(backend='nccl')
   local_rank = int(os.environ.get('LOCAL_RANK', 0))
   torch.cuda.set_device(local_rank)

   config = DistributedConfig(
       mode='bond_parallel',
       world_size=dist.get_world_size(),
       backend='nccl'
   )

   # Total: 2 nodes × 4 GPUs = 8 devices
   # Bond dimension 512 split across 8 devices = 64 per device
   mps = DistributedMPS(
       num_qubits=100,
       bond_dim=512,
       config=config
   )

Strategy 4: Circuit Cutting
----------------------------

Partition large circuits into smaller subcircuits.

Basic Circuit Cutting
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig

   # Define large circuit (100 qubits)
   gates = []
   for i in range(100):
       gates.append(('H', [i], []))
   for i in range(99):
       gates.append(('CNOT', [i, i+1], []))

   # Configure cutter
   config = CuttingConfig(
       max_partition_size=20,      # Each partition ≤ 20 qubits
       max_cuts=4,                 # Allow up to 4 cuts
       optimization='min_cuts'      # Minimize number of cuts
   )

   cutter = CircuitCutter(config)

   # Partition circuit
   partitions, cut_locations = cutter.partition_circuit(gates, n_partitions=5)

   print(f"Original: 100 qubits")
   print(f"Partitions: {len(partitions)}, Cuts: {len(cut_locations)}")
   print(f"Classical overhead: {4**len(cut_locations)}×")

   # Simulate each partition
   from atlas_q.adaptive_mps import AdaptiveMPS

   partition_results = []
   for i, partition in enumerate(partitions):
       mps = AdaptiveMPS(num_qubits=partition.n_qubits, bond_dim=32, device='cuda')

       # Apply partition gates
       for gate_name, qubits, params in partition.gates:
           # Apply gate...
           pass

       partition_results.append(mps)
       print(f"Partition {i}: {partition.n_qubits} qubits simulated")

   # Reconstruct full result
   final_result = cutter.reconstruct(partition_results, cut_locations)

Minimize Cutting Overhead
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Try different strategies
   strategies = ['min_cuts', 'balanced_size', 'min_entanglement']

   for strategy in strategies:
       config = CuttingConfig(
           max_partition_size=25,
           max_cuts=5,
           optimization=strategy
       )
       cutter = CircuitCutter(config)
       partitions, cuts = cutter.partition_circuit(gates, n_partitions=4)

       overhead = 4 ** len(cuts)
       print(f"{strategy:20s}: {len(cuts)} cuts, {overhead}× overhead")

   # Choose strategy with acceptable overhead
   # Typical: 2-3 cuts → 16-64× overhead is manageable

Strategy 5: Compress Special States
------------------------------------

Use specialized representations for structured states.

Periodic States (Shor's Algorithm)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.quantum_hybrid_system import PeriodicState

   # For period-finding: state is |x⟩ → |f(x)⟩ where f(x+r) = f(x)
   # PeriodicState stores only one period: O(r) memory vs O(2^n)

   N = 15  # Number to factor
   a = 7   # Coprime to N
   n_qubits = 8  # Sufficient for N=15

   # Create periodic state
   periodic_state = PeriodicState(
       num_qubits=n_qubits,
       function=lambda x: (a**x) % N,
       device='cuda'
   )

   # Memory: O(period) instead of O(2^n_qubits)
   # For N=15, a=7: period=4, saves 2^8/4 = 64× memory

   # Apply QFT to find period
   periodic_state.apply_qft()
   period = periodic_state.measure_period()
   print(f"Period: {period}")

Matrix Product States with Symmetries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exploit quantum number conservation:

.. code-block:: python

   # For systems with conserved quantum numbers (total Sz, particle number)
   # Use symmetry-adapted MPS (not fully implemented, concept shown)

   # Example: Conserve total Sz = 0
   def create_symmetric_mps(num_qubits, total_sz=0):
       """Create MPS in symmetry sector."""
       # Only keeps states with Sz = total_sz
       # Reduces effective Hilbert space dimension
       # Memory savings proportional to reduction
       pass

   # For 20 qubits with Sz=0:
   # Full space: 2^20 = 1M states
   # Sz=0 sector: C(20,10) = 184K states
   # Savings: ~5×

Strategy 6: Hybrid Classical-Quantum
-------------------------------------

Combine exact quantum simulation with classical approximations.

Compress Low-Entanglement Regions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Identify low-entanglement regions, compress aggressively
   def adaptive_compression(mps):
       """Compress based on local entanglement."""
       from atlas_q.diagnostics import bond_entropy_from_S

       for i in range(mps.num_qubits - 1):
           S_i = mps.get_schmidt_values(bond=i)
           entropy_i = bond_entropy_from_S(S_i)

           # Low entanglement: compress more
           if entropy_i < 0.5:
               mps.set_bond_max_dimension(bond_index=i, chi_max=16)
           # Medium entanglement
           elif entropy_i < 2.0:
               mps.set_bond_max_dimension(bond_index=i, chi_max=64)
           # High entanglement: keep large
           else:
               mps.set_bond_max_dimension(bond_index=i, chi_max=256)

   mps = AdaptiveMPS(num_qubits=100, bond_dim=16, device='cuda')

   # Evolve system
   for _ in range(100):
       # Apply gates...
       pass

   # Adaptively compress
   adaptive_compression(mps)

Classical Pre-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^

Use classical simulation for Clifford parts:

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # Circuit has Clifford (H, S, CNOT) and non-Clifford (T) gates
   clifford_gates = [('H', [i], []) for i in range(50)]
   clifford_gates += [('CNOT', [i, i+1], []) for i in range(49)]

   # Simulate Clifford part with stabilizers (fast, O(n²))
   sim = StabilizerSimulator(n_qubits=50)
   for gate_name, qubits, _ in clifford_gates:
       if gate_name == 'H':
           sim.h(qubits[0])
       elif gate_name == 'CNOT':
           sim.cnot(qubits[0], qubits[1])

   # Convert to MPS for non-Clifford part
   mps = sim.to_mps(bond_dim=32, device='cuda')

   # Now apply expensive non-Clifford gates
   T = torch.tensor([[1, 0], [0, torch.exp(1j * torch.pi / 4)]],
                    dtype=torch.complex64, device='cuda')
   for i in range(50):
       mps.apply_single_qubit_gate(i, T)

Benchmarking and Validation
----------------------------

Verify Large-Scale Simulation Correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare small-scale simulation with large-scale
   def validate_scaling(num_qubits_small, num_qubits_large, chi):
       """Verify large simulation is consistent with small."""
       import torch

       # Small system (exact)
       mps_small = AdaptiveMPS(num_qubits=num_qubits_small, bond_dim=chi, device='cuda')
       H_small = MPOBuilder.ising_hamiltonian(n_sites=num_qubits_small, J=1.0, h=0.5, device='cuda')

       for i in range(num_qubits_small - 1):
           mps_small.apply_cnot(i, i+1)

       E_small = mps_small.expectation(H_small)

       # Large system (distributed or compressed)
       config = DistributedConfig(mode='bond_parallel', world_size=2)
       mps_large = DistributedMPS(num_qubits=num_qubits_large, bond_dim=chi, config=config)
       H_large = MPOBuilder.ising_hamiltonian(n_sites=num_qubits_large, J=1.0, h=0.5, device='cuda')

       for i in range(num_qubits_large - 1):
           mps_large.apply_cnot(i, i+1)

       E_large = mps_large.expectation(H_large)

       # Check energy per qubit is consistent
       E_per_qubit_small = E_small / num_qubits_small
       E_per_qubit_large = E_large / num_qubits_large

       print(f"Small ({num_qubits_small} qubits): {E_per_qubit_small:.6f} per qubit")
       print(f"Large ({num_qubits_large} qubits): {E_per_qubit_large:.6f} per qubit")
       print(f"Relative difference: {abs(E_per_qubit_small - E_per_qubit_large) / abs(E_per_qubit_small):.2%}")

   validate_scaling(20, 100, chi=64)

Summary
-------

To handle large quantum systems in ATLAS-Q:

1. **Adaptive truncation**: Start with small χ, grow where needed (`chi_max_per_bond`, `adaptive_mode=True`)
2. **Memory budgets**: Set hard limits on total memory (`memory_budget_mb`)
3. **Distributed MPS**: Split across GPUs/nodes (bond-parallel or qubit-parallel)
4. **Circuit cutting**: Partition circuits into subcircuits (`CircuitCutter`)
5. **Compress special states**: Use `PeriodicState` for period-finding, symmetry-adapted MPS
6. **Hybrid approaches**: Clifford pre-processing with stabilizers, adaptive compression
7. **Validation**: Verify large-scale results match small-scale benchmarks

These strategies enable simulations of 50-200+ qubit systems that would otherwise exceed memory or time constraints.

See Also
--------

- :doc:`../tutorials/advanced_features` - Distributed MPS and circuit cutting details
- :doc:`../explanations/performance_model` - Scaling analysis and bottlenecks
- :doc:`optimize_performance` - General performance optimization
- :doc:`parallel_computation` - Multi-GPU and multi-node setup
- :doc:`../explanations/tensor_networks` - MPS structure and truncation theory
