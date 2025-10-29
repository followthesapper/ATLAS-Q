How to Optimize Performance
============================

This guide shows practical techniques to speed up ATLAS-Q simulations by 2-100× through GPU acceleration, precision tuning, memory optimization, and algorithm-specific improvements.

Problem
-------

You need to:

- Reduce simulation time for VQE, TDVP, or QAOA
- Handle larger systems (more qubits or higher bond dimensions)
- Run many simulations efficiently (parameter sweeps, ensemble averaging)
- Make best use of available hardware (GPUs, multi-core CPUs)

Prerequisites
-------------

- ATLAS-Q installed (see :doc:`../../installation`)
- CUDA-capable GPU for GPU acceleration (recommended)
- Python 3.9+
- Ability to install optional dependencies

Enable GPU Acceleration
-----------------------

ATLAS-Q provides multiple GPU acceleration backends. Use all available for maximum speedup.

Enable Triton Kernels
^^^^^^^^^^^^^^^^^^^^^

Triton provides custom GPU kernels for MPS operations with 1.5-3× speedup.

**Installation**:

.. code-block:: bash

   pip install triton

Or via setup script:

.. code-block:: bash

   cd /path/to/ATLAS-Q
   ./setup_triton.sh

**Verification**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Triton automatically used if available
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / torch.sqrt(torch.tensor(2.0))

   import time
   start = time.time()
   for i in range(1000):
       mps.apply_single_qubit_gate(0, H)
   elapsed = time.time() - start

   print(f"1000 operations: {elapsed:.2f}s")
   # With Triton: ~0.5s
   # Without: ~1.2s

**Expected speedup**: 1.5-3× for bond dimensions χ > 32.

Enable cuQuantum Backend
^^^^^^^^^^^^^^^^^^^^^^^^

NVIDIA cuQuantum provides highly optimized tensor contractions with 2-10× speedup.

**Installation**:

.. code-block:: bash

   pip install cuquantum-python

**Usage**:

.. code-block:: python

   from atlas_q.cuquantum_backend import CuQuantumBackend, CuQuantumConfig

   # Configure cuQuantum
   config = CuQuantumConfig(
       use_tensor_cores=True,  # Use Tensor Cores on Ampere+ GPUs
       precision='single',     # complex64 for speed
       workspace_size_gb=4     # Scratch memory for contractions
   )

   backend = CuQuantumBackend(config)

   # Use in MPS
   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=128,
       device='cuda',
       backend=backend
   )

   # Operations automatically use cuQuantum
   mps.apply_cnot(20, 21)

**Expected speedup**: 2-5× for large contractions, up to 10× on A100/H100 with Tensor Cores.

Verify GPU Usage
^^^^^^^^^^^^^^^^

Ensure ATLAS-Q is actually using your GPU:

.. code-block:: python

   import torch

   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA device: {torch.cuda.get_device_name(0)}")
   print(f"CUDA version: {torch.version.cuda}")

   # Monitor GPU utilization
   # Run in separate terminal:
   # watch -n 0.5 nvidia-smi

If GPU not used, check device parameter:

.. code-block:: python

   # Correct
   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device='cuda')

   # Incorrect - runs on CPU
   mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device='cpu')

Optimize Bond Dimensions
-------------------------

Bond dimension χ controls accuracy vs performance trade-off.

Choose Appropriate Bond Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start with small χ and increase until convergence:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder

   H = MPOBuilder.heisenberg_hamiltonian(n_sites=20, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

   bond_dims = [16, 32, 64, 128]
   energies = []

   for chi in bond_dims:
       mps = AdaptiveMPS(num_qubits=20, bond_dim=chi, device='cuda')

       # Prepare ground state
       for _ in range(100):
           mps.apply_gate_sweep(H)

       energy = mps.expectation(H)
       energies.append(energy)

       print(f"χ={chi:3d}: E = {energy:.8f}")

   # Check convergence
   import numpy as np
   diffs = np.diff(energies)
   print(f"Energy differences: {diffs}")

   # If last difference < tolerance, χ is sufficient
   if abs(diffs[-1]) < 1e-6:
       optimal_chi = bond_dims[-2]
       print(f"Optimal χ: {optimal_chi}")

Adaptive Truncation
^^^^^^^^^^^^^^^^^^^

Let ATLAS-Q automatically adjust χ based on truncation error:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=16,                    # Initial χ
       chi_max_per_bond=128,           # Maximum χ
       truncation_threshold=1e-10,     # SVD truncation tolerance
       adaptive_mode=True,             # Enable adaptive χ
       device='cuda'
   )

   # Apply gates - χ grows automatically
   for i in range(29):
       mps.apply_cnot(i, i+1)

   # Check final bond dimensions
   bond_dims_actual = mps.get_bond_dimensions()
   print(f"Bond dimensions: {bond_dims_actual}")
   print(f"Max χ used: {max(bond_dims_actual)}")

This automatically balances accuracy and performance.

Per-Bond Dimension Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For heterogeneous entanglement, set different χ per bond:

.. code-block:: python

   # Higher χ in middle (high entanglement), lower at edges
   chi_per_bond = [16, 32, 64, 128, 128, 64, 32, 16]

   mps = AdaptiveMPS(
       num_qubits=9,
       bond_dim=64,  # Default
       device='cuda'
   )

   # Manually set bond dimensions
   for i, chi in enumerate(chi_per_bond):
       mps.set_bond_dimension(i, chi)

Use Mixed Precision
-------------------

Reduce memory and increase speed with complex64 instead of complex128.

Configure Precision Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import DTypePolicy

   # Use single precision (complex64)
   policy = DTypePolicy(
       default=torch.complex64,        # Main datatype
       high_precision=torch.complex128, # For numerically sensitive ops
       threshold=1e-8                   # Switch to high precision if condition number > threshold
   )

   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=64,
       dtype_policy=policy,
       device='cuda'
   )

**Expected speedup**: 2× for most operations, 50% memory reduction.

Precision Validation
^^^^^^^^^^^^^^^^^^^^

Verify accuracy loss is acceptable:

.. code-block:: python

   # High precision (baseline)
   mps_fp64 = AdaptiveMPS(num_qubits=15, bond_dim=32, device='cuda')
   for i in range(14):
       mps_fp64.apply_cnot(i, i+1)
   energy_fp64 = mps_fp64.expectation(H)

   # Single precision (fast)
   policy = DTypePolicy(default=torch.complex64)
   mps_fp32 = AdaptiveMPS(num_qubits=15, bond_dim=32, dtype_policy=policy, device='cuda')
   for i in range(14):
       mps_fp32.apply_cnot(i, i+1)
   energy_fp32 = mps_fp32.expectation(H)

   # Compare
   error = abs(energy_fp64 - energy_fp32)
   relative_error = error / abs(energy_fp64)

   print(f"FP64 energy: {energy_fp64:.12f}")
   print(f"FP32 energy: {energy_fp32:.12f}")
   print(f"Relative error: {relative_error:.2e}")

   # Acceptable if relative error < 1e-5 for most applications

Batch and Parallelize Operations
---------------------------------

Process Multiple Circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use Python multiprocessing for CPU-bound or multi-GPU setups:

.. code-block:: python

   import torch.multiprocessing as mp

   def simulate_circuit(circuit_id, device_id):
       """Simulate one circuit on specific GPU."""
       device = f'cuda:{device_id}'
       mps = AdaptiveMPS(num_qubits=20, bond_dim=32, device=device)

       # Apply circuit
       for gate in circuits[circuit_id]:
           # Apply gate...
           pass

       result = mps.sample(shots=1000)
       return circuit_id, result

   if __name__ == '__main__':
       n_circuits = 100
       n_gpus = 4

       # Distribute circuits across GPUs
       pool = mp.Pool(n_gpus)
       tasks = [(i, i % n_gpus) for i in range(n_circuits)]
       results = pool.starmap(simulate_circuit, tasks)

       print(f"Simulated {n_circuits} circuits on {n_gpus} GPUs")

Vectorize Parameter Sweeps
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For VQE parameter sweeps, evaluate multiple points in parallel:

.. code-block:: python

   import numpy as np
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # Grid search over parameters
   gammas = np.linspace(0, 2*np.pi, 10)
   betas = np.linspace(0, np.pi, 10)

   H = MPOBuilder.heisenberg_hamiltonian(n_sites=10, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

   # Parallel evaluation
   energies = np.zeros((len(gammas), len(betas)))

   import concurrent.futures
   def evaluate_point(gamma, beta):
       config = VQEConfig(ansatz='qaoa', p=1, device='cuda')
       vqe = VQE(H, config, initial_params=[gamma, beta])
       energy, _ = vqe.evaluate()  # Don't optimize, just evaluate
       return energy

   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       futures = {}
       for i, gamma in enumerate(gammas):
           for j, beta in enumerate(betas):
               future = executor.submit(evaluate_point, gamma, beta)
               futures[future] = (i, j)

       for future in concurrent.futures.as_completed(futures):
           i, j = futures[future]
           energies[i, j] = future.result()

   print(f"Evaluated {len(gammas) * len(betas)} points in parallel")

Profile and Identify Bottlenecks
---------------------------------

Use Profiling Tools
^^^^^^^^^^^^^^^^^^^

Find slow operations:

.. code-block:: python

   import cProfile
   import pstats
   from pstats import SortKey

   def run_simulation():
       mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')
       H = MPOBuilder.heisenberg_hamiltonian(n_sites=20, Jx=1.0, Jy=1.0, Jz=1.0, device='cuda')

       for _ in range(50):
           mps.apply_gate_sweep(H)

       return mps.expectation(H)

   # Profile
   profiler = cProfile.Profile()
   profiler.enable()
   energy = run_simulation()
   profiler.disable()

   # Analyze
   stats = pstats.Stats(profiler)
   stats.sort_stats(SortKey.CUMULATIVE)
   stats.print_stats(15)  # Top 15 functions

Common bottlenecks:

- ``apply_two_site_gate``: Entangling operations (expected)
- ``svd``: Truncation overhead (reduce by relaxing threshold)
- ``expectation``: Hamiltonian evaluation (cache if possible)

MPS Diagnostics
^^^^^^^^^^^^^^^

Use built-in diagnostics to understand performance:

.. code-block:: python

   from atlas_q.diagnostics import MPSStatistics

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')
   stats = MPSStatistics(mps)

   # Apply operations
   for i in range(19):
       mps.apply_cnot(i, i+1)

   # Get stats
   summary = stats.summary()

   print(f"Total operations: {summary['total_operations']}")
   print(f"SVD calls: {summary['svd_count']}")
   print(f"Average SVD time: {summary['avg_svd_time_ms']:.2f} ms")
   print(f"Truncation error: {summary['cumulative_truncation_error']:.2e}")
   print(f"Bond dimension distribution: {summary['chi_histogram']}")

Optimize Memory Usage
---------------------

Reduce Memory Footprint
^^^^^^^^^^^^^^^^^^^^^^^^

For large systems, minimize memory:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=32,                # Lower χ
       device='cuda',
       dtype_policy=DTypePolicy(default=torch.complex64),  # Single precision
       checkpointing=True          # Trade compute for memory
   )

   # Clear cache periodically
   import torch
   for i in range(100):
       mps.apply_gate_sweep(H)

       if i % 10 == 0:
           torch.cuda.empty_cache()

Enable Gradient Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For VQE with long circuits:

.. code-block:: python

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=10,
       max_iter=100,
       gradient_checkpointing=True,  # Recompute forward pass during backward
       device='cuda'
   )

   vqe = VQE(H, config)
   energy, params = vqe.optimize()

This reduces memory from O(L) to O(√L) for L layers, at 30% compute cost.

Algorithm-Specific Optimizations
---------------------------------

VQE Optimization
^^^^^^^^^^^^^^^^

.. code-block:: python

   config = VQEConfig(
       ansatz='hardware_efficient',
       n_layers=3,
       optimizer='L-BFGS-B',        # Fast convergence
       max_iter=200,
       tol=1e-6,
       gtol=1e-5,
       bond_dim=48,                 # Moderate χ
       use_jit=True,                # JIT compile ansatz
       cache_hamiltonian=True,      # Cache H for repeated evaluations
       device='cuda'
   )

TDVP Optimization
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.tdvp import TDVP2Site, TDVPConfig

   config = TDVPConfig(
       dt=0.05,
       t_final=10.0,
       krylov_dim=8,                # Reduce from default 20 for speed
       normalize=True,
       adaptive_dt=True,            # Adjust timestep automatically
       error_tol=1e-5,
       device='cuda'
   )

   tdvp = TDVP2Site(H, mps, config)
   tdvp.evolve()

QAOA Optimization
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.vqe_qaoa import QAOA

   qaoa = QAOA(
       H,
       p=3,                         # Moderate depth
       config=VQEConfig(
           optimizer='COBYLA',      # Robust for QAOA
           max_iter=150,
           bond_dim=32,             # QAOA typically low entanglement
           warm_start=True,         # Use classical initialization
           device='cuda'
       )
   )

Use Stabilizer Backend for Clifford
------------------------------------

For Clifford-only circuits (H, S, CNOT, CZ), use exponentially faster stabilizer simulation:

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # Simulate 200-qubit Clifford circuit
   sim = StabilizerSimulator(n_qubits=200)

   # Apply gates (O(n²) per gate vs O(2^n))
   for i in range(200):
       sim.h(i)

   for i in range(199):
       sim.cnot(i, i+1)

   for i in range(0, 200, 2):
       sim.s(i)

   # Measure
   outcomes = [sim.measure(i) for i in range(200)]

   print(f"Simulated 200-qubit Clifford circuit in seconds (vs hours for MPS)")

**Expected speedup**: 20-1000× for Clifford circuits.

Hardware Considerations
-----------------------

Choose Optimal Hardware
^^^^^^^^^^^^^^^^^^^^^^^

Performance by GPU generation:

- **Ampere (A100, RTX 30xx)**: 3-5× faster than Turing, Tensor Core support
- **Hopper (H100)**: 2× faster than A100, best for large χ
- **Ada Lovelace (RTX 40xx)**: Consumer option, 2-3× faster than RTX 30xx

CPU alternatives:

- **AMD EPYC**: 64-128 cores, good for qubit-parallel workloads
- **Intel Xeon**: 32-56 cores, competitive for small χ

Configure CUDA Settings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os

   # Maximize GPU utilization
   os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'  # Serialize kernel launches

   # Enable Tensor Cores
   os.environ['NVIDIA_TF32_OVERRIDE'] = '1'  # Use TF32 on Ampere+

   # Set memory allocation strategy
   os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

Verification
------------

Benchmark your optimizations:

.. code-block:: python

   import time

   # Baseline
   mps_baseline = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')
   start = time.time()
   for i in range(1000):
       mps_baseline.apply_cnot(15, 16)
   time_baseline = time.time() - start

   # Optimized (Triton + cuQuantum + FP32)
   policy = DTypePolicy(default=torch.complex64)
   backend = CuQuantumBackend(CuQuantumConfig(use_tensor_cores=True))
   mps_optimized = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       dtype_policy=policy,
       backend=backend,
       device='cuda'
   )

   start = time.time()
   for i in range(1000):
       mps_optimized.apply_cnot(15, 16)
   time_optimized = time.time() - start

   speedup = time_baseline / time_optimized
   print(f"Baseline: {time_baseline:.2f}s")
   print(f"Optimized: {time_optimized:.2f}s")
   print(f"Speedup: {speedup:.2f}×")

Expected combined speedup: 5-15× for typical workloads.

Summary
-------

To optimize ATLAS-Q performance:

1. **Enable GPU acceleration** (Triton + cuQuantum)
2. **Tune bond dimensions** (start small, increase to convergence)
3. **Use mixed precision** (complex64 for 2× speedup)
4. **Batch operations** (multi-GPU, parallel parameter sweeps)
5. **Profile bottlenecks** (cProfile, MPSStatistics)
6. **Reduce memory** (checkpointing, lower χ, FP32)
7. **Algorithm-specific tuning** (VQE: L-BFGS-B, TDVP: small Krylov)
8. **Clifford fast-path** (stabilizer backend for 20-1000× speedup)
9. **Hardware selection** (Ampere+ GPUs with Tensor Cores)

These techniques routinely achieve 10-100× total speedup.

See Also
--------

- :doc:`../explanations/gpu_acceleration` - GPU acceleration details
- :doc:`../explanations/performance_model` - Performance scaling theory
- :doc:`../tutorials/advanced_features` - Advanced techniques
- :doc:`integrate_cuquantum` - cuQuantum setup details
- :doc:`configure_precision` - Precision configuration guide
