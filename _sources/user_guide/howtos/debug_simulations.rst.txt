========================================
Debug Simulations
========================================

Problem
=======

Debugging quantum simulations is challenging due to:

- **High dimensionality**: States in 2^n Hilbert space are hard to visualize
- **Numerical errors**: Truncation, rounding, and ill-conditioning accumulate
- **Silent failures**: Incorrect results may look plausible
- **Performance issues**: Slow simulations may indicate inefficient algorithms or bugs
- **Complex workflows**: VQE, TDVP, QAOA involve optimization, time evolution, and expectation values

This guide covers debugging strategies for MPS simulations, including error tracking,
numerical stability analysis, gate verification, and performance profiling.

.. seealso::
   :doc:`../explanations/numerical_stability` for understanding numerical issues,
   :doc:`save_load_state` for checkpoint-based debugging,
   :doc:`optimize_performance` for performance debugging,
   :doc:`../faq` for common issues and solutions.

Prerequisites
=============

You need:

- ATLAS-Q installed with development dependencies
- Basic understanding of MPS structure and operations
- Familiarity with your simulation problem (expected results, tolerances)
- Patience for systematic debugging

Strategies
==========

Strategy 1: Installation and Environment Verification
------------------------------------------------------

Verify ATLAS-Q installation and dependencies before debugging simulation logic.

**Check ATLAS-Q installation**:

.. code-block:: python

   import atlas_q
   import torch
   import numpy as np

   # Check versions
   print(f"ATLAS-Q version: {atlas_q.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"NumPy version: {np.__version__}")

   # Check CUDA availability
   print(f"\nCUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU: {torch.cuda.get_device_name(0)}")
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

   # Check cuQuantum
   from atlas_q import get_cuquantum
   cuq = get_cuquantum()
   if cuq['is_cuquantum_available']():
       print(f"cuQuantum version: {cuq['get_cuquantum_version']()}")
   else:
       print("cuQuantum not available (optional)")

**Run diagnostic tests**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   # Test basic MPS creation
   try:
       mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')
       print("MPS creation: OK")
   except Exception as e:
       print(f"MPS creation failed: {e}")
       raise

   # Test basic gate application
   try:
       mps.apply_hadamard(0)
       mps.apply_cnot(0, 1)
       print("Gate application: OK")
   except Exception as e:
       print(f"Gate application failed: {e}")
       raise

   # Test expectation value
   try:
       pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')
       exp_val = mps.expectation_value_single_site(0, pauli_z)
       print(f"Expectation value: {exp_val:.6f} (OK)")
   except Exception as e:
       print(f"Expectation value failed: {e}")
       raise

   print("\nAll diagnostic tests passed!")

Strategy 2: Enable Detailed Logging
------------------------------------

Use Python logging to track MPS operations and identify issues.

**Configure logging levels**:

.. code-block:: python

   import logging

   # Configure logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.StreamHandler(),           # Console output
           logging.FileHandler('debug.log')   # File output
       ]
   )

   # Set specific module log levels
   logging.getLogger('atlas_q.adaptive_mps').setLevel(logging.DEBUG)
   logging.getLogger('atlas_q.vqe_qaoa').setLevel(logging.INFO)
   logging.getLogger('atlas_q.tdvp').setLevel(logging.DEBUG)

   # Now run simulation - all operations logged
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')
   mps.apply_cnot(0, 1)  # Logs: "Applying CNOT to qubits 0-1, χ_before=8, χ_after=16"

**Verbose MPS statistics**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Enable verbose statistics tracking
   mps = AdaptiveMPS(
       num_qubits=20,
       bond_dim=16,
       device='cuda',
       verbose=True,  # Enable verbose output
       track_statistics=True  # Track per-operation statistics
   )

   # Apply gates - each operation prints diagnostics
   for i in range(10):
       mps.apply_cnot(i, i+1)
       # Prints: [Gate 0] CNOT(0,1): χ 16→32, ε=1.2e-08, time=0.012s

   # Print summary
   print(mps.statistics.summary())

Strategy 3: Track Error Propagation
------------------------------------

Monitor truncation error accumulation to detect numerical issues early.

**Global error tracking**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import matplotlib.pyplot as plt

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       truncation_threshold=1e-8,
       device='cuda'
   )

   # Track error accumulation
   errors = []
   for i in range(29):
       mps.apply_cnot(i, i+1)

       # Get cumulative error
       global_error = mps.statistics.total_truncation_error
       errors.append(global_error)

       # Warn if error exceeds threshold
       if global_error > 1e-4:
           print(f"Warning: Global error {global_error:.2e} at gate {i}")
           print(f"  Consider: (1) increasing bond dim, (2) tightening threshold")

   # Plot error growth
   plt.plot(errors)
   plt.xlabel('Gate number')
   plt.ylabel('Cumulative truncation error')
   plt.yscale('log')
   plt.title('Error accumulation over circuit')
   plt.savefig('error_tracking.png')

**Per-bond error analysis**:

.. code-block:: python

   # Analyze which bonds accumulate most error
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply gates
   for i in range(29):
       mps.apply_cnot(i, i+1)

   # Check error by bond
   bond_errors = mps.statistics.truncation_error_per_bond

   print("Truncation error by bond:")
   for bond, error in enumerate(bond_errors):
       if error > 1e-6:
           print(f"  Bond {bond}: {error:.2e} (high error!)")

   # Identify bottleneck bonds
   max_error_bond = bond_errors.argmax()
   print(f"\nBottleneck: Bond {max_error_bond} with error {bond_errors[max_error_bond]:.2e}")
   print(f"  → Increase χ locally or change gate order")

Strategy 4: Verify Gate Correctness
------------------------------------

Test gates for unitarity and correctness before using in simulations.

**Verify gate unitarity**:

.. code-block:: python

   import torch

   def verify_unitary(U, gate_name="Gate", tolerance=1e-10):
       """
       Verify that U is unitary: U @ U† = I.

       Parameters
       ----------
       U : torch.Tensor
           Gate matrix (d×d)
       gate_name : str
           Name for error messages
       tolerance : float
           Numerical tolerance

       Raises
       ------
       AssertionError
           If gate is not unitary within tolerance
       """
       I_actual = U @ U.conj().T
       I_expected = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)

       error = torch.norm(I_actual - I_expected).item()

       if error > tolerance:
           print(f"{gate_name} unitarity check FAILED:")
           print(f"  U @ U† - I norm: {error:.2e}")
           print(f"  Tolerance: {tolerance:.2e}")
           raise AssertionError(f"{gate_name} is not unitary")
       else:
           print(f"{gate_name} unitarity check OK (error: {error:.2e})")

   # Test standard gates
   import math

   # Hadamard
   H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
   verify_unitary(H, "Hadamard")

   # CNOT
   CNOT = torch.tensor([
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]
   ], dtype=torch.complex64)
   verify_unitary(CNOT, "CNOT")

   # Parameterized rotation
   theta = 0.5
   Ry = torch.tensor([
       [math.cos(theta/2), -math.sin(theta/2)],
       [math.sin(theta/2), math.cos(theta/2)]
   ], dtype=torch.complex64)
   verify_unitary(Ry, f"Ry({theta})")

**Test gate application correctness**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import math

   # Test: H|0⟩ = (|0⟩ + |1⟩)/√2
   mps = AdaptiveMPS(num_qubits=1, bond_dim=2, device='cuda')
   mps.apply_hadamard(0)

   # Measure in computational basis
   prob_0 = abs(mps.amplitude([0]))**2
   prob_1 = abs(mps.amplitude([1]))**2

   print(f"After H|0⟩:")
   print(f"  P(|0⟩) = {prob_0:.6f} (expected: 0.5)")
   print(f"  P(|1⟩) = {prob_1:.6f} (expected: 0.5)")

   assert abs(prob_0 - 0.5) < 1e-6, f"P(|0⟩) incorrect: {prob_0}"
   assert abs(prob_1 - 0.5) < 1e-6, f"P(|1⟩) incorrect: {prob_1}"
   print("Hadamard gate test PASSED")

   # Test: CNOT|01⟩ = |01⟩, CNOT|11⟩ = |10⟩
   mps2 = AdaptiveMPS(num_qubits=2, bond_dim=2, device='cuda')
   mps2.apply_pauli_x(1)  # |01⟩
   mps2.apply_cnot(0, 1)

   prob_01 = abs(mps2.amplitude([0, 1]))**2
   print(f"\nCNOT|01⟩ → P(|01⟩) = {prob_01:.6f} (expected: 1.0)")
   assert abs(prob_01 - 1.0) < 1e-6
   print("CNOT gate test PASSED")

Strategy 5: Monitor Numerical Stability
----------------------------------------

Track condition numbers and detect ill-conditioned tensors.

**Condition number monitoring**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   def check_tensor_conditioning(mps, threshold=1e10):
       """
       Check condition numbers of all MPS tensors.

       Parameters
       ----------
       mps : AdaptiveMPS
           MPS to check
       threshold : float
           Condition number threshold for warnings

       Returns
       -------
       dict
           Condition number statistics
       """
       cond_numbers = []

       for i, tensor in enumerate(mps.tensors):
           # Reshape to matrix for SVD
           shape = tensor.shape
           matrix = tensor.reshape(shape[0] * shape[1], shape[2])

           # Compute singular values
           s = torch.linalg.svdvals(matrix)

           # Condition number = σ_max / σ_min
           cond = (s[0] / s[-1]).item()
           cond_numbers.append(cond)

           if cond > threshold:
               print(f"WARNING: Tensor {i} ill-conditioned!")
               print(f"  Condition number: {cond:.2e}")
               print(f"  Max singular value: {s[0].item():.2e}")
               print(f"  Min singular value: {s[-1].item():.2e}")
               print(f"  → Consider using complex128 or reducing bond dimension")

       return {
           'max_cond': max(cond_numbers),
           'mean_cond': sum(cond_numbers) / len(cond_numbers),
           'cond_numbers': cond_numbers
       }

   # Usage: Check conditioning during TDVP
   from atlas_q.tdvp import TDVP

   mps = AdaptiveMPS(num_qubits=30, bond_dim=128, device='cuda')
   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

   for step in range(1000):
       E = tdvp.evolve_step()

       if step % 100 == 0:
           stats = check_tensor_conditioning(mps, threshold=1e8)
           print(f"[Step {step}] E={E:.8f}, max cond={stats['max_cond']:.2e}")

Strategy 6: Checkpoint-Based Debugging
---------------------------------------

Save checkpoints and analyze intermediate states to isolate bugs.

**Save checkpoints for analysis**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch
   import os

   checkpoint_dir = 'debug_checkpoints'
   os.makedirs(checkpoint_dir, exist_ok=True)

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

   # Apply gates and checkpoint periodically
   for i in range(100):
       mps.apply_cnot(i % 19, (i % 19) + 1)

       # Save checkpoint every 10 gates
       if i % 10 == 0:
           checkpoint = {
               'step': i,
               'tensors': [t.cpu() for t in mps.tensors],
               'bond_dims': mps.bond_dims,
               'statistics': mps.statistics.__dict__
           }
           torch.save(checkpoint, os.path.join(checkpoint_dir, f'step_{i:04d}.pt'))

   print(f"Saved {len(os.listdir(checkpoint_dir))} checkpoints")

**Load and analyze checkpoint**:

.. code-block:: python

   # Load checkpoint at specific step
   step_to_analyze = 50
   checkpoint = torch.load(f'debug_checkpoints/step_{step_to_analyze:04d}.pt')

   # Reconstruct MPS
   mps_debug = AdaptiveMPS(
       num_qubits=len(checkpoint['tensors']),
       bond_dim=max(checkpoint['bond_dims']),
       device='cpu'  # CPU for analysis
   )
   mps_debug.tensors = checkpoint['tensors']
   mps_debug.bond_dims = checkpoint['bond_dims']

   # Analyze state
   print(f"Checkpoint at step {step_to_analyze}:")
   print(f"  Bond dimensions: {mps_debug.bond_dims}")
   print(f"  Max bond dim: {max(mps_debug.bond_dims)}")

   # Check specific amplitudes
   amp_000 = mps_debug.amplitude([0, 0, 0])
   print(f"  Amplitude |000⟩: {amp_000}")

   # Compute observables
   # ... custom analysis ...

Strategy 7: Performance Profiling
----------------------------------

Profile simulations to identify performance bottlenecks.

**PyTorch profiler**:

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=128, device='cuda')

   # Profile gate application
   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA
       ],
       record_shapes=True,
       with_stack=True
   ) as prof:
       for i in range(10):
           mps.apply_cnot(i, i+1)

   # Print summary
   print(prof.key_averages().table(
       sort_by="cuda_time_total",
       row_limit=20
   ))

   # Export to Chrome trace format
   prof.export_chrome_trace("trace.json")
   print("Profiling trace saved to trace.json (view in chrome://tracing)")

**Time individual operations**:

.. code-block:: python

   import time
   import torch

   mps = AdaptiveMPS(num_qubits=50, bond_dim=256, device='cuda')

   # Time CNOT application
   torch.cuda.synchronize()
   start = time.time()

   mps.apply_cnot(0, 1)

   torch.cuda.synchronize()
   elapsed = time.time() - start

   print(f"CNOT time: {elapsed*1000:.2f} ms")

   # Time VQE iteration
   from atlas_q.vqe_qaoa import VQE, VQEConfig

   config = VQEConfig(max_iterations=10)
   vqe = VQE(hamiltonian=H, config=config, device='cuda')

   start = time.time()
   energy, params = vqe.optimize()
   elapsed = time.time() - start

   print(f"VQE 10 iterations: {elapsed:.2f}s ({elapsed/10:.2f}s per iteration)")

Troubleshooting
===============

MPS State Looks Wrong
---------------------

**Problem**: Amplitudes or expectation values don't match expected results.

**Solution**: Verify gate order and MPS normalization.

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # Check normalization
   mps = AdaptiveMPS(num_qubits=10, bond_dim=16, device='cuda')
   # ... apply gates ...

   norm = mps.norm()
   print(f"MPS norm: {norm:.10f}")

   if abs(norm - 1.0) > 1e-6:
       print(f"WARNING: MPS not normalized! Norm = {norm}")
       # Renormalize
       mps.normalize()
       print(f"After normalization: {mps.norm():.10f}")

   # Verify specific amplitudes
   amp = mps.amplitude([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   print(f"Amplitude |0...0⟩: {amp}")

   # Check sum of probabilities
   # (Only feasible for small systems)
   if mps.num_qubits <= 10:
       total_prob = 0.0
       for config in itertools.product([0, 1], repeat=mps.num_qubits):
           prob = abs(mps.amplitude(list(config)))**2
           total_prob += prob
       print(f"Total probability: {total_prob:.10f} (should be 1.0)")

Energy Not Decreasing in VQE
-----------------------------

**Problem**: VQE energy plateaus or increases.

**Solution**: Check gradients, reduce learning rate, or inspect Hamiltonian.

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import matplotlib.pyplot as plt

   config = VQEConfig(
       max_iterations=100,
       optimizer='adam',
       learning_rate=0.01
   )

   vqe = VQE(hamiltonian=H, config=config, device='cuda')
   energy, params = vqe.optimize()

   # Plot energy convergence
   plt.plot(vqe.energies)
   plt.xlabel('Iteration')
   plt.ylabel('Energy')
   plt.title('VQE Convergence')
   plt.savefig('vqe_convergence.png')

   # Check if stuck in plateau
   if len(vqe.energies) > 50:
       recent_std = np.std(vqe.energies[-50:])
       if recent_std < 1e-8:
           print("Energy plateaued! Try:")
           print("  - Reduce learning rate (current: 0.01)")
           print("  - Switch optimizer (try 'lbfgs')")
           print("  - Check Hamiltonian spectrum")

Out of Memory During Simulation
--------------------------------

**Problem**: ``RuntimeError: CUDA out of memory``.

**Solution**: Reduce bond dimension, batch size, or use gradient checkpointing.

.. code-block:: python

   import torch

   # Check GPU memory usage
   allocated = torch.cuda.memory_allocated() / 1024**3
   reserved = torch.cuda.memory_reserved() / 1024**3
   total = torch.cuda.get_device_properties(0).total_memory / 1024**3

   print(f"GPU memory:")
   print(f"  Allocated: {allocated:.2f} GB")
   print(f"  Reserved: {reserved:.2f} GB")
   print(f"  Total: {total:.2f} GB")

   if allocated > 0.8 * total:
       print("WARNING: GPU memory usage > 80%")
       print("Solutions:")
       print("  1. Reduce bond dimension")
       print("  2. Use smaller batch size")
       print("  3. Clear cache: torch.cuda.empty_cache()")

   # Reduce bond dimension
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,  # Was 256
       device='cuda'
   )

Summary
=======

Debugging strategies for ATLAS-Q simulations:

1. **Installation verification**: Check versions, CUDA, cuQuantum before debugging logic
2. **Detailed logging**: Enable DEBUG logging to trace operations
3. **Error tracking**: Monitor truncation error accumulation per bond
4. **Gate verification**: Test unitarity and correctness of gates
5. **Numerical stability**: Check condition numbers to detect ill-conditioning
6. **Checkpoint analysis**: Save intermediate states for offline debugging
7. **Performance profiling**: Use PyTorch profiler to identify bottlenecks

Common debugging workflows:

- **Incorrect results**: Verify gates → check normalization → increase bond dim
- **Slow simulation**: Profile → optimize hot spots → consider cuQuantum
- **OOM errors**: Check memory usage → reduce bond dim → clear cache
- **Numerical instability**: Monitor condition numbers → use complex128 → reduce dt

Debugging checklist:

1. Verify installation and dependencies
2. Enable verbose logging
3. Test gates in isolation
4. Monitor error accumulation
5. Check MPS normalization
6. Profile performance
7. Compare with known results (small systems)

See Also
========

- :doc:`../explanations/numerical_stability`: Understanding numerical errors
- :doc:`save_load_state`: Checkpoint-based debugging
- :doc:`optimize_performance`: Performance debugging and profiling
- :doc:`configure_precision`: Precision-related debugging
- :doc:`../faq`: Common issues and solutions
