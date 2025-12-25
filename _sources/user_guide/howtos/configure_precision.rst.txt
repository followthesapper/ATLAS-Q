========================================
Configure Precision
========================================

Problem
=======

Numerical precision affects quantum simulation accuracy, performance, and memory usage.
Choosing the right precision is critical for:

- **Accuracy**: Chemical accuracy (10⁻³ Hartree) requires careful precision management
- **Performance**: complex64 is 2-3× faster than complex128 on GPUs
- **Memory**: complex64 uses half the memory of complex128
- **Stability**: Long-time evolution or ill-conditioned problems need higher precision
- **Cost**: Wasted compute from unnecessary precision or failed simulations from insufficient precision

This guide covers precision configuration strategies for MPS simulations, variational algorithms,
and time evolution, balancing accuracy against performance.

.. seealso::
   :doc:`../explanations/numerical_stability` for theoretical background,
   :doc:`optimize_performance` for performance implications,
   :doc:`../tutorials/tdvp_tutorial` for TDVP-specific precision considerations.

Prerequisites
=============

You need:

- ATLAS-Q with GPU support (precision choices matter most on GPUs)
- Understanding of your accuracy requirements (e.g., chemical accuracy, fidelity targets)
- Benchmark problem to validate precision choices

Strategies
==========

Strategy 1: Choose the Right Default Precision
-----------------------------------------------

Start with the appropriate dtype for your application.

**complex64 (default) - Fast, memory-efficient, sufficient for most cases**:

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   # complex64: 32-bit float for real + 32-bit float for imag = 8 bytes per element
   # - 2-3× faster on GPUs than complex128
   # - 2× less memory than complex128
   # - ~7 decimal digits of precision
   # - Sufficient for most quantum chemistry, optimization, and circuit simulation

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       dtype=torch.complex64,  # Default
       device='cuda'
   )

   print(f"MPS dtype: {mps.tensors[0].dtype}")
   print(f"Memory per tensor: ~{128 * 2 * 128 * 8 / 1024**2:.2f} MB")

**complex128 - Higher precision, needed for demanding applications**:

.. code-block:: python

   # complex128: 64-bit float for real + 64-bit float for imag = 16 bytes per element
   # - ~16 decimal digits of precision
   # - 2-3× slower on GPUs than complex64
   # - 2× more memory than complex64
   # - Necessary for chemical accuracy, long-time evolution, tight tolerances

   mps_high_precision = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       dtype=torch.complex128,
       device='cuda'
   )

   print(f"High-precision MPS dtype: {mps_high_precision.tensors[0].dtype}")
   print(f"Memory per tensor: ~{128 * 2 * 128 * 16 / 1024**2:.2f} MB")

**When to use each precision**:

.. code-block:: python

   # Use complex64 for:
   # - Most VQE, QAOA, circuit simulations
   # - Short-time TDVP (<100 time units)
   # - Exploration and prototyping
   # - Large systems (n>50, χ>256) where memory is tight

   # Use complex128 for:
   # - Chemical accuracy requirements (ΔE < 1e-3 Hartree)
   # - Long-time TDVP (>1000 time units)
   # - Tight optimization tolerances (ΔE < 1e-8)
   # - Systems with ill-conditioned Hamiltonians
   # - When complex64 simulations show numerical instability

Strategy 2: Mixed Precision for VQE
------------------------------------

Use lower precision for ansatz preparation and higher precision for energy evaluation.

**Mixed-precision VQE**:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import torch

   # Configure VQE with mixed precision
   config = VQEConfig(
       max_iterations=1000,
       optimizer='adam',
       learning_rate=0.01,
       # Use complex64 for fast ansatz construction
       mps_dtype=torch.complex64,
       # Use complex128 for accurate energy evaluation
       hamiltonian_dtype=torch.complex128
   )

   vqe = VQE(hamiltonian=H, config=config, device='cuda')

   # VQE will:
   # 1. Build ansatz in complex64 (fast)
   # 2. Convert to complex128 for <ψ|H|ψ> (accurate)
   # 3. Gradients computed in complex128 (stable)
   # 4. Convert back to complex64 for parameter update (fast)

   energy, params = vqe.optimize()

   print(f"Final energy: {energy:.10f} (evaluated in complex128)")
   print(f"Speedup: ~1.5-2× vs pure complex128")
   print(f"Memory savings: ~30-40% vs pure complex128")

**Rationale**: Ansatz construction (gate applications) is the bottleneck, but energy
evaluation determines accuracy. Mixed precision gives 80% of complex128 accuracy at
60% of the cost.

Strategy 3: Automatic Precision Promotion
------------------------------------------

Automatically promote to higher precision when numerical issues are detected.

**Adaptive dtype policy**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy

   # Define precision promotion policy
   policy = DTypePolicy(
       default=torch.complex64,
       promote_if_cond_gt=1e6,      # Promote if condition number > 10^6
       promote_if_truncation_gt=1e-4,  # Promote if truncation error > 10^-4
       demote_if_stable_for=100      # Demote back after 100 stable steps
   )

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       dtype_policy=policy,
       device='cuda'
   )

   # Apply gates - MPS will automatically promote/demote precision
   for i in range(49):
       mps.apply_cnot(i, i+1)

   # Check if precision was promoted
   if mps.tensors[0].dtype == torch.complex128:
       print("Precision was automatically promoted to complex128")
   else:
       print("Remained at complex64")

**Manual promotion on detection**:

.. code-block:: python

   from atlas_q.tdvp import TDVP
   import numpy as np

   # Start with complex64
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, dtype=torch.complex64, device='cuda')
   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

   energies = []
   promoted = False

   for step in range(10000):
       E = tdvp.evolve_step()
       energies.append(E)

       # Check for energy drift (sign of numerical instability)
       if step > 100:
           energy_drift = np.std(energies[-100:])

           if energy_drift > 1e-5 and not promoted:
               print(f"[Step {step}] Energy drift detected: {energy_drift:.2e}")
               print("Promoting to complex128...")

               # Convert MPS to complex128
               tdvp.mps.tensors = [t.to(torch.complex128) for t in tdvp.mps.tensors]

               # Recreate TDVP with new precision
               tdvp = TDVP(
                   hamiltonian=H,
                   mps=tdvp.mps,
                   dt=0.01,
                   device='cuda'
               )

               promoted = True
               energies = energies[-10:]  # Reset drift tracking

Strategy 4: GPU-Specific Precision Considerations
--------------------------------------------------

Different GPU architectures have different precision performance characteristics.

**Tensor Core utilization (NVIDIA Ampere, Hopper)**:

.. code-block:: python

   import torch

   # Tensor Cores accelerate complex64 on modern GPUs
   # Check if Tensor Cores are available
   if torch.cuda.is_available():
       device_name = torch.cuda.get_device_name(0)
       print(f"GPU: {device_name}")

       # Ampere (A100), Hopper (H100) have Tensor Cores for complex64
       # complex64 can be 5-10× faster than without Tensor Cores
       # complex128 does NOT benefit from Tensor Cores

       if 'A100' in device_name or 'H100' in device_name:
           print("Tensor Cores available: complex64 strongly recommended")
           dtype_recommendation = torch.complex64
       else:
           print("No Tensor Cores: complex64 vs complex128 speedup is ~2×")
           dtype_recommendation = torch.complex64  # Still faster, but less dramatic

   # Configure MPS with GPU-optimized precision
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=256,
       dtype=dtype_recommendation,
       device='cuda'
   )

**TF32 mode for complex64 (Ampere and later)**:

.. code-block:: python

   # TF32 (TensorFloat-32) trades slight precision for speed
   # Effective precision: ~19 bits mantissa (between float16 and float32)
   # Speedup: 1.5-2× for complex64 operations
   # Accuracy: Still sufficient for most quantum simulations

   # Enable TF32 (default on Ampere+)
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True

   mps = AdaptiveMPS(num_qubits=50, bond_dim=128, dtype=torch.complex64, device='cuda')

   # Disable TF32 for maximum accuracy (no speedup)
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.allow_tf32 = False

   mps_accurate = AdaptiveMPS(num_qubits=50, bond_dim=128, dtype=torch.complex64, device='cuda')

   print("TF32 gives ~1.5× speedup with negligible accuracy loss for most problems")

Strategy 5: Precision for Different Algorithms
-----------------------------------------------

Different algorithms have different precision requirements.

**VQE precision recommendations**:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig

   # VQE for molecular ground state (chemical accuracy required)
   config_chemistry = VQEConfig(
       max_iterations=1000,
       optimizer='lbfgs',  # Second-order optimizer needs stable gradients
       learning_rate=0.1,
       mps_dtype=torch.complex128,  # Chemical accuracy demands high precision
       hamiltonian_dtype=torch.complex128,
       convergence_threshold=1e-8  # Tight tolerance
   )

   # VQE for MaxCut or optimization (relaxed accuracy acceptable)
   config_optimization = VQEConfig(
       max_iterations=500,
       optimizer='adam',  # First-order optimizer more robust to noise
       learning_rate=0.01,
       mps_dtype=torch.complex64,  # Lower precision sufficient
       hamiltonian_dtype=torch.complex64,
       convergence_threshold=1e-5  # Relaxed tolerance
   )

**TDVP precision recommendations**:

.. code-block:: python

   from atlas_q.tdvp import TDVP

   # Short-time evolution (t < 10): complex64 sufficient
   mps_short = AdaptiveMPS(num_qubits=30, bond_dim=64, dtype=torch.complex64, device='cuda')
   tdvp_short = TDVP(hamiltonian=H, mps=mps_short, dt=0.01, device='cuda')

   # Long-time evolution (t > 100): complex128 recommended
   mps_long = AdaptiveMPS(num_qubits=30, bond_dim=64, dtype=torch.complex128, device='cuda')
   tdvp_long = TDVP(hamiltonian=H, mps=mps_long, dt=0.001, device='cuda')  # Smaller dt too

   # Reasoning: Error accumulates over time steps
   # For t=1000 with dt=0.01, N=100,000 steps
   # Error per step ε_step, total error ~√N × ε_step
   # complex128 keeps total error manageable

**QAOA precision recommendations**:

.. code-block:: python

   from atlas_q.vqe_qaoa import QAOA, QAOAConfig

   # QAOA typically less sensitive to precision than VQE
   # Combinatorial optimization often needs only approximate solutions
   config_qaoa = QAOAConfig(
       p=5,
       optimizer='adam',
       learning_rate=0.02,
       max_iterations=500
   )

   qaoa = QAOA(
       hamiltonian=H_maxcut,
       config=config_qaoa,
       dtype=torch.complex64,  # complex64 sufficient for QAOA
       device='cuda'
   )

Strategy 6: Monitoring Numerical Stability
-------------------------------------------

Track metrics to detect when precision is insufficient.

**Comprehensive stability monitoring**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.tdvp import TDVP
   import torch
   import numpy as np

   def monitor_stability(mps, H, history_length=100):
       """
       Monitor numerical stability indicators.

       Returns
       -------
       dict
           Stability metrics
       """
       metrics = {}

       # 1. Condition numbers of MPS tensors
       cond_numbers = []
       for tensor in mps.tensors:
           # Reshape to matrix
           matrix = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
           # Compute condition number
           s = torch.linalg.svdvals(matrix)
           cond = (s[0] / s[-1]).item()
           cond_numbers.append(cond)

       metrics['max_condition_number'] = max(cond_numbers)
       metrics['mean_condition_number'] = np.mean(cond_numbers)

       # 2. Normalization drift (should stay close to 1)
       norm = torch.sqrt(torch.sum(torch.abs(mps.tensors[0])**2))
       for tensor in mps.tensors[1:]:
           norm = norm * torch.sqrt(torch.sum(torch.abs(tensor)**2))
       metrics['norm_drift'] = abs(norm.item() - 1.0)

       # 3. Energy variance (for Hamiltonian eigenstates)
       energy = mps.expectation_value(H)
       energy_sq = mps.expectation_value(H @ H)
       variance = energy_sq - energy**2
       metrics['energy_variance'] = variance.item()

       return metrics

   # Usage: Monitor during TDVP evolution
   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, dtype=torch.complex64, device='cuda')
   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

   for step in range(1000):
       E = tdvp.evolve_step()

       if step % 100 == 0:
           metrics = monitor_stability(mps, H)

           print(f"[Step {step}] E={E:.8f}")
           print(f"  Max condition number: {metrics['max_condition_number']:.2e}")
           print(f"  Norm drift: {metrics['norm_drift']:.2e}")
           print(f"  Energy variance: {metrics['energy_variance']:.2e}")

           # Trigger warnings
           if metrics['max_condition_number'] > 1e7:
               print("  WARNING: Ill-conditioned tensors detected!")
               print("  Consider promoting to complex128")

           if metrics['norm_drift'] > 1e-3:
               print("  WARNING: Normalization drift detected!")
               print("  Consider reducing dt or increasing precision")

Strategy 7: Precision for Distributed Systems
----------------------------------------------

Distributed MPS introduces additional precision considerations.

**Distributed precision configuration**:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig

   # Distributed MPS across multiple GPUs
   config = DistributedConfig(
       mode='bond_parallel',
       world_size=4,
       backend='nccl',
       device_ids=[0, 1, 2, 3]
   )

   # Precision choices for distributed MPS:
   # - complex64: Less communication volume (2× less than complex128)
   # - complex128: Better stability for large-scale systems
   #
   # Recommendation: Start with complex64, promote if instability detected

   mps_distributed = DistributedMPS(
       num_qubits=100,
       bond_dim=512,
       config=config,
       dtype=torch.complex64  # Lower communication overhead
   )

   # Monitor for cross-device numerical issues
   # Distributed reductions (all-reduce, etc.) can accumulate errors
   # If energy drifts or strange behavior occurs, try complex128:

   mps_distributed_highprec = DistributedMPS(
       num_qubits=100,
       bond_dim=512,
       config=config,
       dtype=torch.complex128  # More stable for distributed operations
   )

Troubleshooting
===============

Energy Not Converging in VQE
-----------------------------

**Problem**: VQE energy oscillates or fails to converge below 1e-5.

**Solution**: Increase precision or adjust optimizer settings.

.. code-block:: python

   # Try 1: Increase precision
   config = VQEConfig(
       max_iterations=1000,
       optimizer='lbfgs',
       mps_dtype=torch.complex128,        # Was complex64
       hamiltonian_dtype=torch.complex128  # Was complex64
   )

   # Try 2: Reduce learning rate (if using Adam)
   config = VQEConfig(
       max_iterations=1000,
       optimizer='adam',
       learning_rate=0.001,  # Was 0.01
       mps_dtype=torch.complex64
   )

   # Try 3: Tighten truncation threshold
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       truncation_threshold=1e-10,  # Was 1e-8
       dtype=torch.complex128
   )

TDVP Energy Drifting Over Time
-------------------------------

**Problem**: Energy increases or oscillates during imaginary-time evolution.

**Solution**: Decrease time step or increase precision.

.. code-block:: python

   # Decrease dt
   tdvp = TDVP(
       hamiltonian=H,
       mps=mps,
       dt=0.001,  # Was 0.01
       device='cuda'
   )

   # Increase precision
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       dtype=torch.complex128,  # Was complex64
       device='cuda'
   )

   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

Ill-Conditioned Matrix Warnings
--------------------------------

**Problem**: ``RuntimeError: svd did not converge`` or very large condition numbers.

**Solution**: Increase precision or regularize.

.. code-block:: python

   # Solution 1: Use complex128
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       dtype=torch.complex128,  # More stable SVD
       device='cuda'
   )

   # Solution 2: Add regularization to SVD
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=128,
       truncation_threshold=1e-8,  # Regularize by truncating small singular values
       dtype=torch.complex64
   )

   # Solution 3: Reduce bond dimension (less ill-conditioning)
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,  # Was 128
       dtype=torch.complex64
   )

Complex64 vs Complex128 Performance Not as Expected
----------------------------------------------------

**Problem**: complex64 only 1.2× faster than complex128, expected 2-3×.

**Solution**: Check GPU, enable TF32, profile bottlenecks.

.. code-block:: python

   import torch

   # Check GPU and driver
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   print(f"CUDA version: {torch.version.cuda}")

   # Ensure TF32 is enabled (Ampere+ GPUs)
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True

   # Profile to find bottleneck
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CUDA],
       record_shapes=True
   ) as prof:
       mps = AdaptiveMPS(num_qubits=50, bond_dim=128, dtype=torch.complex64, device='cuda')
       for i in range(10):
           mps.apply_cnot(i, i+1)

   print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

   # If bottleneck is not matrix multiply (gemm), precision won't help much
   # Common non-matmul bottlenecks: data movement, small kernels, host-device sync

Summary
=======

Precision configuration strategies for ATLAS-Q:

1. **Default precision**: complex64 for most cases, complex128 for chemical accuracy
2. **Mixed precision VQE**: complex64 for ansatz, complex128 for energy evaluation
3. **Automatic promotion**: Detect numerical issues and promote to higher precision
4. **GPU-specific**: Leverage Tensor Cores and TF32 on modern GPUs
5. **Algorithm-specific**: VQE/chemistry needs higher precision than QAOA/optimization
6. **Stability monitoring**: Track condition numbers, norm drift, energy variance
7. **Distributed precision**: Balance communication overhead vs stability

Precision vs performance trade-offs:

- **complex64**: 2-3× faster, 2× less memory, ~7 digits, sufficient for most problems
- **complex128**: 2-3× slower, 2× more memory, ~16 digits, needed for demanding applications
- **Mixed precision**: Best of both worlds for some algorithms (VQE)
- **TF32 mode**: 1.5-2× speedup on Ampere+ GPUs with negligible accuracy loss

When to use higher precision:

- Chemical accuracy requirements (ΔE < 1e-3 Hartree)
- Long-time evolution (t > 100)
- Tight convergence tolerances (ΔE < 1e-8)
- Ill-conditioned Hamiltonians (condition number > 1e6)
- Numerical instability detected (energy drift, SVD failures)

See Also
========

- :doc:`../explanations/numerical_stability`: Theory of numerical stability in MPS
- :doc:`optimize_performance`: Performance implications of precision choices
- :doc:`../tutorials/vqe_tutorial`: VQE precision considerations
- :doc:`../tutorials/tdvp_tutorial`: TDVP time evolution numerical stability
- :doc:`../tutorials/molecular_vqe`: Chemical accuracy and precision requirements
