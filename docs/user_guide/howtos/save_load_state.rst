========================================
Save and Load State
========================================

Problem
=======

Long-running quantum simulations (VQE, TDVP, QAOA) may run for hours or days.
Checkpointing is essential to:

- **Prevent data loss** from crashes, preemption, or hardware failures
- **Resume interrupted optimizations** without restarting from scratch
- **Share results** with collaborators or across compute sessions
- **Debug incrementally** by analyzing intermediate states
- **Manage HPC resources** by splitting work across multiple jobs

This guide covers strategies for checkpointing MPS states, variational parameters,
optimization histories, and distributed simulations.

.. seealso::
   :doc:`handle_large_systems` for distributed checkpointing,
   :doc:`debug_simulations` for loading checkpoints for debugging,
   :doc:`../explanations/design_decisions` for serialization format details.

Prerequisites
=============

You need:

- ATLAS-Q with MPS, VQE, TDVP, or QAOA configured
- Storage space for checkpoints (MPS with n=50, χ=256 ≈ 0.5-2 GB per checkpoint)
- Understanding of your optimization problem structure

Strategies
==========

Strategy 1: Basic MPS Checkpointing
------------------------------------

Save and load MPS state for later analysis or resumption.

**Save complete MPS state**:

.. code-block:: python

   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   # Run simulation
   mps = AdaptiveMPS(num_qubits=50, bond_dim=64, device='cuda')
   # ... apply gates, run algorithm ...

   # Save complete state
   checkpoint = {
       'tensors': [t.cpu() for t in mps.tensors],
       'num_qubits': mps.num_qubits,
       'bond_dims': mps.bond_dims,
       'statistics': {
           'truncation_error': mps.statistics.total_truncation_error,
           'max_bond_dim': mps.statistics.max_bond_dim,
           'num_operations': mps.statistics.num_operations
       },
       'metadata': {
           'timestamp': torch.tensor([time.time()]),
           'atlas_q_version': '0.6.1'
       }
   }

   torch.save(checkpoint, 'mps_checkpoint.pt')
   print(f"Saved MPS state: {sum(t.numel() for t in mps.tensors)} elements")

**Load MPS state**:

.. code-block:: python

   # Load from disk
   checkpoint = torch.load('mps_checkpoint.pt')

   # Reconstruct MPS
   mps = AdaptiveMPS(
       num_qubits=checkpoint['num_qubits'],
       bond_dim=max(checkpoint['bond_dims']),  # Use max bond dim
       device='cuda'
   )

   # Restore tensors and bond dimensions
   mps.tensors = [t.to('cuda') for t in checkpoint['tensors']]
   mps.bond_dims = checkpoint['bond_dims']

   # Restore statistics
   mps.statistics.total_truncation_error = checkpoint['statistics']['truncation_error']
   mps.statistics.max_bond_dim = checkpoint['statistics']['max_bond_dim']

   print(f"Loaded MPS state: {mps.num_qubits} qubits, max χ={max(mps.bond_dims)}")

Strategy 2: Periodic Checkpointing for Long Simulations
--------------------------------------------------------

Save checkpoints at regular intervals during time evolution or optimization.

**Checkpoint TDVP time evolution**:

.. code-block:: python

   from atlas_q.tdvp import TDVP
   import os

   # Configure TDVP
   tdvp = TDVP(
       hamiltonian=H,
       mps=mps,
       dt=0.01,
       device='cuda'
   )

   # Checkpoint directory
   checkpoint_dir = 'tdvp_checkpoints'
   os.makedirs(checkpoint_dir, exist_ok=True)

   # Evolve with periodic checkpointing
   checkpoint_interval = 100  # Steps
   total_steps = 10000

   energies = []
   for step in range(total_steps):
       # Single TDVP step
       E = tdvp.evolve_step()
       energies.append(E)

       # Checkpoint every N steps
       if (step + 1) % checkpoint_interval == 0:
           checkpoint_path = os.path.join(
               checkpoint_dir,
               f'tdvp_step_{step+1:06d}.pt'
           )

           torch.save({
               'step': step + 1,
               'tensors': [t.cpu() for t in tdvp.mps.tensors],
               'bond_dims': tdvp.mps.bond_dims,
               'energy': E,
               'energies': energies.copy(),
               'time': (step + 1) * tdvp.dt
           }, checkpoint_path)

           print(f"[Step {step+1}/{total_steps}] E={E:.6f}, "
                 f"checkpoint saved to {checkpoint_path}")

   print(f"Completed {total_steps} steps, {total_steps // checkpoint_interval} checkpoints saved")

**Resume TDVP from checkpoint**:

.. code-block:: python

   import glob

   # Find latest checkpoint
   checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'tdvp_step_*.pt')))
   if not checkpoints:
       print("No checkpoints found, starting from scratch")
       start_step = 0
       energies = []
   else:
       latest_checkpoint = checkpoints[-1]
       print(f"Resuming from {latest_checkpoint}")

       # Load checkpoint
       checkpoint = torch.load(latest_checkpoint)
       start_step = checkpoint['step']
       energies = checkpoint['energies']

       # Restore MPS
       mps = AdaptiveMPS(
           num_qubits=len(checkpoint['tensors']),
           bond_dim=max(checkpoint['bond_dims']),
           device='cuda'
       )
       mps.tensors = [t.to('cuda') for t in checkpoint['tensors']]
       mps.bond_dims = checkpoint['bond_dims']

       # Recreate TDVP
       tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device='cuda')

       print(f"Resumed from step {start_step}, E={checkpoint['energy']:.6f}")

   # Continue evolution
   for step in range(start_step, total_steps):
       E = tdvp.evolve_step()
       energies.append(E)
       # ... checkpoint as before ...

Strategy 3: Adaptive Checkpointing (Save on Improvement)
---------------------------------------------------------

Save checkpoints only when optimization metrics improve, reducing storage overhead.

**Checkpoint VQE when energy improves**:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import numpy as np

   # Configure VQE with callback
   config = VQEConfig(
       max_iterations=1000,
       optimizer='adam',
       learning_rate=0.01
   )

   vqe = VQE(hamiltonian=H, config=config, device='cuda')

   # Track best energy and parameters
   best_energy = np.inf
   best_params = None
   checkpoint_count = 0

   def checkpoint_callback(params, energy, iteration):
       """Save checkpoint when energy improves."""
       global best_energy, best_params, checkpoint_count

       if energy < best_energy - 1e-6:  # Improvement threshold
           best_energy = energy
           best_params = params.clone()
           checkpoint_count += 1

           torch.save({
               'iteration': iteration,
               'energy': energy,
               'params': params.cpu(),
               'energies': vqe.energies.copy(),
               'param_history': [p.cpu() for p in vqe.param_history],
               'timestamp': time.time()
           }, f'vqe_best_checkpoint_{checkpoint_count:04d}.pt')

           print(f"[Iter {iteration}] New best energy: {energy:.8f} (saved checkpoint)")
       else:
           print(f"[Iter {iteration}] Energy: {energy:.8f} (no improvement)")

   # Run VQE with adaptive checkpointing
   energy, params = vqe.optimize(callback=checkpoint_callback)

   print(f"Final energy: {energy:.8f}")
   print(f"Total checkpoints saved: {checkpoint_count}")

**Resume VQE with warm start**:

.. code-block:: python

   # Load best checkpoint
   checkpoint_files = glob.glob('vqe_best_checkpoint_*.pt')
   if checkpoint_files:
       latest = sorted(checkpoint_files)[-1]
       checkpoint = torch.load(latest)

       print(f"Warm starting VQE from checkpoint at E={checkpoint['energy']:.8f}")

       # Use previous best parameters as initial guess
       initial_params = checkpoint['params'].to('cuda')

       # Continue optimization
       energy, params = vqe.optimize(initial_params=initial_params)
   else:
       # Cold start
       energy, params = vqe.optimize()

Strategy 4: Compressed Checkpoints
-----------------------------------

Reduce checkpoint size using compression and precision reduction.

**Compressed MPS checkpoint**:

.. code-block:: python

   import gzip
   import pickle

   def save_compressed_mps(mps, path, precision='float32'):
       """
       Save MPS with compression.

       Parameters
       ----------
       mps : AdaptiveMPS
           MPS to save
       path : str
           Output path (will append .gz)
       precision : str
           'float32' or 'float16' for reduced precision
       """
       # Convert tensors to CPU and reduce precision if requested
       if precision == 'float16':
           tensors = [t.cpu().half() for t in mps.tensors]
       else:
           tensors = [t.cpu().float() for t in mps.tensors]

       checkpoint = {
           'tensors': tensors,
           'num_qubits': mps.num_qubits,
           'bond_dims': mps.bond_dims,
           'precision': precision
       }

       # Serialize and compress
       serialized = pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)

       with gzip.open(path + '.gz', 'wb', compresslevel=6) as f:
           f.write(serialized)

       # Report compression ratio
       uncompressed_size = len(serialized) / 1024**2
       compressed_size = os.path.getsize(path + '.gz') / 1024**2
       compression_ratio = uncompressed_size / compressed_size

       print(f"Saved compressed MPS: {compressed_size:.2f} MB "
             f"(compression ratio: {compression_ratio:.2f}x)")

   def load_compressed_mps(path, device='cuda'):
       """Load compressed MPS checkpoint."""
       with gzip.open(path, 'rb') as f:
           serialized = f.read()

       checkpoint = pickle.loads(serialized)

       # Restore MPS
       mps = AdaptiveMPS(
           num_qubits=checkpoint['num_qubits'],
           bond_dim=max(checkpoint['bond_dims']),
           device=device
       )

       # Convert back to complex64 on target device
       mps.tensors = [t.to(torch.complex64).to(device) for t in checkpoint['tensors']]
       mps.bond_dims = checkpoint['bond_dims']

       print(f"Loaded compressed MPS (original precision: {checkpoint['precision']})")

       return mps

   # Usage
   save_compressed_mps(mps, 'mps_checkpoint.pt', precision='float32')
   mps_restored = load_compressed_mps('mps_checkpoint.pt.gz', device='cuda')

**Storage comparison**:

.. code-block:: python

   # Example: n=50, χ=128
   # Uncompressed complex64: ~200 MB
   # Compressed complex64 (gzip): ~50-80 MB (3-4x reduction)
   # Compressed float32: ~25-40 MB (5-8x reduction)
   # Compressed float16: ~12-20 MB (10-16x reduction)

   # Trade-off: float16 loses precision but may be acceptable for checkpointing

Strategy 5: Distributed State Checkpointing
--------------------------------------------

Save and load MPS distributed across multiple GPUs or nodes.

**Checkpoint distributed MPS (bond-parallel)**:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   import torch.distributed as dist

   # Distributed MPS across 4 GPUs
   config = DistributedConfig(
       mode='bond_parallel',
       world_size=4,
       backend='nccl',
       device_ids=[0, 1, 2, 3]
   )

   mps = DistributedMPS(
       num_qubits=80,
       bond_dim=512,  # Split across 4 GPUs = 128 per GPU
       config=config
   )

   # Each rank saves its partition
   rank = dist.get_rank()

   # Save local partition
   local_checkpoint = {
       'rank': rank,
       'tensors': [t.cpu() for t in mps.local_tensors],
       'bond_dims': mps.local_bond_dims,
       'num_qubits': mps.num_qubits,
       'world_size': config.world_size
   }

   torch.save(local_checkpoint, f'distributed_mps_rank_{rank}.pt')

   # Rank 0 saves metadata
   if rank == 0:
       metadata = {
           'num_qubits': mps.num_qubits,
           'global_bond_dim': mps.bond_dim,
           'world_size': config.world_size,
           'mode': config.mode
       }
       torch.save(metadata, 'distributed_mps_metadata.pt')

   dist.barrier()  # Synchronize
   print(f"Rank {rank}: saved local partition")

**Load distributed MPS checkpoint**:

.. code-block:: python

   # Load metadata
   metadata = torch.load('distributed_mps_metadata.pt')

   # Reconstruct DistributedConfig
   config = DistributedConfig(
       mode=metadata['mode'],
       world_size=metadata['world_size'],
       backend='nccl',
       device_ids=list(range(metadata['world_size']))
   )

   # Create distributed MPS
   mps = DistributedMPS(
       num_qubits=metadata['num_qubits'],
       bond_dim=metadata['global_bond_dim'],
       config=config
   )

   # Each rank loads its partition
   rank = dist.get_rank()
   local_checkpoint = torch.load(f'distributed_mps_rank_{rank}.pt')

   mps.local_tensors = [t.to(f'cuda:{rank}') for t in local_checkpoint['tensors']]
   mps.local_bond_dims = local_checkpoint['bond_dims']

   dist.barrier()
   print(f"Rank {rank}: loaded local partition")

Strategy 6: Version-Compatible Checkpoints
-------------------------------------------

Ensure checkpoints remain loadable across ATLAS-Q versions.

**Save with version metadata**:

.. code-block:: python

   import atlas_q

   def save_versioned_checkpoint(mps, path, extra_metadata=None):
       """
       Save checkpoint with version metadata for compatibility.

       Parameters
       ----------
       mps : AdaptiveMPS
           MPS to save
       path : str
           Checkpoint path
       extra_metadata : dict, optional
           Additional metadata to save
       """
       checkpoint = {
           # MPS data
           'tensors': [t.cpu() for t in mps.tensors],
           'num_qubits': mps.num_qubits,
           'bond_dims': mps.bond_dims,

           # Version information
           'atlas_q_version': atlas_q.__version__,
           'torch_version': torch.__version__,
           'checkpoint_format_version': '1.0',

           # Save timestamp
           'timestamp': time.time(),
           'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S'),

           # Extra metadata
           'metadata': extra_metadata or {}
       }

       torch.save(checkpoint, path)

       print(f"Saved checkpoint with ATLAS-Q v{atlas_q.__version__}")

   def load_versioned_checkpoint(path, device='cuda'):
       """Load checkpoint with version compatibility checks."""
       checkpoint = torch.load(path)

       # Check version compatibility
       saved_version = checkpoint.get('atlas_q_version', 'unknown')
       current_version = atlas_q.__version__

       if saved_version != current_version:
           print(f"Warning: Checkpoint saved with ATLAS-Q v{saved_version}, "
                 f"loading with v{current_version}")
           # Could implement migration logic here

       # Check format version
       format_version = checkpoint.get('checkpoint_format_version', '0.0')
       if format_version != '1.0':
           raise ValueError(f"Unsupported checkpoint format: {format_version}")

       # Load MPS
       mps = AdaptiveMPS(
           num_qubits=checkpoint['num_qubits'],
           bond_dim=max(checkpoint['bond_dims']),
           device=device
       )
       mps.tensors = [t.to(device) for t in checkpoint['tensors']]
       mps.bond_dims = checkpoint['bond_dims']

       print(f"Loaded checkpoint from {checkpoint['timestamp_readable']}")

       return mps, checkpoint.get('metadata', {})

   # Usage
   save_versioned_checkpoint(mps, 'mps_v1.pt', extra_metadata={'experiment': 'qaoa_maxcut'})
   mps_loaded, metadata = load_versioned_checkpoint('mps_v1.pt')

Strategy 7: Checkpointing QAOA Progress
----------------------------------------

Save intermediate QAOA results for warm-starting and analysis.

**QAOA checkpointing with layer tracking**:

.. code-block:: python

   from atlas_q.vqe_qaoa import QAOA, QAOAConfig

   # Configure QAOA
   config = QAOAConfig(
       p=5,  # 5 layers
       optimizer='adam',
       learning_rate=0.02,
       max_iterations=500
   )

   qaoa = QAOA(hamiltonian=H, config=config, device='cuda')

   # Checkpoint directory
   checkpoint_dir = 'qaoa_checkpoints'
   os.makedirs(checkpoint_dir, exist_ok=True)

   # Optimize with checkpointing every 50 iterations
   iteration_count = 0

   def qaoa_checkpoint_callback(params, energy, iteration):
       """Checkpoint QAOA progress."""
       global iteration_count
       iteration_count = iteration

       if iteration % 50 == 0:
           # Extract gamma and beta parameters
           p = config.p
           gamma = params[:p]
           beta = params[p:]

           checkpoint = {
               'iteration': iteration,
               'energy': energy,
               'params': params.cpu(),
               'gamma': gamma.cpu(),
               'beta': beta.cpu(),
               'p': p,
               'energies': qaoa.energies.copy(),
               'param_history': [p.cpu() for p in qaoa.param_history]
           }

           torch.save(checkpoint, os.path.join(checkpoint_dir, f'qaoa_iter_{iteration:04d}.pt'))
           print(f"[Iter {iteration}] E={energy:.6f}, checkpoint saved")

   # Run QAOA
   energy, params = qaoa.optimize(callback=qaoa_checkpoint_callback)

   # Save final result
   torch.save({
       'final_energy': energy,
       'final_params': params.cpu(),
       'all_energies': qaoa.energies,
       'total_iterations': iteration_count
   }, os.path.join(checkpoint_dir, 'qaoa_final.pt'))

Troubleshooting
===============

Checkpoint Loading Fails
-------------------------

**Problem**: ``torch.load()`` raises ``RuntimeError: storage has wrong size``.

**Solution**: Checkpoint was saved on different device or with different dtype.

.. code-block:: python

   # Always load to CPU first, then move to device
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')
   mps.tensors = [t.to('cuda') for t in checkpoint['tensors']]

Out of Memory When Loading
---------------------------

**Problem**: Loading large checkpoint exhausts CPU or GPU memory.

**Solution**: Load tensors incrementally or use memory mapping.

.. code-block:: python

   # Load tensors one at a time
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')

   mps = AdaptiveMPS(num_qubits=checkpoint['num_qubits'], bond_dim=64, device='cuda')

   for i, tensor in enumerate(checkpoint['tensors']):
       mps.tensors[i] = tensor.to('cuda')
       # Clear CPU cache
       del tensor
       if i % 10 == 0:
           torch.cuda.empty_cache()

Distributed Checkpoint Mismatch
--------------------------------

**Problem**: Loading distributed checkpoint fails with ``world_size`` mismatch.

**Solution**: Saved with 4 GPUs, loading with 2 GPUs requires resharding.

.. code-block:: python

   # Load all partitions to CPU
   metadata = torch.load('distributed_mps_metadata.pt')
   all_tensors = []

   for rank in range(metadata['world_size']):
       partition = torch.load(f'distributed_mps_rank_{rank}.pt')
       all_tensors.extend(partition['tensors'])

   # Reconstruct as single-device MPS
   mps = AdaptiveMPS(
       num_qubits=metadata['num_qubits'],
       bond_dim=metadata['global_bond_dim'],
       device='cuda'
   )
   mps.tensors = [t.to('cuda') for t in all_tensors]

   # Now redistribute if needed

Checkpoint Size Too Large
--------------------------

**Problem**: 50 GB checkpoints fill disk quickly.

**Solution**: Use compression, reduce precision, or checkpoint less frequently.

.. code-block:: python

   # Reduce checkpoint frequency
   checkpoint_interval = 200  # Instead of 50

   # Use compressed float32 instead of complex64
   save_compressed_mps(mps, 'checkpoint.pt', precision='float32')

   # Keep only last N checkpoints
   max_checkpoints = 5
   checkpoints = sorted(glob.glob('checkpoint_*.pt'))
   if len(checkpoints) > max_checkpoints:
       for old_checkpoint in checkpoints[:-max_checkpoints]:
           os.remove(old_checkpoint)
           print(f"Removed old checkpoint: {old_checkpoint}")

Summary
=======

Checkpointing strategies for ATLAS-Q simulations:

1. **Basic checkpointing**: Save/load MPS tensors, bond dimensions, and statistics
2. **Periodic checkpointing**: Save at fixed intervals during long TDVP/VQE runs
3. **Adaptive checkpointing**: Save only when metrics improve to reduce storage
4. **Compressed checkpoints**: Use gzip + reduced precision for 5-16x size reduction
5. **Distributed checkpointing**: Save partitioned state across multiple devices
6. **Version-compatible checkpoints**: Include metadata for cross-version compatibility
7. **Algorithm-specific checkpointing**: QAOA, VQE, TDVP with resume capabilities

Choose checkpointing frequency based on:

- **Simulation length**: Longer runs need more frequent checkpoints
- **Storage budget**: Compression vs. precision trade-off
- **Failure rate**: Unreliable hardware needs more checkpoints
- **Restart cost**: Expensive initialization favors more checkpoints

See Also
========

- :doc:`handle_large_systems`: Distributed MPS checkpointing for multi-GPU systems
- :doc:`debug_simulations`: Load checkpoints to debug intermediate states
- :doc:`optimize_performance`: Checkpoint I/O performance optimization
- :doc:`../explanations/design_decisions`: Serialization format and compatibility
- :doc:`../tutorials/vqe_tutorial`: VQE checkpointing examples
- :doc:`../tutorials/tdvp_tutorial`: TDVP time evolution checkpointing
