========================================
Parallel Computation
========================================

Problem
=======

Quantum simulations are computationally expensive, but many tasks are parallelizable:

- **Large bond dimensions**: Single GPU insufficient for χ > 512
- **Parameter sweeps**: VQE/QAOA hyperparameter optimization requires many trials
- **Statistical sampling**: Need 10,000+ measurement shots for accurate distributions
- **Ensemble averaging**: Molecular dynamics, noise averaging, ensemble studies
- **Multi-task workflows**: Benchmark different algorithms or Hamiltonians simultaneously

This guide covers parallel computation strategies for ATLAS-Q, including multi-GPU
distributed MPS, data parallelism for independent simulations, and task parallelism
for parameter sweeps and ensemble studies.

.. seealso::
   :doc:`handle_large_systems` for distributed MPS details,
   :doc:`integrate_cuquantum` for GPU-accelerated operations,
   :doc:`optimize_performance` for single-GPU performance tuning,
   :doc:`../reference/distributed_mps` for API reference.

Prerequisites
=============

You need:

- **Multiple GPUs**: 2+ NVIDIA GPUs (same or different models)
- **PyTorch distributed**: Installed with CUDA support
- **Network**: For multi-node, InfiniBand or 10+ Gbps Ethernet
- **NCCL backend**: For efficient GPU-to-GPU communication
- **Storage**: Shared filesystem for multi-node checkpoints

Strategies
==========

Strategy 1: Multi-GPU Distributed MPS (Bond-Parallel)
------------------------------------------------------

Split MPS bond dimension across multiple GPUs for large-scale simulations.

**Bond-parallel MPS setup**:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   import torch.distributed as dist

   # Initialize distributed backend
   dist.init_process_group(
       backend='nccl',
       init_method='env://',  # Use environment variables
       world_size=4,          # 4 GPUs
       rank=int(os.environ['RANK'])
   )

   # Configure bond-parallel distribution
   config = DistributedConfig(
       mode='bond_parallel',   # Split bond dimension
       world_size=4,
       backend='nccl',
       device_ids=[0, 1, 2, 3]
   )

   # Create distributed MPS
   # Bond dimension 512 split across 4 GPUs = 128 per GPU
   mps = DistributedMPS(
       num_qubits=80,
       bond_dim=512,
       config=config
   )

   rank = dist.get_rank()
   print(f"Rank {rank}: Managing bond dim partition of {512 // 4}")

   # Apply gates - automatically distributed
   for i in range(79):
       mps.apply_cnot(i, i+1)

   dist.barrier()  # Synchronize all processes
   print(f"Rank {rank}: Completed distributed gate sequence")

**Launch distributed job**:

.. code-block:: bash

   # Single-node multi-GPU (4 GPUs)
   torchrun \
       --nproc_per_node=4 \
       --master_port=29500 \
       distributed_mps_script.py

   # Multi-node (2 nodes, 4 GPUs each = 8 GPUs total)
   # On node 0:
   torchrun \
       --nproc_per_node=4 \
       --nnodes=2 \
       --node_rank=0 \
       --master_addr=node0.example.com \
       --master_port=29500 \
       distributed_mps_script.py

   # On node 1:
   torchrun \
       --nproc_per_node=4 \
       --nnodes=2 \
       --node_rank=1 \
       --master_addr=node0.example.com \
       --master_port=29500 \
       distributed_mps_script.py

**Distributed TDVP example**:

.. code-block:: python

   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   from atlas_q.tdvp import TDVP
   import torch.distributed as dist

   # Initialize distributed backend
   dist.init_process_group(backend='nccl', init_method='env://')
   rank = dist.get_rank()
   world_size = dist.get_world_size()

   # Configure distributed MPS
   config = DistributedConfig(
       mode='bond_parallel',
       world_size=world_size,
       backend='nccl'
   )

   # Create large distributed MPS
   mps = DistributedMPS(
       num_qubits=100,
       bond_dim=1024,  # Split across GPUs
       config=config
   )

   # TDVP time evolution on distributed MPS
   tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.01, device=f'cuda:{rank}')

   for step in range(1000):
       E = tdvp.evolve_step()

       if rank == 0 and step % 100 == 0:
           print(f"[Step {step}] Energy: {E:.8f}")

   dist.barrier()
   dist.destroy_process_group()

Strategy 2: Data Parallelism for Independent Simulations
---------------------------------------------------------

Run multiple independent simulations in parallel for parameter sweeps or ensemble averaging.

**Parallel VQE hyperparameter search**:

.. code-block:: python

   import torch.multiprocessing as mp
   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import torch

   def run_vqe_trial(rank, H, learning_rates, return_dict):
       """
       Run VQE with specific learning rate on assigned GPU.

       Parameters
       ----------
       rank : int
           GPU index
       H : Hamiltonian
           Problem Hamiltonian
       learning_rates : list
           Learning rates to try (one per GPU)
       return_dict : multiprocessing.Manager.dict
           Shared dictionary for results
       """
       torch.cuda.set_device(rank)

       lr = learning_rates[rank]
       config = VQEConfig(
           max_iterations=500,
           optimizer='adam',
           learning_rate=lr
       )

       vqe = VQE(hamiltonian=H, config=config, device=f'cuda:{rank}')
       energy, params = vqe.optimize()

       # Store results
       return_dict[rank] = {
           'learning_rate': lr,
           'final_energy': energy,
           'num_iterations': len(vqe.energies),
           'convergence': vqe.energies
       }

       print(f"GPU {rank}: lr={lr:.4f}, E={energy:.8f}")

   if __name__ == '__main__':
       # Hyperparameter grid
       learning_rates = [0.001, 0.005, 0.01, 0.05]  # 4 trials for 4 GPUs
       num_gpus = len(learning_rates)

       # Shared result storage
       manager = mp.Manager()
       return_dict = manager.dict()

       # Launch parallel VQE trials
       mp.spawn(
           run_vqe_trial,
           args=(H, learning_rates, return_dict),
           nprocs=num_gpus,
           join=True
       )

       # Analyze results
       best_energy = float('inf')
       best_lr = None

       for rank, result in return_dict.items():
           lr = result['learning_rate']
           energy = result['final_energy']
           print(f"lr={lr:.4f}: E={energy:.8f}")

           if energy < best_energy:
               best_energy = energy
               best_lr = lr

       print(f"\nBest learning rate: {best_lr:.4f} (E={best_energy:.8f})")

**Parallel QAOA with different ansatz depths**:

.. code-block:: python

   from atlas_q.vqe_qaoa import QAOA, QAOAConfig
   import torch.multiprocessing as mp

   def run_qaoa_depth(rank, H, depths, results):
       """Run QAOA with specific depth p."""
       torch.cuda.set_device(rank)

       p = depths[rank]
       config = QAOAConfig(
           p=p,
           optimizer='adam',
           learning_rate=0.02,
           max_iterations=500
       )

       qaoa = QAOA(hamiltonian=H, config=config, device=f'cuda:{rank}')
       energy, params = qaoa.optimize()

       results[rank] = {
           'p': p,
           'energy': energy,
           'params': params.cpu()
       }

       print(f"GPU {rank}: p={p}, E={energy:.8f}")

   if __name__ == '__main__':
       # Test different QAOA depths
       depths = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 trials for 8 GPUs
       num_gpus = min(8, torch.cuda.device_count())

       manager = mp.Manager()
       results = manager.dict()

       # Run in parallel
       mp.spawn(
           run_qaoa_depth,
           args=(H_maxcut, depths[:num_gpus], results),
           nprocs=num_gpus,
           join=True
       )

       # Plot depth vs energy
       import matplotlib.pyplot as plt

       ps = [results[i]['p'] for i in range(num_gpus)]
       energies = [results[i]['energy'] for i in range(num_gpus)]

       plt.plot(ps, energies, 'o-')
       plt.xlabel('QAOA depth p')
       plt.ylabel('Final energy')
       plt.title('QAOA performance vs ansatz depth')
       plt.savefig('qaoa_depth_comparison.png')

Strategy 3: Task Parallelism for Ensemble Studies
--------------------------------------------------

Parallelize ensemble averaging or multi-molecule simulations.

**Parallel molecular VQE ensemble**:

.. code-block:: python

   from atlas_q.vqe_qaoa import VQE, VQEConfig
   import torch.multiprocessing as mp

   # Molecular Hamiltonians (e.g., H2 at different bond lengths)
   molecules = [
       ('H2_0.5A', H_h2_05),
       ('H2_0.7A', H_h2_07),
       ('H2_0.9A', H_h2_09),
       ('H2_1.1A', H_h2_11)
   ]

   def compute_molecule_energy(rank, molecule, results):
       """Compute ground state energy for a molecule."""
       name, H = molecule
       torch.cuda.set_device(rank)

       config = VQEConfig(
           max_iterations=1000,
           optimizer='lbfgs',
           convergence_threshold=1e-8
       )

       vqe = VQE(hamiltonian=H, config=config, device=f'cuda:{rank}')
       energy, params = vqe.optimize()

       results[name] = {
           'energy': energy,
           'params': params.cpu()
       }

       print(f"{name} on GPU {rank}: E={energy:.10f} Hartree")

   if __name__ == '__main__':
       num_gpus = len(molecules)
       manager = mp.Manager()
       results = manager.dict()

       # Compute all molecules in parallel
       processes = []
       for rank, molecule in enumerate(molecules):
           p = mp.Process(
               target=compute_molecule_energy,
               args=(rank, molecule, results)
           )
           p.start()
           processes.append(p)

       for p in processes:
           p.join()

       # Plot potential energy curve
       import matplotlib.pyplot as plt

       bond_lengths = [0.5, 0.7, 0.9, 1.1]
       energies = [results[name]['energy'] for name, _ in molecules]

       plt.plot(bond_lengths, energies, 'o-')
       plt.xlabel('Bond length (Å)')
       plt.ylabel('Energy (Hartree)')
       plt.title('H2 potential energy curve')
       plt.savefig('h2_pes.png')

Strategy 4: Parallel Measurement Sampling
------------------------------------------

Parallelize measurement shots across multiple GPUs for faster sampling.

**Parallel shot sampling**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch.multiprocessing as mp
   import torch

   def sample_on_gpu(rank, mps_state, num_shots, results):
       """Sample from MPS on specific GPU."""
       torch.cuda.set_device(rank)

       # Reconstruct MPS on this GPU
       mps = AdaptiveMPS(
           num_qubits=mps_state['num_qubits'],
           bond_dim=max(mps_state['bond_dims']),
           device=f'cuda:{rank}'
       )
       mps.tensors = [t.to(f'cuda:{rank}') for t in mps_state['tensors']]
       mps.bond_dims = mps_state['bond_dims']

       # Sample num_shots on this GPU
       samples = mps.sample(num_shots=num_shots)

       results[rank] = samples.cpu()
       print(f"GPU {rank}: Sampled {num_shots} shots")

   if __name__ == '__main__':
       # Prepare MPS state
       mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda:0')
       # ... apply circuit ...

       # Save state for distribution
       mps_state = {
           'num_qubits': mps.num_qubits,
           'tensors': [t.cpu() for t in mps.tensors],
           'bond_dims': mps.bond_dims
       }

       # Parallel sampling: 40,000 shots across 4 GPUs = 10,000 per GPU
       total_shots = 40000
       num_gpus = 4
       shots_per_gpu = total_shots // num_gpus

       manager = mp.Manager()
       results = manager.dict()

       # Launch sampling on each GPU
       mp.spawn(
           sample_on_gpu,
           args=(mps_state, shots_per_gpu, results),
           nprocs=num_gpus,
           join=True
       )

       # Combine samples
       all_samples = torch.cat([results[i] for i in range(num_gpus)], dim=0)
       print(f"Total samples: {all_samples.shape[0]}")

       # Compute expectation values from samples
       # ... analysis ...

Strategy 5: Circuit Batching Across GPUs
-----------------------------------------

Execute different circuits on different GPUs for benchmarking or algorithm comparison.

**Parallel circuit execution**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch.multiprocessing as mp

   def execute_circuit(rank, circuit, num_qubits, bond_dim, results):
       """Execute circuit on assigned GPU."""
       torch.cuda.set_device(rank)

       mps = AdaptiveMPS(
           num_qubits=num_qubits,
           bond_dim=bond_dim,
           device=f'cuda:{rank}'
       )

       # Apply circuit gates
       for gate in circuit:
           if gate['type'] == 'H':
               mps.apply_hadamard(gate['qubit'])
           elif gate['type'] == 'CNOT':
               mps.apply_cnot(gate['control'], gate['target'])
           # ... other gates ...

       # Compute final observables
       samples = mps.sample(num_shots=10000)
       energy = mps.expectation_value(hamiltonian)

       results[rank] = {
           'samples': samples.cpu(),
           'energy': energy,
           'final_chi': max(mps.bond_dims)
       }

       print(f"GPU {rank}: Circuit completed, E={energy:.6f}")

   if __name__ == '__main__':
       # Different circuits to compare
       circuits = [
           generate_qaoa_circuit(p=1),
           generate_qaoa_circuit(p=2),
           generate_vqe_circuit(layers=4),
           generate_vqe_circuit(layers=8)
       ]

       num_gpus = len(circuits)
       manager = mp.Manager()
       results = manager.dict()

       # Execute in parallel
       processes = []
       for rank, circuit in enumerate(circuits):
           p = mp.Process(
               target=execute_circuit,
               args=(rank, circuit, 20, 64, results)
           )
           p.start()
           processes.append(p)

       for p in processes:
           p.join()

       # Compare results
       for rank, result in results.items():
           print(f"Circuit {rank}: E={result['energy']:.6f}, χ_max={result['final_chi']}")

Strategy 6: Multi-Node Distributed Computing
---------------------------------------------

Scale to multiple compute nodes with InfiniBand or high-speed network.

**Multi-node setup**:

.. code-block:: python

   # multi_node_simulation.py
   from atlas_q.distributed_mps import DistributedMPS, DistributedConfig
   import torch.distributed as dist
   import os

   def main():
       # Initialize distributed backend
       rank = int(os.environ['RANK'])
       world_size = int(os.environ['WORLD_SIZE'])
       master_addr = os.environ['MASTER_ADDR']
       master_port = os.environ['MASTER_PORT']

       dist.init_process_group(
           backend='nccl',
           init_method=f'tcp://{master_addr}:{master_port}',
           world_size=world_size,
           rank=rank
       )

       local_rank = rank % torch.cuda.device_count()
       torch.cuda.set_device(local_rank)

       print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")

       # Configure distributed MPS
       # 4 nodes × 4 GPUs/node = 16 GPUs total
       config = DistributedConfig(
           mode='bond_parallel',
           world_size=world_size,
           backend='nccl'
       )

       # Extremely large MPS: 150 qubits, χ=2048 split across 16 GPUs
       mps = DistributedMPS(
           num_qubits=150,
           bond_dim=2048,  # 128 per GPU
           config=config
       )

       print(f"Rank {rank}: Created distributed MPS")

       # Run simulation
       from atlas_q.tdvp import TDVP
       tdvp = TDVP(hamiltonian=H, mps=mps, dt=0.001, device=f'cuda:{local_rank}')

       for step in range(10000):
           E = tdvp.evolve_step()

           if rank == 0 and step % 100 == 0:
               print(f"[Step {step}] Energy: {E:.10f}")

       dist.barrier()
       dist.destroy_process_group()

   if __name__ == '__main__':
       main()

**SLURM job script for multi-node**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=atlas_q_multi_node
   #SBATCH --nodes=4
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-node=4
   #SBATCH --time=24:00:00

   # Load modules
   module load cuda/12.1
   module load nccl/2.18

   # Set environment
   export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
   export MASTER_PORT=29500

   # Launch distributed training
   srun python multi_node_simulation.py

Troubleshooting
===============

NCCL Initialization Fails
--------------------------

**Problem**: ``RuntimeError: NCCL error: unhandled system error``.

**Solution**: Check network connectivity and NCCL installation.

.. code-block:: bash

   # Test NCCL
   $NCCL_ROOT/bin/nccl-test -g 2  # Test 2 GPUs

   # Check network
   ping -c 5 <other_node>

   # Verify NCCL environment
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL

   # Re-run with verbose logging

Process Group Timeout
---------------------

**Problem**: ``RuntimeError: Timed out initializing process group``.

**Solution**: Increase timeout or check master address/port.

.. code-block:: python

   import torch.distributed as dist
   import datetime

   # Increase timeout to 30 minutes
   dist.init_process_group(
       backend='nccl',
       init_method='env://',
       timeout=datetime.timedelta(minutes=30)
   )

Out of Sync Errors
------------------

**Problem**: GPUs out of sync, incorrect results.

**Solution**: Add explicit barriers and synchronization.

.. code-block:: python

   import torch.distributed as dist

   # Synchronize before and after critical sections
   dist.barrier()
   # ... perform distributed operation ...
   dist.barrier()

   # Ensure CUDA operations complete
   torch.cuda.synchronize()

Memory Imbalance Across GPUs
-----------------------------

**Problem**: Some GPUs run out of memory while others idle.

**Solution**: Balance workload or adjust partitioning.

.. code-block:: python

   # Check memory usage per GPU
   for i in range(torch.cuda.device_count()):
       allocated = torch.cuda.memory_allocated(i) / 1024**3
       total = torch.cuda.get_device_properties(i).total_memory / 1024**3
       print(f"GPU {i}: {allocated:.2f} / {total:.2f} GB")

   # Adjust bond dimension partitioning
   # E.g., assign more work to GPUs with more memory

Summary
=======

Parallel computation strategies for ATLAS-Q:

1. **Bond-parallel MPS**: Split bond dimension across GPUs for χ > 512
2. **Data parallelism**: Run independent simulations for hyperparameter search
3. **Task parallelism**: Parallelize ensemble studies or multi-molecule VQE
4. **Parallel sampling**: Distribute measurement shots across GPUs
5. **Circuit batching**: Execute different circuits on different GPUs
6. **Multi-node**: Scale to 10+ GPUs across multiple compute nodes

Speedup expectations:

- **Bond-parallel MPS**: Near-linear scaling up to communication overhead (~8 GPUs)
- **Data parallelism**: Perfect linear scaling (N GPUs = N× throughput)
- **Parallel sampling**: Linear scaling for large shot counts
- **Multi-node**: 70-90% efficiency with InfiniBand, 40-60% with Ethernet

When to use each strategy:

- **Bond-parallel**: Single large simulation (n > 80, χ > 512)
- **Data parallelism**: Hyperparameter search, ensemble averaging
- **Circuit batching**: Algorithm comparison, benchmarking
- **Multi-node**: Extremely large systems (n > 100, χ > 1024)

See Also
========

- :doc:`handle_large_systems`: Distributed MPS implementation details
- :doc:`integrate_cuquantum`: GPU acceleration with cuQuantum
- :doc:`optimize_performance`: Single-GPU optimization before scaling
- :doc:`../reference/distributed_mps`: Distributed MPS API reference
- :doc:`benchmark_comparison`: Parallel benchmarking strategies
