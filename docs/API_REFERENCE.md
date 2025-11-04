# ATLAS-Q API Reference

**Version 0.6.1** | **Comprehensive API Documentation**

This reference provides complete documentation of all functions, classes, and objects in ATLAS-Q. For conceptual guides and tutorials, see the [Complete Guide](COMPLETE_GUIDE.md).

---

## Table of Contents

### Import Guide
- [How to Import](#import-guide)
- [Module Overview](#module-overview)

### Core Modules
- [`adaptive_mps`](#adaptive_mps---adaptive-matrix-product-states) - Adaptive Matrix Product States
- [`mps_pytorch`](#mps_pytorch---basic-mps-implementation) - Basic MPS Implementation
- [`mpo_ops`](#mpo_ops---matrix-product-operators) - Matrix Product Operators & Hamiltonians
- [`tdvp`](#tdvp---time-evolution) - Time-Dependent Variational Principle

### Variational Algorithms
- [`vqe_qaoa`](#vqe_qaoa---variational-algorithms) - VQE & QAOA
- [`ansatz_uccsd`](#ansatz_uccsd---molecular-chemistry) - UCCSD Molecular Ansatz

### Advanced Features
- [`noise_models`](#noise_models---nisq-simulation) - NISQ Noise Simulation
- [`stabilizer_backend`](#stabilizer_backend---clifford-circuits) - Stabilizer/Clifford Fast Path
- [`circuit_cutting`](#circuit_cutting---large-circuits) - Circuit Partitioning
- [`peps`](#peps---2d-tensor-networks) - 2D Tensor Networks (PEPS)
- [`distributed_mps`](#distributed_mps---multi-gpu) - Multi-GPU Distributed Simulation

### Optional Acceleration
- [`cuquantum_backend`](#cuquantum_backend---nvidia-acceleration) - cuQuantum Integration
- [`planar_2d`](#planar_2d---2d-circuits) - 2D Circuit Mapping

---

## Import Guide

### Recommended Pattern (v0.6.1+)

```python
# Import modules
from atlas_q import mpo_ops, tdvp, vqe_qaoa
from atlas_q.adaptive_mps import AdaptiveMPS

# Import specific classes
from atlas_q.mpo_ops import MPOBuilder, MPO
from atlas_q.tdvp import TDVP1Site, run_tdvp, TDVPConfig
from atlas_q.vqe_qaoa import VQE, VQEConfig
```

### Legacy Pattern (v0.5.0, still supported)

```python
import atlas_q

# Returns dictionaries
mpo_dict = atlas_q.get_mpo_ops()
tdvp_dict = atlas_q.get_tdvp()

MPOBuilder = mpo_dict['MPOBuilder']
TDVP1Site = tdvp_dict['TDVP1Site']
```

---

## Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **adaptive_mps** | Adaptive bond dimension MPS | `AdaptiveMPS`, `DTypePolicy` |
| **mps_pytorch** | Basic MPS implementation | `MatrixProductStatePyTorch` |
| **mpo_ops** | Operators & Hamiltonians | `MPO`, `MPOBuilder` |
| **tdvp** | Time evolution | `TDVP1Site`, `TDVP2Site`, `run_tdvp` |
| **vqe_qaoa** | Variational algorithms | `VQE`, `QAOA` |
| **ansatz_uccsd** | Molecular chemistry | `UCCSDAnsatz` |
| **noise_models** | NISQ noise | `NoiseModel`, `NoiseChannel` |
| **stabilizer_backend** | Clifford fast path | `StabilizerSimulator` |
| **circuit_cutting** | Circuit partitioning | `CircuitCutter` |
| **peps** | 2D tensor networks | `PEPS`, `PatchPEPS` |
| **distributed_mps** | Multi-GPU | `DistributedMPS` |
| **cuquantum_backend** | NVIDIA acceleration | `CuQuantumBackend` |

---

## adaptive_mps - Adaptive Matrix Product States

Adaptive bond dimension MPS with automatic truncation and memory management.

### Classes

#### `AdaptiveMPS`

Matrix Product State with adaptive bond dimensions based on entanglement.

**Constructor:**
```python
AdaptiveMPS(
 num_qubits: int,
 bond_dim: int = 8,
 *,
 eps_bond: float = 1e-6,
 chi_max_per_bond: Union[List[int], int] = 256,
 budget_global_mb: Optional[float] = None,
 dtype_policy: DTypePolicy = DTypePolicy(),
 device: str = 'cuda',
 dtype: Optional[torch.dtype] = None
)
```

**Parameters:**
- `num_qubits` (int): Number of qubits in the system
- `bond_dim` (int, default=8): Initial bond dimension
- `eps_bond` (float, default=1e-6): Truncation tolerance for SVD
- `chi_max_per_bond` (int or List[int], default=256): Maximum bond dimension(s)
- `budget_global_mb` (float, optional): Global memory budget in MB
- `dtype_policy` (DTypePolicy, default=DTypePolicy()): Mixed precision policy
- `device` (str, default='cuda'): Device ('cuda' or 'cpu')
- `dtype` (torch.dtype, optional): Override default dtype (complex64 or complex128)

**Attributes:**
- `num_qubits` (int): Number of qubits
- `tensors` (List[torch.Tensor]): MPS tensors `[χ_left, 2, χ_right]`
- `chi_max_per_bond` (List[int]): Maximum bond dimensions
- `device` (str): Current device
- `dtype` (torch.dtype): Data type of tensors

**Methods:**

##### `apply_single_qubit_gate(qubit: int, gate: torch.Tensor)`
Apply a single-qubit gate.

**Parameters:**
- `qubit` (int): Target qubit index
- `gate` (torch.Tensor): 2×2 unitary gate matrix

**Example:**
```python
import torch
mps = AdaptiveMPS(10, bond_dim=8, device='cuda')

# Hadamard gate
H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
mps.apply_single_qubit_gate(0, H.to('cuda'))
```

##### `apply_two_site_gate(qubit: int, gate: torch.Tensor)`
Apply a two-qubit gate to adjacent qubits.

**Parameters:**
- `qubit` (int): First qubit index (applies to qubit and qubit+1)
- `gate` (torch.Tensor): 4×4 unitary gate matrix

**Example:**
```python
# CNOT gate
CNOT = torch.tensor([
 [1,0,0,0],
 [0,1,0,0],
 [0,0,0,1],
 [0,0,1,0]
], dtype=torch.complex64).reshape(4,4).to('cuda')

mps.apply_two_site_gate(0, CNOT)
```

##### `to_left_canonical() -> None`
Convert to left-canonical form (all tensors left-orthogonal).

##### `to_right_canonical() -> None`
Convert to right-canonical form (all tensors right-orthogonal).

##### `to_statevector() -> torch.Tensor`
Convert MPS to full statevector.

**Returns:**
- `torch.Tensor`: Complex vector of shape `[2^n]`

**Warning:** Exponential memory usage! Only use for small systems (<20 qubits).

##### `memory_usage() -> float`
Get total memory usage in bytes.

**Returns:**
- `float`: Memory usage in bytes

##### `stats_summary() -> Dict`
Get MPS statistics.

**Returns:**
- `dict`: Statistics including `max_chi`, `total_params`, `compression_ratio`

**Example:**
```python
stats = mps.stats_summary()
print(f"Max bond dim: {stats['max_chi']}")
print(f"Parameters: {stats['total_params']}")
print(f"Compression: {stats['compression_ratio']:.2e}×")
```

---

#### `DTypePolicy`

Mixed precision policy for adaptive MPS.

**Constructor:**
```python
DTypePolicy(
 default: torch.dtype = torch.complex64,
 promote_if_cond_gt: float = 1e6
)
```

**Parameters:**
- `default` (torch.dtype): Default precision
- `promote_if_cond_gt` (float): Promote to complex128 if condition number exceeds this

---

### Functions

#### `robust_svd(A: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Robust SVD with automatic fallback.

**Parameters:**
- `A` (torch.Tensor): Input matrix
- `**kwargs`: Additional arguments for torch.linalg.svd

**Returns:**
- `U, S, Vh` (Tuple[torch.Tensor, ...]): SVD decomposition

---

## mpo_ops - Matrix Product Operators

Operators, Hamiltonians, and expectation values.

### Classes

#### `MPO`

Matrix Product Operator representation.

**Constructor:**
```python
MPO(tensors: List[torch.Tensor], n_sites: int)
```

**Parameters:**
- `tensors` (List[torch.Tensor]): MPO tensors `[χ_L, d, d, χ_R]`
- `n_sites` (int): Number of sites

**Attributes:**
- `tensors` (List[torch.Tensor]): MPO tensors
- `n_sites` (int): Number of sites

**Class Methods:**

##### `MPO.identity(n_sites: int, device: str = 'cuda', dtype=torch.complex64) -> MPO`
Create identity MPO.

**Parameters:**
- `n_sites` (int): Number of sites
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: Identity operator

##### `MPO.from_local_ops(ops: List[torch.Tensor], device: str = 'cuda') -> MPO`
Create MPO from local operators.

**Parameters:**
- `ops` (List[torch.Tensor]): List of 2×2 operators (one per site)
- `device` (str): Device

**Returns:**
- `MPO`: MPO representation

**Example:**
```python
import torch
from atlas_q.mpo_ops import MPO

# Pauli Z on every site
Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device='cuda')
mpo = MPO.from_local_ops([Z] * 10, device='cuda')
```

---

#### `MPOBuilder`

Helper class for building common Hamiltonians.

**Static Methods:**

##### `MPOBuilder.ising_hamiltonian(n_sites: int, J: float = 1.0, h: float = 0.5, device: str = 'cuda', dtype=torch.complex64) -> MPO`

Build transverse-field Ising Hamiltonian:
H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ

**Parameters:**
- `n_sites` (int): Number of spins
- `J` (float, default=1.0): Coupling strength
- `h` (float, default=0.5): Transverse field strength
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: Ising Hamiltonian

**Example:**
```python
from atlas_q.mpo_ops import MPOBuilder

H = MPOBuilder.ising_hamiltonian(
 n_sites=10,
 J=1.0,
 h=0.5,
 device='cuda'
)
```

##### `MPOBuilder.heisenberg_hamiltonian(n_sites: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0, device: str = 'cuda', dtype=torch.complex64) -> MPO`

Build Heisenberg Hamiltonian:
H = Σᵢ (Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵧ ZᵢZᵢ₊₁)

**Parameters:**
- `n_sites` (int): Number of spins
- `Jx`, `Jy`, `Jz` (float): Coupling strengths
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: Heisenberg Hamiltonian

##### `MPOBuilder.maxcut_hamiltonian(edges: List[Tuple[int, int]], weights: Optional[List[float]] = None, n_sites: Optional[int] = None, device: str = 'cuda', dtype=torch.complex64) -> MPO`

Build MaxCut QAOA Hamiltonian:
H = Σ_{(i,j)∈E} w_{ij} (1 - ZᵢZ) / 2

**Parameters:**
- `edges` (List[Tuple[int, int]]): Graph edges as (i, j) tuples
- `weights` (List[float], optional): Edge weights (default: all 1.0)
- `n_sites` (int, optional): Number of nodes (inferred if not provided)
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: MaxCut Hamiltonian

**Example:**
```python
# Triangle graph
edges = [(0,1), (1,2), (0,2)]
H = MPOBuilder.maxcut_hamiltonian(edges, device='cuda')
```

##### `MPOBuilder.molecular_hamiltonian_from_specs(molecule: str = 'H2', basis: str = 'sto-3g', charge: int = 0, spin: int = 0, mapping: str = 'jordan_wigner', device: str = 'cuda', dtype=torch.complex64) -> MPO`

Build molecular electronic Hamiltonian using PySCF.

**Parameters:**
- `molecule` (str): Molecule name ('H2', 'LiH', 'H2O') or geometry string
- `basis` (str): Gaussian basis set (sto-3g, 6-31g, cc-pvdz, etc.)
- `charge` (int): Total molecular charge
- `spin` (int): Spin multiplicity (2S)
- `mapping` (str): Fermion-to-qubit mapping ('jordan_wigner')
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: Molecular Hamiltonian

**Example:**
```python
# H2 molecule
H = MPOBuilder.molecular_hamiltonian_from_specs(
 molecule='H2',
 basis='sto-3g',
 device='cuda'
)

# Custom geometry
H_custom = MPOBuilder.molecular_hamiltonian_from_specs(
 molecule='H 0 0 0; H 0 0 0.74',
 basis='6-31g',
 device='cuda'
)
```

---

### Functions

#### `expectation_value(mpo: MPO, mps, use_gpu_optimized: bool = True) -> complex`

Compute ψ|O|ψ expectation value.

**Parameters:**
- `mpo` (MPO): Operator
- `mps` (AdaptiveMPS or MatrixProductStatePyTorch): State
- `use_gpu_optimized` (bool): Use GPU-optimized contractions

**Returns:**
- `complex`: Expectation value

**Example:**
```python
from atlas_q import mpo_ops
from atlas_q.adaptive_mps import AdaptiveMPS

H = mpo_ops.MPOBuilder.ising_hamiltonian(10, J=1.0, h=0.5)
mps = AdaptiveMPS(10, bond_dim=64, device='cuda')

energy = mpo_ops.expectation_value(H, mps)
print(f"Energy: {energy.real:.6f}")
```

#### `apply_mpo_to_mps(mpo: MPO, mps, chi_max: int = 128, eps: float = 1e-8) -> AdaptiveMPS`

Apply MPO to MPS: |ψ' = O |ψ

**Parameters:**
- `mpo` (MPO): Operator to apply
- `mps`: Input state
- `chi_max` (int): Maximum bond dimension after compression
- `eps` (float): Truncation tolerance

**Returns:**
- `AdaptiveMPS`: Resulting state

#### `correlation_function(op1: torch.Tensor, site1: int, op2: torch.Tensor, site2: int, mps) -> complex`

Compute two-point correlation: ψ| O₁(i) O₂(j) |ψ

**Parameters:**
- `op1`, `op2` (torch.Tensor): Operators (2×2)
- `site1`, `site2` (int): Sites
- `mps`: State

**Returns:**
- `complex`: Correlation value

#### `pauli_string_to_mpo(pauli_string: str, device: str = 'cuda', dtype=torch.complex128) -> MPO`

Convert Pauli string to MPO.

**Parameters:**
- `pauli_string` (str): String like "IXYZ" for I⊗X⊗Y⊗Z
- `device` (str): Device
- `dtype`: Data type

**Returns:**
- `MPO`: Pauli operator

**Example:**
```python
from atlas_q.mpo_ops import pauli_string_to_mpo

# Create Z⊗Z⊗I⊗I operator
mpo = pauli_string_to_mpo("ZZII", device='cuda')
```

#### `apply_pauli_exp_to_mps(mps, pauli_string: str, coeff: complex, theta: float, chi_max: int = 128) -> None`

Apply exp(i * theta * coeff * P) to MPS in-place, where P is a Pauli string.

**Parameters:**
- `mps` (AdaptiveMPS): MPS to modify in-place
- `pauli_string` (str): Pauli string like "IXYZ"
- `coeff` (complex): Complex coefficient from operator (e.g., UCCSD generator)
- `theta` (float): Variational parameter
- `chi_max` (int, default=128): Maximum bond dimension after compression

**Notes:**
- Uses exact formula: exp(i * α * P) = cos(α) I + i sin(α) P
- Avoids exponential memory usage by applying rotation formula directly
- Critical for memory-efficient UCCSD implementation

**Example:**
```python
from atlas_q.adaptive_mps import AdaptiveMPS
from atlas_q.mpo_ops import apply_pauli_exp_to_mps

# Initialize MPS
mps = AdaptiveMPS(4, bond_dim=8, device='cuda')

# Apply Pauli exponential (e.g., for UCCSD)
apply_pauli_exp_to_mps(mps, "XYZI", coeff=0.5+0.2j, theta=1.0, chi_max=64)
```

---

## tdvp - Time Evolution

Time-Dependent Variational Principle for Hamiltonian dynamics.

### Classes

#### `TDVPConfig`

Configuration for TDVP evolution.

**Constructor:**
```python
TDVPConfig(
 dt: float = 0.01,
 t_final: float = 10.0,
 order: int = 2,
 chi_max: int = 128,
 eps_bond: float = 1e-8,
 adaptive_dt: bool = False,
 dt_min: float = 1e-5,
 dt_max: float = 0.1,
 error_tol: float = 1e-6,
 use_gpu_optimized: bool = True
)
```

**Parameters:**
- `dt` (float): Time step
- `t_final` (float): Final time
- `order` (int): Trotter order (1 or 2)
- `chi_max` (int): Maximum bond dimension
- `eps_bond` (float): Truncation tolerance
- `adaptive_dt` (bool): Use adaptive time stepping
- `dt_min`, `dt_max` (float): Min/max time steps
- `error_tol` (float): Error tolerance
- `use_gpu_optimized` (bool): Use GPU-optimized operations

---

#### `TDVP1Site`

1-site TDVP evolver (conserves bond dimension).

**Constructor:**
```python
TDVP1Site(hamiltonian: MPO, mps: AdaptiveMPS, dt: float)
```

**Parameters:**
- `hamiltonian` (MPO): Hamiltonian operator
- `mps` (AdaptiveMPS): Initial state
- `dt` (float): Time step

**Methods:**

##### `sweep_forward(dt: float) -> None`
Perform forward sweep.

##### `sweep_backward(dt: float) -> None`
Perform backward sweep.

##### `run() -> Tuple[List[float], List[complex]]`
Run evolution until completion.

**Returns:**
- `times` (List[float]): Time points
- `energies` (List[complex]): Energy at each time

**Example:**
```python
from atlas_q import mpo_ops, tdvp
from atlas_q.tdvp import AdaptiveMPS, TDVP1Site

H = mpo_ops.MPOBuilder.ising_hamiltonian(20, J=1.0, h=0.5)
mps = AdaptiveMPS(20, bond_dim=64, device='cuda')

evolver = TDVP1Site(H, mps, dt=0.01)

# Manual evolution
for step in range(10):
 evolver.sweep_forward(0.01)
 evolver.sweep_backward(0.01)
```

---

#### `TDVP2Site`

2-site TDVP evolver (adaptive bond dimension).

**Constructor:**
```python
TDVP2Site(hamiltonian: MPO, mps: AdaptiveMPS, dt: float)
```

**Parameters:**
- Same as `TDVP1Site`

**Methods:**
- Same as `TDVP1Site`

---

### Functions

#### `run_tdvp(hamiltonian: MPO, initial_mps: AdaptiveMPS, config: Optional[TDVPConfig] = None) -> Tuple[AdaptiveMPS, List[float], List[complex]]`

High-level TDVP evolution function.

**Parameters:**
- `hamiltonian` (MPO): Hamiltonian
- `initial_mps` (AdaptiveMPS): Initial state
- `config` (TDVPConfig, optional): Configuration

**Returns:**
- `final_mps` (AdaptiveMPS): Final state
- `times` (List[float]): Time points
- `energies` (List[complex]): Energies

**Example:**
```python
from atlas_q import mpo_ops, tdvp
from atlas_q.tdvp import AdaptiveMPS, run_tdvp, TDVPConfig

H = mpo_ops.MPOBuilder.ising_hamiltonian(20, 1.0, 0.5, device='cuda')
mps = AdaptiveMPS(20, bond_dim=64, device='cuda')

config = TDVPConfig(dt=0.01, t_final=1.0, chi_max=128)
final_mps, times, energies = run_tdvp(H, mps, config)

print(f"Final energy: {energies[-1]:.6f}")
```

---

## vqe_qaoa - Variational Algorithms

VQE (Variational Quantum Eigensolver) and QAOA for optimization.

### Classes

#### `VQEConfig`

Configuration for VQE.

**Constructor:**
```python
VQEConfig(
 ansatz: str = "hardware_efficient",
 n_layers: int = 3,
 optimizer: str = "COBYLA",
 max_iter: int = 100,
 tol: float = 1e-6,
 chi_max: int = 64,
 device: str = "cuda",
 dtype: torch.dtype = torch.complex128
)
```

**Parameters:**
- `ansatz` (str): Ansatz type ('hardware_efficient', 'custom')
- `n_layers` (int): Number of layers
- `optimizer` (str): Classical optimizer ('COBYLA', 'L-BFGS-B')
- `max_iter` (int): Maximum iterations
- `tol` (float): Convergence tolerance
- `chi_max` (int): Maximum bond dimension for MPS
- `device` (str): Device
- `dtype` (torch.dtype): Data type

---

#### `VQE`

Variational Quantum Eigensolver.

**Constructor:**
```python
VQE(
 hamiltonian: MPO,
 config: VQEConfig,
 custom_ansatz=None
)
```

**Parameters:**
- `hamiltonian` (MPO): Hamiltonian to minimize
- `config` (VQEConfig): Configuration
- `custom_ansatz` (optional): Custom ansatz (e.g., UCCSDAnsatz)

**Attributes:**
- `energies` (List[float]): Energy history
- `iteration` (int): Current iteration

**Methods:**

##### `run(initial_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]`

Run VQE optimization.

**Parameters:**
- `initial_params` (np.ndarray, optional): Initial parameters

**Returns:**
- `optimal_energy` (float): Minimum energy found
- `optimal_params` (np.ndarray): Optimal parameters

**Example:**
```python
from atlas_q import mpo_ops, vqe_qaoa
from atlas_q.ansatz_uccsd import UCCSDAnsatz

# Build molecular Hamiltonian
H = mpo_ops.MPOBuilder.molecular_hamiltonian_from_specs(
 molecule='H2',
 basis='sto-3g',
 device='cuda'
)

# Create UCCSD ansatz
ansatz = UCCSDAnsatz(molecule='H2', basis='sto-3g', device='cuda')

# Run VQE
config = vqe_qaoa.VQEConfig(max_iter=200, chi_max=128)
vqe = vqe_qaoa.VQE(H, config, custom_ansatz=ansatz)

energy, params = vqe.run()
print(f"Ground state energy: {energy:.6f} Ha")
```

---

#### `HardwareEfficientAnsatz`

Hardware-efficient variational ansatz.

**Constructor:**
```python
HardwareEfficientAnsatz(
 n_qubits: int,
 n_layers: int,
 device: str = 'cuda',
 dtype: torch.dtype = torch.complex128
)
```

**Parameters:**
- `n_qubits` (int): Number of qubits
- `n_layers` (int): Number of layers
- `device` (str): Device
- `dtype`: Data type

**Attributes:**
- `n_params` (int): Number of parameters

**Methods:**

##### `apply(mps: AdaptiveMPS, params: np.ndarray) -> None`

Apply ansatz to MPS.

**Parameters:**
- `mps` (AdaptiveMPS): State to modify (in-place)
- `params` (np.ndarray): Variational parameters

---

#### `QAOA`

Quantum Approximate Optimization Algorithm.

**Constructor:**
```python
QAOA(
 cost_hamiltonian: MPO,
 n_layers: int = 3,
 optimizer: str = "COBYLA",
 device: str = "cuda",
 dtype: torch.dtype = torch.complex128
)
```

**Parameters:**
- `cost_hamiltonian` (MPO): Problem Hamiltonian
- `n_layers` (int): QAOA depth (p parameter)
- `optimizer` (str): Classical optimizer
- `device` (str): Device
- `dtype`: Data type

**Methods:**

##### `run(initial_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]`

Run QAOA optimization.

**Returns:**
- `optimal_cost` (float): Optimal cost value
- `optimal_params` (np.ndarray): Optimal (γ, β) parameters

**Example:**
```python
from atlas_q import mpo_ops, vqe_qaoa

# MaxCut on triangle graph
edges = [(0,1), (1,2), (0,2)]
H_cost = mpo_ops.MPOBuilder.maxcut_hamiltonian(edges, device='cuda')

# Run QAOA
qaoa = vqe_qaoa.QAOA(H_cost, n_layers=3, device='cuda')
cost, params = qaoa.run()

print(f"Optimal cost: {cost:.6f}")
```

---

## ansatz_uccsd - Molecular Chemistry

UCCSD (Unitary Coupled-Cluster Singles and Doubles) ansatz for molecular VQE.

### Classes

#### `UCCSDAnsatz`

Chemistry-aware UCCSD ansatz.

**Constructor:**
```python
UCCSDAnsatz(
 molecule: str = 'H2',
 basis: str = 'sto-3g',
 device: str = 'cuda',
 dtype: torch.dtype = torch.complex128
)
```

**Parameters:**
- `molecule` (str): Molecule ('H2', 'LiH', 'H2O', 'BeH2') or geometry string
- `basis` (str): Gaussian basis set
- `device` (str): Device
- `dtype`: Data type

**Attributes:**
- `n_qubits` (int): Number of qubits (2 × n_orbitals)
- `n_parameters` (int): Number of UCCSD parameters
- `n_electrons` (int): Number of electrons
- `hf_energy` (float): Hartree-Fock energy (Ha)
- `hf_state` (np.ndarray): HF occupation vector

**Methods:**

##### `prepare_hf_state(chi_max: int = 64) -> AdaptiveMPS`

Create MPS initialized to Hartree-Fock state.

**Parameters:**
- `chi_max` (int): Maximum bond dimension

**Returns:**
- `AdaptiveMPS`: HF reference state

##### `apply(mps: AdaptiveMPS, params: np.ndarray, chi_max: int = None) -> None`

Apply UCCSD transformation to MPS.

**Parameters:**
- `mps` (AdaptiveMPS): State (modified in-place)
- `params` (np.ndarray): UCCSD amplitudes
- `chi_max` (int, optional): Max bond dimension

**Example:**
```python
from atlas_q.ansatz_uccsd import UCCSDAnsatz
import torch

ansatz = UCCSDAnsatz(molecule='H2', basis='sto-3g', device='cuda')

print(f"Qubits: {ansatz.n_qubits}")
print(f"Parameters: {ansatz.n_parameters}")
print(f"HF energy: {ansatz.hf_energy:.6f} Ha")

# Use with VQE (see VQE example above)
```

---

### Functions

#### `build_uccsd_ansatz(molecule: str, basis: str, ...) -> Dict`

Build UCCSD ansatz data structure.

**Returns:**
- `dict`: Contains `n_qubits`, `n_electrons`, `pauli_strings`, `hf_state`, `hf_energy`

---

## noise_models - NISQ Simulation

Realistic noise simulation for near-term quantum devices.

### Classes

#### `NoiseModel`

NISQ noise model with multiple channels.

**Constructor:**
```python
NoiseModel(device: str = 'cuda')
```

**Methods:**

##### `add_quantum_error(channel: NoiseChannel, gate_types: List[str]) -> None`

Add noise channel to specific gate types.

**Parameters:**
- `channel` (NoiseChannel): Noise channel
- `gate_types` (List[str]): Gate types ('X', 'CNOT', etc.)

##### `apply_noise_to_gate(gate_name: str, mps, qubit: int) -> None`

Apply noise after gate operation.

---

#### `NoiseChannel`

Individual noise channel.

**Static Methods:**

##### `NoiseChannel.depolarizing(p: float, device: str = 'cuda') -> NoiseChannel`

Create depolarizing channel.

**Parameters:**
- `p` (float): Depolarizing probability (0 to 1)
- `device` (str): Device

**Returns:**
- `NoiseChannel`: Depolarizing noise

##### `NoiseChannel.amplitude_damping(gamma: float, device: str = 'cuda') -> NoiseChannel`

Create amplitude damping (T1 decay) channel.

**Parameters:**
- `gamma` (float): Damping rate
- `device` (str): Device

##### `NoiseChannel.dephasing(lam: float, device: str = 'cuda') -> NoiseChannel`

Create dephasing (T2 decay) channel.

**Parameters:**
- `lam` (float): Dephasing rate
- `device` (str): Device

**Example:**
```python
from atlas_q.noise_models import NoiseModel, NoiseChannel
from atlas_q.adaptive_mps import AdaptiveMPS

# Create noise model
noise = NoiseModel(device='cuda')
noise.add_quantum_error(
 NoiseChannel.depolarizing(0.01), # 1% depolarizing
 ['X', 'H', 'RZ']
)
noise.add_quantum_error(
 NoiseChannel.depolarizing(0.02), # 2% for two-qubit gates
 ['CNOT', 'CZ']
)

# Apply to circuit (noise applied automatically)
mps = AdaptiveMPS(10, device='cuda')
# ... apply gates ...
noise.apply_noise_to_gate('X', mps, qubit=0)
```

---

## stabilizer_backend - Clifford Circuits

Fast simulation of Clifford circuits using Gottesman-Knill theorem.

### Classes

#### `StabilizerSimulator`

O(n²) Clifford circuit simulator.

**Constructor:**
```python
StabilizerSimulator(n_qubits: int, device: str = 'cuda')
```

**Methods:**

##### `apply_gate(gate_name: str, qubits: List[int]) -> None`

Apply Clifford gate.

**Parameters:**
- `gate_name` (str): 'H', 'S', 'CNOT', 'CZ', etc.
- `qubits` (List[int]): Target qubit(s)

##### `measure(qubit: int) -> int`

Measure qubit and collapse state.

**Returns:**
- `int`: Measurement outcome (0 or 1)

**Example:**
```python
from atlas_q.stabilizer_backend import StabilizerSimulator

sim = StabilizerSimulator(10, device='cuda')

# Bell pair circuit
sim.apply_gate('H', [0])
sim.apply_gate('CNOT', [0, 1])

# Measure
result = sim.measure(0)
print(f"Measurement: {result}")
```

---

#### `HybridSimulator`

Automatic switching between Stabilizer and MPS.

**Constructor:**
```python
HybridSimulator(n_qubits: int, device: str = 'cuda')
```

**Methods:**
- Same as `StabilizerSimulator`, but auto-switches to MPS when non-Clifford gates are applied

---

## circuit_cutting - Large Circuits

Partition large circuits into smaller subcircuits.

### Classes

#### `CircuitCutter`

Circuit partitioning tool.

**Constructor:**
```python
CircuitCutter(config: CuttingConfig)
```

**Methods:**

##### `cut_circuit(circuit, coupling_graph: CouplingGraph) -> List[CircuitPartition]`

Partition circuit.

**Parameters:**
- `circuit`: Circuit to cut
- `coupling_graph` (CouplingGraph): Hardware connectivity

**Returns:**
- `List[CircuitPartition]`: List of subcircuits

---

#### `CuttingConfig`

Configuration for circuit cutting.

**Constructor:**
```python
CuttingConfig(
 max_subcircuit_size: int = 10,
 max_cuts: int = 5,
 method: str = 'min_cut'
)
```

**Example:**
```python
from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig, CouplingGraph

config = CuttingConfig(max_subcircuit_size=10, max_cuts=3)
cutter = CircuitCutter(config)

# Cut large circuit
partitions = cutter.cut_circuit(large_circuit, coupling_graph)
print(f"Cut into {len(partitions)} subcircuits")
```

---

## peps - 2D Tensor Networks

PEPS (Projected Entangled Pair States) for 2D quantum systems.

### Classes

#### `PEPS`

2D tensor network.

**Constructor:**
```python
PEPS(
 rows: int,
 cols: int,
 bond_dim: int = 8,
 device: str = 'cuda'
)
```

**Methods:**

##### `apply_single_site_gate(row: int, col: int, gate: torch.Tensor) -> None`

Apply gate to site (row, col).

##### `apply_two_site_gate(site1: Tuple[int, int], site2: Tuple[int, int], gate: torch.Tensor) -> None`

Apply two-site gate.

**Example:**
```python
from atlas_q.peps import PEPS
import torch

# 4×4 grid
peps = PEPS(rows=4, cols=4, bond_dim=8, device='cuda')

H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
peps.apply_single_site_gate(0, 0, H.to('cuda'))
```

---

## distributed_mps - Multi-GPU

Distributed MPS simulation across multiple GPUs.

### Classes

#### `DistributedMPS`

Multi-GPU MPS.

**Constructor:**
```python
DistributedMPS(
 num_qubits: int,
 bond_dim: int,
 num_gpus: int,
 config: DistributedConfig
)
```

**Methods:**
- Same gate application methods as `AdaptiveMPS`

**Example:**
```python
from atlas_q.distributed_mps import DistributedMPS, DistributedConfig

config = DistributedConfig(num_gpus=4)
mps = DistributedMPS(100, bond_dim=64, num_gpus=4, config=config)

# Gates distributed automatically across GPUs
```

---

## cuquantum_backend - NVIDIA Acceleration

Optional cuQuantum integration for NVIDIA GPUs.

### Classes

#### `CuQuantumBackend`

cuQuantum acceleration backend.

**Constructor:**
```python
CuQuantumBackend(config: CuQuantumConfig)
```

---

### Functions

#### `is_cuquantum_available() -> bool`

Check if cuQuantum is available.

**Returns:**
- `bool`: True if cuquantum-python is installed

#### `get_cuquantum_version() -> Optional[str]`

Get cuQuantum version.

**Returns:**
- `str`: Version string or None

**Example:**
```python
from atlas_q.cuquantum_backend import is_cuquantum_available

if is_cuquantum_available():
 print("cuQuantum acceleration enabled")
else:
 print("Using PyTorch fallback")
```

---

## planar_2d - 2D Circuits

Mapping for 2D planar circuit architectures.

### Classes

#### `Planar2DCircuit`

2D circuit mapper.

**Constructor:**
```python
Planar2DCircuit(layout: Layout2D, config: MappingConfig)
```

**Methods:**

##### `map_to_hardware(logical_circuit) -> Tuple[physical_circuit, SWAP_count]`

Map logical qubits to physical 2D grid.

---

## See Also

- **[Complete Guide](COMPLETE_GUIDE.md)** - Tutorials and conceptual guides
- **[Feature Status](FEATURE_STATUS.md)** - What's implemented and tested
- **[Changelog](CHANGELOG.md)** - Version history
- **[Examples](../examples/)** - Jupyter notebooks and scripts

---

**Last Updated:** October 2025 | **Version:** 0.6.1
