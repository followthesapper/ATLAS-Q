"""
Variational Quantum Eigensolver (VQE) and QAOA

Implements variational algorithms for optimization and ground state finding:
- VQE for chemistry and physics
- QAOA for combinatorial optimization
- Hardware-efficient ansätze
- Parameter optimization

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. VQE/QAOA optimization will be limited.")

from .adaptive_mps import AdaptiveMPS
from .mpo_ops import MPO, expectation_value


@dataclass
class VQEConfig:
    """Configuration for VQE"""

    ansatz: str = "hardware_efficient"  # 'hardware_efficient', 'uccsd', 'custom'
    n_layers: int = 3
    optimizer: str = "COBYLA"  # 'COBYLA', 'L-BFGS-B', 'Adam'
    max_iter: int = 100
    tol: float = 1e-6
    chi_max: int = 64
    device: str = "cuda"


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz for VQE

    Circuit structure:
    Layer = [Ry(θ) on all qubits] + [CZ on neighboring pairs]

    This mimics real quantum hardware constraints (linear connectivity)
    """

    def __init__(self, n_qubits: int, n_layers: int, device: str = "cuda"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device

        # Parameter count: n_qubits single-qubit rotations per layer
        self.n_params = n_qubits * n_layers

    def apply(self, mps: AdaptiveMPS, params: np.ndarray):
        """
        Apply parameterized circuit to MPS

        Args:
            mps: Initial state (typically |0...0⟩)
            params: Variational parameters (shape: n_params)
        """
        assert len(params) == self.n_params

        param_idx = 0

        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                theta = params[param_idx]
                Ry = self._ry_gate(theta)
                mps.apply_single_qubit_gate(q, Ry)
                param_idx += 1

            # Entangling layer (CZ on neighboring qubits)
            CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64, device=self.device))

            for q in range(0, self.n_qubits - 1, 2):
                # Even pairs (0-1, 2-3, ...)
                mps.apply_two_site_gate(q, CZ)

            for q in range(1, self.n_qubits - 1, 2):
                # Odd pairs (1-2, 3-4, ...)
                mps.apply_two_site_gate(q, CZ)

    def _ry_gate(self, theta: float) -> torch.Tensor:
        """Ry rotation gate"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.complex64, device=self.device)


class VQE:
    """
    Variational Quantum Eigensolver

    Finds ground state of Hamiltonian H by minimizing:
    E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩

    Usage:
        H = MPOBuilder.ising_hamiltonian(n_sites=10, J=1.0, h=0.5)
        vqe = VQE(H, config=VQEConfig(n_layers=3))
        energy, params = vqe.run()
    """

    def __init__(self, hamiltonian: MPO, config: VQEConfig):
        self.H = hamiltonian
        self.config = config

        # Initialize ansatz
        if config.ansatz == "hardware_efficient":
            self.ansatz = HardwareEfficientAnsatz(
                n_qubits=self.H.n_sites, n_layers=config.n_layers, device=config.device
            )
        else:
            raise ValueError(f"Unknown ansatz: {config.ansatz}")

        # Tracking
        self.energies = []
        self.param_history = []
        self.iteration = 0

    def _cost_function(self, params: np.ndarray) -> float:
        """
        Evaluate cost function: E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩

        Args:
            params: Variational parameters

        Returns:
            Energy expectation value (real)
        """
        # Create initial state |0...0⟩
        mps = AdaptiveMPS(
            num_qubits=self.H.n_sites,
            bond_dim=2,
            chi_max_per_bond=self.config.chi_max,
            device=self.config.device,
        )

        # Apply ansatz
        self.ansatz.apply(mps, params)

        # Compute energy
        energy = expectation_value(self.H, mps)

        # Track progress
        self.energies.append(energy.real)
        self.param_history.append(params.copy())
        self.iteration += 1

        if self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: E = {energy.real:.6f}")

        return energy.real

    def run(self, initial_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Run VQE optimization

        Args:
            initial_params: Initial parameter values (random if None)

        Returns:
            optimal_energy: Minimum energy found
            optimal_params: Parameters achieving minimum
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for VQE optimization")

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.randn(self.ansatz.n_params) * 0.1

        # Optimize
        result = minimize(
            self._cost_function,
            initial_params,
            method=self.config.optimizer,
            options={"maxiter": self.config.max_iter, "ftol": self.config.tol},
        )

        print(f"\nVQE converged: E = {result.fun:.6f} after {self.iteration} iterations")

        return result.fun, result.x


class QAOAAnsatz:
    """
    QAOA ansatz for combinatorial optimization

    Circuit structure:
    |ψ(γ, β)⟩ = Πₚ U_B(βₚ) U_C(γₚ) |+⟩^⊗n

    where:
    - U_C(γ) = exp(-i γ H_cost)
    - U_B(β) = exp(-i β H_mixer) with H_mixer = Σᵢ Xᵢ
    """

    def __init__(self, cost_hamiltonian: MPO, n_layers: int, device: str = "cuda"):
        self.H_cost = cost_hamiltonian
        self.n_qubits = cost_hamiltonian.n_sites
        self.n_layers = n_layers
        self.device = device

        # Build mixer Hamiltonian: H_mixer = - Σᵢ Xᵢ
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        ops = [-X] * self.n_qubits
        self.H_mixer = MPO.from_local_ops(ops, device=device)

        # Parameter count: 2 per layer (γ, β)
        self.n_params = 2 * n_layers

    def apply(self, mps: AdaptiveMPS, params: np.ndarray):
        """
        Apply QAOA circuit

        Args:
            mps: Initial state (should be |+⟩^⊗n)
            params: [γ₁, β₁, γ₂, β₂, ..., γₚ, βₚ]
        """
        assert len(params) == self.n_params

        for layer in range(self.n_layers):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]

            # Apply U_C(γ) = exp(-i γ H_cost)
            # Approximate with Trotter: exp(-i γ H) ≈ Πᵢ exp(-i γ hᵢ)
            self._apply_cost_layer(mps, gamma)

            # Apply U_B(β) = exp(-i β H_mixer)
            self._apply_mixer_layer(mps, beta)

    def _apply_cost_layer(self, mps: AdaptiveMPS, gamma: float):
        """Apply cost Hamiltonian evolution"""
        # For simplicity, apply local terms
        # Full implementation needs proper Trotterization of MPO

        # Placeholder: apply local Z rotations
        for q in range(self.n_qubits):
            Rz = self._rz_gate(2 * gamma)  # Factor of 2 from exp(-i γ Z/2)
            mps.apply_single_qubit_gate(q, Rz)

    def _apply_mixer_layer(self, mps: AdaptiveMPS, beta: float):
        """Apply mixer Hamiltonian evolution"""
        # H_mixer = Σᵢ Xᵢ → exp(-i β X) = Rx(2β)
        for q in range(self.n_qubits):
            Rx = self._rx_gate(2 * beta)
            mps.apply_single_qubit_gate(q, Rx)

    def _rz_gate(self, theta: float) -> torch.Tensor:
        """Rz rotation gate"""
        return torch.tensor(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=torch.complex64,
            device=self.device,
        )

    def _rx_gate(self, theta: float) -> torch.Tensor:
        """Rx rotation gate"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64, device=self.device)


class QAOA:
    """
    Quantum Approximate Optimization Algorithm

    For combinatorial optimization problems encoded as Ising Hamiltonians

    Usage:
        # MaxCut on a graph
        H_cost = MPOBuilder.ising_hamiltonian(n_sites=10, J=-1.0, h=0.0)
        qaoa = QAOA(H_cost, n_layers=3)
        energy, params = qaoa.run()
    """

    def __init__(
        self,
        cost_hamiltonian: MPO,
        n_layers: int = 3,
        optimizer: str = "COBYLA",
        device: str = "cuda",
    ):
        self.H_cost = cost_hamiltonian
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.device = device

        self.ansatz = QAOAAnsatz(cost_hamiltonian, n_layers, device)

        self.energies = []
        self.iteration = 0

    def _cost_function(self, params: np.ndarray) -> float:
        """Evaluate QAOA cost function"""
        # Create initial state |+⟩^⊗n
        mps = AdaptiveMPS(
            num_qubits=self.H_cost.n_sites, bond_dim=2, chi_max_per_bond=64, device=self.device
        )

        # Apply Hadamards to get |+⟩^⊗n
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        for q in range(mps.num_qubits):
            mps.apply_single_qubit_gate(q, H)

        # Apply QAOA ansatz
        self.ansatz.apply(mps, params)

        # Compute cost
        energy = expectation_value(self.H_cost, mps)

        self.energies.append(energy.real)
        self.iteration += 1

        if self.iteration % 10 == 0:
            print(f"QAOA Iteration {self.iteration}: Cost = {energy.real:.6f}")

        return energy.real

    def run(self, initial_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Run QAOA optimization"""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for QAOA")

        if initial_params is None:
            # Common initialization: small random values
            initial_params = np.random.randn(self.ansatz.n_params) * 0.1

        result = minimize(
            self._cost_function, initial_params, method=self.optimizer, options={"maxiter": 200}
        )

        print(f"\nQAOA converged: Cost = {result.fun:.6f}")

        return result.fun, result.x


# Chemistry-specific utilities


def build_molecular_hamiltonian(
    h1: np.ndarray, h2: np.ndarray, mapping: str = "jordan_wigner", device: str = "cuda"
) -> MPO:
    """
    Build molecular Hamiltonian MPO from 1- and 2-electron integrals

    Args:
        h1: One-electron integrals [n_orb, n_orb]
        h2: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]
        mapping: 'jordan_wigner' or 'bravyi_kitaev'
        device: torch device

    Returns:
        MPO representation of electronic Hamiltonian
    """
    # Placeholder - proper implementation requires:
    # 1. Second quantization operators
    # 2. Fermion-to-qubit mapping
    # 3. Pauli string collection
    # 4. MPO compression

    n_qubits = h1.shape[0] * 2  # Spin-orbitals

    # For now, return identity
    return MPO.identity(n_qubits, device=device)
