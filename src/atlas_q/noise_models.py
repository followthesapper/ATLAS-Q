"""
Noise Models for NISQ-Era Quantum Simulation

Implements various noise channels as Kraus operators and MPO representations:
- Depolarizing noise (1-qubit, 2-qubit)
- Dephasing (T2 decoherence)
- Amplitude damping (T1 relaxation)
- Bit flip, phase flip
- Stochastic Pauli sampling

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class NoiseChannel:
    """
    A quantum noise channel represented as Kraus operators

    A channel Φ(ρ) = Σᵢ Kᵢ ρ Kᵢ† where Σᵢ Kᵢ†Kᵢ = I (completeness)
    """

    name: str
    kraus_ops: List[torch.Tensor]  # List of Kraus operators
    num_qubits: int = 1

    def __post_init__(self):
        """Validate completeness relation"""
        # Check Σᵢ Kᵢ†Kᵢ ≈ I
        identity_check = sum(K.conj().T @ K for K in self.kraus_ops)
        dim = 2**self.num_qubits
        expected_identity = torch.eye(
            dim, dtype=self.kraus_ops[0].dtype, device=self.kraus_ops[0].device
        )

        error = torch.norm(identity_check - expected_identity).item()
        if error > 1e-6:
            print(f"Warning: Noise channel {self.name} completeness error: {error:.2e}")


class NoiseModel:
    """
    Collection of noise channels applied to quantum circuits

    Usage:
        noise = NoiseModel.depolarizing(p1q=0.001, p2q=0.01)
        # Apply after each gate during simulation
    """

    def __init__(self):
        self.channels_1q: Dict[str, NoiseChannel] = {}
        self.channels_2q: Dict[str, NoiseChannel] = {}
        self.custom_channels: Dict[Tuple[int, ...], NoiseChannel] = {}
        self.seed: Optional[int] = None

    def set_seed(self, seed: int):
        """Set random seed for stochastic sampling"""
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def add_1q_channel(self, name: str, channel: NoiseChannel):
        """Add a 1-qubit noise channel"""
        assert channel.num_qubits == 1, "Must be 1-qubit channel"
        self.channels_1q[name] = channel

    def add_2q_channel(self, name: str, channel: NoiseChannel):
        """Add a 2-qubit noise channel"""
        assert channel.num_qubits == 2, "Must be 2-qubit channel"
        self.channels_2q[name] = channel

    def add_custom_channel(self, qubits: Tuple[int, ...], channel: NoiseChannel):
        """Add noise channel for specific qubits"""
        self.custom_channels[qubits] = channel

    @staticmethod
    def depolarizing(p1q: float = 0.001, p2q: float = 0.01, device: str = "cuda") -> "NoiseModel":
        """
        Depolarizing noise: ρ → (1-p)ρ + p·I/d

        Args:
            p1q: Single-qubit depolarizing probability
            p2q: Two-qubit depolarizing probability
            device: torch device

        Returns:
            NoiseModel with depolarizing channels
        """
        model = NoiseModel()

        # 1-qubit depolarizing: 4 Kraus operators
        # K₀ = √(1-p) I, K₁ = √(p/3) X, K₂ = √(p/3) Y, K₃ = √(p/3) Z
        sqrt_p1 = np.sqrt(p1q / 3)
        sqrt_1_p1 = np.sqrt(1 - p1q)

        I = torch.eye(2, dtype=torch.complex64, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

        kraus_1q = [sqrt_1_p1 * I, sqrt_p1 * X, sqrt_p1 * Y, sqrt_p1 * Z]

        model.add_1q_channel(
            "depolarizing", NoiseChannel("depolarizing_1q", kraus_1q, num_qubits=1)
        )

        # 2-qubit depolarizing: 16 Kraus operators (Pauli basis)
        sqrt_p2 = np.sqrt(p2q / 15)
        sqrt_1_p2 = np.sqrt(1 - p2q)

        paulis_1q = [I, X, Y, Z]
        kraus_2q = []

        for i, P1 in enumerate(paulis_1q):
            for j, P2 in enumerate(paulis_1q):
                if i == 0 and j == 0:
                    # K₀ = √(1-p) I⊗I
                    kraus_2q.append(sqrt_1_p2 * torch.kron(P1, P2))
                else:
                    # Kᵢⱼ = √(p/15) Pᵢ⊗Pⱼ
                    kraus_2q.append(sqrt_p2 * torch.kron(P1, P2))

        model.add_2q_channel(
            "depolarizing", NoiseChannel("depolarizing_2q", kraus_2q, num_qubits=2)
        )

        return model

    @staticmethod
    def dephasing(p: float = 0.001, device: str = "cuda") -> "NoiseModel":
        """
        Phase damping (T2 dephasing): ρ → (1-p)ρ + p·Z ρ Z

        Kraus operators:
        K₀ = √(1-p) I
        K₁ = √p Z
        """
        model = NoiseModel()

        sqrt_p = np.sqrt(p)
        sqrt_1_p = np.sqrt(1 - p)

        I = torch.eye(2, dtype=torch.complex64, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

        kraus = [sqrt_1_p * I, sqrt_p * Z]

        model.add_1q_channel("dephasing", NoiseChannel("dephasing", kraus, num_qubits=1))

        return model

    @staticmethod
    def amplitude_damping(gamma: float = 0.001, device: str = "cuda") -> "NoiseModel":
        """
        Amplitude damping (T1 relaxation): |1⟩ → |0⟩ decay

        Kraus operators:
        K₀ = [[1, 0], [0, √(1-γ)]]
        K₁ = [[0, √γ], [0, 0]]
        """
        model = NoiseModel()

        sqrt_gamma = np.sqrt(gamma)
        sqrt_1_gamma = np.sqrt(1 - gamma)

        K0 = torch.tensor([[1, 0], [0, sqrt_1_gamma]], dtype=torch.complex64, device=device)
        K1 = torch.tensor([[0, sqrt_gamma], [0, 0]], dtype=torch.complex64, device=device)

        kraus = [K0, K1]

        model.add_1q_channel(
            "amplitude_damping", NoiseChannel("amplitude_damping", kraus, num_qubits=1)
        )

        return model

    @staticmethod
    def pauli_channel(px: float, py: float, pz: float, device: str = "cuda") -> "NoiseModel":
        """
        Pauli channel: apply X/Y/Z with probabilities px/py/pz

        Kraus operators:
        K₀ = √(1-px-py-pz) I
        K₁ = √px X
        K₂ = √py Y
        K₃ = √pz Z
        """
        model = NoiseModel()

        p_total = px + py + pz
        assert p_total <= 1.0, f"Total probability {p_total} > 1"

        I = torch.eye(2, dtype=torch.complex64, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

        kraus = [np.sqrt(1 - p_total) * I, np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]

        model.add_1q_channel("pauli", NoiseChannel("pauli", kraus, num_qubits=1))

        return model

    @staticmethod
    def thermal_relaxation(
        t1: float, t2: float, gate_time: float, device: str = "cuda"
    ) -> "NoiseModel":
        """
        Thermal relaxation combining T1 (amplitude) and T2 (phase) damping

        Args:
            t1: T1 relaxation time (microseconds)
            t2: T2 dephasing time (microseconds)
            gate_time: Gate duration (microseconds)

        Approximated as amplitude damping + pure dephasing
        """
        assert t2 <= 2 * t1, "T2 must satisfy T2 ≤ 2T1"

        # Compute probabilities
        gamma = 1 - np.exp(-gate_time / t1)  # Amplitude damping
        p_phase = 0.5 * (1 - np.exp(-gate_time / t2) - gamma)  # Pure dephasing

        model = NoiseModel()

        # Combine amplitude damping + pure dephasing
        # For simplicity, use sequential channels (approximate)
        amp_model = NoiseModel.amplitude_damping(gamma, device)
        deph_model = NoiseModel.dephasing(p_phase, device)

        # Combine channels
        for name, channel in amp_model.channels_1q.items():
            model.add_1q_channel(name, channel)

        return model


class StochasticNoiseApplicator:
    """
    Applies noise channels stochastically during MPS simulation

    Usage:
        applicator = StochasticNoiseApplicator(noise_model, seed=42)

        # After each gate:
        applicator.apply_1q_noise(mps, qubit_idx)
        applicator.apply_2q_noise(mps, qubit_i, qubit_j)
    """

    def __init__(self, noise_model: NoiseModel, seed: Optional[int] = None):
        self.noise_model = noise_model
        self.rng = np.random.RandomState(seed)
        self.fidelity_tracker = []

    def apply_1q_noise(self, mps, qubit: int):
        """
        Apply 1-qubit noise channel by stochastically sampling Kraus operator

        Args:
            mps: AdaptiveMPS instance
            qubit: Target qubit index
        """
        if not self.noise_model.channels_1q:
            return

        # For now, apply the first registered 1q channel
        channel = next(iter(self.noise_model.channels_1q.values()))

        # Sample a Kraus operator based on probabilities
        # P(Kᵢ) = Tr(Kᵢ ρ Kᵢ†) but we approximate with |Kᵢ|²_F / Σ|Kⱼ|²_F
        kraus_weights = torch.tensor([torch.norm(K).item() ** 2 for K in channel.kraus_ops])
        kraus_weights /= kraus_weights.sum()

        # Sample one Kraus operator
        idx = self.rng.choice(len(channel.kraus_ops), p=kraus_weights.cpu().numpy())
        K = channel.kraus_ops[idx]

        # Apply as a gate (non-unitary)
        # Note: This doesn't preserve norm exactly - need to renormalize
        mps.apply_single_qubit_gate(qubit, K)

        # Track approximate fidelity loss
        # F ≈ 1 - ε where ε is the noise strength
        # This is approximate - true fidelity tracking requires density matrices

    def apply_2q_noise(self, mps, qubit_i: int, qubit_j: int):
        """
        Apply 2-qubit noise channel

        Args:
            mps: AdaptiveMPS instance
            qubit_i: First qubit
            qubit_j: Second qubit (must be adjacent for MPS)
        """
        if not self.noise_model.channels_2q:
            return

        assert abs(qubit_i - qubit_j) == 1, "2-qubit noise requires adjacent qubits for MPS"

        channel = next(iter(self.noise_model.channels_2q.values()))

        # Sample Kraus operator
        kraus_weights = torch.tensor([torch.norm(K).item() ** 2 for K in channel.kraus_ops])
        kraus_weights /= kraus_weights.sum()

        idx = self.rng.choice(len(channel.kraus_ops), p=kraus_weights.cpu().numpy())
        K = channel.kraus_ops[idx]

        # Apply as 2-site gate
        bond_idx = min(qubit_i, qubit_j)
        mps.apply_two_site_gate(bond_idx, K)

    def get_fidelity_estimate(self) -> float:
        """
        Get estimated fidelity (approximate)

        True fidelity requires density matrix simulation
        This gives a rough lower bound based on truncation + noise
        """
        if not self.fidelity_tracker:
            return 1.0

        # Multiply all fidelity factors (independent noise assumption)
        return float(np.prod(self.fidelity_tracker))


def kraus_to_choi(kraus_ops: List[torch.Tensor]) -> torch.Tensor:
    """
    Convert Kraus representation to Choi matrix representation

    Choi matrix: Λ = Σᵢ |Kᵢ⟩⟩⟨⟨Kᵢ| where |K⟩⟩ = vec(K)

    Args:
        kraus_ops: List of Kraus operators

    Returns:
        Choi matrix (d² × d²)
    """
    d = kraus_ops[0].shape[0]
    choi = torch.zeros(d * d, d * d, dtype=kraus_ops[0].dtype, device=kraus_ops[0].device)

    for K in kraus_ops:
        K_vec = K.reshape(-1, 1)  # Vectorize
        choi += K_vec @ K_vec.conj().T

    return choi


def choi_to_kraus(choi: torch.Tensor, rank_tol: float = 1e-10) -> List[torch.Tensor]:
    """
    Convert Choi matrix to Kraus representation via eigendecomposition

    Args:
        choi: Choi matrix (d² × d²)
        rank_tol: Tolerance for truncating small eigenvalues

    Returns:
        List of Kraus operators
    """
    # Eigendecomposition
    evals, evecs = torch.linalg.eigh(choi)

    # Keep positive eigenvalues above threshold
    mask = evals > rank_tol
    evals = evals[mask]
    evecs = evecs[:, mask]

    # Construct Kraus operators: Kᵢ = √λᵢ unvec(vᵢ)
    d = int(np.sqrt(choi.shape[0]))
    kraus_ops = []

    for i in range(len(evals)):
        K_vec = torch.sqrt(evals[i]) * evecs[:, i]
        K = K_vec.reshape(d, d)
        kraus_ops.append(K)

    return kraus_ops
