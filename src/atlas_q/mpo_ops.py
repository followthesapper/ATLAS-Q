"""
Matrix Product Operator (MPO) Operations

MPOs represent operators on quantum states in tensor network form:
- Hamiltonian evolution
- Observable expectation values
- Noise channels
- Time evolution

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass

# GPU-optimized operations (if available)
try:
    import sys
    sys.path.insert(0, '/home/admin/ATLAS-Q')
    from triton_kernels.tdvp_mpo_ops import mpo_expectation_step_optimized
    GPU_OPTIMIZED_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZED_AVAILABLE = False


@dataclass
class MPO:
    """
    Matrix Product Operator

    Represents an operator as a chain of 4-tensors:
    O = Σ W[0]_{s₀s₀'} W[1]_{s₁s₁'} ... W[n-1]_{sₙ₋₁sₙ₋₁'}

    Each tensor W[i] has shape [χ_L, d, d, χ_R] where:
    - χ_L, χ_R: left and right bond dimensions
    - d: physical dimension (2 for qubits)
    """
    tensors: List[torch.Tensor]  # List of 4-tensors [χ_L, d, d, χ_R]
    n_sites: int

    def __post_init__(self):
        assert len(self.tensors) == self.n_sites
        # Validate shapes
        for i, W in enumerate(self.tensors):
            assert len(W.shape) == 4, f"MPO tensor {i} must be 4D"
            assert W.shape[1] == W.shape[2], f"Physical dims must match at site {i}"

    @staticmethod
    def identity(n_sites: int, device: str = 'cuda', dtype=torch.complex64) -> 'MPO':
        """Create identity MPO"""
        tensors = []
        for i in range(n_sites):
            # Identity operator: W[σ,σ'] = δ_{σ,σ'}
            W = torch.zeros(1, 2, 2, 1, dtype=dtype, device=device)
            W[0, 0, 0, 0] = 1.0
            W[0, 1, 1, 0] = 1.0
            tensors.append(W)
        return MPO(tensors, n_sites)

    @staticmethod
    def from_local_ops(ops: List[torch.Tensor], device: str = 'cuda') -> 'MPO':
        """
        Create MPO from list of local operators (one per site)

        Args:
            ops: List of 2×2 operators for each site
        """
        n_sites = len(ops)
        tensors = []

        for i, op in enumerate(ops):
            assert op.shape == (2, 2), f"Operator {i} must be 2×2"
            # Wrap operator in MPO tensor [1, 2, 2, 1]
            W = op.view(1, 2, 2, 1).to(device)
            tensors.append(W)

        return MPO(tensors, n_sites)

    @staticmethod
    def from_operators(ops: List[torch.Tensor], device: str = 'cuda') -> 'MPO':
        """Alias for from_local_ops"""
        return MPO.from_local_ops(ops, device=device)


class MPOBuilder:
    """Helper class to build common MPOs"""

    @staticmethod
    def identity_mpo(n_sites: int, device: str = 'cuda', dtype=torch.complex64) -> MPO:
        """Create identity MPO (wrapper for MPO.identity)"""
        return MPO.identity(n_sites, device=device, dtype=dtype)

    @staticmethod
    def ising_hamiltonian(n_sites: int, J: float = 1.0, h: float = 0.5,
                         device: str = 'cuda', dtype=torch.complex64) -> MPO:
        """
        Transverse-field Ising Hamiltonian:
        H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ

        Args:
            n_sites: Number of sites
            J: Coupling strength
            h: Transverse field

        Virtual bond structure (D=3):
        - 0→0: identity track
        - 0→1: emit Z (open ZZ term)
        - 1→2: close with -J Z
        - 2→2: identity tail
        - 0→2: local field -h X
        """
        I = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

        D = 3  # virtual bond dimension
        tensors = []

        for i in range(n_sites):
            if i == 0:
                # left boundary: shape [1, 2, 2, D]
                W = torch.zeros(1, 2, 2, D, dtype=dtype, device=device)
                # 0->0: I
                W[0, :, :, 0] = I
                # 0->1: Z   (open a ZZ term)
                W[0, :, :, 1] = Z
                # 0->2: -h X   (local field)
                if h != 0.0:
                    W[0, :, :, 2] = -h * X
            elif i == n_sites - 1:
                # right boundary: shape [D, 2, 2, 1]
                W = torch.zeros(D, 2, 2, 1, dtype=dtype, device=device)
                # 2->0: I    (close tail)
                W[2, :, :, 0] = I
                # 1->0: -J Z (close a ZZ term)
                W[1, :, :, 0] = -J * Z
                # 0->0: -h X (local field on last site sits on diagonal)
                if h != 0.0:
                    W[0, :, :, 0] = -h * X
            else:
                # bulk: shape [D, 2, 2, D]
                W = torch.zeros(D, 2, 2, D, dtype=dtype, device=device)
                # identity track
                W[0, :, :, 0] = I            # 0->0
                W[2, :, :, 2] = I            # 2->2
                # propagate a single Z
                W[0, :, :, 1] = Z            # 0->1
                # close ZZ with -J Z
                W[1, :, :, 2] = -J * Z       # 1->2
                # local field goes 0->2
                if h != 0.0:
                    W[0, :, :, 2] = -h * X

            tensors.append(W)

        return MPO(tensors, n_sites)

    @staticmethod
    def heisenberg_hamiltonian(n_sites: int, Jx: float = 1.0, Jy: float = 1.0,
                               Jz: float = 1.0, device: str = 'cuda',
                               dtype=torch.complex64) -> MPO:
        """
        Heisenberg Hamiltonian:
        H = Σᵢ (Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵧ ZᵢZᵢ₊₁)
        """
        I = torch.eye(2, dtype=dtype, device=device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

        tensors = []

        for i in range(n_sites):
            if i == 0:
                W = torch.zeros(1, 2, 2, 4, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[0, :, :, 1] = Jx * X
                W[0, :, :, 2] = Jy * Y
                W[0, :, :, 3] = Jz * Z
            elif i == n_sites - 1:
                W = torch.zeros(4, 2, 2, 1, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[1, :, :, 0] = X
                W[2, :, :, 0] = Y
                W[3, :, :, 0] = Z
            else:
                W = torch.zeros(4, 2, 2, 4, dtype=dtype, device=device)
                W[0, :, :, 0] = I
                W[1, :, :, 0] = X
                W[2, :, :, 0] = Y
                W[3, :, :, 0] = Z
                W[0, :, :, 1] = Jx * X
                W[0, :, :, 2] = Jy * Y
                W[0, :, :, 3] = Jz * Z

            tensors.append(W)

        return MPO(tensors, n_sites)


def apply_mpo_to_mps(mpo: MPO, mps, chi_max: int = 128, eps: float = 1e-8) -> 'AdaptiveMPS':
    """
    Apply MPO to MPS: |ψ'⟩ = O |ψ⟩

    Uses zipper/zip-up algorithm for efficient contraction

    Args:
        mpo: Matrix product operator
        mps: Matrix product state (AdaptiveMPS)
        chi_max: Maximum bond dimension after compression
        eps: Truncation tolerance

    Returns:
        New MPS after applying MPO
    """
    from .adaptive_mps import AdaptiveMPS
    from .linalg_robust import robust_svd

    assert mpo.n_sites == mps.num_qubits, "MPO and MPS must have same number of sites"

    n = mpo.n_sites
    device = mps.tensors[0].device
    dtype = mps.tensors[0].dtype

    # Result MPS tensors
    new_tensors = []

    # Contract MPO with MPS site by site
    for i in range(n):
        # W: [l, s, s', r]
        W = mpo.tensors[i].to(device=device, dtype=dtype)
        # A: [a, s', b]
        A = mps.tensors[i]

        # Contract over s'
        # M[l a, s, r b] = Σ_{s′} W[l, s, s′, r] * A[a, s′, b]
        M = torch.einsum('lstr, atb -> lasrb', W, A)          # [l, a, s, r, b]
        l, a, s, r, b = M.shape
        M = M.reshape(l * a, s, r * b)                        # [l a, s, r b]

        new_tensors.append(M)

    # Create new MPS
    result = AdaptiveMPS(n, bond_dim=2, device=device)
    result.tensors = new_tensors

    # Compress back to chi_max using SVD sweeps
    result.to_left_canonical()

    return result


def expectation_value(mpo: MPO, mps, use_gpu_optimized: bool = True) -> complex:
    """
    Compute ⟨ψ|O|ψ⟩ where O is an MPO and |ψ⟩ is an MPS

    Args:
        mpo: Matrix Product Operator
        mps: Matrix Product State
        use_gpu_optimized: Use GPU-optimized contractions (torch.compile)

    Returns:
        Complex expectation value
    """
    n = mpo.n_sites
    assert n == mps.num_qubits

    dtype = mps.tensors[0].dtype
    device = mps.tensors[0].device

    # Use GPU-optimized version if available and enabled
    use_optimized = (GPU_OPTIMIZED_AVAILABLE and
                    use_gpu_optimized and
                    device.type == 'cuda')

    # E has shape [l, ā, a]; start with scalars (1×1×1)
    E = torch.ones(1, 1, 1, dtype=dtype, device=device)

    for i in range(n):
        W = mpo.tensors[i].to(device=device, dtype=dtype)  # [l, s, s', r]
        A = mps.tensors[i]                                 # [a, s, b]

        if use_optimized:
            # GPU-optimized contraction (torch.compile + optimized order)
            E = mpo_expectation_step_optimized(E, A, W)
        else:
            # Standard einsum
            Ac = A.conj()                                      # [ā, s', b̄]
            # E' [χR, aR, bR] = Σ E[χL,aL,bL] * Ac[aL,σ',aR] * W[χL,σ,σ',χR] * A[bL,σ,bR]
            # Indices: L=χL, a=aL, b=bL, t=σ', r=aR, s=σ, R=χR, B=bR
            E = torch.einsum('Lab, atr, LstR, bsB -> RrB', E, Ac, W, A)

    # Now E should be [1, 1, 1] -> scalar
    if E.numel() == 1:
        return complex(E.item())
    else:
        # Extract the [0,0,0] element if not scalar
        return complex(E[0, 0, 0].item())


def correlation_function(op1: torch.Tensor, site1: int, op2: torch.Tensor, site2: int,
                         mps) -> complex:
    """
    Compute two-point correlation function: ⟨ψ| O₁(site1) O₂(site2) |ψ⟩

    Args:
        op1: First operator (2×2)
        site1: First site
        op2: Second operator (2×2)
        site2: Second site
        mps: MPS state

    Returns:
        Correlation ⟨O₁ O₂⟩
    """
    n = mps.num_qubits
    assert 0 <= site1 < n and 0 <= site2 < n

    # Ensure site1 < site2
    if site1 > site2:
        site1, site2 = site2, site1
        op1, op2 = op2, op1

    device = mps.tensors[0].device
    dtype = mps.tensors[0].dtype

    # Build MPO with op1 at site1, op2 at site2, identity elsewhere
    I = torch.eye(2, dtype=dtype, device=device)
    ops = [I] * n
    ops[site1] = op1.to(device=device, dtype=dtype)
    ops[site2] = op2.to(device=device, dtype=dtype)

    mpo = MPO.from_local_ops(ops, device=device)

    return expectation_value(mpo, mps)
