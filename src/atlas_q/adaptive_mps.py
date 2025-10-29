"""
Adaptive Matrix Product State for Moderate-to-High Entanglement

Extends MatrixProductStatePyTorch with:
- Adaptive bond dimension by tolerance
- Per-bond χ caps and global memory budget
- Mixed precision (complex32/complex64) support
- Two-site gate application with automatic SVD truncation
- Comprehensive logging and diagnostics

Mathematical guarantees:
- Local error control: ε_local² = Σ_{i>k} σ_i² ≤ ε_bond²
- Global error bound: ε_global ≤ sqrt(Σ_b ε²_local,b)
- Entropy: S_b = -Σ_i p_i log(p_i) where p_i = σ_i²/Σ_j σ_j²

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .diagnostics import MPSStatistics
from .linalg_robust import robust_qr, robust_svd
from .mps_pytorch import MatrixProductStatePyTorch
from .truncation import check_entropy_sanity, choose_rank_from_sigma

# Triton-accelerated gate operations (if available)
try:
    from triton_kernels.mps_complex import fused_two_qubit_gate_pytorch, fused_two_qubit_gate_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    fused_two_qubit_gate_pytorch = None


@dataclass
class DTypePolicy:
    """Mixed precision policy configuration"""

    default: torch.dtype = torch.complex64
    promote_if_cond_gt: float = 1e6  # Promote to complex128 if cond(S) exceeds this


class AdaptiveMPS(MatrixProductStatePyTorch):
    """
    Adaptive MPS for moderate-to-high entanglement simulation

    Key features:
    - Variable per-bond dimensions with adaptive truncation
    - Energy-based rank selection: keep k such that Σ_{i≤k} σ_i² ≥ (1-ε²) Σ_i σ_i²
    - Per-bond χ caps and global memory budget enforcement
    - Mixed precision with automatic promotion on numerical instability
    - Two-site gate application (TEBD-style)
    - Comprehensive statistics and error tracking

    Example:
        >>> mps = AdaptiveMPS(16, bond_dim=8, eps_bond=1e-6, chi_max_per_bond=64)
        >>> H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64)/torch.sqrt(torch.tensor(2.0))
        >>> for q in range(16):
        >>>     mps.apply_single_qubit_gate(q, H)
        >>> CZ = torch.diag(torch.tensor([1,1,1,-1], dtype=torch.complex64))
        >>> for i in range(0, 15, 2):
        >>>     mps.apply_two_site_gate(i, CZ)
        >>> print(mps.stats_summary())
    """

    def __init__(
        self,
        num_qubits: int,
        bond_dim: int = 8,
        *,
        eps_bond: float = 1e-6,
        chi_max_per_bond: Optional[Union[int, List[int]]] = 256,
        budget_global_mb: Optional[float] = None,
        dtype_policy: DTypePolicy = DTypePolicy(),
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize Adaptive MPS

        Args:
            num_qubits: Number of qubits
            bond_dim: Initial bond dimension
            eps_bond: Energy tolerance for truncation (default 1e-6)
            chi_max_per_bond: Max χ per bond (int or list of ints)
            budget_global_mb: Global memory budget in MB (None = unlimited)
            dtype_policy: Mixed precision policy
            device: 'cuda' or 'cpu'
            dtype: Explicit dtype (overrides dtype_policy.default if provided)
        """
        # Use explicit dtype if provided, otherwise fall back to dtype_policy
        if dtype is None:
            dtype = dtype_policy.default

        super().__init__(num_qubits, bond_dim, device, dtype)

        self.eps_bond = eps_bond
        self.dtype_policy = dtype_policy
        self.budget_global_mb = budget_global_mb

        # Per-bond χ caps
        if isinstance(chi_max_per_bond, int):
            self.chi_max_per_bond = [chi_max_per_bond] * (num_qubits - 1)
        else:
            assert (
                len(chi_max_per_bond) == num_qubits - 1
            ), f"chi_max_per_bond must have length {num_qubits-1}"
            self.chi_max_per_bond = list(chi_max_per_bond)

        # Track actual bond dimensions (initially uniform)
        self.bond_dims = [bond_dim] * (num_qubits - 1)

        # Statistics tracker
        self.statistics = MPSStatistics()
        self._operation_counter = 0

        # Initialize to computational zero state |00...0⟩
        self._initialize_zero_state()

    def _initialize_zero_state(self):
        """Initialize MPS to computational zero state |00...0⟩"""
        # For |00...0⟩, only the [*, 0, *] entry of each tensor is non-zero
        # All tensors: set [:, 1, :] = 0 and [:, 0, :] = identity-like
        for i in range(self.num_qubits):
            T = self.tensors[i]
            # Zero out the |1⟩ component
            T[:, 1, :] = 0
            # Set |0⟩ component to identity (or close to it)
            T[:, 0, :] = torch.eye(T.shape[0], T.shape[2], dtype=T.dtype, device=T.device)
        # Normalize
        self._normalize()

    def _log_operation(self, **kwargs):
        """Log an operation to statistics"""
        self.statistics.record(step=self._operation_counter, **kwargs)
        self._operation_counter += 1

    def stats_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        return self.statistics.summary()

    def global_error_bound(self) -> float:
        """Get global error upper bound"""
        return self.statistics.global_error_bound()

    def reset_stats(self):
        """Reset statistics tracking"""
        self.statistics.reset()
        self._operation_counter = 0

    @torch.no_grad()
    def apply_single_qubit_gate(self, q: int, U2: torch.Tensor):
        """
        Apply single-qubit gate (fast path, no truncation needed)

        Args:
            q: Qubit index
            U2: 2x2 unitary gate

        Complexity: O(χ²)
        """
        assert 0 <= q < self.num_qubits, f"Qubit index {q} out of range"
        assert U2.shape == (2, 2), "U2 must be 2x2"

        T = self.tensors[q]
        # Contract: T[a,s,b] * U[s,t] -> T[a,t,b]
        # Move gate to same device and dtype as tensor
        U2_device = U2.to(device=T.device, dtype=T.dtype)
        self.tensors[q] = torch.einsum("st,asb->atb", U2_device, T)

    @torch.no_grad()
    def apply_two_site_gate(self, i: int, U4: torch.Tensor):
        """
        Apply two-qubit gate with adaptive SVD truncation

        This is the core TEBD operation for moderate entanglement.

        Args:
            i: Bond index (applies to qubits i and i+1)
            U4: 4x4 unitary gate (or 2x2x2x2 tensor)

        Steps:
        1. Merge tensors at sites (i, i+1) into Θ
        2. Apply gate U
        3. SVD: Θ = U S V†
        4. Adaptively select rank k by energy criterion + caps
        5. Split back into two cores with updated χ

        Complexity: O(χ³) for SVD
        """
        assert 0 <= i < self.num_qubits - 1, f"Bond index {i} out of range"

        start_time = time.time()

        A, B = self.tensors[i], self.tensors[i + 1]
        χL, χM, χR = A.shape[0], A.shape[2], B.shape[2]
        device = A.device

        # Choose dtype (start with policy default, may promote later)
        local_dtype = self.dtype_policy.default
        A = A.to(dtype=local_dtype)
        B = B.to(dtype=local_dtype)

        # Reshape U4 to 4x4 if needed
        if U4.shape == (2, 2, 2, 2):
            U_matrix = U4.reshape(4, 4)
        elif U4.shape == (4, 4):
            U_matrix = U4
        else:
            raise ValueError(f"U4 must be (2,2,2,2) or (4,4), got {U4.shape}")

        U_matrix = U_matrix.to(device=device, dtype=local_dtype)

        # Steps 1-3: Fused gate application (using Triton if available on CUDA)
        # This fuses: merge tensors + apply gate + reshape for SVD
        use_triton = TRITON_AVAILABLE and device.type == "cuda"

        if use_triton:
            try:
                # Use Triton-accelerated fused kernel (1.5-3× faster)
                # Input: A[li,2,ri], B[ri,2,rj], U[4,4]
                # Output: X[li*2, 2*rj] ready for SVD
                X = fused_two_qubit_gate_triton(A, B, U_matrix)
            except Exception:
                # Fall back to PyTorch if Triton fails
                X = fused_two_qubit_gate_pytorch(A, B, U_matrix)
        else:
            # Standard PyTorch path (no Triton available or on CPU)
            if fused_two_qubit_gate_pytorch is not None:
                X = fused_two_qubit_gate_pytorch(A, B, U_matrix)
            else:
                # Fallback: manual einsum operations
                Theta = torch.einsum("asm,mtb->astb", A, B)  # [χL, 2, 2, χR]
                U = U_matrix.view(2, 2, 2, 2)
                Theta_new = torch.einsum("stuv,astb->auvb", U, Theta)
                X = Theta_new.reshape(χL * 2, 2 * χR)

        # Step 4: SVD with fallback
        U, S, Vh, driver = robust_svd(X)

        # Step 5: Adaptive rank selection
        cap = self.chi_max_per_bond[i]

        def budget_ok(k: int) -> bool:
            if self.budget_global_mb is None:
                return True
            # Estimate memory delta
            bytes_per_elem = torch.finfo(local_dtype).bits // 8
            before = (χL * χM + χM * χR) * 2 * bytes_per_elem
            after = (χL * k + k * χR) * 2 * bytes_per_elem
            delta = max(0, after - before)
            current_mb = self.memory_usage() / (1024**2)
            return (current_mb + delta / (1024**2)) < self.budget_global_mb

        k, eps_loc, entropy, condS = choose_rank_from_sigma(S, self.eps_bond, cap, budget_ok)

        # Step 6: Check if we need to promote precision
        if math.isfinite(condS) and condS > self.dtype_policy.promote_if_cond_gt:
            # Promote to complex128 and recompute
            X_promoted = X.to(torch.complex128)
            U, S, Vh, driver = robust_svd(X_promoted)
            k, eps_loc, entropy, condS = choose_rank_from_sigma(S, self.eps_bond, cap, budget_ok)
            local_dtype = torch.complex128

        # Step 7: Sanity check entropy
        if not check_entropy_sanity(entropy, χL, χR):
            print(f"Warning: Entropy {entropy:.3f} exceeds physical bound at bond {i}")

        # Step 8: Rebuild cores
        US = U[:, :k] * S[:k]  # [2χL, k]
        VhK = Vh[:k, :]  # [k, 2χR]

        A_new = US.reshape(χL, 2, k)  # [χL, 2, k]
        B_new = VhK.reshape(k, 2, χR)  # [k, 2, χR]

        # Step 9: Update tensors
        self.tensors[i] = A_new
        self.tensors[i + 1] = B_new
        self.bond_dims[i] = k

        # Step 10: Log operation
        elapsed_ms = (time.time() - start_time) * 1000
        self._log_operation(
            bond=i,
            k_star=k,
            chi_before=χM,
            chi_after=k,
            eps_local=eps_loc,
            entropy=entropy,
            svd_driver=driver,
            dtype=str(local_dtype),
            ms_elapsed=elapsed_ms,
            condS=condS,
        )

    @torch.no_grad()
    def to_left_canonical(self):
        """
        Bring MPS into left-canonical form using QR

        After this, each tensor A^[i] satisfies:
        Σ_s (A^[i]_s)† A^[i]_s = I

        Complexity: O(n · χ³)
        """
        for i in range(self.num_qubits - 1):
            A = self.tensors[i]
            χL, p, χR = A.shape

            # QR decomposition
            Q, R, _ = robust_qr(A.reshape(χL * p, χR))
            χmid = Q.shape[1]

            # Update this tensor
            self.tensors[i] = Q.reshape(χL, p, χmid)

            # Absorb R into next tensor
            B = self.tensors[i + 1]
            self.tensors[i + 1] = torch.einsum("ij,jkl->ikl", R, B)

            # Update bond dimension
            if i < len(self.bond_dims):
                self.bond_dims[i] = χmid

        self.is_canonical = True

    @torch.no_grad()
    def to_mixed_canonical(self, center: int):
        """
        Bring MPS into mixed-canonical form with center at specified site

        Sites 0..center-1 are left-canonical
        Sites center+1..n-1 are right-canonical
        Site center holds the normalization

        Args:
            center: Center site index

        Complexity: O(n · χ³)
        """
        assert 0 <= center < self.num_qubits, "Center out of range"

        # Left-canonicalize up to center
        for i in range(center):
            A = self.tensors[i]
            χL, p, χR = A.shape
            Q, R, _ = robust_qr(A.reshape(χL * p, χR))
            χmid = Q.shape[1]
            self.tensors[i] = Q.reshape(χL, p, χmid)
            B = self.tensors[i + 1]
            self.tensors[i + 1] = torch.einsum("ij,jkl->ikl", R, B)

        # Right-canonicalize from end down to center+1
        for i in range(self.num_qubits - 1, center, -1):
            B = self.tensors[i]
            χL, p, χR = B.shape
            # Reshape to [χL, p·χR] for QR
            B_mat = B.permute(1, 2, 0).reshape(p * χR, χL)
            Q, R, _ = robust_qr(B_mat)
            χmid = Q.shape[1]
            # Reshape back
            self.tensors[i] = Q.t().reshape(χmid, p, χR)
            # Absorb R into previous tensor
            if i > 0:
                A = self.tensors[i - 1]
                self.tensors[i - 1] = torch.einsum("ijk,kl->ijl", A, R.t())

    def snapshot(self, path: str):
        """
        Save MPS to file for checkpointing

        Args:
            path: File path to save to
        """
        torch.save(
            {
                "tensors": [t.cpu() for t in self.tensors],
                "bond_dims": self.bond_dims,
                "num_qubits": self.num_qubits,
                "eps_bond": self.eps_bond,
                "chi_max_per_bond": self.chi_max_per_bond,
                "statistics": self.statistics.logs,
            },
            path,
        )

    @staticmethod
    def load_snapshot(path: str, device: str = "cuda") -> "AdaptiveMPS":
        """
        Load MPS from checkpoint file

        Args:
            path: File path to load from
            device: Device to place tensors on

        Returns:
            Loaded AdaptiveMPS instance
        """
        data = torch.load(path)

        mps = AdaptiveMPS(
            num_qubits=data["num_qubits"],
            bond_dim=max(data["bond_dims"]),
            eps_bond=data["eps_bond"],
            chi_max_per_bond=data["chi_max_per_bond"],
            device=device,
        )

        mps.tensors = [t.to(device) for t in data["tensors"]]
        mps.bond_dims = data["bond_dims"]
        if "statistics" in data:
            mps.statistics.logs = data["statistics"]

        return mps

    def get_memory_usage(self) -> int:
        """
        Get total memory usage in bytes

        Returns:
            Total bytes used by all tensors
        """
        return self.memory_usage()

    @torch.no_grad()
    def to_statevector(self) -> torch.Tensor:
        """
        Convert MPS to full statevector (ONLY for small systems!)

        Returns:
            Full statevector of size 2^n

        Warning:
            This scales as O(2^n) and should ONLY be used for testing
            small systems (n ≤ 20). For larger systems, use get_amplitude().
        """
        if self.num_qubits > 20:
            raise ValueError(
                f"Converting {self.num_qubits} qubits to statevector "
                f"would require {2**self.num_qubits} amplitudes. "
                f"Use get_amplitude() for large systems."
            )

        # Contract all tensors: T[0] * T[1] * ... * T[n-1]
        # Determine target dtype (use the highest precision dtype present)
        target_dtype = self.dtype if hasattr(self, 'dtype') else self.tensors[0].dtype

        result = self.tensors[0].to(dtype=target_dtype)  # [1, 2, χ₁]

        for i in range(1, self.num_qubits):
            # result: [..., χᵢ]
            # T[i]: [χᵢ, 2, χᵢ₊₁]
            # Contract over χᵢ dimension
            # Ensure both tensors have same dtype
            tensor_i = self.tensors[i].to(dtype=target_dtype)
            result = torch.einsum("...i,ijk->...jk", result, tensor_i)

        # Final tensor: [2, 2, ..., 2, 1]
        # Squeeze the trailing dimension and flatten
        result = result.squeeze(-1).reshape(-1)

        return result
