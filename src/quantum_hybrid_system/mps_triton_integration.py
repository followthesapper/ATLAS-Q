"""
MPS-Triton Integration Layer
=============================

Drop-in replacement for MPS operations with Triton acceleration.

Usage:
    from src.quantum_hybrid_system.mps_triton_integration import apply_two_qubit_gate

    # Auto-select Triton or PyTorch based on availability
    Ai_new, Aj_new = apply_two_qubit_gate(Ai, Aj, U, max_bond=64, prefer_triton=True)

Author: Claude Code (Phase 3)
Date: October 24, 2025
"""

import torch
from typing import Optional, Tuple

# Try to import Triton kernels
try:
    from triton_kernels.mps_complex import (
        apply_two_qubit_gate_split,
        fused_two_qubit_gate_triton,
        fused_two_qubit_gate_pytorch,
    )
    _HAS_TRITON_MPS = True
except Exception as e:
    _HAS_TRITON_MPS = False
    _IMPORT_ERROR = str(e)


def apply_two_qubit_gate(
    Ai: torch.Tensor,
    Aj: torch.Tensor,
    U: torch.Tensor,
    max_bond: Optional[int] = None,
    cutoff: Optional[float] = None,
    prefer_triton: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply two-qubit gate to neighboring MPS tensors and split back.

    This is a drop-in replacement for standard MPS gate application with
    optional Triton acceleration for improved performance.

    Backend Policy:
        Phase-3 (MPS/SVD) operations DEFAULT to PyTorch/cuBLAS (2-20× faster).
        Set prefer_triton=True only for very large bond dimensions (χ≥64, batch≥16).

    Args:
        Ai: Left MPS tensor [li, 2, ri]
        Aj: Right MPS tensor [ri, 2, rj]
        U: Two-qubit gate matrix [4, 4]
        max_bond: Maximum bond dimension after truncation (None = no limit)
        cutoff: Singular value cutoff threshold (None = no cutoff)
        prefer_triton: Use Triton acceleration (default False, PyTorch is faster for typical workloads)

    Returns:
        (Ai_new, Aj_new): Updated MPS tensors after gate and truncation

    Example:
        >>> Ai = torch.randn(32, 2, 48, dtype=torch.complex64, device='cuda')
        >>> Aj = torch.randn(48, 2, 40, dtype=torch.complex64, device='cuda')
        >>> U = create_cnot_gate()  # 4×4 unitary
        >>> Ai_new, Aj_new = apply_two_qubit_gate(Ai, Aj, U, max_bond=64)
        >>> print(Ai_new.shape)  # [32, 2, chi_new] where chi_new <= 64
    """
    if not _HAS_TRITON_MPS:
        # Fallback: use pure PyTorch implementation
        from triton_kernels.mps_complex import fused_two_qubit_gate_pytorch
        return _pytorch_fallback(Ai, Aj, U, max_bond, cutoff)

    # Use Triton-accelerated version
    use_triton = prefer_triton and Ai.device.type == 'cuda'

    return apply_two_qubit_gate_split(
        Ai, Aj, U,
        max_bond=max_bond,
        cutoff=cutoff,
        use_triton=use_triton,
    )


def _pytorch_fallback(
    Ai: torch.Tensor,
    Aj: torch.Tensor,
    U: torch.Tensor,
    max_bond: Optional[int],
    cutoff: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch fallback if Triton is unavailable."""
    li, _, ri = Ai.shape
    _, _, rj = Aj.shape

    # Contract and apply gate
    Psi = torch.einsum('lpr,rqj->lpqj', Ai, Aj)
    Psi = Psi.permute(1, 2, 0, 3).reshape(4, li * rj)
    Psi_out = (U @ Psi).reshape(2, 2, li, rj).permute(2, 0, 1, 3).reshape(li * 2, 2 * rj)

    # SVD
    U_svd, S, Vh = torch.linalg.svd(Psi_out, full_matrices=False)

    # Truncation
    chi_max = len(S)
    if cutoff is not None:
        chi_new = torch.sum(S > cutoff).item()
    elif max_bond is not None:
        chi_new = min(max_bond, chi_max)
    else:
        chi_new = chi_max

    chi_new = max(1, chi_new)

    # Split
    U_svd = U_svd[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    Ai_new = U_svd.reshape(li, 2, chi_new)

    # Promote S to complex to match Vh dtype (S from SVD is always real)
    S_complex = S.to(dtype=Vh.dtype)
    Aj_new = (torch.diag(S_complex) @ Vh).reshape(chi_new, 2, rj)

    return Ai_new.contiguous(), Aj_new.contiguous()


def is_triton_available() -> bool:
    """Check if Triton MPS kernels are available."""
    return _HAS_TRITON_MPS


def get_triton_status() -> str:
    """Get detailed status of Triton availability."""
    if _HAS_TRITON_MPS:
        return "✅ Triton MPS kernels available"
    else:
        return f"❌ Triton MPS kernels unavailable: {_IMPORT_ERROR}"


# Convenience exports
__all__ = [
    'apply_two_qubit_gate',
    'is_triton_available',
    'get_triton_status',
]
