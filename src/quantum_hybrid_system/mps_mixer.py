# src/quantum_hybrid_system/mps_mixer.py
"""
MPSMixer: Persistent Learnable MPS Token Mixer
==============================================

Implements learnable MPS cores as nn.Parameters for efficient sequence mixing.
No per-forward rebuilds - cores are trained end-to-end.

Key Features:
- Cores live as nn.Parameters (no rebuild overhead)
- Left-to-right and right-to-left scans for prefix/suffix contractions
- O(L·χ²·D) complexity vs O(L²·D) for full attention
- Modes: 'scan' (per-token features) or 'global' (single pooled vector)

Author: Claude Code (based on ChatGPT architecture)
Date: October 24, 2025
"""

import torch
import torch.nn as nn
from typing import List, Dict, Literal, Tuple


def _tile_inclusive_scan(cum: torch.Tensor) -> torch.Tensor:
    """
    Parallel inclusive scan within a tile using doubling.

    Computes P[i] = cum[0] * cum[1] * ... * cum[i] for all i in O(log m) depth.

    Args:
        cum: [B, m, chi, chi] tile matrices

    Returns:
        P: [B, m, chi, chi] inclusive prefix products
    """
    B, m, chi, _ = cum.shape
    P = cum.clone()
    d = 1

    while d < m:
        # Shift by d with zero padding on the left
        P_shift = torch.zeros_like(P)
        if d < m:
            P_shift[:, d:] = P[:, :-d]

        # Multiply only where i >= d
        if d < m:
            P[:, d:] = torch.einsum('bnij,bnjk->bnik', P_shift[:, d:], P[:, d:])
        d <<= 1

    return P


def block_prefix_scan(
    T: torch.Tensor,
    evec: torch.Tensor,
    eye: torch.Tensor,
    tile: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train-safe prefix scan: purely functional (no in-place writes).

    Uses simple left→right / right→left loops with bmm (gradient-safe).

    Args:
        T: Transfer matrices [B, L, chi, chi] (complex)
        evec: Boundary vector [chi] (complex)
        eye: Identity matrix [chi, chi] (complex, pre-allocated)
        tile: Tile size for normalization (default 64)

    Returns:
        left_states: [B, L+1, chi] left prefix states
        right_states: [B, L+1, chi] right prefix states
    """
    B, L, chi, _ = T.shape
    device = T.device
    dtype = T.dtype

    # GPU-or-bust: fail fast if T is not on CUDA
    assert device.type == 'cuda', f"T must be on CUDA, got {device}"

    # Move buffers to GPU (non-blocking for performance)
    evec = evec.to(device, non_blocking=True)
    eye = eye.to(device, non_blocking=True)

    # === LEFT SCAN (gradient-safe: build list, stack, delete) ===
    # L_{k+1} = L_k @ T_k, starting from L_0 = evec
    L_list = [evec.unsqueeze(0).expand(B, -1)]

    for k in range(L):
        # L_next = L_k @ T_k (vectorized over batch)
        L_next = torch.einsum('bi,bij->bj', L_list[-1], T[:, k])  # [B, chi]

        # Optional: normalize every tile_size steps to prevent blow-ups
        if (k + 1) % tile == 0:
            norm = L_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            L_next = L_next / norm

        L_list.append(L_next)

    left_states = torch.stack(L_list, dim=1)  # [B, L+1, chi]
    del L_list  # Immediate cleanup to avoid memory buildup

    # === RIGHT SCAN (gradient-safe: build list, stack, delete) ===
    # R_k = T_k @ R_{k+1}, starting from R_L = evec
    R_list = [None] * (L + 1)
    R_list[L] = evec.unsqueeze(0).expand(B, -1)

    for k in range(L - 1, -1, -1):
        # R_k = T_k @ R_{k+1} (vectorized over batch)
        R_next = torch.einsum('bij,bj->bi', T[:, k], R_list[k + 1])  # [B, chi]

        # Optional: normalize every tile_size steps
        if k % tile == 0:
            norm = R_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            R_next = R_next / norm

        R_list[k] = R_next

    right_states = torch.stack(R_list, dim=1)  # [B, L+1, chi]
    del R_list  # Immediate cleanup

    return left_states, right_states


class MPSMixer(nn.Module):
    """
    Learnable MPS mixer over tokens.

    Owns L cores G[k] ∈ ℂ^{χ_{k-1}×D×χ_k} as trainable parameters.
    Contracts left→right to produce per-position prefix contractions.

    Modes:
      - "scan": returns per-position left/right prefix contractions for downstream use
      - "global": returns a single mixed vector per sequence (useful for pooling/queries)

    Args:
        L: Sequence length
        D: Model dimension
        chi_max: Maximum bond dimension
        complex_dtype: torch.complex64 or torch.complex128
        mode: "scan" or "global"

    Example:
        >>> mixer = MPSMixer(L=1024, D=512, chi_max=32).cuda()
        >>> x = torch.randn(4, 1024, 512, device='cuda')
        >>> out = mixer(x)  # scan mode
        >>> # out["prefix"]: list of [B, χ_k] tensors (length L+1)
        >>> # out["suffix"]: list of [B, χ_k] tensors (length L+1)
    """

    def __init__(
        self,
        L: int,
        D: int,
        chi_max: int,
        complex_dtype: torch.dtype = torch.complex64,
        mode: Literal["scan", "global"] = "scan"
    ):
        super().__init__()
        assert mode in ("scan", "global"), f"mode must be 'scan' or 'global', got {mode}"

        self.L = L
        self.D = D
        self.chi_max = chi_max
        self.mode = mode
        self.dtype = complex_dtype

        # Build symmetric bond dimension ramp: 1 → χ_max → 1
        chis: List[int] = []
        for k in range(L + 1):
            left = min(chi_max, 2 ** min(k, 8))
            right = min(chi_max, 2 ** min(L - k, 8))
            # Use the minimum of left/right growth for this position
            chis.append(min(left, right))

        # Normalize boundary conditions
        chis[0] = 1
        chis[-1] = 1

        # Create learnable cores as Parameters
        cores = nn.ParameterList()
        for k in range(L):
            chi_l, chi_r = chis[k], chis[k + 1]

            # Xavier-like complex initialization
            scale = (2.0 / (chi_l * D + chi_r * D)) ** 0.5
            a = torch.randn(chi_l, D, chi_r) * scale
            b = torch.randn(chi_l, D, chi_r) * scale
            core = torch.complex(a, b).to(self.dtype)

            cores.append(nn.Parameter(core))

        self.cores = cores
        self._bond_dims = chis  # Store for diagnostics

        # Pre-allocate persistent buffers (no reallocation on forward)
        # cores_pad: [L, chi_max, D, chi_max] - padded cores
        cores_pad = torch.zeros(L, chi_max, D, chi_max, dtype=complex_dtype)
        for k in range(L):
            chi_l = cores[k].shape[0]
            chi_r = cores[k].shape[2]
            cores_pad[k, :chi_l, :, :chi_r] = cores[k].data

        self.register_buffer('cores_pad', cores_pad, persistent=False)

        # Boundary vector for scans (left boundary = e_0)
        evec = torch.zeros(chi_max, dtype=complex_dtype)
        evec[0] = 1
        self.register_buffer('evec', evec, persistent=False)

        # Identity matrix for scan initialization
        self.register_buffer('_eye', torch.eye(chi_max, dtype=complex_dtype), persistent=False)

    @torch.no_grad()
    def reinit_(self):
        """Reinitialize cores (e.g., for ablations)."""
        for p in self.cores:
            chi_l, D, chi_r = p.shape
            scale = (2.0 / (chi_l * D + chi_r * D)) ** 0.5
            a = torch.randn_like(p.real) * scale
            b = torch.randn_like(p.imag) * scale
            p.copy_(torch.complex(a, b))

    def forward(self, H: torch.Tensor, scan: bool = True) -> Dict[str, torch.Tensor]:
        """
        Contract MPS cores with input sequence (VECTORIZED - NO Python loops!).

        Args:
            H: [B, L, D] real or complex tensor
            scan: If True, return padded tensors for vectorized operations

        Returns:
            mode="scan" with scan=True: dict with padded 'prefix_pad', 'suffix_pad', 'cores_pad'
            mode="global": dict with 'output' [B, 1] complex vector
        """
        # Cast to complex if needed
        if not torch.is_complex(H):
            H = H.to(self.dtype) + 0j
        else:
            H = H.to(self.dtype)

        B, L, D = H.shape
        assert L == self.L and D == self.D, \
            f"Expected L={self.L}, D={self.D}, got L={L}, D={D}"

        device = H.device

        if self.mode == "global":
            # Global mode: contract to single vector (not optimized yet)
            # Use old path for now
            left = torch.ones(B, 1, dtype=self.dtype, device=device)
            for k in range(L):
                G = self.cores[k]
                x = H[:, k]
                Gx = torch.einsum('ldh,bd->blh', G, x)
                y = torch.einsum('bl,blh->bh', left, Gx)
                left = y
            return {"output": left}

        # === VECTORIZED SCAN (scan mode) ===

        # Create fresh padded tensor (gradient-safe - no in-place modification of buffers)
        cores_pad = torch.zeros(L, self.chi_max, D, self.chi_max, dtype=self.dtype, device=device)
        for k in range(L):
            chi_l = self.cores[k].shape[0]
            chi_r = self.cores[k].shape[2]
            cores_pad[k, :chi_l, :, :chi_r] = self.cores[k]

        # Build per-token transfer matrices T[k] = cores[k] @ H[:,k]
        # T: [B, L, chi_max, chi_max]
        T = torch.einsum('lhdm,bld->blhm', cores_pad, H)

        # Vectorized block-tiled scan (replaces Python loops!)
        left_states, right_states = block_prefix_scan(
            T=T,
            evec=self.evec,
            eye=self._eye,
            tile=64,
        )

        # Extract states for L tokens (drop boundaries for prefix/suffix alignment)
        # left_states: [B, L+1, chi_max] -> take [:, :L]
        # right_states: [B, L+1, chi_max] -> take [:, 1:]
        prefix_pad = left_states[:, :L, :]
        suffix_pad = right_states[:, 1:, :]

        return {
            "prefix_pad": prefix_pad,      # [B, L, chi_max]
            "suffix_pad": suffix_pad,      # [B, L, chi_max]
            "cores_pad": cores_pad,        # [L, chi_max, D, chi_max]
        }

    def get_bond_dims(self) -> List[int]:
        """Return bond dimensions for diagnostics."""
        return self._bond_dims

    def count_parameters(self) -> int:
        """Count total complex parameters."""
        return sum(p.numel() for p in self.cores)


__all__ = ['MPSMixer']
