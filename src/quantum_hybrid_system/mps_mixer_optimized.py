# src/quantum_hybrid_system/mps_mixer_optimized.py
"""
MPSMixer with Optimized Parallel Scan
======================================

Optimized version focusing on reducing sequential depth and better batching.
Uses torch.bmm and einsum more effectively to minimize Python loops.

Key optimizations:
1. Fully vectorized tile scan with batched operations
2. Optimized cross-tile propagation
3. Contiguous memory layout
4. Minimal Python loops

Author: Claude Code (optimizing after Tensor Core experiment)
Date: October 24, 2025
"""

import torch
import torch.nn as nn
from typing import List, Dict, Literal, Tuple


def parallel_prefix_scan_batched(T: torch.Tensor, tile_size: int = 64) -> torch.Tensor:
    """
    Fully vectorized parallel prefix scan using doubling.

    Optimized to minimize Python loops and maximize GPU parallelism.

    Args:
        T: [B, L, chi, chi] transfer matrices
        tile_size: Size of tiles for blocking

    Returns:
        P: [B, L, chi, chi] prefix products (P[i] = T[0] @ T[1] @ ... @ T[i])
    """
    B, L, chi, _ = T.shape
    device = T.device
    dtype = T.dtype

    # Process in tiles to balance memory and parallelism
    ntiles = (L + tile_size - 1) // tile_size

    # Pre-allocate output
    P = torch.empty(B, L, chi, chi, dtype=dtype, device=device)

    # Step 1: In-tile doubling scan (fully parallel within each tile)
    tile_boundaries = torch.zeros(B, ntiles, chi, chi, dtype=dtype, device=device)

    for t in range(ntiles):
        s = t * tile_size
        e = min(L, s + tile_size)
        m = e - s

        # Extract tile: [B, m, chi, chi]
        tile = T[:, s:e]

        # Doubling scan within tile (log m depth)
        scan = tile.clone()
        d = 1
        while d < m:
            if d < m:
                # Shift by d: [B, m, chi, chi]
                shifted = torch.roll(scan, shifts=d, dims=1)
                # Zero out first d elements
                shifted[:, :d] = torch.eye(chi, dtype=dtype, device=device).reshape(1, 1, chi, chi)

                # Batched matmul: [B*m, chi, chi] @ [B*m, chi, chi]
                scan_flat = scan.reshape(B * m, chi, chi)
                shifted_flat = shifted.reshape(B * m, chi, chi)
                result = torch.bmm(shifted_flat, scan_flat)
                scan = result.reshape(B, m, chi, chi)

            d <<= 1

        P[:, s:e] = scan
        tile_boundaries[:, t] = scan[:, -1]  # Last element of each tile

    # Step 2: Cross-tile propagation (sequential but optimized)
    if ntiles > 1:
        # Compute tile prefix products
        tile_prefix = torch.empty(B, ntiles, chi, chi, dtype=dtype, device=device)
        tile_prefix[:, 0] = tile_boundaries[:, 0]

        for t in range(1, ntiles):
            tile_prefix[:, t] = torch.bmm(
                tile_prefix[:, t-1].reshape(B, chi, chi),
                tile_boundaries[:, t].reshape(B, chi, chi)
            ).reshape(B, chi, chi)

        # Propagate tile prefixes back to positions
        for t in range(1, ntiles):
            s = t * tile_size
            e = min(L, s + tile_size)

            # Apply tile prefix to all positions in tile
            # P[s:e] = tile_prefix[t-1] @ P[s:e]
            left = tile_prefix[:, t-1].unsqueeze(1)  # [B, 1, chi, chi]
            right = P[:, s:e]  # [B, m, chi, chi]

            # Batched multiply
            m = e - s
            result = torch.bmm(
                left.expand(B, m, chi, chi).reshape(B * m, chi, chi),
                right.reshape(B * m, chi, chi)
            )
            P[:, s:e] = result.reshape(B, m, chi, chi)

    return P


def block_prefix_scan_optimized(
    T: torch.Tensor,
    evec: torch.Tensor,
    eye: torch.Tensor,
    tile: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized block-tiled prefix scan with better batching.

    Args:
        T: Transfer matrices [B, L, chi, chi] (complex)
        evec: Boundary vector [chi] (complex)
        eye: Identity matrix [chi, chi] (complex)
        tile: Tile size

    Returns:
        left_states: [B, L+1, chi] left prefix states
        right_states: [B, L+1, chi] right prefix states
    """
    B, L, chi, _ = T.shape
    device = T.device
    dtype = T.dtype

    # === LEFT SCAN ===
    # Compute prefix products: P[i] = T[0] @ T[1] @ ... @ T[i]
    T_cum = parallel_prefix_scan_batched(T, tile_size=tile)

    # Extract left states by contracting with boundary vector
    # left[i] = T_cum[i-1] @ evec (with left[0] = evec)
    left_states = torch.empty(B, L + 1, chi, dtype=dtype, device=device)
    left_states[:, 0] = evec.unsqueeze(0).expand(B, -1)

    # Vectorized contraction: [B, L, chi, chi] @ [chi] -> [B, L, chi]
    left_states[:, 1:] = torch.einsum('blij,j->bli', T_cum, evec)

    # === RIGHT SCAN ===
    # Reverse and transpose for right-to-left scan
    T_rev = T.flip(dims=[1]).transpose(-2, -1).contiguous()

    # Compute reversed prefix products
    T_cum_rev = parallel_prefix_scan_batched(T_rev, tile_size=tile)

    # Extract right states
    right_states_rev = torch.empty(B, L + 1, chi, dtype=dtype, device=device)
    right_states_rev[:, 0] = evec.unsqueeze(0).expand(B, -1)
    right_states_rev[:, 1:] = torch.einsum('blij,j->bli', T_cum_rev, evec)

    # Flip back to original order
    right_states = right_states_rev.flip(dims=[1])

    return left_states, right_states


class MPSMixerOptimized(nn.Module):
    """
    Learnable MPS mixer with optimized parallel scan.

    Focuses on reducing sequential depth and better batching rather than
    Tensor Cores (which didn't help for small matrices).

    Args:
        L: Sequence length
        D: Model dimension
        chi_max: Maximum bond dimension
        complex_dtype: torch.complex64 or torch.complex128
        mode: "scan" or "global"
        tile_size: Tile size for scan (default 64, tune per GPU)

    Example:
        >>> mixer = MPSMixerOptimized(L=1024, D=512, chi_max=32).cuda()
        >>> x = torch.randn(4, 1024, 512, device='cuda')
        >>> out = mixer(x)
    """

    def __init__(
        self,
        L: int,
        D: int,
        chi_max: int,
        complex_dtype: torch.dtype = torch.complex64,
        mode: Literal["scan", "global"] = "scan",
        tile_size: int = 64,
    ):
        super().__init__()
        assert mode in ("scan", "global"), f"mode must be 'scan' or 'global', got {mode}"

        self.L = L
        self.D = D
        self.chi_max = chi_max
        self.mode = mode
        self.dtype = complex_dtype
        self.tile_size = tile_size

        # Build bond dimension ramp
        chis: List[int] = []
        for k in range(L + 1):
            left = min(chi_max, 2 ** min(k, 8))
            right = min(chi_max, 2 ** min(L - k, 8))
            chis.append(min(left, right))

        chis[0] = 1
        chis[-1] = 1

        # Learnable cores
        cores = nn.ParameterList()
        for k in range(L):
            chi_l, chi_r = chis[k], chis[k + 1]
            scale = (2.0 / (chi_l * D + chi_r * D)) ** 0.5
            a = torch.randn(chi_l, D, chi_r) * scale
            b = torch.randn(chi_l, D, chi_r) * scale
            core = torch.complex(a, b).to(self.dtype)
            cores.append(nn.Parameter(core))

        self.cores = cores
        self._bond_dims = chis

        # Pre-allocate buffers
        cores_pad = torch.zeros(L, chi_max, D, chi_max, dtype=complex_dtype)
        for k in range(L):
            chi_l = cores[k].shape[0]
            chi_r = cores[k].shape[2]
            cores_pad[k, :chi_l, :, :chi_r] = cores[k].data

        self.register_buffer('cores_pad', cores_pad, persistent=False)

        evec = torch.zeros(chi_max, dtype=complex_dtype)
        evec[0] = 1
        self.register_buffer('evec', evec, persistent=False)

        self.register_buffer('_eye', torch.eye(chi_max, dtype=complex_dtype), persistent=False)

    @torch.no_grad()
    def reinit_(self):
        """Reinitialize cores."""
        for p in self.cores:
            chi_l, D, chi_r = p.shape
            scale = (2.0 / (chi_l * D + chi_r * D)) ** 0.5
            a = torch.randn_like(p.real) * scale
            b = torch.randn_like(p.imag) * scale
            p.copy_(torch.complex(a, b))

    def forward(self, H: torch.Tensor, scan: bool = True) -> Dict[str, torch.Tensor]:
        """
        Contract MPS cores with optimized parallel scan.

        Args:
            H: [B, L, D] real or complex tensor
            scan: If True, return padded tensors

        Returns:
            Dict with 'prefix_pad', 'suffix_pad', 'cores_pad'
        """
        if not torch.is_complex(H):
            H = H.to(self.dtype) + 0j
        else:
            H = H.to(self.dtype)

        B, L, D = H.shape
        assert L == self.L and D == self.D, \
            f"Expected L={self.L}, D={self.D}, got L={L}, D={D}"

        device = H.device

        if self.mode == "global":
            # Fallback for global mode
            left = torch.ones(B, 1, dtype=self.dtype, device=device)
            for k in range(L):
                G = self.cores[k]
                x = H[:, k]
                Gx = torch.einsum('ldh,bd->blh', G, x)
                y = torch.einsum('bl,blh->bh', left, Gx)
                left = y
            return {"output": left}

        # === OPTIMIZED PARALLEL SCAN ===

        # Update cores_pad
        with torch.no_grad():
            self.cores_pad.zero_()
            for k in range(L):
                chi_l = self.cores[k].shape[0]
                chi_r = self.cores[k].shape[2]
                self.cores_pad[k, :chi_l, :, :chi_r].copy_(self.cores[k])

        # Build transfer matrices (single einsum)
        T = torch.einsum('lhdm,bld->blhm', self.cores_pad, H)

        # Ensure contiguous for better bmm performance
        T = T.contiguous()

        # Optimized scan
        left_states, right_states = block_prefix_scan_optimized(
            T=T,
            evec=self.evec,
            eye=self._eye,
            tile=self.tile_size,
        )

        prefix_pad = left_states[:, :L, :]
        suffix_pad = right_states[:, 1:, :]

        return {
            "prefix_pad": prefix_pad,
            "suffix_pad": suffix_pad,
            "cores_pad": self.cores_pad,
        }

    def get_bond_dims(self) -> List[int]:
        """Return bond dimensions."""
        return self._bond_dims

    def count_parameters(self) -> int:
        """Count total complex parameters."""
        return sum(p.numel() for p in self.cores)


__all__ = ['MPSMixerOptimized', 'parallel_prefix_scan_batched']
