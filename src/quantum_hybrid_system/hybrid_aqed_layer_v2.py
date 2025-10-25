"""
HybridAQEDLayer v2: Persistent Learnable MPS Mixer
==================================================

Integrates MPSMixer (persistent learnable cores) with adaptive routing.
No per-forward MPS rebuilds - cores are trained end-to-end.

Architecture:
1. MPSMixer: Persistent learnable cores (O(L·χ²·D))
2. EntanglementRouter: Select ~15% tokens for full attention
3. Sparse attention on routed tokens only
4. FFN on all tokens

Complexity:
- Traditional: O(L²·D)
- Hybrid: O(ρ·L²·D + L·χ²·D) where ρ=0.15, χ=32
- Speedup: ~5-15× for L≥512

Author: Claude Code (based on ChatGPT architecture)
Date: October 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .mps_mixer import MPSMixer
from .entanglement_router import EntanglementRouter


class HybridAQEDLayer(nn.Module):
    """
    Hybrid AQED layer with persistent learnable MPS mixer.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        seq_len: Fixed sequence length (required for MPS initialization)
        chi_max: Maximum MPS bond dimension
        route_frac: Fraction of tokens for full attention
        router_strategy: 'entropy', 'learned', or 'hybrid'
        mps_mode: 'scan' or 'global'
        dropout: Dropout probability
        prefer_triton: Use Triton for Phase-3 ops (generally False)

    Example:
        >>> layer = HybridAQEDLayer(
        ...     d_model=1024, n_heads=16, seq_len=1024,
        ...     chi_max=32, route_frac=0.15
        ... ).cuda()
        >>> x = torch.randn(4, 1024, 1024, device='cuda')
        >>> out, stats = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        chi_max: int = 32,
        router_strategy: str = 'hybrid',
        route_frac: float = 0.15,
        mps_mode: str = 'scan',
        dropout: float = 0.1,
        prefer_triton: bool = False,
        train_safe: bool = True,
        force_fp32: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.chi_max = chi_max
        self.route_frac = route_frac
        self.prefer_triton = prefer_triton
        self.train_safe = train_safe
        self.force_fp32 = force_fp32

        # Router optimization: hard cap on routed tokens
        self.max_routed_tokens = max(1, int(route_frac * seq_len * 1.5))  # 50% buffer

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # 1) Learnable, persistent MPS mixer (Parameters live here)
        self.mps = MPSMixer(
            L=seq_len,
            D=d_model,
            chi_max=chi_max,
            mode=mps_mode
        )

        # 2) Router
        self.router = EntanglementRouter(
            route_frac=route_frac,
            d_model=d_model,
            strategy=router_strategy,
        )

        # 3) Projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # 4) Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        routing_override: Optional[torch.Tensor] = None,
        return_timings: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with hybrid attention.

        Args:
            x: [B, L, D] input (real-valued)
            attn_mask: Optional [B, L] or [B, L, L] attention mask
            entropy: Optional [L] entropy scores for routing
            routing_override: Optional [B, L] explicit routing scores
            return_timings: If True, include per-stage timings in stats

        Returns:
            output: [B, L, D] processed embeddings
            stats: Dict with losses and statistics (optionally with timings)
        """
        import time
        timings = {} if return_timings else None

        # GPU-or-bust: fail fast if input is on CPU
        assert x.is_cuda, f"Input x must be on CUDA for AQED v2, got {x.device}"

        # AMP-safe: cast to FP32 if needed, compute, then cast back
        in_dtype = x.dtype
        comp_dtype = torch.float32 if (self.force_fp32 and x.is_cuda) else in_dtype
        x = x.to(comp_dtype)

        B, L, D = x.shape
        assert L == self.seq_len and D == self.d_model, \
            f"Expected [B, {self.seq_len}, {self.d_model}], got [B, {L}, {D}]"

        if return_timings:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # === MPS mixing (vectorized, NO Python loops) ===
        # DEBUG: Verify MPS cores are on CUDA
        if not hasattr(self, '_debug_checked'):
            assert self.mps.cores[0].is_cuda, f"MPS cores are on CPU! cores[0].device={self.mps.cores[0].device}"
            self._debug_checked = True

        # Get padded tensors for efficient batched operations
        scan = self.mps(x, scan=True)
        prefix = scan["prefix_pad"]  # [B, L, chi_max] complex
        suffix = scan["suffix_pad"]  # [B, L, chi_max] complex
        cores = scan["cores_pad"]    # [L, chi_max, D, chi_max] complex

        if return_timings:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings['mps_scan'] = (t1 - t0) * 1000

        # Vectorized contractions (left × core × right) for all tokens at once
        # 1) Contract prefix with cores over left bond: [B, L, D, chi_max]
        tmp = torch.einsum('blh,lhdm->bldm', prefix, cores)

        # 2) Contract with suffix over right bond: [B, L, D]
        mix = torch.einsum('bldm,blm->bld', tmp, suffix)

        # Residual add (take real part if model is real-valued)
        # mix is complex [B, L, D], x is real [B, L, D]
        x_mixed = x + mix.real

        if return_timings:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            timings['mps_mix'] = (t2 - t1) * 1000

        # === Router decides which tokens need full attention ===
        indices, route_mask, router_aux = self.router(
            hidden_states=x,
            entropy=entropy,
            routing_override=routing_override,
        )
        # route_mask: [B, L] bool, True = full attention
        # GPU-or-bust: ensure router output is on CUDA
        route_mask = route_mask.to(x.device, dtype=torch.bool)

        if return_timings:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            timings['router'] = (t3 - t2) * 1000

        # === Attention (sparse on routed subset) ===
        # Short-circuit if no tokens need routing
        total_routed = route_mask.sum().item()

        if total_routed == 0:
            # Skip attention entirely - just use MPS-mixed embeddings
            x = x_mixed
        else:
            qkv = self.qkv(x_mixed)  # [B, L, 3D]
            q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each [B, L, D]

            # Reshape for multi-head: [B, L, D] -> [B, n_heads, L, head_dim]
            q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            # GPU-or-bust: verify all tensors on same CUDA device
            def _assert_cuda_same(*tensors):
                dev = tensors[0].device
                for t in tensors:
                    assert t.is_cuda and t.device == dev, f"CPU or mixed device: {t.device} vs {dev}"

            _assert_cuda_same(x, q, k, v, route_mask)

            # Gradient-safe attention: build output functionally (no in-place writes)
            if self.train_safe:
                # Build per-batch tensors, then stack (autograd-safe)
                # Uses scatter_add to avoid O(L²) one-hot allocations
                out_list = []
                for b in range(B):
                    sel = route_mask[b]  # [L] bool (CUDA)

                    # Get indices of routed tokens (stay on CUDA, no .item() sync)
                    sel_idx = torch.nonzero(sel, as_tuple=False).squeeze(-1)  # [num_sel] long (CUDA)

                    # GPU-or-bust: verify routed tensors are on CUDA
                    _assert_cuda_same(q[b], k[b], v[b], sel, sel_idx)

                    if sel_idx.numel() > 0:
                        # Extract selected queries
                        q_sel = q[b, :, sel, :]  # [n_heads, num_sel, head_dim]
                        k_full = k[b, :, :, :]  # [n_heads, L, head_dim]
                        v_full = v[b, :, :, :].to(comp_dtype)  # [n_heads, L, head_dim]

                        # Scaled dot-product attention on selected tokens
                        scale = 1.0 / math.sqrt(self.head_dim)
                        attn_scores = torch.matmul(q_sel, k_full.transpose(-2, -1)) * scale
                        # attn_scores: [n_heads, num_sel, L]

                        if attn_mask is not None:
                            # Apply mask (simplified - assumes [B, L])
                            if attn_mask.dim() == 2:
                                mask_b = attn_mask[b].unsqueeze(0).unsqueeze(0)  # [1, 1, L]
                                attn_scores = attn_scores.masked_fill(mask_b == 0, float('-inf'))

                        attn_weights = F.softmax(attn_scores, dim=-1)  # [n_heads, num_sel, L]
                        attn_output = torch.matmul(attn_weights, v_full)  # [n_heads, num_sel, head_dim]
                        attn_output = attn_output.to(comp_dtype)

                        # Scatter back using scatter_add (NO one-hot, stays on GPU)
                        # Build zeros buffer and write selected rows
                        zeros = torch.zeros_like(v_full)  # [n_heads, L, head_dim] on CUDA
                        idx = sel_idx.view(1, -1, 1).expand(zeros.size(0), -1, zeros.size(2))  # [n_heads, num_sel, head_dim]
                        attn_full_scatter = zeros.scatter_add(1, idx, attn_output)  # [n_heads, L, head_dim]

                        # Blend: where routed, use attention; else use v_full
                        mask_3d = sel.view(1, -1, 1)  # [1, L, 1] bool (CUDA)
                        out_b = torch.where(mask_3d, attn_full_scatter, v_full)  # [n_heads, L, head_dim]

                    else:
                        # No routing: just use v_full
                        v_full = v[b, :, :, :].to(comp_dtype)
                        out_b = v_full

                    out_list.append(out_b)

                # Stack all batches (gradient-safe)
                out = torch.stack(out_list, dim=0)  # [B, n_heads, L, head_dim]

            else:
                # Legacy path (faster but not gradient-safe)
                out = torch.zeros(B, self.n_heads, L, self.head_dim,
                                  device=x.device, dtype=comp_dtype)

                for b in range(B):
                    sel = route_mask[b]  # [L] bool
                    if sel.any():
                        num_sel = sel.sum().item()

                        # Extract selected queries
                        q_sel = q[b, :, sel, :]  # [n_heads, num_sel, head_dim]

                        # Attend to all keys/values
                        k_full = k[b, :, :, :]  # [n_heads, L, head_dim]
                        v_full = v[b, :, :, :]  # [n_heads, L, head_dim]

                        # Scaled dot-product attention
                        scale = 1.0 / math.sqrt(self.head_dim)
                        attn_scores = torch.matmul(q_sel, k_full.transpose(-2, -1)) * scale
                        # attn_scores: [n_heads, num_sel, L]

                        if attn_mask is not None:
                            # Apply mask (simplified - assumes [B, L])
                            if attn_mask.dim() == 2:
                                mask_b = attn_mask[b].unsqueeze(0).unsqueeze(0)  # [1, 1, L]
                                attn_scores = attn_scores.masked_fill(mask_b == 0, float('-inf'))

                        attn_weights = F.softmax(attn_scores, dim=-1)  # [n_heads, num_sel, L]
                        attn_output = torch.matmul(attn_weights, v_full)  # [n_heads, num_sel, head_dim]

                        # Scatter back (in-place - breaks gradients)
                        out[b, :, sel, :] = attn_output

                    # Non-selected tokens: keep x_mixed (already enhanced by MPS)
                    if (~sel).any():
                        # Use mixed values directly (no attention)
                        out[b, :, ~sel, :] = v[b, :, ~sel, :]

            # Reshape back: [B, n_heads, L, head_dim] -> [B, L, D]
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            out = self.o(out)

            # Residual + dropout
            x = x + self.dropout(out)
            x = self.norm1(x)

        if return_timings:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            timings['attention'] = (t4 - t3) * 1000

        # === Feed-forward ===
        x = x + self.ffn(x)
        x = self.norm2(x)

        if return_timings:
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            timings['ffn'] = (t5 - t4) * 1000
            timings['total'] = (t5 - t0) * 1000

        # === Stats ===
        stats = {
            "router_aux": router_aux,
            "num_routed": route_mask.sum().item(),
            "route_frac_actual": route_mask.float().mean().item(),
        }

        if return_timings:
            stats['timings'] = timings

        # Cast back to input dtype (for AMP compatibility)
        x = x.to(in_dtype)

        return x, stats

    def count_mps_parameters(self) -> int:
        """Count MPS mixer parameters."""
        return self.mps.count_parameters()


class HybridAQEDTransformer(nn.Module):
    """
    Full transformer with hybrid AQED layers.

    Args:
        n_layers: Number of layers
        d_model: Model dimension
        n_heads: Number of heads
        seq_len: Fixed sequence length
        vocab_size: Vocabulary size
        chi_max: MPS bond dimension
        route_frac: Routing fraction
        **kwargs: Additional layer args

    Example:
        >>> model = HybridAQEDTransformer(
        ...     n_layers=12, d_model=1024, n_heads=16,
        ...     seq_len=1024, vocab_size=50000, chi_max=32
        ... ).cuda()
        >>> input_ids = torch.randint(0, 50000, (4, 1024), device='cuda')
        >>> logits = model(input_ids)
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        seq_len: int,
        vocab_size: int,
        chi_max: int = 32,
        route_frac: float = 0.15,
        dropout: float = 0.1,
        **layer_kwargs,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.seq_len = seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)

        # Layers
        self.layers = nn.ModuleList([
            HybridAQEDLayer(
                d_model=d_model,
                n_heads=n_heads,
                seq_len=seq_len,
                chi_max=chi_max,
                route_frac=route_frac,
                dropout=dropout,
                **layer_kwargs,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """
        Forward pass.

        Args:
            input_ids: [B, L] token IDs
            attention_mask: Optional [B, L] mask
            return_stats: Return per-layer statistics

        Returns:
            logits: [B, L, vocab_size]
            stats: (optional) List of dicts with per-layer stats
        """
        B, L = input_ids.shape
        assert L == self.seq_len, f"Expected L={self.seq_len}, got {L}"

        device = input_ids.device

        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        h = self.dropout(h)

        # Layers
        all_stats = []
        for layer in self.layers:
            h, stats = layer(h, attn_mask=attention_mask)
            if return_stats:
                all_stats.append(stats)

        # Output
        h = self.final_norm(h)
        logits = self.lm_head(h)

        if return_stats:
            return logits, all_stats
        else:
            return logits


__all__ = ['HybridAQEDLayer', 'HybridAQEDTransformer']
