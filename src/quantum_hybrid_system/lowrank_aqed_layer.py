# src/quantum_hybrid_system/lowrank_aqed_layer.py
"""
Real-valued, GPU-native AQED layer using Linformer-style low-rank projections
with Triton-fused kernels for maximum performance.

Key differences from AQED v2 (complex MPS):
- 100% real-valued operations (no complex arithmetic overhead)
- GPU-native matmul/bmm operations (cuBLAS optimized)
- Triton-fused sequence projection (reduces L → r)
- Compatible with torch.compile
- No .item() calls or graph breaks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton-fused projection
try:
    from triton_kernels.linproj_bmm import linproj_bmm_fused
    _HAS_TRITON_LINPROJ = True
except Exception:
    _HAS_TRITON_LINPROJ = False


class LowRankAQEDLayer(nn.Module):
    """
    Real-valued, GPU-native AQED layer with compile-friendly static-shape routing:
      - Router picks a subset of tokens (fraction route_frac) for full attention
      - Routed tokens: full O(L²) attention via SDPA
      - Non-routed tokens: low-rank O(Lr) attention via Linformer-style projection
      - Blending via torch.where (static shapes, no scatter)

    Shapes: B=batch, L=seq_len, D=d_model, H=n_heads, Dh=D/H, r=rank
    """
    def __init__(self, d_model, n_heads, seq_len, rank=64, route_frac=0.10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.rank = rank
        self.route_frac = route_frac
        self.d_head = d_model // n_heads

        # Head projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o   = nn.Linear(d_model, d_model, bias=False)

        # Linformer-style projections (shared across heads)
        # E: [L, r] reduces the sequence dimension; initialized to truncated identity
        E0 = self._init_E(seq_len, rank)  # [L, r]
        self.E = nn.Parameter(E0)  # trainable
        # Cache E^T [r, L] to avoid repeated transpose in forward
        self.register_buffer("E_T_buf", E0.t().contiguous(), persistent=True)

        # Freeze E for benchmarking (stabilizes projection, reduces grads/reductions)
        # Unfreeze later if needed: self.E.requires_grad_(True)
        self.E.requires_grad_(False)

        # Simple router: single linear projection to scalar scores
        self.router_mlp = nn.Linear(d_model, 1, bias=False)

        self.dropout = dropout
        self.ln = nn.LayerNorm(d_model)

    @staticmethod
    def _init_E(L, r):
        """Initialize projection matrix as truncated identity for stability"""
        E = torch.zeros(L, r)
        rows = min(L, r)
        E[:rows, :rows] = torch.eye(rows)
        return E

    @torch.no_grad()
    def warmup_triton(self, device='cuda', dtype=torch.float16):
        """Warmup Triton kernel with a fake batch to avoid autotuning during first real step"""
        if not _HAS_TRITON_LINPROJ:
            return
        # Create fake inputs matching typical shapes
        B, H = 2, self.n_heads
        fake_k = torch.randn(B, H, self.seq_len, self.d_head, device=device, dtype=dtype)
        fake_v = torch.randn(B, H, self.seq_len, self.d_head, device=device, dtype=dtype)
        # Run once to trigger compilation/autotuning
        _ = linproj_bmm_fused(self.E.to(device), fake_k, fake_v)
        torch.cuda.synchronize()  # Wait for kernel compilation to finish

    def forward(self, x):
        """
        x: [B, L, D] real, on CUDA
        returns y: [B, L, D]

        Static-shape routing (compile-friendly):
        - Compute both full-attn and low-rank-attn paths
        - Blend via torch.where based on routing mask
        - No nonzero(), no scatter, no graph breaks
        """
        B, L, D = x.shape
        assert L == self.seq_len and D == self.d_model
        assert x.is_cuda, f"Input must be on CUDA, got {x.device}"

        x_normed = self.ln(x)

        # QKV projection
        qkv = self.qkv(x_normed).view(B, L, 3, self.n_heads, self.d_head)  # [B, L, 3, H, Dh]
        q = qkv[:, :, 0].transpose(1, 2)  # [B, L, H, Dh] -> [B, H, L, Dh]
        k = qkv[:, :, 1].transpose(1, 2)  # [B, L, H, Dh] -> [B, H, L, Dh]
        v = qkv[:, :, 2].transpose(1, 2)  # [B, L, H, Dh] -> [B, H, L, Dh]

        B, H, L, Dh = q.shape
        r = self.rank

        # 1) Linformer-style sequence projection (L -> r) for K,V
        #    Use compile-friendly PyTorch path when compiling, Triton otherwise
        try:
            is_compiling = torch._dynamo.is_compiling()
        except:
            is_compiling = False

        use_triton = (_HAS_TRITON_LINPROJ and
                     not is_compiling and
                     x.dtype in (torch.float16, torch.bfloat16, torch.float32))

        # Make K,V contiguous before projection (better kernel fusion)
        k = k.contiguous()
        v = v.contiguous()

        if use_triton:
            # Triton fused path (faster in eager mode)
            k_proj, v_proj = linproj_bmm_fused(self.E.to(x.device), k, v)
        else:
            # Pure PyTorch path (compile-friendly, Inductor can fuse)
            # Use cached E_T to avoid repeated transpose
            # Update buffer if E changed dtype/device (training with different precision)
            if self.E_T_buf.device != self.E.device or self.E_T_buf.dtype != self.E.dtype:
                self.E_T_buf.data = self.E.t().contiguous()
            E_T = self.E_T_buf  # [r, L]
            # Vectorized matmul: (r,L) @ (L,Dh) -> (r,Dh) for each [B,H]
            k_2d = k.reshape(B*H, L, Dh)
            v_2d = v.reshape(B*H, L, Dh)
            k_proj = (E_T @ k_2d).reshape(B, H, r, Dh)  # [B,H,r,Dh]
            v_proj = (E_T @ v_2d).reshape(B, H, r, Dh)  # [B,H,r,Dh]

        # Make Q and projected K,V contiguous before SDPA (better SDPA kernel selection)
        q = q.contiguous()
        k_proj = k_proj.contiguous()
        v_proj = v_proj.contiguous()

        # 2) Static-shape routing (NO nonzero, NO .item())
        #    Use kthvalue instead of topk (faster threshold selection)
        scores = self.router_mlp(x_normed).squeeze(-1)  # [B, L]
        k_routes = max(1, int(self.route_frac * L))
        # kthvalue is faster than topk for single threshold
        thr, _ = torch.kthvalue(scores, L - k_routes + 1, dim=1, keepdim=True)  # [B, 1]
        route_mask = scores >= thr  # [B, L] bool, static shape

        # 3) Compute BOTH attention paths (static shapes)
        # Use AMP for TF32/fp16 acceleration on Ampere+ GPUs
        from torch.cuda.amp import autocast
        with autocast(dtype=torch.float16):
            # Full attention (L x L) via SDPA
            attn_full = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # [B, H, L, Dh]

            # Low-rank attention (L x r) via SDPA
            attn_lr = F.scaled_dot_product_attention(
                q, k_proj, v_proj,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # [B, H, L, Dh]

        # 4) Blend without scatter (compile-friendly!)
        # Use blended add/mul instead of where for better Inductor fusion
        m = route_mask.view(B, 1, L, 1).to(attn_full.dtype)  # [B,1,L,1] cast to fp16/fp32
        y = attn_lr + m * (attn_full - attn_lr)  # [B, H, L, Dh] - same math as where, better fusion

        # 5) Output projection
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        y = self.o(y)
        return y


__all__ = ['LowRankAQEDLayer']
