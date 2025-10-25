#!/usr/bin/env python3
"""
Profile AQED LowRank to identify kernel bottlenecks causing jagged GPU usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule
import sys
import os

# Add paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'transformers')

# Import lowrank layer components
from quantum_hybrid_system.lowrank_aqed_layer import LowRankAQEDLayer

# Try to import Triton-fused projection
try:
    from triton_kernels.linproj_bmm import linproj_bmm_fused
    _HAS_TRITON_LINPROJ = True
except Exception:
    _HAS_TRITON_LINPROJ = False

# GB10 workaround
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Force Flash SDPA path
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)

class LowRankAQEDLayerNoAutocast(nn.Module):
    """
    Copy of LowRankAQEDLayer but without autocast for profiling
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

        # Linformer projection
        E0 = self._init_E(seq_len, rank)
        self.E = nn.Parameter(E0)
        self.register_buffer("E_T_buf", E0.t().contiguous(), persistent=True)
        self.E.requires_grad_(False)

        # Simple router
        self.router_mlp = nn.Linear(d_model, 1, bias=False)

        self.dropout = dropout
        self.ln = nn.LayerNorm(d_model)

    @staticmethod
    def _init_E(L, r):
        E = torch.zeros(L, r)
        rows = min(L, r)
        E[:rows, :rows] = torch.eye(rows)
        return E

    @torch.no_grad()
    def warmup_triton(self, device='cuda', dtype=torch.float32):
        if not _HAS_TRITON_LINPROJ:
            return
        B, H = 2, self.n_heads
        fake_k = torch.randn(B, H, self.seq_len, self.d_head, device=device, dtype=dtype)
        fake_v = torch.randn(B, H, self.seq_len, self.d_head, device=device, dtype=dtype)
        _ = linproj_bmm_fused(self.E.to(device), fake_k, fake_v)
        torch.cuda.synchronize()

    def forward(self, x):
        B, L, D = x.shape
        assert L == self.seq_len and D == self.d_model

        x_normed = self.ln(x)

        # QKV projection
        qkv = self.qkv(x_normed).view(B, L, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        B, H, L, Dh = q.shape
        r = self.rank

        # Linformer projection
        try:
            is_compiling = torch._dynamo.is_compiling()
        except:
            is_compiling = False

        use_triton = (_HAS_TRITON_LINPROJ and
                     not is_compiling and
                     x.dtype in (torch.float16, torch.bfloat16, torch.float32))

        k = k.contiguous()
        v = v.contiguous()

        if use_triton:
            k_proj, v_proj = linproj_bmm_fused(self.E.to(x.device), k, v)
        else:
            if self.E_T_buf.device != self.E.device or self.E_T_buf.dtype != self.E.dtype:
                self.E_T_buf.data = self.E.t().contiguous()
            E_T = self.E_T_buf
            k_2d = k.reshape(B*H, L, Dh)
            v_2d = v.reshape(B*H, L, Dh)
            k_proj = (E_T @ k_2d).reshape(B, H, r, Dh)
            v_proj = (E_T @ v_2d).reshape(B, H, r, Dh)

        q = q.contiguous()
        k_proj = k_proj.contiguous()
        v_proj = v_proj.contiguous()

        # Routing
        scores = self.router_mlp(x_normed).squeeze(-1)
        k_routes = max(1, int(self.route_frac * L))
        thr, _ = torch.kthvalue(scores, L - k_routes + 1, dim=1, keepdim=True)
        route_mask = scores >= thr

        # BOTH attention paths (NO AUTOCAST FOR PROFILING)
        attn_full = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        attn_lr = F.scaled_dot_product_attention(
            q, k_proj, v_proj,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        # Blend
        m = route_mask.view(B, 1, L, 1).to(attn_full.dtype)
        y = attn_lr + m * (attn_full - attn_lr)

        # Output projection
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.o(y)
        return y


class SimpleLowRankModel(nn.Module):
    """Minimal model with just AQED LowRank layer for profiling"""
    def __init__(self, d_model=512, n_heads=8, seq_len=8192, rank=64, route_frac=0.10):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embed = nn.Embedding(50257, d_model)
        self.aqed = LowRankAQEDLayerNoAutocast(d_model, n_heads, seq_len, rank, route_frac)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 50257, bias=False)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.aqed(x)
        x = self.ln(x)
        return self.head(x)

def main():
    print("="*80)
    print("PROFILING AQED LOWRANK - FINDING KERNEL BOTTLENECKS")
    print("="*80)

    # Config matching test 6
    d_model = 512
    n_heads = 8
    seq_len = 8192
    batch_size = 4
    rank = 64
    route_frac = 0.10

    device = 'cuda'

    print(f"\nConfig:")
    print(f"  seq_len={seq_len}, batch_size={batch_size}")
    print(f"  d_model={d_model}, n_heads={n_heads}")
    print(f"  rank={rank}, route_frac={route_frac}")
    print(f"  (Profiling eager mode, no autocast)")

    model = SimpleLowRankModel(d_model, n_heads, seq_len, rank, route_frac).to(device)

    # Warmup Triton kernels
    print("\nWarming up Triton kernels...")
    model.aqed.warmup_triton(device=device)

    # Create dummy data
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup runs
    print("Warmup runs...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for i in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        print(f"  Warmup {i+1}/3 complete")

    torch.cuda.synchronize()

    # Profile 10 training steps
    print("\nProfiling 10 training steps...")
    print("Looking for kernel-level bottlenecks...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        schedule=schedule(wait=0, warmup=0, active=10, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace("trace_lowrank.json")
    ) as prof:
        for step in range(10):
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.mean()
            loss.backward()
            optimizer.step()
            prof.step()
            print(f"  Step {step+1}/10")

    torch.cuda.synchronize()

    # Print summary using table
    print("\n" + "="*80)
    print("TOP CUDA KERNELS BY TIME")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
        top_level_events_only=False
    ))

    print(f"\n✓ Chrome trace saved to: trace_lowrank.json")
    print("  Open in chrome://tracing to see timeline visualization")
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print("\nLook for in the chrome trace:")
    print("  1. Gaps/bubbles in GPU timeline (idle time)")
    print("  2. Many small kernel launches (high count, short duration)")
    print("  3. Synchronization points (CPU-GPU interaction)")
    print("  4. Kernel launch overhead (host-side gaps)")
    print("\nKey things to check:")
    print("  - Are SDPA kernels taking most time? (expected)")
    print("  - Are there many small matmul/bmm kernels? (overhead)")
    print("  - Is kthvalue or router_mlp slow? (routing overhead)")
    print("  - Are there memcpy/cast operations? (dtype/device movement)")

if __name__ == '__main__':
    main()
