"""
Minimal Benchmark: HybridAQEDLayer v2 vs Full Attention
========================================================

Compares persistent learnable MPS mixer against standard transformer.

Usage:
    python scripts/benchmark_hybrid_aqed_v2.py

Author: Claude Code (Surgical Upgrades)
Date: October 24, 2025
"""

import torch
import torch.nn as nn
import time
from typing import Dict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum_hybrid_system.hybrid_aqed_layer_v2 import HybridAQEDLayer


class FullAttentionLayer(nn.Module):
    """Standard transformer layer for baseline comparison."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full attention on all tokens."""
        B, L, D = x.shape

        # QKV
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        out = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o(out)

        # Residual + norm
        x = x + self.dropout(out)
        x = self.norm1(x)

        # FFN
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x


def benchmark_layer(
    layer: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 5,
    num_trials: int = 20,
) -> Dict[str, float]:
    """
    Benchmark a layer's forward pass.

    Args:
        layer: The layer to benchmark
        x: Input tensor
        num_warmup: Warmup iterations
        num_trials: Measurement iterations

    Returns:
        dict with timing statistics
    """
    layer.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(layer, HybridAQEDLayer):
                _ = layer(x)
            else:
                _ = layer(x)

    # Synchronize
    if x.device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start = time.perf_counter()

            if isinstance(layer, HybridAQEDLayer):
                _ = layer(x)
            else:
                _ = layer(x)

            if x.device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
    }


def measure_memory(layer: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """Measure memory usage."""
    if x.device.type != 'cuda':
        return {'allocated_mb': 0, 'reserved_mb': 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        if isinstance(layer, HybridAQEDLayer):
            _ = layer(x)
        else:
            _ = layer(x)

    allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 2

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
    }


def main():
    print('=' * 70)
    print('Minimal Benchmark: HybridAQEDLayer v2 vs Full Attention')
    print('=' * 70)
    print()

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    # Benchmark configurations
    configs = [
        {'L': 128, 'd_model': 256, 'n_heads': 4, 'chi_max': 16},
        {'L': 256, 'd_model': 512, 'n_heads': 8, 'chi_max': 32},
        {'L': 512, 'd_model': 512, 'n_heads': 8, 'chi_max': 32},
        {'L': 1024, 'd_model': 1024, 'n_heads': 16, 'chi_max': 32},
    ]

    results = []

    for cfg in configs:
        L = cfg['L']
        d_model = cfg['d_model']
        n_heads = cfg['n_heads']
        chi_max = cfg['chi_max']

        print(f'Configuration: L={L}, d_model={d_model}, n_heads={n_heads}, chi_max={chi_max}')
        print('-' * 70)

        # Create layers
        full_attn = FullAttentionLayer(d_model, n_heads).to(device)
        hybrid_aqed = HybridAQEDLayer(
            d_model=d_model,
            n_heads=n_heads,
            seq_len=L,
            chi_max=chi_max,
            route_frac=0.15,
        ).to(device)

        # Input
        batch_size = 4
        x = torch.randn(batch_size, L, d_model, device=device)

        # Benchmark full attention
        print('  Benchmarking Full Attention...')
        full_times = benchmark_layer(full_attn, x)
        full_memory = measure_memory(full_attn, x)

        # Benchmark hybrid AQED
        print('  Benchmarking Hybrid AQED v2...')
        hybrid_times = benchmark_layer(hybrid_aqed, x)
        hybrid_memory = measure_memory(hybrid_aqed, x)

        # Calculate speedup
        speedup = full_times['mean_ms'] / hybrid_times['mean_ms']
        memory_ratio = full_memory['allocated_mb'] / max(hybrid_memory['allocated_mb'], 1)

        # Display results
        print()
        print(f'  Full Attention:    {full_times["mean_ms"]:.2f} ± {full_times["std_ms"]:.2f} ms')
        print(f'  Hybrid AQED v2:    {hybrid_times["mean_ms"]:.2f} ± {hybrid_times["std_ms"]:.2f} ms')
        print(f'  Speedup:           {speedup:.2f}×')
        print()
        print(f'  Memory (Full):     {full_memory["allocated_mb"]:.1f} MB')
        print(f'  Memory (Hybrid):   {hybrid_memory["allocated_mb"]:.1f} MB')
        print(f'  Memory Reduction:  {memory_ratio:.2f}×')
        print()

        results.append({
            'L': L,
            'd_model': d_model,
            'speedup': speedup,
            'full_ms': full_times['mean_ms'],
            'hybrid_ms': hybrid_times['mean_ms'],
            'memory_ratio': memory_ratio,
        })

    # Summary
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    print()
    print(f'{'L':>6} | {'d_model':>8} | {'Full (ms)':>10} | {'Hybrid (ms)':>11} | {'Speedup':>8}')
    print('-' * 70)
    for r in results:
        print(f'{r["L"]:>6} | {r["d_model"]:>8} | {r["full_ms"]:>10.2f} | {r["hybrid_ms"]:>11.2f} | {r["speedup"]:>7.2f}×')

    print()
    print('=' * 70)
    print('Key Findings:')
    max_speedup = max(r['speedup'] for r in results)
    print(f'  - Maximum speedup: {max_speedup:.2f}× at L={[r for r in results if r["speedup"] == max_speedup][0]["L"]}')
    print(f'  - Hybrid AQED uses persistent learnable MPS cores (no rebuilds)')
    print(f'  - Only ~15% tokens receive full attention (sparse routing)')
    print(f'  - Backend: PyTorch/cuBLAS for MPS ops (Phase-3 policy)')
    print('=' * 70)


if __name__ == '__main__':
    main()
