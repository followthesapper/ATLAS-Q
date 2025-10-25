#!/usr/bin/env python3
"""
Benchmark: MPS Scan Optimizations
==================================

Compare different scan implementations:
1. Original (current baseline with doubling)
2. Optimized (better batching, minimal Python loops)

Goal: Reduce MPS scan time (currently 81.6% of total)
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum_hybrid_system.mps_mixer import MPSMixer
from src.quantum_hybrid_system.mps_mixer_optimized import MPSMixerOptimized


def benchmark_mixer(
    mixer_class,
    L: int,
    D: int,
    chi_max: int,
    batch_size: int = 4,
    n_warmup: int = 20,
    n_trials: int = 50,
    **kwargs
):
    """Benchmark a single MPS mixer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mixer = mixer_class(L=L, D=D, chi_max=chi_max, **kwargs).to(device)
    x = torch.randn(batch_size, L, D, device=device)

    # Warmup
    for _ in range(n_warmup):
        out = mixer(x, scan=True)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Timed trials
    times = []
    for _ in range(n_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = mixer(x, scan=True)
        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median_time = sorted(times)[len(times) // 2]
    return median_time


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    print("=" * 80)
    print("MPS Scan Optimization Benchmark")
    print("=" * 80)
    print()
    print("Testing optimizations to reduce scan time (currently 81.6% of total)")
    print()

    configs = [
        {"L": 256, "D": 512, "chi_max": 32, "batch_size": 4},
        {"L": 512, "D": 512, "chi_max": 32, "batch_size": 4},
        {"L": 1024, "D": 1024, "chi_max": 32, "batch_size": 2},
    ]

    results = []

    for config in configs:
        L = config["L"]
        D = config["D"]
        chi_max = config["chi_max"]
        batch_size = config["batch_size"]

        print(f"Configuration: L={L}, D={D}, chi_max={chi_max}, batch_size={batch_size}")
        print("-" * 80)

        # Original (doubling scan)
        print("  Benchmarking original (doubling scan)...")
        time_original = benchmark_mixer(
            MPSMixer,
            L=L,
            D=D,
            chi_max=chi_max,
            batch_size=batch_size,
        )
        print(f"    Time: {time_original:.2f} ms")

        # Optimized (better batching)
        print("  Benchmarking optimized (parallel scan)...")
        time_optimized = benchmark_mixer(
            MPSMixerOptimized,
            L=L,
            D=D,
            chi_max=chi_max,
            batch_size=batch_size,
            tile_size=64,
        )
        print(f"    Time: {time_optimized:.2f} ms")

        speedup = time_original / time_optimized
        saved = time_original - time_optimized
        print(f"    Speedup: {speedup:.2f}×")
        print(f"    Time saved: {saved:.2f} ms")
        print()

        results.append({
            "L": L,
            "D": D,
            "chi_max": chi_max,
            "time_original": time_original,
            "time_optimized": time_optimized,
            "speedup": speedup,
            "saved_ms": saved,
        })

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'L':<6} {'D':<6} {'chi':<5} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10} {'Saved (ms)':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['L']:<6} {r['D']:<6} {r['chi_max']:<5} "
              f"{r['time_original']:>13.2f}   {r['time_optimized']:>13.2f}   "
              f"{r['speedup']:>8.2f}×  {r['saved_ms']:>10.2f}")

    print()

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    total_saved = sum(r['saved_ms'] for r in results)

    print(f"Average speedup: {avg_speedup:.2f}×")
    print(f"Total time saved: {total_saved:.2f} ms")
    print()

    # Impact on end-to-end
    print("Impact on End-to-End Performance:")
    print("-" * 80)
    print("Scan was 81.6% of total time (76ms out of 93ms)")
    if avg_speedup > 1.0:
        # Estimated end-to-end improvement
        scan_reduction = (1 - 1/avg_speedup) * 0.816  # Fraction of scan time saved
        estimated_speedup = 1 / (1 - scan_reduction)
        print(f"Estimated end-to-end speedup: {estimated_speedup:.2f}×")
        print(f"(from reducing 81.6% of runtime by {avg_speedup:.2f}×)")
    else:
        print(f"⚠️  No improvement - optimization didn't help")

    print()

    # Assessment
    if avg_speedup >= 1.5:
        print(f"✅ SUCCESS: {avg_speedup:.2f}× speedup on scan operations")
    elif avg_speedup >= 1.1:
        print(f"✓ MODERATE: {avg_speedup:.2f}× speedup (modest improvement)")
    else:
        print(f"❌ MINIMAL: {avg_speedup:.2f}× speedup (not significant)")
        print("   May need custom CUDA kernel or different approach")

    print()


if __name__ == "__main__":
    main()
