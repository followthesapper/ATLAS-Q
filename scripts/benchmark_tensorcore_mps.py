#!/usr/bin/env python3
"""
Benchmark: MPS Mixer Tensor Core Acceleration
=============================================

Compare original complex operations vs Tensor Core accelerated version.

Expected: 1.5-2.5× speedup with real 2×2 blocks + FP16/BF16.
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum_hybrid_system.mps_mixer import MPSMixer
from src.quantum_hybrid_system.mps_mixer_tensorcore import MPSMixerTensorCore


def benchmark_mps_mixer(
    mixer_class,
    L: int,
    D: int,
    chi_max: int,
    batch_size: int = 4,
    n_warmup: int = 20,
    n_trials: int = 50,
    **kwargs
):
    """Benchmark a single MPS mixer configuration."""
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
        times.append((t1 - t0) * 1000)  # ms

    median_time = sorted(times)[len(times) // 2]
    return median_time


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    print("=" * 80)
    print("MPS Mixer Tensor Core Acceleration Benchmark")
    print("=" * 80)
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

        # Original (complex operations)
        print("  Benchmarking original (complex ops)...")
        time_original = benchmark_mps_mixer(
            MPSMixer,
            L=L,
            D=D,
            chi_max=chi_max,
            batch_size=batch_size,
            n_warmup=20,
            n_trials=50,
        )
        print(f"    Time: {time_original:.2f} ms")

        # Tensor Core (real 2×2 blocks + AMP)
        print("  Benchmarking Tensor Core (real 2×2 + AMP)...")
        time_tensorcore = benchmark_mps_mixer(
            MPSMixerTensorCore,
            L=L,
            D=D,
            chi_max=chi_max,
            batch_size=batch_size,
            use_amp=True,
            n_warmup=20,
            n_trials=50,
        )
        print(f"    Time: {time_tensorcore:.2f} ms")

        speedup = time_original / time_tensorcore
        print(f"    Speedup: {speedup:.2f}×")
        print()

        results.append({
            "L": L,
            "D": D,
            "chi_max": chi_max,
            "batch_size": batch_size,
            "time_original": time_original,
            "time_tensorcore": time_tensorcore,
            "speedup": speedup,
        })

    # Summary table
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'L':<6} {'D':<6} {'chi':<5} {'Original (ms)':<15} {'TensorCore (ms)':<18} {'Speedup':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['L']:<6} {r['D']:<6} {r['chi_max']:<5} "
              f"{r['time_original']:>13.2f}   {r['time_tensorcore']:>15.2f}   "
              f"{r['speedup']:>8.2f}×")

    print()
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}×")
    print()

    # Check if we hit target
    if avg_speedup >= 1.5:
        print(f"✅ SUCCESS: Achieved {avg_speedup:.2f}× speedup (target: 1.5-2.5×)")
    else:
        print(f"⚠️  WARNING: Only {avg_speedup:.2f}× speedup (target: 1.5-2.5×)")
        print("   Consider tuning tile size or checking Tensor Core utilization.")

    print()


if __name__ == "__main__":
    main()
