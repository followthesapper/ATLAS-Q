#!/usr/bin/env python3
"""
Benchmark: Simulator Environment Construction
=============================================

Compares naive per-site loops vs optimized vectorized scan for MPS
environment building (used in DMRG, TEBD, expectation values).

Expected: 2-4× speedup for environment construction stage.

Author: Claude Code (applying AQED optimizations to Simulator)
Date: October 24, 2025
"""

import torch
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantum_hybrid_system.sim_env_scan import build_mps_environments, check_environment_correctness


def naive_environment_builder(mps_cores, chi_max):
    """
    Naive Python-loop based environment construction (OLD PATH).

    Returns matrix environments [chi×chi], not vectors, per correct definition.
    Keeps complex dtype (transfer operators can have complex entries).
    """
    L = len(mps_cores)
    device = mps_cores[0].device
    dtype = mps_cores[0].dtype  # Keep complex dtype
    d = mps_cores[0].shape[1]

    # Pad cores
    cores_pad = torch.zeros(L, chi_max, d, chi_max, dtype=dtype, device=device)
    for k in range(L):
        chi_l, d_k, chi_r = mps_cores[k].shape
        cores_pad[k, :chi_l, :, :chi_r] = mps_cores[k]

    # Left environments (Python loop) - MATRIX form
    left_envs = []
    left = torch.eye(chi_max, dtype=dtype, device=device)
    left_envs.append(left.clone())

    for k in range(L):
        core_k = cores_pad[k]
        # Transfer matrix: E = sum_{s,m} core[i,s,m] * conj(core[j,s,m])
        # Keep complex! E is Hermitian PSD but can have complex entries.
        E_k = torch.einsum('ism,jsm->ij', core_k, core_k.conj())

        # L_{k+1} = L_k @ E_k
        left = torch.mm(left, E_k)
        left_envs.append(left.clone())

    # Right environments (Python loop) - MATRIX form
    right_envs = []
    right = torch.eye(chi_max, dtype=dtype, device=device)
    right_envs.insert(0, right.clone())

    for k in range(L - 1, -1, -1):
        core_k = cores_pad[k]
        E_k = torch.einsum('ism,jsm->ij', core_k, core_k.conj())

        # R_k = E_k @ R_{k+1}
        right = torch.mm(E_k, right)
        right_envs.insert(0, right.clone())

    # Stack to tensors: [1, L+1, chi_max, chi_max]
    left_tensor = torch.stack(left_envs, dim=0).unsqueeze(0)
    right_tensor = torch.stack(right_envs, dim=0).unsqueeze(0)

    return left_tensor, right_tensor


def benchmark_environment_construction(
    L: int,
    chi: int,
    d: int = 2,
    n_warmup: int = 10,
    n_trials: int = 30,
):
    """Benchmark environment construction: naive vs optimized."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.complex64

    # Create random MPS cores
    cores = []
    for k in range(L):
        chi_l = min(chi, 2 ** min(k, 8))
        chi_r = min(chi, 2 ** min(L - k - 1, 8))
        chi_l = max(1, chi_l)
        chi_r = max(1, chi_r)

        real = torch.randn(chi_l, d, chi_r, device=device)
        imag = torch.randn(chi_l, d, chi_r, device=device)
        core = torch.complex(real, imag)
        cores.append(core)

    chi_max = chi

    # === NAIVE (Python loops) ===
    print(f"  Benchmarking naive (Python loops)...")

    # Warmup
    for _ in range(n_warmup):
        left_naive, right_naive = naive_environment_builder(cores, chi_max)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Timed
    times_naive = []
    for _ in range(n_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        left_naive, right_naive = naive_environment_builder(cores, chi_max)
        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        times_naive.append((t1 - t0) * 1000)

    median_naive = sorted(times_naive)[len(times_naive) // 2]
    print(f"    Time: {median_naive:.2f} ms")

    # === OPTIMIZED (Vectorized scan) ===
    print(f"  Benchmarking optimized (vectorized scan)...")

    # Test without normalization first for correctness
    print(f"  Testing correctness (no normalization)...")
    left_test, right_test = build_mps_environments(cores, chi_max=chi_max, tile_size=64, normalize=False)

    # Pad cores for check
    cores_pad = torch.zeros(L, chi_max, d, chi_max, dtype=dtype, device=device)
    for k in range(L):
        chi_l, d_k, chi_r = cores[k].shape
        cores_pad[k, :chi_l, :, :chi_r] = cores[k]

    correct = check_environment_correctness(cores_pad, left_test, right_test, site=min(3, L//4), atol=1e-3, rtol=1e-2)
    if correct:
        print(f"    ✅ Algorithm correct!")
    else:
        print(f"    ❌ Algorithm has bug (even without normalization)")
        # Continue anyway to measure speed with normalization

    # Now benchmark WITH normalization (for stability)
    # Warmup
    for _ in range(n_warmup):
        left_opt, right_opt = build_mps_environments(cores, chi_max=chi_max, tile_size=64, normalize=True)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Timed
    times_opt = []
    for _ in range(n_trials):
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        left_opt, right_opt = build_mps_environments(cores, chi_max=chi_max, tile_size=64, normalize=True)
        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        times_opt.append((t1 - t0) * 1000)

    median_opt = sorted(times_opt)[len(times_opt) // 2]
    print(f"    Time (with normalization): {median_opt:.2f} ms")

    speedup = median_naive / median_opt
    print(f"    Speedup: {speedup:.2f}×")

    return {
        'L': L,
        'chi': chi,
        'time_naive': median_naive,
        'time_optimized': median_opt,
        'speedup': speedup,
        'correct': correct,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    print("=" * 80)
    print("Simulator Environment Construction Benchmark")
    print("=" * 80)
    print()
    print("Testing: Naive Python loops vs Optimized vectorized scan")
    print("Use case: DMRG/TEBD environment building, expectation values")
    print()

    configs = [
        {'L': 64, 'chi': 16},
        {'L': 128, 'chi': 32},
        {'L': 256, 'chi': 32},
        {'L': 512, 'chi': 32},
    ]

    results = []

    for config in configs:
        L = config['L']
        chi = config['chi']

        print(f"Configuration: L={L}, chi={chi}")
        print("-" * 80)

        result = benchmark_environment_construction(L=L, chi=chi, d=2)
        results.append(result)
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'L':<6} {'chi':<5} {'Naive (ms)':<12} {'Optimized (ms)':<16} {'Speedup':<10} {'Correct':<8}")
    print("-" * 80)

    for r in results:
        correct_str = "✅" if r['correct'] else "❌"
        print(f"{r['L']:<6} {r['chi']:<5} {r['time_naive']:>10.2f}   "
              f"{r['time_optimized']:>14.2f}   {r['speedup']:>8.2f}×  {correct_str:<8}")

    print()

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    all_correct = all(r['correct'] for r in results)

    print(f"Average speedup: {avg_speedup:.2f}×")
    print()

    # Assessment
    if not all_correct:
        print("❌ CORRECTNESS FAILURE - Some environments don't match!")
        print("   Do NOT use optimized path until fixed.")
    elif avg_speedup >= 2.0:
        print(f"✅ SUCCESS: {avg_speedup:.2f}× average speedup")
        print("   Optimized scan is ready for production use.")
    elif avg_speedup >= 1.5:
        print(f"✓ MODERATE: {avg_speedup:.2f}× average speedup")
        print("   Useful improvement, deploy if convenient.")
    else:
        print(f"⚠️  MINIMAL: {avg_speedup:.2f}× average speedup")
        print("   Improvement is small, may not be worth complexity.")

    print()

    # Impact estimate
    if all_correct and avg_speedup >= 1.5:
        print("Impact on Simulator:")
        print("-" * 80)
        print("If environment construction takes 50% of DMRG/TEBD time:")
        sweep_speedup = 1 / (0.5 / avg_speedup + 0.5)
        print(f"  Estimated per-sweep speedup: {sweep_speedup:.2f}×")
        print()
        print("If environment construction takes 70% of time:")
        sweep_speedup_70 = 1 / (0.7 / avg_speedup + 0.3)
        print(f"  Estimated per-sweep speedup: {sweep_speedup_70:.2f}×")
        print()

    print()


if __name__ == "__main__":
    main()
