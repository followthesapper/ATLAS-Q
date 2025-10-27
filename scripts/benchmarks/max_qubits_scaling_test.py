#!/usr/bin/env python3
"""
Moderate-Entanglement Capacity Benchmark
==========================================

Answers the question: "How many qubits of moderate entanglement can we simulate?"

This benchmark:
- Applies a brickwork circuit (H + CZ gates in alternating layers)
- Tracks entanglement entropy and bond dimension growth
- Reports the maximum n where entanglement stays in the "moderate" band
- Defines "moderate" as p95 entropy between S_MIN and S_MAX bits

Usage:
    python bench/bench_moderate_capacity.py

    # Customize parameters:
    CHI_CAP=64 EPS=1e-6 LAYERS=8 S_MIN=2.0 S_MAX=8.0 python bench/bench_moderate_capacity.py

    # Stricter error tolerance:
    EPS=1e-7 TARGET_ERR=1e-4 python bench/bench_moderate_capacity.py

    # Higher χ cap for more entanglement:
    CHI_CAP=128 S_MAX=10 python bench/bench_moderate_capacity.py

Author: ATLAS-Q Contributors
Date: October 2025
"""

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from atlas_q.adaptive_mps import AdaptiveMPS


def brickwork_layers(mps, layers=6):
    """
    Apply brickwork circuit: alternating Hadamard + CZ layers

    This creates moderate entanglement that grows with depth.
    """
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))
    n = mps.num_qubits

    for L in range(layers):
        # Hadamard layer
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)

        # Entangling layer (alternating offset)
        offset = L % 2
        for i in range(offset, n - 1, 2):
            mps.apply_two_site_gate(i, CZ)


def run_benchmark(n, chi_cap, eps, layers, device):
    """Run single benchmark for n qubits"""
    mps = AdaptiveMPS(
        num_qubits=n,
        bond_dim=min(4, chi_cap),
        eps_bond=eps,
        chi_max_per_bond=chi_cap,
        device=device
    )

    t0 = time.time()
    brickwork_layers(mps, layers=layers)
    dt = time.time() - t0

    stats = mps.stats_summary()

    return {
        "n": n,
        "layers": layers,
        "chi_cap": chi_cap,
        "eps": eps,
        "max_chi": int(stats["max_chi"]),
        "mean_chi": float(stats["mean_chi"]),
        "p95_entropy": float(stats["p95_entropy"]),
        "mean_entropy": float(stats["mean_entropy"]),
        "ops": int(stats["total_operations"]),
        "ops_per_s": stats["total_operations"] / max(dt, 1e-6),
        "mem_MB": mps.get_memory_usage() / (1024**2),
        "global_err": float(mps.global_error_bound()),
        "max_eps": float(stats["max_eps"]),
        "cuda_svd_pct": float(stats["cuda_svd_pct"]),
        "time_s": dt,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  MODERATE-ENTANGLEMENT CAPACITY BENCHMARK")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration from environment or defaults
    chi_cap = int(os.environ.get("CHI_CAP", 64))
    eps = float(os.environ.get("EPS", 1e-6))
    layers = int(os.environ.get("LAYERS", 8))

    # Define "moderate entanglement": p95 entropy in [S_MIN, S_MAX] bits
    S_MIN = float(os.environ.get("S_MIN", 2.0))
    S_MAX = float(os.environ.get("S_MAX", 8.0))
    target_err = float(os.environ.get("TARGET_ERR", 5e-4))

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  χ_cap: {chi_cap}")
    print(f"  ε_bond: {eps}")
    print(f"  Layers: {layers}")
    print(f"  Moderate entanglement band: {S_MIN} ≤ p95_entropy ≤ {S_MAX} bits")
    print(f"  Target error: {target_err}")
    print()

    # Test grid
    n_grid = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(f"{'n':<8} {'χ_cap':<8} {'p95_S':<10} {'maxχ':<8} {'err':<12} {'mem(MB)':<12} {'status'}")
    print("-" * 70)

    results = []
    for n in n_grid:
        try:
            r = run_benchmark(n, chi_cap, eps, layers, device)

            # Check if within "moderate" band
            ok_entropy = (S_MIN <= r["p95_entropy"] <= S_MAX)
            ok_err = (r["global_err"] <= target_err)
            ok_cap = (r["max_chi"] < chi_cap)  # Not saturated

            r["moderate_ok"] = bool(ok_entropy and ok_err)

            status = "✓" if r["moderate_ok"] else " "
            if not ok_entropy:
                if r["p95_entropy"] < S_MIN:
                    status += " (too low)"
                else:
                    status += " (too high)"
            if not ok_err:
                status += " (err>"
                status += f"{target_err:.0e})"

            print(f"{n:<8} {chi_cap:<8} {r['p95_entropy']:<10.2f} "
                  f"{r['max_chi']:<8} {r['global_err']:<12.2e} "
                  f"{r['mem_MB']:<12.2f} {status}")

            results.append(r)

        except RuntimeError as e:
            print(f"{n:<8} OOM/Fail: {e}")
            break
        except Exception as e:
            print(f"{n:<8} Error: {e}")
            break

    # Report largest n in the "moderate" band
    ok_results = [r for r in results if r["moderate_ok"]]

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if ok_results:
        best = ok_results[-1]
        print(f"\n✓ Maximum n within moderate-entanglement band: {best['n']} qubits")
        print(f"  ├─ p95 entropy: {best['p95_entropy']:.2f} bits")
        print(f"  ├─ Mean entropy: {best['mean_entropy']:.2f} bits")
        print(f"  ├─ Max χ: {best['max_chi']}")
        print(f"  ├─ Mean χ: {best['mean_chi']:.2f}")
        print(f"  ├─ Global error: {best['global_err']:.2e}")
        print(f"  ├─ Max local error: {best['max_eps']:.2e}")
        print(f"  ├─ Memory: {best['mem_MB']:.2f} MB")
        print(f"  ├─ Time: {best['time_s']:.2f} s")
        print(f"  ├─ Throughput: {best['ops_per_s']:.1f} ops/sec")
        print(f"  └─ CUDA SVD: {best['cuda_svd_pct']:.1f}%")

        # Summary table
        print(f"\n{'n':<10} {'p95_S':<12} {'maxχ':<10} {'mem(MB)':<12} {'time(s)'}")
        print("-" * 60)
        for r in ok_results:
            print(f"{r['n']:<10} {r['p95_entropy']:<12.2f} "
                  f"{r['max_chi']:<10} {r['mem_MB']:<12.2f} {r['time_s']:<.2f}")
    else:
        print("\n❌ No n met the moderate-entanglement criteria.")
        print("\nSuggestions:")
        print("  - Increase χ_cap (CHI_CAP=128)")
        print("  - Increase eps tolerance (EPS=1e-5)")
        print("  - Reduce layers (LAYERS=6)")
        print("  - Widen entropy band (S_MIN=1.0 S_MAX=10.0)")

    print()
