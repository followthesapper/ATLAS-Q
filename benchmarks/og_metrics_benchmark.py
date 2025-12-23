#!/usr/bin/env python3
"""
Observability Geometry (OG) Metrics Benchmark
===============================================

Demonstrates the OG-enhanced capabilities of ATLAS-Q:
1. OG metrics computation performance (CPU vs Triton/GPU)
2. OG-guided shot allocation optimization
3. OG-based circuit quality evaluation
4. Comparison of OG vs traditional approaches

Author: ATLAS-Q Development Team
Date: December 2025
"""

import time
from typing import Dict, List, Tuple

import numpy as np

# Import OG metrics
from atlas_q.coherence import (
    compute_og_metrics,
    compute_effective_hbar,
    compute_representational_cost,
    compute_coherence_gradient,
    compute_optimal_shot_allocation,
    evaluate_circuit_quality,
    classify_observability_regime,
    track_transition_dynamics,
    R_BAR_CRITICAL,
    H_EFF_CRITICAL,
    ObservabilityRegime,
)


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """Print section header."""
    print(f"\n{title}")
    print("-" * len(title))


def benchmark_og_metrics_computation():
    """Benchmark OG metrics computation performance."""
    print_header("BENCHMARK 1: OG Metrics Computation Performance")

    sizes = [1000, 10000, 100000]

    print_section("Pure Python/NumPy Performance")
    for size in sizes:
        # Generate random R_bar values
        R_bars = np.random.uniform(0.01, 0.99, size)

        # Time effective Planck constant
        t0 = time.perf_counter()
        for R in R_bars[:1000]:  # Sample for speed
            _ = compute_effective_hbar(R)
        t_heff = (time.perf_counter() - t0) / 1000 * size

        # Time coherence gradient
        t0 = time.perf_counter()
        _ = compute_coherence_gradient(R_bars)
        t_grad = time.perf_counter() - t0

        # Time full OG metrics
        t0 = time.perf_counter()
        for R in R_bars[:100]:  # Sample
            _ = compute_og_metrics(R_bar=R)
        t_full = (time.perf_counter() - t0) / 100 * size

        print(f"  Size={size:>6}:  h_eff={t_heff*1000:.2f}ms  "
              f"grad={t_grad*1000:.2f}ms  full={t_full*1000:.2f}ms")

    # Try Triton if available
    try:
        import torch
        if torch.cuda.is_available():
            print_section("Triton/GPU Performance")
            from triton_kernels.og_kernels import (
                compute_effective_hbar_triton,
                compute_og_metrics_from_amplitudes_triton,
            )

            for size in sizes:
                # Create GPU tensors
                R_bars_gpu = torch.rand(size, device='cuda', dtype=torch.float32) * 0.98 + 0.01
                amplitudes_gpu = torch.randn(size, dtype=torch.complex64, device='cuda')
                amplitudes_gpu = amplitudes_gpu / torch.norm(amplitudes_gpu)

                # Warmup
                for _ in range(3):
                    _ = compute_effective_hbar_triton(R_bars_gpu)
                    _ = compute_og_metrics_from_amplitudes_triton(amplitudes_gpu)
                torch.cuda.synchronize()

                # Time h_eff
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = compute_effective_hbar_triton(R_bars_gpu)
                torch.cuda.synchronize()
                t_heff_gpu = (time.perf_counter() - t0) / 10

                # Time full metrics from amplitudes
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = compute_og_metrics_from_amplitudes_triton(amplitudes_gpu)
                torch.cuda.synchronize()
                t_full_gpu = (time.perf_counter() - t0) / 10

                print(f"  Size={size:>6}:  h_eff={t_heff_gpu*1000:.3f}ms  "
                      f"full_metrics={t_full_gpu*1000:.3f}ms")
    except ImportError:
        print("\n  (Triton not available, skipping GPU benchmark)")


def benchmark_shot_allocation():
    """Benchmark OG-guided vs uniform shot allocation."""
    print_header("BENCHMARK 2: OG-Guided Shot Allocation")

    print_section("Scenario: VQE with varying coherence across measurement groups")

    # Simulate measurement groups with different coherence levels
    group_R_bars = [0.9, 0.7, 0.5, 0.3, 0.15]
    total_shots = 10000

    print(f"\nGroup coherence (R_bar): {group_R_bars}")
    print(f"Total shots available: {total_shots}")

    # OG-guided allocation
    og_shots = compute_optimal_shot_allocation(group_R_bars, total_shots)

    # Uniform allocation
    uniform_shots = [total_shots // len(group_R_bars)] * len(group_R_bars)

    print(f"\nOG-guided allocation:  {og_shots}")
    print(f"Uniform allocation:    {uniform_shots}")

    # Analyze the difference
    print("\nAnalysis:")
    print("  Group | R_bar | V_phi | OG Shots | Uniform | Ratio")
    print("  " + "-" * 55)

    for i, R_bar in enumerate(group_R_bars):
        V_phi = -2 * np.log(R_bar)
        ratio = og_shots[i] / uniform_shots[i]
        print(f"    {i+1}   | {R_bar:.2f}  | {V_phi:.2f}  |   {og_shots[i]:4}   |  {uniform_shots[i]:4}   | {ratio:.2f}x")

    # Simulate measurement quality
    print("\nSimulated measurement quality (lower is better):")

    np.random.seed(42)
    n_trials = 1000

    # OG allocation quality
    og_errors = []
    for _ in range(n_trials):
        group_errors = []
        for R_bar, shots in zip(group_R_bars, og_shots):
            # Higher R_bar = lower error, more shots = lower error
            error = (1 - R_bar) / np.sqrt(shots) + np.random.randn() * 0.01
            group_errors.append(abs(error))
        og_errors.append(np.mean(group_errors))

    # Uniform allocation quality
    uniform_errors = []
    for _ in range(n_trials):
        group_errors = []
        for R_bar, shots in zip(group_R_bars, uniform_shots):
            error = (1 - R_bar) / np.sqrt(shots) + np.random.randn() * 0.01
            group_errors.append(abs(error))
        uniform_errors.append(np.mean(group_errors))

    print(f"  OG-guided mean error:  {np.mean(og_errors):.4f} +/- {np.std(og_errors):.4f}")
    print(f"  Uniform mean error:    {np.mean(uniform_errors):.4f} +/- {np.std(uniform_errors):.4f}")
    print(f"  OG improvement:        {(np.mean(uniform_errors) - np.mean(og_errors)) / np.mean(uniform_errors) * 100:.1f}%")


def benchmark_circuit_quality():
    """Benchmark OG-based circuit quality evaluation."""
    print_header("BENCHMARK 3: OG-Based Circuit Quality Evaluation")

    print_section("Comparing circuit quality metrics across different configurations")

    # Different circuit configurations
    configs = {
        "Optimal (uniform high)": [0.9, 0.9, 0.9, 0.9, 0.9],
        "Degraded uniform": [0.5, 0.5, 0.5, 0.5, 0.5],
        "Non-uniform (gradient)": [0.9, 0.7, 0.5, 0.3, 0.1],
        "One bad qubit": [0.9, 0.9, 0.1, 0.9, 0.9],
        "Two bad qubits": [0.9, 0.1, 0.9, 0.1, 0.9],
    }

    print("\n  Configuration           | Mean R | Min R | Gradient | Quality | Regime")
    print("  " + "-" * 75)

    for name, R_bars in configs.items():
        result = evaluate_circuit_quality(R_bars)
        print(f"  {name:<24} | {result['mean_R_bar']:.3f}  | {result['min_R_bar']:.3f} | "
              f"{result['coherence_gradient']:.4f}  | {result['quality_score']:.3f}   | {result['regime']}")

    print("\nKey insights from OG theory:")
    print("  - Quality depends on BOTH mean coherence AND uniformity")
    print("  - Non-uniform coherence creates 'observability gradients'")
    print("  - Circuit quality is limited by the bottleneck qubit (min_R_bar)")
    print("  - The gradient term penalizes non-uniform configurations")


def benchmark_transition_tracking():
    """Benchmark OG transition dynamics tracking."""
    print_header("BENCHMARK 4: OG Transition Dynamics Tracking")

    print_section("Simulating VQE coherence evolution and prediction")

    # Simulate VQE coherence history
    np.random.seed(42)
    n_steps = 20

    # Scenario 1: Successful optimization (improving)
    improving = [0.3]
    for _ in range(n_steps - 1):
        new_R = improving[-1] + 0.03 + np.random.randn() * 0.01
        improving.append(np.clip(new_R, 0.01, 0.99))

    # Scenario 2: Degrading (noise dominated)
    degrading = [0.8]
    for _ in range(n_steps - 1):
        new_R = degrading[-1] - 0.02 + np.random.randn() * 0.01
        degrading.append(np.clip(new_R, 0.01, 0.99))

    # Scenario 3: Oscillating
    oscillating = [0.5]
    for i in range(n_steps - 1):
        new_R = oscillating[-1] + 0.05 * np.sin(i * 0.5) + np.random.randn() * 0.02
        oscillating.append(np.clip(new_R, 0.01, 0.99))

    scenarios = {
        "Improving": improving,
        "Degrading": degrading,
        "Oscillating": oscillating,
    }

    print("\nScenario Analysis:")
    print("  Scenario    | Start | End  | Direction | Velocity | Steps to e^-2")
    print("  " + "-" * 65)

    for name, history in scenarios.items():
        dynamics = track_transition_dynamics(history)
        steps_str = f"{dynamics.steps_to_critical:.1f}" if dynamics.steps_to_critical else "N/A"
        print(f"  {name:<12} | {history[0]:.3f} | {history[-1]:.3f}| "
              f"{dynamics.direction:<9} | {dynamics.velocity:+.4f}  | {steps_str}")

    # Early stopping analysis
    print("\nOG-based Early Stopping Analysis:")
    print("  The e^-2 threshold (R_bar_c ~ 0.135) defines the observability boundary")
    print("  Below this threshold, quantum structure becomes unobservable")

    for name, history in scenarios.items():
        dynamics = track_transition_dynamics(history)
        final_regime = classify_observability_regime(history[-1])

        if final_regime == ObservabilityRegime.AIR:
            recommendation = "STOP - Below observability threshold"
        elif dynamics.direction == 'degrading' and dynamics.velocity < -0.02:
            recommendation = "WARNING - Rapid degradation, consider restart"
        elif dynamics.direction == 'improving':
            recommendation = "CONTINUE - Coherence improving"
        else:
            recommendation = "MONITOR - Stable or slow changes"

        print(f"  {name}: {recommendation}")


def benchmark_heff_insights():
    """Demonstrate effective Planck constant insights."""
    print_header("BENCHMARK 5: Effective Planck Constant Analysis")

    print_section("OG-derived h_eff = 2*sqrt(-2*ln(R))*sqrt(R)")

    print("\nKey theoretical predictions from OG:")
    print(f"  - Critical point: R_bar_c = e^-2 = {R_BAR_CRITICAL:.4f}")
    print(f"  - Critical h_eff: h_eff_c = {H_EFF_CRITICAL:.4f}")
    print(f"  - Maximum h_eff at R = e^-1 = {np.exp(-1):.4f}")

    # Compute h_eff curve
    R_values = np.linspace(0.01, 0.99, 50)
    h_values = [compute_effective_hbar(R) for R in R_values]

    max_idx = np.argmax(h_values)
    max_R = R_values[max_idx]
    max_h = h_values[max_idx]

    print(f"\nComputed maximum: h_eff = {max_h:.4f} at R_bar = {max_R:.4f}")
    print(f"Theoretical maximum: R = e^-1 = {np.exp(-1):.4f}")

    # Regime analysis
    print("\nh_eff values across regimes:")
    regime_samples = [
        ("AIR (R=0.03)", 0.03),
        ("Transition (R=0.15)", 0.15),
        ("Critical (R=e^-2)", R_BAR_CRITICAL),
        ("Upper Trans (R=0.35)", 0.35),
        ("IR (R=0.6)", 0.6),
        ("High IR (R=0.9)", 0.9),
    ]

    print("  R_bar  | h_eff | Regime      | Interpretation")
    print("  " + "-" * 55)

    for name, R in regime_samples:
        h = compute_effective_hbar(R)
        regime = classify_observability_regime(R)
        if h < 0.5:
            interp = "Fine resolution possible"
        elif h < 1.0:
            interp = "Moderate quantum granularity"
        elif h < 1.5:
            interp = "Near-critical resolution"
        else:
            interp = "Maximum quantum granularity"
        print(f"  {R:.4f} | {h:.3f} | {regime.value:<11} | {interp}")


def run_all_benchmarks():
    """Run all OG benchmarks."""
    print("\n" + "=" * 70)
    print("  ATLAS-Q OBSERVABILITY GEOMETRY (OG) BENCHMARK SUITE")
    print("=" * 70)
    print("\nThis benchmark demonstrates the pre-quantum theoretical foundation")
    print("provided by Observability Geometry for quantum simulation optimization.")

    benchmark_og_metrics_computation()
    benchmark_shot_allocation()
    benchmark_circuit_quality()
    benchmark_transition_tracking()
    benchmark_heff_insights()

    print_header("BENCHMARK SUMMARY")
    print("""
OG provides ATLAS-Q with:

1. EFFECTIVE PLANCK CONSTANT (h_eff)
   - Basis-dependent quantum granularity
   - Predicts measurement resolution limits
   - Falsifiable prediction distinguishing OG from standard QM

2. REPRESENTATIONAL COST (C = 1/V_phi)
   - Resource metric for observability extraction
   - Guides shot allocation for optimal precision

3. COHERENCE GRADIENT (nabla V_phi)
   - Measures spatial non-uniformity
   - Minimizing this minimizes the OG action functional
   - Optimal circuits have uniform coherence

4. TRANSITION DYNAMICS
   - e^-2 threshold defines observability boundary
   - Tracking enables early stopping decisions
   - Predicts when quantum structure becomes extractable

5. REGIME CLASSIFICATION
   - IR: Structure observable (R > 0.37)
   - TRANSITION: Critical dynamics (0.05 < R < 0.37)
   - AIR: Structure hidden (R < 0.05)

These capabilities derive from OG's pre-quantum theoretical foundation,
enabling ATLAS-Q to make optimization decisions based on fundamental
physics rather than heuristics.
""")


if __name__ == "__main__":
    run_all_benchmarks()
