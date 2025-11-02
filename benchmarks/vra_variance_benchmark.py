"""
VRA Variance Reduction Benchmark
=================================

Demonstrates VRA's variance reduction in Hamiltonian measurements.

Simulates shot-based measurement variance and shows how VRA grouping
reduces variance for the same number of total shots.

This is a simplified benchmark focusing purely on the measurement aspect,
independent of the full VQE optimization loop.

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

import numpy as np
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atlas_q.vra_enhanced import vra_hamiltonian_grouping


def simulate_hamiltonian_measurement(
    coeffs: np.ndarray,
    pauli_strings: List[str],
    groups: List[List[int]],
    shots_per_group: np.ndarray,
    n_samples: int = 1000,
    true_expectation: float = 0.0
) -> Tuple[np.ndarray, float, float]:
    """
    Simulate shot-based measurement with realistic variance.

    Parameters
    ----------
    coeffs : np.ndarray
        Hamiltonian coefficients
    pauli_strings : List[str]
        Pauli strings for each term
    groups : List[List[int]]
        Term groupings
    shots_per_group : np.ndarray
        Shots allocated to each group
    n_samples : int
        Number of measurement samples
    true_expectation : float
        True expectation value of Hamiltonian

    Returns
    -------
    measurements : np.ndarray
        Array of n_samples measurements
    mean : float
        Mean of measurements
    std : float
        Standard deviation of measurements
    """
    measurements = []

    for _ in range(n_samples):
        total_energy = 0.0

        for group, shots_g in zip(groups, shots_per_group):
            # For each group, simulate measurement
            c_g = coeffs[group]

            # True group energy (sum of coefficients times assumed expectation=1)
            # In real VQE, this would be <ψ|P_i|ψ> for each Pauli
            # For simulation: assume expectation values ~ 1 with variance
            true_group_energy = np.sum(c_g)

            # Variance for this group
            # For independent Pauli measurements: Var = Σ c_i²
            # With shots: Var = (Σ c_i²) / shots_g
            variance = np.sum(c_g**2) / shots_g

            # Simulate measurement
            measured_group = np.random.normal(true_group_energy, np.sqrt(variance))
            total_energy += measured_group

        measurements.append(total_energy)

    measurements = np.array(measurements)
    return measurements, np.mean(measurements), np.std(measurements)


def benchmark_h2_variance():
    """Benchmark variance reduction on H2 molecular Hamiltonian."""
    print("\n" + "="*70)
    print("VRA Variance Reduction Benchmark: H2 Molecule")
    print("="*70)

    # H2 Hamiltonian
    coeffs = np.array([-0.81054, 0.17218, -0.22575, 0.12091, 0.16862])
    paulis = ["II", "ZI", "IZ", "ZZ", "XX"]

    print(f"\nHamiltonian: {len(coeffs)} Pauli terms")
    print(f"Terms: {list(zip(coeffs, paulis))}")

    total_shots = 10000
    n_samples = 1000

    # Baseline: per-term measurement
    print(f"\n{'─'*70}")
    print("BASELINE: Per-term measurement")
    print(f"{'─'*70}")

    n_terms = len(coeffs)
    shots_per_term = total_shots // n_terms
    baseline_groups = [[i] for i in range(n_terms)]
    baseline_shots = np.array([shots_per_term] * n_terms)

    print(f"Groups: {baseline_groups}")
    print(f"Shots per term: {shots_per_term}")

    baseline_measurements, baseline_mean, baseline_std = simulate_hamiltonian_measurement(
        coeffs, paulis, baseline_groups, baseline_shots, n_samples=n_samples
    )

    print(f"Mean: {baseline_mean:.6f}")
    print(f"Std Dev: {baseline_std:.6f}")
    print(f"Variance: {baseline_std**2:.6f}")

    # VRA: grouped measurement
    print(f"\n{'─'*70}")
    print("VRA-ENHANCED: Grouped measurement with Neyman allocation")
    print(f"{'─'*70}")

    result = vra_hamiltonian_grouping(
        coeffs,
        pauli_strings=paulis,
        total_shots=total_shots,
        max_group_size=5
    )

    print(f"Groups: {result.groups}")
    print(f"Shot allocation: {result.shots_per_group}")
    print(f"Predicted variance reduction: {result.variance_reduction:.2f}×")

    vra_measurements, vra_mean, vra_std = simulate_hamiltonian_measurement(
        coeffs, paulis, result.groups, result.shots_per_group, n_samples=n_samples
    )

    print(f"Mean: {vra_mean:.6f}")
    print(f"Std Dev: {vra_std:.6f}")
    print(f"Variance: {vra_std**2:.6f}")

    # Comparison
    print(f"\n{'='*70}")
    print("VARIANCE REDUCTION RESULTS")
    print(f"{'='*70}")

    variance_reduction_actual = (baseline_std**2) / (vra_std**2)
    std_reduction_actual = baseline_std / vra_std

    print(f"Baseline variance:      {baseline_std**2:.6f}")
    print(f"VRA variance:           {vra_std**2:.6f}")
    print(f"Variance reduction:     {variance_reduction_actual:.2f}×")
    print(f"Std dev reduction:      {std_reduction_actual:.2f}×")
    print(f"")
    print(f"VRA prediction:         {result.variance_reduction:.2f}×")
    print(f"Actual measurement:     {variance_reduction_actual:.2f}×")
    print(f"Match quality:          {abs(result.variance_reduction - variance_reduction_actual):.2f} (lower = better)")

    # Statistical significance
    print(f"\n{'─'*70}")
    print("Statistical Significance")
    print(f"{'─'*70}")

    # For same precision, how many fewer shots needed?
    shot_reduction = baseline_std**2 / vra_std**2
    shots_needed_vra = total_shots / shot_reduction

    print(f"For {baseline_std:.6f} precision:")
    print(f"  Baseline needs: {total_shots} shots")
    print(f"  VRA needs:      {int(shots_needed_vra)} shots")
    print(f"  Shot savings:   {total_shots - int(shots_needed_vra)} shots ({(1 - shots_needed_vra/total_shots)*100:.1f}%)")

    # Save plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(baseline_measurements, bins=50, alpha=0.7, label='Baseline', color='blue')
    plt.hist(vra_measurements, bins=50, alpha=0.7, label='VRA', color='green')
    plt.xlabel('Measured Energy')
    plt.ylabel('Frequency')
    plt.title(f'H2 Measurement Distribution ({n_samples} samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    methods = ['Baseline\n(per-term)', 'VRA\n(grouped)']
    variances = [baseline_std**2, vra_std**2]
    colors = ['blue', 'green']
    bars = plt.bar(methods, variances, color=colors, alpha=0.7)
    plt.ylabel('Variance')
    plt.title('Variance Comparison')
    plt.grid(True, alpha=0.3, axis='y')

    # Add reduction factor annotation
    plt.text(0.5, max(variances)*0.9,
             f'{variance_reduction_actual:.2f}× reduction',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/admin/ATLAS-Q/benchmarks/h2_variance_reduction.png', dpi=150)
    print(f"\nPlot saved to: benchmarks/h2_variance_reduction.png")

    return variance_reduction_actual


def benchmark_scaling():
    """Benchmark how variance reduction scales with Hamiltonian size."""
    print("\n" + "="*70)
    print("VRA Variance Reduction Scaling Benchmark")
    print("="*70)

    hamiltonian_sizes = [5, 8, 10, 12, 15]
    total_shots = 10000
    n_samples = 1000

    results = []

    for n_terms in hamiltonian_sizes:
        print(f"\n{'─'*70}")
        print(f"Hamiltonian size: {n_terms} terms")
        print(f"{'─'*70}")

        # Create synthetic Hamiltonian with exponential decay
        coeffs = np.exp(-np.arange(n_terms) / 4.0)

        # Create Pauli strings
        pauli_strings = []
        for i in range(n_terms):
            # Alternate between different Pauli types
            if i % 4 == 0:
                pauli_strings.append("ZZ" + "I"*(n_terms//2))
            elif i % 4 == 1:
                pauli_strings.append("XX" + "I"*(n_terms//2))
            elif i % 4 == 2:
                pauli_strings.append("YY" + "I"*(n_terms//2))
            else:
                pauli_strings.append("ZI" + "I"*(n_terms//2))

        # Truncate to reasonable length
        pauli_strings = [p[:min(4, n_terms)] for p in pauli_strings]

        # Baseline
        shots_per_term = total_shots // n_terms
        baseline_groups = [[i] for i in range(n_terms)]
        baseline_shots = np.array([shots_per_term] * n_terms)

        baseline_measurements, _, baseline_std = simulate_hamiltonian_measurement(
            coeffs, pauli_strings, baseline_groups, baseline_shots, n_samples=n_samples
        )

        # VRA
        vra_result = vra_hamiltonian_grouping(
            coeffs,
            pauli_strings=pauli_strings,
            total_shots=total_shots,
            max_group_size=5
        )

        vra_measurements, _, vra_std = simulate_hamiltonian_measurement(
            coeffs, pauli_strings, vra_result.groups, vra_result.shots_per_group, n_samples=n_samples
        )

        variance_reduction = (baseline_std**2) / (vra_std**2)

        print(f"  VRA groups: {len(vra_result.groups)}")
        print(f"  Baseline std: {baseline_std:.6f}")
        print(f"  VRA std:      {vra_std:.6f}")
        print(f"  Reduction:    {variance_reduction:.2f}×")

        results.append((n_terms, variance_reduction))

    # Plot scaling
    plt.figure(figsize=(10, 6))
    sizes, reductions = zip(*results)
    plt.plot(sizes, reductions, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Hamiltonian Size (number of terms)')
    plt.ylabel('Variance Reduction Factor')
    plt.title('VRA Variance Reduction vs Hamiltonian Size')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/admin/ATLAS-Q/benchmarks/variance_reduction_scaling.png', dpi=150)
    print(f"\nScaling plot saved to: benchmarks/variance_reduction_scaling.png")

    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    for n_terms, reduction in results:
        print(f"  {n_terms:2d} terms: {reduction:5.2f}× variance reduction")


if __name__ == "__main__":
    # Benchmark H2
    h2_reduction = benchmark_h2_variance()

    # Benchmark scaling
    benchmark_scaling()

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"H2 variance reduction: {h2_reduction:.2f}×")
    print(f"See plots in benchmarks/ directory")
    print(f"{'='*70}\n")
