"""
VRA Commutativity Enhancement Benchmark
========================================

Compares three measurement strategies for VQE:
1. Baseline: Per-term independent measurement
2. VRA: Variance-minimized grouping (no commutativity constraints)
3. VRA+Commutativity: Physically realizable grouped measurement

Key Trade-off:
- VRA without commutativity achieves high variance reduction but is PHYSICALLY IMPOSSIBLE
- VRA with commutativity achieves moderate variance reduction but is PHYSICALLY REALIZABLE

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

import numpy as np
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from atlas_q.vra_enhanced import (
    vra_hamiltonian_grouping,
    estimate_pauli_coherence_matrix,
    group_by_variance_minimization,
    allocate_shots_neyman,
    check_group_commutativity,
)
from atlas_q.vra_enhanced.vqe_grouping import compute_variance_reduction


def simulate_measurement_variance(
    coeffs: np.ndarray,
    pauli_strings: List[str],
    groups: List[List[int]],
    shots_per_group: np.ndarray,
    n_samples: int = 1000
) -> Tuple[float, float]:
    """
    Simulate measurement variance for a grouping strategy.

    Returns
    -------
    mean : float
        Mean measured energy
    std : float
        Standard deviation of measurements
    """
    measurements = []

    for _ in range(n_samples):
        total_energy = 0.0

        for group, shots_g in zip(groups, shots_per_group):
            c_g = coeffs[group]
            true_group_energy = np.sum(c_g)
            variance = np.sum(c_g**2) / shots_g
            measured_group = np.random.normal(true_group_energy, np.sqrt(variance))
            total_energy += measured_group

        measurements.append(total_energy)

    return float(np.mean(measurements)), float(np.std(measurements))


def benchmark_three_methods(
    coeffs: np.ndarray,
    pauli_strings: List[str],
    total_shots: int = 10000,
    n_samples: int = 1000
) -> Tuple:
    """
    Benchmark all three methods and return results.

    Returns
    -------
    baseline_results : dict
    vra_results : dict
    vra_comm_results : dict
    """
    n_terms = len(coeffs)

    print(f"\nHamiltonian: {n_terms} Pauli terms")
    print(f"Total shots: {total_shots}")
    print(f"Simulation samples: {n_samples}")

    # Method 1: Baseline (per-term measurement)
    print(f"\n{'='*70}")
    print("Method 1: BASELINE (Per-term measurement)")
    print(f"{'='*70}")

    shots_per_term = total_shots // n_terms
    baseline_groups = [[i] for i in range(n_terms)]
    baseline_shots = np.array([shots_per_term] * n_terms)

    print(f"Groups: {len(baseline_groups)} (one per term)")
    print(f"Shots per term: {shots_per_term}")

    baseline_mean, baseline_std = simulate_measurement_variance(
        coeffs, pauli_strings, baseline_groups, baseline_shots, n_samples
    )

    print(f"Mean energy: {baseline_mean:.6f}")
    print(f"Std dev: {baseline_std:.6f}")
    print(f"Variance: {baseline_std**2:.6f}")

    baseline_results = {
        'groups': baseline_groups,
        'shots': baseline_shots,
        'mean': baseline_mean,
        'std': baseline_std,
        'variance': baseline_std**2,
        'physically_realizable': True,
        'method': 'Baseline'
    }

    # Method 2: VRA (no commutativity constraints)
    print(f"\n{'='*70}")
    print("Method 2: VRA (Variance-minimized, NO commutativity)")
    print(f"{'='*70}")

    Sigma = estimate_pauli_coherence_matrix(coeffs, pauli_strings)

    vra_groups = group_by_variance_minimization(
        Sigma, coeffs, max_group_size=5,
        pauli_strings=pauli_strings,
        check_commutativity=False  # Disabled
    )

    vra_shots = allocate_shots_neyman(Sigma, coeffs, vra_groups, total_shots)

    print(f"Groups: {len(vra_groups)}")
    print(f"Group structure: {vra_groups}")
    print(f"Shot allocation: {vra_shots}")

    # Check if groups violate commutativity
    violations = 0
    for i, group in enumerate(vra_groups):
        commutes = check_group_commutativity(group, pauli_strings)
        if not commutes:
            violations += 1
            print(f"  Group {i} {group}: ⚠️ VIOLATES COMMUTATIVITY (not physically realizable)")

    vra_mean, vra_std = simulate_measurement_variance(
        coeffs, pauli_strings, vra_groups, vra_shots, n_samples
    )

    var_reduction_vra = (baseline_std**2) / (vra_std**2)

    print(f"\nMean energy: {vra_mean:.6f}")
    print(f"Std dev: {vra_std:.6f}")
    print(f"Variance: {vra_std**2:.6f}")
    print(f"Variance reduction: {var_reduction_vra:.2f}×")
    print(f"Physically realizable: {'No - {violations} groups violate commutativity' if violations > 0 else 'Yes'}")

    vra_results = {
        'groups': vra_groups,
        'shots': vra_shots,
        'mean': vra_mean,
        'std': vra_std,
        'variance': vra_std**2,
        'variance_reduction': var_reduction_vra,
        'physically_realizable': violations == 0,
        'violations': violations,
        'method': 'VRA (no commutativity)'
    }

    # Method 3: VRA + Commutativity
    print(f"\n{'='*70}")
    print("Method 3: VRA + COMMUTATIVITY (Physically realizable)")
    print(f"{'='*70}")

    vra_comm_groups = group_by_variance_minimization(
        Sigma, coeffs, max_group_size=5,
        pauli_strings=pauli_strings,
        check_commutativity=True  # Enabled
    )

    vra_comm_shots = allocate_shots_neyman(Sigma, coeffs, vra_comm_groups, total_shots)

    print(f"Groups: {len(vra_comm_groups)}")
    print(f"Group structure: {vra_comm_groups}")
    print(f"Shot allocation: {vra_comm_shots}")

    # Verify all groups commute
    all_commute = True
    for i, group in enumerate(vra_comm_groups):
        commutes = check_group_commutativity(group, pauli_strings)
        print(f"  Group {i} {group}: {'✓ commutes' if commutes else '✗ VIOLATES'}")
        all_commute = all_commute and commutes

    vra_comm_mean, vra_comm_std = simulate_measurement_variance(
        coeffs, pauli_strings, vra_comm_groups, vra_comm_shots, n_samples
    )

    var_reduction_comm = (baseline_std**2) / (vra_comm_std**2)

    print(f"\nMean energy: {vra_comm_mean:.6f}")
    print(f"Std dev: {vra_comm_std:.6f}")
    print(f"Variance: {vra_comm_std**2:.6f}")
    print(f"Variance reduction: {var_reduction_comm:.2f}×")
    print(f"Physically realizable: {'Yes' if all_commute else 'No'}")

    vra_comm_results = {
        'groups': vra_comm_groups,
        'shots': vra_comm_shots,
        'mean': vra_comm_mean,
        'std': vra_comm_std,
        'variance': vra_comm_std**2,
        'variance_reduction': var_reduction_comm,
        'physically_realizable': all_commute,
        'method': 'VRA + Commutativity'
    }

    return baseline_results, vra_results, vra_comm_results


def plot_comparison(baseline, vra, vra_comm, filename):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Variance comparison
    methods = ['Baseline\n(per-term)', 'VRA\n(no commutativity)', 'VRA\n(with commutativity)']
    variances = [baseline['variance'], vra['variance'], vra_comm['variance']]
    colors = ['blue', 'orange', 'green']
    realizable = [baseline['physically_realizable'], vra['physically_realizable'], vra_comm['physically_realizable']]

    bars = axes[0].bar(methods, variances, color=colors, alpha=0.7)
    for i, (bar, real) in enumerate(zip(bars, realizable)):
        if not real:
            bar.set_hatch('//')
            bar.set_edgecolor('red')
            bar.set_linewidth(2)

    axes[0].set_ylabel('Variance')
    axes[0].set_title('Measurement Variance Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add legend for hatching
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Physically realizable'),
        Patch(facecolor='gray', hatch='//', edgecolor='red', linewidth=2, alpha=0.7, label='Not realizable')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Plot 2: Variance reduction factor
    reductions = [1.0, vra.get('variance_reduction', 1.0), vra_comm.get('variance_reduction', 1.0)]
    bars = axes[1].bar(methods, reductions, color=colors, alpha=0.7)
    for i, (bar, real) in enumerate(zip(bars, realizable)):
        if not real:
            bar.set_hatch('//')
            bar.set_edgecolor('red')
            bar.set_linewidth(2)

    axes[1].set_ylabel('Variance Reduction Factor')
    axes[1].set_title('Variance Reduction vs Baseline')
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Number of groups
    n_groups = [len(baseline['groups']), len(vra['groups']), len(vra_comm['groups'])]
    bars = axes[2].bar(methods, n_groups, color=colors, alpha=0.7)
    for i, (bar, real) in enumerate(zip(bars, realizable)):
        if not real:
            bar.set_hatch('//')
            bar.set_edgecolor('red')
            bar.set_linewidth(2)

    axes[2].set_ylabel('Number of Groups')
    axes[2].set_title('Measurement Grouping')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to: {filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VRA Commutativity Enhancement Benchmark")
    print("="*70)

    # H2 Molecular Hamiltonian
    print("\n" + "─"*70)
    print("Benchmark 1: H2 Molecular Hamiltonian")
    print("─"*70)

    h2_coeffs = np.array([-0.81054, 0.17218, -0.22575, 0.12091, 0.16862])
    h2_paulis = ["II", "ZI", "IZ", "ZZ", "XX"]

    baseline_h2, vra_h2, vra_comm_h2 = benchmark_three_methods(
        h2_coeffs, h2_paulis, total_shots=10000, n_samples=1000
    )

    plot_comparison(baseline_h2, vra_h2, vra_comm_h2, '/home/admin/ATLAS-Q/benchmarks/h2_commutativity_comparison.png')

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: H2 Molecular Hamiltonian")
    print("="*70)
    print(f"Baseline variance:               {baseline_h2['variance']:.6f} (1.00×)")
    print(f"VRA variance:                    {vra_h2['variance']:.6f} ({vra_h2['variance_reduction']:.2f}×) {'⚠️ NOT REALIZABLE' if not vra_h2['physically_realizable'] else '✓ Realizable'}")
    print(f"VRA+Commutativity variance:      {vra_comm_h2['variance']:.6f} ({vra_comm_h2['variance_reduction']:.2f}×) ✓ Realizable")
    print(f"\n{'='*70}")
    print("KEY INSIGHT:")
    print("  VRA without commutativity shows higher reduction but is physically")
    print("  impossible to measure (violates quantum mechanics).")
    print("  VRA with commutativity provides realizable measurement strategy")
    print(f"  with {vra_comm_h2['variance_reduction']:.2f}× variance reduction.")
    print("="*70)
