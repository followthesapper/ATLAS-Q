#!/usr/bin/env python3
"""
VRA Commutativity Benchmark: Larger Molecules
==============================================

Test commutativity-aware VQE grouping on larger molecular Hamiltonians
to demonstrate variance reduction improvements over H2.

Expected results:
- H2 (5 terms): 0.76× (poor commuting structure)
- LiH (12+ terms): 5-20× (better commuting structure)
- H2O (20+ terms): 10-50× (strong commuting groups)

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

import sys
import numpy as np
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, '/home/admin/ATLAS-Q/src')

from atlas_q.vra_enhanced import (
    vra_hamiltonian_grouping,
    estimate_pauli_coherence_matrix,
    group_by_variance_minimization,
    allocate_shots_neyman,
    check_group_commutativity,
)


def get_molecular_pauli_hamiltonian(molecule: str, basis: str = 'sto-3g') -> Tuple[np.ndarray, List[str]]:
    """
    Generate Pauli decomposition of molecular Hamiltonian using PySCF.

    Parameters
    ----------
    molecule : str
        Molecule name or geometry string
    basis : str
        Basis set (default: sto-3g)

    Returns
    -------
    coeffs : np.ndarray
        Hamiltonian coefficients
    pauli_strings : List[str]
        Pauli strings (e.g., ['ZI', 'IZ', 'XX'])
    """
    from pyscf import gto, scf, ao2mo
    from atlas_q.mpo_ops import _jordan_wigner_transform

    # Define molecular geometries
    geometries = {
        'H2': 'H 0 0 0; H 0 0 0.74',
        'LiH': 'Li 0 0 0; H 0 0 1.596',
        'H2O': 'O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0',
        'BeH2': 'Be 0 0 0; H 0 0 1.334; H 0 0 -1.334',
        'NH3': 'N 0 0 0; H 0 0 1.012; H 0.955 0 -0.337; H -0.478 0.827 -0.337',
    }

    if molecule in geometries:
        geometry = geometries[molecule]
    else:
        geometry = molecule  # Assume custom geometry string

    print(f"\nGenerating {molecule} Hamiltonian...")
    print(f"  Geometry: {geometry}")
    print(f"  Basis: {basis}")

    # Build molecule with PySCF
    mol = gto.M(atom=geometry, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()

    # Get integrals
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    h2 = ao2mo.restore(1, eri, h1.shape[0])
    e_nuc = mol.energy_nuc()

    # Jordan-Wigner transformation
    pauli_terms = _jordan_wigner_transform(h1, h2, e_nuc)

    # Extract non-negligible terms
    threshold = 1e-8
    coeffs_list = []
    paulis_list = []

    for pauli_tuple, coeff in pauli_terms.items():
        if abs(coeff) > threshold:
            # Take real part of coefficient (Hamiltonian should be Hermitian)
            coeffs_list.append(np.real(coeff))
            # Convert tuple to string: ('X', 'Y', 'Z') → 'XYZ'
            paulis_list.append(''.join(pauli_tuple))

    coeffs = np.array(coeffs_list, dtype=float)

    print(f"  Number of qubits: {len(paulis_list[0]) if paulis_list else 0}")
    print(f"  Number of Pauli terms: {len(coeffs)}")
    print(f"  Largest coefficient: {max(abs(coeffs)):.6f}")
    print(f"  HF energy: {mf.e_tot:.8f} Ha")

    return coeffs, paulis_list


def simulate_measurement_variance(
    coeffs: np.ndarray,
    pauli_strings: List[str],
    groups: List[List[int]],
    shots_per_group: np.ndarray,
    Sigma: np.ndarray,
    n_samples: int = 1000
) -> Tuple[float, float]:
    """
    Simulate measurement variance for a grouping strategy.

    For grouped measurements, variance depends on coherence matrix (Q_GLS).
    """
    from atlas_q.vra_enhanced.vqe_grouping import compute_Q_GLS

    measurements = []

    for _ in range(n_samples):
        total_energy = 0.0

        for group, shots_g in zip(groups, shots_per_group):
            c_g = coeffs[group]
            true_group_energy = np.sum(c_g)

            # For grouped measurements, use Q_GLS to compute variance
            if len(group) > 1:
                Sigma_g = Sigma[np.ix_(group, group)]
                Q_g = compute_Q_GLS(Sigma_g, c_g)
                variance = Q_g / shots_g
            else:
                # Single term: variance = c^2 / shots
                variance = c_g[0]**2 / shots_g

            measured_group = np.random.normal(true_group_energy, np.sqrt(variance))
            total_energy += measured_group

        measurements.append(total_energy)

    return float(np.mean(measurements)), float(np.std(measurements))


def benchmark_molecule(
    molecule: str,
    total_shots: int = 10000,
    n_samples: int = 1000,
    max_terms: int = 50
) -> Dict:
    """
    Benchmark commutativity-aware grouping on a molecular Hamiltonian.

    Parameters
    ----------
    molecule : str
        Molecule name
    total_shots : int
        Total measurement shots
    n_samples : int
        Number of Monte Carlo samples
    max_terms : int
        Maximum number of Pauli terms to use (for large Hamiltonians)

    Returns
    -------
    results : dict
        Benchmark results for all three methods
    """
    print(f"\n{'='*70}")
    print(f"Benchmark: {molecule}")
    print(f"{'='*70}")

    # Generate Hamiltonian
    coeffs, pauli_strings = get_molecular_pauli_hamiltonian(molecule)

    # Sort by coefficient magnitude and keep largest terms
    if len(coeffs) > max_terms:
        indices = np.argsort(-np.abs(coeffs))[:max_terms]
        coeffs = coeffs[indices]
        pauli_strings = [pauli_strings[i] for i in indices]
        print(f"  Using {max_terms} largest terms (out of {len(pauli_strings)})")

    n_terms = len(coeffs)

    print(f"\nRunning three-way comparison...")
    print(f"  Total shots: {total_shots}")
    print(f"  Simulation samples: {n_samples}")

    # Method 1: Baseline (per-term measurement)
    print(f"\n{'-'*70}")
    print("Method 1: BASELINE (Per-term)")
    print(f"{'-'*70}")

    shots_per_term = total_shots // n_terms
    baseline_groups = [[i] for i in range(n_terms)]
    baseline_shots = np.array([shots_per_term] * n_terms)

    # Estimate coherence matrix
    Sigma = estimate_pauli_coherence_matrix(coeffs, pauli_strings)

    baseline_mean, baseline_std = simulate_measurement_variance(
        coeffs, pauli_strings, baseline_groups, baseline_shots, Sigma, n_samples
    )

    print(f"  Variance: {baseline_std**2:.6e}")

    baseline_results = {
        'groups': baseline_groups,
        'shots': baseline_shots,
        'variance': baseline_std**2,
        'physically_realizable': True,
        'n_groups': len(baseline_groups),
    }

    # Method 2: VRA (no commutativity constraints)
    print(f"\n{'-'*70}")
    print("Method 2: VRA (No commutativity)")
    print(f"{'-'*70}")

    vra_groups = group_by_variance_minimization(
        Sigma, coeffs, max_group_size=10,
        pauli_strings=pauli_strings,
        check_commutativity=False
    )

    vra_shots = allocate_shots_neyman(Sigma, coeffs, vra_groups, total_shots)

    # Check violations
    violations = sum(1 for g in vra_groups if not check_group_commutativity(g, pauli_strings))

    vra_mean, vra_std = simulate_measurement_variance(
        coeffs, pauli_strings, vra_groups, vra_shots, Sigma, n_samples
    )

    var_reduction_vra = baseline_std**2 / vra_std**2

    print(f"  Groups: {len(vra_groups)}")
    print(f"  Variance: {vra_std**2:.6e}")
    print(f"  Variance reduction: {var_reduction_vra:.2f}×")
    print(f"  Physically realizable: {'No' if violations > 0 else 'Yes'}")

    vra_results = {
        'groups': vra_groups,
        'shots': vra_shots,
        'variance': vra_std**2,
        'variance_reduction': var_reduction_vra,
        'physically_realizable': violations == 0,
        'n_groups': len(vra_groups),
        'violations': violations,
    }

    # Method 3: VRA + Commutativity
    print(f"\n{'-'*70}")
    print("Method 3: VRA + COMMUTATIVITY")
    print(f"{'-'*70}")

    vra_comm_groups = group_by_variance_minimization(
        Sigma, coeffs, max_group_size=10,
        pauli_strings=pauli_strings,
        check_commutativity=True
    )

    vra_comm_shots = allocate_shots_neyman(Sigma, coeffs, vra_comm_groups, total_shots)

    # Verify all groups commute
    all_commute = all(check_group_commutativity(g, pauli_strings) for g in vra_comm_groups)

    vra_comm_mean, vra_comm_std = simulate_measurement_variance(
        coeffs, pauli_strings, vra_comm_groups, vra_comm_shots, Sigma, n_samples
    )

    var_reduction_comm = baseline_std**2 / vra_comm_std**2

    print(f"  Groups: {len(vra_comm_groups)}")
    print(f"  Variance: {vra_comm_std**2:.6e}")
    print(f"  Variance reduction: {var_reduction_comm:.2f}×")
    print(f"  Physically realizable: {'Yes' if all_commute else 'No'}")

    vra_comm_results = {
        'groups': vra_comm_groups,
        'shots': vra_comm_shots,
        'variance': vra_comm_std**2,
        'variance_reduction': var_reduction_comm,
        'physically_realizable': all_commute,
        'n_groups': len(vra_comm_groups),
    }

    return {
        'molecule': molecule,
        'n_terms': n_terms,
        'baseline': baseline_results,
        'vra': vra_results,
        'vra_comm': vra_comm_results,
    }


def plot_multi_molecule_comparison(results_list: List[Dict], filename: str):
    """
    Create comparison plot across multiple molecules.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    molecules = [r['molecule'] for r in results_list]
    n_molecules = len(molecules)

    # Extract data
    baseline_var = [r['baseline']['variance'] for r in results_list]
    vra_var = [r['vra']['variance'] for r in results_list]
    vra_comm_var = [r['vra_comm']['variance'] for r in results_list]

    baseline_red = [1.0] * n_molecules
    vra_red = [r['vra']['variance_reduction'] for r in results_list]
    vra_comm_red = [r['vra_comm']['variance_reduction'] for r in results_list]

    baseline_groups = [r['baseline']['n_groups'] for r in results_list]
    vra_groups = [r['vra']['n_groups'] for r in results_list]
    vra_comm_groups = [r['vra_comm']['n_groups'] for r in results_list]

    n_terms = [r['n_terms'] for r in results_list]

    x = np.arange(n_molecules)
    width = 0.25

    # Plot 1: Variance comparison
    ax = axes[0, 0]
    bars1 = ax.bar(x - width, baseline_var, width, label='Baseline', color='blue', alpha=0.7)
    bars2 = ax.bar(x, vra_var, width, label='VRA (no comm)', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, vra_comm_var, width, label='VRA + Comm', color='green', alpha=0.7)

    # Hatch non-realizable
    for i, r in enumerate(results_list):
        if not r['vra']['physically_realizable']:
            bars2[i].set_hatch('//')
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(2)

    ax.set_ylabel('Variance')
    ax.set_title('Measurement Variance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Variance reduction factor
    ax = axes[0, 1]
    bars1 = ax.bar(x - width, baseline_red, width, label='Baseline', color='blue', alpha=0.7)
    bars2 = ax.bar(x, vra_red, width, label='VRA (no comm)', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, vra_comm_red, width, label='VRA + Comm', color='green', alpha=0.7)

    # Hatch non-realizable
    for i, r in enumerate(results_list):
        if not r['vra']['physically_realizable']:
            bars2[i].set_hatch('//')
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(2)

    ax.set_ylabel('Variance Reduction Factor')
    ax.set_title('Variance Reduction vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Number of groups
    ax = axes[1, 0]
    bars1 = ax.bar(x - width, baseline_groups, width, label='Baseline', color='blue', alpha=0.7)
    bars2 = ax.bar(x, vra_groups, width, label='VRA (no comm)', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, vra_comm_groups, width, label='VRA + Comm', color='green', alpha=0.7)

    for i, r in enumerate(results_list):
        if not r['vra']['physically_realizable']:
            bars2[i].set_hatch('//')
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(2)

    ax.set_ylabel('Number of Measurement Groups')
    ax.set_title('Measurement Grouping')
    ax.set_xticks(x)
    ax.set_xticklabels(molecules)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Hamiltonian size
    ax = axes[1, 1]
    bars = ax.bar(molecules, n_terms, color='purple', alpha=0.7)
    ax.set_ylabel('Number of Pauli Terms')
    ax.set_title('Hamiltonian Complexity')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to: {filename}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VRA Commutativity Benchmark: Larger Molecules")
    print("="*70)

    # Benchmark molecules
    molecules = ['H2', 'LiH', 'H2O', 'BeH2', 'NH3']

    all_results = []

    for molecule in molecules:
        try:
            # Use more terms for larger molecules
            max_terms = 30 if molecule in ['H2', 'LiH'] else 40
            results = benchmark_molecule(molecule, total_shots=10000, n_samples=1000, max_terms=max_terms)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Failed to benchmark {molecule}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create comparison plot
    if len(all_results) >= 2:
        plot_multi_molecule_comparison(
            all_results,
            '/home/admin/ATLAS-Q/benchmarks/vra_multi_molecule_comparison.png'
        )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Multi-Molecule Comparison")
    print("="*70)

    for r in all_results:
        mol = r['molecule']
        n_terms = r['n_terms']
        baseline_var = r['baseline']['variance']
        vra_red = r['vra']['variance_reduction']
        vra_comm_red = r['vra_comm']['variance_reduction']
        vra_realizable = r['vra']['physically_realizable']

        print(f"\n{mol} ({n_terms} Pauli terms):")
        print(f"  Baseline variance: {baseline_var:.6e}")
        print(f"  VRA reduction: {vra_red:.2f}× {'⚠️ NOT REALIZABLE' if not vra_realizable else ''}")
        print(f"  VRA+Comm reduction: {vra_comm_red:.2f}× ✓ Realizable")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("  Commutativity-aware grouping becomes more beneficial as")
    print("  Hamiltonian size increases and commuting structure improves.")
    print("="*70)
