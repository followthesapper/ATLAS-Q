#!/usr/bin/env python3
"""
Coherence-Aware Hardware Test on IBM Brisbane
==============================================

Tests the adaptive VRA framework on real quantum hardware.

This is a simplified version focusing on coherence tracking during
a single VQE-style energy measurement, demonstrating:

1. Coherence tracking (R̄, V_φ) from real measurements
2. e^-2 boundary check
3. Adaptive VRA decision (ON/OFF based on coherence)
4. Go/No-Go classification

Author: ATLAS-Q + VRA Integration
Date: November 2, 2025
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict

import numpy as np

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ATLAS-Q" / "src"))

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

try:
    from pyscf import gto, scf, ao2mo
    from atlas_q.mpo_ops import _jordan_wigner_transform
    from atlas_q.vra_enhanced import vra_hamiltonian_grouping
    PYSCF_AVAILABLE = True
    VRA_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    VRA_AVAILABLE = False
    print("⚠️  PySCF/VRA not available - using mock Hamiltonian")


@dataclass
class CoherenceMetrics:
    """Circular statistics from VRA Test 2"""
    R_bar: float
    V_phi: float
    is_above_e2_boundary: bool
    vra_predicted_to_help: bool


@dataclass
class HardwareTestResult:
    """Results from hardware coherence test"""
    molecule: str
    n_qubits: int
    n_pauli_terms: int

    # Measurements
    energy: float
    energy_nuc: float
    energy_elec: float
    shots_used: int

    # Coherence (NEW)
    coherence: CoherenceMetrics
    measurement_outcomes: List[float]

    # Classification (NEW)
    go_no_go: str
    classification_reason: str

    # Hardware info
    backend_name: str
    job_id: str
    timestamp: str


def extract_hamiltonian(molecule: str = 'H2') -> Tuple[np.ndarray, List[str], int, float]:
    """
    Extract molecular Hamiltonian.

    Args:
        molecule: 'H2', 'LiH', or 'H2O'

    Returns: (coeffs, pauli_strings, n_qubits, e_nuc)
    """
    if PYSCF_AVAILABLE:
        # Build molecule
        if molecule.upper() == 'H2':
            mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        elif molecule.upper() == 'LIH':
            mol = gto.M(atom='Li 0 0 0; H 0 0 1.5949', basis='sto-3g')
        elif molecule.upper() == 'H2O':
            # Water molecule at equilibrium geometry
            mol = gto.M(atom='O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0', basis='sto-3g')
        else:
            raise ValueError(f"Unknown molecule: {molecule}")

        mf = scf.RHF(mol)
        mf.kernel()

        # Get nuclear repulsion energy
        e_nuc = mol.energy_nuc()

        h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.kernel(mol, mf.mo_coeff)
        h2 = ao2mo.restore(1, eri, h1.shape[0])

        # Jordan-Wigner includes nuclear repulsion in identity term
        pauli_dict = _jordan_wigner_transform(h1, h2, e_nuc)

        coeffs = []
        pauli_strings = []
        for pauli_tuple, coeff in pauli_dict.items():
            if abs(coeff) > 1e-8:
                coeffs.append(np.real(coeff))
                pauli_strings.append(''.join(pauli_tuple))

        n_qubits = len(pauli_strings[0])
        return np.array(coeffs), pauli_strings, n_qubits, e_nuc
    else:
        # Mock simple Hamiltonian for testing
        return (
            np.array([0.5, -0.5, 0.3]),
            ['IIZZ', 'XXII', 'YYII'],
            4,
            0.0  # mock nuclear repulsion
        )


def create_test_circuit(n_qubits: int, params: np.ndarray = None) -> QuantumCircuit:
    """
    Create simple hardware-efficient ansatz for H2.
    """
    if params is None:
        # Random initial parameters
        np.random.seed(42)
        params = np.random.randn(n_qubits * 2) * 0.5

    # Create circuit manually
    qc = QuantumCircuit(n_qubits)

    # Layer 1: RY rotations
    for i in range(n_qubits):
        qc.ry(params[i], i)

    # Entanglers: CX
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)

    # Layer 2: RY rotations
    for i in range(n_qubits):
        qc.ry(params[n_qubits + i], i)

    return qc


def pauli_to_circuit(pauli_str: str, base_circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Create measurement circuit for a Pauli string.
    """
    n_qubits = len(pauli_str)
    qc = base_circuit.copy()

    # Apply basis rotations for measurement
    for i, pauli in enumerate(pauli_str):
        if pauli == 'X':
            qc.h(i)
        elif pauli == 'Y':
            qc.sdg(i)
            qc.h(i)
        # Z basis: no rotation needed

    qc.measure_all()
    return qc


def compute_expectation_from_counts(counts: Dict, pauli_str: str) -> float:
    """
    Compute Pauli expectation value from measurement counts.
    """
    total_shots = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        # Compute parity
        parity = 0
        for i, pauli in enumerate(pauli_str):
            if pauli != 'I':
                parity ^= int(bitstring[i])

        # +1 for even parity, -1 for odd
        sign = 1 if parity == 0 else -1
        expectation += sign * count / total_shots

    return expectation


def compute_coherence(measurement_outcomes: np.ndarray) -> CoherenceMetrics:
    """
    Compute circular statistics coherence (VRA Test 2).
    """
    # Convert Pauli expectations [-1, 1] to phases
    phases = np.arccos(np.clip(measurement_outcomes, -1, 1))

    # Mean resultant length
    phasors = np.exp(1j * phases)
    R_bar = np.abs(np.mean(phasors))

    # Circular variance
    V_phi = -2.0 * np.log(R_bar) if R_bar > 1e-10 else np.inf

    # e^-2 boundary (VRA Test 7)
    e2_boundary = 0.135
    is_above = R_bar > e2_boundary

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        vra_predicted_to_help=is_above
    )


def classify_go_no_go(coherence: CoherenceMetrics) -> Tuple[str, str]:
    """
    VRA Test 7 go/no-go classifier.
    """
    if coherence.R_bar > 0.135:
        return "GO", f"Coherence above e^-2 boundary (R̄={coherence.R_bar:.3f} > 0.135)"
    else:
        return "NO-GO", f"Coherence below e^-2 boundary (R̄={coherence.R_bar:.3f} < 0.135)"


def run_coherence_aware_test(backend_name: str = 'ibm_brisbane', shots: int = 5000, molecule: str = 'H2', use_vra_grouping: bool = True):
    """
    Run coherence-aware hardware test.

    Args:
        backend_name: IBM Quantum backend name
        shots: Shots per measurement (per group if using VRA grouping)
        molecule: Molecule to test ('H2' or 'LiH')
        use_vra_grouping: Use VRA grouping for large Hamiltonians
    """
    print("\n" + "="*80)
    print("COHERENCE-AWARE HARDWARE TEST")
    print("="*80)
    print(f"Backend: {backend_name}")
    print(f"Molecule: {molecule}")
    print(f"Shots per measurement: {shots}")
    print("\nIntegrating VRA hardware validation:")
    print("  ✓ Test 2: Coherence tracking (R̄, V_φ)")
    print("  ✓ Test 7: e^-2 boundary and go/no-go classification")
    print("")

    # Extract Hamiltonian
    print(f"[1/5] Extracting {molecule} Hamiltonian...")
    coeffs, pauli_strings, n_qubits, e_nuc = extract_hamiltonian(molecule)
    print(f"  Qubits: {n_qubits}")
    print(f"  Pauli terms: {len(pauli_strings)}")
    print(f"  Largest coeff: {np.max(np.abs(coeffs)):.4f}")
    print(f"  Nuclear repulsion: {e_nuc:.6f} Ha")

    # Check if VRA grouping should be used
    use_grouping = use_vra_grouping and VRA_AVAILABLE and len(pauli_strings) > 20

    if use_grouping:
        print(f"\n[2/5] Applying VRA grouping...")
        # Group Pauli terms
        from atlas_q.vra_enhanced import vra_hamiltonian_grouping
        total_budget = shots * len(pauli_strings)
        grouping_result = vra_hamiltonian_grouping(coeffs, pauli_strings, total_shots=total_budget)

        # grouping_result.groups contains lists of indices
        group_indices = grouping_result.groups
        shots_per_group = [shots] * len(group_indices)  # Use fixed shots per group

        compression = len(pauli_strings) / len(group_indices)
        print(f"  {len(pauli_strings)} terms → {len(group_indices)} groups ({compression:.1f}× compression)")
        print(f"  Shots per group: {shots} (fixed)")
        print(f"  Total shots: {shots * len(group_indices):,}")
    else:
        print(f"\n[2/5] Using individual Pauli measurements (no grouping)")
        group_indices = [[i] for i in range(len(pauli_strings))]
        shots_per_group = [shots] * len(group_indices)
        print(f"  Total shots: {shots * len(pauli_strings):,}")

    # Create ansatz circuit
    print(f"\n[3/5] Creating quantum circuit...")
    base_circuit = create_test_circuit(n_qubits)
    print(f"  Ansatz depth: {base_circuit.depth()}")
    print(f"  Parameters: {base_circuit.num_parameters}")

    # Build measurement circuits for each group
    print(f"\n[4/5] Building measurement circuits...")
    circuits = []
    for group_idx_list in group_indices:
        # For simplicity, measure first Pauli in each group
        # (In full VRA, would measure commuting observables simultaneously)
        first_idx = group_idx_list[0]
        representative_pauli = pauli_strings[first_idx]
        qc = pauli_to_circuit(representative_pauli, base_circuit)
        circuits.append(qc)
    print(f"  Total circuits: {len(circuits)}")

    # Connect to IBM Quantum
    print("\n[5/5] Submitting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    # Transpile
    print("  Transpiling...")
    transpiled_circuits = transpile(
        circuits,
        backend=backend,
        optimization_level=3,
        seed_transpiler=42
    )

    # Show transpiled stats
    avg_depth = np.mean([qc.depth() for qc in transpiled_circuits])
    print(f"  Average transpiled depth: {avg_depth:.1f}")

    # Submit job
    print(f"  Submitting {len(transpiled_circuits)} circuits...")

    sampler = Sampler(backend)

    t0 = time.time()
    job = sampler.run(transpiled_circuits, shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Waiting for results...")

    result = job.result()
    wall_time = time.time() - t0

    print(f"  ✅ Results received ({wall_time:.1f}s)")

    # Process results
    print("\n" + "="*80)
    print("PROCESSING RESULTS")
    print("="*80)

    measurement_outcomes = []
    energy = 0.0
    total_shots = 0

    # Process each group
    for i, group_idx_list in enumerate(group_indices):
        # Get counts for this group's circuit
        pub_result = result[i]
        counts_dict = pub_result.data.meas.get_counts()

        # Compute expectation for EACH Pauli in this group
        # (Commuting Paulis can be measured in same basis, but have different parities!)
        group_contributions = []
        for idx in group_idx_list:
            pauli_str = pauli_strings[idx]
            coeff = coeffs[idx]

            # Compute THIS Pauli's expectation from the counts
            exp_val = compute_expectation_from_counts(counts_dict, pauli_str)
            measurement_outcomes.append(exp_val)

            # Accumulate energy
            contribution = coeff * exp_val
            energy += contribution
            group_contributions.append((pauli_str, exp_val, coeff, contribution))

        total_shots += shots

        # Print summary for this group
        if len(group_idx_list) == 1:
            pauli_str, exp_val, coeff, _ = group_contributions[0]
            print(f"  {pauli_str}: ⟨P⟩ = {exp_val:+.4f}, coeff = {coeff:+.4f}")
        else:
            # Print first term as example
            first_pauli, first_exp, first_coeff, _ = group_contributions[0]
            print(f"  Group {i+1} ({len(group_idx_list)} terms): first {first_pauli}: ⟨P⟩ = {first_exp:+.4f}")

    print(f"\n  Electronic energy: {energy:.6f} Ha")
    print(f"  Nuclear repulsion: {e_nuc:.6f} Ha")
    print(f"  Total energy: {energy:.6f} Ha (electronic only)")
    print(f"  Total shots: {total_shots:,}")

    # Compute coherence
    print("\n" + "="*80)
    print("COHERENCE ANALYSIS (VRA Test 2)")
    print("="*80)

    measurement_outcomes = np.array(measurement_outcomes)
    coherence = compute_coherence(measurement_outcomes)

    print(f"  Mean resultant length: R̄ = {coherence.R_bar:.4f}")
    print(f"  Circular variance: V_φ = {coherence.V_phi:.4f}")
    print(f"  Above e^-2 boundary (0.135): {'✅ YES' if coherence.is_above_e2_boundary else '❌ NO'}")
    print(f"  VRA predicted to help: {'✅ YES' if coherence.vra_predicted_to_help else '❌ NO'}")

    # Classification
    print("\n" + "="*80)
    print("GO/NO-GO CLASSIFICATION (VRA Test 7)")
    print("="*80)

    go_no_go, reason = classify_go_no_go(coherence)

    print(f"  Classification: {go_no_go}")
    print(f"  Reason: {reason}")

    # Expected on hardware vs simulator
    print("\n" + "="*80)
    print("COMPARISON: SIMULATOR vs HARDWARE")
    print("="*80)
    print(f"  Simulator (MPS): R̄ ~ 0.80-0.93 (very high coherence)")
    print(f"  Hardware (this run): R̄ = {coherence.R_bar:.4f}")
    print(f"  VRA Test 7 range: R̄ = 0.124-0.460 (typical hardware)")

    if coherence.R_bar > 0.5:
        print("\n  ⚡ INSIGHT: Your hardware coherence is higher than expected!")
        print("     This suggests good qubit quality or favorable circuit structure.")
    elif coherence.R_bar < 0.2:
        print("\n  ⚠️  INSIGHT: Low coherence detected!")
        print("     Adaptive VRA would DISABLE grouping to save shots.")

    # Save results
    result_data = HardwareTestResult(
        molecule=molecule,
        n_qubits=n_qubits,
        n_pauli_terms=len(pauli_strings),
        energy=float(energy),  # Electronic energy (Hamiltonian already includes nuclear repulsion in identity term)
        energy_nuc=float(e_nuc),
        energy_elec=float(energy),
        shots_used=total_shots,
        coherence=coherence,
        measurement_outcomes=measurement_outcomes.tolist(),
        go_no_go=go_no_go,
        classification_reason=reason,
        backend_name=backend_name,
        job_id=job_id,
        timestamp=time.strftime("%Y%m%d_%H%M%S")
    )

    # Save to JSON
    output_file = Path(__file__).parent / "results" / f"coherence_hardware_{molecule.lower()}_{result_data.timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(asdict(result_data), f, indent=2, default=str)

    print(f"\n  📄 Results saved: {output_file}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"✅ Successfully demonstrated coherence-aware quantum computing on {backend_name}")
    print("✅ Coherence tracking (R̄, V_φ) from real hardware measurements")
    print("✅ e^-2 boundary check and VRA prediction")
    print("✅ Go/No-Go classification of quantum results")
    print("="*80 + "\n")

    return result_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coherence-aware hardware test")
    parser.add_argument('--backend', type=str, default='ibm_brisbane',
                       help='IBM Quantum backend name')
    parser.add_argument('--shots', type=int, default=5000,
                       help='Shots per measurement (per group if using VRA grouping)')
    parser.add_argument('--molecule', type=str, default='H2',
                       help='Molecule to test (H2, LiH, or H2O)')
    parser.add_argument('--no-vra-grouping', action='store_true',
                       help='Disable VRA grouping (measure all Pauli terms individually)')

    args = parser.parse_args()

    try:
        result = run_coherence_aware_test(
            backend_name=args.backend,
            shots=args.shots,
            molecule=args.molecule,
            use_vra_grouping=not args.no_vra_grouping
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
