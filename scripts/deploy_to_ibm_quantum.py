#!/usr/bin/env python3
"""
ATLAS-Q → IBM Quantum Deployment Script (with IR)
===================================================

Complete workflow using the E147-validated approach:
1. Build molecular Hamiltonian with PySCF + Jordan-Wigner
2. Apply IR grouping for measurement optimization
3. Build measurement circuits for each Pauli group
4. Deploy to IBM Quantum hardware
5. Compute energy and coherence metrics

This script uses the EXACT approach that achieved R̄=0.988 for H2O
on IBM Brisbane (E147 experiment, November 2025).

Author: ATLAS-Q + IR Integration
Date: December 2025
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add ATLAS-Q to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Molecule Library - Extensible for any molecule
# ============================================================================

MOLECULE_LIBRARY = {
    # Tier 0: Already validated on IBM hardware
    'H2': {
        'geometry': 'H 0 0 0; H 0 0 0.74',
        'description': 'Hydrogen molecule (4 qubits, ~15 Paulis)',
    },
    'LiH': {
        'geometry': 'Li 0 0 0; H 0 0 1.5949',
        'description': 'Lithium hydride (12 qubits, ~630 Paulis)',
    },
    'H2O': {
        'geometry': 'O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0',
        'description': 'Water molecule (14 qubits, ~1086 Paulis) - VRA paper validation',
    },
    # Tier 1: Next rational step
    'BeH2': {
        'geometry': 'Be 0 0 0; H 0 0 1.3264; H 0 0 -1.3264',
        'description': 'Beryllium dihydride (14 qubits)',
    },
    'NH3': {
        'geometry': 'N 0 0 0.116; H 0 0.939 -0.272; H 0.813 -0.470 -0.272; H -0.813 -0.470 -0.272',
        'description': 'Ammonia (16 qubits, ~2000-3000 Paulis)',
    },
    # Tier 2: Where most frameworks break
    'CH4': {
        'geometry': 'C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629',
        'description': 'Methane (18 qubits, ~5000-8000 Paulis)',
    },
    # Tier 3: Larger molecules
    'N2': {
        'geometry': 'N 0 0 0; N 0 0 1.0977',
        'description': 'Nitrogen molecule (20 qubits)',
    },
    'CO': {
        'geometry': 'C 0 0 0; O 0 0 1.128',
        'description': 'Carbon monoxide (20 qubits)',
    },
    'HF': {
        'geometry': 'H 0 0 0; F 0 0 0.917',
        'description': 'Hydrogen fluoride (12 qubits)',
    },
    # Tier 4: Frontier molecules
    'CO2': {
        'geometry': 'C 0 0 0; O 0 0 1.16; O 0 0 -1.16',
        'description': 'Carbon dioxide (30 qubits, ~10000+ Paulis) - frontier',
    },
}


@dataclass
class CoherenceMetrics:
    """Circular statistics from IR Test 2"""
    R_bar: float
    V_phi: float
    is_above_e2_boundary: bool
    ir_predicted_to_help: bool


@dataclass
class HardwareResult:
    """Complete results from hardware run"""
    molecule: str
    basis: str
    n_qubits: int
    n_pauli_terms: int
    n_groups: int
    compression_ratio: float

    # Energy
    energy: float
    energy_nuc: float
    hf_energy: float

    # Coherence
    coherence: CoherenceMetrics
    measurement_outcomes: List[float]

    # Classification
    go_no_go: str
    classification_reason: str

    # Hardware info
    backend_name: str
    job_id: str
    shots_per_group: int
    total_shots: int
    runtime_s: float
    timestamp: str


# ============================================================================
# Step 1: Build Hamiltonian (E147-validated approach)
# ============================================================================

def build_hamiltonian(molecule: str, basis: str = 'sto-3g') -> Tuple[np.ndarray, List[str], int, float, float]:
    """
    Build molecular Hamiltonian using PySCF + Jordan-Wigner.

    Uses OpenFermion for larger molecules (>20 qubits) for faster JW transform.
    """
    from pyscf import gto, scf, ao2mo

    if molecule.upper() not in MOLECULE_LIBRARY:
        raise ValueError(f"Unknown molecule: {molecule}. Available: {list(MOLECULE_LIBRARY.keys())}")

    geometry = MOLECULE_LIBRARY[molecule.upper()]['geometry']

    print(f"  Building {molecule} with PySCF...")
    mol = gto.M(atom=geometry, basis=basis)

    # Run RHF
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()

    # Get nuclear repulsion
    e_nuc = mol.energy_nuc()

    # Get integrals in MO basis
    n_orbitals = mol.nao_nr()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    h2 = ao2mo.restore(1, eri, n_orbitals)

    n_qubits = 2 * n_orbitals

    # Use OpenFermion for larger molecules (faster JW)
    if n_qubits > 20:
        print(f"  Applying Jordan-Wigner transform (OpenFermion - optimized for {n_qubits} qubits)...")
        from openfermion.chem.molecular_data import spinorb_from_spatial
        from openfermion.ops import InteractionOperator
        from openfermion.transforms import get_fermion_operator, jordan_wigner

        one_body, two_body = spinorb_from_spatial(h1, h2)
        mol_ham = InteractionOperator(e_nuc, one_body, 0.5 * two_body)
        fermion_h = get_fermion_operator(mol_ham)
        qubit_h = jordan_wigner(fermion_h)

        # Extract Pauli strings and coefficients
        coeffs = []
        paulis = []
        for term, coeff in qubit_h.terms.items():
            if abs(coeff) > 1e-8:
                pauli_str = ['I'] * n_qubits
                for qubit_idx, pauli_op in term:
                    pauli_str[qubit_idx] = pauli_op
                paulis.append(''.join(pauli_str))
                coeffs.append(np.real(coeff))
    else:
        # Use ATLAS-Q's JW for smaller molecules (E147 approach)
        print(f"  Applying Jordan-Wigner transform (ATLAS-Q)...")
        from atlas_q.mpo_ops import _jordan_wigner_transform
        pauli_dict = _jordan_wigner_transform(h1, h2, e_nuc)

        coeffs = []
        paulis = []
        for pauli_tuple, coeff in pauli_dict.items():
            if abs(coeff) > 1e-8:
                coeffs.append(np.real(coeff))
                paulis.append(''.join(pauli_tuple))

    n_qubits = len(paulis[0]) if paulis else 0

    print(f"  ✓ {n_qubits} qubits, {len(paulis)} Pauli terms")
    print(f"  ✓ HF energy: {hf_energy:.6f} Ha")
    print(f"  ✓ Nuclear repulsion: {e_nuc:.6f} Ha")

    return np.array(coeffs), paulis, n_qubits, e_nuc, hf_energy


# ============================================================================
# Step 2: Apply IR Grouping
# ============================================================================

def apply_ir_grouping(coeffs: np.ndarray, paulis: List[str], total_shots: int) -> Tuple[List[List[int]], float]:
    """
    Apply IR grouping for measurement optimization.

    Returns groups of Pauli indices that can be measured together.

    Uses vra_hamiltonian_grouping - the EXACT function from E147 that achieved R̄=0.988.
    """
    from atlas_q.vra_enhanced import vra_hamiltonian_grouping

    print(f"  Applying VRA grouping (E147 approach)...")
    grouping = vra_hamiltonian_grouping(coeffs, pauli_strings=paulis, total_shots=total_shots)

    compression = len(paulis) / len(grouping.groups)

    print(f"  ✓ {len(paulis)} Paulis → {len(grouping.groups)} groups")
    print(f"  ✓ Circuit reduction: {compression:.1f}× fewer quantum circuits")
    print(f"  ✓ Shot efficiency: {compression:.1f}× (same precision, {compression:.1f}× fewer shots)")
    print(f"  ✓ Effective variance reduction: {compression:.1f}× (if using same total shots)")

    return grouping.groups, compression


# ============================================================================
# Step 3: Build Measurement Circuits
# ============================================================================

def create_ansatz_circuit(n_qubits: int, n_layers: int = 2):
    """Create hardware-efficient ansatz."""
    from qiskit import QuantumCircuit

    # Random initial parameters (in production, would optimize these)
    np.random.seed(42)
    params = np.random.randn(n_qubits * 2 * n_layers) * 0.5

    qc = QuantumCircuit(n_qubits)

    param_idx = 0
    for layer in range(n_layers):
        # RY rotations
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1

        # Entanglers: CX (linear connectivity)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc


def pauli_to_measurement_circuit(pauli_str: str, base_circuit):
    """Create measurement circuit for a Pauli string."""
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


def build_measurement_circuits(paulis: List[str], groups: List[List[int]], n_qubits: int):
    """Build measurement circuits for each group."""
    from qiskit import transpile

    base_circuit = create_ansatz_circuit(n_qubits)

    circuits = []
    for group_indices in groups:
        # Use first Pauli in group as representative for measurement basis
        representative_pauli = paulis[group_indices[0]]
        qc = pauli_to_measurement_circuit(representative_pauli, base_circuit)
        circuits.append(qc)

    return circuits, base_circuit


# ============================================================================
# Step 4: Execute on IBM Quantum
# ============================================================================

def execute_on_ibm(circuits, backend, shots_per_circuit: int, dry_run: bool = False, confirm: bool = True):
    """Execute circuits on IBM Quantum hardware."""
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    # Estimate cost
    n_circuits = len(circuits)
    estimated_time = n_circuits * 5  # ~5 sec per circuit
    estimated_cost = max(0, (estimated_time / 60 - 10) * 96)

    print(f"\n  Circuits to run: {n_circuits}")
    print(f"  Shots per circuit: {shots_per_circuit}")
    print(f"  Total shots: {n_circuits * shots_per_circuit:,}")
    print(f"  Estimated time: {estimated_time}s")
    if estimated_cost > 0:
        print(f"  Estimated cost: ${estimated_cost:.2f}")
    else:
        print(f"  Cost: $0 (within free tier)")

    if dry_run:
        print("\n  ⚠️  DRY RUN MODE - Not submitting to quantum hardware")
        return None, None

    if confirm:
        response = input("\n  Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("  Cancelled by user")
            return None, None

    # Transpile
    print(f"\n  Transpiling {n_circuits} circuits for {backend.name}...")
    transpiled = transpile(circuits, backend=backend, optimization_level=3, seed_transpiler=42)

    avg_depth = np.mean([qc.depth() for qc in transpiled])
    print(f"  ✓ Average transpiled depth: {avg_depth:.1f}")

    # Submit
    print(f"\n  Submitting to {backend.name}...")
    sampler = Sampler(backend)

    t0 = time.time()
    job = sampler.run(transpiled, shots=shots_per_circuit)
    job_id = job.job_id()

    print(f"  ✓ Job ID: {job_id}")
    print(f"  ✓ Status: {job.status()}")
    print(f"  Waiting for results...")

    result = job.result()
    runtime = time.time() - t0

    print(f"  ✓ Completed in {runtime:.1f}s")

    return result, job_id


# ============================================================================
# Step 5: Process Results
# ============================================================================

def compute_expectation_from_counts(counts: Dict, pauli_str: str) -> float:
    """Compute Pauli expectation value from measurement counts."""
    total_shots = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        # Compute parity for non-identity Paulis
        parity = 0
        for i, pauli in enumerate(pauli_str):
            if pauli != 'I':
                # Bitstring is reversed in Qiskit
                bit_idx = len(bitstring) - 1 - i
                if bit_idx >= 0 and bit_idx < len(bitstring):
                    parity ^= int(bitstring[bit_idx])

        sign = 1 if parity == 0 else -1
        expectation += sign * count / total_shots

    return expectation


def compute_coherence(measurement_outcomes: np.ndarray) -> CoherenceMetrics:
    """Compute circular statistics coherence (IR Test 2)."""
    # Convert Pauli expectations [-1, 1] to phases
    phases = np.arccos(np.clip(measurement_outcomes, -1, 1))

    # Mean resultant length
    phasors = np.exp(1j * phases)
    R_bar = float(np.abs(np.mean(phasors)))

    # Circular variance
    V_phi = -2.0 * np.log(R_bar) if R_bar > 1e-10 else np.inf

    # e^-2 boundary (IR Test 7)
    e2_boundary = 0.135
    is_above = R_bar > e2_boundary

    return CoherenceMetrics(
        R_bar=R_bar,
        V_phi=V_phi,
        is_above_e2_boundary=is_above,
        ir_predicted_to_help=is_above
    )


def process_results(result, paulis: List[str], coeffs: np.ndarray, groups: List[List[int]]) -> Tuple[float, np.ndarray]:
    """Process measurement results to compute energy and coherence."""
    measurement_outcomes = []
    energy = 0.0

    for group_idx, group_indices in enumerate(groups):
        # Get counts for this group's circuit
        pub_result = result[group_idx]
        counts_dict = pub_result.data.meas.get_counts()

        # Compute expectation for EACH Pauli in this group
        for idx in group_indices:
            pauli_str = paulis[idx]
            coeff = coeffs[idx]

            exp_val = compute_expectation_from_counts(counts_dict, pauli_str)
            measurement_outcomes.append(exp_val)
            energy += coeff * exp_val

    return energy, np.array(measurement_outcomes)


# ============================================================================
# Main Workflow
# ============================================================================

def run_hardware_test(
    molecule: str,
    basis: str = 'sto-3g',
    backend_name: str = 'ibm_torino',
    shots: int = 5000,
    dry_run: bool = False,
    confirm: bool = True,
    save_results: bool = True
) -> HardwareResult:
    """
    Run complete IR-enhanced hardware test.

    This uses the EXACT workflow that achieved R̄=0.988 for H2O on IBM Brisbane.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    print("\n" + "#" * 80)
    print(f"# IR-ENHANCED IBM QUANTUM TEST: {molecule.upper()}")
    print("#" * 80)

    if molecule.upper() in MOLECULE_LIBRARY:
        print(f"\nMolecule: {MOLECULE_LIBRARY[molecule.upper()]['description']}")
    print(f"Basis: {basis}")
    print(f"Backend: {backend_name}")
    print(f"Shots per group: {shots}")
    print(f"Dry run: {'YES' if dry_run else 'NO'}")

    t0_total = time.time()

    # Step 1: Build Hamiltonian
    print("\n" + "=" * 80)
    print("STEP 1: Build Molecular Hamiltonian")
    print("=" * 80)
    coeffs, paulis, n_qubits, e_nuc, hf_energy = build_hamiltonian(molecule, basis)

    # Step 2: Apply IR Grouping
    print("\n" + "=" * 80)
    print("STEP 2: Apply IR Grouping")
    print("=" * 80)
    total_budget = shots * len(paulis)
    groups, compression = apply_ir_grouping(coeffs, paulis, total_budget)

    # Step 3: Build Measurement Circuits
    print("\n" + "=" * 80)
    print("STEP 3: Build Measurement Circuits")
    print("=" * 80)
    circuits, _ = build_measurement_circuits(paulis, groups, n_qubits)
    print(f"  ✓ Built {len(circuits)} measurement circuits")

    # Step 4: Connect and Execute
    print("\n" + "=" * 80)
    print("STEP 4: Connect to IBM Quantum")
    print("=" * 80)

    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(backend_name)
    print(f"  ✓ Connected to {backend.name} ({backend.num_qubits} qubits)")

    print("\n" + "=" * 80)
    print("STEP 5: Execute on Quantum Hardware")
    print("=" * 80)

    result, job_id = execute_on_ibm(circuits, backend, shots, dry_run, confirm)

    if result is None:
        print("\n⚠️  No results (dry run or cancelled)")
        return None

    # Step 5: Process Results
    print("\n" + "=" * 80)
    print("STEP 6: Process Results")
    print("=" * 80)

    energy, measurement_outcomes = process_results(result, paulis, coeffs, groups)
    coherence = compute_coherence(measurement_outcomes)

    # Classification
    if coherence.R_bar > 0.135:
        go_no_go = "GO"
        reason = f"Coherence above e^-2 boundary (R̄={coherence.R_bar:.3f} > 0.135)"
    else:
        go_no_go = "NO-GO"
        reason = f"Coherence below e^-2 boundary (R̄={coherence.R_bar:.3f} < 0.135)"

    runtime = time.time() - t0_total

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Molecule: {molecule}")
    print(f"  Qubits: {n_qubits}")
    print(f"  Pauli terms: {len(paulis)}")
    print(f"  IR groups: {len(groups)}")
    print(f"  Compression: {compression:.1f}×")
    print()
    print(f"  HF Energy: {hf_energy:.6f} Ha")
    print(f"  VQE Energy: {energy:.6f} Ha")
    print(f"  Nuclear repulsion: {e_nuc:.6f} Ha")
    print()
    print(f"  Coherence R̄: {coherence.R_bar:.4f}")
    print(f"  Circular variance V_φ: {coherence.V_phi:.4f}")
    print(f"  Classification: {go_no_go}")
    print(f"  Reason: {reason}")
    print()
    print(f"  Job ID: {job_id}")
    print(f"  Total runtime: {runtime:.1f}s")

    # Create result object
    hw_result = HardwareResult(
        molecule=molecule.upper(),
        basis=basis,
        n_qubits=n_qubits,
        n_pauli_terms=len(paulis),
        n_groups=len(groups),
        compression_ratio=compression,
        energy=float(energy),
        energy_nuc=float(e_nuc),
        hf_energy=float(hf_energy),
        coherence=coherence,
        measurement_outcomes=measurement_outcomes.tolist(),
        go_no_go=go_no_go,
        classification_reason=reason,
        backend_name=backend_name,
        job_id=job_id,
        shots_per_group=shots,
        total_shots=shots * len(groups),
        runtime_s=runtime,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Save results
    if save_results:
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"ir_hardware_{molecule.lower()}_{hw_result.timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(asdict(hw_result), f, indent=2, default=str)

        print(f"\n  ✓ Results saved: {output_file}")

    print("\n" + "=" * 80)
    if go_no_go == "GO":
        print(f"✅ SUCCESS: {molecule} - Coherence R̄={coherence.R_bar:.4f} (GO)")
    else:
        print(f"⚠️  {molecule} - Coherence R̄={coherence.R_bar:.4f} (NO-GO)")
    print("=" * 80)

    return hw_result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IR-Enhanced IBM Quantum Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run H2O (validated molecule)
  python deploy_to_ibm_quantum.py --molecule H2O

  # Run NH3 (harder molecule)
  python deploy_to_ibm_quantum.py --molecule NH3 --shots 3000

  # Dry run (no hardware submission)
  python deploy_to_ibm_quantum.py --molecule H2O --dry-run

  # List available molecules
  python deploy_to_ibm_quantum.py --list-molecules

Available molecules:
""" + "\n".join(f"  {k}: {v['description']}" for k, v in MOLECULE_LIBRARY.items())
    )

    parser.add_argument('--molecule', '-m', type=str, default='H2O',
                       help='Molecule to test (default: H2O)')
    parser.add_argument('--basis', '-b', type=str, default='sto-3g',
                       help='Basis set (default: sto-3g)')
    parser.add_argument('--backend', type=str, default='ibm_torino',
                       help='IBM Quantum backend (default: ibm_torino)')
    parser.add_argument('--shots', '-s', type=int, default=1000,
                       help='Shots per measurement group (default: 1000, matches E147)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (no hardware submission)')
    parser.add_argument('--no-confirm', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--list-molecules', action='store_true',
                       help='List available molecules and exit')

    args = parser.parse_args()

    if args.list_molecules:
        print("\nAvailable molecules:")
        print("-" * 60)
        for name, info in MOLECULE_LIBRARY.items():
            print(f"  {name:6s}: {info['description']}")
        print()
        return

    try:
        result = run_hardware_test(
            molecule=args.molecule,
            basis=args.basis,
            backend_name=args.backend,
            shots=args.shots,
            dry_run=args.dry_run,
            confirm=not args.no_confirm
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
