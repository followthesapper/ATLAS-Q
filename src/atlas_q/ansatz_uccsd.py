#!/usr/bin/env python3
"""
ATLAS-Q: UCCSD Ansatz Construction
----------------------------------

Implements the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz
for molecular VQE simulations.

Features:
- Builds fermionic excitation operators from PySCF molecular integrals
- Uses OpenFermion for operator algebra and Jordan–Wigner mapping
- Produces qubit-space parameterized operators compatible with ATLAS-Q VQE
- Fully compatible with GPU execution (torch)

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from openfermion import (
        MolecularData,
        QubitOperator,
        get_fermion_operator,
        jordan_wigner,
        uccsd_singlet_generator,
        uccsd_singlet_paramsize,
    )
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False


def _to_pauli_strings(qubit_op: QubitOperator, n_qubits: int) -> List[Tuple[complex, str]]:
    """
    Convert an OpenFermion QubitOperator to a list of (coeff, pauli_string) tuples.

    This avoids building exponentially large matrices by keeping Pauli strings
    as lightweight string representations.

    Args:
        qubit_op: OpenFermion QubitOperator
        n_qubits: Number of qubits

    Returns:
        List of (coefficient, pauli_string) tuples
        pauli_string is like "IXYZ" for I⊗X⊗Y⊗Z
    """
    ops = []
    for term, coeff in qubit_op.terms.items():
        if abs(coeff) < 1e-12:
            continue

        # Build Pauli string representation
        # term is like ((0, 'X'), (2, 'Y')) for X_0 Y_2
        pauli_string = ['I'] * n_qubits
        for qubit_idx, pauli_op in term:
            pauli_string[qubit_idx] = pauli_op

        ops.append((complex(coeff), ''.join(pauli_string)))

    return ops


def _perm_interleaved_to_blocked(n_orb: int) -> List[int]:
    """
    Build permutation to convert interleaved → blocked spin-orbital ordering.

    OpenFermion UCCSD uses interleaved: [α0, β0, α1, β1, ..., α(n-1), β(n-1)]
    ATLAS-Q Hamiltonian uses blocked: [α0, α1, ..., α(n-1), β0, β1, ..., β(n-1)]

    Args:
        n_orb: Number of spatial orbitals

    Returns:
        Permutation list where perm[new_idx] = old_idx
    """
    perm = []
    # Alpha orbitals: interleaved positions 0,2,4,... → blocked positions 0,1,2,...
    for r in range(n_orb):
        perm.append(r * 2)  # α_r at interleaved position 2r → blocked position r
    # Beta orbitals: interleaved positions 1,3,5,... → blocked positions n_orb, n_orb+1, ...
    for r in range(n_orb):
        perm.append(r * 2 + 1)  # β_r at interleaved position 2r+1 → blocked position n_orb+r
    return perm


def _permute_pauli_string(ps: str, perm: List[int]) -> str:
    """
    Permute qubits in a Pauli string according to given permutation.

    Args:
        ps: Pauli string like "IXYZ" (in interleaved ordering from OpenFermion)
        perm: Permutation list where perm[blocked_idx] = interleaved_idx

    Returns:
        Permuted Pauli string (in blocked ordering for ATLAS-Q)
    """
    src = list(ps)  # Source string in interleaved ordering
    out = ['I'] * len(src)
    # Inverse permutation: if perm[blocked_idx] = interleaved_idx,
    # then out[blocked_idx] = src[interleaved_idx]
    for blocked_idx, interleaved_idx in enumerate(perm):
        out[blocked_idx] = src[interleaved_idx]
    return ''.join(out)


def _build_param_groups(n_qubits: int, n_elec: int) -> List[List[Tuple[complex, str]]]:
    """
    Build per-parameter groups for UCCSD ansatz.

    Each UCCSD excitation parameter k expands to multiple Pauli strings with fixed
    coefficients. This function isolates each parameter to determine which Pauli
    strings belong to it.

    Args:
        n_qubits: Number of qubits
        n_elec: Number of electrons

    Returns:
        List where entry k contains all (coeff, pauli_string) pairs for parameter k
    """
    from openfermion import jordan_wigner, uccsd_singlet_generator, uccsd_singlet_paramsize

    groups: List[List[Tuple[complex, str]]] = []
    n_params = uccsd_singlet_paramsize(n_qubits, n_elec)

    for k in range(n_params):
        # Isolate excitation k by setting only its amplitude to 1
        amps = np.zeros(n_params, dtype=float)
        amps[k] = 1.0

        # Build generator for this single excitation
        fermion_gen_k = uccsd_singlet_generator(
            amps, n_qubits, n_elec, anti_hermitian=True
        )

        # Convert to qubit operators (uses interleaved spin-orbital ordering)
        qubit_op_k = jordan_wigner(fermion_gen_k)

        # Convert to Pauli strings (interleaved indices from OpenFermion)
        group_raw = _to_pauli_strings(qubit_op_k, n_qubits)

        # TEMPORARY: Disable permutation to test if it's causing the issue
        # TODO: Investigate correct spin-orbital ordering
        # Permute to BLOCKED indices to match Hamiltonian + HF init
        # OpenFermion uses interleaved [α0,β0,α1,β1,...] but ATLAS-Q uses blocked [α0,α1,...,β0,β1,...]
        # n_orb = n_qubits // 2
        # perm = _perm_interleaved_to_blocked(n_orb)
        # group = [(coeff, _permute_pauli_string(ps, perm)) for (coeff, ps) in group_raw]
        group = group_raw  # NO PERMUTATION for now

        groups.append(group)

    return groups


def build_uccsd_ansatz(
    molecule: str = "H2",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    mapping: str = "jordan_wigner",
    device: str = "cuda",
    dtype=torch.complex128,
) -> Dict[str, Any]:
    """
    Build a UCCSD ansatz for a given molecule.

    The UCCSD ansatz is the gold standard for molecular VQE, providing:
    - Particle number conservation
    - Spin symmetry
    - Systematic improvement with system size
    - Chemical accuracy for small molecules

    Args:
        molecule: Molecular geometry string or name ('H2', 'LiH', 'H2O')
        basis: Gaussian basis set (sto-3g, 6-31g, cc-pvdz, etc.)
        charge: Total molecular charge
        spin: Spin multiplicity (2S, where S is total spin)
        mapping: Fermion-to-qubit mapping ('jordan_wigner')
        device: torch device ('cuda' or 'cpu')
        dtype: torch dtype (torch.complex128 recommended)

    Returns:
        Dictionary containing:
            - n_qubits: Number of qubits
            - n_electrons: Number of electrons
            - hf_state: Hartree-Fock reference state
            - uccsd_ops: List of (coeff, operator) tuples
            - qubit_op: OpenFermion QubitOperator
            - hf_energy: Hartree-Fock energy (Ha)
            - nuclear_repulsion: Nuclear repulsion energy (Ha)
    """
    if not OPENFERMION_AVAILABLE:
        raise ImportError(
            "OpenFermion required for UCCSD ansatz. "
            "Install with: pip install openfermion openfermionpyscf"
        )

    try:
        from openfermionpyscf import run_pyscf
        from pyscf import ao2mo, gto, scf
    except ImportError:
        raise ImportError(
            "PySCF and OpenFermion-PySCF required. "
            "Install with: pip install pyscf openfermionpyscf"
        )

    # Parse molecule specification
    if molecule in ['H2', 'h2']:
        mol_spec = [['H', [0, 0, 0]], ['H', [0, 0, 0.74]]]
    elif molecule in ['LiH', 'lih']:
        mol_spec = [['Li', [0, 0, 0]], ['H', [0, 0, 1.5949]]]
    elif molecule in ['H2O', 'h2o']:
        mol_spec = [
            ['O', [0.0000, 0.0000, 0.1173]],
            ['H', [0.0000, 0.7572, -0.4692]],
            ['H', [0.0000, -0.7572, -0.4692]],
        ]
    elif molecule in ['BeH2', 'beh2']:
        mol_spec = [['Be', [0, 0, 0]], ['H', [0, 0, 1.3264]], ['H', [0, 0, -1.3264]]]
    else:
        # Assume it's a custom geometry string
        mol_spec = molecule

    # Build molecular system with PySCF
    mol = gto.M(atom=mol_spec, basis=basis, charge=charge, spin=spin)
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.kernel()

    hf_energy = mf.e_tot
    nuclear_repulsion = mol.energy_nuc()

    # Get molecular data using OpenFermion-PySCF
    molecule_of = MolecularData(
        geometry=mol_spec,
        basis=basis,
        charge=charge,
        multiplicity=spin + 1,
    )

    molecule_of = run_pyscf(molecule_of, run_scf=1, run_mp2=0, run_cisd=0, run_ccsd=0, run_fci=0)

    n_orb = molecule_of.n_orbitals
    n_elec = molecule_of.n_electrons
    n_qubits = 2 * n_orb

    # Build UCCSD generator (singlet version for closed-shell)
    # This generates the anti-Hermitian operator T - T†

    # Get number of UCCSD parameters
    n_params = uccsd_singlet_paramsize(n_qubits, n_elec)

    # Build parameter groups: each UCCSD excitation parameter k expands to
    # multiple Pauli strings with fixed coefficients. We need to group them
    # correctly so that all Pauli strings belonging to one excitation share
    # the same θ parameter during optimization.
    param_groups = _build_param_groups(n_qubits, n_elec)

    # Construct HF reference state
    # IMPORTANT: Must match the spin-orbital ordering used in molecular_hamiltonian_from_specs
    # The Hamiltonian uses BLOCKED ordering: [alpha_0, alpha_1, ..., beta_0, beta_1, ...]
    # So for n_orb=2, n_elec=2 (H2): occupy alpha_0 and beta_0 → [1, 0, 1, 0]
    hf_state = np.zeros(n_qubits, dtype=int)

    # Fill electrons into orbitals with spin-pairing (RHF assumption)
    n_doubly_occupied = n_elec // 2
    n_singly_occupied = n_elec % 2

    # Doubly occupied orbitals: put alpha and beta electrons
    for i in range(n_doubly_occupied):
        hf_state[i] = 1              # Alpha electron in orbital i
        hf_state[i + n_orb] = 1      # Beta electron in orbital i

    # Singly occupied orbitals (if odd number of electrons): put alpha electron
    for i in range(n_singly_occupied):
        hf_state[n_doubly_occupied + i] = 1  # Alpha electron

    # Log spin-orbital ordering
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"UCCSD ansatz for {molecule}/{basis}")
    logger.info(f"  HF state uses BLOCKED spin-orbital ordering: [α₀...αₙ₋₁, β₀...βₙ₋₁]")
    logger.info(f"  HF occupation: {hf_state}")
    logger.info(f"  Number of qubits: {n_qubits}, electrons: {n_elec}")

    return {
        "n_qubits": n_qubits,
        "n_electrons": n_elec,
        "n_orbitals": n_orb,
        "hf_state": hf_state,
        "param_groups": param_groups,  # List of lists: param_groups[k] = [(coeff, "IXYZ...")] for parameter k
        "hf_energy": hf_energy,
        "nuclear_repulsion": nuclear_repulsion,
        "n_parameters": n_params,  # TRUE UCCSD parameter count (not number of Pauli strings!)
        "device": device,
        "dtype": dtype,
    }


class UCCSDAnsatz:
    """
    UCCSD Ansatz for molecular VQE compatible with ATLAS-Q AdaptiveMPS

    Usage:
        ansatz = UCCSDAnsatz(molecule='H2', basis='sto-3g', device='cuda')
        mps = ansatz.prepare_hf_state()  # Initialize to Hartree-Fock
        ansatz.apply(mps, params)  # Apply UCCSD transformation
    """

    def __init__(self, molecule: str = 'H2', basis: str = 'sto-3g', device: str = 'cuda', dtype=torch.complex128):
        self.uccsd = build_uccsd_ansatz(molecule, basis, device=device, dtype=dtype)
        self.n_qubits = self.uccsd['n_qubits']
        self.n_parameters = self.uccsd['n_parameters']
        self.device = device
        self.dtype = dtype
        self.param_groups = self.uccsd['param_groups']  # List of lists: param_groups[k] = [(coeff, "IXYZ...")] for parameter k
        self.n_params = self.n_parameters  # For VQE compatibility
        self.hf_state = self.uccsd['hf_state']  # HF occupation vector
        self.hf_energy = self.uccsd['hf_energy']
        self.n_electrons = self.uccsd['n_electrons']

    def get_param_shift(self, k: int) -> float:
        """
        Return parameter-shift amount s_k for group k, so that
        dE/dθ_k = [E(θ_k + s_k) - E(θ_k - s_k)] / 2.

        We use s_k = π / (2 * a_max), where a_max = max |a| over coeffs in group k
        when λ = θ * a (coeff = i*a for anti-Hermitian pieces).
        """
        group = self.param_groups[k]
        a_max = 0.0
        for coeff, _ in group:
            a = abs(coeff.imag) if abs(coeff.imag) > 1e-14 else abs(coeff.real)
            if a > a_max:
                a_max = a
        if a_max <= 1e-14:
            # fallback: pure π/2 shift if group coefficients are (numerically) zero
            return np.pi / 2.0
        return np.pi / (2.0 * a_max)

    def prepare_hf_state(self, chi_max: int = 64):
        """
        Create MPS initialized to Hartree-Fock reference state.

        Returns:
            AdaptiveMPS initialized to |HF⟩
        """
        from .adaptive_mps import AdaptiveMPS

        mps = AdaptiveMPS(
            num_qubits=self.n_qubits,
            bond_dim=2,
            chi_max_per_bond=chi_max,
            device=self.device,
        )

        # Initialize to computational basis state matching HF occupation
        # HF state is like [1,1,0,0,...] for n_electrons in lowest orbitals
        for i in range(self.n_qubits):
            if self.hf_state[i] == 1:
                # Apply X gate to flip |0⟩ to |1⟩
                X = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
                mps.apply_single_qubit_gate(i, X)

        return mps

    def apply(self, mps, params: np.ndarray, chi_max: int = None):
        """
        Apply parameterized UCCSD circuit to MPS

        This applies the UCCSD unitary U(θ) = exp(Σᵢ θᵢ Gᵢ) to the MPS,
        where Gᵢ are the UCCSD generators (anti-Hermitian operators).

        Each generator is decomposed into Pauli strings with complex coefficients.
        For anti-Hermitian generators, coeff = i·a (purely imaginary), so:
        exp(theta * coeff * P) = exp(theta * i·a * P) = exp(i * (theta*a) * P) → unitary

        Args:
            mps: AdaptiveMPS instance (modified in-place)
            params: Parameter values for each UCCSD amplitude
            chi_max: Maximum bond dimension (uses mps.chi_max_per_bond if None)
        """
        from .mpo_ops import apply_pauli_exp_to_mps

        assert len(params) == self.n_parameters, f"Expected {self.n_parameters} parameters, got {len(params)}"

        if chi_max is None:
            if hasattr(mps, 'chi_max_per_bond'):
                # chi_max_per_bond might be int or list; convert to int
                chi_max_attr = mps.chi_max_per_bond
                chi_max = max(chi_max_attr) if isinstance(chi_max_attr, list) else chi_max_attr
            else:
                chi_max = 128

        # Check which backend to use (DEFAULT: mpo = existing path, safe for compatibility)
        backend = getattr(self, 'unitary_backend', 'mpo')

        # Helper function to apply a single Pauli rotation
        def _apply_term(P: str, lam: float, original_coeff: complex):
            if backend == 'dense':
                from .unitary_backends import apply_pauli_string_rotation_dense
                apply_pauli_string_rotation_dense(mps, P, float(lam), chi_max=chi_max)
            elif backend == 'circuit':
                from .unitary_backends import apply_pauli_string_rotation_circuit
                apply_pauli_string_rotation_circuit(mps, P, float(lam), chi_max=chi_max)
            else:  # 'mpo' backend
                # For MPO backend, keep original coefficient and scale theta
                # lam = theta_k * coeff.imag, so to get half-step: theta_half = lam / coeff.imag
                apply_pauli_exp_to_mps(mps, pauli_string=P, coeff=original_coeff, theta=float(lam / original_coeff.imag if abs(original_coeff.imag) > 1e-14 else lam / original_coeff.real), chi_max=chi_max)

        # Second-order Trotter (Suzuki) for each UCCSD parameter k
        # U ≈ ∏_j exp(i (λ_j/2) P_j) · ∏_j^rev exp(i (λ_j/2) P_j)
        # This reduces Trotter error and improves local curvature for optimizers
        for k, group in enumerate(self.param_groups):
            theta_k = params[k]

            # Skip if parameter is effectively zero
            if abs(theta_k) < 1e-16:
                continue

            # Prepare forward and backward sequences
            seq_forward = group
            seq_backward = list(reversed(group))

            # Forward half-steps
            for coeff, P in seq_forward:
                lam = (theta_k * coeff.imag) if abs(coeff.imag) > 1e-14 else (theta_k * coeff.real)
                if abs(lam) > 1e-16:
                    _apply_term(P, 0.5 * lam, coeff)

            # Backward half-steps (symmetric Trotter)
            for coeff, P in seq_backward:
                lam = (theta_k * coeff.imag) if abs(coeff.imag) > 1e-14 else (theta_k * coeff.real)
                if abs(lam) > 1e-16:
                    _apply_term(P, 0.5 * lam, coeff)
