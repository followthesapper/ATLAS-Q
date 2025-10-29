#!/usr/bin/env python3
"""
Unit tests for molecular VQE normalization fixes

Tests to prevent regressions in:
1. HF baseline with blocked spin-orbital ordering
2. FCI eigenvalue accuracy
3. Spin-orbital ordering for different molecules
"""

import sys
import pytest
import torch
import numpy as np

# Import ATLAS-Q modules
from atlas_q.mpo_ops import MPOBuilder, expectation_value
from atlas_q.ansatz_uccsd import UCCSDAnsatz


class TestMolecularVQEFixes:
    """Test suite for VQE normalization and spin-orbital ordering fixes"""

    @pytest.fixture
    def device(self):
        """Use CPU for deterministic tests"""
        return 'cpu'

    @pytest.fixture
    def dtype(self):
        """Use complex128 for high precision"""
        return torch.complex128

    def test_h2_hf_energy_blocked_ordering(self, device, dtype):
        """
        Test that H2 HF energy is correct with blocked spin-orbital ordering.

        This test ensures:
        - Hamiltonian uses blocked ordering: [α₀, α₁, β₀, β₁]
        - UCCSD ansatz creates correct HF state |1010⟩
        - HF energy matches PySCF reference: -1.1167593 Ha
        """
        # Build H2 Hamiltonian
        H = MPOBuilder.molecular_hamiltonian_from_specs(
            'H2', 'sto-3g', device=device, dtype=dtype
        )

        # Build UCCSD ansatz
        ansatz = UCCSDAnsatz('H2', 'sto-3g', device=device, dtype=dtype)

        # Prepare HF state
        mps = ansatz.prepare_hf_state(chi_max=64)

        # Compute HF energy
        E_hf = expectation_value(H, mps).real

        # Reference HF energy from PySCF
        E_hf_ref = -1.1167593

        # Assert HF energy is correct (within numerical precision)
        assert abs(E_hf - E_hf_ref) < 1e-6, \
            f"H2 HF energy {E_hf:.8f} Ha differs from reference {E_hf_ref:.8f} Ha"

        # Check HF occupation is blocked: [1, 0, 1, 0] for H2
        assert ansatz.hf_state.tolist() == [1, 0, 1, 0], \
            f"H2 HF state should be [1,0,1,0], got {ansatz.hf_state.tolist()}"

    def test_h2_fci_eigenvalue_matches_reference(self, device, dtype):
        """
        Test that H2 FCI energy (ground state eigenvalue) matches reference.

        This ensures the Hamiltonian matrix is constructed correctly.
        Expected FCI energy: -1.1372838 Ha
        """
        # Build H2 Hamiltonian
        H = MPOBuilder.molecular_hamiltonian_from_specs(
            'H2', 'sto-3g', device=device, dtype=dtype
        )

        # Get eigenvalues from dense Hamiltonian
        assert hasattr(H, 'full_matrix'), "Hamiltonian should have full_matrix attribute"

        H_matrix = H.full_matrix.to(device=device, dtype=dtype)
        eigenvalues = torch.linalg.eigvalsh(H_matrix).real

        E_fci = eigenvalues.min().item()
        E_fci_ref = -1.1372838

        # Assert FCI energy is correct
        assert abs(E_fci - E_fci_ref) < 2e-6, \
            f"H2 FCI energy {E_fci:.8f} Ha differs from reference {E_fci_ref:.8f} Ha"

    def test_lih_hf_occupation_blocked(self, device):
        """
        Test that LiH HF state uses blocked spin-orbital ordering.

        LiH has 4 electrons in 6 orbitals (12 spin-orbitals).
        Expected HF occupation (blocked): [1,1,0,0,0,0, 1,1,0,0,0,0]
        - α₀, α₁, β₀, β₁ occupied
        """
        # Build LiH UCCSD ansatz
        ansatz = UCCSDAnsatz('LiH', 'sto-3g', device=device)

        n_orb = ansatz.n_qubits // 2
        hf_state = ansatz.hf_state

        # Check that first 2 alpha orbitals are occupied
        assert hf_state[0] == 1, "LiH α₀ should be occupied"
        assert hf_state[1] == 1, "LiH α₁ should be occupied"

        # Check that first 2 beta orbitals are occupied
        assert hf_state[n_orb] == 1, "LiH β₀ should be occupied"
        assert hf_state[n_orb + 1] == 1, "LiH β₁ should be occupied"

        # Check total number of electrons
        assert hf_state.sum() == ansatz.n_electrons, \
            f"HF state should have {ansatz.n_electrons} electrons, got {hf_state.sum()}"

    def test_h2o_hf_occupation_blocked(self, device):
        """
        Test that H2O HF state uses blocked spin-orbital ordering.

        H2O has 10 electrons in 7 orbitals (14 spin-orbitals).
        Expected: 5 doubly occupied orbitals
        """
        # Build H2O UCCSD ansatz
        ansatz = UCCSDAnsatz('H2O', 'sto-3g', device=device)

        n_orb = ansatz.n_qubits // 2
        hf_state = ansatz.hf_state

        # Check that first 5 alpha orbitals are occupied
        for i in range(5):
            assert hf_state[i] == 1, f"H2O α_{i} should be occupied"

        # Check that first 5 beta orbitals are occupied
        for i in range(5):
            assert hf_state[n_orb + i] == 1, f"H2O β_{i} should be occupied"

        # Check total number of electrons
        assert hf_state.sum() == ansatz.n_electrons, \
            f"HF state should have {ansatz.n_electrons} electrons, got {hf_state.sum()}"

    def test_expectation_value_normalization(self, device, dtype):
        """
        Test that expectation_value correctly normalizes unnormalized states.

        This ensures the normalization fixes prevent unphysical energies.
        """
        # Build H2 Hamiltonian
        H = MPOBuilder.molecular_hamiltonian_from_specs(
            'H2', 'sto-3g', device=device, dtype=dtype
        )

        # Build UCCSD ansatz
        ansatz = UCCSDAnsatz('H2', 'sto-3g', device=device, dtype=dtype)
        mps = ansatz.prepare_hf_state(chi_max=64)

        # Artificially scale the MPS norm by 3
        for i in range(mps.num_qubits):
            mps.tensors[i] = mps.tensors[i] * 1.5  # Each tensor scaled → total norm ≈ 3

        # Compute energy (should automatically normalize)
        E = expectation_value(H, mps).real

        # Energy should still be close to HF energy
        E_hf_ref = -1.1167593
        assert abs(E - E_hf_ref) < 0.01, \
            f"Unnormalized state gave energy {E:.6f}, expected ~{E_hf_ref:.6f}"

    def test_spin_orbital_ordering_consistency(self, device):
        """
        Test that all molecules use consistent blocked spin-orbital ordering.
        """
        molecules = ['H2', 'LiH']

        for mol in molecules:
            ansatz = UCCSDAnsatz(mol, 'sto-3g', device=device)
            n_orb = ansatz.n_qubits // 2
            n_elec = ansatz.n_electrons
            hf_state = ansatz.hf_state

            # For closed-shell RHF: n_elec should be even
            if n_elec % 2 == 0:
                n_doubly_occ = n_elec // 2

                # Check alpha electrons are in first n_doubly_occ orbitals
                for i in range(n_doubly_occ):
                    assert hf_state[i] == 1, \
                        f"{mol}: α_{i} should be occupied in blocked ordering"

                # Check beta electrons are in first n_doubly_occ orbitals
                for i in range(n_doubly_occ):
                    assert hf_state[n_orb + i] == 1, \
                        f"{mol}: β_{i} should be occupied in blocked ordering"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
