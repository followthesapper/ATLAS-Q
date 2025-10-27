"""
Test Molecular Hamiltonian builder with PySCF integration
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q import get_mpo_ops


def test_molecular_hamiltonian_h2_basic():
    """Test H2 molecular Hamiltonian construction with default settings"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Build H2 Hamiltonian with minimal basis
    H = MPOBuilder.molecular_hamiltonian_from_specs(
        molecule='H2',
        basis='sto-3g',
        device='cpu'
    )

    # Check basic properties
    assert H.n_sites == 4, f"H2 with sto-3g should have 4 qubits, got {H.n_sites}"
    assert len(H.tensors) == 4, f"Expected 4 tensors, got {len(H.tensors)}"

    # Check all tensors have correct shape (4D)
    for i, tensor in enumerate(H.tensors):
        assert len(tensor.shape) == 4, f"Tensor {i} should be 4D, got {len(tensor.shape)}D"
        assert tensor.shape[1] == 2, f"Physical dim should be 2, got {tensor.shape[1]}"
        assert tensor.shape[2] == 2, f"Physical dim should be 2, got {tensor.shape[2]}"

    print("✅ test_molecular_hamiltonian_h2_basic passed")


def test_molecular_hamiltonian_with_expectation():
    """Test that we can compute expectation values with molecular Hamiltonian"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']
    expectation_value = mpo_mod['expectation_value']

    # Import MPS
    from atlas_q.adaptive_mps import AdaptiveMPS

    # Build H2 Hamiltonian
    H = MPOBuilder.molecular_hamiltonian_from_specs(
        molecule='H2',
        basis='sto-3g',
        device='cpu'
    )

    # Create simple MPS state (4 qubits for H2 with sto-3g)
    mps = AdaptiveMPS(4, bond_dim=4, device='cpu')

    # Compute expectation value (should not crash)
    energy = expectation_value(H, mps)

    assert isinstance(energy, complex), f"Expected complex, got {type(energy)}"
    print(f"Energy: {energy}")

    print("✅ test_molecular_hamiltonian_with_expectation passed")


def test_molecular_hamiltonian_lih():
    """Test LiH molecular Hamiltonian construction"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Build LiH Hamiltonian
    H = MPOBuilder.molecular_hamiltonian_from_specs(
        molecule='LiH',
        basis='sto-3g',
        device='cpu'
    )

    # LiH with sto-3g should have more qubits than H2
    assert H.n_sites >= 6, f"LiH should have >= 6 qubits, got {H.n_sites}"
    assert len(H.tensors) == H.n_sites

    print(f"LiH has {H.n_sites} qubits with sto-3g basis")
    print("✅ test_molecular_hamiltonian_lih passed")


def test_molecular_hamiltonian_custom_geometry():
    """Test molecular Hamiltonian with custom geometry string"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Custom H2 geometry with specific bond length
    custom_h2 = "H 0 0 0; H 0 0 0.74"  # 0.74 Angstrom bond length

    H = MPOBuilder.molecular_hamiltonian_from_specs(
        molecule=custom_h2,
        basis='sto-3g',
        device='cpu'
    )

    assert H.n_sites == 4, f"Custom H2 should have 4 qubits, got {H.n_sites}"

    print("✅ test_molecular_hamiltonian_custom_geometry passed")


if __name__ == "__main__":
    print("Running Molecular Hamiltonian tests...\n")

    try:
        test_molecular_hamiltonian_h2_basic()
        test_molecular_hamiltonian_with_expectation()
        test_molecular_hamiltonian_lih()
        test_molecular_hamiltonian_custom_geometry()

        print("\n" + "="*50)
        print("✅ All Molecular Hamiltonian tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
