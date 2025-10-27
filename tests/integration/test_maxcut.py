"""
Test MaxCut Hamiltonian builder and QAOA integration
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q import get_mpo_ops, get_vqe_qaoa


def test_maxcut_hamiltonian_basic():
    """Test MaxCut Hamiltonian construction for simple triangle graph"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Triangle graph: nodes 0, 1, 2
    edges = [(0, 1), (1, 2), (0, 2)]

    H = MPOBuilder.maxcut_hamiltonian(edges, device='cpu')

    # Check basic properties
    assert H.n_sites == 3, f"Expected 3 sites, got {H.n_sites}"
    assert len(H.tensors) == 3, f"Expected 3 tensors, got {len(H.tensors)}"

    # Check all tensors have correct shape (4D)
    for i, tensor in enumerate(H.tensors):
        assert len(tensor.shape) == 4, f"Tensor {i} should be 4D, got {len(tensor.shape)}D"
        assert tensor.shape[1] == 2, f"Physical dim should be 2, got {tensor.shape[1]}"
        assert tensor.shape[2] == 2, f"Physical dim should be 2, got {tensor.shape[2]}"

    print("✅ test_maxcut_hamiltonian_basic passed")


def test_maxcut_hamiltonian_weighted():
    """Test MaxCut with weighted edges"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Square graph with weights
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    weights = [1.0, 2.0, 1.0, 2.0]

    H = MPOBuilder.maxcut_hamiltonian(edges, weights=weights, device='cpu')

    assert H.n_sites == 4
    assert len(H.tensors) == 4

    print("✅ test_maxcut_hamiltonian_weighted passed")


def test_maxcut_with_expectation():
    """Test that we can compute expectation values with MaxCut Hamiltonian"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']
    expectation_value = mpo_mod['expectation_value']

    # Import MPS
    from atlas_q.adaptive_mps import AdaptiveMPS

    # Simple 3-node graph
    edges = [(0, 1), (1, 2)]
    H = MPOBuilder.maxcut_hamiltonian(edges, device='cpu')

    # Create simple MPS state
    mps = AdaptiveMPS(3, bond_dim=4, device='cpu')

    # Apply some gates to create non-trivial state
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    mps.apply_single_qubit_gate(0, X)

    # Compute expectation value (should not crash)
    energy = expectation_value(H, mps)

    assert isinstance(energy, complex), f"Expected complex, got {type(energy)}"
    print(f"Energy: {energy}")

    print("✅ test_maxcut_with_expectation passed")


def test_maxcut_hamiltonian_explicit_sites():
    """Test MaxCut with explicit number of sites"""
    mpo_mod = get_mpo_ops()
    MPOBuilder = mpo_mod['MPOBuilder']

    # Graph with gap in node indices
    edges = [(0, 2), (2, 4)]
    H = MPOBuilder.maxcut_hamiltonian(edges, n_sites=5, device='cpu')

    assert H.n_sites == 5, f"Expected 5 sites, got {H.n_sites}"

    print("✅ test_maxcut_hamiltonian_explicit_sites passed")


if __name__ == "__main__":
    print("Running MaxCut Hamiltonian tests...\n")

    try:
        test_maxcut_hamiltonian_basic()
        test_maxcut_hamiltonian_weighted()
        test_maxcut_hamiltonian_explicit_sites()
        test_maxcut_with_expectation()

        print("\n" + "="*50)
        print("✅ All MaxCut tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
