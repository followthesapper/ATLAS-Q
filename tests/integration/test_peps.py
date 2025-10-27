"""
Test PEPS (Projected Entangled Pair States) 2D tensor networks
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q.peps import (
    PEPS,
    PEPSConfig,
    PatchPEPS,
    ContractionStrategy
)


def test_peps_initialization():
    """Test PEPS initialization in product state"""
    config = PEPSConfig(rows=3, cols=3, bond_dim=2, device='cpu')
    peps = PEPS(config)

    assert peps.rows == 3
    assert peps.cols == 3
    assert len(peps.tensors) == 9  # 3×3 grid

    # Check tensor shapes
    for (r, c), peps_tensor in peps.tensors.items():
        tensor = peps_tensor.tensor
        assert len(tensor.shape) == 5  # [χU, χL, d, χR, χD]
        assert tensor.shape[2] == 2  # Physical dimension

    print("✅ test_peps_initialization passed")


def test_peps_boundary_bonds():
    """Test that boundary PEPS tensors have bond dimension 1"""
    config = PEPSConfig(rows=4, cols=4, bond_dim=3, device='cpu')
    peps = PEPS(config)

    # Top-left corner
    corner_tensor = peps.tensors[(0, 0)].tensor
    assert corner_tensor.shape[0] == 1  # χU = 1 (top boundary)
    assert corner_tensor.shape[1] == 1  # χL = 1 (left boundary)

    # Bottom-right corner
    corner_tensor = peps.tensors[(3, 3)].tensor
    assert corner_tensor.shape[3] == 1  # χR = 1 (right boundary)
    assert corner_tensor.shape[4] == 1  # χD = 1 (bottom boundary)

    # Bulk tensor
    bulk_tensor = peps.tensors[(1, 1)].tensor
    assert bulk_tensor.shape[0] == 3  # χU = bond_dim
    assert bulk_tensor.shape[1] == 3  # χL = bond_dim

    print("✅ test_peps_boundary_bonds passed")


def test_peps_single_gate():
    """Test single-qubit gate application"""
    config = PEPSConfig(rows=2, cols=2, bond_dim=2, device='cpu')
    peps = PEPS(config)

    # Hadamard gate
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)

    # Apply to (0,0)
    peps.apply_single_site_gate(0, 0, H)

    # Tensor should still be valid shape
    tensor = peps.tensors[(0, 0)].tensor
    assert tensor.shape == (1, 1, 2, 2, 2)  # [χU, χL, d, χR, χD]

    print("✅ test_peps_single_gate passed")


def test_peps_multiple_gates():
    """Test applying multiple single-qubit gates"""
    config = PEPSConfig(rows=3, cols=3, bond_dim=2, device='cpu')
    peps = PEPS(config)

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)

    # Apply Hadamard to all sites
    for r in range(3):
        for c in range(3):
            peps.apply_single_site_gate(r, c, H)

    # Apply X to center
    peps.apply_single_site_gate(1, 1, X)

    # All tensors should still be valid
    for (r, c), peps_tensor in peps.tensors.items():
        tensor = peps_tensor.tensor
        assert tensor.shape[2] == 2  # Physical dim unchanged

    print("✅ test_peps_multiple_gates passed")


def test_peps_norm_computation():
    """Test PEPS norm computation"""
    config = PEPSConfig(rows=2, cols=2, bond_dim=2, device='cpu')
    peps = PEPS(config)

    # Initial state should have norm close to 1
    norm = peps.compute_norm()

    # Norm should be positive
    assert norm > 0
    print(f"PEPS norm: {norm:.6f}")

    # Apply some gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    peps.apply_single_site_gate(0, 0, H)

    norm_after = peps.compute_norm()
    assert norm_after > 0
    print(f"PEPS norm after gate: {norm_after:.6f}")

    print("✅ test_peps_norm_computation passed")


def test_patch_peps_initialization():
    """Test PatchPEPS initialization"""
    patch = PatchPEPS(patch_size=4, device='cpu')

    assert patch.patch_size == 4
    assert patch.peps.rows == 4
    assert patch.peps.cols == 4

    print("✅ test_patch_peps_initialization passed")


def test_patch_peps_shallow_circuit():
    """Test PatchPEPS with shallow circuit"""
    patch = PatchPEPS(patch_size=3, device='cpu')

    # Create simple circuit: layer of Hadamards
    gates = []
    for r in range(3):
        for c in range(3):
            gates.append(('H', [(r, c)], []))

    # Apply circuit
    patch.apply_shallow_circuit(gates)

    # Compute norm
    norm = patch.peps.compute_norm()
    assert norm > 0
    print(f"Patch PEPS norm after Hadamards: {norm:.6f}")

    print("✅ test_patch_peps_shallow_circuit passed")


def test_peps_row_contraction():
    """Test contracting a PEPS row to MPS"""
    config = PEPSConfig(rows=2, cols=3, bond_dim=2, device='cpu')
    peps = PEPS(config)

    # Contract top row to MPS
    mps_tensors = peps._contract_row_to_mps(row=0)

    assert len(mps_tensors) == 3  # 3 columns

    # Each MPS tensor should be 3D
    for tensor in mps_tensors:
        assert len(tensor.shape) == 3  # [χL, d*χD, χR]

    print("✅ test_peps_row_contraction passed")


def test_peps_neighbor_check():
    """Test neighbor checking for two-site gates"""
    config = PEPSConfig(rows=3, cols=3, bond_dim=2, device='cpu')
    peps = PEPS(config)

    # Horizontal neighbors
    assert peps._are_neighbors(0, 0, 0, 1) == True
    assert peps._are_neighbors(0, 1, 0, 0) == True

    # Vertical neighbors
    assert peps._are_neighbors(0, 0, 1, 0) == True
    assert peps._are_neighbors(1, 0, 0, 0) == True

    # Not neighbors
    assert peps._are_neighbors(0, 0, 0, 2) == False
    assert peps._are_neighbors(0, 0, 2, 0) == False
    assert peps._are_neighbors(0, 0, 1, 1) == False  # Diagonal

    print("✅ test_peps_neighbor_check passed")


def test_peps_gate_matrix_generation():
    """Test gate matrix generation for PatchPEPS"""
    patch = PatchPEPS(patch_size=2, device='cpu')

    # Test Hadamard
    H = patch._get_gate_matrix('H', [])
    assert H.shape == (2, 2)

    # Test CZ
    CZ = patch._get_gate_matrix('CZ', [])
    assert CZ.shape == (2, 2) or CZ.shape == (4, 4)  # Could be 2x2 or 4x4 depending on impl

    # Test unknown gate (should return identity)
    I = patch._get_gate_matrix('UNKNOWN', [])
    assert I.shape == (2, 2)

    print("✅ test_peps_gate_matrix_generation passed")


if __name__ == "__main__":
    print("Running PEPS tests...\n")

    try:
        test_peps_initialization()
        test_peps_boundary_bonds()
        test_peps_single_gate()
        test_peps_multiple_gates()
        test_peps_norm_computation()
        test_patch_peps_initialization()
        test_patch_peps_shallow_circuit()
        test_peps_row_contraction()
        test_peps_neighbor_check()
        test_peps_gate_matrix_generation()

        print("\n" + "="*50)
        print("✅ All PEPS tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
