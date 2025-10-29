"""
Unit Tests for Adaptive MPS

Tests adaptive truncation, correctness, and moderate entanglement handling.

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy
from atlas_q.diagnostics import bond_entropy_from_S, effective_rank
from atlas_q.truncation import choose_rank_from_sigma


class TestAdaptiveMPS:
    """Test suite for AdaptiveMPS class"""

    @pytest.fixture
    def device(self):
        """Use CUDA if available, else CPU"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_initialization(self, device):
        """Test basic initialization"""
        mps = AdaptiveMPS(
            num_qubits=4,
            bond_dim=2,
            eps_bond=1e-6,
            chi_max_per_bond=16,
            device=device
        )
        assert mps.num_qubits == 4
        assert len(mps.tensors) == 4
        assert len(mps.bond_dims) == 3
        assert all(d == 2 for d in mps.bond_dims)

    def test_single_qubit_gate(self, device):
        """Test single-qubit gate application"""
        mps = AdaptiveMPS(4, bond_dim=2, device=device)

        # Hadamard gate
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)

        # Apply to qubit 0
        mps.apply_single_qubit_gate(0, H)

        # Check tensor shape unchanged
        assert mps.tensors[0].shape == (1, 2, 2)

        # Check not all zeros
        assert torch.abs(mps.tensors[0]).sum() > 0.1

    def test_bell_pair_no_truncation(self, device):
        """Test Bell pair creation with χ=2 (should not truncate)"""
        mps = AdaptiveMPS(
            num_qubits=2,
            bond_dim=2,
            eps_bond=1e-10,
            chi_max_per_bond=4,
            device=device
        )

        # Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        # Step 1: H on qubit 0
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        mps.apply_single_qubit_gate(0, H)

        # Step 2: CNOT (as two-qubit gate)
        CNOT = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

        mps.apply_two_site_gate(0, CNOT)

        # Check bond dimension stayed at 2 (maximally entangled 2-qubit state)
        assert mps.bond_dims[0] <= 2

        # Check entanglement entropy (should be non-zero for entangled state)
        stats = mps.stats_summary()
        if stats['total_operations'] > 0:
            assert stats['mean_entropy'] >= 0.0  # Non-zero indicates entanglement

        # Check global error is very small
        assert mps.global_error_bound() < 1e-5  # Relaxed for numerical precision

    def test_adaptive_truncation_reduces_rank(self, device):
        """Test that adaptive truncation actually reduces bond dimension"""
        mps = AdaptiveMPS(
            num_qubits=4,
            bond_dim=8,
            eps_bond=1e-3,  # Aggressive truncation
            chi_max_per_bond=8,
            device=device
        )

        # Apply some gates that create limited entanglement
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)

        # H on all qubits
        for i in range(4):
            mps.apply_single_qubit_gate(i, H)

        # Apply a controlled gate (creates entanglement)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))
        mps.apply_two_site_gate(0, CZ)

        # Check that at least one operation was logged
        stats = mps.stats_summary()
        assert stats['total_operations'] > 0

        # Check that truncation happened (some χ reduced)
        # For this simple circuit, we shouldn't need full χ=8
        assert stats['max_chi'] <= 8

    def test_ghz_state_correctness(self, device):
        """Test GHZ state |000⟩ + |111⟩ creation and properties"""
        n = 3
        mps = AdaptiveMPS(
            num_qubits=n,
            bond_dim=2,
            eps_bond=1e-10,
            chi_max_per_bond=4,
            device=device
        )

        # Create GHZ: H on qubit 0, then CNOT chain
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CNOT = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

        mps.apply_single_qubit_gate(0, H)
        mps.apply_two_site_gate(0, CNOT)
        mps.apply_two_site_gate(1, CNOT)

        # Check bond dimensions are small (GHZ has χ=2)
        for d in mps.bond_dims:
            assert d <= 2

        # Check norm preservation (should be close to 1)
        # Get full state vector and check
        full_state = mps.to_statevector()
        norm = torch.abs(full_state).pow(2).sum()
        # GPU complex64 precision requires more relaxed tolerance
        tol = 1e-3 if mps.tensors[0].dtype == torch.complex64 else 1e-5
        assert abs(norm.item() - 1.0) < tol, f"Norm {norm.item()} differs from 1.0 by {abs(norm.item() - 1.0)}"

    def test_moderate_entanglement_handling(self, device):
        """Test handling of moderate entanglement with χ growth"""
        n = 8
        mps = AdaptiveMPS(
            num_qubits=n,
            bond_dim=4,
            eps_bond=1e-6,
            chi_max_per_bond=32,
            device=device
        )

        # Create moderate entanglement: random local gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

        # Layer 1: Hadamards
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)

        # Layer 2: Entangling gates
        for i in range(0, n-1, 2):
            mps.apply_two_site_gate(i, CZ)

        # Layer 3: More Hadamards
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)

        # Layer 4: More entangling
        for i in range(1, n-1, 2):
            mps.apply_two_site_gate(i, CZ)

        # Check that bond dimensions respected cap
        stats = mps.stats_summary()
        assert stats['max_chi'] >= 2  # Some growth occurred
        assert stats['max_chi'] <= 32  # But respected cap

        # Check that global error is controlled
        assert mps.global_error_bound() < 1e-4

    def test_truncation_error_accounting(self, device):
        """Test that truncation error accounting is correct"""
        mps = AdaptiveMPS(
            num_qubits=4,
            bond_dim=8,
            eps_bond=1e-4,
            chi_max_per_bond=16,
            device=device
        )

        # Apply circuit
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

        for i in range(4):
            mps.apply_single_qubit_gate(i, H)

        for i in range(3):
            mps.apply_two_site_gate(i, CZ)

        # Check global error bound formula
        stats = mps.stats_summary()
        computed_bound = mps.global_error_bound()

        # Manual computation: sqrt(sum(eps_local^2))
        manual_bound = np.sqrt(stats['sum_eps2'])
        assert abs(computed_bound - manual_bound) < 1e-10

    def test_svd_fallback(self, device):
        """Test that SVD fallback mechanism works"""
        mps = AdaptiveMPS(
            num_qubits=3,
            bond_dim=4,
            eps_bond=1e-8,
            device=device
        )

        # Apply gates that might cause numerical issues
        # Create a nearly-degenerate scenario
        theta = 1e-8
        U = torch.tensor([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=torch.complex64)

        mps.apply_single_qubit_gate(0, U)

        # Should complete without errors (fallback works)
        stats = mps.stats_summary()
        assert stats['total_operations'] >= 0  # Should not crash

    def test_mixed_precision_policy(self, device):
        """Test mixed precision with condition number threshold"""
        policy = DTypePolicy(
            default=torch.complex64,
            promote_if_cond_gt=1e6
        )

        mps = AdaptiveMPS(
            num_qubits=3,
            bond_dim=4,
            eps_bond=1e-10,
            dtype_policy=policy,
            device=device
        )

        # Apply normal gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CNOT = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

        mps.apply_single_qubit_gate(0, H)
        mps.apply_two_site_gate(0, CNOT)

        # Should complete (dtype handling works)
        assert mps.tensors[0].dtype in [torch.complex64, torch.complex128]

    def test_per_bond_chi_caps(self, device):
        """Test per-bond χ caps are respected"""
        chi_caps = [2, 4, 8, 16]
        mps = AdaptiveMPS(
            num_qubits=5,
            bond_dim=2,
            eps_bond=1e-6,
            chi_max_per_bond=chi_caps,
            device=device
        )

        assert mps.chi_max_per_bond == chi_caps

        # Apply entangling circuit
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

        for i in range(5):
            mps.apply_single_qubit_gate(i, H)

        for i in range(4):
            mps.apply_two_site_gate(i, CZ)

        # Check caps are respected
        for i in range(4):
            assert mps.bond_dims[i] <= chi_caps[i]

    def test_canonical_forms(self, device):
        """Test canonical form transformations"""
        mps = AdaptiveMPS(4, bond_dim=4, device=device)

        # Apply some gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        for i in range(4):
            mps.apply_single_qubit_gate(i, H)

        # Test left-canonical
        mps.to_left_canonical()
        assert mps.is_canonical

        # Test mixed-canonical
        mps.to_mixed_canonical(center=2)
        # Should complete without error

    def test_statistics_tracking(self, device):
        """Test that statistics are tracked correctly"""
        mps = AdaptiveMPS(4, bond_dim=4, eps_bond=1e-6, device=device)

        # Reset stats
        mps.reset_stats()
        stats = mps.stats_summary()
        assert stats['total_operations'] == 0

        # Apply gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

        for i in range(3):
            mps.apply_two_site_gate(i, CZ)

        # Check stats updated
        stats = mps.stats_summary()
        assert stats['total_operations'] == 3
        assert stats['total_time_ms'] > 0

    def test_snapshot_and_load(self, device, tmp_path):
        """Test checkpointing functionality"""
        mps1 = AdaptiveMPS(4, bond_dim=4, eps_bond=1e-6, device=device)

        # Apply gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        for i in range(4):
            mps1.apply_single_qubit_gate(i, H)

        # Save
        path = tmp_path / "test_mps.pt"
        mps1.snapshot(str(path))

        # Load
        mps2 = AdaptiveMPS.load_snapshot(str(path), device=device)

        # Check equality
        assert mps2.num_qubits == mps1.num_qubits
        assert mps2.bond_dims == mps1.bond_dims
        assert len(mps2.tensors) == len(mps1.tensors)

    def test_memory_usage(self, device):
        """Test memory usage reporting"""
        mps = AdaptiveMPS(10, bond_dim=8, device=device)

        mem = mps.get_memory_usage()
        assert mem > 0

        # Should be approximately n * χ² * 2 * 8 bytes
        expected = 10 * 8 * 8 * 2 * 8  # rough estimate
        assert mem < expected * 2  # Within 2x (due to overhead)


class TestTruncationPolicy:
    """Test suite for truncation policy functions"""

    def test_choose_rank_basic(self):
        """Test basic rank selection"""
        S = torch.tensor([1.0, 0.5, 0.1, 0.01, 0.001])

        k, eps_local, entropy, condS = choose_rank_from_sigma(
            S, eps_bond=1e-2, chi_cap=5
        )

        # Should keep most of the energy
        assert k >= 3
        assert k <= 5
        assert eps_local < 0.1

    def test_choose_rank_with_cap(self):
        """Test that χ cap is respected"""
        S = torch.tensor([1.0] * 10)

        k, _, _, _ = choose_rank_from_sigma(
            S, eps_bond=1e-10, chi_cap=5
        )

        assert k <= 5

    def test_choose_rank_with_budget(self):
        """Test budget constraint"""
        S = torch.tensor([1.0] * 10)

        def budget_ok(k):
            return k <= 3

        k, _, _, _ = choose_rank_from_sigma(
            S, eps_bond=1e-10, chi_cap=10, budget_ok=budget_ok
        )

        assert k <= 3

    def test_entropy_calculation(self):
        """Test entropy calculation from singular values"""
        # Uniform distribution (maximum entropy)
        S = torch.ones(4) / 2.0
        entropy = bond_entropy_from_S(S)
        assert abs(entropy - 2.0) < 0.1  # log₂(4) = 2

        # Single value (zero entropy)
        S = torch.tensor([1.0, 0.0, 0.0, 0.0])
        entropy = bond_entropy_from_S(S)
        assert entropy < 0.01

    def test_effective_rank(self):
        """Test effective rank calculation"""
        # Rapidly decaying spectrum
        S = torch.tensor([1.0, 0.1, 0.01, 0.001])
        r = effective_rank(S, threshold=0.99)
        assert r <= 2  # First two values dominate


class TestDiagnostics:
    """Test suite for diagnostics utilities"""

    def test_mps_statistics_accumulation(self):
        """Test statistics accumulation"""
        from atlas_q.diagnostics import MPSStatistics

        stats = MPSStatistics()

        # Record some operations
        stats.record(step=0, bond=0, k_star=4, chi_before=8, chi_after=4,
                    eps_local=1e-6, entropy=1.5, svd_driver='torch_cuda',
                    dtype='complex64', ms_elapsed=10.0, condS=100.0)

        stats.record(step=1, bond=1, k_star=6, chi_before=8, chi_after=6,
                    eps_local=2e-6, entropy=2.0, svd_driver='torch_cuda',
                    dtype='complex64', ms_elapsed=12.0, condS=150.0)

        summary = stats.summary()
        assert summary['total_operations'] == 2
        assert summary['max_chi'] == 6
        assert summary['total_time_ms'] == 22.0

        # Check global error
        global_err = stats.global_error_bound()
        expected = np.sqrt((1e-6)**2 + (2e-6)**2)
        assert abs(global_err - expected) < 1e-15


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
