"""
Comprehensive Tests for IR (Informational Relativity) Enhancements
===================================================================

Tests all IR enhancements implemented across ATLAS-Q:
1. Response field coherence (L8 Placement Principle)
2. Spectral lifting M_ij matrix
3. Four-layer hierarchy feedback
4. Exponential decoherence model (L5)
5. Rust backend coherence tracking
6. Triton coherence-aware kernels

Author: ATLAS-Q Development Team
Date: December 2025
"""

import math
import sys

import numpy as np
import pytest
import torch

# =============================================================================
# Test 1: Response Field Coherence (coherence/metrics.py)
# =============================================================================


class TestResponseFieldCoherence:
    """Test IR Law L8: Coherence on response fields."""

    def test_compute_response_coherence_pure_state(self):
        """Pure |0> state should have high coherence."""
        from atlas_q.coherence import compute_response_coherence

        # |0> state: all amplitude at index 0
        amplitudes = np.array([1.0 + 0j, 0.0, 0.0, 0.0])
        metrics = compute_response_coherence(amplitudes)

        assert metrics.R_bar > 0.99, f"Pure state should have R_bar ~ 1.0, got {metrics.R_bar}"
        assert metrics.is_above_e2_boundary, "Pure state should be GO regime"

    def test_compute_response_coherence_ghz_state(self):
        """GHZ state (|00> + |11>)/sqrt(2) should have high coherence."""
        from atlas_q.coherence import compute_response_coherence

        # GHZ state: amplitudes at 0 and 3 (|00> and |11>)
        amplitudes = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=complex)
        metrics = compute_response_coherence(amplitudes)

        # Both amplitudes have same phase (0), so coherence should be high
        assert metrics.R_bar > 0.9, f"GHZ state should have high R_bar, got {metrics.R_bar}"
        assert metrics.is_above_e2_boundary

    def test_compute_response_coherence_random_phases(self):
        """Random phases should have lower coherence."""
        from atlas_q.coherence import compute_response_coherence

        np.random.seed(42)
        n = 64
        phases = np.random.uniform(0, 2 * np.pi, n)
        amplitudes = np.exp(1j * phases) / np.sqrt(n)

        metrics = compute_response_coherence(amplitudes)

        # Random phases should give low R_bar
        assert metrics.R_bar < 0.5, f"Random phases should have low R_bar, got {metrics.R_bar}"

    def test_compute_relational_coherence(self):
        """Test spectral lifting coherence computation."""
        from atlas_q.coherence import compute_relational_coherence

        # Coherent system: aligned phases
        n = 16
        responses = np.ones(n) / np.sqrt(n)
        phases = np.zeros(n)  # All aligned

        metrics = compute_relational_coherence(responses, phases)

        # With aligned phases, spectral coherence should be high
        assert metrics.R_bar > 0.5, f"Aligned phases should have higher coherence"


# =============================================================================
# Test 2: Spectral Lifting (ir_enhanced/spectral_lifting.py)
# =============================================================================


class TestSpectralLifting:
    """Test IR spectral lifting M_ij matrix."""

    def test_compute_relational_matrix(self):
        """Test M_ij = chi_i * chi_j * cos(theta_i - theta_j)."""
        from atlas_q.ir_enhanced import compute_relational_matrix

        responses = np.array([1.0, 0.5, 0.25, 0.1])
        phases = np.array([0.0, 0.0, 0.0, 0.0])  # All aligned

        M = compute_relational_matrix(responses, phases)

        # With aligned phases, M_ij = chi_i * chi_j * cos(0) = chi_i * chi_j
        assert M.shape == (4, 4)
        assert np.allclose(M[0, 0], 1.0)  # chi_0 * chi_0
        assert np.allclose(M[0, 1], 0.5)  # chi_0 * chi_1
        assert np.allclose(M, M.T)  # Symmetric

    def test_compute_relational_matrix_orthogonal_phases(self):
        """Test M_ij with orthogonal phases."""
        from atlas_q.ir_enhanced import compute_relational_matrix

        responses = np.array([1.0, 1.0])
        phases = np.array([0.0, np.pi / 2])  # 90 degrees apart

        M = compute_relational_matrix(responses, phases)

        # cos(0 - pi/2) = cos(-pi/2) = 0
        assert np.allclose(M[0, 1], 0.0, atol=1e-10)
        assert np.allclose(M[1, 0], 0.0, atol=1e-10)

    def test_spectral_lifting_analysis(self):
        """Test full spectral lifting analysis."""
        from atlas_q.ir_enhanced import spectral_lifting_analysis

        # Coherent system
        n = 8
        responses = np.ones(n) / np.sqrt(n)
        phases = np.zeros(n)

        result = spectral_lifting_analysis(responses, phases)

        assert result.M.shape == (n, n)
        assert len(result.eigenvalues) >= 1
        assert 0 <= result.spectral_coherence <= 1
        assert result.spectral_gap >= 1 or np.isinf(result.spectral_gap)

    def test_coherent_structure_score(self):
        """Test coherent structure scoring."""
        from atlas_q.ir_enhanced import coherent_structure_score

        # High coherence system
        responses = np.array([1.0, 0.9, 0.8])
        phases = np.array([0.0, 0.1, 0.05])  # Nearly aligned

        score, is_ir, desc = coherent_structure_score(responses, phases)

        assert 0 <= score <= 1
        assert isinstance(is_ir, bool)
        assert "regime" in desc.lower()


# =============================================================================
# Test 3: Four-Layer Hierarchy Feedback (truncation.py)
# =============================================================================


class TestHierarchyFeedback:
    """Test Observable -> Structure feedback."""

    def test_adaptive_chi_from_coherence(self):
        """Test chi adaptation based on coherence history."""
        from atlas_q.truncation import adaptive_chi_from_coherence

        # High coherence -> chi should grow
        high_coh = [0.8, 0.85, 0.9, 0.88, 0.87]
        new_chi = adaptive_chi_from_coherence(high_coh, current_chi=64, chi_max=256)
        assert new_chi > 64, f"High coherence should increase chi, got {new_chi}"

        # Low coherence -> chi should shrink
        low_coh = [0.05, 0.06, 0.04, 0.05, 0.03]
        new_chi = adaptive_chi_from_coherence(low_coh, current_chi=64, chi_min=2)
        assert new_chi < 64, f"Low coherence should decrease chi, got {new_chi}"

    def test_coherence_aware_truncation_policy(self):
        """Test per-site truncation policy generation."""
        from atlas_q.truncation import coherence_aware_truncation_policy

        # Mixed coherence profile
        site_coherences = [0.5, 0.3, 0.1, 0.05, 0.2]

        eps_per_site, chi_per_site = coherence_aware_truncation_policy(
            site_coherences, base_eps=1e-6, base_chi=64
        )

        assert len(eps_per_site) == len(site_coherences)
        assert len(chi_per_site) == len(site_coherences)

        # High coherence sites should have tighter eps (smaller value)
        # Low coherence sites should have looser eps (larger value)
        # Site 0 has highest coherence (0.5)
        # Site 3 has lowest coherence (0.05)
        assert eps_per_site[0] < eps_per_site[3], "High coherence should have tighter truncation"


# =============================================================================
# Test 4: Exponential Decoherence Model (L5)
# =============================================================================


class TestDecoherenceModel:
    """Test IR Law L5: Exponential Decoherence."""

    def test_choose_rank_with_coherence(self):
        """Test coherence-aware rank selection."""
        from atlas_q.truncation import choose_rank_with_coherence

        # Create singular values
        S = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

        # High coherence -> conservative truncation
        k_high, eps_high, _, _, adj_high = choose_rank_with_coherence(
            S, eps_bond=0.1, chi_cap=10, coherence=0.8
        )

        # Low coherence -> aggressive truncation
        k_low, eps_low, _, _, adj_low = choose_rank_with_coherence(
            S, eps_bond=0.1, chi_cap=10, coherence=0.01
        )

        # High coherence should keep more ranks (more conservative)
        assert k_high >= k_low, f"High coherence should keep more ranks: {k_high} vs {k_low}"
        assert adj_high > adj_low, "High coherence should have higher adjustment"

    def test_choose_rank_with_decoherence(self):
        """Test L5-based rank selection with exponential decay."""
        from atlas_q.truncation import choose_rank_with_decoherence

        S = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.05])

        # Site close to preparation (low decoherence)
        k_near, _, _, _, coh_near = choose_rank_with_decoherence(
            S, eps_bond=0.1, chi_cap=10, decoherence_rate=0.5, site_index=0
        )

        # Site far from preparation (high decoherence)
        k_far, _, _, _, coh_far = choose_rank_with_decoherence(
            S, eps_bond=0.1, chi_cap=10, decoherence_rate=0.5, site_index=5
        )

        # Coherence should decay with distance
        assert coh_near > coh_far, f"Coherence should decay: {coh_near} vs {coh_far}"

    def test_compute_spectral_coherence(self):
        """Test spectral coherence from singular values."""
        from atlas_q.truncation import compute_spectral_coherence

        # Highly concentrated spectrum
        S_concentrated = torch.tensor([1.0, 0.01, 0.001])
        coh_conc = compute_spectral_coherence(S_concentrated)

        # Spread spectrum
        S_spread = torch.tensor([0.5, 0.4, 0.3])
        coh_spread = compute_spectral_coherence(S_spread)

        assert coh_conc > coh_spread, "Concentrated spectrum should have higher coherence"


# =============================================================================
# Test 5: Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for IR enhancements."""

    def test_coherence_metrics_consistency(self):
        """Test that different coherence methods give consistent results."""
        from atlas_q.coherence import compute_coherence, compute_response_coherence

        # For a simple case, both should indicate similar regime
        # Pure state
        amplitudes = np.array([1.0 + 0j, 0.0, 0.0, 0.0])
        response_metrics = compute_response_coherence(amplitudes)

        # Pauli expectations from pure |0> state
        expectations = np.array([1.0, 1.0, 1.0])  # ZZZ all +1
        pauli_metrics = compute_coherence(expectations)

        # Both should indicate GO regime for pure state
        assert response_metrics.is_above_e2_boundary
        assert pauli_metrics.is_above_e2_boundary

    def test_spectral_lifting_with_coherence(self):
        """Test that spectral lifting produces coherent metrics."""
        from atlas_q.ir_enhanced import spectral_lifting_from_amplitudes

        # Create a coherent quantum state (GHZ-like)
        n = 8
        amplitudes = np.zeros(2**n, dtype=complex)
        amplitudes[0] = 1 / np.sqrt(2)  # |00000000>
        amplitudes[-1] = 1 / np.sqrt(2)  # |11111111>

        result = spectral_lifting_from_amplitudes(amplitudes)

        # Should have some structure
        assert result.spectral_coherence > 0


# =============================================================================
# Test 6: Triton Kernels (if CUDA available)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonKernels:
    """Test Triton IR coherence kernels."""

    def test_compute_response_coherence_triton(self):
        """Test Triton response coherence computation."""
        from triton_kernels import compute_response_coherence_triton

        # Pure state on GPU
        state = torch.zeros(1024, dtype=torch.complex64, device="cuda")
        state[0] = 1.0

        r_bar, v_phi, is_above = compute_response_coherence_triton(state)

        assert r_bar > 0.99, f"Pure state should have R_bar ~ 1.0, got {r_bar}"
        assert is_above, "Pure state should be GO regime"

    def test_compute_spectral_coherence_triton(self):
        """Test Triton spectral coherence computation."""
        from triton_kernels import compute_spectral_coherence_triton

        # Pure state
        state = torch.zeros(1024, dtype=torch.complex64, device="cuda")
        state[0] = 1.0

        spectral = compute_spectral_coherence_triton(state)
        assert spectral > 0.99, f"Expected ~ 1.0 for pure state, got {spectral}"

    def test_coherence_aware_truncation_triton(self):
        """Test Triton coherence-aware truncation."""
        from triton_kernels import coherence_aware_truncation_triton

        S = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.05], device="cuda")
        state = torch.zeros(1024, dtype=torch.complex64, device="cuda")
        state[0] = 1.0  # Pure state = high coherence

        k, eps, regime = coherence_aware_truncation_triton(S, state, base_eps=0.1, chi_cap=10)

        assert regime == 2, f"Pure state should be IR regime (2), got {regime}"
        assert k >= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
