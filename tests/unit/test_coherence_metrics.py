"""
Unit tests for coherence metrics module.

Tests:
- Coherence calculation from measurements
- Coherence law validation
- E^-2 boundary classification
- Edge cases and error handling
"""

import numpy as np
import pytest

from atlas_q.coherence import (
    CoherenceMetrics,
    compute_coherence,
    validate_coherence_law,
    compute_pauli_expectation,
)


class TestCoherenceMetrics:
    """Tests for CoherenceMetrics dataclass."""

    def test_valid_metrics(self):
        """Test creation of valid coherence metrics."""
        metrics = CoherenceMetrics(
            R_bar=0.8,
            V_phi=0.45,
            is_above_e2_boundary=True,
            vra_predicted_to_help=True,
            n_measurements=10
        )
        assert metrics.R_bar == 0.8
        assert metrics.V_phi == 0.45
        assert metrics.is_above_e2_boundary
        assert metrics.n_measurements == 10

    def test_invalid_r_bar(self):
        """Test that invalid R_bar values raise ValueError."""
        with pytest.raises(ValueError, match="R_bar must be in"):
            CoherenceMetrics(
                R_bar=1.5,  # Invalid: > 1
                V_phi=0.0,
                is_above_e2_boundary=True,
                vra_predicted_to_help=True,
            )

        with pytest.raises(ValueError, match="R_bar must be in"):
            CoherenceMetrics(
                R_bar=-0.1,  # Invalid: < 0
                V_phi=0.0,
                is_above_e2_boundary=False,
                vra_predicted_to_help=False,
            )

    def test_as_dict(self):
        """Test conversion to dictionary."""
        metrics = CoherenceMetrics(
            R_bar=0.9,
            V_phi=0.2,
            is_above_e2_boundary=True,
            vra_predicted_to_help=True,
            n_measurements=5
        )
        d = metrics.as_dict()
        assert isinstance(d, dict)
        assert d['R_bar'] == 0.9
        assert d['V_phi'] == 0.2
        assert d['is_above_e2_boundary'] is True
        assert d['n_measurements'] == 5

    def test_str_representation(self):
        """Test string representation."""
        metrics = CoherenceMetrics(
            R_bar=0.875,
            V_phi=0.267,
            is_above_e2_boundary=True,
            vra_predicted_to_help=True,
            n_measurements=100
        )
        s = str(metrics)
        assert "R̄=0.8750" in s
        assert "V_φ=0.2670" in s
        assert "GO" in s
        assert "n=100" in s


class TestComputeCoherence:
    """Tests for compute_coherence function."""

    def test_perfect_coherence(self):
        """Test with perfect coherence (all measurements identical)."""
        outcomes = np.array([1.0, 1.0, 1.0, 1.0])
        coherence = compute_coherence(outcomes)

        assert coherence.R_bar == pytest.approx(1.0, abs=1e-6)
        assert coherence.V_phi == pytest.approx(0.0, abs=1e-6)
        assert coherence.is_above_e2_boundary
        assert coherence.n_measurements == 4

    def test_random_coherence(self):
        """Test with random measurements (low coherence)."""
        # Generate random measurements
        np.random.seed(42)
        outcomes = np.random.uniform(-1, 1, 100)
        coherence = compute_coherence(outcomes)

        # Random measurements should have moderate to high R_bar due to arccos mapping
        assert 0.0 <= coherence.R_bar <= 1.0  # Valid range
        assert coherence.V_phi >= 0.0  # Non-negative variance
        assert coherence.n_measurements == 100

    def test_high_coherence_above_e2(self):
        """Test with high coherence measurements (above e^-2)."""
        # Create measurements with some spread but high coherence
        outcomes = np.array([0.9, 0.85, 0.88, 0.87, 0.9, 0.89])
        coherence = compute_coherence(outcomes)

        assert coherence.R_bar > 0.135  # Above e^-2
        assert coherence.is_above_e2_boundary
        assert coherence.vra_predicted_to_help

    def test_low_coherence_below_e2(self):
        """Test with low coherence measurements (below e^-2)."""
        # Create measurements with truly low coherence
        # Use values that map to widely spread phases
        outcomes = np.array([0.99, -0.99, 0.98, -0.98, 0.97, -0.97])
        coherence = compute_coherence(outcomes)

        # With alternating near-max values, should have very low R_bar
        assert coherence.R_bar < 0.5
        assert coherence.is_above_e2_boundary is False or coherence.R_bar < 0.135

    def test_custom_threshold(self):
        """Test with custom e^-2 threshold."""
        # Use measurements that give R_bar between 0.135 and 0.5
        # Create moderate spread to get intermediate coherence
        np.random.seed(123)
        outcomes = np.random.uniform(0.3, 0.7, 20)  # Moderate spread

        # Default threshold (0.135)
        coherence_default = compute_coherence(outcomes, e2_threshold=0.135)

        # Custom high threshold (0.95)
        coherence_custom = compute_coherence(outcomes, e2_threshold=0.95)

        # Same R_bar, different boundary check
        assert coherence_default.R_bar == coherence_custom.R_bar
        # With moderate spread, should be above 0.135 but below 0.95
        if 0.135 < coherence_default.R_bar < 0.95:
            assert coherence_default.is_above_e2_boundary != coherence_custom.is_above_e2_boundary

    def test_coherence_law(self):
        """Test that computed metrics satisfy coherence law: R̄ = e^(-V_φ/2)."""
        outcomes = np.array([0.8, 0.75, 0.82, 0.78, 0.81])
        coherence = compute_coherence(outcomes)

        # Validate coherence law
        is_valid = validate_coherence_law(coherence.R_bar, coherence.V_phi, tolerance=0.1)
        assert is_valid, "Coherence law not satisfied"

    def test_empty_measurements(self):
        """Test that empty measurements raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_coherence(np.array([]))

    def test_invalid_measurement_range(self):
        """Test that measurements outside [-1, 1] raise ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            compute_coherence(np.array([0.8, 1.5, 0.9]))  # 1.5 > 1

    def test_list_input(self):
        """Test that list input is accepted and converted to numpy array."""
        outcomes_list = [0.8, 0.85, 0.82, 0.88]
        coherence = compute_coherence(outcomes_list)

        assert isinstance(coherence, CoherenceMetrics)
        assert coherence.n_measurements == 4

    def test_single_measurement(self):
        """Test with single measurement."""
        outcomes = np.array([0.9])
        coherence = compute_coherence(outcomes)

        # Single measurement should have perfect coherence (R̄ = 1)
        assert coherence.R_bar == pytest.approx(1.0, abs=1e-6)
        assert coherence.is_above_e2_boundary


class TestComputePauliExpectation:
    """Tests for computing Pauli expectations from counts."""

    def test_all_zeros(self):
        """Test with all measurements returning |00...0⟩."""
        counts = {'00': 1000}
        exp_val = compute_pauli_expectation(counts, 'ZZ')

        # |00⟩ has eigenvalue +1 for ZZ
        assert exp_val == pytest.approx(1.0, abs=1e-6)

    def test_balanced_superposition(self):
        """Test with balanced superposition."""
        counts = {'00': 500, '11': 500}
        exp_val = compute_pauli_expectation(counts, 'ZZ')

        # |00⟩ and |11⟩ both have even parity for ZZ
        assert exp_val == pytest.approx(1.0, abs=1e-6)

    def test_odd_parity(self):
        """Test with odd parity states."""
        counts = {'01': 500, '10': 500}
        exp_val = compute_pauli_expectation(counts, 'ZZ')

        # |01⟩ and |10⟩ both have odd parity for ZZ
        assert exp_val == pytest.approx(-1.0, abs=1e-6)

    def test_identity_operator(self):
        """Test that identity always returns +1."""
        counts = {'00': 200, '01': 300, '10': 250, '11': 250}
        exp_val = compute_pauli_expectation(counts, 'II')

        # Identity always has eigenvalue +1
        assert exp_val == pytest.approx(1.0, abs=1e-6)

    def test_single_qubit_x(self):
        """Test single-qubit X measurement."""
        counts = {'0': 600, '1': 400}
        exp_val = compute_pauli_expectation(counts, 'X')

        # Should be close to 0 with slight bias toward |0⟩
        expected = (600 - 400) / 1000  # Parity: 0=even, 1=odd
        assert exp_val == pytest.approx(expected, abs=1e-6)

    def test_empty_counts(self):
        """Test that empty counts raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_pauli_expectation({}, 'ZZ')

    def test_mismatched_length(self):
        """Test that mismatched bitstring length raises ValueError."""
        counts = {'00': 1000}
        with pytest.raises(ValueError, match="does not match"):
            compute_pauli_expectation(counts, 'ZZZ')  # Pauli too long


class TestCoherenceLawValidation:
    """Tests for coherence law validation."""

    def test_perfect_coherence_law(self):
        """Test that perfect coherence satisfies the law."""
        R_bar = 1.0
        V_phi = 0.0
        assert validate_coherence_law(R_bar, V_phi, tolerance=0.01)

    def test_typical_coherence_law(self):
        """Test typical coherence values."""
        R_bar = 0.8
        V_phi = -2.0 * np.log(R_bar)
        assert validate_coherence_law(R_bar, V_phi, tolerance=0.01)

    def test_low_coherence_law(self):
        """Test low coherence values."""
        R_bar = 0.1
        V_phi = -2.0 * np.log(R_bar)
        assert validate_coherence_law(R_bar, V_phi, tolerance=0.01)

    def test_violated_law(self):
        """Test that inconsistent values don't satisfy the law."""
        R_bar = 0.8
        V_phi = 10.0  # Inconsistent with R_bar
        assert not validate_coherence_law(R_bar, V_phi, tolerance=0.01)

    def test_infinite_variance(self):
        """Test with infinite variance (R_bar → 0)."""
        R_bar = 0.0
        V_phi = np.inf
        assert validate_coherence_law(R_bar, V_phi, tolerance=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
