"""
Comprehensive Tests for Observability Geometry (OG) Metrics
============================================================

Tests all OG theory implementations:
- Effective Planck constant (h_eff)
- Representational cost (C)
- Coherence gradient (nabla V_phi)
- Action density (S)
- Regime classification
- Transition dynamics
- Shot allocation optimization
- Circuit quality evaluation

Author: ATLAS-Q Development Team
Date: December 2025
"""

import math
import numpy as np
import pytest

from atlas_q.coherence.og_metrics import (
    # Core metrics
    compute_effective_hbar,
    compute_representational_cost,
    compute_coherence_gradient,
    compute_action_density,
    compute_og_metrics,
    # Classification
    classify_observability_regime,
    compute_transition_parameter,
    compute_observability_score,
    ObservabilityRegime,
    # Advanced
    predict_measurement_precision,
    compute_optimal_shot_allocation,
    evaluate_circuit_quality,
    suggest_basis_for_measurement,
    # Dynamics
    track_transition_dynamics,
    TransitionDynamics,
    # Constants
    R_BAR_CRITICAL,
    TRANSITION_LOWER,
    TRANSITION_UPPER,
    H_EFF_CRITICAL,
    OGMetrics,
)


class TestOGConstants:
    """Test OG theoretical constants."""

    def test_critical_point(self):
        """R_bar_c = e^-2 is the critical observability threshold."""
        assert abs(R_BAR_CRITICAL - np.exp(-2)) < 1e-10
        assert abs(R_BAR_CRITICAL - 0.1353352832366127) < 1e-10

    def test_transition_bounds(self):
        """Transition bounds are e^-3 and e^-1."""
        assert abs(TRANSITION_LOWER - R_BAR_CRITICAL / np.e) < 1e-10
        assert abs(TRANSITION_UPPER - R_BAR_CRITICAL * np.e) < 1e-10

    def test_effective_hbar_at_critical(self):
        """h_eff at critical point is 4 * e^-1 ~ 1.47."""
        expected = 2 * np.sqrt(4) * np.sqrt(np.exp(-2))  # 4 * e^-1
        assert abs(H_EFF_CRITICAL - expected) < 1e-10
        assert abs(H_EFF_CRITICAL - 1.4715177646857693) < 1e-10


class TestEffectivePlanckConstant:
    """Test h_eff = 2 * sqrt(-2 * ln(R)) * sqrt(R)."""

    def test_heff_at_critical(self):
        """h_eff at R_bar_c should equal H_EFF_CRITICAL."""
        h_eff = compute_effective_hbar(R_BAR_CRITICAL)
        assert abs(h_eff - H_EFF_CRITICAL) < 1e-6

    def test_heff_at_unity(self):
        """h_eff at R_bar=1 should approach 0 (classical limit)."""
        h_eff = compute_effective_hbar(0.9999)
        assert h_eff < 0.1

    def test_heff_at_zero(self):
        """h_eff at R_bar->0 approaches 0 (sqrt(R) dominates)."""
        # h_eff = 2*sqrt(-2*ln(R))*sqrt(R)
        # As R->0: sqrt(-2*ln(R)) -> inf but sqrt(R) -> 0 faster
        # So h_eff -> 0 at R=0
        h_eff = compute_effective_hbar(0.0001)
        # At R=0.0001: h_eff = 2*sqrt(18.4)*0.01 = 2*4.3*0.01 = 0.086
        assert h_eff < 1.0  # Small value

    def test_heff_behavior(self):
        """h_eff has maximum at R=e^-1, approaches 0 at R=0 and R=1."""
        # h_eff = 2*sqrt(-2*ln(R))*sqrt(R) has maximum at R = e^-1 ~ 0.368
        max_R = np.exp(-1)
        h_max = compute_effective_hbar(max_R)

        # Should be larger than values at extremes
        h_low = compute_effective_hbar(0.01)
        h_high = compute_effective_hbar(0.99)

        assert h_max > h_low, f"h_max ({h_max}) should exceed h_eff at R=0.01 ({h_low})"
        assert h_max > h_high, f"h_max ({h_max}) should exceed h_eff at R=0.99 ({h_high})"

        # Verify decreasing behavior from max to R=1
        R_bars = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        h_effs = [compute_effective_hbar(R) for R in R_bars]

        for i in range(len(h_effs) - 1):
            assert h_effs[i] > h_effs[i + 1], \
                f"h_eff should decrease above R=e^-1: {h_effs[i]} > {h_effs[i+1]}"

    def test_heff_specific_values(self):
        """Test specific h_eff values."""
        # At R_bar = 0.5:
        # h_eff = 2 * sqrt(-2 * ln(0.5)) * sqrt(0.5)
        #       = 2 * sqrt(2 * 0.693) * 0.707
        #       ~ 1.665
        h_eff_05 = compute_effective_hbar(0.5)
        expected = 2 * np.sqrt(-2 * np.log(0.5)) * np.sqrt(0.5)
        assert abs(h_eff_05 - expected) < 1e-6


class TestRepresentationalCost:
    """Test C = 1/V_phi."""

    def test_cost_basic(self):
        """C = 1/V_phi exactly."""
        for V_phi in [0.5, 1.0, 2.0, 4.0]:
            C = compute_representational_cost(V_phi)
            assert abs(C - 1/V_phi) < 1e-10

    def test_cost_at_low_vphi(self):
        """Low V_phi (high coherence) means high cost."""
        C = compute_representational_cost(0.01)
        assert C == 100.0

    def test_cost_at_high_vphi(self):
        """High V_phi (low coherence) means low cost."""
        C = compute_representational_cost(100.0)
        assert C == 0.01

    def test_cost_infinity_handling(self):
        """V_phi=0 should give infinite cost."""
        C = compute_representational_cost(0.0)
        assert np.isinf(C)


class TestCoherenceGradient:
    """Test |nabla V_phi|^2 computation."""

    def test_uniform_coherence(self):
        """Uniform coherence should have zero gradient."""
        R_bars = [0.8, 0.8, 0.8, 0.8, 0.8]
        gradient = compute_coherence_gradient(R_bars)
        assert gradient < 1e-10

    def test_nonuniform_coherence(self):
        """Non-uniform coherence should have non-zero gradient."""
        R_bars = [0.9, 0.7, 0.5, 0.3, 0.1]
        gradient = compute_coherence_gradient(R_bars)
        assert gradient > 0

    def test_gradient_magnitude(self):
        """Test gradient magnitude for known case."""
        # Linear R_bar from 0.2 to 0.8 over 5 points
        R_bars = np.linspace(0.2, 0.8, 5)
        gradient = compute_coherence_gradient(R_bars)
        # Gradient should be non-zero for varying coherence
        assert gradient > 0

    def test_single_point(self):
        """Single point should have zero gradient."""
        gradient = compute_coherence_gradient([0.5])
        assert gradient == 0.0

    def test_two_points(self):
        """Two points should give gradient based on difference."""
        R_bars = [0.9, 0.1]  # Large difference
        gradient = compute_coherence_gradient(R_bars)
        assert gradient > 0


class TestActionDensity:
    """Test S = integral (nabla V_phi)^2 dx."""

    def test_uniform_action(self):
        """Uniform coherence should have zero action."""
        R_bars = np.full(100, 0.5)
        action = compute_action_density(R_bars)
        assert action < 0.01

    def test_varying_action(self):
        """Varying coherence should have non-zero action."""
        R_bars = np.linspace(0.2, 0.8, 100)
        action = compute_action_density(R_bars)
        assert action > 0

    def test_action_with_positions(self):
        """Test action with custom positions."""
        R_bars = np.linspace(0.2, 0.8, 10)
        positions = np.linspace(0, 10, 10)  # Total length 10
        action = compute_action_density(R_bars, positions)
        assert action > 0


class TestRegimeClassification:
    """Test observability regime classification."""

    def test_ir_regime(self):
        """High R_bar should be IR regime."""
        regime = classify_observability_regime(0.5)
        assert regime == ObservabilityRegime.IR

        regime = classify_observability_regime(0.9)
        assert regime == ObservabilityRegime.IR

    def test_air_regime(self):
        """Low R_bar should be AIR regime."""
        regime = classify_observability_regime(0.01)
        assert regime == ObservabilityRegime.AIR

        regime = classify_observability_regime(0.03)
        assert regime == ObservabilityRegime.AIR

    def test_transition_regime(self):
        """R_bar near e^-2 should be TRANSITION."""
        regime = classify_observability_regime(R_BAR_CRITICAL)
        assert regime == ObservabilityRegime.TRANSITION

        regime = classify_observability_regime(0.1)
        assert regime == ObservabilityRegime.TRANSITION

        regime = classify_observability_regime(0.2)
        assert regime == ObservabilityRegime.TRANSITION

    def test_regime_boundaries(self):
        """Test exact boundary behavior."""
        # Just below lower bound -> AIR
        regime = classify_observability_regime(TRANSITION_LOWER - 0.001)
        assert regime == ObservabilityRegime.AIR

        # Just above upper bound -> IR
        regime = classify_observability_regime(TRANSITION_UPPER + 0.001)
        assert regime == ObservabilityRegime.IR


class TestTransitionParameter:
    """Test tau = (R_bar - R_c) / R_c."""

    def test_at_critical(self):
        """tau should be 0 at critical point."""
        tau = compute_transition_parameter(R_BAR_CRITICAL)
        assert abs(tau) < 1e-10

    def test_above_critical(self):
        """tau should be positive above critical."""
        tau = compute_transition_parameter(0.5)
        assert tau > 0

    def test_below_critical(self):
        """tau should be negative below critical."""
        tau = compute_transition_parameter(0.05)
        assert tau < 0

    def test_tau_magnitude(self):
        """Test specific tau values."""
        # At R_bar = 2*R_c, tau = 1
        tau = compute_transition_parameter(2 * R_BAR_CRITICAL)
        assert abs(tau - 1.0) < 1e-10

        # At R_bar = R_c/2, tau = -0.5
        tau = compute_transition_parameter(R_BAR_CRITICAL / 2)
        assert abs(tau - (-0.5)) < 1e-10


class TestObservabilityScore:
    """Test observability score computation."""

    def test_score_at_critical(self):
        """Score should be ~0.5 at critical point."""
        score = compute_observability_score(R_BAR_CRITICAL)
        assert abs(score - 0.5) < 0.01

    def test_score_bounds(self):
        """Score should be in [0, 1]."""
        for R_bar in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            score = compute_observability_score(R_bar)
            assert 0 <= score <= 1

    def test_score_monotonicity(self):
        """Score should increase with R_bar."""
        R_bars = [0.1, 0.3, 0.5, 0.7, 0.9]
        scores = [compute_observability_score(R) for R in R_bars]

        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


class TestOGMetricsDataclass:
    """Test OGMetrics dataclass and compute_og_metrics function."""

    def test_metrics_from_rbar(self):
        """Create metrics from R_bar."""
        metrics = compute_og_metrics(R_bar=0.5)

        assert abs(metrics.R_bar - 0.5) < 1e-10
        assert abs(metrics.V_phi - (-2 * np.log(0.5))) < 1e-6
        assert metrics.h_eff > 0
        assert metrics.rep_cost > 0
        assert metrics.regime == ObservabilityRegime.IR
        assert 0 <= metrics.observability_score <= 1

    def test_metrics_from_vphi(self):
        """Create metrics from V_phi."""
        V_phi = 2.0
        metrics = compute_og_metrics(V_phi=V_phi)

        assert abs(metrics.V_phi - V_phi) < 1e-10
        # R_bar = exp(-V_phi/2) = exp(-1) ~ 0.368
        expected_R = np.exp(-V_phi / 2)
        assert abs(metrics.R_bar - expected_R) < 1e-6

    def test_metrics_from_amplitudes(self):
        """Create metrics from state amplitudes."""
        # Pure state |0>
        amplitudes = np.array([1, 0, 0, 0], dtype=complex)
        metrics = compute_og_metrics(amplitudes=amplitudes)

        assert metrics.R_bar > 0.99
        assert metrics.regime == ObservabilityRegime.IR
        assert metrics.spectral_coherence > 0.99

    def test_metrics_from_measurements(self):
        """Create metrics from Pauli measurements."""
        outcomes = np.array([0.9, 0.85, 0.88, 0.87])
        metrics = compute_og_metrics(measurement_outcomes=outcomes)

        assert 0 < metrics.R_bar < 1
        assert metrics.V_phi > 0

    def test_metrics_with_qubit_data(self):
        """Test gradient computation with per-qubit data."""
        qubit_R_bars = np.array([0.9, 0.8, 0.7, 0.6])
        metrics = compute_og_metrics(R_bar=0.75, qubit_R_bars=qubit_R_bars)

        assert metrics.coherence_gradient is not None
        assert metrics.action_density is not None
        assert metrics.coherence_gradient > 0  # Non-uniform

    def test_metrics_as_dict(self):
        """Test conversion to dictionary."""
        metrics = compute_og_metrics(R_bar=0.5)
        d = metrics.as_dict()

        assert 'R_bar' in d
        assert 'V_phi' in d
        assert 'h_eff' in d
        assert 'regime' in d
        assert d['regime'] == 'ir'

    def test_metrics_str(self):
        """Test string representation."""
        metrics = compute_og_metrics(R_bar=0.5)
        s = str(metrics)

        assert 'R̄=' in s or 'R_bar' in s
        assert 'V_φ=' in s or 'V_phi' in s


class TestPrecisionPrediction:
    """Test measurement precision prediction."""

    def test_precision_high_coherence(self):
        """High coherence should give good precision."""
        result = predict_measurement_precision(0.9, n_shots=1000)

        assert result['h_eff'] < H_EFF_CRITICAL
        assert result['effective_precision'] < 0.1

    def test_precision_low_coherence(self):
        """Low coherence results in different h_eff."""
        result = predict_measurement_precision(0.1, n_shots=1000)

        # h_eff varies with coherence (not monotonically)
        assert result['h_eff'] > 0
        assert result['precision_limit'] > 0

    def test_precision_shot_dependence(self):
        """More shots should improve shot precision."""
        result_100 = predict_measurement_precision(0.5, n_shots=100)
        result_1000 = predict_measurement_precision(0.5, n_shots=1000)

        assert result_1000['shot_precision'] < result_100['shot_precision']


class TestShotAllocation:
    """Test optimal shot allocation."""

    def test_allocation_conserves_total(self):
        """Total shots should be conserved."""
        R_bars = [0.9, 0.5, 0.2]
        total = 1000
        shots = compute_optimal_shot_allocation(R_bars, total)

        assert sum(shots) == total

    def test_allocation_favors_low_coherence(self):
        """Low coherence groups should get more shots (harder to measure)."""
        R_bars = [0.9, 0.1]  # High and low coherence
        shots = compute_optimal_shot_allocation(R_bars, total_shots=1000)

        # Low coherence (R_bar=0.1) has higher V_phi = harder to measure
        # Should get more shots
        assert shots[1] > shots[0], f"Low coherence should get more shots: {shots}"

    def test_allocation_minimum(self):
        """Each group should get at least minimum shots."""
        R_bars = [0.9, 0.9, 0.9]
        min_shots = 50
        shots = compute_optimal_shot_allocation(R_bars, total_shots=200,
                                                min_shots_per_group=min_shots)

        for s in shots:
            assert s >= min_shots

    def test_allocation_empty(self):
        """Empty list should return empty."""
        shots = compute_optimal_shot_allocation([], total_shots=100)
        assert shots == []


class TestCircuitQuality:
    """Test circuit quality evaluation."""

    def test_uniform_quality(self):
        """Uniform high coherence should give high quality."""
        qubit_R_bars = [0.9, 0.9, 0.9, 0.9]
        result = evaluate_circuit_quality(qubit_R_bars)

        assert result['mean_R_bar'] == 0.9
        assert result['min_R_bar'] == 0.9
        assert result['coherence_gradient'] < 0.01
        assert result['quality_score'] > 0.8

    def test_nonuniform_quality(self):
        """Non-uniform coherence should lower quality."""
        qubit_R_bars = [0.9, 0.5, 0.3, 0.1]
        result = evaluate_circuit_quality(qubit_R_bars)

        assert result['mean_R_bar'] < 0.9
        assert result['min_R_bar'] == 0.1
        assert result['coherence_gradient'] > 0
        assert result['quality_score'] < 0.5  # Penalized by non-uniformity

    def test_bottleneck_regime(self):
        """Regime should be classified by minimum coherence."""
        qubit_R_bars = [0.9, 0.9, 0.03, 0.9]  # One clearly AIR qubit (0.03 < TRANSITION_LOWER)
        result = evaluate_circuit_quality(qubit_R_bars)

        assert result['regime'] == 'air'  # Bottleneck determines regime


class TestBasisSuggestion:
    """Test measurement basis suggestion."""

    def test_ir_regime_suggestion(self):
        """IR regime should suggest computational basis."""
        result = suggest_basis_for_measurement(0.9)

        assert result['regime'] == 'ir'
        assert result['preferred_basis'] == 'Z'

    def test_air_regime_suggestion(self):
        """AIR regime should suggest alternative basis."""
        result = suggest_basis_for_measurement(0.03)

        assert result['regime'] == 'air'
        assert result['preferred_basis'] == 'Y'

    def test_transition_suggestion(self):
        """Transition regime should suggest superposition basis."""
        result = suggest_basis_for_measurement(0.15)

        assert result['regime'] == 'transition'
        assert result['preferred_basis'] == 'X'


class TestTransitionDynamics:
    """Test transition dynamics tracking."""

    def test_degrading_trajectory(self):
        """Decreasing R_bar should be 'degrading'."""
        history = [0.8, 0.6, 0.4, 0.3, 0.2]
        dynamics = track_transition_dynamics(history)

        assert dynamics.direction == 'degrading'
        assert dynamics.velocity < 0
        assert dynamics.current_R_bar == 0.2
        assert dynamics.previous_R_bar == 0.3

    def test_improving_trajectory(self):
        """Increasing R_bar should be 'improving'."""
        history = [0.2, 0.4, 0.5, 0.6, 0.8]
        dynamics = track_transition_dynamics(history)

        assert dynamics.direction == 'improving'
        assert dynamics.velocity > 0

    def test_stable_trajectory(self):
        """Constant R_bar should be 'stable'."""
        history = [0.5, 0.5, 0.5, 0.5]
        dynamics = track_transition_dynamics(history)

        assert dynamics.direction == 'stable'
        assert abs(dynamics.velocity) < 1e-6

    def test_threshold_crossing_prediction(self):
        """Should predict threshold crossing."""
        history = [0.3, 0.25, 0.2, 0.15]  # Heading toward critical
        dynamics = track_transition_dynamics(history)

        # Should predict crossing since we're heading down
        if dynamics.velocity < 0 and dynamics.current_R_bar > R_BAR_CRITICAL:
            assert dynamics.steps_to_critical is not None

    def test_will_cross_threshold(self):
        """Test threshold crossing prediction method."""
        history = [0.2, 0.18, 0.16, 0.14]  # Just above critical, going down
        dynamics = track_transition_dynamics(history)

        # With velocity ~ -0.02 per step and starting at 0.14
        # Critical is 0.135, so should cross in ~1-2 steps
        assert dynamics.will_cross_threshold(5)

    def test_single_point(self):
        """Single point history should be handled."""
        dynamics = track_transition_dynamics([0.5])

        assert dynamics.current_R_bar == 0.5
        assert dynamics.direction == 'stable'


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_extreme_rbar_values(self):
        """Test extreme R_bar values."""
        # Very small R_bar - h_eff approaches 0 as sqrt(R) dominates
        metrics = compute_og_metrics(R_bar=1e-10)
        assert metrics.regime == ObservabilityRegime.AIR
        assert metrics.h_eff < 0.01  # h_eff is small at extreme R_bar

        # Very large R_bar - h_eff approaches 0
        metrics = compute_og_metrics(R_bar=0.9999999)
        assert metrics.regime == ObservabilityRegime.IR
        assert metrics.h_eff < 0.1

    def test_boundary_values(self):
        """Test exact boundary values."""
        metrics = compute_og_metrics(R_bar=0.0)
        assert metrics.R_bar == 0.0
        assert np.isinf(metrics.V_phi)

        metrics = compute_og_metrics(R_bar=1.0)
        assert metrics.R_bar == 1.0

    def test_complex_amplitudes(self):
        """Test with complex amplitudes."""
        # Superposition state
        amplitudes = np.array([1, 1j, -1, -1j], dtype=complex) / 2
        metrics = compute_og_metrics(amplitudes=amplitudes)

        assert 0 <= metrics.R_bar <= 1
        assert metrics.V_phi >= 0

    def test_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            compute_og_metrics()  # No input

        # Note: R_bar is clipped to [0, 1], so -0.5 becomes 0.0 (not an error)
        # Test that missing all inputs raises error
        with pytest.raises(ValueError):
            compute_og_metrics(amplitudes=None, measurement_outcomes=None)


class TestIntegration:
    """Integration tests combining multiple OG components."""

    def test_full_pipeline(self):
        """Test complete OG analysis pipeline."""
        # Generate random state
        np.random.seed(42)
        amplitudes = np.random.randn(16) + 1j * np.random.randn(16)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        # Compute metrics
        metrics = compute_og_metrics(amplitudes=amplitudes)

        # Predict precision
        precision = predict_measurement_precision(metrics.R_bar, n_shots=1000)

        # Get basis suggestion
        basis = suggest_basis_for_measurement(metrics.R_bar)

        # All should be consistent
        assert (metrics.regime == ObservabilityRegime.IR) == (basis['regime'] == 'ir')
        assert precision['h_eff'] == metrics.h_eff

    def test_vqe_simulation(self):
        """Simulate VQE coherence tracking."""
        # Initial high coherence
        R_bar_history = [0.9]

        # Simulate degradation during optimization
        for _ in range(10):
            # Add some noise to simulate VQE iteration
            new_R = R_bar_history[-1] - 0.02 + np.random.randn() * 0.01
            new_R = np.clip(new_R, 0.01, 0.99)
            R_bar_history.append(new_R)

        # Track dynamics
        dynamics = track_transition_dynamics(R_bar_history)

        # Compute final metrics
        metrics = compute_og_metrics(R_bar=R_bar_history[-1])

        # Make decisions based on regime
        if metrics.regime == ObservabilityRegime.AIR:
            # Would typically stop or restart
            assert metrics.R_bar < TRANSITION_LOWER

    def test_circuit_optimization(self):
        """Simulate circuit optimization using OG metrics."""
        # Initial non-uniform coherence
        initial_R_bars = [0.9, 0.5, 0.3, 0.7]
        initial_quality = evaluate_circuit_quality(initial_R_bars)

        # "Optimized" coherence - more uniform
        optimized_R_bars = [0.7, 0.65, 0.6, 0.68]
        optimized_quality = evaluate_circuit_quality(optimized_R_bars)

        # More uniform coherence should have lower gradient
        assert optimized_quality['coherence_gradient'] < initial_quality['coherence_gradient']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
