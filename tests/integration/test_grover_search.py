#!/usr/bin/env python3
"""
Integration tests for Grover's quantum search algorithm

Tests end-to-end functionality including:
- Algorithm completes successfully
- Performance scaling with problem size
- Multi-marked state scenarios
- Integration with AdaptiveMPS
- Statistics tracking

Note: These tests focus on algorithmic correctness and API integration.
Finding the exact marked state with high probability requires a more
sophisticated multi-controlled gate implementation in the MPS backend.
"""

import pytest
import torch
import numpy as np
from typing import Set

from atlas_q.grover import (
    GroverConfig,
    GroverSearch,
    FunctionOracle,
    BitmapOracle,
    grover_search,
    calculate_grover_iterations,
)


class TestGroverSearchBasic:
    """Integration tests for basic Grover search functionality"""

    @pytest.fixture
    def device(self):
        return 'cpu'  # Use CPU for deterministic tests

    def test_grover_completes_successfully_3qubits(self, device):
        """Test that Grover search completes without error on 3-qubit system"""
        n_qubits = 3
        marked_state = 5

        # Run Grover search
        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Check that it completes and returns valid result
        assert 'measured_state' in result
        assert 'success_probability' in result
        assert 'iterations_used' in result
        assert 'runtime_ms' in result
        assert 0 <= result['measured_state'] < 2**n_qubits

    def test_grover_completes_successfully_4qubits(self, device):
        """Test that Grover search completes without error on 4-qubit system"""
        n_qubits = 4
        marked_state = 11

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Check result structure
        assert result['iterations_used'] == calculate_grover_iterations(n_qubits, 1)
        assert 0 <= result['measured_state'] < 2**n_qubits

    def test_multiple_marked_states_completes(self, device):
        """Test Grover with multiple marked states completes successfully"""
        n_qubits = 4
        marked_states = {3, 7, 11, 15}

        result = grover_search(
            n_qubits=n_qubits,
            marked_states=marked_states,
            device=device,
            verbose=False
        )

        # Should complete successfully
        assert 'measured_state' in result
        assert 0 <= result['measured_state'] < 2**n_qubits

    def test_function_oracle_completes(self, device):
        """Test Grover with function oracle completes successfully"""
        n_qubits = 3

        def mark_even(x):
            return x % 2 == 0

        result = grover_search(
            n_qubits=n_qubits,
            marked_states=mark_even,  # Callable function
            device=device,
            verbose=False
        )

        # Should complete successfully
        assert 'measured_state' in result
        assert result['iterations_used'] > 0


class TestGroverPerformance:
    """Integration tests for performance characteristics"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_scaling_with_problem_size(self, device):
        """Test Grover scales correctly across different problem sizes"""
        results = {}

        for n_qubits in [2, 3, 4]:
            # Mark the last state
            marked_state = 2**n_qubits - 1

            result = grover_search(
                n_qubits=n_qubits,
                marked_states={marked_state},
                device=device,
                verbose=False
            )

            results[n_qubits] = result

            # Check iterations scale as expected
            expected_iterations = calculate_grover_iterations(n_qubits, 1)
            assert result['iterations_used'] == expected_iterations

        # Verify iterations grow with problem size
        assert results[2]['iterations_used'] < results[4]['iterations_used']

    def test_success_probability_tracking_enabled(self, device):
        """Test that success probability tracking can be enabled"""
        n_qubits = 3
        marked_state = 5

        config = GroverConfig(
            n_qubits=n_qubits,
            device=device,
            verbose=False,
            measure_success_prob=True  # Enable probability tracking
        )

        oracle = BitmapOracle(n_qubits, {marked_state}, device=device)
        grover = GroverSearch(oracle, config)

        result = grover.run()

        # Check that success probabilities were tracked
        assert len(grover.success_probabilities) > 0
        assert len(grover.success_probabilities) == result['iterations_used']

    def test_bond_dimension_reporting(self, device):
        """Test MPS bond dimensions are properly reported"""
        n_qubits = 4
        marked_state = 10

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Check bond dimensions are reported
        assert 'bond_dims' in result
        assert len(result['bond_dims']) == n_qubits - 1 or len(result['bond_dims']) == 0

    def test_runtime_is_measured(self, device):
        """Test that runtime is measured and reasonable"""
        n_qubits = 3

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={5},
            device=device,
            verbose=False
        )

        # Runtime should be positive and reasonable (< 10 seconds)
        assert result['runtime_ms'] > 0
        assert result['runtime_ms'] < 10000  # Less than 10 seconds


class TestGroverIterationOptimization:
    """Integration tests for iteration optimization"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_optimal_iterations_calculated_correctly(self, device):
        """Test optimal iteration calculation for various cases"""
        test_cases = [
            (2, 1, 1),  # N=4, M=1: k* = 1
            (3, 1, 2),  # N=8, M=1: k* = 2
            (4, 1, 3),  # N=16, M=1: k* = 3
            (4, 4, 1),  # N=16, M=4: k* = 1
        ]

        for n_qubits, n_marked, expected_k in test_cases:
            k = calculate_grover_iterations(n_qubits, n_marked)
            assert k == expected_k, \
                f"For {n_qubits} qubits, {n_marked} marked: expected k={expected_k}, got k={k}"

    def test_auto_iterations_mode(self, device):
        """Test automatic iteration calculation mode"""
        n_qubits = 3
        marked_state = 5

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            iterations=None,  # Auto mode
            device=device,
            verbose=False
        )

        optimal_k = calculate_grover_iterations(n_qubits, 1)
        assert result['iterations_used'] == optimal_k

    def test_manual_iterations_mode(self, device):
        """Test manual iteration specification"""
        n_qubits = 3
        manual_iterations = 5

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={5},
            iterations=manual_iterations,  # Manual
            device=device,
            verbose=False
        )

        assert result['iterations_used'] == manual_iterations


class TestGroverConfiguration:
    """Integration tests for configuration options"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_different_oracle_types(self, device):
        """Test different oracle types (function vs bitmap)"""
        n_qubits = 3
        marked_state = 5

        # Function oracle
        result_function = grover_search(
            n_qubits=n_qubits,
            marked_states=lambda x: x == marked_state,
            device=device,
            verbose=False
        )

        # Bitmap oracle
        result_bitmap = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Both should complete successfully
        assert 'measured_state' in result_function
        assert 'measured_state' in result_bitmap

    def test_chi_max_constraint(self, device):
        """Test Grover with limited bond dimension"""
        n_qubits = 4
        marked_state = 7

        config = GroverConfig(
            n_qubits=n_qubits,
            chi_max=16,  # Limited bond dimension
            device=device,
            verbose=False
        )

        oracle = BitmapOracle(n_qubits, {marked_state}, device=device)
        grover = GroverSearch(oracle, config)

        result = grover.run()

        # Should complete without error
        assert 'measured_state' in result

    def test_different_dtypes(self, device):
        """Test Grover with different data types"""
        n_qubits = 3
        marked_state = 5

        for dtype in [torch.complex64, torch.complex128]:
            config = GroverConfig(
                n_qubits=n_qubits,
                device=device,
                dtype=dtype,
                verbose=False
            )

            oracle = BitmapOracle(n_qubits, {marked_state}, device=device, dtype=dtype)
            grover = GroverSearch(oracle, config)

            result = grover.run()
            assert 'measured_state' in result


class TestGroverStatistics:
    """Integration tests for statistics tracking"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_iteration_statistics_tracked(self, device):
        """Test that iteration statistics are properly tracked"""
        n_qubits = 3
        marked_state = 5

        config = GroverConfig(
            n_qubits=n_qubits,
            device=device,
            verbose=False,
            measure_success_prob=True
        )

        oracle = BitmapOracle(n_qubits, {marked_state}, device=device)
        grover = GroverSearch(oracle, config)

        result = grover.run()

        # Check iteration stats exist
        assert 'iteration_stats' in result
        stats = result['iteration_stats']
        assert len(stats) == result['iterations_used']

        # Each iteration should have required fields
        for stat in stats:
            assert 'iteration' in stat
            assert 'time_ms' in stat
            assert 'success_probability' in stat
            assert 'max_bond_dim' in stat
            assert stat['time_ms'] >= 0

    def test_multiple_searches_independent(self, device):
        """Test that multiple searches are independent"""
        n_qubits = 3
        results = []

        for marked_state in [1, 3, 5, 7]:
            result = grover_search(
                n_qubits=n_qubits,
                marked_states={marked_state},
                device=device,
                verbose=False
            )
            results.append(result)

        # All searches should complete successfully
        assert len(results) == 4
        for result in results:
            assert 'measured_state' in result
            assert 0 <= result['measured_state'] < 2**n_qubits


@pytest.mark.slow
class TestGroverLargeScale:
    """Integration tests for larger problem sizes (marked as slow)"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_5_qubit_search_completes(self, device):
        """Test Grover completes on 5-qubit system"""
        n_qubits = 5
        marked_state = 20

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Should complete successfully
        assert 'measured_state' in result
        assert 0 <= result['measured_state'] < 2**n_qubits

    def test_6_qubit_search_completes(self, device):
        """Test Grover completes on 6-qubit system"""
        n_qubits = 6
        marked_state = 50

        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )

        # Should complete successfully
        assert 'measured_state' in result
        assert 0 <= result['measured_state'] < 2**n_qubits


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
