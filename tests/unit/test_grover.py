#!/usr/bin/env python3
"""
Unit tests for Grover's quantum search algorithm

Tests core functionality including:
- Configuration and validation
- Oracle implementations (FunctionOracle, BitmapOracle)
- Optimal iteration calculation
- Search functionality via convenience function
- Edge cases and error handling
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
    DiffusionOperator,
    grover_search,
    calculate_grover_iterations,
)


class TestGroverConfig:
    """Test suite for GroverConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GroverConfig(n_qubits=4)
        assert config.n_qubits == 4
        assert config.oracle_type == 'function'
        assert config.auto_iterations is True
        assert config.max_iterations == 1000
        assert config.chi_max == 256
        assert config.device == 'cuda'
        assert config.dtype == torch.complex128
        assert config.verbose is False
        assert config.measure_success_prob is False

    def test_custom_config(self):
        """Test custom configuration"""
        config = GroverConfig(
            n_qubits=8,
            oracle_type='bitmap',
            auto_iterations=False,
            chi_max=128,
            device='cpu',
            dtype=torch.complex64,
            verbose=True,
        )
        assert config.n_qubits == 8
        assert config.oracle_type == 'bitmap'
        assert config.auto_iterations is False
        assert config.chi_max == 128
        assert config.device == 'cpu'
        assert config.dtype == torch.complex64
        assert config.verbose is True


class TestCalculateIterations:
    """Test suite for optimal iteration calculation"""

    def test_single_marked_state(self):
        """Test iteration count for single marked state"""
        # For N=16, M=1: k* ≈ floor(π/4 * √16) ≈ floor(π/4 * 4) ≈ 3
        k = calculate_grover_iterations(n_qubits=4, n_marked=1)
        assert k == 3

    def test_multiple_marked_states(self):
        """Test iteration count for multiple marked states"""
        # For N=16, M=4: k* ≈ floor(π/4 * √(16/4)) ≈ floor(π/4 * 2) ≈ 1
        k = calculate_grover_iterations(n_qubits=4, n_marked=4)
        assert k == 1

    def test_half_marked_states(self):
        """Test iteration count when half the states are marked"""
        # For N=8, M=4: k* ≈ floor(π/4 * √(8/4)) ≈ floor(π/4 * √2) ≈ 1
        k = calculate_grover_iterations(n_qubits=3, n_marked=4)
        assert k == 1

    def test_large_space(self):
        """Test iteration count for larger search space"""
        # For N=1024 (2^10), M=1: k* ≈ floor(π/4 * √1024) ≈ floor(π/4 * 32) ≈ 25
        k = calculate_grover_iterations(n_qubits=10, n_marked=1)
        assert 24 <= k <= 26  # Allow small floating point variation

    def test_edge_case_all_marked(self):
        """Test edge case where all states are marked"""
        # When M=N, no amplification needed (but return at least 0)
        k = calculate_grover_iterations(n_qubits=3, n_marked=8)
        assert k == 0

    def test_edge_case_zero_marked(self):
        """Test edge case with zero marked states"""
        with pytest.raises(ValueError, match="n_marked must be in range"):
            calculate_grover_iterations(n_qubits=4, n_marked=0)

    def test_edge_case_too_many_marked(self):
        """Test edge case with more marked states than exist"""
        with pytest.raises(ValueError, match="n_marked must be in range"):
            calculate_grover_iterations(n_qubits=2, n_marked=10)


class TestFunctionOracle:
    """Test suite for FunctionOracle"""

    @pytest.fixture
    def device(self):
        """Test device"""
        return 'cpu'  # Use CPU for deterministic tests

    @pytest.fixture
    def dtype(self):
        """Test dtype"""
        return torch.complex128

    def test_oracle_initialization(self, device, dtype):
        """Test oracle initialization with marking function"""
        def mark_3_and_5(x):
            return x == 3 or x == 5

        oracle = FunctionOracle(
            n_qubits=3,
            marking_fn=mark_3_and_5,
            device=device,
            dtype=dtype
        )
        assert oracle.n_qubits == 3
        assert oracle.n_marked == 2
        assert oracle.marking_fn(3) is True
        assert oracle.marking_fn(5) is True
        assert oracle.marking_fn(0) is False

    def test_oracle_marks_single_state(self, device, dtype):
        """Test oracle correctly counts single marked state"""
        def mark_5(x):
            return x == 5

        oracle = FunctionOracle(n_qubits=3, marking_fn=mark_5, device=device, dtype=dtype)
        assert oracle.n_marked == 1
        assert oracle.get_marked_count() == 1

    def test_oracle_marks_multiple_states(self, device, dtype):
        """Test oracle correctly counts multiple marked states"""
        def mark_even(x):
            return x % 2 == 0

        oracle = FunctionOracle(n_qubits=3, marking_fn=mark_even, device=device, dtype=dtype)
        assert oracle.n_marked == 4  # States: 0, 2, 4, 6
        assert oracle.get_marked_count() == 4


class TestBitmapOracle:
    """Test suite for BitmapOracle"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    @pytest.fixture
    def dtype(self):
        return torch.complex128

    def test_bitmap_initialization(self, device, dtype):
        """Test bitmap oracle initialization"""
        marked = {3, 5, 7}
        oracle = BitmapOracle(
            n_qubits=3,
            marked_states=marked,
            device=device,
            dtype=dtype
        )
        assert oracle.n_qubits == 3
        assert oracle.n_marked == 3
        assert oracle.marked_states == marked

    def test_bitmap_empty_marked_set(self, device, dtype):
        """Test bitmap oracle with empty marked set"""
        with pytest.raises(ValueError, match="marked_states must contain at least one state"):
            BitmapOracle(n_qubits=3, marked_states=set(), device=device, dtype=dtype)

    def test_bitmap_invalid_state(self, device, dtype):
        """Test bitmap oracle with invalid state value"""
        # State 10 is out of range for 3 qubits (max 7)
        with pytest.raises(ValueError, match="State .* out of range"):
            BitmapOracle(n_qubits=3, marked_states={10}, device=device, dtype=dtype)

    def test_bitmap_negative_state(self, device, dtype):
        """Test bitmap oracle with negative state"""
        with pytest.raises(ValueError, match="State .* out of range"):
            BitmapOracle(n_qubits=3, marked_states={-1}, device=device, dtype=dtype)


class TestDiffusionOperator:
    """Test suite for DiffusionOperator"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    @pytest.fixture
    def dtype(self):
        return torch.complex128

    def test_diffusion_initialization(self, device, dtype):
        """Test diffusion operator initialization"""
        diffusion = DiffusionOperator(n_qubits=3, device=device, dtype=dtype)
        assert diffusion.n_qubits == 3
        assert diffusion.device == device
        assert diffusion.dtype == dtype


class TestGroverSearch:
    """Test suite for GroverSearch main algorithm"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    @pytest.fixture
    def dtype(self):
        return torch.complex64  # Use lighter dtype for faster tests

    def test_grover_initialization_with_function_oracle(self, device, dtype):
        """Test GroverSearch initialization with function oracle"""
        config = GroverConfig(n_qubits=3, device=device, dtype=dtype, verbose=False)

        def mark_5(x):
            return x == 5

        oracle = FunctionOracle(3, mark_5, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)
        assert grover.config.n_qubits == 3
        assert grover.oracle.n_marked == 1

    def test_grover_initialization_with_bitmap_oracle(self, device, dtype):
        """Test GroverSearch initialization with bitmap oracle"""
        config = GroverConfig(
            n_qubits=3,
            oracle_type='bitmap',
            device=device,
            dtype=dtype,
            verbose=False
        )

        oracle = BitmapOracle(3, {3, 7}, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)
        assert grover.config.n_qubits == 3
        assert grover.oracle.n_marked == 2

    def test_grover_oracle_qubit_mismatch(self, device, dtype):
        """Test that oracle and config must have matching n_qubits"""
        config = GroverConfig(n_qubits=4, device=device, dtype=dtype)
        oracle = BitmapOracle(3, {5}, device=device, dtype=dtype)  # 3 qubits

        with pytest.raises(ValueError, match="Oracle has .* qubits but config specifies"):
            GroverSearch(oracle, config)

    def test_grover_simple_2qubit_search(self, device, dtype):
        """Test Grover search on 2-qubit system (4 states)"""
        config = GroverConfig(
            n_qubits=2,
            device=device,
            dtype=dtype,
            verbose=False,
            chi_max=16
        )

        # Search for state |11> = 3
        oracle = BitmapOracle(2, {3}, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)

        # Run search with optimal iterations (should be 1 for N=4, M=1)
        result = grover.run(iterations=1)

        # Check result structure
        assert 'measured_state' in result
        assert 'success_probability' in result
        assert 'iterations_used' in result
        assert 'runtime_ms' in result

        assert result['iterations_used'] == 1
        assert 0 <= result['measured_state'] < 4

    def test_grover_3qubit_search(self, device, dtype):
        """Test Grover search on 3-qubit system (8 states)"""
        config = GroverConfig(
            n_qubits=3,
            device=device,
            dtype=dtype,
            verbose=False,
            chi_max=32
        )

        # Search for state |101> = 5
        def mark_5(x):
            return x == 5

        oracle = FunctionOracle(3, mark_5, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)

        # Optimal iterations for N=8, M=1: floor(π/4 * √8) ≈ 2
        result = grover.run(iterations=2)

        assert result['iterations_used'] == 2
        assert 0 <= result['measured_state'] < 8

    def test_grover_auto_iterations(self, device, dtype):
        """Test automatic iteration calculation"""
        config = GroverConfig(
            n_qubits=3,
            auto_iterations=True,
            device=device,
            dtype=dtype,
            verbose=False
        )

        oracle = BitmapOracle(3, {5}, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)

        # Run with auto iterations (should calculate k* = 2)
        result = grover.run()

        assert result['iterations_used'] == 2

    def test_grover_multiple_marked_states(self, device, dtype):
        """Test Grover with multiple marked states"""
        config = GroverConfig(
            n_qubits=3,
            device=device,
            dtype=dtype,
            verbose=False
        )

        # Mark states {2, 5, 7}
        oracle = BitmapOracle(3, {2, 5, 7}, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)

        # For N=8, M=3: k* ≈ floor(π/4 * √(8/3)) ≈ 1
        result = grover.run(iterations=1)

        assert result['iterations_used'] == 1
        assert 0 <= result['measured_state'] < 8

    def test_grover_zero_iterations_raises_error(self, device, dtype):
        """Test that zero iterations raises ValueError"""
        config = GroverConfig(n_qubits=2, device=device, dtype=dtype, verbose=False)
        oracle = BitmapOracle(2, {3}, device=device, dtype=dtype)
        grover = GroverSearch(oracle, config)

        with pytest.raises(ValueError, match="Cannot run with 0 iterations"):
            grover.run(iterations=0)


class TestGroverConvenience:
    """Test suite for convenience function grover_search()"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    def test_convenience_function_basic(self, device):
        """Test convenience function with basic parameters"""
        result = grover_search(
            n_qubits=2,
            marked_states={3},
            iterations=1,
            device=device,
            verbose=False
        )

        assert 'measured_state' in result
        assert result['iterations_used'] == 1

    def test_convenience_function_auto_iterations(self, device):
        """Test convenience function with automatic iterations"""
        result = grover_search(
            n_qubits=3,
            marked_states={5},
            device=device,
            verbose=False
        )

        # Should automatically calculate k* = 2 for N=8, M=1
        assert result['iterations_used'] == 2

    def test_convenience_function_with_list(self, device):
        """Test convenience function with list of marked states"""
        result = grover_search(
            n_qubits=2,
            marked_states=[1, 3],  # List instead of set
            iterations=1,
            device=device,
            verbose=False
        )

        assert result['iterations_used'] == 1
        assert 0 <= result['measured_state'] < 4

    def test_convenience_function_with_callable(self, device):
        """Test convenience function with marking function"""
        def mark_odd(x):
            return x % 2 == 1

        result = grover_search(
            n_qubits=2,
            marked_states=mark_odd,  # Callable function
            iterations=1,
            device=device,
            verbose=False
        )

        assert result['iterations_used'] == 1

    def test_convenience_function_invalid_marked_state(self, device):
        """Test convenience function with invalid marked state"""
        with pytest.raises(ValueError, match="State .* out of range"):
            grover_search(
                n_qubits=2,
                marked_states={10},  # Out of range for 2 qubits
                device=device
            )


class TestGroverEdgeCases:
    """Test suite for edge cases and error handling"""

    @pytest.fixture
    def device(self):
        return 'cpu'

    @pytest.fixture
    def dtype(self):
        return torch.complex64

    def test_single_qubit_search(self, device, dtype):
        """Test Grover on 1-qubit system"""
        result = grover_search(
            n_qubits=1,
            marked_states={1},
            iterations=1,
            device=device,
            verbose=False
        )

        assert result['iterations_used'] == 1
        assert result['measured_state'] in [0, 1]

    def test_excessive_iterations(self, device, dtype):
        """Test that excessive iterations still work (though not optimal)"""
        result = grover_search(
            n_qubits=2,
            marked_states={3},
            iterations=10,
            device=device,
            verbose=False
        )

        assert result['iterations_used'] == 10
        assert 0 <= result['measured_state'] < 4


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
