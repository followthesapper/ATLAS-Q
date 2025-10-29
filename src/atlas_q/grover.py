"""
Grover's Search Algorithm for Quantum Database Search

Implements Grover's amplitude amplification algorithm for searching unstructured
databases with quadratic speedup over classical methods.

Mathematical Background:
- Search space: N = 2^n elements
- Marked items: M items to find
- Classical complexity: O(N)
- Quantum complexity: O(√(N/M)) iterations
- Optimal iterations: k* = π/4 * √(N/M)

Features:
- Oracle construction for various marking patterns
- Amplitude amplification with diffusion operator
- Automatic optimal iteration calculation
- Integration with AdaptiveMPS for scalability
- Support for multiple marked items
- Flexible oracle types (function-based, bitmap, structured)

Author: ATLAS-Q Contributors
Date: November 2025
License: MIT
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from .adaptive_mps import AdaptiveMPS

# Optional visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class GroverConfig:
    """
    Configuration for Grover's algorithm

    Attributes:
        n_qubits: Number of qubits (search space size = 2^n_qubits)
        oracle_type: Type of oracle ('function', 'bitmap', 'structured')
        auto_iterations: Automatically calculate optimal iterations
        max_iterations: Maximum iterations (safety limit)
        chi_max: Maximum bond dimension for MPS
        device: 'cuda' or 'cpu'
        dtype: torch.complex64 or torch.complex128
        verbose: Print progress information
        measure_success_prob: Track success probability after each iteration
    """
    n_qubits: int
    oracle_type: str = 'function'
    auto_iterations: bool = True
    max_iterations: int = 1000
    chi_max: int = 256
    device: str = 'cuda'
    dtype: torch.dtype = torch.complex128
    verbose: bool = False
    measure_success_prob: bool = False


class OracleBase:
    """Base class for quantum oracles"""

    def __init__(self, n_qubits: int, device: str = 'cuda',
                 dtype: torch.dtype = torch.complex128):
        self.n_qubits = n_qubits
        self.device = device
        self.dtype = dtype
        self.n_marked = 0  # Number of marked items

    def mark(self, mps: AdaptiveMPS, state: int) -> AdaptiveMPS:
        """Mark a specific basis state by flipping its phase"""
        raise NotImplementedError

    def apply(self, mps: AdaptiveMPS) -> AdaptiveMPS:
        """
        Apply oracle to MPS (marks all target states)

        Returns:
            New MPS after oracle application
        """
        raise NotImplementedError

    def get_marked_count(self) -> int:
        """Return number of marked items"""
        return self.n_marked


class FunctionOracle(OracleBase):
    """
    Oracle based on a marking function

    Marks states where f(x) = True by applying phase flip.
    Implements O_f |x⟩ = (-1)^f(x) |x⟩

    Example:
        # Mark state |5⟩
        oracle = FunctionOracle(n_qubits=4, marking_fn=lambda x: x == 5)
    """

    def __init__(self, n_qubits: int, marking_fn: Callable[[int], bool],
                 device: str = 'cuda', dtype: torch.dtype = torch.complex128):
        super().__init__(n_qubits, device, dtype)
        self.marking_fn = marking_fn

        # Count marked items
        N = 2 ** n_qubits
        self.n_marked = sum(1 for x in range(N) if marking_fn(x))

        if self.n_marked == 0:
            warnings.warn("Oracle marks zero items - Grover will not work")

    def apply(self, mps: AdaptiveMPS) -> AdaptiveMPS:
        """
        Apply oracle by marking all states where marking_fn returns True

        Returns:
            New MPS after oracle application
        """
        N = 2 ** self.n_qubits
        result_mps = mps
        for x in range(N):
            if self.marking_fn(x):
                result_mps = self._mark_state(result_mps, x)
        return result_mps

    def _mark_state(self, mps: AdaptiveMPS, state: int) -> AdaptiveMPS:
        """
        Mark a single state by applying phase flip using MPO

        Builds an MPO that represents O = I - 2|state⟩⟨state|
        This is exact and works for any marked state.

        Returns:
            New MPS after oracle application
        """
        # Import MPO operations
        from .mpo_ops import MPO, apply_mpo_to_mps

        # Build oracle MPO for marking this specific state
        oracle_mpo = self._build_oracle_mpo(state)

        # Apply oracle to MPS - returns NEW MPS!
        return apply_mpo_to_mps(oracle_mpo, mps, chi_max=mps.chi_max_per_bond)

    def _build_oracle_mpo(self, state: int) -> 'MPO':
        """
        Build MPO that marks a specific state

        For state x with binary representation b_n...b_1:
        Oracle = I - 2|x⟩⟨x|

        This is implemented as an MPO with bond dimension 2 that applies
        -1 phase to state |x⟩ and +1 to all others.
        """
        from .mpo_ops import MPO

        # Get binary representation (MSB first to match MPS convention)
        # MPS uses: bits[0] = MSB (leftmost), bits[n-1] = LSB (rightmost)
        bits = [(state >> (self.n_qubits - 1 - i)) & 1 for i in range(self.n_qubits)]

        tensors = []

        for i in range(self.n_qubits):
            if i == 0:
                # First site: bond dim [1, 2]
                # Layout: [[I, P_i]] where P_i projects onto bit i
                W = torch.zeros(1, 2, 2, 2, dtype=self.dtype, device=self.device)

                # I channel (bond 0 -> 0)
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # P_i channel (bond 0 -> 1): project onto bit value
                if bits[i] == 0:
                    W[0, 0, 0, 1] = 1.0  # |0⟩⟨0|
                else:
                    W[0, 1, 1, 1] = 1.0  # |1⟩⟨1|

            elif i == self.n_qubits - 1:
                # Last site: bond dim [2, 1]
                # Layout: [[I], [P_i]]
                W = torch.zeros(2, 2, 2, 1, dtype=self.dtype, device=self.device)

                # I channel (bond 0 -> 0)
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # Complete projector (bond 1 -> 0): P_i - 2P_total
                # This gives I - 2|x⟩⟨x| when combined
                if bits[i] == 0:
                    W[1, 0, 0, 0] = -2.0  # |0⟩⟨0|
                else:
                    W[1, 1, 1, 0] = -2.0  # |1⟩⟨1|

            else:
                # Middle sites: bond dim [2, 2]
                # Layout: [[I, 0], [P_i, I]]
                W = torch.zeros(2, 2, 2, 2, dtype=self.dtype, device=self.device)

                # I channel (bond 0 -> 0)
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # Projector chain (bond 1 -> 1)
                if bits[i] == 0:
                    W[1, 0, 0, 1] = 1.0  # |0⟩⟨0|
                else:
                    W[1, 1, 1, 1] = 1.0  # |1⟩⟨1|

                # Pass through identity (bond 1 -> 0) - not used in this structure

            tensors.append(W)

        return MPO(tensors=tensors, n_sites=self.n_qubits)


class BitmapOracle(OracleBase):
    """
    Oracle based on explicit bitmap of marked states

    More efficient than FunctionOracle for small marked sets.

    Example:
        oracle = BitmapOracle(n_qubits=4, marked_states={3, 7, 11})
    """

    def __init__(self, n_qubits: int, marked_states: Set[int],
                 device: str = 'cuda', dtype: torch.dtype = torch.complex128):
        super().__init__(n_qubits, device, dtype)

        # Validate empty set
        if not marked_states:
            raise ValueError("marked_states must contain at least one state")

        self.marked_states = marked_states
        self.n_marked = len(marked_states)

        # Validate states are in range
        N = 2 ** n_qubits
        for state in marked_states:
            if state < 0 or state >= N:
                raise ValueError(f"State {state} out of range [0, {N-1}]")

    def apply(self, mps: AdaptiveMPS) -> AdaptiveMPS:
        """
        Apply oracle by marking all states in bitmap

        Returns:
            New MPS after oracle application
        """
        # Reuse FunctionOracle's marking strategy
        oracle = FunctionOracle(
            self.n_qubits,
            marking_fn=lambda x: x in self.marked_states,
            device=self.device,
            dtype=self.dtype
        )
        return oracle.apply(mps)


class DiffusionOperator:
    """
    Grover diffusion operator (inversion about average)

    Implements: D = 2|ψ⟩⟨ψ| - I where |ψ⟩ = |+⟩^⊗n

    Mathematical form:
        D = H^⊗n (2|0⟩⟨0| - I) H^⊗n

    This reflects amplitudes about their average, amplifying marked states.
    """

    def __init__(self, n_qubits: int, device: str = 'cuda',
                 dtype: torch.dtype = torch.complex128):
        self.n_qubits = n_qubits
        self.device = device
        self.dtype = dtype

    def apply(self, mps: AdaptiveMPS) -> AdaptiveMPS:
        """
        Apply diffusion operator: 2|+⟩⟨+| - I

        Decomposition:
        1. H^⊗n  (transform to computational basis)
        2. 2|0⟩⟨0| - I  (conditional phase shift using MPO)
        3. H^⊗n  (transform back)

        Returns:
            New MPS after diffusion
        """
        from .mpo_ops import MPO, apply_mpo_to_mps

        # Step 1: Apply Hadamard to all qubits
        H = self._hadamard_gate()
        for i in range(self.n_qubits):
            mps.apply_single_qubit_gate(i, H)

        # Step 2: Apply 2|0⟩⟨0| - I using MPO - returns NEW MPS!
        # This is -(I - 2|0⟩⟨0|), so we mark state 0 and add global phase
        zero_oracle_mpo = self._build_zero_oracle_mpo()
        mps = apply_mpo_to_mps(zero_oracle_mpo, mps, chi_max=mps.chi_max_per_bond)

        # Global phase -1 (absorbed into first tensor)
        mps.tensors[0] = -1 * mps.tensors[0]

        # Step 3: Apply Hadamard to all qubits again
        for i in range(self.n_qubits):
            mps.apply_single_qubit_gate(i, H)

        return mps

    def _hadamard_gate(self) -> torch.Tensor:
        """Hadamard gate"""
        return torch.tensor(
            [[1, 1], [1, -1]], dtype=self.dtype, device=self.device
        ) / math.sqrt(2)

    def _build_zero_oracle_mpo(self) -> 'MPO':
        """
        Build MPO that applies I - 2|0⟩⟨0|

        This marks the all-zero state |00...0⟩.
        For the diffusion operator, we then add a global -1 phase to get 2|0⟩⟨0| - I.
        """
        from .mpo_ops import MPO

        tensors = []

        for i in range(self.n_qubits):
            if i == 0:
                # First site: bond dim [1, 2]
                W = torch.zeros(1, 2, 2, 2, dtype=self.dtype, device=self.device)

                # I channel
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # |0⟩⟨0| channel
                W[0, 0, 0, 1] = 1.0

            elif i == self.n_qubits - 1:
                # Last site: bond dim [2, 1]
                W = torch.zeros(2, 2, 2, 1, dtype=self.dtype, device=self.device)

                # I channel
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # Complete projector: -2|0⟩⟨0|
                W[1, 0, 0, 0] = -2.0

            else:
                # Middle sites: bond dim [2, 2]
                W = torch.zeros(2, 2, 2, 2, dtype=self.dtype, device=self.device)

                # I channel
                W[0, 0, 0, 0] = 1.0
                W[0, 1, 1, 0] = 1.0

                # |0⟩⟨0| chain
                W[1, 0, 0, 1] = 1.0

            tensors.append(W)

        return MPO(tensors=tensors, n_sites=self.n_qubits)


class GroverSearch:
    """
    Grover's Search Algorithm Implementation

    Performs quantum search on an unstructured database with O(√N) queries.

    Algorithm:
        1. Initialize to |+⟩^⊗n (uniform superposition)
        2. Repeat k times:
            a. Apply oracle O (mark target states)
            b. Apply diffusion operator D (amplify marked states)
        3. Measure to find marked item with high probability

    Example:
        >>> config = GroverConfig(n_qubits=4)
        >>> oracle = FunctionOracle(4, lambda x: x == 7)
        >>> grover = GroverSearch(oracle, config)
        >>> result = grover.run()
        >>> print(f"Found item: {result['measured_state']}")
        Found item: 7
    """

    def __init__(self, oracle: OracleBase, config: GroverConfig):
        self.oracle = oracle
        self.config = config
        self.n_qubits = config.n_qubits

        # Validate oracle matches config
        if oracle.n_qubits != config.n_qubits:
            raise ValueError(
                f"Oracle has {oracle.n_qubits} qubits but config specifies {config.n_qubits}"
            )

        # Initialize components
        self.diffusion = DiffusionOperator(
            config.n_qubits, config.device, config.dtype
        )

        # Statistics tracking
        self.iteration_stats = []
        self.success_probabilities = []

    def _initialize_state(self) -> AdaptiveMPS:
        """Initialize to uniform superposition |+⟩^⊗n"""
        mps = AdaptiveMPS(
            num_qubits=self.n_qubits,
            bond_dim=2,
            chi_max_per_bond=self.config.chi_max,
            device=self.config.device,
            dtype=self.config.dtype
        )

        # Apply Hadamard to all qubits to create |+⟩^⊗n
        H = torch.tensor(
            [[1, 1], [1, -1]], dtype=self.config.dtype, device=self.config.device
        ) / math.sqrt(2)

        for i in range(self.n_qubits):
            mps.apply_single_qubit_gate(i, H)

        return mps

    def _calculate_optimal_iterations(self) -> int:
        """
        Calculate optimal number of Grover iterations

        Formula: k* = floor(π/4 * √(N/M))
        where N = 2^n is search space size, M is number of marked items
        """
        N = 2 ** self.n_qubits
        M = self.oracle.get_marked_count()

        if M == 0:
            warnings.warn("No marked items - cannot run Grover")
            return 0

        # Optimal iterations
        k_opt = int(math.floor(math.pi / 4 * math.sqrt(N / M)))

        # Cap at max_iterations
        k = min(k_opt, self.config.max_iterations)

        if self.config.verbose:
            print(f"Search space: N = {N}")
            print(f"Marked items: M = {M}")
            print(f"Optimal iterations: {k_opt}")
            print(f"Using iterations: {k}")

        return k

    def _measure_success_probability(self, mps: AdaptiveMPS) -> float:
        """
        Measure probability of finding a marked item

        Computes: P_success = Σ_{x: f(x)=1} |⟨x|ψ⟩|²
        """
        # For marked states, compute probability
        total_prob = 0.0

        # If oracle is FunctionOracle or BitmapOracle, we know marked states
        if isinstance(self.oracle, BitmapOracle):
            marked_states = self.oracle.marked_states
        elif isinstance(self.oracle, FunctionOracle):
            N = 2 ** self.n_qubits
            marked_states = {x for x in range(N) if self.oracle.marking_fn(x)}
        else:
            # Unknown oracle type
            return -1.0

        # Sum probabilities
        for state in marked_states:
            prob = mps.get_probability(state)
            total_prob += prob

        return total_prob

    def run(self, iterations: Optional[int] = None) -> dict:
        """
        Run Grover's algorithm

        Args:
            iterations: Number of iterations (None = auto-calculate optimal)

        Returns:
            Dictionary with results:
                - measured_state: Most likely measurement outcome
                - success_probability: Probability of measuring marked item
                - iterations_used: Number of iterations performed
                - runtime_ms: Execution time in milliseconds
                - bond_dims: Final MPS bond dimensions
        """
        start_time = time.time()

        # Determine iterations
        if iterations is None and self.config.auto_iterations:
            iterations = self._calculate_optimal_iterations()
        elif iterations is None:
            iterations = 1  # Default to 1 iteration

        if iterations == 0:
            raise ValueError("Cannot run with 0 iterations")

        # Initialize state
        mps = self._initialize_state()

        # Grover iterations
        for k in range(iterations):
            iter_start = time.time()

            # Apply oracle - returns NEW MPS!
            mps = self.oracle.apply(mps)

            # Apply diffusion - returns NEW MPS!
            mps = self.diffusion.apply(mps)

            iter_time = (time.time() - iter_start) * 1000

            # Track statistics
            if self.config.measure_success_prob:
                success_prob = self._measure_success_probability(mps)
                self.success_probabilities.append(success_prob)
            else:
                success_prob = -1.0

            self.iteration_stats.append({
                'iteration': k + 1,
                'time_ms': iter_time,
                'success_probability': success_prob,
                'max_bond_dim': max(mps.bond_dims) if (hasattr(mps, 'bond_dims') and len(mps.bond_dims) > 0) else self.config.chi_max
            })

            if self.config.verbose:
                print(f"Iteration {k+1}/{iterations}: "
                      f"P_success = {success_prob:.4f}, "
                      f"time = {iter_time:.2f}ms")

        # Measure final state (find most likely outcome)
        measured_state = self._measure_most_likely(mps)
        final_success_prob = self._measure_success_probability(mps)

        runtime_ms = (time.time() - start_time) * 1000

        return {
            'measured_state': measured_state,
            'success_probability': final_success_prob,
            'iterations_used': iterations,
            'runtime_ms': runtime_ms,
            'bond_dims': mps.bond_dims if hasattr(mps, 'bond_dims') else ([self.config.chi_max] * max(0, self.n_qubits - 1)),
            'iteration_stats': self.iteration_stats
        }

    def _measure_most_likely(self, mps: AdaptiveMPS) -> int:
        """Find most likely measurement outcome"""
        N = 2 ** self.n_qubits

        # For small N, check all states
        if N <= 1024:
            max_prob = 0.0
            max_state = 0

            for state in range(N):
                prob = mps.get_probability(state)
                if prob > max_prob:
                    max_prob = prob
                    max_state = state

            return max_state
        else:
            # For large N, sample
            samples = mps.sweep_sample(num_shots=100)
            # Return most common sample
            from collections import Counter
            counts = Counter(samples)
            return counts.most_common(1)[0][0]

    def plot_convergence(self, save_path: Optional[str] = None):
        """
        Plot success probability vs iteration

        Args:
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available, cannot plot")
            return

        if not self.success_probabilities:
            warnings.warn("No success probability data to plot")
            return

        plt.figure(figsize=(10, 6))
        iterations = list(range(1, len(self.success_probabilities) + 1))
        plt.plot(iterations, self.success_probabilities, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Success Probability', fontsize=12)
        plt.title("Grover's Algorithm Convergence", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


# Convenience functions

def grover_search(
    n_qubits: int,
    marked_states: Union[Set[int], List[int], Callable[[int], bool]],
    iterations: Optional[int] = None,
    device: str = 'cuda',
    verbose: bool = False
) -> dict:
    """
    Convenience function to run Grover's search

    Args:
        n_qubits: Number of qubits
        marked_states: Either set/list of marked states or marking function
        iterations: Number of iterations (None = auto)
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        Results dictionary

    Example:
        >>> result = grover_search(4, marked_states={7}, verbose=True)
        >>> print(f"Found: {result['measured_state']}")
        Found: 7
    """
    config = GroverConfig(n_qubits=n_qubits, device=device, verbose=verbose,
                         auto_iterations=(iterations is None))

    # Create oracle
    if callable(marked_states):
        oracle = FunctionOracle(n_qubits, marked_states, device=device)
    else:
        if isinstance(marked_states, list):
            marked_states = set(marked_states)
        oracle = BitmapOracle(n_qubits, marked_states, device=device)

    # Run search
    grover = GroverSearch(oracle, config)
    return grover.run(iterations=iterations)


def calculate_grover_iterations(n_qubits: int, n_marked: int) -> int:
    """
    Calculate optimal number of Grover iterations

    Args:
        n_qubits: Number of qubits
        n_marked: Number of marked items

    Returns:
        Optimal iteration count
    """
    N = 2 ** n_qubits
    if n_marked == 0 or n_marked > N:
        raise ValueError(f"n_marked must be in range [1, {N}]")

    k_opt = int(math.floor(math.pi / 4 * math.sqrt(N / n_marked)))
    return k_opt
