"""
Coherence Utilities
===================

Utility functions for coherence-aware quantum computing.

Author: ATLAS-Q Development Team
Date: November 2025
"""

from typing import Dict

import numpy as np


def compute_pauli_expectation(counts: Dict[str, int], pauli_str: str) -> float:
    """
    Compute Pauli expectation value ⟨P⟩ from measurement counts.

    For a Pauli operator P, the expectation value is computed as:
        ⟨P⟩ = (n_even - n_odd) / n_total

    where n_even is the count of bitstrings with even parity and n_odd
    is the count of bitstrings with odd parity.

    The parity is determined by the XOR of all qubits where P acts
    non-trivially (i.e., where P is X, Y, or Z, not I).

    Args:
        counts: Dictionary mapping bitstrings to measurement counts
                (e.g., {'00': 800, '11': 200})
        pauli_str: Pauli string (e.g., 'IXZY', 'ZZII')

    Returns:
        Expectation value ⟨P⟩ ∈ [-1, 1]

    Raises:
        ValueError: If counts is empty or pauli_str has mismatched length

    Example:
        >>> counts = {'00': 900, '11': 100}
        >>> exp_val = compute_pauli_expectation(counts, 'ZZ')
        >>> print(f"⟨ZZ⟩ = {exp_val:.4f}")
        ⟨ZZ⟩ = 0.8000

    Notes:
        - Bitstrings in counts are assumed to be in Qiskit's ordering (q[n-1]...q[0])
        - Identity operators (I) do not contribute to parity
        - X, Y, Z all contribute identically to parity (XOR of qubit value)
    """
    if not counts:
        raise ValueError("counts dictionary cannot be empty")

    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("total shots cannot be zero")

    expectation = 0.0

    for bitstring, count in counts.items():
        # Validate bitstring length
        if len(bitstring) != len(pauli_str):
            raise ValueError(
                f"Bitstring length ({len(bitstring)}) does not match "
                f"Pauli string length ({len(pauli_str)})"
            )

        # Compute parity: XOR of all qubits where Pauli acts non-trivially
        parity = 0
        for i, pauli in enumerate(pauli_str):
            if pauli != 'I':
                # XOR with bit at position i
                parity ^= int(bitstring[i])

        # +1 for even parity (parity=0), -1 for odd parity (parity=1)
        sign = 1 if parity == 0 else -1
        expectation += sign * count / total_shots

    return expectation


def pauli_commute(pauli1: str, pauli2: str) -> bool:
    """
    Check if two Pauli strings commute.

    Two Pauli operators commute if they have an even number of positions
    where they both act non-trivially with different operators.

    Args:
        pauli1: First Pauli string (e.g., 'IXZY')
        pauli2: Second Pauli string (e.g., 'ZIIX')

    Returns:
        True if Pauli strings commute, False otherwise

    Raises:
        ValueError: If Pauli strings have different lengths

    Example:
        >>> pauli_commute('IXZY', 'IZXY')
        True
        >>> pauli_commute('XX', 'YY')
        False

    Notes:
        - Identity (I) always commutes with everything
        - Same Pauli operators (XX, YY, ZZ) commute
        - Different non-identity Paulis (XY, YZ, XZ) anticommute
    """
    if len(pauli1) != len(pauli2):
        raise ValueError(
            f"Pauli strings must have same length, got {len(pauli1)} and {len(pauli2)}"
        )

    # Count positions where both act non-trivially with different operators
    anticommute_count = 0
    for p1, p2 in zip(pauli1, pauli2):
        if p1 != 'I' and p2 != 'I' and p1 != p2:
            anticommute_count += 1

    # Commute if even number of anticommuting positions
    return anticommute_count % 2 == 0


def qubit_wise_commute(pauli1: str, pauli2: str) -> bool:
    """
    Check if two Pauli strings are qubit-wise commuting (QWC).

    Two Paulis are QWC if at each qubit position, they either:
    - Both act with I
    - Both act with Z
    - Both act with X or Y

    This is a stronger condition than general commutativity and is
    used for measurement grouping.

    Args:
        pauli1: First Pauli string
        pauli2: Second Pauli string

    Returns:
        True if Pauli strings are qubit-wise commuting

    Raises:
        ValueError: If Pauli strings have different lengths

    Example:
        >>> qubit_wise_commute('IXZY', 'IZZY')
        True
        >>> qubit_wise_commute('IXZY', 'IXYX')
        True
        >>> qubit_wise_commute('IXZ', 'IZX')
        False

    Notes:
        - QWC grouping enables simultaneous measurement in the same basis
        - (I, Z) form one measurement family (Z-basis)
        - (X, Y) form another measurement family (rotated basis)
    """
    if len(pauli1) != len(pauli2):
        raise ValueError(
            f"Pauli strings must have same length, got {len(pauli1)} and {len(pauli2)}"
        )

    # Define measurement families
    z_family = {'I', 'Z'}
    xy_family = {'X', 'Y'}

    for p1, p2 in zip(pauli1, pauli2):
        # Check if both in same family
        both_z = (p1 in z_family) and (p2 in z_family)
        both_xy = (p1 in xy_family) and (p2 in xy_family)

        if not (both_z or both_xy):
            return False

    return True


def group_paulis_qwc(pauli_strings: list, coefficients: np.ndarray = None) -> list:
    """
    Group Pauli strings using qubit-wise commuting (QWC) strategy.

    This is a greedy algorithm that groups Paulis that can be measured
    simultaneously in the same basis.

    Args:
        pauli_strings: List of Pauli strings
        coefficients: Optional weights for prioritizing large-coefficient terms

    Returns:
        List of groups, where each group is a list of indices into pauli_strings

    Example:
        >>> paulis = ['IIZZ', 'IZIZ', 'XXII', 'XYII']
        >>> groups = group_paulis_qwc(paulis)
        >>> print(f"Groups: {groups}")
        Groups: [[0, 1], [2, 3]]

    Notes:
        - First group contains Z-basis measurements
        - Subsequent groups are formed greedily
        - Large-coefficient terms are prioritized if coefficients provided
    """
    if not pauli_strings:
        return []

    n_paulis = len(pauli_strings)

    # Sort by coefficient magnitude if provided
    if coefficients is not None:
        if len(coefficients) != n_paulis:
            raise ValueError("coefficients must have same length as pauli_strings")
        # Sort indices by descending coefficient magnitude
        sorted_indices = np.argsort(-np.abs(coefficients))
    else:
        sorted_indices = list(range(n_paulis))

    groups = []
    remaining = set(sorted_indices)

    while remaining:
        # Start new group with first remaining Pauli
        seed_idx = min(remaining)  # Take first in remaining set
        new_group = [seed_idx]
        remaining.remove(seed_idx)
        seed_pauli = pauli_strings[seed_idx]

        # Try to add compatible Paulis to this group
        to_remove = []
        for idx in remaining:
            if qubit_wise_commute(seed_pauli, pauli_strings[idx]):
                # Check if compatible with all Paulis in group
                compatible = all(
                    qubit_wise_commute(pauli_strings[idx], pauli_strings[group_idx])
                    for group_idx in new_group
                )
                if compatible:
                    new_group.append(idx)
                    to_remove.append(idx)

        for idx in to_remove:
            remaining.remove(idx)

        groups.append(new_group)

    return groups
