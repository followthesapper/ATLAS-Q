"""
VRA-Enhanced QAOA Measurement Grouping
=======================================

Applies VRA coherence-based grouping to QAOA cost Hamiltonians for
variance reduction in combinatorial optimization.

Key Insight:
-----------
MaxCut Hamiltonians consist of ZiZj terms which often commute:
- Z_i Z_j and Z_k Z_l commute if {i,j} ∩ {k,l} = ∅ (no shared qubits)
- Can group non-overlapping edges for simultaneous measurement
- VRA coherence analysis optimizes grouping and shot allocation

Target: 10-500× variance reduction for medium-to-large graphs

Author: ATLAS-Q + VRA Integration
Date: November 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class QAOAGroupingResult:
    """Result of QAOA Hamiltonian grouping"""

    groups: List[List[int]]  # Groups of edge indices
    shots_per_group: np.ndarray  # Optimal shot allocation
    variance_reduction: float  # vs baseline (per-edge measurement)
    method: str  # Grouping method used
    n_edges: int  # Total number of edges
    n_groups: int  # Number of measurement groups


def edges_commute(edge1: Tuple[int, int], edge2: Tuple[int, int]) -> bool:
    """
    Check if two ZiZj terms commute.

    Two ZiZj terms commute if they don't share any qubits.

    Parameters
    ----------
    edge1 : Tuple[int, int]
        First edge (i, j)
    edge2 : Tuple[int, int]
        Second edge (k, l)

    Returns
    -------
    bool
        True if edges can be measured simultaneously

    Examples
    --------
    >>> edges_commute((0, 1), (2, 3))  # No shared qubits
    True
    >>> edges_commute((0, 1), (1, 2))  # Share qubit 1
    False
    >>> edges_commute((0, 1), (0, 2))  # Share qubit 0
    False
    """
    i1, j1 = edge1
    i2, j2 = edge2

    # Check if sets {i1, j1} and {i2, j2} are disjoint
    return len({i1, j1} & {i2, j2}) == 0


def check_group_commutativity_edges(
    group: List[int],
    edges: List[Tuple[int, int]]
) -> bool:
    """
    Check if all edges in a group mutually commute.

    Parameters
    ----------
    group : List[int]
        Indices of edges to check
    edges : List[Tuple[int, int]]
        List of all edges

    Returns
    -------
    bool
        True if all pairs in group commute
    """
    for i, idx1 in enumerate(group):
        for idx2 in group[i+1:]:
            if not edges_commute(edges[idx1], edges[idx2]):
                return False
    return True


def estimate_edge_coherence_matrix(
    weights: np.ndarray,
    edges: List[Tuple[int, int]],
    method: str = "exponential"
) -> np.ndarray:
    """
    Estimate coherence matrix for graph edges.

    Coherence captures correlation between edge measurements based on:
    - Shared vertices (higher correlation if edges share qubits)
    - Weight similarity (similar weights → similar variance)

    Parameters
    ----------
    weights : np.ndarray
        Edge weights (MaxCut coefficients)
    edges : List[Tuple[int, int]]
        Graph edges
    method : str
        "exponential" (weight-based) or "geometric" (topology-based)

    Returns
    -------
    Sigma : np.ndarray, shape (n_edges, n_edges)
        Coherence/correlation matrix
    """
    n_edges = len(edges)
    Sigma = np.eye(n_edges)

    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            edge_i = edges[i]
            edge_j = edges[j]

            # Geometric correlation: shared vertices
            shared_vertices = len(set(edge_i) & set(edge_j))

            if method == "exponential":
                # Exponential decay based on weight difference and topology
                weight_diff = abs(weights[i] - weights[j])
                weight_avg = (abs(weights[i]) + abs(weights[j])) / 2

                if weight_avg > 0:
                    normalized_diff = weight_diff / weight_avg
                else:
                    normalized_diff = 0

                # Combine topology and weight similarity
                if shared_vertices > 0:
                    # Shared vertices → higher correlation
                    coherence = np.exp(-normalized_diff) * (1 + shared_vertices * 0.5)
                else:
                    # No shared vertices → lower correlation
                    coherence = np.exp(-normalized_diff) * 0.1

            elif method == "geometric":
                # Topology-only correlation
                if shared_vertices > 0:
                    coherence = 1.0 / (1.0 + normalized_diff if 'normalized_diff' in locals() else 1.0)
                else:
                    coherence = 0.1

            # Ensure coherence in [0, 1]
            coherence = max(0.0, min(1.0, coherence))

            Sigma[i, j] = Sigma[j, i] = coherence

    # Ensure positive definiteness
    eigenvalues = np.linalg.eigvalsh(Sigma)
    if eigenvalues.min() < 1e-10:
        Sigma += np.eye(n_edges) * (1e-10 - eigenvalues.min())

    return Sigma


def group_edges_by_commutativity(
    Sigma: np.ndarray,
    weights: np.ndarray,
    edges: List[Tuple[int, int]],
    max_group_size: int = 10
) -> List[List[int]]:
    """
    Group edges by commutativity constraint with variance minimization.

    Greedy algorithm:
    1. Start with highest-weight edge
    2. Add commuting edges that minimize Q_GLS increase
    3. Repeat until all edges grouped

    Parameters
    ----------
    Sigma : np.ndarray
        Coherence matrix
    weights : np.ndarray
        Edge weights
    max_group_size : int
        Maximum edges per group
    edges : List[Tuple[int, int]]
        Graph edges

    Returns
    -------
    groups : List[List[int]]
        Edge groupings
    """
    from .vqe_grouping import compute_Q_GLS

    n_edges = len(edges)
    remaining = set(range(n_edges))
    groups = []

    # Sort edges by weight magnitude (prioritize high-weight edges)
    sorted_indices = np.argsort(-np.abs(weights))

    while remaining:
        # Start new group with highest remaining weight edge
        start_idx = None
        for idx in sorted_indices:
            if idx in remaining:
                start_idx = idx
                break

        if start_idx is None:
            start_idx = min(remaining)

        group = [start_idx]
        remaining.remove(start_idx)

        # Greedy: add edges that commute and minimize Q_GLS
        while len(group) < max_group_size and remaining:
            best_idx = None
            best_Q = float('inf')

            for candidate in list(remaining):
                # Check commutativity with all edges in group
                test_group = group + [candidate]
                if not check_group_commutativity_edges(test_group, edges):
                    continue  # Skip non-commuting edges

                # Compute Q_GLS for test group
                c_test = weights[test_group]
                Sigma_test = Sigma[np.ix_(test_group, test_group)]
                Q_test = compute_Q_GLS(Sigma_test, c_test)

                if Q_test < best_Q:
                    best_Q = Q_test
                    best_idx = candidate

            if best_idx is not None:
                group.append(best_idx)
                remaining.remove(best_idx)
            else:
                break  # No more commuting edges found

        groups.append(sorted(group))

    return groups


def allocate_shots_neyman_edges(
    Sigma: np.ndarray,
    weights: np.ndarray,
    groups: List[List[int]],
    total_shots: int
) -> np.ndarray:
    """
    Optimal shot allocation via Neyman allocation for edge groups.

    Allocates more shots to groups with higher variance.

    Parameters
    ----------
    Sigma : np.ndarray
        Coherence matrix
    weights : np.ndarray
        Edge weights
    groups : List[List[int]]
        Edge groupings
    total_shots : int
        Total measurement budget

    Returns
    -------
    shots_per_group : np.ndarray
        Optimal shot allocation
    """
    from .vqe_grouping import allocate_shots_neyman, compute_Q_GLS

    return allocate_shots_neyman(Sigma, weights, groups, total_shots)


def compute_variance_reduction_qaoa(
    Sigma: np.ndarray,
    weights: np.ndarray,
    groups: List[List[int]],
    total_shots: int
) -> float:
    """
    Compute variance reduction factor vs baseline (per-edge measurement).

    Parameters
    ----------
    Sigma : np.ndarray
        Coherence matrix
    weights : np.ndarray
        Edge weights
    groups : List[List[int]]
        Edge groupings
    total_shots : int
        Total measurement budget

    Returns
    -------
    float
        Variance reduction factor (baseline_var / grouped_var)
    """
    from .vqe_grouping import compute_variance_reduction

    return compute_variance_reduction(Sigma, weights, groups, total_shots)


def vra_qaoa_grouping(
    weights: np.ndarray,
    edges: List[Tuple[int, int]],
    total_shots: int = 10000,
    max_group_size: int = 10,
    coherence_method: str = "exponential"
) -> QAOAGroupingResult:
    """
    VRA-enhanced grouping for QAOA MaxCut Hamiltonians.

    Automatically groups commuting edges and allocates shots optimally.

    Parameters
    ----------
    weights : np.ndarray
        Edge weights from MaxCut Hamiltonian
    edges : List[Tuple[int, int]]
        Graph edges (node pairs)
    total_shots : int
        Total measurement budget
    max_group_size : int
        Maximum edges per group
    coherence_method : str
        "exponential" or "geometric"

    Returns
    -------
    QAOAGroupingResult
        Grouping strategy with shot allocation and variance reduction

    Examples
    --------
    >>> # Triangle graph
    >>> weights = np.array([1.0, 1.0, 1.0])
    >>> edges = [(0, 1), (1, 2), (0, 2)]
    >>> result = vra_qaoa_grouping(weights, edges, total_shots=10000)
    >>> print(f"Groups: {result.groups}")
    >>> print(f"Variance reduction: {result.variance_reduction:.2f}×")
    """
    n_edges = len(edges)

    # Estimate coherence matrix
    Sigma = estimate_edge_coherence_matrix(weights, edges, method=coherence_method)

    # Group edges by commutativity
    groups = group_edges_by_commutativity(Sigma, weights, edges, max_group_size)

    # Allocate shots optimally
    shots_per_group = allocate_shots_neyman_edges(Sigma, weights, groups, total_shots)

    # Compute variance reduction
    variance_reduction = compute_variance_reduction_qaoa(Sigma, weights, groups, total_shots)

    return QAOAGroupingResult(
        groups=groups,
        shots_per_group=shots_per_group,
        variance_reduction=variance_reduction,
        method="vra_qaoa_commuting",
        n_edges=n_edges,
        n_groups=len(groups)
    )
