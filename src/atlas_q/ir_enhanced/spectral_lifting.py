"""
IR Spectral Lifting Module
===========================

Implements the relational spectral lifting representation from Informational Relativity.

The key insight from IR is that global structure can be extracted through the
relational matrix M_ij = χ_i χ_j cos(θ_i - θ_j), where dominant eigenmodes
encode coherent structure invisible to pointwise statistics.

Key Functions:
- compute_relational_matrix(): Build M_ij spectral lifting matrix
- extract_structure_modes(): Extract dominant eigenmodes for structure identification
- spectral_grouping(): Group measurements by spectral coherence structure

References:
- IR Paper Section 5: "The Representation Layer (R)"
- IR Paper Equation 7: M_ij = χ_i χ_j cos(θ_i - θ_j)
- IR Paper Figure 4: Spectral Representation

Author: ATLAS-Q Development Team
Date: December 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpectralLiftingResult:
    """
    Result of spectral lifting analysis.

    Attributes:
        M: Relational spectral lifting matrix M_ij
        eigenvalues: Eigenvalues sorted in descending order
        eigenvectors: Corresponding eigenvectors (columns)
        spectral_coherence: Coherence from dominant eigenmode concentration
        structure_indices: Indices of structurally coherent elements
        spectral_gap: Gap ratio between first and second eigenvalues
    """
    M: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_coherence: float
    structure_indices: np.ndarray
    spectral_gap: float


def compute_relational_matrix(
    responses: np.ndarray,
    phases: np.ndarray
) -> np.ndarray:
    """
    Compute IR relational spectral lifting matrix.

    M_ij = χ_i * χ_j * cos(θ_i - θ_j)

    This matrix encodes pairwise phase relationships weighted by response
    magnitudes. The dominant eigenvector localizes at structurally coherent
    regions, enabling identification of hidden global structure.

    Args:
        responses: Response field magnitudes χ_i (n-dimensional array)
        phases: Response field phases θ_i (n-dimensional array, radians)

    Returns:
        M: n×n relational spectral lifting matrix

    Example:
        >>> # Coherent system: all phases aligned
        >>> responses = np.array([1.0, 0.9, 0.8, 0.7])
        >>> phases = np.array([0.0, 0.1, 0.05, 0.02])  # Nearly aligned
        >>> M = compute_relational_matrix(responses, phases)
        >>> # M will have high values (positive correlation)

    References:
        - IR Paper Equation 7
    """
    n = len(responses)
    if len(phases) != n:
        raise ValueError(f"responses and phases must have same length, got {n} and {len(phases)}")

    # Vectorized computation of M_ij = χ_i * χ_j * cos(θ_i - θ_j)
    # Phase difference matrix
    phase_diff = np.subtract.outer(phases, phases)

    # Response product matrix
    response_product = np.outer(responses, responses)

    # Full relational matrix
    M = response_product * np.cos(phase_diff)

    return M


def compute_relational_matrix_from_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """
    Compute relational matrix directly from complex amplitudes.

    Convenience function that extracts magnitudes and phases from complex
    amplitudes and builds the relational matrix.

    Args:
        amplitudes: Complex amplitude array

    Returns:
        M: Relational spectral lifting matrix
    """
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    responses = np.abs(amplitudes)
    phases = np.angle(amplitudes)
    return compute_relational_matrix(responses, phases)


def extract_structure_modes(
    M: np.ndarray,
    top_k: int = 5,
    localization_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dominant eigenmodes encoding global structure.

    The dominant eigenvector of M localizes at structurally coherent regions,
    enabling identification of hidden structure invisible to pointwise statistics.

    Args:
        M: Relational spectral lifting matrix (n×n)
        top_k: Number of dominant modes to extract
        localization_threshold: Threshold for structure localization (|v_i|² > threshold)

    Returns:
        eigenvalues: Top-k eigenvalues (descending order)
        eigenvectors: Top-k eigenvectors (columns)
        structure_indices: Indices where dominant eigenvector localizes

    Example:
        >>> M = compute_relational_matrix(responses, phases)
        >>> eigenvalues, eigenvectors, structure_idx = extract_structure_modes(M, top_k=3)
        >>> print(f"Dominant eigenvalue: {eigenvalues[0]:.4f}")
        >>> print(f"Structure localized at indices: {structure_idx}")
    """
    # Ensure M is symmetric (should be by construction)
    M = (M + M.T) / 2

    # Full eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Sort by descending eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract top-k modes
    top_k = min(top_k, len(eigenvalues))
    top_eigenvalues = eigenvalues[:top_k]
    top_eigenvectors = eigenvectors[:, :top_k]

    # Find structure localization from dominant eigenvector
    dominant_v = top_eigenvectors[:, 0]
    v_squared = np.abs(dominant_v) ** 2
    structure_indices = np.where(v_squared > localization_threshold * np.max(v_squared))[0]

    return top_eigenvalues, top_eigenvectors, structure_indices


def spectral_lifting_analysis(
    responses: np.ndarray,
    phases: np.ndarray,
    top_k: int = 5,
    localization_threshold: float = 0.1
) -> SpectralLiftingResult:
    """
    Complete spectral lifting analysis for coherence structure identification.

    Combines matrix construction, eigendecomposition, and structure identification
    into a single analysis pipeline.

    Args:
        responses: Response field magnitudes χ_i
        phases: Response field phases θ_i (radians)
        top_k: Number of dominant modes to extract
        localization_threshold: Threshold for structure localization

    Returns:
        SpectralLiftingResult with full analysis
    """
    # Build relational matrix
    M = compute_relational_matrix(responses, phases)

    # Extract structure modes
    eigenvalues, eigenvectors, structure_indices = extract_structure_modes(
        M, top_k=top_k, localization_threshold=localization_threshold
    )

    # Compute spectral coherence (concentration in dominant mode)
    total_power = np.sum(np.abs(eigenvalues))
    if total_power > 1e-15:
        spectral_coherence = float(np.abs(eigenvalues[0]) / total_power)
    else:
        spectral_coherence = 0.0

    # Compute spectral gap
    if len(eigenvalues) > 1 and np.abs(eigenvalues[1]) > 1e-15:
        spectral_gap = float(np.abs(eigenvalues[0]) / np.abs(eigenvalues[1]))
    else:
        spectral_gap = np.inf

    return SpectralLiftingResult(
        M=M,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        spectral_coherence=spectral_coherence,
        structure_indices=structure_indices,
        spectral_gap=spectral_gap
    )


def spectral_lifting_from_amplitudes(
    amplitudes: np.ndarray,
    top_k: int = 5,
    localization_threshold: float = 0.1
) -> SpectralLiftingResult:
    """
    Spectral lifting analysis directly from complex amplitudes.

    Convenience function for quantum states represented as amplitude vectors.

    Args:
        amplitudes: Complex quantum state amplitudes
        top_k: Number of dominant modes to extract
        localization_threshold: Threshold for structure localization

    Returns:
        SpectralLiftingResult with full analysis
    """
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    responses = np.abs(amplitudes)
    phases = np.angle(amplitudes)
    return spectral_lifting_analysis(responses, phases, top_k, localization_threshold)


def spectral_grouping(
    items: List,
    responses: np.ndarray,
    phases: np.ndarray,
    n_groups: int = 5,
    min_group_size: int = 1
) -> List[List[int]]:
    """
    Group items by spectral coherence structure.

    Uses the dominant eigenvectors of the relational matrix to identify
    coherent clusters for optimal grouping.

    Algorithm:
        1. Compute relational matrix M
        2. Extract dominant eigenvectors
        3. Cluster items by eigenvector components (k-means in eigenvector space)

    Args:
        items: List of items to group (e.g., Pauli strings, parameters)
        responses: Response field magnitudes for each item
        phases: Response field phases for each item
        n_groups: Target number of groups
        min_group_size: Minimum items per group

    Returns:
        List of groups, where each group is a list of indices into items
    """
    n = len(items)
    if n == 0:
        return []

    n_groups = min(n_groups, n)

    if n <= n_groups:
        # Each item in its own group
        return [[i] for i in range(n)]

    # Spectral analysis
    result = spectral_lifting_analysis(responses, phases, top_k=min(n_groups, n))

    # Use top eigenvectors for clustering
    # Project items onto eigenvector space
    n_components = min(n_groups, len(result.eigenvalues))
    features = result.eigenvectors[:, :n_components]

    # Simple k-means-style clustering in eigenvector space
    groups = _kmeans_cluster(features, n_groups, min_group_size)

    return groups


def _kmeans_cluster(
    features: np.ndarray,
    n_clusters: int,
    min_size: int
) -> List[List[int]]:
    """
    Simple k-means clustering for spectral grouping.

    Args:
        features: n×d feature matrix (eigenvector projections)
        n_clusters: Number of clusters
        min_size: Minimum cluster size

    Returns:
        List of clusters (lists of indices)
    """
    n = len(features)
    if n <= n_clusters:
        return [[i] for i in range(n)]

    # Initialize centroids using k-means++ style
    centroids = [features[0]]
    for _ in range(1, n_clusters):
        # Distance to nearest centroid
        dists = np.array([
            min(np.linalg.norm(f - c) for c in centroids)
            for f in features
        ])
        # Sample proportional to squared distance
        probs = dists ** 2
        if probs.sum() > 0:
            probs /= probs.sum()
            idx = np.random.choice(n, p=probs)
        else:
            idx = np.random.randint(n)
        centroids.append(features[idx])

    centroids = np.array(centroids)

    # Iterate k-means
    for _ in range(20):  # Max iterations
        # Assign to nearest centroid
        assignments = np.array([
            np.argmin([np.linalg.norm(f - c) for c in centroids])
            for f in features
        ])

        # Update centroids
        new_centroids = []
        for k in range(n_clusters):
            cluster_points = features[assignments == k]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Build groups from assignments
    groups = [[] for _ in range(n_clusters)]
    for i, k in enumerate(assignments):
        groups[k].append(i)

    # Remove empty groups
    groups = [g for g in groups if len(g) >= min_size]

    # Handle items in groups below min_size
    if not groups:
        # Fallback: single group
        return [list(range(n))]

    return groups


def coherent_structure_score(
    responses: np.ndarray,
    phases: np.ndarray,
    e2_threshold: float = 0.135
) -> Tuple[float, bool, str]:
    """
    Compute a coherent structure score and regime classification.

    Uses spectral lifting to determine:
    - IR regime (R̄ > e^-2): Structure directly observable
    - AIR regime (R̄ < e^-2): Structure locally detectable but globally hidden

    Args:
        responses: Response field magnitudes
        phases: Response field phases
        e2_threshold: e^-2 threshold (default: 0.135)

    Returns:
        score: Spectral coherence score (0-1)
        is_ir_regime: True if in IR regime (structure observable)
        regime_description: Human-readable regime description
    """
    result = spectral_lifting_analysis(responses, phases)

    score = result.spectral_coherence
    is_ir = score > e2_threshold

    if is_ir:
        description = f"IR regime (R̄={score:.3f} > {e2_threshold}): Structure directly observable"
    else:
        description = f"AIR regime (R̄={score:.3f} < {e2_threshold}): Structure globally hidden"

    return score, is_ir, description
