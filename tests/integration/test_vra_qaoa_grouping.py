"""
Test VRA-Enhanced QAOA Grouping
================================

Validates commutativity-aware edge grouping for QAOA MaxCut problems.

Target: 10-500× variance reduction for medium-to-large graphs
"""

import pytest
import numpy as np
from atlas_q.vra_enhanced import (
    vra_qaoa_grouping,
    edges_commute,
    check_group_commutativity_edges,
)


class TestEdgeCommutativity:
    """Test edge commutativity checking"""

    def test_disjoint_edges_commute(self):
        """Edges with no shared vertices commute"""
        assert edges_commute((0, 1), (2, 3))  # Disjoint
        assert edges_commute((0, 1), (3, 4))  # Disjoint
        assert edges_commute((5, 6), (7, 8))  # Disjoint
        print("✓ Disjoint edges commute")

    def test_shared_vertex_anticommute(self):
        """Edges sharing a vertex don't commute"""
        assert not edges_commute((0, 1), (1, 2))  # Share vertex 1
        assert not edges_commute((0, 1), (0, 2))  # Share vertex 0
        assert not edges_commute((2, 3), (3, 4))  # Share vertex 3
        print("✓ Edges with shared vertices anti-commute")

    def test_same_edge(self):
        """Same edge always commutes with itself"""
        assert edges_commute((0, 1), (0, 1))
        assert edges_commute((5, 6), (5, 6))
        print("✓ Same edge commutes with itself")


class TestGroupCommutativityEdges:
    """Test group-level commutativity for edges"""

    def test_all_disjoint_edges(self):
        """Group of disjoint edges all commute"""
        edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
        group = [0, 1, 2, 3]  # All edges

        assert check_group_commutativity_edges(group, edges)
        print("✓ All disjoint edges commute")

    def test_triangle_graph(self):
        """Triangle graph has no commuting pairs"""
        edges = [(0, 1), (1, 2), (0, 2)]

        # No two edges can be grouped
        assert not check_group_commutativity_edges([0, 1], edges)
        assert not check_group_commutativity_edges([1, 2], edges)
        assert not check_group_commutativity_edges([0, 2], edges)
        print("✓ Triangle edges don't commute pairwise")

    def test_square_graph(self):
        """Square graph has two commuting pairs"""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        # Opposite edges commute
        assert check_group_commutativity_edges([0, 2], edges)  # (0,1) and (2,3)
        assert check_group_commutativity_edges([1, 3], edges)  # (1,2) and (3,0)

        # Adjacent edges don't commute
        assert not check_group_commutativity_edges([0, 1], edges)
        assert not check_group_commutativity_edges([2, 3], edges)
        print("✓ Square graph commutativity validated")


class TestQAOAGrouping:
    """Test VRA QAOA grouping algorithm"""

    def test_triangle_graph_grouping(self):
        """Triangle: No edges commute → 3 groups"""
        weights = np.array([1.0, 1.0, 1.0])
        edges = [(0, 1), (1, 2), (0, 2)]

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\nTriangle Graph:")
        print(f"  Edges: {edges}")
        print(f"  Groups: {result.groups}")
        print(f"  Shots: {result.shots_per_group}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Should have 3 groups (one per edge)
        assert result.n_groups == 3
        assert result.n_edges == 3

        # All groups should commute
        for group in result.groups:
            assert check_group_commutativity_edges(group, edges)

        print("✓ Triangle graph grouping validated")

    def test_square_graph_grouping(self):
        """Square: Opposite edges commute → 2 groups"""
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\nSquare Graph:")
        print(f"  Edges: {edges}")
        print(f"  Groups: {result.groups}")
        print(f"  Shots: {result.shots_per_group}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Should have 2 groups (opposite edges)
        assert result.n_groups == 2
        assert result.n_edges == 4

        # All groups should commute
        for group in result.groups:
            assert check_group_commutativity_edges(group, edges)

        # Should show variance reduction vs baseline (4 → 2 groups)
        assert result.variance_reduction >= 1.0

        print("✓ Square graph grouping validated")

    def test_disjoint_edges_grouping(self):
        """Fully disjoint edges → 1 group"""
        weights = np.array([1.0, 1.0, 1.0])
        edges = [(0, 1), (2, 3), (4, 5)]

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\nDisjoint Edges:")
        print(f"  Edges: {edges}")
        print(f"  Groups: {result.groups}")
        print(f"  Shots: {result.shots_per_group}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Should group all edges together
        assert result.n_groups == 1
        assert len(result.groups[0]) == 3

        # Should show significant variance reduction
        assert result.variance_reduction > 1.0

        print("✓ Disjoint edges grouped efficiently")

    def test_petersen_graph_grouping(self):
        """Petersen graph (10 vertices, 15 edges) - realistic example"""
        # Petersen graph edges
        edges = [
            # Outer pentagon
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            # Inner pentagon
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
            # Connections
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        weights = np.ones(len(edges))

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\nPetersen Graph:")
        print(f"  Edges: {len(edges)}")
        print(f"  Groups: {result.n_groups}")
        print(f"  Largest group: {max(len(g) for g in result.groups)}")
        print(f"  Shots distribution: {result.shots_per_group}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Should have fewer groups than edges
        assert result.n_groups < len(edges)

        # All groups should commute
        for group in result.groups:
            assert check_group_commutativity_edges(group, edges)

        # Should show variance reduction
        assert result.variance_reduction >= 1.0

        print("✓ Petersen graph grouping validated")

    def test_weighted_edges(self):
        """Test with non-uniform edge weights"""
        weights = np.array([2.0, 1.0, 0.5, 3.0])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7)]  # All disjoint

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\nWeighted Disjoint Edges:")
        print(f"  Weights: {weights}")
        print(f"  Groups: {result.groups}")
        print(f"  Shots: {result.shots_per_group}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Should group all together (disjoint)
        assert result.n_groups == 1

        # Higher weight edges should get more shots in Neyman allocation
        # (but since grouped together, just one group gets all shots)
        assert np.sum(result.shots_per_group) == 10000

        print("✓ Weighted edges handled correctly")


class TestScaling:
    """Test scaling behavior with graph size"""

    def test_complete_graph_scaling(self):
        """Complete graphs K_n have n(n-1)/2 edges, all share vertices"""
        # K_5 complete graph
        n = 5
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        weights = np.ones(len(edges))

        result = vra_qaoa_grouping(weights, edges, total_shots=10000, max_group_size=5)

        print(f"\nComplete Graph K_{n}:")
        print(f"  Edges: {len(edges)}")
        print(f"  Groups: {result.n_groups}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Complete graphs have many shared vertices → more groups
        # Expect groups ≈ ceil(edges / max_independent_set_size)

        # All groups must commute
        for group in result.groups:
            assert check_group_commutativity_edges(group, edges)

        print("✓ Complete graph scaling validated")

    def test_grid_graph_scaling(self):
        """2D grid graph (many disjoint edges possible)"""
        # 3×3 grid (12 edges)
        edges = [
            # Horizontal
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (6, 7), (7, 8),
            # Vertical
            (0, 3), (3, 6),
            (1, 4), (4, 7),
            (2, 5), (5, 8),
        ]
        weights = np.ones(len(edges))

        result = vra_qaoa_grouping(weights, edges, total_shots=10000)

        print(f"\n3×3 Grid Graph:")
        print(f"  Edges: {len(edges)}")
        print(f"  Groups: {result.n_groups}")
        print(f"  Average group size: {len(edges) / result.n_groups:.1f}")
        print(f"  Variance reduction: {result.variance_reduction:.2f}×")

        # Grid graphs should allow good grouping (many disjoint edge sets)
        assert result.n_groups < len(edges)  # Should compress

        # All groups must commute
        for group in result.groups:
            assert check_group_commutativity_edges(group, edges)

        print("✓ Grid graph scaling validated")


def test_end_to_end_qaoa_grouping():
    """
    End-to-end test demonstrating variance reduction for QAOA.
    """
    print("\n" + "="*70)
    print("VRA QAOA Grouping - End-to-End Test")
    print("="*70)

    # Medium graph (20 vertices, various connectivity)
    np.random.seed(42)
    n_vertices = 20
    edge_probability = 0.3

    # Generate random graph
    edges = []
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            if np.random.rand() < edge_probability:
                edges.append((i, j))

    weights = np.random.uniform(0.5, 2.0, len(edges))

    print(f"\nRandom Graph:")
    print(f"  Vertices: {n_vertices}")
    print(f"  Edges: {len(edges)}")
    print(f"  Density: {len(edges) / (n_vertices * (n_vertices - 1) / 2):.2%}")

    # Baseline grouping (per-edge)
    baseline_groups = [[i] for i in range(len(edges))]
    print(f"\nBaseline (per-edge): {len(baseline_groups)} groups")

    # VRA grouping
    result = vra_qaoa_grouping(weights, edges, total_shots=10000, max_group_size=10)

    print(f"\nVRA Grouping:")
    print(f"  Groups: {result.n_groups}")
    print(f"  Compression: {len(edges)} → {result.n_groups} ({len(edges)/result.n_groups:.1f}× reduction)")
    print(f"  Largest group: {max(len(g) for g in result.groups)} edges")
    print(f"  Smallest group: {min(len(g) for g in result.groups)} edges")
    print(f"  Average group size: {len(edges) / result.n_groups:.1f}")

    # Verify commutativity
    for i, group in enumerate(result.groups):
        commutes = check_group_commutativity_edges(group, edges)
        if not commutes:
            print(f"  ❌ Group {i} violates commutativity!")
        assert commutes

    print(f"\n✅ All {result.n_groups} groups satisfy commutativity")
    print(f"✅ Variance reduction: {result.variance_reduction:.2f}×")
    print(f"✅ Shot allocation: {result.shots_per_group[:5]}... (first 5 groups)")

    # Expected compression for 30% density graph
    expected_compression = 2.0  # Conservative estimate
    assert result.n_groups < len(edges) / expected_compression

    print(f"\n{'='*70}")
    print("QAOA VRA Integration: SUCCESS")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Run end-to-end test
    test_end_to_end_qaoa_grouping()
