"""
Test Circuit Cutting integration with AdaptiveMPS
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from atlas_q.circuit_cutting import (
    CouplingGraph,
    CircuitCutter,
    CuttingConfig,
    MinCutPartitioner
)


def test_coupling_graph_basic():
    """Test basic coupling graph construction"""
    graph = CouplingGraph(n_qubits=5)

    # Add some two-qubit gates
    graph.add_two_qubit_gate(0, 1)
    graph.add_two_qubit_gate(1, 2)
    graph.add_two_qubit_gate(2, 3)
    graph.add_two_qubit_gate(3, 4)

    # Check edges
    assert len(graph.edges) == 4
    assert graph.edges[(0, 1)] == 1
    assert graph.edges[(1, 2)] == 1

    # Check degrees
    assert graph.get_degree(0) == 1  # Connected to 1 only
    assert graph.get_degree(2) == 2  # Connected to 1 and 3

    print("✅ test_coupling_graph_basic passed")


def test_coupling_graph_heatmap():
    """Test entanglement heatmap computation"""
    graph = CouplingGraph(n_qubits=4)

    # Create diamond pattern
    graph.add_two_qubit_gate(0, 1)
    graph.add_two_qubit_gate(0, 2)
    graph.add_two_qubit_gate(1, 3)
    graph.add_two_qubit_gate(2, 3)

    heatmap = graph.compute_entanglement_heatmap()

    assert heatmap.shape == (4, 4)
    assert heatmap[0, 1] > 0  # Edge exists
    assert heatmap[0, 3] == 0  # No direct edge

    print("✅ test_coupling_graph_heatmap passed")


def test_bottleneck_edges():
    """Test finding bottleneck edges for cutting"""
    graph = CouplingGraph(n_qubits=6)

    # Create two highly connected clusters with weak link
    # Cluster 1: sites 0,1,2
    graph.add_two_qubit_gate(0, 1, weight=5.0)
    graph.add_two_qubit_gate(1, 2, weight=5.0)
    graph.add_two_qubit_gate(0, 2, weight=5.0)

    # Weak link
    graph.add_two_qubit_gate(2, 3, weight=1.0)

    # Cluster 2: sites 3,4,5
    graph.add_two_qubit_gate(3, 4, weight=5.0)
    graph.add_two_qubit_gate(4, 5, weight=5.0)
    graph.add_two_qubit_gate(3, 5, weight=5.0)

    bottlenecks = graph.find_bottleneck_edges(k=1)

    # Weakest edge should be (2,3)
    assert bottlenecks[0][0] == 2
    assert bottlenecks[0][1] == 3
    assert bottlenecks[0][2] == 1.0

    print("✅ test_bottleneck_edges passed")


def test_min_cut_partitioner():
    """Test min-cut partitioning algorithm"""
    graph = CouplingGraph(n_qubits=8)

    # Create chain: 0-1-2-3-4-5-6-7
    for i in range(7):
        graph.add_two_qubit_gate(i, i+1)

    config = CuttingConfig(max_partition_size=4, max_cuts=1)
    partitioner = MinCutPartitioner(config)

    partitions = partitioner.partition(graph, n_partitions=2)

    assert len(partitions) == 2
    assert partitions[0].partition_id == 0
    assert partitions[1].partition_id == 1

    # Check that qubits are distributed
    total_qubits = len(partitions[0].qubits) + len(partitions[1].qubits)
    assert total_qubits == 8

    # Check cuts exist
    assert len(partitions[0].cut_points) > 0

    print("✅ test_min_cut_partitioner passed")


def test_circuit_cutter_analyze():
    """Test circuit analysis and graph construction"""
    # Create simple circuit
    gates = [
        ('H', [0], []),
        ('H', [1], []),
        ('CNOT', [0, 1], []),
        ('CNOT', [1, 2], []),
        ('CNOT', [2, 3], []),
    ]

    config = CuttingConfig(max_partition_size=2)
    cutter = CircuitCutter(config)

    graph = cutter.analyze_circuit(gates)

    assert graph.n_qubits == 4
    assert len(graph.edges) == 3  # Three CNOT gates

    print("✅ test_circuit_cutter_analyze passed")


def test_circuit_cutter_partition():
    """Test circuit partitioning"""
    # 6-qubit circuit
    gates = [
        ('H', [i], []) for i in range(6)
    ] + [
        ('CNOT', [i, i+1], []) for i in range(5)
    ]

    config = CuttingConfig(max_partition_size=3)
    cutter = CircuitCutter(config)

    partitions = cutter.partition_circuit(gates, n_partitions=2)

    assert len(partitions) == 2

    # Each partition should have some gates
    for p in partitions:
        assert len(p.gates) > 0
        assert len(p.qubits) > 0

    print(f"Partition 0: {len(p.qubits)} qubits, {len(partitions[0].gates)} gates")
    print(f"Partition 1: {len(p.qubits)} qubits, {len(partitions[1].gates)} gates")
    print(f"Cut points: {len(partitions[0].cut_points)}")

    print("✅ test_circuit_cutter_partition passed")


def test_circuit_cutting_neighbors():
    """Test neighbor detection in coupling graph"""
    graph = CouplingGraph(n_qubits=5)

    graph.add_two_qubit_gate(0, 1)
    graph.add_two_qubit_gate(1, 2)

    neighbors_1 = graph.get_neighbors(1)

    assert 0 in neighbors_1
    assert 2 in neighbors_1
    assert 3 not in neighbors_1

    print("✅ test_circuit_cutting_neighbors passed")


if __name__ == "__main__":
    print("Running Circuit Cutting tests...\n")

    try:
        test_coupling_graph_basic()
        test_coupling_graph_heatmap()
        test_bottleneck_edges()
        test_min_cut_partitioner()
        test_circuit_cutter_analyze()
        test_circuit_cutter_partition()
        test_circuit_cutting_neighbors()

        print("\n" + "="*50)
        print("✅ All Circuit Cutting tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
