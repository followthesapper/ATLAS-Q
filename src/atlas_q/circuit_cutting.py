"""
Circuit Cutting & Entanglement Forging

Enables simulation of circuits beyond MPS connectivity limits by:
- Partitioning circuits into subcircuits
- Classical stitching of results
- Variance reduction techniques

Features:
- Coupling graph analysis
- Min-cut partitioning algorithms
- Cut-point operator insertion
- Classical post-processing
- Entanglement heatmap visualization

Author: ATLAS-Q Contributors
Date: October 2025
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


@dataclass
class CutPoint:
    """Represents a cut location in the circuit"""

    qubit1: int
    qubit2: int
    time_step: int  # When in the circuit this cut occurs
    partition1: int  # Which partition qubit1 belongs to
    partition2: int  # Which partition qubit2 belongs to


@dataclass
class CircuitPartition:
    """Represents a partition of the circuit"""

    partition_id: int
    qubits: Set[int]
    gates: List[Tuple]  # List of (gate_type, qubits, params)
    cut_points: List[CutPoint]


@dataclass
class CuttingConfig:
    """Configuration for circuit cutting"""

    max_partition_size: int = 10  # Max qubits per partition
    max_cuts: int = 3  # Maximum number of cuts
    algorithm: str = "min_cut"  # 'min_cut', 'spectral', 'manual'
    variance_reduction: bool = True
    samples_per_cut: int = 100  # Classical samples per cut


class CouplingGraph:
    """
    Represents the coupling/entanglement structure of a circuit.

    Used for analyzing connectivity and finding optimal partitions.
    """

    def __init__(self, n_qubits: int):
        """
        Initialize coupling graph.

        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
        self.edges = defaultdict(int)  # (q1, q2) -> weight (number of 2q gates)
        self.adjacency = np.zeros((n_qubits, n_qubits), dtype=int)

    def add_two_qubit_gate(self, qubit1: int, qubit2: int, weight: float = 1.0):
        """
        Add a two-qubit gate to the coupling graph.

        Args:
            qubit1: First qubit
            qubit2: Second qubit
            weight: Importance weight (default 1)
        """
        edge = tuple(sorted([qubit1, qubit2]))
        self.edges[edge] += weight
        self.adjacency[qubit1, qubit2] += weight
        self.adjacency[qubit2, qubit1] += weight

    def get_degree(self, qubit: int) -> int:
        """Get degree (number of connections) for a qubit"""
        return self.adjacency[qubit, :].sum()

    def get_neighbors(self, qubit: int) -> List[int]:
        """Get neighboring qubits"""
        return [i for i in range(self.n_qubits) if self.adjacency[qubit, i] > 0]

    def compute_entanglement_heatmap(self) -> np.ndarray:
        """
        Compute entanglement heatmap (normalized adjacency matrix).

        Returns:
            n_qubits × n_qubits matrix of entanglement strengths
        """
        max_weight = self.adjacency.max()
        if max_weight == 0:
            return np.zeros_like(self.adjacency, dtype=float)

        return self.adjacency.astype(float) / max_weight

    def find_bottleneck_edges(self, k: int = 3) -> List[Tuple[int, int, int]]:
        """
        Find the k edges with lowest weight (best candidates for cutting).

        Args:
            k: Number of edges to return

        Returns:
            List of (qubit1, qubit2, weight) tuples
        """
        sorted_edges = sorted(self.edges.items(), key=lambda x: x[1])
        return [(e[0], e[1], w) for (e, w) in sorted_edges[:k]]


class MinCutPartitioner:
    """
    Partition circuit using min-cut algorithm.

    Finds a partition that minimizes the number of edges (cuts) needed.
    """

    def __init__(self, config: CuttingConfig):
        self.config = config

    def partition(self, graph: CouplingGraph, n_partitions: int = 2) -> List[CircuitPartition]:
        """
        Partition the coupling graph.

        Args:
            graph: CouplingGraph to partition
            n_partitions: Number of partitions to create

        Returns:
            List of CircuitPartition objects
        """
        if n_partitions == 2:
            return self._partition_two_way(graph)
        else:
            return self._partition_recursive(graph, n_partitions)

    def _partition_two_way(self, graph: CouplingGraph) -> List[CircuitPartition]:
        """Simple two-way partition using spectral bisection"""
        # Compute Laplacian
        D = np.diag(graph.adjacency.sum(axis=1))
        L = D - graph.adjacency

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Second smallest eigenvector (Fiedler vector)
        fiedler = eigenvectors[:, 1]

        # Partition based on sign
        partition1_qubits = set(np.where(fiedler >= 0)[0].tolist())
        partition2_qubits = set(np.where(fiedler < 0)[0].tolist())

        # Create partitions
        partitions = [
            CircuitPartition(partition_id=0, qubits=partition1_qubits, gates=[], cut_points=[]),
            CircuitPartition(partition_id=1, qubits=partition2_qubits, gates=[], cut_points=[]),
        ]

        # Find cut points
        for (q1, q2), weight in graph.edges.items():
            if q1 in partition1_qubits and q2 in partition2_qubits:
                cut = CutPoint(
                    qubit1=q1,
                    qubit2=q2,
                    time_step=0,  # Would need circuit structure to determine
                    partition1=0,
                    partition2=1,
                )
                partitions[0].cut_points.append(cut)
                partitions[1].cut_points.append(cut)

        return partitions

    def _partition_recursive(
        self, graph: CouplingGraph, n_partitions: int
    ) -> List[CircuitPartition]:
        """Recursive partitioning for n > 2"""
        # Start with 2-way partition
        partitions = self._partition_two_way(graph)

        # Recursively partition largest partition until we have n_partitions
        while len(partitions) < n_partitions:
            # Find largest partition
            largest = max(partitions, key=lambda p: len(p.qubits))

            # Create subgraph
            subgraph = CouplingGraph(graph.n_qubits)
            for (q1, q2), weight in graph.edges.items():
                if q1 in largest.qubits and q2 in largest.qubits:
                    subgraph.add_two_qubit_gate(q1, q2, weight)

            # Split it
            sub_partitions = self._partition_two_way(subgraph)

            # Replace largest with its splits
            partitions.remove(largest)
            partitions.extend(sub_partitions)

        return partitions


class CutOperator:
    """
    Represents a cut operator for circuit cutting.

    Implements the wire-cutting protocol:
    - Original gate G → sample from decomposition {I, X, Y, Z}
    - Classical post-processing to reconstruct expectation values
    """

    def __init__(self, cut_point: CutPoint):
        self.cut_point = cut_point
        self.decomposition = self._build_pauli_decomposition()

    def _build_pauli_decomposition(self) -> Dict[str, torch.Tensor]:
        """
        Build Pauli decomposition for cut operator.

        Returns dictionary of Pauli operators {I, X, Y, Z}
        """
        return {
            "I": torch.eye(2, dtype=torch.complex64),
            "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
            "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
            "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
        }

    def sample(self, rng: np.random.Generator) -> Tuple[str, torch.Tensor]:
        """
        Sample a Pauli operator from the decomposition.

        Args:
            rng: Random number generator

        Returns:
            (operator_name, operator_matrix) tuple
        """
        # Uniform sampling over Pauli group
        pauli_names = ["I", "X", "Y", "Z"]
        choice = rng.choice(pauli_names)
        return choice, self.decomposition[choice]


class CircuitCutter:
    """
    Main interface for circuit cutting and entanglement forging.

    Handles:
    - Circuit partitioning
    - Subcircuit execution
    - Classical stitching
    """

    def __init__(self, config: CuttingConfig):
        self.config = config
        self.partitioner = MinCutPartitioner(config)

    def analyze_circuit(self, gates: List[Tuple]) -> CouplingGraph:
        """
        Analyze circuit and build coupling graph.

        Args:
            gates: List of (gate_type, qubits, params) tuples

        Returns:
            CouplingGraph representing circuit connectivity
        """
        # Determine number of qubits
        max_qubit = max(max(qubits) for (_, qubits, _) in gates)
        n_qubits = max_qubit + 1

        graph = CouplingGraph(n_qubits)

        # Build graph from gates
        for gate_type, qubits, _ in gates:
            if len(qubits) == 2:
                graph.add_two_qubit_gate(qubits[0], qubits[1])

        return graph

    def partition_circuit(
        self, gates: List[Tuple], n_partitions: int = 2
    ) -> List[CircuitPartition]:
        """
        Partition circuit into subcircuits.

        Args:
            gates: Circuit gates
            n_partitions: Number of partitions

        Returns:
            List of CircuitPartition objects
        """
        graph = self.analyze_circuit(gates)
        partitions = self.partitioner.partition(graph, n_partitions)

        # Assign gates to partitions
        for partition in partitions:
            partition.gates = [
                (gt, qs, ps) for (gt, qs, ps) in gates if all(q in partition.qubits for q in qs)
            ]

        return partitions

    def execute_with_cuts(
        self, partitions: List[CircuitPartition], observable: torch.Tensor, device: str = "cuda"
    ) -> Tuple[float, float]:
        """
        Execute partitioned circuit and stitch results.

        Args:
            partitions: Circuit partitions from partition_circuit()
            observable: Observable to measure
            device: Torch device

        Returns:
            (expectation_value, variance) tuple
        """
        from atlas_q.adaptive_mps import AdaptiveMPS

        # Classical sampling over cut configurations
        n_samples = self.config.samples_per_cut ** len(partitions[0].cut_points)
        rng = np.random.default_rng(seed=42)

        results = []

        for _ in range(n_samples):
            # Sample cut operators
            cut_ops = {}
            for cut_point in partitions[0].cut_points:
                cut_op = CutOperator(cut_point)
                pauli_name, pauli_mat = cut_op.sample(rng)
                cut_ops[cut_point] = (pauli_name, pauli_mat)

            # Execute each partition
            partition_results = []
            for partition in partitions:
                mps = AdaptiveMPS(num_qubits=len(partition.qubits), bond_dim=64, device=device)

                # Apply gates (simplified - would need full gate application logic)
                for gate_type, qubits, params in partition.gates:
                    pass  # TODO: Apply gates

                # Apply cut operators at boundaries
                # ...

                # Measure observable on this partition
                # partition_exp = measure_observable(mps, observable)
                partition_exp = 0.0  # Placeholder
                partition_results.append(partition_exp)

            # Classical post-processing: combine partition results
            total_result = np.prod(partition_results)
            results.append(total_result)

        # Estimate expectation and variance
        expectation = np.mean(results)
        variance = np.var(results)

        return float(expectation), float(variance)


def visualize_entanglement_heatmap(
    graph: CouplingGraph, filename: Optional[str] = None
) -> np.ndarray:
    """
    Visualize entanglement heatmap.

    Args:
        graph: CouplingGraph to visualize
        filename: Optional filename to save plot

    Returns:
        Heatmap matrix
    """
    heatmap = graph.compute_entanglement_heatmap()

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar(label="Entanglement Strength")
        plt.xlabel("Qubit")
        plt.ylabel("Qubit")
        plt.title("Circuit Entanglement Heatmap")

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")

    return heatmap


# Example usage
if __name__ == "__main__":
    print("Circuit Cutting Example")
    print("=" * 50)

    # Example circuit: 8-qubit chain with entangling gates
    gates = [("H", [i], []) for i in range(8)] + [("CNOT", [i, i + 1], []) for i in range(7)]

    # Create cutter
    config = CuttingConfig(max_partition_size=4, max_cuts=2)
    cutter = CircuitCutter(config)

    # Analyze and partition
    graph = cutter.analyze_circuit(gates)
    print(f"Circuit has {len(graph.edges)} two-qubit gates")

    partitions = cutter.partition_circuit(gates, n_partitions=2)
    print(f"Partitioned into {len(partitions)} subcircuits:")
    for p in partitions:
        print(f"  Partition {p.partition_id}: {len(p.qubits)} qubits, {len(p.cut_points)} cuts")

    # Find bottleneck edges
    bottlenecks = graph.find_bottleneck_edges(k=3)
    print("\nBest cut candidates:")
    for q1, q2, w in bottlenecks:
        print(f"  Edge ({q1}, {q2}) weight={w}")
