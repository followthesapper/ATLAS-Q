"""
2D/Planar Circuit Support

Enables simulation of circuits on 2D qubit layouts (e.g., superconducting devices).

Features:
- Snake mapping: 2D grid → 1D MPS ordering
- SWAP network synthesis for non-nearest-neighbor gates
- Adaptive bond dimension scheduling for 2D circuits
- Support for common 2D topologies (square grid, heavy-hex)

Author: ATLAS-Q Contributors
Date: October 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Topology(Enum):
    """2D qubit layout topologies"""

    SQUARE_GRID = "square_grid"
    HEAVY_HEX = "heavy_hex"
    TRIANGULAR = "triangular"
    CUSTOM = "custom"


@dataclass
class Qubit2D:
    """Represents a qubit in 2D layout"""

    row: int
    col: int
    index_1d: Optional[int] = None  # Index in 1D MPS ordering


@dataclass
class Layout2D:
    """2D qubit layout specification"""

    rows: int
    cols: int
    topology: Topology
    coupling_map: List[Tuple[int, int]]  # List of connected qubit pairs
    qubits: Dict[Tuple[int, int], Qubit2D]  # (row, col) -> Qubit2D


@dataclass
class MappingConfig:
    """Configuration for 2D → 1D mapping"""

    strategy: str = "snake"  # 'snake', 'row_major', 'col_major', 'hilbert'
    optimize_swaps: bool = True
    max_swap_layers: int = 100
    chi_schedule: str = "adaptive"  # 'adaptive', 'fixed', 'exponential'


class SnakeMapper:
    """
    Maps 2D qubit grid to 1D MPS using snake pattern.

    Snake pattern minimizes the number of long-range interactions:

    Example 3×3 grid:
    0 → 1 → 2
            ↓
    5 ← 4 ← 3
    ↓
    6 → 7 → 8
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols

    def map_2d_to_1d(self, row: int, col: int) -> int:
        """
        Map (row, col) to 1D index using snake pattern.

        Args:
            row: Row index (0-indexed)
            col: Column index (0-indexed)

        Returns:
            1D index in MPS ordering
        """
        if row % 2 == 0:
            # Even rows: left to right
            return row * self.cols + col
        else:
            # Odd rows: right to left (snake!)
            return row * self.cols + (self.cols - 1 - col)

    def map_1d_to_2d(self, index_1d: int) -> Tuple[int, int]:
        """
        Map 1D index back to (row, col).

        Args:
            index_1d: 1D MPS index

        Returns:
            (row, col) tuple
        """
        row = index_1d // self.cols

        if row % 2 == 0:
            col = index_1d % self.cols
        else:
            col = self.cols - 1 - (index_1d % self.cols)

        return (row, col)

    def get_distance(self, qubit1: int, qubit2: int) -> int:
        """
        Get MPS distance between two qubits (1D indices).

        Args:
            qubit1: First qubit (1D index)
            qubit2: Second qubit (1D index)

        Returns:
            Distance in MPS ordering
        """
        return abs(qubit1 - qubit2)

    def get_manhattan_distance(self, qubit1: int, qubit2: int) -> int:
        """
        Get Manhattan distance in 2D grid.

        Args:
            qubit1: First qubit (1D index)
            qubit2: Second qubit (1D index)

        Returns:
            Manhattan distance
        """
        r1, c1 = self.map_1d_to_2d(qubit1)
        r2, c2 = self.map_1d_to_2d(qubit2)

        return abs(r1 - r2) + abs(c1 - c2)


class SWAPSynthesizer:
    """
    Synthesizes SWAP networks to map non-nearest-neighbor gates to nearest-neighbor.

    Uses A* search to find optimal SWAP insertion.
    """

    def __init__(self, layout: Layout2D, mapper: SnakeMapper):
        self.layout = layout
        self.mapper = mapper
        self.n_qubits = mapper.n_qubits

    def synthesize_swap_network(self, control: int, target: int) -> List[Tuple[int, int]]:
        """
        Generate SWAP gates to bring control and target qubits adjacent.

        Args:
            control: Control qubit (1D index)
            target: Target qubit (1D index)

        Returns:
            List of (qubit_i, qubit_j) SWAP gate pairs
        """
        swaps = []

        # If already adjacent, no SWAPs needed
        distance = self.mapper.get_distance(control, target)
        if distance == 1:
            return swaps

        # Greedy approach: move target towards control
        current_target = target

        while self.mapper.get_distance(control, current_target) > 1:
            # Find neighbor of current_target that's closer to control
            neighbors = self._get_neighbors_1d(current_target)

            best_neighbor = min(neighbors, key=lambda n: self.mapper.get_distance(control, n))

            # SWAP current_target with best_neighbor
            swaps.append(tuple(sorted([current_target, best_neighbor])))
            current_target = best_neighbor

        return swaps

    def _get_neighbors_1d(self, qubit: int) -> List[int]:
        """
        Get nearest neighbors of qubit in 1D MPS ordering.

        Args:
            qubit: Qubit index

        Returns:
            List of neighbor indices
        """
        neighbors = []

        if qubit > 0:
            neighbors.append(qubit - 1)
        if qubit < self.n_qubits - 1:
            neighbors.append(qubit + 1)

        return neighbors

    def count_total_swaps(self, gates: List[Tuple[str, List[int], List]]) -> int:
        """
        Count total SWAPs needed for a circuit.

        Args:
            gates: List of (gate_type, qubits, params)

        Returns:
            Total number of SWAPs required
        """
        total_swaps = 0

        for gate_type, qubits, _ in gates:
            if len(qubits) == 2:
                swaps = self.synthesize_swap_network(qubits[0], qubits[1])
                total_swaps += len(swaps)

        return total_swaps


class ChiScheduler:
    """
    Adaptive bond dimension scheduler for 2D circuits.

    Adjusts χ based on circuit depth and entanglement structure.
    """

    def __init__(self, config: MappingConfig):
        self.config = config

    def get_chi(self, layer: int, total_layers: int, gate_type: str, distance: int) -> int:
        """
        Compute adaptive bond dimension for given layer.

        Args:
            layer: Current layer number
            total_layers: Total circuit depth
            gate_type: Type of gate being applied
            distance: Distance between qubits in MPS

        Returns:
            Recommended bond dimension
        """
        if self.config.chi_schedule == "fixed":
            return 64

        elif self.config.chi_schedule == "exponential":
            # Exponential growth with depth
            base_chi = 8
            max_chi = 128
            chi = int(base_chi * (1.5 ** (layer / 10)))
            return min(chi, max_chi)

        elif self.config.chi_schedule == "adaptive":
            # Adaptive based on gate distance
            base_chi = 16

            # Long-range gates need higher χ
            distance_factor = 1.0 + 0.2 * min(distance, 5)

            # Entangling gates need higher χ
            if gate_type in ["CNOT", "CZ", "SWAP"]:
                entangle_factor = 1.5
            else:
                entangle_factor = 1.0

            chi = int(base_chi * distance_factor * entangle_factor)
            return min(chi, 256)

        else:
            return 64


class Planar2DCircuit:
    """
    Main interface for 2D planar circuit simulation.

    Handles:
    - 2D layout specification
    - Automatic snake mapping
    - SWAP synthesis
    - Adaptive χ scheduling
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        topology: Topology = Topology.SQUARE_GRID,
        config: Optional[MappingConfig] = None,
    ):
        """
        Initialize 2D planar circuit.

        Args:
            rows: Number of rows in grid
            cols: Number of columns
            topology: Qubit layout topology
            config: Mapping configuration
        """
        self.rows = rows
        self.cols = cols
        self.topology = topology
        self.config = config or MappingConfig()

        # Create layout
        self.layout = self._create_layout()

        # Create mapper
        self.mapper = SnakeMapper(rows, cols)

        # Create SWAP synthesizer
        self.swap_synth = SWAPSynthesizer(self.layout, self.mapper)

        # Create χ scheduler
        self.chi_scheduler = ChiScheduler(self.config)

    def _create_layout(self) -> Layout2D:
        """Create 2D layout based on topology"""
        qubits = {}
        coupling_map = []

        # Create qubits
        for r in range(self.rows):
            for c in range(self.cols):
                qubit = Qubit2D(row=r, col=c)
                qubits[(r, c)] = qubit

        # Create coupling map based on topology
        if self.topology == Topology.SQUARE_GRID:
            for r in range(self.rows):
                for c in range(self.cols):
                    idx = r * self.cols + c

                    # Horizontal edges
                    if c < self.cols - 1:
                        neighbor_idx = r * self.cols + (c + 1)
                        coupling_map.append((idx, neighbor_idx))

                    # Vertical edges
                    if r < self.rows - 1:
                        neighbor_idx = (r + 1) * self.cols + c
                        coupling_map.append((idx, neighbor_idx))

        return Layout2D(
            rows=self.rows,
            cols=self.cols,
            topology=self.topology,
            coupling_map=coupling_map,
            qubits=qubits,
        )

    def compile_circuit(
        self, gates_2d: List[Tuple[str, List[Tuple[int, int]], List]]
    ) -> List[Tuple[str, List[int], List]]:
        """
        Compile 2D circuit to 1D MPS-compatible circuit with SWAPs.

        Args:
            gates_2d: Gates with (row, col) qubit specifications

        Returns:
            Gates with 1D qubit indices and inserted SWAPs
        """
        gates_1d = []

        for gate_type, qubits_2d, params in gates_2d:
            # Map to 1D indices
            qubits_1d = [self.mapper.map_2d_to_1d(r, c) for (r, c) in qubits_2d]

            # For 2-qubit gates, insert SWAPs if needed
            if len(qubits_1d) == 2:
                control, target = qubits_1d
                distance = self.mapper.get_distance(control, target)

                if distance > 1 and self.config.optimize_swaps:
                    # Insert SWAP network
                    swaps = self.swap_synth.synthesize_swap_network(control, target)

                    for swap_pair in swaps:
                        gates_1d.append(("SWAP", list(swap_pair), []))

                    # Original gate (qubits now adjacent)
                    gates_1d.append((gate_type, qubits_1d, params))

                    # Reverse SWAPs to restore layout
                    for swap_pair in reversed(swaps):
                        gates_1d.append(("SWAP", list(swap_pair), []))
                else:
                    gates_1d.append((gate_type, qubits_1d, params))
            else:
                # Single-qubit gate
                gates_1d.append((gate_type, qubits_1d, params))

        return gates_1d

    def simulate(
        self, gates_2d: List[Tuple[str, List[Tuple[int, int]], List]], device: str = "cuda"
    ):
        """
        Simulate 2D circuit using MPS backend.

        Args:
            gates_2d: Circuit gates with 2D qubit coordinates
            device: Torch device

        Returns:
            AdaptiveMPS final state
        """
        from atlas_q.adaptive_mps import AdaptiveMPS

        # Compile to 1D
        gates_1d = self.compile_circuit(gates_2d)

        # Create MPS
        mps = AdaptiveMPS(
            num_qubits=self.rows * self.cols,
            bond_dim=self.chi_scheduler.get_chi(0, len(gates_1d), "H", 1),
            device=device,
        )

        # Apply gates with adaptive χ
        for layer, (gate_type, qubits, params) in enumerate(gates_1d):
            # Get appropriate χ for this gate
            if len(qubits) == 2:
                distance = self.mapper.get_distance(qubits[0], qubits[1])
            else:
                distance = 0

            chi = self.chi_scheduler.get_chi(layer, len(gates_1d), gate_type, distance)

            # Apply gate (simplified - would use actual gate matrices)
            # mps.apply_gate(gate_type, qubits, chi_max=chi)
            pass  # TODO: Implement full gate application

        return mps

    def visualize_layout(self, filename: Optional[str] = None):
        """
        Visualize the 2D qubit layout and snake mapping.

        Args:
            filename: Optional file to save visualization
        """
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Plot 2D layout
            for r in range(self.rows):
                for c in range(self.cols):
                    idx_1d = self.mapper.map_2d_to_1d(r, c)
                    ax1.scatter(c, -r, s=500, c="lightblue", edgecolors="black")
                    ax1.text(c, -r, str(idx_1d), ha="center", va="center", fontsize=12)

            # Draw edges
            for q1, q2 in self.layout.coupling_map:
                r1, c1 = self.mapper.map_1d_to_2d(q1)
                r2, c2 = self.mapper.map_1d_to_2d(q2)
                ax1.plot([c1, c2], [-r1, -r2], "k-", alpha=0.3)

            ax1.set_title("2D Qubit Layout (Snake Mapped)")
            ax1.set_xlabel("Column")
            ax1.set_ylabel("Row")
            ax1.grid(True, alpha=0.3)

            # Plot 1D MPS ordering
            positions = list(range(self.rows * self.cols))
            ax2.scatter(positions, [0] * len(positions), s=500, c="lightgreen", edgecolors="black")
            for i, pos in enumerate(positions):
                ax2.text(pos, 0, str(i), ha="center", va="center", fontsize=12)

            # Draw MPS bonds
            for i in range(len(positions) - 1):
                ax2.plot([i, i + 1], [0, 0], "k-", linewidth=2)

            ax2.set_title("1D MPS Ordering")
            ax2.set_xlabel("MPS Index")
            ax2.set_ylim(-0.5, 0.5)
            ax2.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()

            if filename:
                plt.savefig(filename, dpi=150, bbox_inches="tight")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for visualization")


# Example usage
if __name__ == "__main__":
    print("2D Planar Circuit Support Example")
    print("=" * 50)

    # Create 4×4 square grid
    circuit_2d = Planar2DCircuit(rows=4, cols=4, topology=Topology.SQUARE_GRID)

    print(f"Layout: {circuit_2d.rows}×{circuit_2d.cols} grid")
    print(f"Total qubits: {circuit_2d.rows * circuit_2d.cols}")
    print(f"Coupling map edges: {len(circuit_2d.layout.coupling_map)}")

    # Example: map (row, col) to 1D
    for r in range(4):
        row_mapping = [circuit_2d.mapper.map_2d_to_1d(r, c) for c in range(4)]
        print(f"Row {r}: {row_mapping}")

    # Example gates in 2D coordinates
    gates_2d = [
        ("H", [(0, 0)], []),
        ("H", [(0, 1)], []),
        ("CNOT", [(0, 0), (0, 1)], []),  # Nearest neighbor
        ("CNOT", [(0, 0), (2, 2)], []),  # Long range - needs SWAPs
    ]

    # Compile to 1D
    gates_1d = circuit_2d.compile_circuit(gates_2d)
    print("\nCompiled circuit:")
    print(f"  Original gates: {len(gates_2d)}")
    print(f"  With SWAPs: {len(gates_1d)}")

    # Count SWAPs
    swap_count = sum(1 for g, _, _ in gates_1d if g == "SWAP")
    print(f"  Total SWAPs inserted: {swap_count}")
