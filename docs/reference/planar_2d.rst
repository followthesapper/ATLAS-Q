2D/Planar Circuits
==================

Simulation of quantum circuits on 2D qubit layouts (superconducting, trapped-ion, neutral atom devices).

.. currentmodule:: atlas_q.planar_2d

Overview
--------

Most near-term quantum processors have qubits arranged in 2D geometries: square grids (superconducting qubits), triangular lattices (trapped ions), or arbitrary 2D patterns (neutral atoms). Simulating these circuits efficiently requires mapping the 2D layout to a 1D Matrix Product State while minimizing the entanglement cost.

**The Challenge**

MPS are inherently 1D: they efficiently represent states with nearest-neighbor entanglement on a chain. A 2D circuit must be "flattened" to 1D, but this mapping determines simulation cost:

- **Good mapping**: Nearby qubits in 2D remain nearby in 1D → low bond dimension χ
- **Bad mapping**: 2D-adjacent qubits far apart in 1D → high χ, expensive simulation

**Snake Ordering**

The snake (or boustrophedon) pattern is optimal for square grids [Markov08]_:

.. code-block:: text

   2D Grid (3×3):          1D MPS Ordering:

   0 - 1 - 2              0 → 1 → 2
   |   |   |                      ↓
   3 - 4 - 5              5 ← 4 ← 3
   |   |   |              ↓
   6 - 7 - 8              6 → 7 → 8

This ensures:

- Horizontal neighbors: distance 1 in MPS
- Vertical neighbors: distance 1 in MPS (via snake wrap)
- Average distance: O(1) vs. O(√n) for row-major ordering

**SWAP Network Synthesis**

For gates between non-adjacent qubits (after 1D mapping), insert SWAP gates to move qubits together:

.. math::

   \text{CX}(q_i, q_j) \to \text{SWAP}_i^{j-1} \cdot \text{CX}(q_{j-1}, q_j) \cdot \text{SWAP}_i^{j-1}

Cost: O(|i-j|) SWAP gates, O(|i-j|) circuit depth increase.

**Key Features**

- **Automatic mapping**: Snake, Hilbert curve, or custom orderings
- **SWAP optimization**: Minimize inserted SWAPs via A* search or greedy heuristics
- **Topology-aware**: Supports square grid, heavy-hex (IBM), triangular, arbitrary
- **Adaptive χ**: Increase bond dimension for entanglement hotspots
- **2D measurement**: Correlations and observables in original 2D coordinates

Topologies
----------

.. class:: Topology

   2D qubit layout topologies.

   .. attribute:: SQUARE_GRID

      Regular square lattice

   .. attribute:: HEAVY_HEX

      IBM heavy-hex topology

   .. attribute:: TRIANGULAR

      Triangular lattice

   .. attribute:: CUSTOM

      User-defined topology

Data Structures
---------------

.. class:: Qubit2D

   Represents a qubit in 2D layout.

   :param int row: Row index
   :param int col: Column index
   :param int index_1d: Index in 1D MPS ordering (optional)

.. class:: Layout2D

   2D qubit layout specification.

   :param int rows: Number of rows
   :param int cols: Number of columns
   :param Topology topology: Layout topology
   :param list coupling_map: List of connected qubit pairs
   :param dict qubits: Dictionary mapping (row, col) to Qubit2D

.. class:: MappingConfig

   Configuration for 2D → 1D mapping.

   :param str strategy: Mapping strategy ('snake', 'row_major', 'col_major', 'hilbert')
   :param bool optimize_swaps: Optimize SWAP network
   :param int max_swap_layers: Maximum SWAP layers (default 100)
   :param str chi_schedule: Bond dimension schedule ('adaptive', 'fixed', 'exponential')

Classes
-------

.. class:: SnakeMapper(rows, cols)

   Maps 2D qubit grid to 1D MPS using snake pattern.

   Snake pattern minimizes long-range interactions:

   .. code-block:: text

       0 → 1 → 2
               ↓
       5 ← 4 ← 3
       ↓
       6 → 7 → 8

   :param int rows: Number of rows
   :param int cols: Number of columns

   **Methods:**

   .. method:: map_2d_to_1d(row, col)

      Map (row, col) to 1D index.

      :param int row: Row index
      :param int col: Column index
      :return: 1D index in MPS ordering
      :rtype: int

   .. method:: map_1d_to_2d(index_1d)

      Map 1D index back to (row, col).

      :param int index_1d: 1D index
      :return: (row, col) tuple
      :rtype: tuple

   .. method:: get_distance(row1, col1, row2, col2)

      Compute Manhattan distance in snake ordering.

      :param int row1: First row
      :param int col1: First column
      :param int row2: Second row
      :param int col2: Second column
      :return: Distance in 1D MPS ordering
      :rtype: int

.. class:: SWAPRouter(layout)

   SWAP network synthesis for non-nearest-neighbor gates.

   :param Layout2D layout: 2D qubit layout

   **Methods:**

   .. method:: route_gate(qubit1, qubit2)

      Find SWAP path to make qubits adjacent.

      :param int qubit1: First qubit (1D index)
      :param int qubit2: Second qubit (1D index)
      :return: List of SWAP operations
      :rtype: list

   .. method:: optimize_swap_network(gates)

      Optimize full SWAP network for gate sequence.

      :param list gates: List of gates
      :return: Optimized gate sequence with SWAPs inserted
      :rtype: list

.. class:: PlanarCircuitSimulator(layout, config)

   Simulator for 2D planar circuits.

   :param Layout2D layout: 2D qubit layout
   :param MappingConfig config: Mapping configuration

   **Methods:**

   .. method:: compile_circuit(gates_2d)

      Compile 2D circuit to 1D MPS-compatible circuit.

      :param list gates_2d: Gates with 2D qubit indices
      :return: Gates with 1D indices and inserted SWAPs
      :rtype: list

   .. method:: run_circuit(gates_2d)

      Run 2D circuit using MPS backend.

      :param list gates_2d: Gates with 2D qubit indices
      :return: Final MPS state
      :rtype: AdaptiveMPS

Examples
--------

Snake mapping for square grid:

.. code-block:: python

   from atlas_q.planar_2d import SnakeMapper

   # 4×4 grid
   mapper = SnakeMapper(rows=4, cols=4)

   # Map 2D coordinates
   idx_1d = mapper.map_2d_to_1d(row=1, col=2)
   print(f"(1,2) → {idx_1d}")  # Output: (1,2) → 5

   # Reverse mapping
   row, col = mapper.map_1d_to_2d(idx_1d)
   print(f"{idx_1d} → ({row},{col})")

   # Check distance
   dist = mapper.get_distance(0, 0, 3, 3)
   print(f"Distance from (0,0) to (3,3): {dist}")

Simulate 2D circuit:

.. code-block:: python

   from atlas_q.planar_2d import Layout2D, Topology, PlanarCircuitSimulator, MappingConfig

   # Create 5×5 square grid layout
   layout = Layout2D(
       rows=5,
       cols=5,
       topology=Topology.SQUARE_GRID,
       coupling_map=[(i, i+1) for i in range(24)],  # Simplified
       qubits={}
   )

   config = MappingConfig(strategy='snake', optimize_swaps=True)
   sim = PlanarCircuitSimulator(layout, config)

   # Define 2D circuit
   gates_2d = [
       ('H', [(0, 0)], []),
       ('H', [(0, 1)], []),
       ('CZ', [(0, 0), (0, 1)], []),  # Adjacent in 2D
       ('CZ', [(0, 0), (4, 4)], []),  # Non-adjacent, requires SWAPs
   ]

   # Compile and run
   mps = sim.run_circuit(gates_2d)
   print(f"Final bond dimensions: {[t.shape for t in mps.tensors]}")

SWAP routing:

.. code-block:: python

   from atlas_q.planar_2d import SWAPRouter

   router = SWAPRouter(layout)

   # Route gate between distant qubits
   swaps = router.route_gate(qubit1=0, qubit2=24)  # Corners of 5×5 grid
   print(f"Required SWAPs: {swaps}")

   # Optimize full circuit
   optimized_gates = router.optimize_swap_network(gates_2d)

Adaptive bond dimension for 2D:

.. code-block:: python

   config = MappingConfig(
       strategy='snake',
       chi_schedule='adaptive'  # Increase χ for long-range gates
   )
   sim = PlanarCircuitSimulator(layout, config)

Performance Notes
-----------------

**Mapping Quality Comparison**

.. list-table:: 1D Mapping Strategies for 5×5 Grid
   :header-rows: 1
   :widths: 25 20 25 30

   * - Mapping
     - Avg Distance
     - Max Distance
     - χ Required
   * - Row-major
     - 2.5
     - 8
     - χ ~ 64-128
   * - Column-major
     - 2.5
     - 8
     - χ ~ 64-128
   * - Snake
     - 1.4
     - 4
     - χ ~ 32-48
   * - Hilbert curve
     - 1.2
     - 3
     - χ ~ 24-32

Snake and Hilbert curve mappings significantly reduce required bond dimension.

**SWAP Overhead Scaling**

For n×n grid with random gates:

- **No optimization**: ~n² SWAPs per layer
- **Greedy optimization**: ~n SWAPs per layer
- **Optimal (A*)**: ~√n SWAPs per layer (NP-hard, only practical for small circuits)

**Bond Dimension Requirements**

.. list-table:: χ for Various 2D Grid Sizes (Snake Mapping)
   :header-rows: 1
   :widths: 20 25 30 25

   * - Grid Size
     - Total Qubits
     - Recommended χ
     - Memory (GPU)
   * - 3×3
     - 9
     - 16-24
     - 100 MB
   * - 5×5
     - 25
     - 32-48
     - 500 MB
   * - 7×7
     - 49
     - 48-64
     - 2 GB
   * - 10×10
     - 100
     - 64-96
     - 8 GB

**Simulation Time** (per gate, NVIDIA A100)

- 3×3 grid, χ=24: ~3 ms/gate
- 5×5 grid, χ=32: ~12 ms/gate
- 7×7 grid, χ=48: ~45 ms/gate

Best Practices
--------------

**Choosing Mapping Strategy**

1. **Snake ordering** (default): Best for square grids, simple and effective
2. **Hilbert curve**: Slightly better for very large grids (>10×10), more complex
3. **Row/column-major**: Avoid unless specific hardware constraints require it
4. **Custom**: For irregular topologies (heavy-hex, triangular)

**SWAP Optimization**

.. code-block:: python

   # Enable SWAP optimization (recommended)
   config = MappingConfig(
       strategy='snake',
       optimize_swaps=True,      # Use greedy SWAP minimization
       max_swap_layers=100       # Limit search depth
   )

   # For small circuits, use optimal routing
   if num_gates < 50:
       config.router_algorithm = 'astar'  # Optimal but slow
   else:
       config.router_algorithm = 'greedy'  # Fast heuristic

**Adaptive Bond Dimension**

For circuits with variable entanglement:

.. code-block:: python

   config = MappingConfig(
       strategy='snake',
       chi_schedule='adaptive',
       chi_min=16,
       chi_max=64,
       chi_threshold=1e-8  # Truncation threshold
   )

   sim = PlanarCircuitSimulator(layout, config)

χ starts at chi_min and grows to chi_max based on entanglement.

**Monitoring Performance**

.. code-block:: python

   mps = sim.run_circuit(gates_2d)

   # Check bond dimensions
   stats = mps.stats_summary()
   print(f"Max χ: {stats['max_chi']}")
   print(f"Avg χ: {stats['avg_chi']:.1f}")

   # Count inserted SWAPs
   compiled_gates = sim.compile_circuit(gates_2d)
   swap_count = sum(1 for g in compiled_gates if g[0] == 'SWAP')
   print(f"Inserted SWAPs: {swap_count}")

   # If too many SWAPs, circuit may not be 2D-local
   if swap_count > len(gates_2d):
       print("Warning: SWAP overhead > gate count, consider circuit cutting")

**Troubleshooting**

- **High χ required**: Check if gates are truly local in 2D; consider circuit cutting for long-range gates
- **Too many SWAPs**: Gates not respecting 2D locality; reorder operations or use different mapping
- **Memory errors**: Reduce χ_max or use smaller grid sections
- **Slow compilation**: Use 'greedy' router instead of 'astar' for large circuits

Limitations
-----------

**MPS vs. PEPS for 2D**

.. list-table:: 2D Simulation Method Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Method
     - Advantages
     - Limitations
   * - **MPS + Snake**
     - Fast, standard tools, χ ~ 32-64
     - SWAP overhead, χ grows with depth
   * - **PEPS**
     - No SWAP, natural 2D, χ ~ 4-8
     - Slower contractions, experimental
   * - **Circuit Cutting**
     - Handles arbitrary connectivity
     - Exponential sampling cost

**Use MPS + Snake when**:

- Grid is small to medium (≤ 10×10)
- Circuit depth moderate (≤ 20 layers)
- Gates mostly 2D-local

**Use PEPS when**:

- Grid is large (> 10×10)
- Circuit has deep 2D structure
- Can tolerate slower per-gate cost for lower χ

**Use Circuit Cutting when**:

- Grid connectivity complex (heavy-hex, arbitrary)
- Long-range gates frequent
- Willing to pay classical sampling overhead

Use Cases
---------

**Superconducting Qubit Devices**

- IBM Quantum processors (heavy-hex topology)
- Google Sycamore (54-qubit grid)
- Rigetti Aspen (square grid variants)

Example: Simulating Google's 53-qubit supremacy circuit on 7×8 grid with snake mapping, χ=48.

**Trapped-Ion Systems**

- IonQ (all-to-all connectivity, but often use 2D layouts)
- Honeywell (linear chains, extendable to 2D)

Example: 10×10 triangular lattice for quantum simulation of magnetic models.

**Neutral Atom Processors**

- QuEra/Harvard (arbitrary 2D patterns)
- Pasqal (programmable geometries)

Example: Custom 2D layout optimized for QAOA on MaxCut problems.

**Quantum Simulation**

- 2D Ising model dynamics
- Hubbard model on square lattice
- Surface code error correction

See Also
--------

- :doc:`peps` - True 2D tensor networks for lattice systems
- :doc:`circuit_cutting` - Alternative for high-connectivity circuits
- :doc:`distributed_mps` - Multi-GPU for very large 2D grids
- :doc:`../user_guide/howtos/handle_large_systems` - Scaling strategies
- :doc:`../user_guide/explanations/tensor_networks` - MPS theory

References
----------

.. [Markov08] I. L. Markov & Y. Shi, "Simulating quantum computation by contracting tensor networks," *SIAM Journal on Computing* 38, 963 (2008).

.. [Pednault17] E. Pednault et al., "Breaking the 49-qubit barrier in the simulation of quantum circuits," *arXiv:1710.05867* (2017).

.. [Boixo18] S. Boixo et al., "Characterizing quantum supremacy in near-term devices," *Nature Physics* 14, 595 (2018).

.. [Daley22] A. J. Daley et al., "Practical quantum advantage in quantum simulation," *Nature* 607, 667 (2022).
