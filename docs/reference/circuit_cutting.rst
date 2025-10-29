Circuit Cutting
===============

Partition large quantum circuits to enable simulation beyond hardware connectivity limits.

.. currentmodule:: atlas_q.circuit_cutting

Overview
--------

Circuit cutting enables simulation of large quantum circuits by decomposing them into smaller, simulatable subcircuits. This is essential when:

- Circuit has non-local connectivity (all-to-all, high-degree graphs)
- Number of qubits exceeds single-simulator capacity
- Circuit width is constrained by hardware topology

**Mathematical Foundation**

Circuit cutting exploits the linearity of quantum mechanics. A two-qubit gate :math:`U_{ij}` acting across a cut can be decomposed using quasi-probability decomposition [Peng20]_:

.. math::

   U_{ij} = \sum_{\alpha,\beta} c_{\alpha\beta} L^\alpha_i \otimes R^\beta_j

where :math:`L^\alpha`, :math:`R^\beta` are local operations and :math:`c_{\alpha\beta}` are (possibly negative) coefficients. The expectation value is reconstructed via:

.. math::

   \langle O \rangle = \sum_{\alpha,\beta} c_{\alpha\beta} \langle O_\alpha \rangle_L \langle O_\beta \rangle_R

where subscripts L, R denote left and right partitions.

**Key Challenges**

1. **Sampling overhead**: Exponential in number of cuts (:math:`\sim 4^k` for k cuts)
2. **Variance**: Negative quasi-probabilities increase statistical variance
3. **Partitioning quality**: Good cuts minimize overhead

**Algorithm Workflow**

1. **Coupling graph analysis**: Build graph G = (V, E) where edges represent two-qubit gates
2. **Graph partitioning**: Minimize edge cuts using min-cut, spectral, or METIS algorithms
3. **Cut insertion**: Replace cut gates with measurement/preparation operators
4. **Classical stitching**: Sample subcircuits and reconstruct full distribution

**Sampling Overhead**

For k cuts, the quasi-probability decomposition requires sampling :math:`\gamma \geq 4^k` configurations. More precisely:

.. math::

   \gamma(k) = \prod_{i=1}^{k} \gamma_i

where :math:`\gamma_i` is the gate-specific overhead (typically 4-9 depending on gate type).

**When to Use Circuit Cutting**

- **High connectivity**: Circuit has long-range entanglement or all-to-all connectivity
- **Device constraints**: Target hardware has limited qubit count
- **MPS inefficiency**: Circuit would require χ > 10⁴ for accurate MPS simulation
- **Small cut count**: k ≤ 3 cuts (overhead manageable)

**Implementation Approach**

This module provides:

- Automated partitioning (min-cut, spectral clustering, manual)
- Quasi-probability decomposition for standard gates (CNOT, CZ, SWAP)
- Variance reduction via importance sampling
- Integration with MPS backend for subcircuit simulation

Configuration
-------------

.. class:: CuttingConfig

   Configuration for circuit cutting.

   :param int max_partition_size: Maximum qubits per partition (default 10)
   :param int max_cuts: Maximum number of cuts (default 3)
   :param str algorithm: Partitioning algorithm ('min_cut', 'spectral', 'manual')
   :param bool variance_reduction: Use variance reduction techniques
   :param int samples_per_cut: Classical samples per cut (default 100)

Data Structures
---------------

.. class:: CutPoint

   Represents a cut location in the circuit.

   :param int qubit1: First qubit index
   :param int qubit2: Second qubit index
   :param int time_step: Time step of cut
   :param int partition1: Partition containing qubit1
   :param int partition2: Partition containing qubit2

.. class:: CircuitPartition

   Represents a partition of the circuit.

   :param int partition_id: Partition identifier
   :param set qubits: Set of qubit indices in this partition
   :param list gates: List of gates (gate_type, qubits, params)
   :param list cut_points: List of CutPoint objects at partition boundary

Classes
-------

.. class:: CouplingGraph(n_qubits)

   Represents coupling/entanglement structure of a circuit.

   :param int n_qubits: Number of qubits

   **Attributes:**

   .. attribute:: edges

      Dictionary mapping (q1, q2) tuples to edge weights

   .. attribute:: adjacency

      NumPy adjacency matrix [n_qubits, n_qubits]

   **Methods:**

   .. method:: add_two_qubit_gate(qubit1, qubit2, weight=1.0)

      Add two-qubit gate to coupling graph.

      :param int qubit1: First qubit
      :param int qubit2: Second qubit
      :param float weight: Edge weight (default 1)

   .. method:: get_degree(qubit)

      Get degree (number of connections) for a qubit.

      :param int qubit: Qubit index
      :return: Degree
      :rtype: int

   .. method:: get_neighbors(qubit)

      Get neighboring qubits.

      :param int qubit: Qubit index
      :return: List of neighbor indices
      :rtype: list

   .. method:: compute_entanglement_heatmap()

      Compute entanglement heatmap from gate connectivity.

      :return: 2D heatmap array
      :rtype: np.ndarray

.. class:: CircuitCutter(config)

   Main circuit cutting implementation.

   :param CuttingConfig config: Cutting configuration

   **Methods:**

   .. method:: analyze_circuit(gates)

      Analyze circuit and build coupling graph.

      :param list gates: List of (gate_type, qubits, params)
      :return: CouplingGraph
      :rtype: CouplingGraph

   .. method:: partition_circuit(gates, n_partitions)

      Partition circuit into subcircuits.

      :param list gates: List of gates
      :param int n_partitions: Number of partitions
      :return: List of CircuitPartition objects
      :rtype: list

   .. method:: insert_cut_operators(partition)

      Insert cut-point operators at partition boundaries.

      :param CircuitPartition partition: Circuit partition
      :return: Modified gate list with cut operators
      :rtype: list

   .. method:: simulate_cut_circuit(partitions)

      Simulate cut circuit using classical stitching.

      :param list partitions: List of CircuitPartition objects
      :return: Dictionary of measurement outcomes
      :rtype: dict

   .. method:: estimate_sampling_overhead(n_cuts)

      Estimate classical sampling overhead.

      :param int n_cuts: Number of cuts
      :return: Sampling overhead factor (≈ 4^n_cuts)
      :rtype: int

Examples
--------

Basic circuit cutting:

.. code-block:: python

   from atlas_q.circuit_cutting import CircuitCutter, CuttingConfig

   # Configure cutting
   config = CuttingConfig(
       max_partition_size=8,
       max_cuts=2,
       algorithm='min_cut'
   )
   cutter = CircuitCutter(config)

   # Define circuit
   gates = [
       ('H', [0], []),
       ('H', [1], []),
       ('CNOT', [0, 1], []),
       ('CNOT', [1, 2], []),
       # ... more gates
   ]

   # Partition circuit
   partitions = cutter.partition_circuit(gates, n_partitions=2)

   # Simulate
   results = cutter.simulate_cut_circuit(partitions)
   print(f"Measurement outcomes: {results}")

Analyze circuit connectivity:

.. code-block:: python

   from atlas_q.circuit_cutting import CouplingGraph

   # Build coupling graph
   graph = CouplingGraph(n_qubits=20)
   for gate_type, qubits, _ in gates:
       if len(qubits) == 2:
           graph.add_two_qubit_gate(qubits[0], qubits[1])

   # Analyze connectivity
   heatmap = graph.compute_entanglement_heatmap()

   # Find highly connected qubits
   for i in range(20):
       degree = graph.get_degree(i)
       print(f"Qubit {i}: degree {degree}")

Manual cutting at specific locations:

.. code-block:: python

   config = CuttingConfig(algorithm='manual')
   cutter = CircuitCutter(config)

   # Manually specify cut points
   cut_points = [
       CutPoint(qubit1=5, qubit2=6, time_step=10, partition1=0, partition2=1),
       CutPoint(qubit1=10, qubit2=11, time_step=15, partition1=1, partition2=2)
   ]

   # Apply cuts
   partitions = cutter.partition_with_cuts(gates, cut_points)

Performance Notes
-----------------

**Sampling Overhead Scaling**

.. list-table:: Circuit Cutting Overhead
   :header-rows: 1
   :widths: 15 25 30 30

   * - Cuts (k)
     - Samples Required
     - Time Factor
     - Practical Use
   * - 1
     - 4¹ = 4
     - 4×
     - Excellent
   * - 2
     - 4² = 16
     - 16×
     - Good
   * - 3
     - 4³ = 64
     - 64×
     - Acceptable
   * - 4
     - 4⁴ = 256
     - 256×
     - Marginal
   * - 5
     - 4⁵ = 1024
     - 1024×
     - Impractical

**Recommendation**: Limit k ≤ 3 for production simulations.

**Gate-Specific Overhead**

Different gates have different cutting costs γ:

- **CNOT**: γ = 4 (Pauli decomposition)
- **CZ**: γ = 4 (same as CNOT)
- **SWAP**: γ = 9 (higher overhead)
- **Toffoli**: γ = 16 (very expensive to cut)

**Partitioning Quality**

Good partitions minimize cuts while balancing partition sizes:

- **Min-cut**: Optimal for 2 partitions, O(n³) time
- **Spectral**: Good for k > 2 partitions, O(n² log n)
- **METIS**: Fast heuristic, O(n log n), near-optimal
- **Manual**: Full control, requires domain knowledge

**Memory Usage**

Each partition with n qubits and χ bond dimension requires:

- **MPS storage**: O(nχ²) per partition
- **Classical samples**: O(4^k × batch_size) outcome storage
- **Reconstruction**: O(4^k) coefficient storage

Total memory dominated by classical sampling for k ≥ 3.

**Accuracy vs. Samples**

Statistical error decreases as :math:`1/\sqrt{N_{\text{samples}}}`:

- 100 samples per config: ~10% error
- 1000 samples: ~3% error
- 10000 samples: ~1% error

With k cuts, total samples = samples_per_config × 4^k.

Best Practices
--------------

**Partitioning Strategies**

1. **Analyze connectivity first**: Use ``CouplingGraph`` to identify bottlenecks
2. **Start with min-cut**: Try automatic partitioning before manual cuts
3. **Balance partition sizes**: Uneven partitions waste resources
4. **Cut early in circuit**: Early cuts reduce downstream entanglement

**Minimizing Overhead**

- Use **gate commutation** to move cuts to less entangling regions
- Apply **circuit optimization** (gate cancellation, synthesis) before cutting
- **Batch simulations** to amortize setup costs
- Enable **variance reduction** for faster convergence

**Error Management**

.. code-block:: python

   # Monitor reconstruction error
   results = cutter.simulate_cut_circuit(partitions, samples=1000)

   # Estimate standard error
   std_error = np.std(results['samples']) / np.sqrt(len(results['samples']))
   print(f"Standard error: {std_error:.4f}")

   # Increase samples if error too large
   if std_error > 0.01:
       results = cutter.simulate_cut_circuit(partitions, samples=10000)

**Debugging Cut Circuits**

.. code-block:: python

   # Verify partitioning
   for i, partition in enumerate(partitions):
       print(f"Partition {i}: {len(partition.qubits)} qubits, {len(partition.gates)} gates")
       print(f"  Cut points: {len(partition.cut_points)}")

   # Check overhead estimate
   overhead = cutter.estimate_sampling_overhead(n_cuts=len(cut_points))
   print(f"Estimated sampling overhead: {overhead}×")

   # Test on small example first
   if overhead > 100:
       print("Warning: High overhead, consider reducing cuts")

Limitations
-----------

**Fundamental Limits**

- Exponential scaling in number of cuts (unavoidable)
- Negative quasi-probabilities cause high variance
- Some gates (Toffoli) very expensive to cut

**Practical Constraints**

- k > 5 cuts: sampling overhead prohibitive (>1000×)
- High-degree graphs: may require many cuts
- Deep circuits: accumulate more cuts

**When NOT to Use**

- Circuit already simulatable with standard MPS (χ < 100)
- Too many cuts required (k > 5)
- Need exact results (cutting introduces statistical error)
- Circuit has natural tensor network structure (use PEPS instead)

Use Cases
---------

**Ideal Applications**

1. **All-to-all connectivity**: QAOA on complete graphs, VQE for molecular systems
2. **Hardware compilation**: Adapt circuits to device topology
3. **Large-scale benchmarking**: Simulate beyond single-GPU capacity
4. **Hybrid algorithms**: Combine classical and quantum resources

**Research Applications**

- Quantum advantage experiments (Google, IBM circuits)
- Circuit optimization and synthesis
- Fault-tolerant circuit decomposition
- Distributed quantum computing protocols

See Also
--------

- :doc:`../user_guide/howtos/handle_large_systems` - Scaling strategies
- :doc:`../user_guide/explanations/tensor_networks` - Theoretical foundations
- :doc:`distributed_mps` - Multi-GPU alternative to cutting
- :doc:`planar_2d` - 2D circuit layout optimization
- :doc:`vqe_qaoa` - Variational algorithms with cutting

References
----------

.. [Peng20] T. Peng et al., "Simulating large quantum circuits on a small quantum computer," *Physical Review Letters* 125, 150504 (2020).

.. [Tang21] W. Tang et al., "Cutting quantum circuits with entanglement recovery," *Physical Review A* 103, 012603 (2021).

.. [Mitarai21] K. Mitarai & K. Fujii, "Constructing a virtual two-qubit gate from single-qubit operations," *Physical Review Research* 3, 013129 (2021).

.. [Lowe22] A. Lowe et al., "Fast quantum circuit cutting with randomized measurements," *Quantum* 7, 934 (2023).
