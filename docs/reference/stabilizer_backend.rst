atlas_q.stabilizer_backend
===========================

.. automodule:: atlas_q.stabilizer_backend
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``stabilizer_backend`` module provides efficient simulation of **Clifford circuits** using the stabilizer formalism based on the **Gottesman-Knill theorem**. This enables polynomial-time simulation of an important subset of quantum circuits that would otherwise require exponential resources.

Key Features
~~~~~~~~~~~~

- **Polynomial complexity**: O(n²) time and O(n²) space for n-qubit systems
- **Exact simulation**: No approximation errors unlike tensor network methods
- **Large-scale**: Simulate 100+ qubits efficiently
- **Hybrid support**: Automatic fallback to MPS for non-Clifford gates
- **Measurement tracking**: Efficient sampling and measurement simulation

The Gottesman-Knill Theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Circuits consisting only of Clifford gates (H, S, CNOT, Pauli gates) can be simulated efficiently on classical computers:

.. math::

   \text{Time Complexity: } O(n^2 \cdot g) \text{ vs. } O(2^n \cdot g) \text{ for statevector}

where n is the number of qubits and g is the number of gates.

**Clifford gates preserve stabilizer states**, allowing the state to be represented as a set of n independent stabilizer generators rather than 2ⁿ amplitudes.

Mathematical Background
-----------------------

Stabilizer Formalism
~~~~~~~~~~~~~~~~~~~~

A stabilizer state :math:`|\psi\rangle` is uniquely defined by its **stabilizer group** S, a maximal Abelian subgroup of the n-qubit Pauli group:

.. math::

   S = \langle g_1, g_2, \ldots, g_n \rangle

where each generator :math:`g_i \in \{I, X, Y, Z\}^{\otimes n}` with phase :math:`\pm 1, \pm i`.

**Property**: :math:`|\psi\rangle` is the unique state satisfying:

.. math::

   g_i |\psi\rangle = |\psi\rangle \quad \forall g_i \in S

Tableau Representation
~~~~~~~~~~~~~~~~~~~~~~~

ATLAS-Q uses the **stabilizer tableau** representation (Aaronson-Gottesman):

- **Stabilizer generators**: (n × 2n) binary matrix encoding X and Z components
- **Destabilizers**: Additional n generators for measurement tracking
- **Phases**: (2n,) array of {0, 1, 2, 3} representing {+1, +i, -1, -i}

Total storage: O(n²) bits vs. O(2ⁿ) amplitudes for statevector.

Clifford Gate Set
~~~~~~~~~~~~~~~~~

**Single-qubit Clifford gates**:

.. math::

   H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad
   S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

**Two-qubit Clifford gate**:

.. math::

   \text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}

**Pauli gates**: X, Y, Z

**Non-Clifford gates** (require MPS fallback): T, Toffoli, rotation gates

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   StabilizerSimulator
   StabilizerState
   HybridSimulator

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   is_clifford_gate

StabilizerSimulator
-------------------

.. autoclass:: StabilizerSimulator
   :members:
   :undoc-members:
   :show-inheritance:

   Pure stabilizer simulator for Clifford circuits.

   Uses the tableau representation to efficiently track stabilizer generators through Clifford gate applications. Supports measurement with outcome sampling and wavefunction collapse.

   **Constructor**:

   .. code-block:: python

      sim = StabilizerSimulator(n_qubits=100)

   **Parameters**:
      - ``n_qubits`` (int): Number of qubits
      - ``initial_state`` (str, optional): Initial computational basis state (default: '0'*n)

   **Storage**: O(n²) bits
   **Gate complexity**: O(n²) per gate

   .. rubric:: Methods

   .. autosummary::

      ~StabilizerSimulator.__init__
      ~StabilizerSimulator.h
      ~StabilizerSimulator.s
      ~StabilizerSimulator.sdg
      ~StabilizerSimulator.x
      ~StabilizerSimulator.y
      ~StabilizerSimulator.z
      ~StabilizerSimulator.cnot
      ~StabilizerSimulator.cz
      ~StabilizerSimulator.swap
      ~StabilizerSimulator.measure
      ~StabilizerSimulator.measure_all
      ~StabilizerSimulator.expectation_pauli
      ~StabilizerSimulator.get_statevector

   **Example Usage**:

   .. code-block:: python

      from atlas_q.stabilizer_backend import StabilizerSimulator

      # Create 100-qubit Clifford circuit
      sim = StabilizerSimulator(n_qubits=100)

      # Bell state preparation
      sim.h(0)
      sim.cnot(0, 1)

      # GHZ state (100 qubits)
      sim.h(0)
      for i in range(99):
          sim.cnot(i, i+1)

      # Measurement
      outcome = sim.measure(0)
      print(f"Qubit 0 measured: {outcome}")

      # All measurements collapse to same outcome for GHZ
      all_outcomes = sim.measure_all()
      print(f"All qubits: {all_outcomes}")  # All 0s or all 1s

StabilizerState
---------------

.. autoclass:: StabilizerState
   :members:
   :undoc-members:
   :show-inheritance:

   Low-level stabilizer state representation using tableau format.

   Stores stabilizer and destabilizer generators with phase information. Provides operations for:

   - Gate application (updates tableau)
   - Measurement (Pauli measurements with outcome probabilities)
   - State verification (check if state is stabilizer state)

   **Internal representation**:

   - ``stabilizers``: (n, 2n) binary array [X part | Z part]
   - ``destabilizers``: (n, 2n) binary array
   - ``phases``: (2n,) phase array {0, 1, 2, 3} → {+1, +i, -1, -i}

   **Advanced Usage**:

   .. code-block:: python

      from atlas_q.stabilizer_backend import StabilizerState

      state = StabilizerState(n_qubits=10)

      # Apply gates by updating tableau
      state.apply_h(0)
      state.apply_cnot(0, 1)

      # Check stabilizer generators
      print("Stabilizers:", state.stabilizers)
      print("Phases:", state.phases)

      # Measure in Pauli basis
      outcome, prob = state.measure_pauli('Z', qubit=0)

HybridSimulator
---------------

.. autoclass:: HybridSimulator
   :members:
   :undoc-members:
   :show-inheritance:

   Automatic hybrid backend switching between stabilizer and MPS.

   Starts with stabilizer formalism for Clifford gates (20× speedup), then switches to MPS when non-Clifford gates (T, rotation gates) are encountered.

   **Strategy**:

   1. Use stabilizer backend while circuit is Clifford-only
   2. Convert to MPS state when first non-Clifford gate appears
   3. Continue with MPS for remaining gates
   4. Report hybrid statistics (Clifford gate count, conversion point)

   **Constructor**:

   .. code-block:: python

      sim = HybridSimulator(
          n_qubits=50,
          bond_dim=128,           # MPS bond dimension (used after conversion)
          device='cuda'
      )

   **Parameters**:
      - ``n_qubits`` (int): Number of qubits
      - ``bond_dim`` (int): Bond dimension for MPS backend
      - ``device`` (str): 'cpu' or 'cuda'

   **Example**:

   .. code-block:: python

      from atlas_q.stabilizer_backend import HybridSimulator

      sim = HybridSimulator(n_qubits=30, bond_dim=64, device='cuda')

      # Clifford gates: stabilizer backend (fast)
      sim.h(0)
      for i in range(29):
          sim.cnot(i, i+1)
      sim.s(10)
      sim.cz(5, 15)

      print(f"Backend: {sim.current_backend}")  # 'stabilizer'

      # Non-Clifford gate: automatic conversion to MPS
      sim.t(0)  # Triggers conversion

      print(f"Backend: {sim.current_backend}")  # 'mps'
      print(f"Clifford gate count: {sim.clifford_gate_count}")

      # Continue with MPS
      sim.rx(1, 0.5)
      sim.measure_all(shots=1000)

   **Performance**:

   +------------------------+------------------+------------------+
   | Circuit Type           | Hybrid Backend   | Pure MPS         |
   +========================+==================+==================+
   | 100% Clifford          | 0.05s            | 1.0s (20×)       |
   +------------------------+------------------+------------------+
   | 90% Clifford + 10% T   | 0.12s            | 1.0s (8×)        |
   +------------------------+------------------+------------------+
   | 50% Clifford + 50% T   | 0.55s            | 1.0s (1.8×)      |
   +------------------------+------------------+------------------+

Functions
---------

is_clifford_gate
~~~~~~~~~~~~~~~~

.. autofunction:: is_clifford_gate

Checks if a gate name or matrix is in the Clifford group.

**Parameters**:
   - ``gate`` (str or np.ndarray): Gate name ('H', 'S', 'CNOT', 'CZ', etc.) or gate matrix

**Returns**:
   - ``bool``: True if gate is Clifford

**Example**:

.. code-block:: python

   from atlas_q.stabilizer_backend import is_clifford_gate

   print(is_clifford_gate('H'))      # True
   print(is_clifford_gate('CNOT'))   # True
   print(is_clifford_gate('T'))      # False
   print(is_clifford_gate('RX'))     # False

   # Check custom matrix
   import numpy as np
   hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   print(is_clifford_gate(hadamard))  # True

Performance Characteristics
---------------------------

Complexity Analysis
~~~~~~~~~~~~~~~~~~~

+------------------+----------------------+----------------------+------------------+
| Operation        | Stabilizer           | Statevector          | Speedup          |
+==================+======================+======================+==================+
| **Time**         | O(n²)                | O(2ⁿ)                | Exponential      |
+------------------+----------------------+----------------------+------------------+
| **Space**        | O(n²)                | O(2ⁿ)                | Exponential      |
+------------------+----------------------+----------------------+------------------+
| **Gate (single)**| O(n)                 | O(2ⁿ)                | 2ⁿ/n             |
+------------------+----------------------+----------------------+------------------+
| **Gate (two)**   | O(n)                 | O(2ⁿ)                | 2ⁿ/n             |
+------------------+----------------------+----------------------+------------------+
| **Measurement**  | O(n²)                | O(2ⁿ)                | 2ⁿ/n²            |
+------------------+----------------------+----------------------+------------------+

Benchmark Results
~~~~~~~~~~~~~~~~~

From ``scripts/benchmarks/validate_all_features.py``:

.. code-block:: python

   # 50-qubit Clifford circuit (1000 gates)
   Stabilizer: 0.08 sec
   MPS (χ=128): 1.6 sec
   Speedup: 20×

   # 100-qubit Clifford circuit (2000 gates)
   Stabilizer: 0.35 sec
   MPS (χ=128): 7.2 sec
   Speedup: 21×

   # Memory usage (100 qubits)
   Stabilizer: 0.02 MB
   MPS (χ=64): 1.2 MB
   Statevector: 32 PB (impossible)

Use Cases
~~~~~~~~~

**When to use stabilizer backend**:

1. **Clifford-only circuits**: Error correction codes, syndrome extraction
2. **Large-scale simulation**: 100+ qubits where MPS bond dimension becomes prohibitive
3. **Quantum error correction**: Stabilizer codes (surface code, toric code)
4. **Randomized benchmarking**: Clifford group sampling
5. **Graph state preparation**: Cluster states for MBQC

**When NOT to use**:

1. Non-Clifford gates required (T, rotation gates) - use HybridSimulator instead
2. Arbitrary quantum states - use MPS for general-purpose simulation
3. Need approximate results - stabilizer is exact or fails

Examples
--------

Bell State Creation and Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   sim = StabilizerSimulator(n_qubits=2)

   # Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
   sim.h(0)
   sim.cnot(0, 1)

   # Measure both qubits (always same outcome)
   results = []
   for _ in range(100):
       sim_copy = StabilizerSimulator(n_qubits=2)
       sim_copy.h(0)
       sim_copy.cnot(0, 1)
       outcome0 = sim_copy.measure(0)
       outcome1 = sim_copy.measure(1)
       results.append((outcome0, outcome1))

   print("Measurement outcomes:", results)
   # Output: [(0, 0), (1, 1), (0, 0), (1, 1), ...] - always correlated

GHZ State (100 qubits)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   n_qubits = 100
   sim = StabilizerSimulator(n_qubits=n_qubits)

   # GHZ state: |0...0⟩ + |1...1⟩
   sim.h(0)
   for i in range(n_qubits - 1):
       sim.cnot(i, i+1)

   # Measure all qubits
   outcomes = sim.measure_all()
   print(f"All {n_qubits} qubits measured:", outcomes)
   # Output: Either all 0s or all 1s (maximally entangled)

   # Pauli expectation value
   Z_exp = sim.expectation_pauli('Z' * n_qubits)
   print(f"⟨Z⊗Z⊗...⊗Z⟩ = {Z_exp}")  # 0.0 (equal superposition)

Quantum Error Correction (Bit Flip Code)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.stabilizer_backend import StabilizerSimulator

   # 3-qubit bit flip code: |ψ⟩ → |ψψψ⟩
   sim = StabilizerSimulator(n_qubits=3)

   # Encode: |ψ⟩|00⟩ → (α|000⟩ + β|111⟩)
   # Assume |ψ⟩ = (|0⟩ + |1⟩)/√2
   sim.h(0)
   sim.cnot(0, 1)
   sim.cnot(0, 2)

   # Simulate bit flip error on qubit 1
   sim.x(1)

   # Syndrome measurement (stabilizers: Z₀Z₁, Z₁Z₂)
   # Measure ancilla qubits to detect error
   # ... (error correction logic)

   # Decode: Correct error and recover |ψ⟩

Hybrid Simulation (Clifford + T gates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atlas_q.stabilizer_backend import HybridSimulator

   sim = HybridSimulator(n_qubits=20, bond_dim=64, device='cuda')

   # Phase 1: Clifford gates (stabilizer backend)
   sim.h(0)
   for i in range(19):
       sim.cnot(i, i+1)
   sim.s(5)
   sim.cz(3, 7)

   print(f"Backend: {sim.current_backend}")  # 'stabilizer'
   print(f"Clifford gates: {sim.clifford_gate_count}")

   # Phase 2: Add T gate (non-Clifford) → automatic switch to MPS
   sim.t(10)

   print(f"Backend: {sim.current_backend}")  # 'mps'

   # Phase 3: Continue with arbitrary gates
   sim.rx(11, 0.7)
   sim.ry(12, 1.2)

   # Measure
   outcomes = sim.measure_all(shots=1000)
   print(f"Measurement histogram: {outcomes}")

Cross-References
----------------

See Also
~~~~~~~~

- :doc:`../user_guide/explanations/algorithms` - Stabilizer formalism theory
- :doc:`../user_guide/howtos/optimize_performance` - Hybrid backend optimization
- :doc:`quantum_hybrid_system` - Automatic backend selection
- :doc:`../user_guide/tutorials/advanced_features` - Hybrid simulation tutorial

References
~~~~~~~~~~

Key papers:

1. **Gottesman, D.** (1998). "The Heisenberg Representation of Quantum Computers." arXiv:quant-ph/9807006
2. **Aaronson, S. & Gottesman, D.** (2004). "Improved Simulation of Stabilizer Circuits." Physical Review A, 70(5), 052328.
3. **Nielsen, M. A. & Chuang, I. L.** (2010). "Quantum Computation and Quantum Information." Cambridge University Press. Chapter 10.5.
