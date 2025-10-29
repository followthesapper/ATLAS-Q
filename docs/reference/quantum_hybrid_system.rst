atlas_q.quantum_hybrid_system
==============================

.. automodule:: atlas_q.quantum_hybrid_system
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``quantum_hybrid_system`` module implements compressed quantum state representations and period-finding algorithms for Shor's factorization algorithm. Key features include:

- Memory-efficient quantum state representations (O(1) to O(n) memory)
- Analytic QFT for periodic states
- Hybrid classical-quantum period-finding
- Fast modular exponentiation on GPU
- Tensor network support for moderate entanglement

This module enables factorization of large numbers by leveraging the structure of periodic quantum states without requiring exponential memory.

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CompressedQuantumState
   PeriodicState
   ProductState
   MatrixProductState
   PeriodResult
   PeriodFinder
   QuantumClassicalHybrid
   QuantumCircuit
   GPUAccelerator

Quantum State Representations
------------------------------

CompressedQuantumState
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CompressedQuantumState
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for memory-efficient quantum state representations. Provides interface for:

   - Amplitude queries for specific basis states
   - Probability computations
   - Measurement sampling (with automatic MPS fallback for large systems)

PeriodicState
^^^^^^^^^^^^^

.. autoclass:: PeriodicState
   :members:
   :undoc-members:
   :show-inheritance:

   O(1) memory representation of periodic quantum states. Perfect for Shor's algorithm.

   Represents states of the form:

   .. math::

      |\psi\rangle = \frac{1}{\sqrt{k}} \sum_{j=0}^{k-1} |\text{offset} + j \cdot \text{period}\rangle

   Features analytic QFT sampling for exact period extraction without explicit QFT computation.

ProductState
^^^^^^^^^^^^

.. autoclass:: ProductState
   :members:
   :undoc-members:
   :show-inheritance:

   O(n) memory representation for product (unentangled) states:

   .. math::

      |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes \cdots \otimes |\psi_n\rangle

   Efficient for states with no entanglement.

MatrixProductState
^^^^^^^^^^^^^^^^^^

.. autoclass:: MatrixProductState
   :members:
   :undoc-members:
   :show-inheritance:

   O(n·χ²) memory representation for moderately entangled states using tensor networks. Provides:

   - Sweep-based sampling
   - Canonical form transformations
   - SVD-based compression

Period-Finding
--------------

PeriodResult
^^^^^^^^^^^^

.. autoclass:: PeriodResult
   :members:
   :undoc-members:
   :show-inheritance:

   Dataclass storing period-finding results:

   - Detected period (r)
   - Confidence score
   - Execution time
   - Success flag

PeriodFinder
^^^^^^^^^^^^

.. autoclass:: PeriodFinder
   :members:
   :undoc-members:
   :show-inheritance:

   Core period-finding algorithm using O(√r) complexity hybrid approach:

   1. Classical preprocessing for small periods
   2. Quantum period detection for medium periods
   3. Continued fractions for period extraction

QuantumClassicalHybrid
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QuantumClassicalHybrid
   :members:
   :undoc-members:
   :show-inheritance:

   Complete factorization system combining:

   - Period-finding via quantum simulation
   - Classical GCD for factor extraction
   - Trial division for small factors
   - Automatic fallback strategies

   Primary interface for factoring integers.

Supporting Classes
------------------

QuantumCircuit
^^^^^^^^^^^^^^

.. autoclass:: QuantumCircuit
   :members:
   :undoc-members:
   :show-inheritance:

   Quantum circuit builder for period-finding circuits. Supports:

   - Gate application (H, CNOT, phase gates)
   - Modular exponentiation circuits
   - QFT implementation

GPUAccelerator
^^^^^^^^^^^^^^

.. autoclass:: GPUAccelerator
   :members:
   :undoc-members:
   :show-inheritance:

   GPU-accelerated modular exponentiation and tensor operations. Provides 100-1000× speedup over CPU for period-finding.

Examples
--------

Basic factorization:

.. code-block:: python

   from atlas_q import get_quantum_sim

   # Get factorization system
   QCH, _, _, _ = get_quantum_sim()

   # Factor a number
   sim = QCH()
   factors = sim.factor_number(221)

   print(f"221 = {factors[0]} × {factors[1]}")  # Output: 221 = 13 × 17

Using PeriodicState:

.. code-block:: python

   from atlas_q.quantum_hybrid_system import PeriodicState

   # Create periodic state with period=7, offset=3, in 20-qubit space
   state = PeriodicState(
       num_qubits=20,
       period=7,
       offset=3
   )

   # Query amplitude (O(1) operation, no memory usage)
   amp = state.get_amplitude(10)  # |10⟩
   prob = state.get_probability(10)

   print(f"Amplitude: {amp}")
   print(f"Probability: {prob}")

   # Sample measurements (uses analytic QFT)
   samples = state.measure(num_shots=1000)
   print(f"Sample: {samples[0]} (binary: {bin(samples[0])})")

Direct period-finding:

.. code-block:: python

   from atlas_q.quantum_hybrid_system import PeriodFinder

   finder = PeriodFinder()

   # Find period of a^x mod N
   result = finder.find_period(a=7, N=15, num_qubits=8)

   if result.success:
       print(f"Period: {result.period}")
       print(f"Confidence: {result.confidence:.2%}")
       print(f"Time: {result.time_sec:.3f}s")

Using MPS for entangled states:

.. code-block:: python

   from atlas_q.quantum_hybrid_system import MatrixProductState
   import numpy as np

   # Create MPS with bond dimension 16
   mps = MatrixProductState(num_qubits=30, bond_dim=16)

   # Apply gates to create entanglement
   # (This is a simplified example; in practice use AdaptiveMPS)

   # Sample from the state
   samples = mps.measure(num_shots=100)
   print(f"Sampled states: {samples[:10]}")

GPU-accelerated modular exponentiation:

.. code-block:: python

   from atlas_q.quantum_hybrid_system import GPUAccelerator

   gpu = GPUAccelerator()

   # Compute a^x mod N on GPU
   a, N = 7, 221
   x_values = list(range(100))

   results = gpu.batch_modular_exp(a, x_values, N)
   print(f"7^50 mod 221 = {results[50]}")

Advanced: Custom factorization with parameters:

.. code-block:: python

   from atlas_q import get_quantum_sim

   QCH, _, _, _ = get_quantum_sim()

   sim = QCH()

   # Factor with custom parameters
   N = 10403  # Product of two primes

   # Attempt factorization
   try:
       factors = sim.factor_number(N)
       if factors:
           p, q = factors
           print(f"{N} = {p} × {q}")
           assert p * q == N
   except ValueError as e:
       print(f"Factorization failed: {e}")

Performance Characteristics
---------------------------

Memory Usage
^^^^^^^^^^^^

- PeriodicState: O(1) - constant memory regardless of qubit count
- ProductState: O(n) - linear in qubit count
- MatrixProductState: O(n·χ²) - depends on bond dimension χ

Computational Complexity
^^^^^^^^^^^^^^^^^^^^^^^^

- Period-finding: O(√r) average case with quantum acceleration
- Classical preprocessing: O(log N) for trial division
- Modular exponentiation: O(log N) per operation
- GPU acceleration: 100-1000× speedup over CPU

Applications
------------

Shor's Algorithm
^^^^^^^^^^^^^^^^

The primary application is Shor's factorization algorithm:

1. Choose random a < N
2. Find period r of f(x) = a^x mod N using QuantumClassicalHybrid
3. If r is even and a^(r/2) ≠ -1 mod N, factors are gcd(a^(r/2) ± 1, N)
4. Repeat if unsuccessful

Period-Finding in Cryptography
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond factorization, period-finding has applications in:

- Discrete logarithm problem
- Order-finding in groups
- Hidden subgroup problem

Best Practices
--------------

**Parameter Selection**

- **register_size**: Use 2n+1 bits for n-bit numbers (ensures sufficient precision)
- **chi_max**: 32-64 sufficient for period-finding up to 10-12 qubits
- **samples**: 1000-10000 for reliable period detection

**Optimization**

1. Use GPU for circuits with > 15 qubits
2. Enable statistics tracking to monitor entanglement
3. Batch multiple period-finding runs for different bases
4. Cache QFT circuit if running multiple times

**Troubleshooting**

- **Wrong period detected**: Increase register_size or samples
- **Memory errors**: Reduce chi_max or circuit size
- **Slow performance**: Enable GPU, use Triton kernels

Use Cases
---------

**Educational**

- Teaching Shor's algorithm concepts
- Demonstrating quantum speedup
- Period-finding problem examples

**Research**

- Benchmarking quantum simulators
- Studying entanglement in hybrid algorithms
- Developing improved factorization methods

**Not Suitable For**

- Actual cryptographic attacks (classical algorithms faster for practical key sizes)
- Production cryptography (use battle-tested libraries)

See Also
--------

- :doc:`adaptive_mps` - Tensor network simulations for entangled states
- :doc:`grover` - Grover's search algorithm implementation
- :doc:`vqe_qaoa` - Other quantum algorithms
- :doc:`../user_guide/tutorials/beginners` - Tutorial including period-finding examples
- :doc:`../user_guide/explanations/algorithms` - Algorithm details

References
----------

.. [Shor94] P. W. Shor, "Algorithms for quantum computation: Discrete logarithms and factoring," *FOCS 1994* (1994).

.. [Nielsen10] M. A. Nielsen & I. L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press (2010).

.. [Mosca01] M. Mosca, "The quantum order finding algorithm," *arXiv:quant-ph/0110167* (2001).
