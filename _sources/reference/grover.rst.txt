atlas_q.grover
===============

.. automodule:: atlas_q.grover
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``grover`` module implements Grover's quantum search algorithm for unstructured database search with quadratic speedup. This algorithm finds marked items in a search space of size :math:`N` using :math:`O(\sqrt{N})` queries, compared to :math:`O(N)` classically.

The implementation integrates with ATLAS-Q's Matrix Product State (MPS) backend for scalable simulation of larger systems.

Mathematical Background
----------------------

Grover's algorithm searches for marked items in an unstructured database using amplitude amplification:

1. **Initialize** to uniform superposition :math:`|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}`
2. **Repeat** :math:`k^* = \lfloor\frac{\pi}{4}\sqrt{N/M}\rfloor` times:

   a. Apply oracle :math:`O_f|x\rangle = (-1)^{f(x)}|x\rangle` (marks target states)
   b. Apply diffusion operator :math:`D = 2|s\rangle\langle s| - I` (amplifies marked states)

3. **Measure** to find marked item with high probability (:math:`\gtrsim 95\%`)

Where:
  - :math:`N = 2^n` is the search space size
  - :math:`M` is the number of marked items
  - :math:`k^*` is the optimal number of iterations

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GroverSearch
   GroverConfig
   OracleBase
   FunctionOracle
   BitmapOracle
   DiffusionOperator

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   grover_search
   calculate_grover_iterations

GroverSearch
------------

.. autoclass:: GroverSearch
   :members:
   :undoc-members:
   :show-inheritance:

   Main implementation of Grover's quantum search algorithm.

   The algorithm amplifies the probability amplitude of marked states through repeated
   application of the oracle and diffusion operators.

   .. rubric:: Constructor

   .. code-block:: python

      grover = GroverSearch(oracle, config)

   **Parameters:**
      - ``oracle`` (OracleBase): Oracle that marks target states
      - ``config`` (GroverConfig): Algorithm configuration

   .. rubric:: Methods

   .. autosummary::

      ~GroverSearch.__init__
      ~GroverSearch.run
      ~GroverSearch.plot_convergence

   .. rubric:: Example

   .. code-block:: python

      from atlas_q.grover import GroverSearch, GroverConfig, BitmapOracle

      # Configure algorithm
      config = GroverConfig(n_qubits=4, device='cpu')

      # Create oracle marking state 7
      oracle = BitmapOracle(4, {7}, device='cpu')

      # Run search
      grover = GroverSearch(oracle, config)
      result = grover.run()

      print(f"Found state: {result['measured_state']}")
      print(f"Success probability: {result['success_probability']:.3f}")

GroverConfig
------------

.. autoclass:: GroverConfig
   :members:
   :undoc-members:

   Configuration for Grover's algorithm.

   .. attribute:: n_qubits

      Number of qubits (search space size = :math:`2^{\\text{n_qubits}}`).

   .. attribute:: oracle_type

      Oracle type: ``'function'`` or ``'bitmap'`` (default: ``'function'``).

   .. attribute:: auto_iterations

      Automatically calculate optimal iterations (default: ``True``).

   .. attribute:: max_iterations

      Maximum iterations safety limit (default: 1000).

   .. attribute:: chi_max

      Maximum MPS bond dimension (default: 256).

   .. attribute:: device

      Compute device: ``'cuda'`` or ``'cpu'`` (default: ``'cuda'``).

   .. attribute:: dtype

      Data type: ``torch.complex64`` or ``torch.complex128`` (default: ``torch.complex128``).

   .. attribute:: verbose

      Print progress information (default: ``False``).

   .. attribute:: measure_success_prob

      Track success probability per iteration (default: ``False``).

OracleBase
----------

.. autoclass:: OracleBase
   :members:
   :undoc-members:
   :show-inheritance:

   Abstract base class for quantum oracles.

   Oracles mark target states by applying a phase flip:

   .. math::

      O_f|x\rangle = (-1)^{f(x)}|x\rangle

   .. rubric:: Methods

   .. autosummary::

      ~OracleBase.apply
      ~OracleBase.get_marked_count

FunctionOracle
--------------

.. autoclass:: FunctionOracle
   :members:
   :undoc-members:
   :show-inheritance:

   Oracle based on a marking function :math:`f: \\{0,1\\}^n \\to \\{0,1\\}`.

   Marks states where ``marking_fn(x)`` returns ``True``.

   .. rubric:: Constructor

   .. code-block:: python

      oracle = FunctionOracle(n_qubits, marking_fn, device='cpu')

   **Parameters:**
      - ``n_qubits`` (int): Number of qubits
      - ``marking_fn`` (callable): Function that returns ``True`` for marked states
      - ``device`` (str): Compute device
      - ``dtype`` (torch.dtype): Data type

   .. rubric:: Example

   .. code-block:: python

      from atlas_q.grover import FunctionOracle

      # Mark even numbers
      oracle = FunctionOracle(
          n_qubits=4,
          marking_fn=lambda x: x % 2 == 0,
          device='cpu'
      )

      # Mark powers of two
      is_power_of_two = lambda x: x > 0 and (x & (x - 1)) == 0
      oracle = FunctionOracle(4, is_power_of_two)

BitmapOracle
------------

.. autoclass:: BitmapOracle
   :members:
   :undoc-members:
   :show-inheritance:

   Oracle based on explicit bitmap of marked states.

   More efficient than ``FunctionOracle`` for small marked sets.

   .. rubric:: Constructor

   .. code-block:: python

      oracle = BitmapOracle(n_qubits, marked_states, device='cpu')

   **Parameters:**
      - ``n_qubits`` (int): Number of qubits
      - ``marked_states`` (Set[int]): Set of marked state indices
      - ``device`` (str): Compute device
      - ``dtype`` (torch.dtype): Data type

   .. rubric:: Example

   .. code-block:: python

      from atlas_q.grover import BitmapOracle

      # Mark states 3, 7, and 11
      oracle = BitmapOracle(
          n_qubits=4,
          marked_states={3, 7, 11},
          device='cpu'
      )

DiffusionOperator
-----------------

.. autoclass:: DiffusionOperator
   :members:
   :undoc-members:
   :show-inheritance:

   Grover diffusion operator (inversion about average).

   Implements :math:`D = 2|s\rangle\langle s| - I` where :math:`|s\rangle = H^{\otimes n}|0\rangle^{\otimes n}`.

   Mathematical form:

   .. math::

      D = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}

   This reflects amplitudes about their average, amplifying marked states.

Convenience Functions
--------------------

grover_search
~~~~~~~~~~~~~

.. autofunction:: grover_search

   Convenience function for quick Grover searches.

   **Parameters:**
      - ``n_qubits`` (int): Number of qubits
      - ``marked_states`` (Set[int], List[int], or callable): Marked states or marking function
      - ``iterations`` (Optional[int]): Number of iterations (``None`` = auto)
      - ``device`` (str): ``'cuda'`` or ``'cpu'`` (default: ``'cuda'``)
      - ``verbose`` (bool): Print progress (default: ``False``)

   **Returns:**
      Dictionary with search results

   .. rubric:: Example

   .. code-block:: python

      from atlas_q.grover import grover_search

      # Search for state 7 in 4-qubit space
      result = grover_search(n_qubits=4, marked_states={7}, device='cpu')
      print(f"Found: {result['measured_state']}")

      # Search using function
      result = grover_search(
          n_qubits=5,
          marked_states=lambda x: x % 3 == 0,  # Multiples of 3
          device='cpu'
      )

calculate_grover_iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_grover_iterations

   Calculate optimal number of Grover iterations.

   Formula: :math:`k^* = \lfloor\frac{\pi}{4}\sqrt{N/M}\rfloor`

   **Parameters:**
      - ``n_qubits`` (int): Number of qubits
      - ``n_marked`` (int): Number of marked items

   **Returns:**
      Optimal iteration count (int)

   .. rubric:: Example

   .. code-block:: python

      from atlas_q.grover import calculate_grover_iterations

      # For N=16, M=1
      k = calculate_grover_iterations(n_qubits=4, n_marked=1)
      print(f"Optimal iterations: {k}")  # Output: 3

      # For N=1024, M=4
      k = calculate_grover_iterations(n_qubits=10, n_marked=4)
      print(f"Optimal iterations: {k}")  # Output: 12

Performance Considerations
--------------------------

**Scaling:**
  - Classical complexity: :math:`O(N)` where :math:`N = 2^n`
  - Quantum complexity: :math:`O(\sqrt{N/M})` iterations
  - Each iteration: :math:`O(n \cdot \chi^2)` where :math:`\chi` is bond dimension

**Bond Dimension:**
  Higher ``chi_max`` improves accuracy but increases computation time. Recommended values:

  - Small systems (≤5 qubits): ``chi_max=32``
  - Medium systems (6-8 qubits): ``chi_max=64-128``
  - Large systems (≥9 qubits): ``chi_max=256-512``

**Device Selection:**
  - CPU: Better for small systems (≤6 qubits) or debugging
  - CUDA: Significantly faster for larger systems (≥7 qubits)

**Oracle Choice:**
  - ``BitmapOracle``: Best for small explicit marked sets
  - ``FunctionOracle``: Best for pattern-based marking (e.g., "all even numbers")

Implementation Details
----------------------

**MPO-Based Oracles:**
  The oracle implementation uses exact Matrix Product Operators (MPOs) to represent phase-flip
  operators. This provides:

  - Exact oracle implementation (no approximations)
  - Bond dimension 2 for single-state oracles
  - Efficient contraction with MPS via ``apply_mpo_to_mps``
  - 94-100% success probability matching theoretical predictions

**MPS Representation:**
  The quantum state is represented using Matrix Product States with controlled entanglement:

  - Accuracy controlled by ``chi_max`` bond dimension parameter
  - Recommended: ``chi_max=64-128`` for 4-6 qubits, ``chi_max=256`` for 7-8 qubits
  - Higher ``chi_max`` improves accuracy at cost of performance
  - For critical applications, run multiple searches and take majority vote

**Performance Characteristics:**
  - 2-4 qubit systems: ~10-15 ms/iteration (CPU)
  - 5-7 qubit systems: ~20-25 ms/iteration (CPU)
  - GPU acceleration available for larger systems (≥7 qubits)

Best Practices
--------------

**Optimal Iteration Count**

For N items with M solutions, optimal iterations ≈ π/4 × √(N/M):

- Single solution: ~0.785√N iterations
- Monitor success probability to avoid over-rotation
- Can empirically tune by testing different iteration counts

**Oracle Design**

- Keep oracle simple (few gates) to minimize circuit depth
- Use phase kickback for efficient implementation
- Test oracle independently before full Grover circuit

**Performance Optimization**

1. Use χ=16-32 for up to 10 qubits (sufficient for Grover)
2. Enable GPU for n ≥ 8 qubits
3. Batch multiple searches if testing different oracles
4. Use stabilizer backend for Clifford-only oracles (faster)

Use Cases
---------

**Educational Applications**

- Teaching quantum algorithms and amplitude amplification
- Demonstrating quantum speedup vs. classical search
- Understanding oracle-based quantum algorithms

**Research Applications**

- Benchmarking quantum simulators
- Studying entanglement in search algorithms
- Developing improved search variants
- Testing new oracle implementations

**Practical Considerations**

- For small N (< 100): Classical search faster
- For large N: Physical quantum computers needed for advantage
- Simulator useful for: algorithm development, education, benchmarking

See Also
--------

- :doc:`adaptive_mps` - MPS backend used by Grover's algorithm
- :doc:`vqe_qaoa` - Other variational quantum algorithms
- :doc:`quantum_hybrid_system` - Period-finding and Shor's algorithm
- :doc:`stabilizer_backend` - Clifford circuit simulator
- :doc:`../user_guide/tutorials/beginners` - Grover algorithm tutorial

References
----------

.. [Grover96] L. K. Grover, "A fast quantum mechanical algorithm for database search,"
   Proceedings of the 28th Annual ACM Symposium on Theory of Computing (1996).
   https://arxiv.org/abs/quant-ph/9605043

.. [Nielsen00] M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
   Cambridge University Press (2000), Section 6.1.

.. [Brassard02] G. Brassard, P. Høyer, M. Mosca, and A. Tapp, "Quantum Amplitude Amplification
   and Estimation," Contemporary Mathematics, 305:53-74 (2002).
   https://arxiv.org/abs/quant-ph/0005055
