Grover's Quantum Search
=======================

This guide introduces Grover's quantum search algorithm and shows how to use it with ATLAS-Q for practical search problems.

Introduction
------------

Grover's algorithm provides quadratic speedup for searching unstructured databases. While classical search requires :math:`O(N)` queries on average, Grover's algorithm finds marked items in :math:`O(\sqrt{N})` queries.

**When to use Grover's algorithm:**

- Searching unstructured databases
- Finding solutions to constraint satisfaction problems
- Quantum amplitude amplification
- Pattern matching in large datasets

**Key advantages:**

- Quadratic speedup: √N vs N queries
- Works on arbitrary search criteria
- Provably optimal quantum search algorithm

Quick Start
-----------

The simplest way to use Grover's algorithm is with the convenience function:

.. code-block:: python

   from atlas_q.grover import grover_search

   # Search for state 7 in 4-qubit space (16 states)
   result = grover_search(
       n_qubits=4,
       marked_states={7},  # Mark state |0111⟩
       device='cpu'
   )

   print(f"Found state: {result['measured_state']}")
   print(f"Success probability: {result['success_probability']:.3f}")
   print(f"Iterations used: {result['iterations_used']}")

**Output:**

.. code-block:: text

   Found state: 7
   Success probability: 0.946
   Iterations used: 3

Basic Examples
--------------

Example 1: Finding a Specific Item
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for a specific database entry:

.. code-block:: python

   from atlas_q.grover import grover_search

   # Search 256-entry database (8 qubits) for entry 150
   result = grover_search(
       n_qubits=8,
       marked_states={150},
       device='cpu',
       verbose=True
   )

   print(f"\\nFound entry: {result['measured_state']}")
   print(f"Search took: {result['runtime_ms']:.2f} ms")
   print(f"Used {result['iterations_used']} iterations")

Example 2: Finding Multiple Items
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for multiple marked items simultaneously:

.. code-block:: python

   from atlas_q.grover import grover_search

   # Find any of {5, 10, 15, 20} in 5-qubit space
   result = grover_search(
       n_qubits=5,
       marked_states={5, 10, 15, 20},
       device='cpu'
   )

   print(f"Found one of the marked items: {result['measured_state']}")

**Note:** Grover finds *one* of the marked items with high probability.

Example 3: Pattern-Based Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a function to define search criteria:

.. code-block:: python

   from atlas_q.grover import grover_search

   # Find numbers divisible by 3
   def divisible_by_3(x):
       return x % 3 == 0

   result = grover_search(
       n_qubits=4,
       marked_states=divisible_by_3,  # Function oracle
       device='cpu'
   )

   print(f"Found: {result['measured_state']}")
   assert result['measured_state'] % 3 == 0

Example 4: Finding Prime Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for prime numbers in a range:

.. code-block:: python

   from atlas_q.grover import grover_search

   def is_prime(n):
       if n < 2:
           return False
       if n == 2:
           return True
       if n % 2 == 0:
           return False
       for i in range(3, int(n**0.5) + 1, 2):
           if n % i == 0:
               return False
       return True

   # Find primes in range [0, 15]
   result = grover_search(
       n_qubits=4,
       marked_states=is_prime,
       device='cpu'
   )

   print(f"Found prime: {result['measured_state']}")
   # Likely: 2, 3, 5, 7, 11, or 13

Advanced Usage
--------------

Using the Full API
~~~~~~~~~~~~~~~~~~

For more control, use the full ``GroverSearch`` class:

.. code-block:: python

   from atlas_q.grover import (
       GroverSearch, GroverConfig, BitmapOracle
   )

   # Configure algorithm
   config = GroverConfig(
       n_qubits=6,
       chi_max=128,  # MPS bond dimension
       device='cpu',
       verbose=True,
       measure_success_prob=True  # Track probabilities
   )

   # Create oracle
   oracle = BitmapOracle(
       n_qubits=6,
       marked_states={42},
       device='cpu'
   )

   # Run search
   grover = GroverSearch(oracle, config)
   result = grover.run()

   # Access detailed statistics
   for i, stat in enumerate(result['iteration_stats'], 1):
       print(f"Iteration {i}: P_success = {stat['success_probability']:.4f}")

Custom Iteration Count
~~~~~~~~~~~~~~~~~~~~~~

Override automatic iteration calculation:

.. code-block:: python

   from atlas_q.grover import (
       grover_search, calculate_grover_iterations
   )

   n_qubits = 5
   marked_states = {10, 20}

   # Calculate optimal iterations
   n_marked = len(marked_states)
   optimal_k = calculate_grover_iterations(n_qubits, n_marked)
   print(f"Optimal iterations: {optimal_k}")

   # Use custom iteration count
   result = grover_search(
       n_qubits=n_qubits,
       marked_states=marked_states,
       iterations=optimal_k + 2,  # Experiment with more iterations
       device='cpu'
   )

GPU Acceleration
~~~~~~~~~~~~~~~~

Use CUDA for faster execution on larger systems:

.. code-block:: python

   from atlas_q.grover import grover_search
   import torch

   # Check CUDA availability
   if torch.cuda.is_available():
       device = 'cuda'
       print(f"Using GPU: {torch.cuda.get_device_name()}")
   else:
       device = 'cpu'
       print("Using CPU")

   # Run on GPU (much faster for n_qubits >= 7)
   result = grover_search(
       n_qubits=10,  # 1024 states
       marked_states={512},
       device=device
   )

Practical Applications
----------------------

Application 1: Sudoku Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Grover to find Sudoku solutions:

.. code-block:: python

   from atlas_q.grover import grover_search

   def is_valid_sudoku_solution(x):
       """Check if x encodes valid 4x4 Sudoku solution"""
       # Encode solution as 16-bit integer
       # Bits 0-3: top row, 4-7: second row, etc.
       # This is simplified - real Sudoku needs more qubits
       grid = [(x >> (4*i)) & 0xF for i in range(4)]

       # Check each row has unique values
       for row in grid:
           if len(set([row >> i & 1 for i in range(4)])) != 4:
               return False
       return True

   # Search for valid solution
   result = grover_search(
       n_qubits=16,
       marked_states=is_valid_sudoku_solution,
       device='cpu'
   )

Application 2: 3-SAT Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find satisfying assignments for Boolean formulas:

.. code-block:: python

   from atlas_q.grover import grover_search

   def satisfies_3sat(x):
       """Check if variable assignment satisfies formula"""
       # Example: (x1 OR x2 OR ~x3) AND (~x1 OR x3 OR x4) AND ...
       # Variables encoded in bits of x
       x1 = (x >> 0) & 1
       x2 = (x >> 1) & 1
       x3 = (x >> 2) & 1
       x4 = (x >> 3) & 1

       clause1 = x1 or x2 or (not x3)
       clause2 = (not x1) or x3 or x4
       clause3 = x2 or (not x3) or (not x4)

       return clause1 and clause2 and clause3

   result = grover_search(
       n_qubits=4,
       marked_states=satisfies_3sat,
       device='cpu'
   )

   print(f"Satisfying assignment: {result['measured_state']:04b}")

Application 3: Finding Collisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find hash collisions (for educational purposes):

.. code-block:: python

   from atlas_q.grover import grover_search

   def simple_hash(x):
       """Simple hash function"""
       return (x * 31 + 17) % 256

   target_hash = simple_hash(42)

   def has_collision(x):
       """Check if x hashes to target but x != 42"""
       return x != 42 and simple_hash(x) == target_hash

   # Search for collision
   result = grover_search(
       n_qubits=8,  # Search 256 values
       marked_states=has_collision,
       device='cpu'
   )

   if result['measured_state'] != 0:
       print(f"Found collision: {result['measured_state']}")
       print(f"Hash: {simple_hash(result['measured_state'])}")

Performance Tuning
------------------

Choosing Bond Dimension
~~~~~~~~~~~~~~~~~~~~~~~~

The ``chi_max`` parameter controls MPS accuracy vs speed tradeoff:

.. code-block:: python

   from atlas_q.grover import GroverSearch, GroverConfig, BitmapOracle
   import time

   n_qubits = 6
   marked_state = 42

   for chi_max in [32, 64, 128, 256]:
       config = GroverConfig(
           n_qubits=n_qubits,
           chi_max=chi_max,
           device='cpu'
       )

       oracle = BitmapOracle(n_qubits, {marked_state}, device='cpu')
       grover = GroverSearch(oracle, config)

       start = time.time()
       result = grover.run()
       elapsed = time.time() - start

       print(f"χ={chi_max:3d}: {elapsed:.3f}s, "
             f"found={result['measured_state']}")

**Guidelines:**

- Small systems (≤5 qubits): ``chi_max=32`` sufficient
- Medium systems (6-7 qubits): ``chi_max=64-128``
- Large systems (≥8 qubits): ``chi_max=256-512``

Benchmarking Your Search
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the benchmark suite to analyze performance:

.. code-block:: bash

   # Run full benchmark suite
   python benchmarks/benchmark_grover.py --device cpu

   # Save results to file
   python benchmarks/benchmark_grover.py --device cuda --save results.json

Visualizing Convergence
~~~~~~~~~~~~~~~~~~~~~~~~

Track and plot success probability over iterations:

.. code-block:: python

   from atlas_q.grover import GroverSearch, GroverConfig, BitmapOracle
   import matplotlib.pyplot as plt

   config = GroverConfig(
       n_qubits=4,
       measure_success_prob=True,  # Enable tracking
       device='cpu'
   )

   oracle = BitmapOracle(4, {7}, device='cpu')
   grover = GroverSearch(oracle, config)
   result = grover.run()

   # Plot convergence
   grover.plot_convergence(save_path='grover_convergence.png')

   # Or plot manually
   plt.figure(figsize=(10, 6))
   plt.plot(range(1, len(grover.success_probabilities) + 1),
            grover.success_probabilities, 'b-o')
   plt.xlabel('Iteration')
   plt.ylabel('Success Probability')
   plt.title("Grover's Algorithm Convergence")
   plt.grid(True)
   plt.show()

Understanding Results
---------------------

Result Dictionary
~~~~~~~~~~~~~~~~~

The ``grover_search()`` and ``GroverSearch.run()`` methods return a dictionary:

.. code-block:: python

   result = {
       'measured_state': 7,         # Found state index
       'success_probability': 0.946, # P(measuring marked state)
       'iterations_used': 3,         # Number of Grover iterations
       'runtime_ms': 12.5,          # Execution time (ms)
       'bond_dims': [4, 8, 4],      # MPS bond dimensions
       'iteration_stats': [...]     # Per-iteration statistics
   }

Interpreting Success Probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The success probability indicates confidence in the result:

- **> 0.9**: High confidence - likely correct
- **0.7 - 0.9**: Moderate confidence - probably correct
- **< 0.7**: Low confidence - may need more iterations or higher chi_max

.. code-block:: python

   result = grover_search(n_qubits=5, marked_states={20}, device='cpu')

   if result['success_probability'] > 0.9:
       print(f"High confidence: found {result['measured_state']}")
   elif result['success_probability'] > 0.7:
       print(f"Moderate confidence: likely {result['measured_state']}")
   else:
       print("Low confidence - consider re-running with higher chi_max")

Multiple Runs for Reliability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For critical applications, run multiple searches:

.. code-block:: python

   from collections import Counter
   from atlas_q.grover import grover_search

   n_qubits = 4
   marked_states = {7}
   n_runs = 10

   results = []
   for _ in range(n_runs):
       result = grover_search(
           n_qubits=n_qubits,
           marked_states=marked_states,
           device='cpu'
       )
       results.append(result['measured_state'])

   # Take majority vote
   counts = Counter(results)
   most_common = counts.most_common(1)[0]
   print(f"Most common result: {most_common[0]} "
         f"(appeared {most_common[1]}/{n_runs} times)")

Troubleshooting
---------------

Problem: Low Success Probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** ``success_probability < 0.7``

**Solutions:**

1. Increase bond dimension:

   .. code-block:: python

      config = GroverConfig(n_qubits=6, chi_max=256)  # Higher chi_max

2. Verify marked states exist:

   .. code-block:: python

      from atlas_q.grover import calculate_grover_iterations

      n_marked = len(marked_states)
      assert n_marked > 0, "No marked states!"

      k = calculate_grover_iterations(n_qubits, n_marked)
      print(f"Will use {k} iterations for {n_marked} marked states")

3. Check oracle implementation:

   .. code-block:: python

      # Test oracle manually
      oracle = BitmapOracle(4, {7}, device='cpu')
      assert 7 in oracle.marked_states
      assert oracle.n_marked == 1

Problem: Slow Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Takes too long for your system size

**Solutions:**

1. Use GPU:

   .. code-block:: python

      result = grover_search(..., device='cuda')

2. Reduce bond dimension (trades accuracy for speed):

   .. code-block:: python

      config = GroverConfig(n_qubits=8, chi_max=64)  # Lower chi_max

3. Use smaller systems or approximate search

Problem: Incorrect Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Consistently returns wrong state

**Possible causes:**

1. **Oracle definition error** - Double-check marking function
2. **Too many iterations** - Try optimal k* value
3. **Marked state doesn't exist** - Verify constraints are satisfiable

**Debug approach:**

.. code-block:: python

   from atlas_q.grover import FunctionOracle

   def mark_even(x):
       return x % 2 == 0

   # Test oracle
   oracle = FunctionOracle(4, mark_even, device='cpu')
   print(f"Marked {oracle.n_marked} states")

   # Should mark 8 states: 0, 2, 4, 6, 8, 10, 12, 14
   assert oracle.n_marked == 8

Best Practices
--------------

1. **Start small:** Test with 3-4 qubits before scaling up
2. **Verify oracles:** Test marking functions on small inputs first
3. **Use optimal iterations:** Let ``auto_iterations=True`` (default)
4. **Monitor probabilities:** Enable ``measure_success_prob`` for debugging
5. **Benchmark first:** Run benchmarks to understand your system's limits
6. **GPU for scale:** Use CUDA for systems with ≥7 qubits
7. **Multiple runs:** For critical applications, run 5-10 times and vote

Further Reading
---------------

- `API Reference <../reference/grover.html>`_ - Complete API documentation
- `Benchmarks </benchmarks/benchmark_grover.py>`_ - Performance benchmarking suite
- `Grover's Original Paper <https://arxiv.org/abs/quant-ph/9605043>`_ (1996)
- `Quantum Amplitude Amplification <https://arxiv.org/abs/quant-ph/0005055>`_ (2002)

Related Algorithms
------------------

- **QAOA** (:doc:`vqe_qaoa`) - Quantum approximate optimization
- **VQE** (:doc:`vqe_qaoa`) - Variational quantum eigensolver
- **Quantum Counting** - Extension of Grover for counting solutions

Next Steps
----------

Now that you understand Grover's algorithm, try:

1. Implementing your own oracle for a specific problem
2. Experimenting with different problem sizes
3. Comparing classical vs quantum search times
4. Combining with other quantum algorithms (hybrid approaches)

Happy quantum searching!
