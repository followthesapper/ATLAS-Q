QAOA Tutorial
=============

This tutorial demonstrates the Quantum Approximate Optimization Algorithm (QAOA) with ATLAS-Q, covering combinatorial optimization problems including MaxCut, graph coloring, and satisfiability problems with practical optimization strategies.

Prerequisites
-------------

Before starting this tutorial, you should:

- Understand variational quantum algorithms (see :doc:`vqe_tutorial`)
- Have familiarity with combinatorial optimization problems
- Know basics of graph theory (vertices, edges, cuts)
- Understand Hamiltonian formulation of optimization problems
- Be comfortable with NetworkX for graph manipulation

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

1. Formulate combinatorial optimization problems as QAOA instances
2. Build problem and mixer Hamiltonians for various graph problems
3. Optimize QAOA parameters for MaxCut, graph coloring, and satisfiability
4. Understand the trade-off between circuit depth (p) and solution quality
5. Interpret measurement outcomes and extract optimal solutions
6. Apply warm-starting techniques to improve convergence
7. Compare QAOA performance to classical heuristic algorithms
8. Troubleshoot common QAOA optimization challenges

Part 1: Fundamentals of QAOA
-----------------------------

What is QAOA?
^^^^^^^^^^^^^

The Quantum Approximate Optimization Algorithm (QAOA) is a variational quantum algorithm designed for solving combinatorial optimization problems. Given a problem encoded as a cost function :math:`C(z)` over binary strings :math:`z \in \{0,1\}^n`, QAOA prepares a quantum state that approximates the optimal solution.

The algorithm constructs a parameterized quantum circuit alternating between two Hamiltonians:

.. math::

   |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}

where:

- :math:`U_C(\gamma) = e^{-i \gamma H_C}` applies the **problem Hamiltonian** :math:`H_C` encoding the cost function
- :math:`U_B(\beta) = e^{-i \beta H_B}` applies the **mixer Hamiltonian** :math:`H_B` (typically :math:`\sum_i X_i`)
- :math:`p` is the number of layers (depth parameter)
- :math:`\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_p)` and :math:`\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)` are variational parameters

The goal is to minimize the expectation value:

.. math::

   F(\boldsymbol{\gamma}, \boldsymbol{\beta}) = \langle \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) | H_C | \psi(\boldsymbol{\gamma}, \boldsymbol{\beta}) \rangle

Why QAOA Works
^^^^^^^^^^^^^^

QAOA combines quantum superposition with classical optimization to explore the solution space efficiently:

- **Initialization**: Start in uniform superposition :math:`|+\rangle^{\otimes n}` (equal amplitude on all bit strings)
- **Problem phase**: :math:`U_C(\gamma)` phases basis states according to their cost, favoring low-cost solutions
- **Mixer phase**: :math:`U_B(\beta)` creates quantum tunneling between basis states, enabling exploration
- **Layering**: Multiple layers (larger :math:`p`) allow finer control and better approximations
- **Measurement**: Final state has high probability of measuring near-optimal solutions

As :math:`p \to \infty`, QAOA becomes equivalent to quantum annealing and can reach the optimal solution. For practical :math:`p`, QAOA provides approximate solutions with guarantees on approximation ratio.

Basic QAOA Workflow
^^^^^^^^^^^^^^^^^^^

1. **Formulate problem**: Encode optimization as cost Hamiltonian :math:`H_C`
2. **Choose depth**: Select number of layers :math:`p` (typically 1-5)
3. **Initialize parameters**: Set :math:`\boldsymbol{\gamma}, \boldsymbol{\beta}` (random or heuristic)
4. **Optimize**: Use classical optimizer to minimize :math:`\langle H_C \rangle`
5. **Measure**: Sample bit strings from optimized state
6. **Decode**: Interpret bit strings as problem solutions

Part 2: The QAOA Ansatz
------------------------

Problem Hamiltonian (Cost Function)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The problem Hamiltonian :math:`H_C` encodes the cost function. For a binary optimization problem with cost :math:`C(z)`, we construct:

.. math::

   H_C = \sum_z C(z) |z\rangle\langle z|

For most problems, this simplifies to sums of Pauli operators. For example, MaxCut on a graph :math:`G=(V,E)`:

.. math::

   H_C = -\frac{1}{2} \sum_{(i,j) \in E} (I - Z_i Z_j)

where :math:`Z_i = \pm 1` indicates which partition vertex :math:`i` belongs to. Minimizing :math:`H_C` maximizes the cut.

Mixer Hamiltonian
^^^^^^^^^^^^^^^^^

The mixer Hamiltonian :math:`H_B` creates superpositions and enables transitions between basis states. The standard mixer is:

.. math::

   H_B = \sum_{i=1}^n X_i

This applies bit flips, allowing exploration of all :math:`2^n` bit strings. For constrained problems, custom mixers preserve feasibility.

Layered Structure
^^^^^^^^^^^^^^^^^

QAOA depth :math:`p` controls the number of alternating applications:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx

   # Create graph
   G = nx.karate_club_graph()

   # Build MaxCut Hamiltonian
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Compare different depths
   for p in [1, 2, 4, 8]:
       config = VQEConfig(max_iter=100, device='cuda')
       qaoa = QAOA(H, p=p, config=config)
       energy, params = qaoa.optimize()

       print(f"p={p}: Energy = {energy:.6f}, Parameters: {len(params)}")

   # Expected:
   # p=1: 2 parameters (γ₁, β₁)
   # p=2: 4 parameters (γ₁, β₁, γ₂, β₂)
   # p=4: 8 parameters
   # p=8: 16 parameters

Higher :math:`p` provides more expressivity but increases optimization difficulty (more parameters, deeper circuit).

Parameter Landscape
^^^^^^^^^^^^^^^^^^^

QAOA optimization landscapes exhibit interesting structure:

- **Concentration**: For small :math:`p`, optimal parameters concentrate near specific values
- **Barren plateaus**: For large systems, gradients can vanish exponentially
- **Symmetries**: Landscape often has periodic structure due to Hamiltonian symmetries
- **Local minima**: Can trap optimizers, especially at higher :math:`p`

Understanding landscape helps choose initialization and optimization strategies.

Part 3: MaxCut Problem
----------------------

Graph Partitioning
^^^^^^^^^^^^^^^^^^

MaxCut partitions graph vertices into two sets to maximize edges between sets. This is NP-hard in general but has many applications: network design, statistical physics (Ising model), VLSI design.

For a graph :math:`G=(V,E)`, a cut :math:`S \subset V` has value:

.. math::

   \text{cut}(S) = |\{(i,j) \in E : i \in S, j \notin S\}|

Building MaxCut Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ATLAS-Q provides convenient MaxCut Hamiltonian construction:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx

   # Create example graph (5-node cycle + diagonal)
   G = nx.Graph()
   G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)])

   # Build MaxCut Hamiltonian
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   print(f"Graph: {G.number_of_nodes()} vertices, {G.number_of_edges()} edges")
   print(f"Hamiltonian: {H.num_sites} qubits")

   # Run QAOA
   config = VQEConfig(max_iter=150, optimizer='COBYLA', device='cuda')
   qaoa = QAOA(H, p=3, config=config)
   energy, params = qaoa.optimize()

   print(f"QAOA energy: {energy:.6f}")
   print(f"MaxCut value (approx): {-energy:.6f}")

The negative energy gives the approximate MaxCut value.

Solution Interpretation
^^^^^^^^^^^^^^^^^^^^^^^^

Extract partition from QAOA state by measurement:

.. code-block:: python

   # Get optimized state
   mps_solution = qaoa.get_state()

   # Sample bit strings
   samples = []
   for _ in range(1000):
       bitstring = mps_solution.sample()
       samples.append(bitstring)

   # Count occurrences
   from collections import Counter
   counts = Counter(samples)

   # Get most frequent (likely optimal)
   best_bitstring, count = counts.most_common(1)[0]
   print(f"Most probable solution: {best_bitstring} (seen {count}/1000 times)")

   # Compute actual cut value
   def compute_cut(G, bitstring):
       """Compute cut value for given bit string."""
       cut_value = 0
       for i, j in G.edges():
           if bitstring[i] != bitstring[j]:
               cut_value += 1
       return cut_value

   cut_val = compute_cut(G, best_bitstring)
   print(f"Actual cut value: {cut_val}")

Quality Metrics
^^^^^^^^^^^^^^^

Evaluate QAOA performance:

.. code-block:: python

   # Classical bound (Goemans-Williamson: 0.878 approximation)
   import numpy as np

   def maxcut_bruteforce(G):
       """Find exact MaxCut by brute force (small graphs only)."""
       n = G.number_of_nodes()
       max_cut = 0
       best_partition = None

       for i in range(2**n):
           bitstring = [(i >> j) & 1 for j in range(n)]
           cut = compute_cut(G, bitstring)
           if cut > max_cut:
               max_cut = cut
               best_partition = bitstring

       return max_cut, best_partition

   max_cut_exact, _ = maxcut_bruteforce(G)
   max_cut_qaoa = cut_val

   approximation_ratio = max_cut_qaoa / max_cut_exact
   print(f"Exact MaxCut: {max_cut_exact}")
   print(f"QAOA MaxCut: {max_cut_qaoa}")
   print(f"Approximation ratio: {approximation_ratio:.3f}")

For well-tuned QAOA with sufficient :math:`p`, ratios typically exceed 0.9 for small graphs.

Part 4: Other Combinatorial Problems
-------------------------------------

Graph Coloring
^^^^^^^^^^^^^^

Assign colors to vertices such that no adjacent vertices share colors:

.. code-block:: python

   def graph_coloring_hamiltonian(G, num_colors=3):
       """
       Build Hamiltonian for graph coloring.
       Each vertex has num_colors qubits (one-hot encoding).
       """
       from atlas_q.mpo_ops import MPOBuilder
       import torch

       n_vertices = G.number_of_nodes()
       n_qubits = n_vertices * num_colors

       # Penalty for adjacent vertices having same color
       # H = sum_{(i,j) in E} sum_{c} (color_i^c AND color_j^c)
       # Implement using Pauli operators...

       # (Simplified - actual implementation in MPOBuilder)
       H = MPOBuilder.graph_coloring_hamiltonian(
           G,
           num_colors=num_colors,
           device='cuda'
       )

       return H

   # Example: 3-coloring of Petersen graph
   G_petersen = nx.petersen_graph()
   H_coloring = graph_coloring_hamiltonian(G_petersen, num_colors=3)

   config = VQEConfig(max_iter=200, device='cuda')
   qaoa = QAOA(H_coloring, p=4, config=config)
   energy, params = qaoa.optimize()

   print(f"Coloring energy: {energy:.6f}")

Vertex Cover
^^^^^^^^^^^^

Find minimum set of vertices such that every edge is incident to at least one vertex:

.. code-block:: python

   def vertex_cover_hamiltonian(G, penalty_weight=5.0):
       """
       Hamiltonian for minimum vertex cover.
       H = sum_{i} Z_i (minimize vertices) + penalty * sum_{(i,j)} (1-Z_i)(1-Z_j)
       """
       from atlas_q.mpo_ops import MPOBuilder

       H = MPOBuilder.vertex_cover_hamiltonian(
           G,
           penalty_weight=penalty_weight,
           device='cuda'
       )
       return H

   # Example
   G = nx.Graph()
   G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])

   H_vc = vertex_cover_hamiltonian(G, penalty_weight=10.0)

   config = VQEConfig(max_iter=150, device='cuda')
   qaoa = QAOA(H_vc, p=3, config=config)
   energy, params = qaoa.optimize()

   # Extract vertex cover from bit string
   mps = qaoa.get_state()
   bitstring = mps.sample()
   vertex_cover = [i for i, b in enumerate(bitstring) if b == 1]

   print(f"Vertex cover: {vertex_cover}")
   print(f"Cover size: {len(vertex_cover)}")

Satisfiability (3-SAT)
^^^^^^^^^^^^^^^^^^^^^^

Determine if Boolean formula in conjunctive normal form is satisfiable:

.. code-block:: python

   def sat_hamiltonian(clauses):
       """
       Build Hamiltonian for 3-SAT.
       Each clause contributes penalty if unsatisfied.
       """
       # clauses: list of (var, var, var, negation_flags)
       # Example: (x1 OR NOT x2 OR x3) represented as (0, 1, 2, (False, True, False))

       from atlas_q.mpo_ops import MPOBuilder

       H = MPOBuilder.sat_hamiltonian(clauses, device='cuda')
       return H

   # Example: (x1 OR x2 OR x3) AND (NOT x1 OR NOT x2 OR x3) AND (x1 OR NOT x3 OR NOT x4)
   clauses = [
       ((0, 1, 2), (False, False, False)),
       ((0, 1, 2), (True, True, False)),
       ((0, 2, 3), (False, True, True))
   ]

   H_sat = sat_hamiltonian(clauses)

   config = VQEConfig(max_iter=200, device='cuda')
   qaoa = QAOA(H_sat, p=4, config=config)
   energy, params = qaoa.optimize()

   if energy < 0.1:  # Near-zero energy indicates satisfiability
       print("Formula is satisfiable")
       mps = qaoa.get_state()
       assignment = mps.sample()
       print(f"Satisfying assignment: {assignment}")
   else:
       print("Formula may be unsatisfiable or more layers needed")

Part 5: Parameter Optimization
-------------------------------

Classical Optimizers
^^^^^^^^^^^^^^^^^^^^

QAOA parameters are optimized classically. Different optimizers suit different regimes:

.. code-block:: python

   from atlas_q.mpo_ops import MPOBuilder
   from atlas_q.vqe_qaoa import QAOA, VQEConfig
   import networkx as nx
   import time

   G = nx.karate_club_graph()
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   optimizers = ['COBYLA', 'L-BFGS-B', 'SLSQP', 'Nelder-Mead']
   results = []

   for opt in optimizers:
       config = VQEConfig(
           optimizer=opt,
           max_iter=200,
           tol=1e-6,
           device='cuda'
       )

       start = time.time()
       qaoa = QAOA(H, p=2, config=config)
       energy, params = qaoa.optimize()
       elapsed = time.time() - start

       results.append({
           'optimizer': opt,
           'energy': energy,
           'time': elapsed,
           'iterations': len(qaoa.get_history()['energies'])
       })

       print(f"{opt:15s}: E = {energy:.6f}, time = {elapsed:.2f}s, iters = {results[-1]['iterations']}")

   # Typical output:
   # COBYLA:        E = -5.423, time = 12.3s, iters = 87
   # L-BFGS-B:      E = -5.421, time = 8.7s, iters = 45
   # SLSQP:         E = -5.422, time = 9.1s, iters = 52
   # Nelder-Mead:   E = -5.415, time = 18.4s, iters = 156

For QAOA, COBYLA (derivative-free) is robust for small :math:`p`, while L-BFGS-B (gradient-based) is faster for larger :math:`p`.

Parameter Initialization Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Good initialization accelerates convergence:

.. code-block:: python

   import numpy as np

   # Strategy 1: Random initialization
   def random_init(p):
       gamma = np.random.uniform(0, 2*np.pi, p)
       beta = np.random.uniform(0, np.pi, p)
       return np.concatenate([gamma, beta])

   # Strategy 2: Heuristic initialization
   def heuristic_init(p):
       # Start with parameters known to work well for p=1
       # Extrapolate for higher p
       gamma = np.linspace(0.5, 1.5, p)
       beta = np.linspace(0.2, 0.8, p)
       return np.concatenate([gamma, beta])

   # Strategy 3: Transfer from lower p
   def transfer_init(qaoa_prev, p_new):
       """Initialize p_new layer QAOA from optimized p_old parameters."""
       params_old = qaoa_prev.optimal_params
       p_old = len(params_old) // 2

       # Interpolate to p_new
       gamma_old = params_old[:p_old]
       beta_old = params_old[p_old:]

       gamma_new = np.interp(
           np.linspace(0, p_old-1, p_new),
           np.arange(p_old),
           gamma_old
       )
       beta_new = np.interp(
           np.linspace(0, p_old-1, p_new),
           np.arange(p_old),
           beta_old
       )

       return np.concatenate([gamma_new, beta_new])

   # Compare strategies
   for init_name, init_func in [('random', random_init), ('heuristic', heuristic_init)]:
       config = VQEConfig(max_iter=100, device='cuda')
       qaoa = QAOA(H, p=3, config=config, initial_params=init_func(3))
       energy, _ = qaoa.optimize()

       print(f"{init_name} init: final energy = {energy:.6f}")

Layer-by-Layer Training
^^^^^^^^^^^^^^^^^^^^^^^^

Optimize incrementally, transferring parameters between depths:

.. code-block:: python

   # Layer-by-layer QAOA
   config = VQEConfig(max_iter=100, device='cuda')

   qaoa_p1 = QAOA(H, p=1, config=config)
   energy_p1, params_p1 = qaoa_p1.optimize()
   print(f"p=1: E = {energy_p1:.6f}")

   # Use p=1 solution to initialize p=2
   params_p2_init = transfer_init(qaoa_p1, p_new=2)
   qaoa_p2 = QAOA(H, p=2, config=config, initial_params=params_p2_init)
   energy_p2, params_p2 = qaoa_p2.optimize()
   print(f"p=2: E = {energy_p2:.6f}")

   # Continue to p=3
   params_p3_init = transfer_init(qaoa_p2, p_new=3)
   qaoa_p3 = QAOA(H, p=3, config=config, initial_params=params_p3_init)
   energy_p3, params_p3 = qaoa_p3.optimize()
   print(f"p=3: E = {energy_p3:.6f}")

This approach often converges faster than directly optimizing high-:math:`p` QAOA.

Part 6: Depth vs Performance Trade-offs
----------------------------------------

Effect of Circuit Depth
^^^^^^^^^^^^^^^^^^^^^^^^

Increasing :math:`p` improves approximation quality but increases computational cost:

.. code-block:: python

   import matplotlib.pyplot as plt

   G = nx.karate_club_graph()
   H = MPOBuilder.maxcut_hamiltonian(G, device='cuda')

   # Exact solution
   max_cut_exact, _ = maxcut_bruteforce(G)

   p_values = range(1, 9)
   energies = []
   approx_ratios = []
   times = []

   for p in p_values:
       config = VQEConfig(max_iter=150, device='cuda')

       start = time.time()
       qaoa = QAOA(H, p=p, config=config)
       energy, _ = qaoa.optimize()
       elapsed = time.time() - start

       energies.append(-energy)
       approx_ratios.append(-energy / max_cut_exact)
       times.append(elapsed)

       print(f"p={p}: cut = {-energy:.2f}, ratio = {approx_ratios[-1]:.3f}, time = {elapsed:.1f}s")

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   ax1.plot(p_values, approx_ratios, 'o-')
   ax1.axhline(1.0, color='r', linestyle='--', label='Optimal')
   ax1.set_xlabel('QAOA depth p')
   ax1.set_ylabel('Approximation ratio')
   ax1.set_title('Solution Quality vs Depth')
   ax1.grid(True)
   ax1.legend()

   ax2.plot(p_values, times, 'o-')
   ax2.set_xlabel('QAOA depth p')
   ax2.set_ylabel('Optimization time (s)')
   ax2.set_title('Computational Cost vs Depth')
   ax2.grid(True)

   plt.tight_layout()
   plt.savefig('qaoa_depth_tradeoff.png', dpi=150)

Typical behavior: approximation ratio improves with :math:`p` but saturates, while time grows linearly or faster.

Approximation Guarantees
^^^^^^^^^^^^^^^^^^^^^^^^^

For MaxCut, QAOA with :math:`p=1` guarantees approximation ratio :math:`\geq 0.6924` (worst case). Higher :math:`p` improves this bound. In practice, :math:`p=3-5` often achieves ratios > 0.95 for random graphs.

Part 7: Advanced QAOA Techniques
---------------------------------

Warm-Starting from Classical Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Initialize QAOA from classical heuristic solution:

.. code-block:: python

   # Get classical greedy solution
   def greedy_maxcut(G):
       """Greedy heuristic for MaxCut."""
       partition = [0] * G.number_of_nodes()
       for node in G.nodes():
           # Assign to partition maximizing cut
           cut_if_0 = sum(1 for neighbor in G.neighbors(node) if partition[neighbor] == 1)
           cut_if_1 = sum(1 for neighbor in G.neighbors(node) if partition[neighbor] == 0)
           partition[node] = 1 if cut_if_1 > cut_if_0 else 0
       return partition

   classical_solution = greedy_maxcut(G)
   classical_cut = compute_cut(G, classical_solution)
   print(f"Classical greedy: cut = {classical_cut}")

   # Initialize QAOA state to classical solution
   from atlas_q.adaptive_mps import AdaptiveMPS

   def warm_start_state(bitstring):
       """Create MPS initialized to given bit string."""
       n = len(bitstring)
       mps = AdaptiveMPS(num_qubits=n, bond_dim=2, device='cuda')

       # Prepare computational basis state |bitstring⟩
       for i, bit in enumerate(bitstring):
           if bit == 1:
               X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
               mps.apply_single_qubit_gate(i, X)

       return mps

   # Use warm-started QAOA
   mps_init = warm_start_state(classical_solution)
   qaoa_warm = QAOA(H, p=2, config=config, initial_state=mps_init)
   energy_warm, _ = qaoa_warm.optimize()

   print(f"Warm-started QAOA: cut = {-energy_warm:.2f}")

Warm-starting can significantly reduce optimization time for hard instances.

Custom Mixer Hamiltonians
^^^^^^^^^^^^^^^^^^^^^^^^^^

For constrained problems, design mixers preserving feasibility:

.. code-block:: python

   # Example: MaxCut with constraint that partition sizes differ by at most 1
   # Use XY mixer instead of X mixer:
   # H_B = sum_{i < j} (X_i X_j + Y_i Y_j)

   def custom_mixer_qaoa(H_problem, mixer_type='XY', p=2):
       """QAOA with custom mixer."""
       from atlas_q.mpo_ops import MPOBuilder

       if mixer_type == 'XY':
           H_mixer = MPOBuilder.xy_mixer_hamiltonian(
               num_qubits=H_problem.num_sites,
               device='cuda'
           )
       else:
           # Standard X mixer
           H_mixer = MPOBuilder.x_mixer_hamiltonian(
               num_qubits=H_problem.num_sites,
               device='cuda'
           )

       config = VQEConfig(max_iter=150, device='cuda')
       qaoa = QAOA(
           H_problem,
           p=p,
           config=config,
           mixer_hamiltonian=H_mixer
       )
       energy, params = qaoa.optimize()

       return energy, params

   energy_xy, _ = custom_mixer_qaoa(H, mixer_type='XY', p=3)
   print(f"XY mixer energy: {energy_xy:.6f}")

Custom mixers can improve performance on structured problems.

Part 8: Interpreting Results
-----------------------------

Measurement and Sampling
^^^^^^^^^^^^^^^^^^^^^^^^^

Extract solutions from QAOA state:

.. code-block:: python

   # Optimize QAOA
   qaoa = QAOA(H, p=3, config=config)
   energy, params = qaoa.optimize()

   # Get quantum state
   mps = qaoa.get_state()

   # Sample multiple bit strings
   n_samples = 10000
   samples = []
   for _ in range(n_samples):
       bitstring = mps.sample()
       cut_val = compute_cut(G, bitstring)
       samples.append((bitstring, cut_val))

   # Analyze distribution
   from collections import Counter
   cut_values = [cut for _, cut in samples]
   cut_counter = Counter(cut_values)

   print("Cut value distribution:")
   for cut_val in sorted(cut_counter.keys(), reverse=True):
       prob = cut_counter[cut_val] / n_samples
       print(f"  Cut {cut_val}: {prob*100:.1f}%")

   # Best observed solution
   best_sample = max(samples, key=lambda x: x[1])
   print(f"\nBest sampled solution: cut = {best_sample[1]}")

Visualization of Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize graph partitions:

.. code-block:: python

   import matplotlib.pyplot as plt
   import networkx as nx

   def visualize_cut(G, bitstring):
       """Draw graph with colored partition."""
       pos = nx.spring_layout(G, seed=42)

       # Color nodes by partition
       colors = ['red' if b == 0 else 'blue' for b in bitstring]

       # Draw
       plt.figure(figsize=(8, 6))
       nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.8)
       nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

       # Color edges: green if cut, gray otherwise
       edge_colors = []
       for i, j in G.edges():
           if bitstring[i] != bitstring[j]:
               edge_colors.append('green')
           else:
               edge_colors.append('gray')

       nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.6)

       cut_val = compute_cut(G, bitstring)
       plt.title(f'MaxCut Solution (cut value = {cut_val})')
       plt.axis('off')
       plt.tight_layout()
       plt.savefig('maxcut_solution.png', dpi=150)

   # Visualize QAOA solution
   best_bitstring = best_sample[0]
   visualize_cut(G, best_bitstring)

Comparison to Classical Heuristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmark against classical algorithms:

.. code-block:: python

   # Classical algorithms for MaxCut
   def random_cut(G, n_trials=1000):
       """Random partition."""
       best_cut = 0
       for _ in range(n_trials):
           bitstring = [random.randint(0, 1) for _ in range(G.number_of_nodes())]
           cut = compute_cut(G, bitstring)
           best_cut = max(best_cut, cut)
       return best_cut

   def simulated_annealing_cut(G, max_iter=10000):
       """Simulated annealing for MaxCut."""
       import random
       import math

       n = G.number_of_nodes()
       current = [random.randint(0, 1) for _ in range(n)]
       current_cut = compute_cut(G, current)
       best_cut = current_cut

       T = 1.0
       for iteration in range(max_iter):
           # Flip random bit
           neighbor = current.copy()
           flip_idx = random.randint(0, n-1)
           neighbor[flip_idx] = 1 - neighbor[flip_idx]
           neighbor_cut = compute_cut(G, neighbor)

           # Accept with probability
           delta = neighbor_cut - current_cut
           if delta > 0 or random.random() < math.exp(delta / T):
               current = neighbor
               current_cut = neighbor_cut
               best_cut = max(best_cut, current_cut)

           # Cool down
           T *= 0.9995

       return best_cut

   # Compare
   max_cut_exact, _ = maxcut_bruteforce(G)
   cut_random = random_cut(G)
   cut_greedy = classical_cut
   cut_sa = simulated_annealing_cut(G)
   cut_qaoa = -energy

   print(f"{'Method':<20s} {'Cut Value':<12s} {'Approximation':<15s}")
   print("-" * 50)
   print(f"{'Random':<20s} {cut_random:<12.2f} {cut_random/max_cut_exact:<15.3f}")
   print(f"{'Greedy':<20s} {cut_greedy:<12.2f} {cut_greedy/max_cut_exact:<15.3f}")
   print(f"{'Simulated Annealing':<20s} {cut_sa:<12.2f} {cut_sa/max_cut_exact:<15.3f}")
   print(f"{'QAOA (p=3)':<20s} {cut_qaoa:<12.2f} {cut_qaoa/max_cut_exact:<15.3f}")
   print(f"{'Exact':<20s} {max_cut_exact:<12.2f} {'1.000':<15s}")

Part 9: Troubleshooting
------------------------

Problem: QAOA Converges to Poor Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Approximation ratio < 0.7, far from optimal

**Solutions**:

1. Increase number of layers:

   .. code-block:: python

      qaoa = QAOA(H, p=5, config=config)  # Try p=5 instead of p=2

2. Improve parameter initialization:

   .. code-block:: python

      # Use heuristic or layer-by-layer initialization
      params_init = heuristic_init(p=3)
      qaoa = QAOA(H, p=3, config=config, initial_params=params_init)

3. Try different optimizer:

   .. code-block:: python

      config.optimizer = 'L-BFGS-B'  # Switch from COBYLA

Problem: Optimization Takes Too Long
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Hours to converge, stuck in local minima

**Solutions**:

1. Reduce number of optimizer iterations:

   .. code-block:: python

      config.max_iter = 100  # Instead of 300

2. Use faster (less accurate) QAOA:

   .. code-block:: python

      qaoa = QAOA(H, p=2, config=config)  # Lower p

3. Warm-start from classical solution:

   .. code-block:: python

      classical_sol = greedy_maxcut(G)
      mps_init = warm_start_state(classical_sol)
      qaoa = QAOA(H, p=2, config=config, initial_state=mps_init)

Problem: Sampled Solutions Don't Match Energy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Low energy but samples give poor cuts

**Solutions**:

1. Increase sampling count:

   .. code-block:: python

      n_samples = 100000  # More samples for rare good solutions

2. Check for degenerate ground states:

   .. code-block:: python

      # Multiple near-optimal solutions may exist
      # Take best among top samples
      top_samples = sorted(samples, key=lambda x: x[1], reverse=True)[:100]

3. Post-process with local search:

   .. code-block:: python

      def local_search(G, bitstring):
          """1-flip local search to improve solution."""
          current = list(bitstring)
          improved = True

          while improved:
              improved = False
              for i in range(len(current)):
                  current[i] = 1 - current[i]
                  if compute_cut(G, current) > compute_cut(G, bitstring):
                      bitstring = current.copy()
                      improved = True
                  else:
                      current[i] = 1 - current[i]

          return bitstring

      improved_sol = local_search(G, best_bitstring)

Problem: QAOA Worse Than Classical for Small Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Symptoms**: Classical greedy outperforms QAOA

**Solutions**:

This is expected. QAOA quantum advantage appears for:

- Larger graphs (50+ nodes)
- Dense graphs (high edge-to-node ratio)
- Structured problem instances (not random)
- Higher circuit depth (:math:`p \geq 3`)

For small graphs, classical algorithms are competitive. Use QAOA for regimes where classical methods struggle.

Summary
-------

This tutorial covered QAOA with ATLAS-Q:

✓ Formulated combinatorial problems (MaxCut, coloring, SAT) as QAOA Hamiltonians
✓ Built problem and mixer Hamiltonians for various optimization tasks
✓ Optimized QAOA parameters using classical optimizers
✓ Explored depth-performance trade-offs with varying :math:`p`
✓ Sampled and interpreted solutions from QAOA states
✓ Applied warm-starting and custom mixers for improved performance
✓ Compared QAOA to classical heuristics (greedy, simulated annealing)
✓ Troubleshot convergence and solution quality issues

QAOA bridges quantum computation and classical optimization, providing a near-term quantum algorithm for combinatorial problems with practical applications.

Next Steps
----------

- :doc:`vqe_tutorial` - Related variational algorithm principles
- :doc:`advanced_features` - Multi-objective optimization and constraints
- :doc:`../howtos/custom_hamiltonians` - Build problem Hamiltonians
- :doc:`../explanations/algorithms` - QAOA theory and guarantees

Practice Exercises
------------------

1. Implement QAOA for weighted MaxCut where edges have different weights
2. Compare approximation ratios for QAOA with :math:`p=1,2,3,4,5` on a 20-node random graph
3. Warm-start QAOA from a greedy solution and measure speedup
4. Design a custom mixer for graph coloring that preserves one-hot constraints
5. Implement tournament selection to combine multiple QAOA samples into best solution

See Also
--------

- :doc:`../../../reference/vqe_qaoa` - QAOA API reference
- :doc:`../../../reference/mpo_ops` - Problem Hamiltonian builders
- :doc:`../explanations/algorithms` - Theoretical foundations
- :doc:`../howtos/optimize_performance` - Performance tuning for large instances
