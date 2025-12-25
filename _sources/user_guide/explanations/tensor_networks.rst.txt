========================================
Tensor Networks
========================================

Overview
========

Tensor networks provide an efficient representation of quantum many-body states by decomposing
the exponentially large wavefunction into a network of smaller tensors connected by contraction
indices. This representation enables simulation of quantum systems that would be intractable
using full statevector methods.

Key advantages:

- **Memory efficiency**: O(n·χ²·d) vs O(d^n) for statevector
- **Computational tractability**: Polynomial vs exponential scaling
- **Physical insight**: Explicitly captures entanglement structure
- **Scalability**: 50+ qubits with moderate entanglement

This document explains the theoretical foundations of tensor networks, their mathematical structure,
and why they enable efficient quantum simulation.

.. seealso::
   :doc:`../tutorials/mps_basics` for practical MPS usage,
   :doc:`adaptive_truncation` for truncation theory,
   :doc:`algorithms` for tensor network algorithms.

Tensor Network Fundamentals
============================

Basic Tensor Representation
----------------------------

A tensor is a multidimensional array with indices representing different physical or virtual degrees of freedom.

**Rank-3 tensor** (MPS building block):

.. math::

   A^{[i]}_{α,s,β}

where:

- α, β: Virtual indices (bond dimensions) connecting to neighboring tensors
- s: Physical index representing local quantum state (s = 0, 1 for qubits)
- i: Tensor position in the chain

**Graphical notation**:

Tensors are represented as shapes with legs for each index:

- Physical indices: Point up/down
- Virtual (bond) indices: Point left/right
- Contraction: Connect matching indices

**Example** - MPS for 3 qubits::

    Physical:  |    |    |
              [A¹] [A²] [A³]
    Virtual:  ─────────────

Tensor Contraction
------------------

Tensor contraction is generalized matrix multiplication summing over shared indices.

**Matrix multiplication** as tensor contraction:

.. math::

   C_{ij} = \sum_k A_{ik} B_{kj}

**Tensor network contraction**:

.. math::

   |\psi\rangle = \sum_{α_1,α_2,...,s_1,s_2,...} A^{[1]}_{α_1,s_1,α_2} A^{[2]}_{α_2,s_2,α_3} \cdots A^{[n]}_{α_n,s_n,α_{n+1}} |s_1 s_2 \ldots s_n\rangle

Computational complexity:

- Order matters: O(χ³) vs O(χ⁶) for different contraction orders
- Optimal order: NP-hard in general
- 1D chains (MPS): Naturally efficient O(n·χ³) contraction

Matrix Product States (MPS)
============================

MPS Representation
------------------

An n-qubit quantum state in MPS form:

.. math::

   |\psi\rangle = \sum_{s_1,s_2,\ldots,s_n=0}^{d-1} \text{Tr}(A^{[1]}_{s_1} A^{[2]}_{s_2} \cdots A^{[n]}_{s_n}) |s_1 s_2 \ldots s_n\rangle

where:

- Each :math:`A^{[i]}_{s_i}` is a :math:`χ_{i-1} \times χ_i` matrix
- :math:`s_i \in \{0, 1, \ldots, d-1\}` is the physical index (d=2 for qubits)
- :math:`χ_i` is the bond dimension between sites i and i+1
- Boundary conditions: :math:`χ_0 = χ_n = 1`

**Explicit form** for 3 qubits:

.. math::

   |\psi\rangle = \sum_{s_1,s_2,s_3=0}^{1} A^{[1]}_{s_1} A^{[2]}_{s_2} A^{[3]}_{s_3} |s_1 s_2 s_3\rangle

where :math:`A^{[1]}_{s_1}` is a :math:`1 \times χ_1` matrix (row vector),
:math:`A^{[2]}_{s_2}` is a :math:`χ_1 \times χ_2` matrix,
and :math:`A^{[3]}_{s_3}` is a :math:`χ_2 \times 1` matrix (column vector).

Memory Scaling
--------------

**Full statevector**:

- Memory: :math:`O(2^n)` complex numbers
- Example: 30 qubits = 1,073,741,824 amplitudes × 16 bytes = 17 GB

**MPS representation**:

- Memory: :math:`O(n \cdot χ^2 \cdot d)` complex numbers
- Example: 30 qubits, χ=64 = 30 × 64² × 2 × 16 bytes ≈ 4 MB

**Compression ratio**: For n=30, χ=64: ~4,000× compression

Canonical Forms
===============

MPS can be put into canonical forms that simplify operations and provide computational advantages.

Left-Canonical Form
-------------------

In left-canonical form, tensors satisfy the orthonormality condition:

.. math::

   \sum_{α,s} A^{[i]}_{α,s,β} (A^{[i]}_{α,s,β'})^* = δ_{β,β'}

Graphically, contracting a left-canonical tensor with its conjugate gives identity::

    |
   [A]─┬─ = ─┬─
       │    │
   [A*]─┴─   └─ = Identity

**Properties**:

- Normalized: Inner product computation is O(χ²) instead of O(2^n)
- Efficient expectation values on left part of chain
- Singular values concentrated on right bond

Right-Canonical Form
--------------------

In right-canonical form, tensors satisfy:

.. math::

   \sum_{β,s} A^{[i]}_{α,s,β} (A^{[i]}_{α',s,β})^* = δ_{α,α'}

**Properties**:

- Normalized: Inner product computation is O(χ²)
- Efficient expectation values on right part of chain
- Singular values concentrated on left bond

Mixed-Canonical Form
--------------------

Mixed-canonical form has a partition point k where:

- Sites 1 to k-1 are left-canonical
- Sites k+1 to n are right-canonical
- Site k is the orthogonality center

.. math::

   |\psi\rangle = \sum \underbrace{L^{[1]} \cdots L^{[k-1]}}_{\text{left-canonical}} C^{[k]} \underbrace{R^{[k+1]} \cdots R^{[n]}}_{\text{right-canonical}} |s_1 \ldots s_n\rangle

**Applications**:

- DMRG: Sweep orthogonality center through chain
- TDVP: Time evolution at orthogonality center
- Truncation: SVD at orthogonality center gives optimal approximation

Entanglement and Bond Dimensions
=================================

Entanglement Entropy
--------------------

For a bipartition of the chain at bond i, the entanglement entropy is:

.. math::

   S_i = -\sum_α λ_α^2 \log(λ_α^2)

where :math:`λ_α` are the Schmidt values (singular values) across bond i.

**Bond dimension requirement**:

To exactly represent a state with entanglement entropy S:

.. math::

   χ_i \geq e^{S_i}

**Area law**: For 1D systems with local interactions, :math:`S \propto \log(n)`, so χ grows slowly.

**Volume law**: For highly entangled states, :math:`S \propto n`, requiring exponential χ.

When MPS is Efficient
---------------------

**Efficient** (χ polynomial in n):

- 1D systems with local interactions
- Ground states of gapped Hamiltonians (area law)
- Short-time evolution of product states
- Shallow quantum circuits
- Thermal states at low temperature

**Inefficient** (χ exponential in n):

- Random quantum states (volume law)
- Deep quantum circuits with long-range gates
- Highly entangled states (GHZ, W states may have low entanglement)
- Critical systems at quantum phase transitions (logarithmic divergence)

Truncation and Approximation
-----------------------------

When χ exceeds a threshold, truncate by keeping largest singular values:

**SVD truncation**:

.. math::

   A^{[i]}_{α,s,β} \approx \sum_{γ=1}^{χ_{\text{max}}} U_{α,γ} λ_γ V_{γ,β}^†

where λ₁ ≥ λ₂ ≥ ... ≥ λ_χ are sorted singular values.

**Truncation error**:

.. math::

   ε = \sqrt{\sum_{γ > χ_{\text{max}}} λ_γ^2}

This is the Frobenius norm of the discarded part, minimizing distance to exact state.

Matrix Product Operators (MPO)
===============================

MPO Structure
-------------

A matrix product operator represents an operator as:

.. math::

   \hat{O} = \sum_{s_1,s_1',...,s_n,s_n'} W^{[1]}_{s_1,s_1'} W^{[2]}_{s_2,s_2'} \cdots W^{[n]}_{s_n,s_n'} |s_1 \ldots s_n\rangle \langle s_1' \ldots s_n'|

Each :math:`W^{[i]}` is a rank-4 tensor with shape :math:`[χ_L, d, d, χ_R]`:

- :math:`χ_L, χ_R`: Virtual indices (operator bond dimension)
- First d: Output physical index (ket)
- Second d: Input physical index (bra)

Hamiltonian as MPO
------------------

**Transverse-field Ising Hamiltonian**:

.. math::

   H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i

Can be represented as MPO with bond dimension χ = 3:

.. math::

   W^{[i]} = \begin{pmatrix}
   I & 0 & 0 \\
   Z & 0 & 0 \\
   hX & -JZ & I
   \end{pmatrix}

**Graphical representation**::

      |  |
     [W¹][W²] ... [Wⁿ]
      |  |

**General nearest-neighbor Hamiltonians**: χ = 3
**Long-range interactions**: χ grows with range
**Time evolution operator** :math:`e^{-iHt}`: Can be approximated by MPO

MPO-MPS Operations
------------------

**Applying MPO to MPS** (W|ψ⟩):

Contract MPO with MPS to get new MPS with bond dimension χ_new = χ_MPS × χ_MPO:

.. math::

   |\phi\rangle = \hat{O} |\psi\rangle

Computational cost: :math:`O(n \cdot χ_{\text{MPS}}^2 \cdot χ_{\text{MPO}}^2 \cdot d^3)`

**Expectation value** ⟨ψ|O|ψ⟩:

Contract bra-MPO-ket network from left to right:

.. math::

   \langle \psi | \hat{O} | \psi \rangle = \text{Tr}(E^{[1]} E^{[2]} \cdots E^{[n]})

where :math:`E^{[i]}` is the environment tensor at site i.

Computational cost: :math:`O(n \cdot χ_{\text{MPS}}^2 \cdot χ_{\text{MPO}} \cdot d^2)`

Tensor Network Contraction
===========================

Contraction Order
-----------------

**Problem**: Given a tensor network, in what order should indices be contracted?

**Complexity**: Depends critically on order
- Bad order: :math:`O(χ^6)` or worse
- Good order: :math:`O(χ^3)`

**Finding optimal order**: NP-hard in general

**For MPS/MPO chains**: Natural left-to-right or right-to-left order is optimal

Efficient Contraction Algorithms
---------------------------------

**1. Sequential contraction** (MPS):

Contract tensors left-to-right:

.. code-block:: text

   A¹ → (A¹·A²) → ((A¹·A²)·A³) → ...

Cost per step: O(χ³·d)
Total cost: O(n·χ³·d)

**2. Binary tree contraction**:

Pair tensors and contract in tree structure:

.. code-block:: text

   Level 1: (A¹·A²), (A³·A⁴), ...
   Level 2: ((A¹·A²)·(A³·A⁴)), ...

Can be more cache-efficient but same asymptotic cost for MPS.

**3. GPU-accelerated einsum**:

Use optimized libraries (cuBLAS, cuTENSOR, Triton) for tensor contractions:
- Batch operations when possible
- Fuse contractions to reduce memory movement
- Utilize Tensor Cores on modern GPUs

Higher-Dimensional Tensor Networks
===================================

Projected Entangled Pair States (PEPS)
---------------------------------------

PEPS extend MPS to 2D and higher dimensions.

**Structure**: Each tensor has:
- 1 physical index (qubit state)
- 4 virtual indices (connecting to neighbors in 2D grid)

.. math::

   T^{[i,j]}_{α,β,γ,δ,s}

where α, β, γ, δ connect to left, right, up, down neighbors, and s is physical index.

**Memory scaling**: :math:`O(L^2 \cdot χ^4 \cdot d)` for L×L grid

**Contraction complexity**: NP-hard in general
- Approximate methods: boundary MPS, CTMRG
- Exact contraction: exponential cost

**Use cases in ATLAS-Q**:
- 2D quantum circuits
- Qubit connectivity maps to 2D grid
- Shallow circuits where contraction is tractable

Tree Tensor Networks (TTN)
---------------------------

Organize tensors in a tree structure rather than 1D chain.

**Advantages**:
- Flexible topology matching problem structure
- Lower entanglement for some systems
- Hierarchical decomposition

**Disadvantages**:
- More complex algorithms
- Less mature software support

Multi-Scale Entanglement Renormalization Ansatz (MERA)
-------------------------------------------------------

Hierarchical tensor network with entanglement renormalization:

- Isometry tensors: Coarse-grain system
- Disentangler tensors: Remove short-range entanglement
- Logarithmic bond dimension for critical systems

**Use cases**:
- Critical systems
- Scale-invariant states
- Conformal field theories

Computational Complexity
========================

Scaling Analysis
----------------

**Full statevector**:

- Memory: :math:`O(2^n)`
- Gate application: :math:`O(2^n)` for 1-qubit, :math:`O(2^{n+1})` for 2-qubit
- Exact up to ~30 qubits with 16 GB RAM

**MPS (adaptive bond dimension)**:

- Memory: :math:`O(n \cdot χ^2 \cdot d)`
- Gate application: :math:`O(χ^3 \cdot d^2)` per gate
- Truncation: :math:`O(χ^3 \cdot d)` (SVD)
- Scales to 50-100+ qubits with χ ≤ 512

**Crossover**: MPS advantageous when :math:`n \cdot χ^2 \ll 2^n`

For χ=64: Crossover at n ≈ 15-20 qubits

Bottlenecks
-----------

**For small χ (< 128)**:
- Bottleneck: Gate application (tensor contractions)
- Solution: GPU acceleration, Triton kernels

**For large χ (≥ 256)**:
- Bottleneck: SVD truncation
- Solution: cuQuantum, randomized SVD, truncation-free methods

**For very large n (> 100)**:
- Bottleneck: Memory bandwidth
- Solution: Distributed MPS across multiple GPUs

Connection to Other Formalisms
===============================

Quantum Circuits
----------------

MPS naturally represents quantum circuit evolution:

1. Initialize product state (χ=1)
2. Apply gates sequentially
3. Bond dimension grows with entanglement
4. Truncate when χ exceeds threshold

**Advantage**: Automatic compression vs statevector

Density Matrices
----------------

Purified density matrix as MPS:

.. math::

   \rho = \text{Tr}_{\text{aux}}(|\psi\rangle \langle \psi|)

where |ψ⟩ is MPS with doubled Hilbert space.

**Matrix Product Density Operators (MPDO)**: Direct representation of mixed states

Schmidt Decomposition
---------------------

For bipartition at bond i:

.. math::

   |\psi\rangle = \sum_α λ_α |L_α\rangle |R_α\rangle

where λ_α are Schmidt coefficients, |L_α⟩ and |R_α⟩ are orthonormal bases.

**MPS connection**: Bond dimension χ_i equals number of non-zero Schmidt values.

Wave Function Ansätze
---------------------

MPS is a variational ansatz for wave functions:

- **Variational principle**: Minimize ⟨ψ|H|ψ⟩ over MPS manifold
- **Tangent space**: Derivatives with respect to MPS tensors
- **Optimization**: DMRG, TDVP, gradient descent

Advantages and Limitations
===========================

When to Use MPS
---------------

**Ideal scenarios**:
- 1D systems or systems with 1D-like entanglement structure
- Ground state search (DMRG)
- Short-to-moderate time evolution (TDVP)
- Moderate entanglement (χ < 512)
- Large system size (n > 30)

**Performance sweet spot**: 30-80 qubits, χ=64-256

When Not to Use MPS
-------------------

**Poor performance**:
- Random quantum states (volume law entanglement)
- Deep quantum circuits with all-to-all connectivity
- Long-time evolution with entanglement spreading
- Small systems (n < 15): Statevector faster

**Alternative methods**:
- Small n: Exact statevector
- 2D systems: PEPS (more expensive)
- Classical systems: Classical simulation
- Highly entangled: Other methods or approximations

Comparison with Other Methods
------------------------------

**vs Statevector**:
- MPS: Polynomial memory/time, approximate
- Statevector: Exponential memory/time, exact

**vs Monte Carlo**:
- MPS: Deterministic, coherent evolution
- Monte Carlo: Stochastic, better for equilibrium properties

**vs PEPS**:
- MPS: 1D, O(n·χ²), exact contraction
- PEPS: 2D, O(n²·χ⁴), approximate contraction

**vs Hardware**:
- MPS: Classical simulation, unlimited qubits (with compression)
- Hardware: Real qubits, limited size, noisy

Summary
=======

Tensor networks, particularly Matrix Product States, provide:

1. **Efficient representation**: O(n·χ²) vs O(2^n) memory
2. **Physical insight**: Entanglement structure explicitly encoded in bond dimensions
3. **Scalability**: 50-100+ qubits with moderate entanglement
4. **Flexibility**: MPS, MPO, PEPS for different problems
5. **Approximation control**: Truncation error bounds from SVD

**Key concepts**:
- Tensor contraction and graphical notation
- Canonical forms for efficient computation
- Bond dimension χ controls accuracy vs cost
- Area law entanglement enables efficiency
- MPO for operators and Hamiltonians
- Higher-dimensional generalizations (PEPS, TTN, MERA)

**Computational complexity**:
- Memory: O(n·χ²·d)
- Time: O(n·χ³·d²) per operation
- Truncation: O(χ³·d) per site

**Limitations**:
- Exponential χ for volume law entanglement
- 1D structure limits 2D system efficiency
- Truncation introduces controllable error

Tensor networks are the foundation enabling ATLAS-Q to simulate quantum systems
beyond the reach of exact statevector methods while maintaining accuracy control
through adaptive truncation.

See Also
========

- :doc:`../tutorials/mps_basics`: Practical MPS usage and examples
- :doc:`adaptive_truncation`: Theory of SVD truncation and error bounds
- :doc:`algorithms`: DMRG, TDVP, and other tensor network algorithms
- :doc:`numerical_stability`: Numerical considerations for tensor networks
- :doc:`performance_model`: Computational complexity and scaling analysis
