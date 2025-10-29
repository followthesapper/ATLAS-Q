========================================
Adaptive Truncation
========================================

Overview
========

Adaptive truncation is the key technique enabling efficient MPS simulation by dynamically
adjusting bond dimensions based on entanglement structure. By truncating small singular values,
MPS maintains controllable approximation error while keeping memory and computational costs
polynomial in system size.

Without truncation, bond dimensions grow exponentially: χ → 2χ after each two-site gate,
quickly becoming intractable. With adaptive truncation, χ grows only as needed to capture
physical entanglement, enabling simulation of 50+ qubit systems.

Key concepts:

- **SVD truncation**: Discard small singular values to reduce bond dimension
- **Optimal approximation**: SVD minimizes distance to exact state
- **Error bounds**: Truncation error quantified by discarded singular values
- **Adaptive growth**: χ increases only when entanglement requires it
- **Memory budgets**: Enforce global memory constraints across all bonds

This document explains the theory of adaptive truncation, error analysis, and implementation
strategies used in ATLAS-Q.

.. seealso::
   :doc:`tensor_networks` for tensor network foundations,
   :doc:`numerical_stability` for numerical considerations,
   :doc:`../tutorials/mps_basics` for practical truncation usage.

SVD and Optimal Truncation
===========================

Singular Value Decomposition
-----------------------------

**Schmidt decomposition** of a bipartite quantum state:

.. math::

   |\psi\rangle_{AB} = \sum_{i=1}^r λ_i |u_i\rangle_A \otimes |v_i\rangle_B

where:

- λᵢ ≥ 0 are Schmidt coefficients (λ₁ ≥ λ₂ ≥ ... ≥ λᵣ)
- |uᵢ⟩, |vᵢ⟩ are orthonormal Schmidt bases
- r = min(dim A, dim B) is Schmidt rank
- Normalization: Σᵢ λᵢ² = 1

**SVD** is the computational realization of Schmidt decomposition:

.. math::

   Θ_{α,s,β} = \sum_{γ=1}^r U_{α,γ} λ_γ V_{γ,β}^†

where:

- Θ is the two-site tensor after gate application
- U, V† are unitary matrices
- λ are positive singular values (λ = Schmidt coefficients)
- r = χ_left × 2 × 2 × χ_right (before truncation)

Optimality of SVD Truncation
-----------------------------

**Theorem** (Eckart-Young-Mirsky):

Truncating SVD to keep k largest singular values minimizes the Frobenius norm error:

.. math::

   \min_{\text{rank}(M)=k} \|Θ - M\|_F = \sqrt{\sum_{i=k+1}^r λ_i^2}

This means SVD truncation gives the **optimal rank-k approximation** to the exact state.

**Interpretation**:

- Keeping k singular values: Best possible approximation with bond dimension χ = k
- No other truncation method can do better (in Frobenius norm)
- Error is simply sum of discarded singular values squared

Truncation Error
----------------

**Local truncation error** at bond i:

.. math::

   ε_i = \sqrt{\sum_{α=χ_i+1}^r λ_{i,α}^2}

where χᵢ is the chosen bond dimension after truncation.

**Distance to exact state**:

.. math::

   \| |\psi_{\text{approx}}\rangle - |\psi_{\text{exact}}\rangle \|^2 = ε_i^2

**Fidelity**:

.. math::

   F = |\langle \psi_{\text{exact}} | \psi_{\text{approx}} \rangle|^2 \approx 1 - ε_i^2

for small ε_i.

Truncation Criteria
===================

Energy-Based Truncation
-----------------------

Keep singular values until retained energy exceeds threshold:

.. math::

   \sum_{α=1}^{χ} λ_α^2 \geq (1 - ε^2) \sum_{α=1}^r λ_α^2

where ε is the truncation tolerance (``truncation_threshold`` in ATLAS-Q).

**Algorithm**:

1. Compute cumulative energy: :math:`E_k = \sum_{α=1}^k λ_α^2`
2. Find smallest k such that :math:`E_k \geq (1 - ε^2) E_r`
3. Truncate to χ = min(k, χ_max)

**Interpretation**: Retain at least (1-ε²) ≈ (1-2ε) of the wavefunction norm.

**Typical values**:

- ε = 1e-8: High accuracy, large χ
- ε = 1e-6: Good accuracy, moderate χ (recommended default)
- ε = 1e-4: Lower accuracy, small χ

Fixed Bond Dimension
--------------------

Simply keep the k largest singular values:

.. math::

   χ = \min(k_{\text{fixed}}, r)

**Advantages**:

- Predictable memory usage
- Consistent computational cost
- Simple to implement

**Disadvantages**:

- May waste resources (too large χ for low entanglement)
- May lose accuracy (too small χ for high entanglement)

Combined Criterion
------------------

Use both energy-based and fixed maximum:

.. math::

   χ = \min(k_{\text{energy}}, χ_{\max})

This ensures:

1. Accuracy: Retain enough singular values (energy criterion)
2. Bounded cost: Never exceed χ_max (memory/time constraint)

**ATLAS-Q implementation**:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,                    # Initial χ
       chi_max_per_bond=256,           # Maximum χ per bond
       truncation_threshold=1e-6,      # Energy criterion (ε)
       adaptive_mode=True              # Enable adaptive growth
   )

Error Accumulation and Bounds
==============================

Global Error Accumulation
-------------------------

Truncation errors at different bonds are independent, so total error accumulates:

.. math::

   ε_{\text{global}}^2 = \sum_{i=1}^{n-1} ε_i^2

where the sum is over all bonds in the MPS.

**For N gate applications**:

.. math::

   ε_{\text{global}} \leq \sqrt{N} · ε_{\text{local}}

assuming each gate introduces local error ε_local.

**Example**:

- 100 gates, ε_local = 1e-6 per gate
- Global error ≤ √100 × 1e-6 = 1e-5

Worst-Case Bound
----------------

**Theorem**: After N two-site gates with local truncation error ε each:

.. math::

   \| |\psi_N\rangle - |\psi_N^{\text{exact}}\rangle \| \leq \sqrt{N} · ε

**Proof sketch**:

1. Each gate introduces independent error ε
2. Errors add in quadrature (triangle inequality)
3. Total distance bounded by :math:`\sqrt{\sum_i ε_i^2} \leq \sqrt{N} · ε`

Practical Error Tracking
-------------------------

ATLAS-Q automatically tracks global error:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=32, device='cuda')

   # Apply gates
   for i in range(100):
       mps.apply_cnot(i % 29, (i % 29) + 1)

   # Check accumulated error
   global_error = mps.statistics.total_truncation_error
   print(f"Global truncation error: {global_error:.2e}")

   # Per-bond error distribution
   bond_errors = mps.statistics.truncation_error_per_bond
   max_error_bond = bond_errors.argmax()
   print(f"Worst bond: {max_error_bond}, error: {bond_errors[max_error_bond]:.2e}")

Tightening Error Bounds
------------------------

When global error exceeds tolerance:

**Option 1**: Reduce truncation threshold

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       truncation_threshold=1e-8,  # Was 1e-6
       chi_max_per_bond=512        # May need larger χ_max
   )

**Option 2**: Increase maximum bond dimension

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=64,
       truncation_threshold=1e-6,
       chi_max_per_bond=512  # Was 256
   )

**Option 3**: Checkpoint and restart with tighter tolerances

Adaptive Truncation Strategies
===============================

Entropy-Based Adaptive Truncation
----------------------------------

Adjust χ based on entanglement entropy:

.. math::

   S_{\text{vN}} = -\sum_α λ_α^2 \log(λ_α^2)

**Strategy**:

1. Compute entanglement entropy S from singular values
2. Set χ = min(2^S, χ_max) to capture entanglement
3. Ensures bond dimension matches physical entanglement

**When S is small** (low entanglement):
- Small χ sufficient
- Memory savings

**When S is large** (high entanglement):
- Large χ required
- Automatic adaptation

Per-Bond Adaptive Caps
-----------------------

Different bonds have different entanglement, so use different χ_max:

.. math::

   χ_i = \min(k_{\text{energy},i}, χ_{\max,i})

**ATLAS-Q implementation**:

.. code-block:: python

   # Heterogeneous bond dimension caps
   num_qubits = 20
   chi_caps = []

   for i in range(num_qubits - 1):
       if 5 <= i <= 14:
           # Center bonds: high entanglement
           chi_caps.append(256)
       else:
           # Edge bonds: low entanglement
           chi_caps.append(64)

   mps = AdaptiveMPS(
       num_qubits=num_qubits,
       bond_dim=16,
       chi_max_per_bond=chi_caps
   )

**Rationale**:

- Center of chain: Higher entanglement from accumulated gates
- Edges: Lower entanglement (boundary effects)
- Saves memory without sacrificing accuracy

Growth Rate Control
-------------------

Control how quickly χ grows during simulation:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=8,                   # Start small
       chi_max_per_bond=256,         # Can grow up to 256
       truncation_threshold=1e-6,
       adaptive_mode=True,
       growth_rate='conservative'    # 'aggressive', 'moderate', 'conservative'
   )

**Growth strategies**:

- **Aggressive**: Allow χ to double quickly, prioritize accuracy
- **Moderate**: Balance between growth and memory
- **Conservative**: Grow χ slowly, prioritize memory efficiency

Memory Budget Management
========================

Global Memory Constraints
--------------------------

Total MPS memory:

.. math::

   M_{\text{total}} = \sum_{i=1}^n d · χ_{i-1} · χ_i · \text{sizeof}(\text{dtype})

where:

- d = 2 for qubits (local dimension)
- χᵢ = bond dimension at bond i
- dtype = complex64 (8 bytes) or complex128 (16 bytes)

**Memory budget** enforcement:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=100,
       bond_dim=16,
       budget_global_mb=8192,  # 8 GB maximum
       adaptive_mode=True
   )

ATLAS-Q dynamically adjusts per-bond χ to respect the budget.

Budget Allocation Strategies
-----------------------------

**1. Uniform allocation**:

All bonds get equal budget:

.. math::

   χ_i = \sqrt{\frac{M_{\text{budget}}}{n · d · \text{sizeof}(\text{dtype})}}

**2. Entropy-proportional allocation**:

Allocate more memory to high-entanglement bonds:

.. math::

   χ_i \propto e^{S_i}

where Sᵢ is entanglement entropy at bond i.

**3. Hierarchical allocation**:

Center bonds get more resources:

.. math::

   χ_i = χ_{\max} · \exp\left(-\frac{|i - n/2|^2}{2σ^2}\right)

Dynamic Rebalancing
-------------------

As simulation progresses, entanglement redistributes. Periodically rebalance χ:

.. code-block:: python

   # Every 100 gates, rebalance bond dimensions
   for epoch in range(10):
       for _ in range(100):
           # Apply gates
           mps.apply_cnot(...)

       # Rebalance based on current entanglement
       mps.rebalance_bond_dimensions(
           strategy='entropy_based',
           total_budget_mb=8192
       )

**Rebalancing algorithm**:

1. Measure entanglement entropy at each bond
2. Redistribute total χ budget proportional to S_i
3. Move to canonical form and re-truncate

Special Cases and Optimizations
================================

Product States
--------------

**Product state**: :math:`|\psi\rangle = |ψ_1\rangle \otimes |ψ_2\rangle \otimes \cdots \otimes |ψ_n\rangle`

**Entanglement**: Zero, S = 0

**Optimal χ**: χ = 1 (no entanglement)

**ATLAS-Q automatically recognizes** product states:

- All singular values except one are zero
- Truncates to χ = 1
- Memory: O(n) instead of O(n·χ²)

Low-Entanglement States
------------------------

**Area law states**: S ∝ log(n)

**Examples**:

- Ground states of gapped 1D Hamiltonians
- Short-time evolution from product states
- Thermal states at low temperature

**Required χ**: χ ∝ n^α for small α

**Performance**: Polynomial scaling in n

Critical States
---------------

**Critical states**: S ∝ log(L) where L is subsystem size

**Examples**:

- Ground states at quantum phase transitions
- Conformal field theory states

**Required χ**: χ ∝ L^α for small α

**Challenge**: Logarithmic growth of χ with system size

**ATLAS-Q strategy**:

- Use MERA for critical systems (if available)
- Or accept larger χ and tighter truncation thresholds

Volume Law Entanglement
------------------------

**Volume law**: S ∝ n

**Examples**:

- Random quantum states
- Deep quantum circuits
- Highly entangled states

**Required χ**: χ ∝ 2^n (exponential!)

**Performance**: MPS inefficient, consider alternative methods

Truncation-Free Methods
========================

Projected Time Evolution
-------------------------

Avoid explicit SVD truncation by projecting onto low-bond-dimension manifold:

.. math::

   |\psi(t + dt)\rangle = P_χ \left[ e^{-iH dt} |\psi(t)\rangle \right]

where P_χ is projection onto χ-dimensional bond space.

**Advantage**: No SVD cost (O(χ³) → 0)

**Disadvantage**: More complex implementation

Time-Dependent Variational Principle (TDVP)
--------------------------------------------

Evolve MPS in tangent space of MPS manifold:

.. math::

   \frac{d}{dt} |\psi(t)\rangle = -i P_{\text{MPS}} H |\psi(t)\rangle

**Advantage**: Automatic truncation without explicit SVD

**Implementation**: ATLAS-Q's TDVP algorithm

**See**: :doc:`algorithms` for TDVP details

Implementation in ATLAS-Q
=========================

Truncation Workflow
-------------------

**Step-by-step** ATLAS-Q truncation:

1. **Gate application**: Contract gate with two neighboring MPS tensors

   .. math::

      Θ_{α,s,s',β} = \sum_{γ} A^{[i]}_{α,s,γ} · G_{s,s',t,t'} · A^{[i+1]}_{γ,t,β}

2. **Reshape**: Θ → matrix of shape (χ_left × d) × (d × χ_right)

3. **SVD**: Θ = U Σ V†

4. **Energy criterion**:

   .. code-block:: python

      cumsum = torch.cumsum(singular_values**2, dim=0)
      total = cumsum[-1]
      threshold = (1 - eps**2) * total
      chi_energy = torch.searchsorted(cumsum, threshold) + 1

5. **Apply caps**:

   .. code-block:: python

      chi = min(chi_energy, chi_max_per_bond)

6. **Truncate**:

   .. code-block:: python

      U_trunc = U[:, :chi]
      S_trunc = S[:chi]
      Vt_trunc = Vt[:chi, :]

7. **Split tensors**: Form new MPS tensors A[i] and A[i+1]

8. **Track error**:

   .. code-block:: python

      truncation_error = torch.sqrt(torch.sum(S[chi:]**2))
      mps.statistics.total_truncation_error += truncation_error**2

GPU Optimization
----------------

**ATLAS-Q optimizations**:

- Use cuQuantum for large SVD (χ > 128)
- Batch SVD operations when possible
- Fuse truncation with other operations
- Asynchronous SVD on GPU while CPU prepares next gate

**Performance**: 2-5× speedup for χ > 128

Numerical Stability
-------------------

**Issues**:

- Small singular values near machine precision
- Conditioning of SVD
- Accumulation of rounding errors

**Solutions**:

- Use relative thresholds: discard λ < max(λ) × 1e-14
- Regularize ill-conditioned SVD
- Use higher precision (complex128) when needed

**See**: :doc:`numerical_stability` for details

Comparison with Other Methods
==============================

Fixed χ Truncation
------------------

**Method**: Always truncate to fixed χ regardless of entanglement

**Pros**:
- Predictable cost
- Simple implementation

**Cons**:
- Wastes resources (low entanglement)
- Loses accuracy (high entanglement)

**ATLAS-Q**: Adaptive χ superior for most use cases

Randomized SVD
--------------

**Method**: Approximate SVD using randomized algorithms

**Pros**:
- Faster for very large χ (> 1000)
- O(χ²) instead of O(χ³) for full SVD

**Cons**:
- Approximate (additional error)
- Complex implementation

**ATLAS-Q**: Uses for χ > 512 when available

Density Matrix Truncation
--------------------------

**Method**: Form reduced density matrix, diagonalize, keep top eigenvectors

**Pros**:
- Directly optimizes fidelity
- Equivalent to SVD for pure states

**Cons**:
- More expensive (O(χ⁴) vs O(χ³) for SVD)
- Only applicable to pure states

**ATLAS-Q**: SVD preferred (equivalent, faster)

Summary
=======

Adaptive truncation enables efficient MPS simulation by:

1. **Optimal approximation**: SVD minimizes distance to exact state
2. **Error bounds**: Truncation error quantified by discarded singular values
3. **Adaptive growth**: Bond dimension grows only when entanglement requires it
4. **Memory budgets**: Global constraints automatically enforced
5. **Numerical stability**: Robust implementation with error tracking

**Key concepts**:

- SVD truncation is optimal (Eckart-Young-Mirsky theorem)
- Energy criterion: Keep singular values until (1-ε²) norm retained
- Global error: :math:`ε_{\text{global}} \leq \sqrt{N} · ε_{\text{local}}`
- Adaptive strategies: Entropy-based, per-bond caps, memory budgets
- Special cases: Product states (χ=1), area law (small χ), volume law (large χ)

**Truncation parameters**:

- ``truncation_threshold``: Controls accuracy (1e-6 recommended)
- ``chi_max_per_bond``: Caps maximum χ per bond
- ``budget_global_mb``: Enforces total memory constraint
- ``adaptive_mode``: Enables adaptive χ growth

**Practical guidelines**:

- Start with truncation_threshold = 1e-6, chi_max = 256
- Monitor global error with ``mps.statistics.total_truncation_error``
- Tighten thresholds if error exceeds tolerance
- Use per-bond caps to optimize memory allocation
- Consider memory budgets for very large systems

Adaptive truncation is the algorithmic innovation enabling ATLAS-Q to simulate
50-100+ qubit systems while maintaining controllable approximation error.

See Also
========

- :doc:`tensor_networks`: Tensor network foundations
- :doc:`algorithms`: DMRG, TDVP, and other algorithms using truncation
- :doc:`numerical_stability`: Numerical considerations for truncation
- :doc:`../tutorials/mps_basics`: Practical truncation usage
- :doc:`../howtos/handle_large_systems`: Large-scale simulations with truncation
