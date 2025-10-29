atlas_q.truncation
===================

.. automodule:: atlas_q.truncation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``truncation`` module implements adaptive rank selection for Matrix Product States based on energy-based criteria. Key features include:

- Energy-based truncation with configurable tolerance
- Per-bond dimension caps (``chi_max_per_bond``)
- Global memory budget enforcement
- Entropy and condition number diagnostics
- Local and global error bound computation

Mathematical Foundation
-----------------------

The module implements truncation based on energy retention:

.. math::

   \text{Keep smallest } k \text{ such that } \sum_{i \leq k} \sigma_i^2 \geq (1 - \varepsilon^2) \sum_i \sigma_i^2

Local truncation error:

.. math::

   \varepsilon_{\text{local}}^2 = \sum_{i > k} \sigma_i^2

Global error bound (Frobenius norm):

.. math::

   \varepsilon_{\text{global}} \leq \sqrt{\sum_b \varepsilon_{\text{local},b}^2}

Entanglement entropy at a bond:

.. math::

   S = -\sum_i p_i \log(p_i) \quad \text{where} \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   choose_rank_from_sigma
   compute_global_error_bound
   check_entropy_sanity

choose_rank_from_sigma
----------------------

.. autofunction:: choose_rank_from_sigma

Core adaptive truncation function that selects bond dimension from singular values. Applies multiple constraints in sequence:

1. Energy criterion: retain (1-ε²) of total energy
2. Per-bond cap: enforce ``chi_max_per_bond``
3. Memory budget: reduce rank if global memory exceeded
4. Compute diagnostics: local error, entropy, condition number

Returns selected rank k along with diagnostic information.

compute_global_error_bound
--------------------------

.. autofunction:: compute_global_error_bound

Computes upper bound on global state error from local truncation errors using Frobenius norm. This provides a certificate that the MPS approximation is within specified tolerance of the exact state.

check_entropy_sanity
--------------------

.. autofunction:: check_entropy_sanity

Validates that computed entanglement entropy is physically reasonable. Maximum entropy at a bond is bounded by the smaller Hilbert space dimension. This function checks if measured entropy exceeds theoretical maximum, which would indicate numerical issues.

Examples
--------

Basic truncation decision:

.. code-block:: python

   import torch
   from atlas_q.truncation import choose_rank_from_sigma

   # Singular values from SVD
   S = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1, 0.01])

   # Choose rank with ε=1e-3 tolerance, max χ=4
   k, eps_local, entropy, condS = choose_rank_from_sigma(
       S,
       eps_bond=1e-3,
       chi_cap=4
   )

   print(f"Selected rank: {k}")
   print(f"Local error: {eps_local:.2e}")
   print(f"Entropy: {entropy:.3f}")
   print(f"Condition number: {condS:.2e}")

Truncation with memory budget:

.. code-block:: python

   from atlas_q.truncation import choose_rank_from_sigma

   current_memory = 2000  # MB
   budget = 4096  # MB

   def budget_ok(k):
       # Estimate memory for rank k
       estimated_memory = current_memory + k * k * 16  # bytes per element
       return estimated_memory < budget * 1024 * 1024

   k, eps_local, entropy, condS = choose_rank_from_sigma(
       S,
       eps_bond=1e-6,
       chi_cap=128,
       budget_ok=budget_ok
   )

   print(f"Selected rank respecting budget: {k}")

Computing global error:

.. code-block:: python

   from atlas_q.truncation import compute_global_error_bound

   # Local errors from multiple truncations
   local_errors = [1e-6, 2e-6, 1.5e-6, 3e-6, 1e-6]

   global_error = compute_global_error_bound(local_errors)
   print(f"Global error bound: {global_error:.2e}")

   # Check if within tolerance
   target_error = 1e-5
   if global_error < target_error:
       print("Simulation within target accuracy")

Entropy validation:

.. code-block:: python

   from atlas_q.truncation import check_entropy_sanity

   # Bond with dimensions χ_L=32, χ_R=64
   entropy = 8.5
   is_valid = check_entropy_sanity(entropy, chi_left=32, chi_right=64)

   if not is_valid:
       print("Warning: Entropy exceeds physical bound")

Integration with AdaptiveMPS:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # AdaptiveMPS uses truncation module internally
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=16,
       eps_bond=1e-6,              # Truncation tolerance
       chi_max_per_bond=128,       # Per-bond cap
       budget_global_mb=4096,      # Memory budget
       device='cuda'
   )

   # Apply gates - truncation automatically applied
   # ...

   # Check global error
   global_error = mps.global_error_bound()
   print(f"Total truncation error: {global_error:.2e}")

   # Access per-operation truncation data
   for step, eps in enumerate(mps.statistics.logs['eps_local']):
       bond = mps.statistics.logs['bond'][step]
       print(f"Bond {bond}: ε_local = {eps:.2e}")

Advanced Usage: Custom Truncation Strategy
-------------------------------------------

Implement custom budget function for adaptive memory management:

.. code-block:: python

   from atlas_q.truncation import choose_rank_from_sigma
   import torch

   class AdaptiveBudget:
       def __init__(self, total_bonds, budget_mb):
           self.total_bonds = total_bonds
           self.budget_mb = budget_mb
           self.current_usage = [0] * total_bonds

       def budget_ok_for_bond(self, bond_idx):
           def checker(k):
               # Estimate if setting bond to rank k fits budget
               new_usage = self.current_usage.copy()
               new_usage[bond_idx] = k * k * 16  # bytes
               return sum(new_usage) < self.budget_mb * 1024 * 1024
           return checker

   budget = AdaptiveBudget(total_bonds=20, budget_mb=4096)

   # Per-bond truncation decisions
   for bond in range(20):
       S = torch.randn(128).abs()  # Mock singular values
       S = S.sort(descending=True)[0]

       k, eps, entropy, cond = choose_rank_from_sigma(
           S,
           eps_bond=1e-6,
           chi_cap=128,
           budget_ok=budget.budget_ok_for_bond(bond)
       )

       budget.current_usage[bond] = k * k * 16
       print(f"Bond {bond}: χ={k}, ε={eps:.2e}")

Performance Notes
-----------------

**Computational Cost**

Truncation itself is O(χ):

- Sorting singular values: O(χ log χ)
- Computing cumulative sum: O(χ)
- Entropy calculation: O(χ)

Total cost negligible compared to SVD (O(χ³)).

**Memory-Efficient Strategies**

For large systems, use per-bond caps to limit peak memory:

.. code-block:: python

   # Vary χ by position in chain
   chi_max_per_bond = [32] * 10 + [64] * 10 + [32] * 10  # Higher in middle

This reduces memory by 2-4× for 1D chains with localized entanglement.

**Adaptive vs. Fixed**

- **Fixed χ**: Predictable memory, may over/under-provision
- **Adaptive with ε**: Minimal χ for given accuracy, variable memory
- **Hybrid**: Per-bond caps + energy threshold (recommended)

Best Practices
--------------

**Choosing ε_bond**

- ε = 10⁻⁸: High accuracy, larger χ
- ε = 10⁻⁶: Balanced (recommended)
- ε = 10⁻⁴: Fast but less accurate

Monitor global error; if > 0.01, decrease ε or increase χ_max.

**Per-Bond Optimization**

Analyze entropy profile to set per-bond budgets:

.. code-block:: python

   entropies = [mps.get_entropy(i) for i in range(N-1)]
   chi_max_per_bond = [min(64, int(2**S)) for S in entropies]

Allocates χ proportional to local entanglement.

**Troubleshooting**

- **χ always hits cap**: Increase χ_max or use 2-site algorithm
- **High global error**: Decrease ε_bond or check circuit structure
- **Memory overflow**: Use per-bond caps or reduce χ_max

Use Cases
---------

**Adaptive Truncation**

Essential for:

- Unknown entanglement structure (exploratory simulations)
- Variable entanglement (quantum quenches, time evolution)
- Memory-constrained environments

**Fixed Truncation**

Sufficient for:

- Known entanglement (e.g., ground states of local Hamiltonians)
- Benchmarking (reproducible χ)
- Real-time applications (predictable performance)

See Also
--------

- :doc:`adaptive_mps` - MPS with adaptive truncation
- :doc:`diagnostics` - Entropy and spectral gap analysis
- :doc:`linalg_robust` - SVD providing singular values for truncation
- :doc:`../user_guide/explanations/adaptive_truncation` - Truncation theory

References
----------

.. [Vidal03] G. Vidal, "Efficient classical simulation of slightly entangled quantum computations," *Physical Review Letters* 91, 147902 (2003).

.. [White92] S. R. White, "Density matrix formulation for quantum renormalization groups," *Physical Review Letters* 69, 2863 (1992).

.. [Orus14] R. Orús, "A practical introduction to tensor networks," *Annals of Physics* 349, 117 (2014).
