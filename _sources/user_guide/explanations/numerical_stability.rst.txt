====================
Numerical Stability
====================

Numerical stability is critical for reliable quantum simulation with MPS. This document explains sources of numerical error, condition number analysis, robust linear algebra techniques, mixed precision strategies, and best practices for maintaining accuracy throughout long simulations.

.. contents::
   :local:
   :depth: 2

Overview
========

Why Numerical Stability Matters
--------------------------------

MPS simulations involve:

1. **Many operations**: 1000+ gate applications in deep circuits
2. **Ill-conditioned linear algebra**: SVD of matrices with widely varying singular values
3. **Error accumulation**: Small errors compound over time
4. **Truncation approximations**: Discarding information must be controlled

**Consequences of poor stability**:

- SVD convergence failures (simulation crash)
- Negative probabilities in measurement
- Norm drift (:math:`\|\psi\| \neq 1`)
- Unphysical results (energy not conserved in TDVP)
- Wrong ground state in VQE

**Goals**:

1. Maintain accuracy within user-specified tolerances
2. Detect numerical issues early
3. Apply corrective measures automatically when possible
4. Provide diagnostics for debugging

Floating-Point Arithmetic
==========================

IEEE 754 Standard
-----------------

Modern GPUs and CPUs use IEEE 754 floating-point representation:

**Complex64 (single precision complex)**:

- **Real/Imag parts**: 32-bit float (1 sign + 8 exponent + 23 mantissa bits)
- **Machine epsilon**: :math:`\epsilon \approx 1.2 \times 10^{-7}`
- **Relative accuracy**: ~7 decimal digits
- **Range**: :math:`\pm 10^{\pm 38}`

**Complex128 (double precision complex)**:

- **Real/Imag parts**: 64-bit float (1 sign + 11 exponent + 52 mantissa bits)
- **Machine epsilon**: :math:`\epsilon \approx 2.2 \times 10^{-16}`
- **Relative accuracy**: ~16 decimal digits
- **Range**: :math:`\pm 10^{\pm 308}`

Rounding Errors
---------------

Every floating-point operation introduces rounding error:

.. math::

   \text{fl}(a \odot b) = (a \odot b)(1 + \delta), \quad |\delta| \leq \epsilon_{\text{mach}}

where :math:`\odot` is any arithmetic operation (+, -, ×, ÷).

**Example**: Sum of n numbers accumulates error :math:`O(n \epsilon)`:

.. code-block:: python

   import torch

   # Accumulation of rounding errors
   n = 10000
   x = torch.ones(n, dtype=torch.float32) / n

   # Exact sum should be 1.0
   sum_forward = x.sum()
   print(f"Forward sum: {sum_forward:.15f}")  # ~1.000000119

   # Better: Kahan summation (compensated summation)
   from atlas_q.numerics import kahan_sum
   sum_compensated = kahan_sum(x)
   print(f"Compensated sum: {sum_compensated:.15f}")  # Closer to 1.0

Loss of Significance
--------------------

Subtracting nearly equal numbers causes catastrophic cancellation:

.. math::

   a - b = (1.23456789 - 1.23456780) \times 10^{8} = 9.0 \times 10^{1}

Leading digits cancel, leaving only low-order (inaccurate) digits.

**ATLAS-Q mitigation**: Use numerically stable formulas. Example:

.. code-block:: python

   # Bad: Numerical cancellation when x ≈ 0
   def bad_formula(x):
       return (torch.exp(x) - 1) / x

   # Good: Use Taylor series for small |x|
   def stable_formula(x):
       # exp(x) - 1 = x + x²/2 + x³/6 + ...
       # (exp(x) - 1) / x = 1 + x/2 + x²/6 + ...
       small_x = torch.abs(x) < 1e-4
       result = torch.empty_like(x)
       result[small_x] = 1.0 + x[small_x] / 2.0 + x[small_x]**2 / 6.0
       result[~small_x] = (torch.exp(x[~small_x]) - 1) / x[~small_x]
       return result

Sources of Numerical Error
===========================

1. Floating-Point Arithmetic
-----------------------------

**Magnitude**: :math:`O(\epsilon_{\text{mach}})` per operation

**Accumulation**: After :math:`n` operations, error grows to :math:`O(n \epsilon_{\text{mach}})`

**Example**: Gate application involves ~10 tensor contractions

- Complex64: Error per gate ~ :math:`10^{-6}`
- Complex128: Error per gate ~ :math:`10^{-15}`

For 1000 gates:
- Complex64: Accumulated error ~ :math:`10^{-3}` (may be significant)
- Complex128: Accumulated error ~ :math:`10^{-12}` (usually negligible)

2. Ill-Conditioned Matrices
----------------------------

**Definition**: Matrix :math:`A` is ill-conditioned if small perturbations cause large changes in solutions.

**Condition number**:

.. math::

   \kappa(A) = \|A\| \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}

**Error amplification**:

.. math::

   \frac{\|\Delta x\|}{\|x\|} \leq \kappa(A) \frac{\|\Delta b\|}{\|b\|}

where :math:`Ax = b`.

**Example in MPS**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   import torch

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply many gates - bond tensors become ill-conditioned
   for i in range(100):
       for j in range(29):
           mps.apply_cnot(j, j+1)

   # Check condition number of a bond tensor
   tensor = mps.tensors[15].reshape(-1, mps.tensors[15].shape[-1])
   U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
   cond = S.max() / S.min()
   print(f"Condition number: {cond:.2e}")  # May be 10^6 or higher

   if cond > 1e6:
       print("Warning: Ill-conditioned tensor - consider using complex128")

3. SVD Truncation Error
------------------------

Truncation discards small singular values:

.. math::

   \|\Theta - \tilde{\Theta}\|_F = \sqrt{\sum_{i=\chi+1}^{r} \sigma_i^2}

**Local error** (single truncation): Typically :math:`10^{-8}` to :math:`10^{-12}`

**Global error** (many truncations): Bounded by root-sum-square:

.. math::

   \epsilon_{\text{global}} \leq \sqrt{\sum_{t=1}^{N} \epsilon_t^2}

**Monitoring in ATLAS-Q**:

.. code-block:: python

   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=64,
       truncation_threshold=1e-10,
       device='cuda'
   )

   # Apply gates
   for i in range(100):
       mps.apply_h(i % 40)
       mps.apply_cnot(i % 39, (i % 39) + 1)

   # Check accumulated error
   global_error = mps.statistics.total_truncation_error
   print(f"Global truncation error: {global_error:.2e}")

   if global_error > 1e-5:
       print("Warning: Accumulated truncation error may affect results")

4. Canonical Form Drift
------------------------

MPS tensors should maintain orthonormality constraints (canonical form), but rounding errors cause drift:

.. math::

   \sum_{\alpha,s} A^{[i]}_{\alpha,s,\beta} (A^{[i]}_{\alpha,s,\beta'})^* \approx \delta_{\beta,\beta'}

**Drift accumulation**: After many operations without recanonicalizing, orthonormality is lost.

**Consequences**:

- Incorrect overlap calculations
- Norm drift (:math:`\langle\psi|\psi\rangle \neq 1`)
- Unstable TDVP evolution

**Mitigation**:

.. code-block:: python

   # Recanonicalize periodically
   for step in range(1000):
       mps.apply_cnot(step % 39, (step % 39) + 1)

       if step % 100 == 0:
           mps.canonicalize(center=20)  # Restore orthonormality
           mps.normalize()               # Ensure unit norm

Condition Numbers
=================

Mathematical Definition
-----------------------

For a matrix :math:`A`, the condition number in the 2-norm is:

.. math::

   \kappa(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}

**Interpretation**:

- :math:`\kappa(A) = 1`: Perfectly conditioned (e.g., unitary matrices)
- :math:`\kappa(A) < 10^3`: Well-conditioned
- :math:`10^3 < \kappa(A) < 10^6`: Moderately conditioned
- :math:`\kappa(A) > 10^6`: Ill-conditioned (numerical issues likely)
- :math:`\kappa(A) > 1/\epsilon_{\text{mach}}`: Effectively singular

Computing Condition Numbers
---------------------------

.. code-block:: python

   import torch

   def condition_number(tensor):
       """
       Compute condition number of 2D tensor.
       """
       U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
       return (S.max() / S.min()).item()

   # Example: Check MPS tensor conditioning
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply deep circuit
   for layer in range(50):
       for i in range(29):
           mps.apply_cnot(i, i+1)

   # Check conditioning at each site
   for i, tensor in enumerate(mps.tensors):
       mat = tensor.reshape(-1, tensor.shape[-1])
       cond = condition_number(mat)
       if cond > 1e6:
           print(f"Site {i}: condition number = {cond:.2e} (ill-conditioned!)")

Impact on Linear Algebra
-------------------------

**SVD stability**: SVD is numerically stable even for ill-conditioned matrices. Small singular values are computed accurately (relative to their magnitude).

**Matrix inversion**: Inverting ill-conditioned matrix amplifies errors by factor :math:`\kappa(A)`.

**Linear solves**: Solving :math:`Ax = b` loses :math:`\log_{10}(\kappa(A))` digits of accuracy.

**Example**:

.. code-block:: python

   # Ill-conditioned 4×4 Hilbert matrix
   H = torch.tensor([
       [1.0, 1/2, 1/3, 1/4],
       [1/2, 1/3, 1/4, 1/5],
       [1/3, 1/4, 1/5, 1/6],
       [1/4, 1/5, 1/6, 1/7]
   ], dtype=torch.float64)

   cond = condition_number(H)
   print(f"Condition number: {cond:.2e}")  # ~15,514

   # Solve Hx = b
   b = torch.randn(4, dtype=torch.float64)
   x = torch.linalg.solve(H, b)

   # Check residual
   residual = torch.norm(H @ x - b) / torch.norm(b)
   print(f"Relative residual: {residual:.2e}")  # Should be small

   # Perturb b slightly
   b_perturbed = b + 1e-10 * torch.randn(4, dtype=torch.float64)
   x_perturbed = torch.linalg.solve(H, b_perturbed)

   # Large change in solution due to ill-conditioning
   relative_change = torch.norm(x_perturbed - x) / torch.norm(x)
   print(f"Relative change in x: {relative_change:.2e}")  # ~ κ(H) * 1e-10

Canonical Forms and Orthogonality
==================================

Canonical forms maintain numerical stability by enforcing orthonormality constraints.

Left-Canonical Form
-------------------

Site tensors satisfy:

.. math::

   \sum_{\alpha,s} A^{[i]}_{\alpha,s,\beta} (A^{[i]}_{\alpha,s,\beta'})^* = \delta_{\beta,\beta'}

**Interpretation**: Right indices of :math:`A^{[i]}` form orthonormal basis.

**Construction via QR decomposition**:

.. code-block:: python

   import torch

   def left_canonicalize_site(tensor):
       """
       Left-canonicalize MPS tensor via QR.

       Input: tensor of shape (chi_left, d, chi_right)
       Output: Q (left-canonical), R (to absorb into next site)
       """
       chi_left, d, chi_right = tensor.shape
       mat = tensor.reshape(chi_left * d, chi_right)
       Q, R = torch.linalg.qr(mat)
       Q_tensor = Q.reshape(chi_left, d, -1)
       return Q_tensor, R

   # Example: Left-canonicalize entire MPS
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

   # Apply some gates
   for i in range(19):
       mps.apply_cnot(i, i+1)

   # Left-canonicalize (sweep left-to-right)
   mps.canonicalize(direction='left')

   # Verify orthonormality
   tensor = mps.tensors[10]
   chi_left, d, chi_right = tensor.shape
   mat = tensor.reshape(chi_left * d, chi_right)
   should_be_identity = mat.T.conj() @ mat
   error = torch.norm(should_be_identity - torch.eye(chi_right, device='cuda'))
   print(f"Orthonormality error: {error:.2e}")  # Should be ~ 1e-7 (complex64)

Right-Canonical Form
--------------------

Site tensors satisfy:

.. math::

   \sum_{\beta,s} A^{[i]}_{\alpha,s,\beta} (A^{[i]}_{\alpha',s,\beta})^* = \delta_{\alpha,\alpha'}

**Interpretation**: Left indices of :math:`A^{[i]}` form orthonormal basis.

**Construction via RQ decomposition** (QR on transpose).

Mixed-Canonical Form
--------------------

Combination of left and right canonical:

- Sites :math:`i < c`: Left-canonical
- Site :math:`c`: Center (no constraint)
- Sites :math:`i > c`: Right-canonical

**Benefits**:

1. **Efficient norm calculation**: :math:`\|\psi\|^2 = \text{Tr}(A^{[c]\dagger} A^{[c]})`
2. **Efficient overlap**: :math:`\langle\phi|\psi\rangle` computed via single contraction at center
3. **TDVP stability**: Tangent space projection requires canonical form

**ATLAS-Q canonicalization**:

.. code-block:: python

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply gates (loses canonicalization)
   for i in range(200):
       mps.apply_cnot(i % 29, (i % 29) + 1)

   # Recanonicalize with center at site 15
   mps.canonicalize(center=15)

   # Now sites 0-14 are left-canonical, 16-29 are right-canonical

Importance for TDVP
-------------------

TDVP requires canonical form for numerical stability:

.. code-block:: python

   from atlas_q.tdvp import TDVP

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')
   # mps starts in canonical form

   # TDVP maintains canonicalization internally
   tdvp = TDVP(
       mps=mps,
       hamiltonian=hamiltonian_mpo,
       dt=0.05,
       method='one_site',
       normalize=True,       # Renormalize each step
       check_canonical=True  # Verify canonical form (debug mode)
   )

   for step in range(100):
       energy = tdvp.evolve_step()

       if step % 10 == 0:
           # Check if canonicalization maintained
           orthogonality_error = mps.check_orthogonality()
           print(f"Step {step}: orthogonality error = {orthogonality_error:.2e}")

Truncation Error Analysis
==========================

Local Truncation Error
----------------------

Single truncation discards singular values :math:`\sigma_{\chi+1}, \sigma_{\chi+2}, \ldots`:

.. math::

   \epsilon_{\text{local}} = \|\Theta - \tilde{\Theta}\|_F = \sqrt{\sum_{i=\chi+1}^{r} \sigma_i^2}

**Optimal approximation** (Eckart-Young-Mirsky theorem): SVD truncation minimizes Frobenius norm error.

**Energy-based truncation**:

.. math::

   \sum_{\alpha=1}^{\chi} \sigma_\alpha^2 \geq (1 - \epsilon^2) \sum_{\alpha=1}^{r} \sigma_\alpha^2

Retains singular values until target fraction of weight captured.

Global Error Bounds
-------------------

After :math:`N` truncations with local errors :math:`\epsilon_1, \epsilon_2, \ldots, \epsilon_N`:

**Best-case bound** (errors cancel):

.. math::

   \epsilon_{\text{global}} \leq \sum_{i=1}^{N} \epsilon_i

**Typical bound** (root-sum-square):

.. math::

   \epsilon_{\text{global}} \leq \sqrt{\sum_{i=1}^{N} \epsilon_i^2}

**Conservative bound**: Depends on circuit structure and error accumulation pattern.

**ATLAS-Q tracking**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=64,
       truncation_threshold=1e-10,
       track_error=True,  # Enable error tracking
       device='cuda'
   )

   # Apply 500 gates
   for i in range(500):
       mps.apply_h(i % 40)
       mps.apply_cnot(i % 39, (i % 39) + 1)

   # Access error statistics
   stats = mps.statistics
   print(f"Total truncations: {stats.num_truncations}")
   print(f"Max local error: {stats.max_truncation_error:.2e}")
   print(f"Average local error: {stats.avg_truncation_error:.2e}")
   print(f"Global error (RSS): {stats.total_truncation_error:.2e}")

   # Per-bond error distribution
   import matplotlib.pyplot as plt
   plt.bar(range(len(stats.truncation_errors_per_bond)),
           stats.truncation_errors_per_bond)
   plt.xlabel('Bond index')
   plt.ylabel('Accumulated error')
   plt.title('Truncation Error Distribution')
   plt.show()

Error Mitigation Strategies
----------------------------

**1. Adaptive truncation threshold**:

.. code-block:: python

   # Start with aggressive truncation, tighten if error grows
   mps = AdaptiveMPS(
       num_qubits=50,
       bond_dim=64,
       truncation_threshold=1e-8,  # Initial threshold
       adaptive_mode=True,
       device='cuda'
   )

   max_global_error = 1e-5

   for i in range(1000):
       mps.apply_cnot(i % 49, (i % 49) + 1)

       # Check global error periodically
       if i % 100 == 0:
           global_error = mps.statistics.total_truncation_error
           if global_error > max_global_error * 0.8:
               # Tighten threshold
               mps.truncation_threshold *= 0.1
               print(f"Tightened truncation to {mps.truncation_threshold:.2e}")

**2. Increase bond dimension**:

If global error exceeds tolerance despite tight truncation threshold, increase :math:`\chi`.

**3. Periodic compression**:

Apply truncation less frequently to reduce accumulated error:

.. code-block:: python

   # Truncate every 10 gates instead of every gate
   mps = AdaptiveMPS(num_qubits=30, bond_dim=128, device='cuda')

   for i in range(100):
       # Apply 10 gates without truncation
       for j in range(10):
           gate_index = i * 10 + j
           mps.apply_cnot(gate_index % 29, (gate_index % 29) + 1,
                         truncate=False)

       # Single truncation after 10 gates
       mps.compress()  # Global compression

Robust Linear Algebra
======================

Robust SVD
----------

ATLAS-Q implements robust SVD with multiple fallback strategies:

.. code-block:: python

   from atlas_q.linalg_robust import robust_svd
   import torch

   def robust_svd(tensor, threshold=1e-14):
       """
       Robust SVD with automatic fallbacks.

       Fallback sequence:
       1. torch.linalg.svd on GPU
       2. Add Tikhonov regularization if condition number high
       3. Fallback to CPU if GPU fails
       4. Try different LAPACK drivers (gesvd, gesvdj)
       """
       try:
           # Try standard GPU SVD
           U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
       except RuntimeError as e:
           # GPU SVD failed - check for ill-conditioning
           if "convergence" in str(e).lower():
               # Add Tikhonov regularization
               reg = threshold * torch.eye(tensor.shape[-1],
                                            device=tensor.device,
                                            dtype=tensor.dtype)
               regularized = tensor @ tensor.T.conj() + reg
               U, S_squared, _ = torch.linalg.svd(regularized)
               S = torch.sqrt(S_squared)
               Vh = (U.T.conj() @ tensor) / S.unsqueeze(1)
           else:
               # Fallback to CPU
               tensor_cpu = tensor.cpu()
               U, S, Vh = torch.linalg.svd(tensor_cpu, full_matrices=False)
               U = U.to(tensor.device)
               S = S.to(tensor.device)
               Vh = Vh.to(tensor.device)

       # Filter small singular values
       keep = S > threshold
       return U[:, keep], S[keep], Vh[keep, :]

**Usage in ATLAS-Q**:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS

   # robust_svd used automatically in adaptive truncation
   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=128,
       use_robust_linalg=True,  # Enable robust linear algebra
       device='cuda'
   )

Robust QR Decomposition
-----------------------

QR with column pivoting for rank-deficient matrices:

.. code-block:: python

   def robust_qr(tensor, threshold=1e-14):
       """
       QR decomposition with column pivoting for numerical stability.
       """
       Q, R, P = torch.linalg.qr(tensor, mode='reduced', pivot=True)

       # Truncate small diagonal elements of R
       diag_R = torch.diag(R)
       keep = torch.abs(diag_R) > threshold

       Q_truncated = Q[:, keep]
       R_truncated = R[keep, :]

       # Undo permutation
       P_inv = torch.argsort(P)
       R_truncated = R_truncated[:, P_inv]

       return Q_truncated, R_truncated

Eigendecomposition Stability
-----------------------------

For Hermitian matrices (e.g., density matrices, Hamiltonians):

.. code-block:: python

   def robust_eigh(tensor, threshold=1e-14):
       """
       Robust eigendecomposition for Hermitian matrices.
       """
       # Ensure Hermiticity (symmetrize)
       tensor_sym = (tensor + tensor.T.conj()) / 2

       # Eigendecomposition
       eigenvalues, eigenvectors = torch.linalg.eigh(tensor_sym)

       # Filter small/negative eigenvalues
       keep = eigenvalues > threshold
       return eigenvalues[keep], eigenvectors[:, keep]

   # Example: Reduced density matrix
   from atlas_q.adaptive_mps import AdaptiveMPS

   mps = AdaptiveMPS(num_qubits=20, bond_dim=64, device='cuda')

   # Reduced density matrix for sites 0-9
   rho = mps.reduced_density_matrix(sites=range(10))

   # Eigendecomposition
   eigenvalues, eigenvectors = robust_eigh(rho)

   # Check: eigenvalues should sum to 1, all non-negative
   assert torch.abs(eigenvalues.sum() - 1.0) < 1e-6
   assert torch.all(eigenvalues >= -1e-10)  # Allow small numerical negative

Mixed Precision Strategies
===========================

Complex64 vs Complex128
-----------------------

**Complex64** (default):

- **Advantages**: 2× faster, 2× less memory
- **Sufficient for**: Most quantum circuits, VQE with shallow ansätze, QAOA
- **Limitations**: Accumulates error faster, less stable for ill-conditioned problems

**Complex128**:

- **Advantages**: Higher precision, better stability for long simulations
- **Required for**: Long-time TDVP, deep circuits (>1000 gates), high-precision chemistry
- **Cost**: 2× slower, 2× more memory

Automatic Precision Promotion
------------------------------

ATLAS-Q can automatically promote to higher precision when detecting instability:

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy
   import torch

   # Configure automatic promotion policy
   policy = DTypePolicy(
       default=torch.complex64,
       promote_if_cond_gt=1e6,        # Promote if condition number > 10^6
       promote_if_error_gt=1e-5,      # Promote if global error > 10^-5
       promote_to=torch.complex128
   )

   mps = AdaptiveMPS(
       num_qubits=40,
       bond_dim=64,
       dtype_policy=policy,
       device='cuda'
   )

   # Simulation proceeds with complex64
   for i in range(500):
       mps.apply_cnot(i % 39, (i % 39) + 1)

       # Automatic promotion if instability detected
       if mps.statistics.promoted_to_higher_precision:
           print(f"Promoted to complex128 at gate {i}")
           break

Manual Precision Control
------------------------

.. code-block:: python

   # Always use complex128 for maximum stability
   mps = AdaptiveMPS(
       num_qubits=30,
       bond_dim=128,
       dtype=torch.complex128,
       device='cuda'
   )

   # Or convert existing MPS
   mps = mps.to(dtype=torch.complex128)

Diagnostics and Monitoring
===========================

Built-in Diagnostics
--------------------

.. code-block:: python

   from atlas_q.adaptive_mps import AdaptiveMPS
   from atlas_q.diagnostics import MPSDiagnostics

   mps = AdaptiveMPS(num_qubits=30, bond_dim=64, device='cuda')

   # Apply circuit
   for i in range(200):
       mps.apply_cnot(i % 29, (i % 29) + 1)

   # Run comprehensive diagnostics
   diag = MPSDiagnostics(mps)

   # Check norm
   norm = diag.check_norm()
   print(f"MPS norm: {norm:.10f}")  # Should be close to 1.0
   if abs(norm - 1.0) > 1e-6:
       print("Warning: Norm drift detected")

   # Check orthogonality (if in canonical form)
   orth_error = diag.check_orthogonality()
   print(f"Orthogonality error: {orth_error:.2e}")

   # Check for unphysical values
   has_nan = diag.check_for_nan()
   has_inf = diag.check_for_inf()
   print(f"Contains NaN: {has_nan}, Contains Inf: {has_inf}")

   # Condition numbers
   max_cond = diag.max_condition_number()
   print(f"Max condition number: {max_cond:.2e}")

   # Entanglement entropy sanity check
   max_entropy = diag.max_entanglement_entropy()
   theoretical_max = torch.log(torch.tensor(2.0)) * 15  # For half-system
   print(f"Max entropy: {max_entropy:.2f}, Theoretical max: {theoretical_max:.2f}")

Energy Conservation (TDVP)
--------------------------

For Hamiltonian time evolution, energy should be conserved:

.. code-block:: python

   from atlas_q.tdvp import TDVP

   tdvp = TDVP(mps=mps, hamiltonian=hamiltonian_mpo, dt=0.05, method='one_site')

   energies = []
   for step in range(500):
       E = tdvp.evolve_step()
       energies.append(E)

       if step % 50 == 0:
           energy_drift = abs(E - energies[0])
           print(f"Step {step}: E = {E:.8f}, drift = {energy_drift:.2e}")

   # Check energy conservation
   max_drift = max(abs(E - energies[0]) for E in energies)
   if max_drift > 1e-6:
       print(f"Warning: Energy drift {max_drift:.2e} exceeds tolerance")
       print("Consider: reducing time step, using complex128, or 2-site TDVP")

Best Practices
==============

General Guidelines
------------------

1. **Start with complex64**, promote to complex128 if needed
2. **Monitor condition numbers** periodically
3. **Canonicalize regularly** (every 100-500 gates)
4. **Track truncation error**, ensure global error acceptable
5. **Check norm** periodically (:math:`\|\psi\| \approx 1`)
6. **Use appropriate bond dimension** (too small → large truncation error)
7. **Enable robust linear algebra** for production runs
8. **Profile first**, optimize second (don't prematurely use complex128)

Algorithm-Specific Recommendations
-----------------------------------

**TDVP**:

- Use complex128 for long-time evolution (t > 100)
- Monitor energy conservation
- Use 2-site TDVP for better accuracy
- Reduce time step if energy drifts

**VQE**:

- complex64 sufficient for most ansätze
- complex128 for chemical accuracy (kcal/mol precision)
- Monitor gradient norm (should decrease to near-zero at convergence)

**QAOA**:

- complex64 usually sufficient (shallow circuits)
- Check approximation ratio consistency across runs

Common Issues and Solutions
===========================

Issue 1: SVD Convergence Failure
---------------------------------

**Symptom**: ``RuntimeError: SVD did not converge``

**Causes**:

- Ill-conditioned matrix (κ > 10^10)
- NaN or Inf in tensor
- GPU numerical instability

**Solutions**:

1. Enable robust SVD: ``use_robust_linalg=True``
2. Increase precision: ``dtype=torch.complex128``
3. Add regularization (automatic in robust_svd)
4. Fallback to CPU for this operation

Issue 2: Negative Probabilities
--------------------------------

**Symptom**: Measurement probabilities < 0

**Causes**:

- Truncation error accumulation
- Loss of canonicalization
- Numerical instability

**Solutions**:

1. Tighten truncation threshold
2. Recanonicalize: ``mps.canonicalize()``
3. Increase bond dimension
4. Use complex128

Issue 3: Norm Drift
--------------------

**Symptom**: :math:`\|\psi\|^2 \neq 1`

**Causes**:

- Rounding error accumulation
- Improper canonicalization
- Truncation without renormalization

**Solutions**:

1. Renormalize: ``mps.normalize()``
2. Recanonicalize: ``mps.canonicalize()``
3. Enable automatic renormalization in algorithm (e.g., TDVP ``normalize=True``)

Issue 4: Energy Not Conserved (TDVP)
-------------------------------------

**Symptom**: Energy drifts over time

**Causes**:

- Time step too large
- Bond dimension insufficient
- Numerical instability (complex64)

**Solutions**:

1. Reduce time step: ``dt = dt / 2``
2. Use 2-site TDVP: ``method='two_site'``
3. Increase bond dimension
4. Use complex128
5. Tighten truncation threshold

Summary
=======

Numerical stability in MPS simulation requires careful attention to:

**Error Sources**:

- **Floating-point arithmetic**: :math:`O(\epsilon_{\text{mach}})` per operation
- **Ill-conditioned matrices**: Amplify errors by factor :math:`\kappa(A)`
- **Truncation**: Controllable via threshold and bond dimension
- **Accumulation**: Errors compound over many operations

**Key Techniques**:

1. **Canonical forms**: Maintain orthonormality for stability
2. **Robust linear algebra**: Automatic fallbacks for ill-conditioned problems
3. **Mixed precision**: complex64 default, complex128 when needed
4. **Error tracking**: Monitor global truncation error
5. **Diagnostics**: Check norm, orthogonality, condition numbers

**Practical Workflow**:

1. Start with complex64, :math:`\chi` = 64, ``truncation_threshold`` = :math:`10^{-10}`
2. Monitor diagnostics (norm, energy conservation, global error)
3. If issues arise: tighten threshold → increase :math:`\chi` → use complex128
4. Enable robust linear algebra for production runs
5. Canonicalize periodically (every 100-500 operations)

**When to Use Complex128**:

- Long-time TDVP evolution (t > 100)
- Deep circuits (> 1000 gates)
- Chemical accuracy requirements (VQE for molecules)
- Condition numbers :math:`> 10^6` detected
- Persistent instability symptoms

For algorithm-specific stability considerations, see:

- :doc:`algorithms` - TDVP, VQE, DMRG stability properties
- :doc:`adaptive_truncation` - Error control in truncation
- :doc:`../howtos/debug_simulations` - Debugging numerical issues
- :doc:`../howtos/optimize_performance` - Performance vs accuracy tradeoffs
