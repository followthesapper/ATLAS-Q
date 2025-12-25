Explanations
============

Understanding-oriented documentation that clarifies concepts, design decisions, and theoretical foundations of ATLAS-Q.

.. toctree::
   :maxdepth: 2

   tensor_networks
   adaptive_truncation
   algorithms
   gpu_acceleration
   numerical_stability
   performance_model
   design_decisions
   comparisons

Explanation Topics
------------------

:doc:`tensor_networks`
   Matrix Product States (MPS), Matrix Product Operators (MPO), and Projected Entangled Pair States (PEPS). Explains tensor network representation, contraction schemes, and canonical forms.

:doc:`adaptive_truncation`
   Bond dimension selection strategies, energy-based truncation, entropy criteria, and global error bounds. Discusses trade-offs between accuracy and efficiency.

:doc:`algorithms`
   Mathematical foundations of TDVP, VQE, QAOA, stabilizer formalism, and period-finding. Includes complexity analysis and convergence properties.

:doc:`gpu_acceleration`
   GPU optimization strategies, custom Triton kernels, memory management, and cuQuantum integration. Explains performance characteristics and bottlenecks.

:doc:`numerical_stability`
   Sources of numerical error, ill-conditioned matrices, mixed-precision strategies, and robust linear algebra. Discusses condition number monitoring and automatic promotion.

:doc:`performance_model`
   Computational complexity, memory scaling, and performance predictions for different simulation regimes. Helps choose appropriate methods for specific problem sizes.

:doc:`design_decisions`
   Architectural choices, API design rationale, and implementation trade-offs. Explains why ATLAS-Q works the way it does.

:doc:`comparisons`
   Comparison with other quantum simulation frameworks (Qiskit Aer, Cirq, ITensor, TeNPy). Discusses when to use each tool.

Theoretical Background
----------------------

Mathematical notation used throughout ATLAS-Q documentation:

- :math:`|\psi\rangle`: Quantum state vector
- :math:`\chi`: Bond dimension
- :math:`\epsilon`: Truncation tolerance
- :math:`S`: Entanglement entropy
- :math:`\sigma_i`: Singular values
- :math:`\hat{H}`: Hamiltonian operator

Matrix Product State (MPS) representation:

.. math::

   |\psi\rangle = \sum_{s_1,\ldots,s_n} A^{[1]}_{s_1} A^{[2]}_{s_2} \cdots A^{[n]}_{s_n} |s_1 s_2 \ldots s_n\rangle

where :math:`A^{[i]}_{s_i}` are rank-3 tensors with shape :math:`[\chi_{i-1}, 2, \chi_i]`.

Truncation criterion:

.. math::

   \text{keep } k \text{ such that } \sum_{i=1}^k \sigma_i^2 \geq (1 - \epsilon^2) \sum_{i=1}^{r} \sigma_i^2

References
----------

Key papers and resources:

- Verstraete et al., "Matrix product states, projected entangled pair states, and variational renormalization group methods for quantum spin systems," Advances in Physics (2008)
- Haegeman et al., "Time-Dependent Variational Principle for Quantum Lattices," Physical Review Letters (2011)
- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor," Nature Communications (2014)
- Farhi & Goldstone, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028 (2014)
- Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits," Physical Review A (2004)

See :doc:`../../citing` for how to cite ATLAS-Q in academic work.
