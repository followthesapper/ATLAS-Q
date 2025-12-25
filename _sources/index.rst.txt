ATLAS-Q Documentation
======================

ATLAS-Q (Adaptive Tensor Learning And Simulation – Quantum) is a GPU-accelerated quantum tensor network simulator implementing Matrix Product States (MPS), Matrix Product Operators (MPO), Projected Entangled Pair States (PEPS), and variational quantum algorithms. The framework provides memory-efficient quantum state representation with adaptive bond dimensions, custom GPU kernels, and specialized backends for Clifford circuits and period-finding.

Version 0.7.0 (December 2025) - IR v1.1.0

**NEW: IR Integration - Pre-Computation Diagnosis**

ATLAS-Q v0.7.0 introduces complete Informational Relativity (IR) integration with **pre-computation regime diagnosis**. This breakthrough enables you to know whether quantum algorithms will succeed *before* running them.

IR v1.1.0 Features:

- **Regime Analyzer**: Diagnose problems as IR (observable), Transition, or AIR (hidden) regimes
- **Pre-Computation GO/NO-GO**: Physics-derived e^-2 threshold (0.135) for trustworthiness
- **4× VQE Circuit Reduction**: Coherence-based measurement grouping
- **42% Period Finding Shot Reduction**: IR preprocessing for QPE
- **100 Qubit MPS Simulation**: 1.56 seconds with 10^25× memory compression
- **Spectral Lifting**: Full M_ij relational matrix analysis for structure detection

Key capabilities:

- **Coherence-Aware VQE/QAOA**: Real-time coherence tracking with GO/NO-GO classification
- **IR Integration**: Circular statistics and RMT-based quality metrics (R̄, V_φ)
- **Hardware-Validated**: Tested on IBM Brisbane with near-ideal coherence (R̄=0.988 for H2O)
- Adaptive MPS with per-bond dimension control and global memory budgets
- Variational algorithms: VQE, QAOA with hardware-efficient and UCCSD ansätze
- Grover's quantum search algorithm with MPO-based oracles (94-100% accuracy)
- Time evolution via TDVP (1-site and 2-site)
- PEPS for 2D tensor networks
- Stabilizer backend for Clifford circuits (20× speedup)
- Circuit cutting and entanglement forging
- Distributed MPS for multi-GPU simulation
- Molecular Hamiltonians via PySCF integration
- Custom Triton kernels for gate operations (1.5-3× speedup)
- cuQuantum backend integration

Performance: 100 qubits in 1.56 seconds, 10^25× memory compression at 100 qubits, 4× circuit reduction with IR grouping.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/tutorials/index
   user_guide/howtos/index
   user_guide/explanations/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   developer/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples/index
   faq
   changelog
   citing

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* `GitHub Repository <https://github.com/followthesapper/ATLAS-Q>`_
* `Issue Tracker <https://github.com/followthesapper/ATLAS-Q/issues>`_
* `PyPI Package <https://pypi.org/project/atlas-quantum/>`_

Citation
--------

.. code-block:: bibtex

   @software{atlasq2025,
     title={ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum},
     author={ATLAS-Q Development Team},
     year={2025},
     url={https://github.com/followthesapper/ATLAS-Q},
     version={0.7.0}
   }

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
