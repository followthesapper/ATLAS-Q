Tutorials
=========

These tutorials provide step-by-step learning paths for ATLAS-Q. Each tutorial is self-contained and builds from basic to advanced concepts.

.. toctree::
   :maxdepth: 2

   beginners
   mps_basics
   vqe_tutorial
   coherence_aware_vqe
   tdvp_tutorial
   molecular_vqe
   qaoa_tutorial
   advanced_features

Tutorials Overview
------------------

:doc:`beginners`
   Introduction to quantum simulation with ATLAS-Q. Covers basic concepts, installation verification, and first simulations. Start here if you are new to ATLAS-Q.

:doc:`mps_basics`
   Matrix Product States fundamentals. Learn tensor network representation, bond dimensions, truncation, and gate application.

:doc:`vqe_tutorial`
   Variational Quantum Eigensolver for ground state finding. Covers ansätze selection, parameter optimization, and convergence analysis.

:doc:`tdvp_tutorial`
   Time-Dependent Variational Principle for quantum dynamics. Learn real-time evolution, quench dynamics, and correlation functions.

:doc:`molecular_vqe`
   Quantum chemistry with VQE. Build molecular Hamiltonians using PySCF, choose appropriate ansätze (Hardware-Efficient vs UCCSD), and compute ground state energies.

:doc:`qaoa_tutorial`
   Quantum Approximate Optimization Algorithm for combinatorial problems. Apply QAOA to MaxCut, graph coloring, and other optimization problems.

:doc:`advanced_features`
   Advanced simulation techniques: circuit cutting, PEPS for 2D circuits, distributed MPS, stabilizer backend, and noise models.

Prerequisites
-------------

All tutorials assume:

- Python 3.9+ installed
- ATLAS-Q installed (see :doc:`../../installation`)
- Basic understanding of quantum mechanics (qubits, gates, measurements)
- Familiarity with NumPy and PyTorch

For molecular chemistry tutorials, install PySCF:

.. code-block:: bash

   pip install pyscf openfermion openfermionpyscf

Jupyter Notebooks
-----------------

Many tutorials are available as interactive Jupyter notebooks in the repository:

.. code-block:: bash

   git clone https://github.com/followthesapper/ATLAS-Q.git
   cd ATLAS-Q
   jupyter notebook ATLAS_Q_Demo.ipynb

Or open directly in Google Colab:

`ATLAS-Q Demo Notebook <https://colab.research.google.com/github/followthesapper/ATLAS-Q/blob/ATLAS-Q/ATLAS_Q_Demo.ipynb>`_
