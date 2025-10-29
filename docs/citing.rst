Citing ATLAS-Q
==============

If you use ATLAS-Q in academic work, please cite the software using one of the formats below.

BibTeX
------

.. code-block:: bibtex

   @software{atlasq2025,
     title={ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum},
     author={ATLAS-Q Development Team},
     year={2025},
     url={https://github.com/followthsapper/ATLAS-Q},
     version={0.6.1},
     doi={10.5281/zenodo.XXXXXXX}  # Update when DOI is assigned
   }

APA Format
----------

ATLAS-Q Development Team. (2025). *ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum* (Version 0.6.1) [Computer software]. https://github.com/followthsapper/ATLAS-Q

MLA Format
----------

ATLAS-Q Development Team. *ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum*. Version 0.6.1, 2025, https://github.com/followthsapper/ATLAS-Q.

Chicago Format
--------------

ATLAS-Q Development Team. 2025. "ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum." Version 0.6.1. https://github.com/followthsapper/ATLAS-Q.

Text Citation
-------------

For inline citations in text:

    We performed quantum simulations using ATLAS-Q (ATLAS-Q Development Team, 2025), a GPU-accelerated tensor network simulator.

Citing Specific Features
-------------------------

When citing specific algorithms or features, include relevant academic references:

Matrix Product States
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @article{Verstraete2008,
     title={Matrix product states, projected entangled pair states, and variational renormalization group methods for quantum spin systems},
     author={Verstraete, F. and Cirac, J. I. and Murg, V.},
     journal={Advances in Physics},
     volume={57},
     number={2},
     pages={143--224},
     year={2008},
     publisher={Taylor \& Francis}
   }

TDVP Time Evolution
^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @article{Haegeman2011,
     title={Time-Dependent Variational Principle for Quantum Lattices},
     author={Haegeman, Jutho and Cirac, J. Ignacio and Osborne, Tobias J. and Pižorn, Iztok and Verschelde, Henri and Verstraete, Frank},
     journal={Physical Review Letters},
     volume={107},
     pages={070601},
     year={2011},
     publisher={American Physical Society}
   }

Variational Quantum Eigensolver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @article{Peruzzo2014,
     title={A variational eigenvalue solver on a photonic quantum processor},
     author={Peruzzo, Alberto and McClean, Jarrod and Shadbolt, Peter and Yung, Man-Hong and Zhou, Xiao-Qi and Love, Peter J. and Aspuru-Guzik, Alán and O'Brien, Jeremy L.},
     journal={Nature Communications},
     volume={5},
     pages={4213},
     year={2014},
     publisher={Nature Publishing Group}
   }

QAOA
^^^^

.. code-block:: bibtex

   @article{Farhi2014,
     title={A Quantum Approximate Optimization Algorithm},
     author={Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam},
     journal={arXiv preprint arXiv:1411.4028},
     year={2014}
   }

Stabilizer Formalism
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

   @article{AaronsonGottesman2004,
     title={Improved Simulation of Stabilizer Circuits},
     author={Aaronson, Scott and Gottesman, Daniel},
     journal={Physical Review A},
     volume={70},
     pages={052328},
     year={2004},
     publisher={American Physical Society}
   }

Example Citation in Paper
-------------------------

Sample citation in a research paper:

    We simulated the quantum dynamics using the Time-Dependent Variational Principle (TDVP) [Haegeman2011] as implemented in ATLAS-Q [atlasq2025], a GPU-accelerated tensor network framework. The adaptive Matrix Product State representation [Verstraete2008] allowed efficient simulation of 50-qubit systems with bond dimension χ=128, achieving memory compression of 500,000× compared to full statevector methods while maintaining truncation error below 10⁻⁸.

Acknowledgments
---------------

When ATLAS-Q contributes significantly to your work, consider including an acknowledgment:

    This work benefited from simulations performed using ATLAS-Q, a GPU-accelerated quantum tensor network simulator developed by the ATLAS-Q Development Team.

Version Pinning
---------------

For reproducibility, specify the exact version used:

.. code-block:: bibtex

   @software{atlasq2025,
     title={ATLAS-Q: Adaptive Tensor Learning And Simulation – Quantum},
     author={ATLAS-Q Development Team},
     year={2025},
     url={https://github.com/followthsapper/ATLAS-Q},
     version={0.6.1},
     note={git commit: 3d7d144}
   }

Publication List
----------------

If you publish work using ATLAS-Q, we encourage you to let us know so we can list it in the documentation. Submit a pull request or open an issue with your publication details.

Contributing Citations
----------------------

To add your publication to the ATLAS-Q citation list:

1. Fork the repository
2. Add your citation to ``docs/citing.rst``
3. Submit a pull request with the title "Add publication: [Your Paper Title]"

License and Terms
-----------------

ATLAS-Q is released under the MIT License. There are no licensing restrictions on academic or commercial use. Citation is requested but not required.

See :doc:`installation` for installation instructions and :doc:`quickstart` for getting started with ATLAS-Q.
