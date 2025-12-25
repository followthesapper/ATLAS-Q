Developer Documentation
=======================

.. toctree::
   :maxdepth: 2

   contributing
   development_setup
   testing
   documentation
   release_process

Overview
--------

This section documents the development process for ATLAS-Q contributors.

:doc:`contributing`
   Guidelines for contributing code, documentation, and bug reports.

:doc:`development_setup`
   Setting up a development environment, building from source, and configuring tools.

:doc:`testing`
   Test suite organization, writing tests, and running test suites.

:doc:`documentation`
   Building and contributing to documentation.

:doc:`release_process`
   Release workflow, versioning, and deployment procedures.

Quick Start for Contributors
-----------------------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/ATLAS-Q.git
      cd ATLAS-Q

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      venv\Scripts\activate     # Windows

3. Install in development mode:

   .. code-block:: bash

      pip install -e .[dev,gpu]

4. Run tests:

   .. code-block:: bash

      pytest tests/ -v

5. Build documentation:

   .. code-block:: bash

      cd docs
      make html

Code Style
----------

ATLAS-Q follows standard Python conventions:

- PEP 8 style guide
- Black for code formatting (line length 100)
- Ruff for linting
- Type hints throughout
- Numpydoc-style docstrings

Format code before committing:

.. code-block:: bash

   black src/ tests/ --line-length 100
   ruff check src/ tests/ --fix

Pre-commit hooks are recommended:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Architecture Overview
---------------------

ATLAS-Q is organized into layers:

1. **Core Tensor Networks** (``adaptive_mps.py``, ``mps_pytorch.py``)

   - MPS/MPO representation
   - Tensor operations
   - Canonicalization

2. **Algorithms** (``tdvp.py``, ``vqe_qaoa.py``)

   - Variational algorithms
   - Time evolution
   - Optimization

3. **Specialized Backends** (``stabilizer_backend.py``, ``peps.py``, ``cuquantum_backend.py``)

   - Clifford simulation
   - 2D tensor networks
   - GPU acceleration

4. **Utilities** (``diagnostics.py``, ``linalg_robust.py``, ``truncation.py``)

   - Error tracking
   - Robust linear algebra
   - Truncation strategies

5. **GPU Kernels** (``triton_kernels/``)

   - Custom Triton kernels
   - cuQuantum integration

Module dependencies:

.. code-block:: text

   adaptive_mps
   ├── mps_pytorch (base class)
   ├── linalg_robust (SVD/QR)
   ├── truncation (rank selection)
   └── diagnostics (statistics)

   vqe_qaoa
   ├── adaptive_mps (state)
   ├── mpo_ops (Hamiltonian)
   └── scipy (optimization)

   tdvp
   ├── adaptive_mps (state)
   ├── mpo_ops (Hamiltonian)
   └── linalg_robust (matrix exponential)

Test Organization
-----------------

Tests are organized by type:

.. code-block:: text

   tests/
   ├── unit/              # Unit tests for individual functions/classes
   ├── integration/       # Integration tests for complete workflows
   ├── performance/       # Performance benchmarks
   └── legacy/            # Legacy tests (quantum-inspired ML)

Run specific test suites:

.. code-block:: bash

   pytest tests/unit/ -v                    # Unit tests only
   pytest tests/integration/ -v             # Integration tests
   pytest tests/ -m "not slow"              # Skip slow tests
   pytest tests/ -k "test_mps"              # Tests matching pattern

Contributing Workflow
---------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/my-feature

2. Make changes and write tests

3. Run test suite:

   .. code-block:: bash

      pytest tests/ -v

4. Format code:

   .. code-block:: bash

      black src/ tests/ --line-length 100
      ruff check src/ tests/ --fix

5. Commit with descriptive message:

   .. code-block:: bash

      git commit -m "Add feature: brief description"

6. Push and create pull request:

   .. code-block:: bash

      git push origin feature/my-feature

Pull Request Guidelines
------------------------

- Include clear description of changes
- Reference related issues
- Add tests for new features
- Update documentation
- Ensure all tests pass
- Follow code style guidelines

For detailed guidelines, see :doc:`contributing`.

Community
---------

- GitHub Issues: https://github.com/followthesapper/ATLAS-Q/issues
- GitHub Discussions: https://github.com/followthesapper/ATLAS-Q/discussions
- PyPI: https://pypi.org/project/atlas-quantum/

License
-------

ATLAS-Q is released under the MIT License. See the LICENSE file in the repository for details.
