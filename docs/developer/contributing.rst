Contributing Guide
==================

Contributions to ATLAS-Q are welcome. This guide outlines the contribution process and standards.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a virtual environment
4. Install development dependencies

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/ATLAS-Q.git
   cd ATLAS-Q
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   pip install -e .[dev,gpu]

Types of Contributions
----------------------

Bug Reports
^^^^^^^^^^^

Submit bug reports via GitHub Issues. Include:

- ATLAS-Q version (``atlas_q.__version__``)
- Python version
- PyTorch version
- GPU model and CUDA version (if applicable)
- Minimal code to reproduce the bug
- Expected vs actual behavior
- Error messages and stack traces

Feature Requests
^^^^^^^^^^^^^^^^

Submit feature requests via GitHub Issues or Discussions. Include:

- Clear description of proposed feature
- Use case and motivation
- Example API or usage pattern
- Potential implementation approach (optional)

Documentation
^^^^^^^^^^^^^

Documentation improvements are highly valued:

- Fix typos and grammatical errors
- Clarify existing documentation
- Add examples
- Improve docstrings
- Create tutorials

Code Contributions
^^^^^^^^^^^^^^^^^^

Code contributions should:

- Address an existing issue or provide clear motivation
- Include tests
- Follow code style guidelines
- Update documentation
- Pass all CI checks

Development Workflow
--------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make changes and write tests

3. Run test suite:

   .. code-block:: bash

      pytest tests/ -v

4. Format code:

   .. code-block:: bash

      black src/ tests/ --line-length 100
      ruff check src/ tests/ --fix

5. Build documentation:

   .. code-block:: bash

      cd docs
      make html

6. Commit changes with descriptive message

7. Push to your fork and create pull request

Code Standards
--------------

Style Guide
^^^^^^^^^^^

ATLAS-Q follows:

- PEP 8 style guide
- Black formatting (line length 100)
- Ruff linting rules
- Type hints for function signatures
- Numpydoc-style docstrings

Format all code before committing:

.. code-block:: bash

   black src/ tests/ --line-length 100
   ruff check src/ tests/

Pre-commit hooks (recommended):

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Type Hints
^^^^^^^^^^

Use type hints for all public APIs:

.. code-block:: python

   from typing import Optional, List, Tuple
   import torch

   def example_function(
       param1: int,
       param2: torch.Tensor,
       option: Optional[str] = None
   ) -> Tuple[torch.Tensor, float]:
       ...

Docstrings
^^^^^^^^^^

All public functions and classes must have numpydoc-style docstrings:

.. code-block:: python

   def compute_energy(mps, hamiltonian, use_gpu=True):
       """
       Compute expectation value of Hamiltonian.

       Parameters
       ----------
       mps : AdaptiveMPS
           Matrix Product State.
       hamiltonian : MPO
           Hamiltonian operator.
       use_gpu : bool, optional
           Use GPU-optimized contractions. Default is True.

       Returns
       -------
       energy : complex
           Expectation value ⟨ψ|H|ψ⟩.

       Examples
       --------
       >>> mps = AdaptiveMPS(10, bond_dim=8)
       >>> H = MPOBuilder.ising_hamiltonian(10)
       >>> energy = compute_energy(mps, H)
       """
       ...

Testing Guidelines
------------------

All code contributions must include tests.

Test Organization
^^^^^^^^^^^^^^^^^

Tests are organized by type:

- ``tests/unit/`` - Unit tests for individual functions
- ``tests/integration/`` - Integration tests for workflows
- ``tests/performance/`` - Performance benchmarks

Writing Tests
^^^^^^^^^^^^^

Use pytest:

.. code-block:: python

   import pytest
   import torch
   from atlas_q.adaptive_mps import AdaptiveMPS

   def test_mps_creation():
       """Test MPS initialization."""
       mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cpu')
       assert mps.num_qubits == 10
       assert len(mps.tensors) == 10

   def test_gate_application():
       """Test single-qubit gate application."""
       mps = AdaptiveMPS(num_qubits=5, bond_dim=4, device='cpu')
       H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
       mps.apply_single_qubit_gate(0, H)
       # Verify gate was applied correctly
       ...

   @pytest.mark.gpu
   def test_gpu_acceleration():
       """Test GPU-specific functionality."""
       if not torch.cuda.is_available():
           pytest.skip("CUDA not available")
       mps = AdaptiveMPS(num_qubits=10, bond_dim=8, device='cuda')
       ...

Running Tests
^^^^^^^^^^^^^

Run all tests:

.. code-block:: bash

   pytest tests/ -v

Run specific test suites:

.. code-block:: bash

   pytest tests/unit/ -v                    # Unit tests
   pytest tests/integration/ -v             # Integration tests
   pytest tests/ -m "not slow"              # Skip slow tests
   pytest tests/ -k "test_mps"              # Pattern matching

Test Coverage
^^^^^^^^^^^^^

Check test coverage:

.. code-block:: bash

   pytest tests/ --cov=atlas_q --cov-report=html
   open htmlcov/index.html

Aim for >80% coverage for new code.

Pull Request Process
--------------------

1. **Before submitting:**

   - Run test suite: ``pytest tests/ -v``
   - Format code: ``black src/ tests/ --line-length 100``
   - Lint code: ``ruff check src/ tests/``
   - Build docs: ``cd docs && make html``
   - Verify changes address issue or provide clear value

2. **Pull request description should include:**

   - Clear title summarizing changes
   - Reference to related issues (e.g., "Fixes #123")
   - Description of changes and motivation
   - Testing performed
   - Any breaking changes or migration notes

3. **Review process:**

   - Maintainers will review pull request
   - Address review comments
   - CI must pass (tests, linting, documentation build)
   - Once approved, maintainer will merge

4. **After merging:**

   - Your contribution will be included in next release
   - You will be added to contributors list

Code Review Criteria
--------------------

Reviewers will check:

- **Correctness**: Does the code work as intended?
- **Tests**: Are there sufficient tests? Do they pass?
- **Documentation**: Are docstrings complete? Is user-facing documentation updated?
- **Style**: Does code follow style guidelines?
- **Performance**: Are there obvious performance issues?
- **API design**: Is the API intuitive and consistent?
- **Backwards compatibility**: Does change break existing code?

Contribution Areas
------------------

Areas where contributions are especially welcome:

- **Examples and tutorials**: More example code and tutorials
- **Documentation**: Clarifications, corrections, new sections
- **Integration**: Qiskit/Cirq converters, export formats
- **Performance**: Optimization, profiling, benchmarking
- **Testing**: Increase test coverage, add edge cases
- **Features**: New algorithms, backends, utilities

See GitHub Issues for specific opportunities labeled "good first issue" or "help wanted".

Communication
-------------

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions

Be respectful and constructive in all interactions.

License
-------

By contributing to ATLAS-Q, you agree that your contributions will be licensed under the MIT License.

Attribution
-----------

Contributors are acknowledged in:

- CHANGELOG.md
- GitHub contributors page
- Documentation (for significant contributions)

For questions about contributing, open a GitHub Discussion or Issue.
