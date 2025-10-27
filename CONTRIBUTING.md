# Contributing to ATLAS-Q

Thank you for your interest in contributing to ATLAS-Q! We welcome contributions from the community.

---

## Areas of Contribution

### Core Features

**Tensor Networks & MPS:**
- Improved truncation strategies
- Additional tensor network structures (PEPS, MERA)
- Optimized contraction algorithms
- Mixed precision strategies

**Quantum Algorithms:**
- New variational algorithms beyond VQE/QAOA
- Enhanced time evolution methods
- Additional Hamiltonian builders
- Circuit optimization techniques

**GPU Acceleration:**
- Custom CUDA/Triton kernels
- cuQuantum integration improvements
- Multi-GPU support enhancements
- Performance optimizations

**Noise Models:**
- New noise channels
- Hardware-specific noise models
- Error mitigation techniques
- Realistic NISQ device simulation

### Integration & Ecosystem

- Qiskit/Cirq backend adapters
- Circuit conversion tools
- Export/import for other simulators
- Integration with quantum chemistry packages (PySCF, OpenFermion)

### Documentation & Examples

- Tutorial notebooks
- Use case demonstrations
- Performance benchmarks
- Blog posts and guides

### Testing & Quality

- Additional unit tests
- Integration tests
- Performance benchmarks
- Bug fixes

---

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional but recommended)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/followthsapper/ATLAS-Q.git
cd ATLAS-Q

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .[dev,gpu]

# Verify installation
pytest tests/unit/ -v
```

### GPU Setup (Optional)

```bash
# Install Triton
pip install triton>=2.0.0

# Set environment variables
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"

# Or use automated setup
./setup_triton.sh
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Follow the code style guidelines below. Make atomic commits with clear messages:

```bash
git commit -m "Add support for custom noise channels

- Implement NoiseChannel class
- Add tests for Kraus completeness
- Update documentation"
```

### 3. Add Tests

All new features must include tests:

```bash
# Unit tests
tests/unit/test_your_feature.py

# Integration tests (if applicable)
tests/integration/test_feature_integration.py
```

### 4. Run Tests Locally

```bash
# All tests
make test

# Unit tests only
make test-unit

# Skip GPU tests
pytest -m "not gpu"

# With coverage
pytest --cov=atlas_q --cov-report=html
```

### 5. Update Documentation

- Update `docs/COMPLETE_GUIDE.md` if adding user-facing features
- Update `docs/FEATURE_STATUS.md` to mark feature as implemented
- Add docstrings to all public APIs
- Update `docs/CHANGELOG.md` with your changes

### 6. Run Benchmarks (if applicable)

If your changes affect performance:

```bash
# Run relevant benchmarks
python scripts/benchmarks/validate_all_features.py

# Document results in PR
```

### 7. Submit Pull Request

- Push your branch: `git push origin feature/your-feature-name`
- Open PR on GitHub
- Fill out PR template
- Link related issues

---

## Code Style

### Python Style

**Formatting:**
- Use Black with line length 100
- Run: `black src/ tests/`

**Linting:**
- Use Ruff
- Run: `ruff check src/`

**Type Hints:**
- Required for public APIs
- Preferred for internal functions
- Run: `mypy src/atlas_q/`

### Docstrings

Use Google style docstrings for all public APIs:

```python
def my_function(param1: int, param2: str = 'default') -> bool:
    """
    Brief description of function.

    Longer description if needed. Explain what the function does,
    any important details about the algorithm, etc.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 'default')

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        RuntimeError: When computation fails

    Example:
        >>> result = my_function(42, 'test')
        >>> print(result)
        True
    """
    # Implementation...
```

### Commit Messages

Follow conventional commits format:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvements
- `test`: Adding tests
- `refactor`: Code restructuring
- `chore`: Maintenance tasks

**Example:**
```
feat: Add custom noise channel support

- Implement NoiseChannel class with Kraus operators
- Add validation for Kraus completeness
- Include tests with 100% coverage
- Update COMPLETE_GUIDE.md with examples

Closes #123
```

---

## Pull Request Guidelines

### PR Title

Use conventional commit format:
- `feat: Add MERA tensor network support`
- `fix: Correct SVD convergence in adaptive_mps`
- `docs: Update VQE examples in COMPLETE_GUIDE`

### PR Description

Include:

1. **What** - What does this PR do?
2. **Why** - Why is this change needed?
3. **How** - How does it work?
4. **Testing** - What tests were added?
5. **Performance** - Any performance impact? (include benchmarks)
6. **Breaking Changes** - Any API changes?

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (Black, Ruff)
- [ ] All tests pass (`make test`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] `FEATURE_STATUS.md` updated if adding features
- [ ] Benchmarks run if performance-related
- [ ] No merge conflicts
- [ ] Commit messages are clear

### Review Process

1. Automated checks run (CI)
2. Maintainer review
3. Address feedback
4. Approval and merge

---

## Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Isolated component tests
├── integration/    # Multi-component tests
├── performance/    # GPU/performance benchmarks
└── legacy/         # Old tests (being phased out)
```

### Writing Tests

```python
import pytest
import torch
from atlas_q import get_adaptive_mps

def test_mps_gate_application():
    """Test applying gates to MPS."""
    # Setup
    mps_modules = get_adaptive_mps()
    AdaptiveMPS = mps_modules['AdaptiveMPS']
    mps = AdaptiveMPS(5, bond_dim=4, device='cpu')

    # Apply gate
    H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
    mps.apply_single_qubit_gate(0, H)

    # Assert
    stats = mps.stats_summary()
    assert stats['total_operations'] == 1

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test GPU-specific features."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # GPU test...
```

### Running Tests

```bash
# All tests
pytest

# Unit tests
pytest tests/unit/

# Skip GPU tests
pytest -m "not gpu"

# Specific test
pytest tests/unit/test_adaptive_mps.py::test_mps_gate_application

# With coverage
pytest --cov=atlas_q --cov-report=term-missing
```

---

## Reporting Issues

### Bug Reports

Use [GitHub Issues](https://github.com/followthsapper/ATLAS-Q/issues/new) with:

**Title:** Clear, concise description (e.g., "SVD fails for bond dimension > 128")

**Description:**
```markdown
**Bug Description**
Clear description of what's wrong.

**To Reproduce**
```python
# Minimal code to reproduce
from atlas_q import get_adaptive_mps
mps_modules = get_adaptive_mps()
AdaptiveMPS = mps_modules['AdaptiveMPS']
mps = AdaptiveMPS(10, bond_dim=256, device='cuda')
# Error occurs here...
```

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens (include full error message).

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10.12
- ATLAS-Q version: 0.5.0
- PyTorch version: 2.1.0
- CUDA version: 12.1
- GPU: NVIDIA H100

**Additional Context**
Any other relevant information.
```

### Feature Requests

**Title:** Clear feature request (e.g., "Add support for MERA tensor networks")

**Description:**
```markdown
**Problem/Use Case**
Describe what problem this solves or what use case it enables.

**Proposed Solution**
How would this work? Include:
- API design
- Example usage
- Performance expectations

**Alternatives Considered**
Other approaches you've thought about.

**Willingness to Contribute**
Are you willing to implement this? (optional but appreciated)
```

---

## Performance Contributions

If contributing performance improvements:

### Before Submitting

1. **Run benchmarks** before and after changes
2. **Document results** in PR
3. **Explain approach** - what optimization and why
4. **Consider tradeoffs** - memory vs speed, accuracy vs performance

### Benchmark Format

```markdown
## Performance Results

**Test:** MPS gate application (1000 iterations, 20 qubits, χ=32)

**Before:**
- CPU: 2.34s (427 ops/sec)
- GPU: 0.45s (2222 ops/sec)
- Memory: 45 MB

**After:**
- CPU: 2.31s (433 ops/sec) [+1.4%]
- GPU: 0.31s (3226 ops/sec) [+45%]
- Memory: 45 MB [no change]

**Optimization:** Fused Triton kernel for two-site gate application
```

---

## Documentation Contributions

### Documentation Standards

- **Accuracy:** Every code example must be tested
- **Completeness:** Document all parameters, return values, exceptions
- **Clarity:** Write for users at different skill levels
- **Examples:** Include working examples for all features

### Before Submitting Docs

1. **Test all code examples**
2. **Check links** (no broken links)
3. **Verify accuracy** against actual code
4. **Run spell check**
5. **Preview rendering** (Markdown preview)

---

## License

By contributing to ATLAS-Q, you agree that your contributions will be licensed under the MIT License.

---

## Code of Conduct

### Our Standards

- **Be respectful** - Treat everyone with respect
- **Be constructive** - Focus on constructive feedback
- **Be inclusive** - Welcome newcomers
- **Be honest** - No false claims or misleading documentation
- **Give credit** - Acknowledge others' work

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Publishing false information
- Personal attacks

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to project maintainers.

---

## Questions?

- **Getting Started:** Try `ATLAS_Q_Demo.ipynb` for interactive examples
- **General questions:** Open a GitHub Issue
- **Discussion:** Use GitHub Discussions
- **Documentation:** Check `docs/COMPLETE_GUIDE.md` first
- **Features:** See `docs/FEATURE_STATUS.md` for what's implemented
- **Online Docs:** Browse https://followthsapper.github.io/ATLAS-Q/

---

## Recognition

Contributors will be recognized in:
- `CHANGELOG.md` for each release
- GitHub contributors page
- Special mentions for significant contributions

---

**Thank you for contributing to ATLAS-Q!**

Making quantum simulation accessible through honest, working code.
