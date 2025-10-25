# Contributing to Quantum Hybrid Simulator

Thank you for your interest in contributing! This project welcomes contributions from the community.

## Areas of Interest

### AQED (Transformer Optimization)
- Adaptive routing algorithms
- Hierarchical mixing strategies
- Custom CUDA/Triton kernels
- Benchmarking on new models/datasets
- Integration with HuggingFace ecosystem

### Quantum Simulator
- New compressed state representations
- Faster period-finding algorithms
- Improved tensor network methods
- GPU optimization
- Quantum-inspired ML features

### Documentation & Examples
- Tutorial notebooks
- Use case demonstrations
- Benchmark reports
- Blog posts

## Development Setup

```bash
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[dev]
pytest  # Run tests
```

## Code Style

- **Formatting:** Black (line length 100)
- **Linting:** Ruff
- **Type hints:** Preferred (checked with mypy)
- **Docstrings:** Required for public APIs (Google style)

## Testing

```bash
# All tests
pytest

# Skip GPU tests
pytest -m "not gpu"

# With coverage
pytest --cov=quantum_hybrid_system --cov-report=html
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/amazing-feature`
3. **Make your changes** with clear, atomic commits
4. **Add tests** for new functionality
5. **Run tests** locally: `pytest`
6. **Update documentation** if needed
7. **Submit PR** with clear description

### PR Guidelines

- Clear title and description
- Link related issues
- Include benchmarks for performance changes
- Update CHANGELOG.md (if applicable)
- Ensure CI passes

## Reporting Bugs

Use [GitHub Issues](https://github.com/your-org/quantum-hybrid-simulator/issues) with:

- **Clear title** describing the issue
- **Steps to reproduce** (minimal example)
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU if applicable)
- **Error messages** (full stack trace)

## Feature Requests

We welcome feature requests! Please:

- **Check existing issues** first
- **Describe the use case** (why is this needed?)
- **Propose solution** (if you have one)
- **Offer to implement** (optional but appreciated!)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers
- Credit others' work

## Questions?

- Open an issue for general questions
- Use Discussions for broader topics
- Check documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🎉
