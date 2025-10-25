# GitHub Workflows

This directory can contain GitHub Actions workflows for CI/CD.

## Suggested Workflows

### test.yml - Continuous Integration
Run tests on every push/PR:
- pytest (skip GPU tests in CI)
- Code style checks (black, ruff)
- Type checking (mypy)

### benchmark.yml - Performance Testing
Run AQED benchmarks on schedule (weekly):
- Test AQED vs baseline at L=2048, 4096, 8192
- Report throughput and loss metrics
- Compare to previous runs

### docs.yml - Documentation
Build and deploy documentation:
- Sphinx/MkDocs build
- Deploy to GitHub Pages
- API reference generation

## Example test.yml

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -e .[dev]
      - run: pytest -m "not gpu"
```
