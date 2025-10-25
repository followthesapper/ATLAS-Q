
# Test Suite (PyTest)

Run everything:

```bash
pytest -q
```

Useful subsets:

```bash
pytest -k period
pytest -m gpu
pytest -m "not slow"
```

Notes:
- GPU tests are skipped automatically if the GPU path or CuPy isn't available.
- Property-based tests use `hypothesis` if installed; otherwise they skip.
- Tests aim to be reproducible and quick on CI.
