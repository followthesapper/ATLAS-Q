# Contributing

Thanks for your interest in improving the Quantum Hybrid Simulator!

## Environment
- Python 3.9+
- `pip install -r requirements-dev.txt`
- Optional: `scikit-learn`, `torch`

## Tests
- Run all tests: `pytest -q`
- Skip GPU: `pytest -q -k "not gpu"`

## Notebooks
All example notebooks are designed to run offline. Optional deps are skipped gracefully.

## Code style
- Keep functions small and documented.
- Prefer pure NumPy; gate optional deps with try/except.
