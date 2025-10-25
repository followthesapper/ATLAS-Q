# Test Suite

Comprehensive test suite for the Quantum Hybrid Simulator.

## Running Tests

```bash
# All tests
pytest

# Skip GPU tests (if no GPU available)
pytest -m "not gpu"

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=quantum_hybrid_system --cov-report=html

# Specific test file
pytest tests/test_quantum_system.py

# Specific test function
pytest tests/test_quantum_system.py::test_period_finding
```

## Test Organization

### Quantum Simulator Tests
- `test_quantum_system.py` - Core quantum simulation
- `test_period_finding.py` - Period-finding algorithms
- `test_qft_sampling.py` - QFT and sampling
- `test_state_memory_and_sampling.py` - State representations
- `test_mps_*.py` - Matrix Product State tests
- `test_circuit_*.py` - Quantum circuit tests

### AQED Tests
- `test_hybrid_aqed.py` - Hybrid AQED transformer
- `test_phase3_mps.py` - MPS-based attention
- `test_triton_integration.py` - Triton kernel integration

### ML Features Tests
- `test_qih_pat_smoke.py` - QIH pattern recognition
- `test_learned_period_head.py` - Neural period detection
- `test_chirp_features.py` - Chirp feature extraction
- `test_periodic_mixture_features.py` - Multi-period features
- `test_tn_layers_smoke.py` - Tensor network layers

### Infrastructure Tests
- `test_cpu_gpu_consistency.py` - CPU/GPU consistency
- `test_gpu_paths.py` - GPU acceleration paths
- `test_regression_api_surface.py` - API regression tests

## Test Coverage

Run with coverage to see which code is tested:

```bash
pytest --cov=quantum_hybrid_system --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Configuration

- `conftest.py` - Pytest configuration and fixtures
- `pytest.ini` - Pytest settings

## Markers

```bash
# Mark slow tests
pytest -m "slow"

# Mark GPU tests
pytest -m "gpu"

# Run only fast tests
pytest -m "not slow and not gpu"
```

## Adding New Tests

1. Create test file: `test_<feature>.py`
2. Import fixtures from conftest if needed
3. Use descriptive test names: `test_<specific_behavior>`
4. Mark GPU tests: `@pytest.mark.gpu`
5. Mark slow tests: `@pytest.mark.slow`

Example:
```python
import pytest

def test_my_feature():
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...

@pytest.mark.gpu
def test_my_gpu_feature():
    import cupy as cp
    ...
```
