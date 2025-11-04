# ATLAS-Q Test Suite

Comprehensive test suite for ATLAS-Q quantum tensor network simulator.

## Running Tests

```bash
# All tests
pytest

# Specific test directory
pytest tests/unit/ # Unit tests only
pytest tests/integration/ # Integration tests only
pytest tests/performance/ # Performance tests only

# Skip GPU tests (if no GPU available)
pytest -m "not gpu"

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=atlas_q --cov-report=html

# Specific test file
pytest tests/unit/test_quantum_system.py

# Specific test function
pytest tests/unit/test_quantum_system.py::test_period_finding
```

## Test Organization

### Unit Tests (`unit/`)
Core component tests for individual modules:
- `test_adaptive_mps.py` - Adaptive MPS with GPU support
- `test_mpo_ops.py` - MPO operations (Hamiltonians)
- `test_tdvp.py` - TDVP time evolution
- `test_vqe_qaoa.py` - VQE and QAOA algorithms
- `test_noise_models.py` - NISQ noise models
- `test_stabilizer_backend.py` - Clifford circuit simulation
- `test_mps_pytorch.py` - PyTorch MPS backend
- `test_period_finding.py` - Period-finding algorithms
- `test_quantum_system.py` - Core quantum simulation

### Integration Tests (`integration/`)
Multi-component tests and API verification:
- `test_circuit_api.py` - Circuit builder API
- `test_cpu_gpu_consistency.py` - CPU/GPU consistency
- `test_qft_sampling.py` - QFT and sampling
- `test_state_memory_and_sampling.py` - State representations
- `test_mps_gates_batched.py` - Batched gate operations
- `test_regression_api_surface.py` - API regression tests

### Performance Tests (`performance/`)
GPU acceleration and performance validation:
- `test_gpu_paths.py` - GPU acceleration paths
- `test_triton_integration.py` - Custom Triton kernels

### Legacy Tests (`legacy/`)
Older tests that may need updating or archiving (15 tests)

## Test Coverage

Run with coverage to see which code is tested:

```bash
pytest --cov=atlas_q --cov-report=html
open htmlcov/index.html # View coverage report
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
