# ATLAS-Q Makefile - Developer shortcuts
.PHONY: help test test-unit test-integration test-performance test-nogpu bench bench-quick demo demo-qaoa probe install dev-install clean

help:
	@echo "ATLAS-Q Development Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install package in editable mode"
	@echo "  make dev-install      - Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-performance - Run performance tests only"
	@echo "  make test-nogpu       - Run tests without GPU"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench            - Run comprehensive benchmarks"
	@echo "  make bench-quick      - Run quick benchmark"
	@echo ""
	@echo "Demonstrations:"
	@echo "  make demo             - Run feature demonstrations"
	@echo "  make demo-qaoa        - Run QAOA demo"
	@echo "  make probe            - Run capacity probe"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e .
	pip install pytest pytest-cov black ruff mypy

test:
	pytest -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v

test-nogpu:
	pytest -v -m "not gpu"

bench:
	@echo "Running comprehensive benchmarks..."
	python3 benchmarks/comprehensive_benchmark.py

bench-quick:
	@echo "Running quick benchmark..."
	python3 scripts/benchmarks/benchmark_simulator.py

demo:
	@echo "Running ATLAS-Q feature demonstrations..."
	python3 scripts/demos/demo_adaptive_mps.py
	python3 scripts/demos/demo_new_features.py

demo-qaoa:
	@echo "Running QAOA demonstration..."
	python3 scripts/demos/qaoa_maxcut_grid.py --n_qubits 8

probe:
	@echo "Running capacity probe..."
	python3 scripts/probes/qubit_capacity_probe.py --safety_frac 0.7

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
