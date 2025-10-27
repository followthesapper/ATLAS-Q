# ATLAS-Q Makefile - Developer shortcuts
.PHONY: help test test-unit test-integration test-performance test-nogpu bench demo install dev-install clean

help:
	@echo "ATLAS-Q Development Commands:"
	@echo ""
	@echo "  make install          - Install package in editable mode"
	@echo "  make dev-install      - Install with dev dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-performance - Run performance tests only"
	@echo "  make test-nogpu       - Run tests without GPU"
	@echo "  make bench            - Run comprehensive benchmarks"
	@echo "  make demo             - Run feature demonstrations"
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
	python3 benchmarks/comprehensive_benchmark.py

demo:
	@echo "Running ATLAS-Q feature demonstrations..."
	python3 scripts/demo_adaptive_mps.py
	python3 scripts/demo_new_features.py

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
