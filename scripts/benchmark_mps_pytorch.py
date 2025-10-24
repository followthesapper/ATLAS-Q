"""
Benchmark PyTorch MPS vs NumPy MPS

Compares performance of PyTorch-based MPS implementation against NumPy version.
Measures speedup across different operations and configurations.

Target speedup: 1.5-2× for typical operations

Author: Claude Code
Date: October 2025
"""

import sys
import os
import time
import numpy as np
import torch
import csv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum_hybrid_system.quantum_hybrid_system import MatrixProductState
from src.quantum_hybrid_system.mps_pytorch import MatrixProductStatePyTorch, create_compiled_mps


def benchmark_initialization(num_qubits, bond_dim, num_trials=100):
    """Benchmark MPS initialization time"""
    print(f"\n[Initialization: {num_qubits} qubits, χ={bond_dim}]")

    # NumPy version
    start = time.perf_counter()
    for _ in range(num_trials):
        mps = MatrixProductState(num_qubits, bond_dim)
    numpy_time = time.perf_counter() - start

    # PyTorch version
    start = time.perf_counter()
    for _ in range(num_trials):
        mps = MatrixProductStatePyTorch(num_qubits, bond_dim, device='cuda')
        torch.cuda.synchronize()  # Wait for GPU
    torch_time = time.perf_counter() - start

    speedup = numpy_time / torch_time

    print(f"  NumPy:   {numpy_time*1000/num_trials:.2f} ms/init")
    print(f"  PyTorch: {torch_time*1000/num_trials:.2f} ms/init")
    print(f"  Speedup: {speedup:.2f}×")

    return {
        'operation': 'initialization',
        'num_qubits': num_qubits,
        'bond_dim': bond_dim,
        'numpy_time_ms': numpy_time * 1000 / num_trials,
        'torch_time_ms': torch_time * 1000 / num_trials,
        'speedup': speedup
    }


def benchmark_canonicalization(num_qubits, bond_dim, num_trials=50):
    """Benchmark MPS canonicalization (QR decomposition)"""
    print(f"\n[Canonicalization: {num_qubits} qubits, χ={bond_dim}]")

    # NumPy version
    mps_numpy = MatrixProductState(num_qubits, bond_dim)
    start = time.perf_counter()
    for _ in range(num_trials):
        mps_numpy.canonicalize_left_to_right()
    numpy_time = time.perf_counter() - start

    # PyTorch version
    mps_torch = MatrixProductStatePyTorch(num_qubits, bond_dim, device='cuda')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        mps_torch.canonicalize_left_to_right()
        torch.cuda.synchronize()
    torch_time = time.perf_counter() - start

    speedup = numpy_time / torch_time

    print(f"  NumPy:   {numpy_time*1000/num_trials:.2f} ms/canonicalization")
    print(f"  PyTorch: {torch_time*1000/num_trials:.2f} ms/canonicalization")
    print(f"  Speedup: {speedup:.2f}×")

    return {
        'operation': 'canonicalization',
        'num_qubits': num_qubits,
        'bond_dim': bond_dim,
        'numpy_time_ms': numpy_time * 1000 / num_trials,
        'torch_time_ms': torch_time * 1000 / num_trials,
        'speedup': speedup
    }


def benchmark_amplitude_calculation(num_qubits, bond_dim, num_samples=100):
    """Benchmark amplitude calculation for random basis states"""
    print(f"\n[Amplitude Calculation: {num_qubits} qubits, χ={bond_dim}]")

    # Create both MPS
    mps_numpy = MatrixProductState(num_qubits, bond_dim)
    mps_torch = MatrixProductStatePyTorch(num_qubits, bond_dim, device='cuda')

    # Random basis states to query
    max_state = 2 ** num_qubits - 1
    basis_states = [np.random.randint(0, max_state + 1) for _ in range(num_samples)]

    # NumPy version
    start = time.perf_counter()
    for state in basis_states:
        amp = mps_numpy.get_amplitude(state)
    numpy_time = time.perf_counter() - start

    # PyTorch version (includes GPU transfer time)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for state in basis_states:
        amp = mps_torch.get_amplitude(state)
    torch.cuda.synchronize()
    torch_time = time.perf_counter() - start

    speedup = numpy_time / torch_time

    print(f"  NumPy:   {numpy_time*1000/num_samples:.3f} ms/amplitude")
    print(f"  PyTorch: {torch_time*1000/num_samples:.3f} ms/amplitude")
    print(f"  Speedup: {speedup:.2f}×")

    return {
        'operation': 'amplitude_calc',
        'num_qubits': num_qubits,
        'bond_dim': bond_dim,
        'numpy_time_ms': numpy_time * 1000 / num_samples,
        'torch_time_ms': torch_time * 1000 / num_samples,
        'speedup': speedup
    }


def benchmark_sampling(num_qubits, bond_dim, num_shots=100):
    """Benchmark MPS sampling (sweep sampling)"""
    print(f"\n[Sampling: {num_qubits} qubits, χ={bond_dim}, {num_shots} shots]")

    # Create both MPS
    mps_numpy = MatrixProductState(num_qubits, bond_dim)
    mps_torch = MatrixProductStatePyTorch(num_qubits, bond_dim, device='cuda')

    # NumPy version
    start = time.perf_counter()
    samples_numpy = mps_numpy.sweep_sample(num_shots)
    numpy_time = time.perf_counter() - start

    # PyTorch version
    torch.cuda.synchronize()
    start = time.perf_counter()
    samples_torch = mps_torch.sweep_sample(num_shots)
    torch.cuda.synchronize()
    torch_time = time.perf_counter() - start

    speedup = numpy_time / torch_time

    print(f"  NumPy:   {numpy_time*1000:.2f} ms ({num_shots} shots)")
    print(f"  PyTorch: {torch_time*1000:.2f} ms ({num_shots} shots)")
    print(f"  Speedup: {speedup:.2f}×")

    return {
        'operation': 'sampling',
        'num_qubits': num_qubits,
        'bond_dim': bond_dim,
        'num_shots': num_shots,
        'numpy_time_ms': numpy_time * 1000,
        'torch_time_ms': torch_time * 1000,
        'speedup': speedup
    }


def benchmark_compiled_version(num_qubits, bond_dim):
    """Benchmark torch.compile speedup"""
    print(f"\n[torch.compile: {num_qubits} qubits, χ={bond_dim}]")

    if not hasattr(torch, 'compile'):
        print("  ⚠ torch.compile not available (requires PyTorch 2.0+)")
        return None

    # Regular PyTorch version
    mps_regular = MatrixProductStatePyTorch(num_qubits, bond_dim, device='cuda')

    # Compiled version
    mps_compiled = create_compiled_mps(num_qubits, bond_dim, device='cuda', compile=True)

    # Warmup
    for _ in range(5):
        mps_compiled.canonicalize_left_to_right()
    torch.cuda.synchronize()

    # Benchmark regular
    num_trials = 20
    start = time.perf_counter()
    for _ in range(num_trials):
        mps_regular.canonicalize_left_to_right()
        torch.cuda.synchronize()
    regular_time = time.perf_counter() - start

    # Benchmark compiled
    start = time.perf_counter()
    for _ in range(num_trials):
        mps_compiled.canonicalize_left_to_right()
        torch.cuda.synchronize()
    compiled_time = time.perf_counter() - start

    speedup = regular_time / compiled_time

    print(f"  Regular:  {regular_time*1000/num_trials:.2f} ms/canonicalization")
    print(f"  Compiled: {compiled_time*1000/num_trials:.2f} ms/canonicalization")
    print(f"  Speedup:  {speedup:.2f}×")

    return {
        'operation': 'torch_compile',
        'num_qubits': num_qubits,
        'bond_dim': bond_dim,
        'regular_time_ms': regular_time * 1000 / num_trials,
        'compiled_time_ms': compiled_time * 1000 / num_trials,
        'speedup': speedup
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across multiple configurations"""
    print("=" * 70)
    print("PyTorch MPS Performance Benchmark")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    results = []

    # Configuration sets
    configs = [
        # (num_qubits, bond_dim, description)
        (5, 8, "Small system"),
        (10, 16, "Medium system"),
        (15, 16, "Large system (more qubits)"),
        (10, 32, "Large system (larger bond dim)"),
    ]

    for num_qubits, bond_dim, description in configs:
        print(f"\n{'='*70}")
        print(f"{description}: {num_qubits} qubits, χ={bond_dim}")
        print(f"{'='*70}")

        # Run benchmarks
        results.append(benchmark_initialization(num_qubits, bond_dim))
        results.append(benchmark_canonicalization(num_qubits, bond_dim))
        results.append(benchmark_amplitude_calculation(num_qubits, bond_dim))
        results.append(benchmark_sampling(num_qubits, bond_dim, num_shots=100))

    # torch.compile benchmark (separate)
    print(f"\n{'='*70}")
    print("torch.compile Speedup Test")
    print(f"{'='*70}")
    compile_result = benchmark_compiled_version(10, 16)
    if compile_result:
        results.append(compile_result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Calculate average speedups by operation
    operations = {}
    for result in results:
        op = result['operation']
        if op not in operations:
            operations[op] = []
        if 'speedup' in result:
            operations[op].append(result['speedup'])

    print("\nAverage Speedups by Operation:")
    for op, speedups in operations.items():
        avg_speedup = np.mean(speedups)
        min_speedup = np.min(speedups)
        max_speedup = np.max(speedups)
        print(f"  {op:20s}: {avg_speedup:.2f}× (range: {min_speedup:.2f}× - {max_speedup:.2f}×)")

    # Overall speedup
    all_speedups = [s for speedup_list in operations.values() for s in speedup_list]
    overall_speedup = np.mean(all_speedups)
    print(f"\n  {'Overall Average':20s}: {overall_speedup:.2f}×")

    # Check if we met target
    target_speedup = 1.5
    if overall_speedup >= target_speedup:
        print(f"\n✓ Target speedup of {target_speedup}× ACHIEVED!")
    else:
        print(f"\n⚠ Target speedup of {target_speedup}× not reached (got {overall_speedup:.2f}×)")

    # Save to CSV
    csv_file = "runs/mps_pytorch_benchmark.csv"
    os.makedirs("runs", exist_ok=True)

    with open(csv_file, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {csv_file}")

    return overall_speedup >= target_speedup


if __name__ == "__main__":
    try:
        success = run_comprehensive_benchmark()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
