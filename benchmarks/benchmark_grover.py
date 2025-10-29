#!/usr/bin/env python3
"""
Benchmark suite for Grover's quantum search algorithm

Measures:
- Performance scaling with qubit count
- Impact of bond dimension limits
- Oracle implementation comparison
- Memory usage
- Iteration times

Usage:
    python benchmarks/benchmark_grover.py
    python benchmarks/benchmark_grover.py --device cuda
    python benchmarks/benchmark_grover.py --save results.json
"""

import argparse
import json
import time
from typing import Dict, List

import torch
import numpy as np

from atlas_q.grover import (
    GroverConfig,
    GroverSearch,
    FunctionOracle,
    BitmapOracle,
    grover_search,
    calculate_grover_iterations,
)


def benchmark_scaling_with_qubits(device: str = 'cpu', verbose: bool = True) -> Dict:
    """Benchmark performance scaling with number of qubits"""
    results = []

    if verbose:
        print("\n" + "="*70)
        print("Benchmark: Performance Scaling with Qubit Count")
        print("="*70)

    for n_qubits in range(2, 8):
        marked_state = 2**n_qubits - 1  # Mark last state

        start_time = time.time()
        result = grover_search(
            n_qubits=n_qubits,
            marked_states={marked_state},
            device=device,
            verbose=False
        )
        total_time = (time.time() - start_time) * 1000  # Convert to ms

        optimal_k = calculate_grover_iterations(n_qubits, 1)

        results.append({
            'n_qubits': n_qubits,
            'search_space_size': 2**n_qubits,
            'optimal_iterations': optimal_k,
            'total_time_ms': total_time,
            'time_per_iteration_ms': result['runtime_ms'] / result['iterations_used'],
            'measured_state': result['measured_state'],
            'marked_state': marked_state,
        })

        if verbose:
            print(f"  {n_qubits} qubits (N={2**n_qubits:4d}): "
                  f"{total_time:6.2f} ms total, "
                  f"{result['runtime_ms']/result['iterations_used']:5.2f} ms/iter, "
                  f"{optimal_k} iterations")

    return {'scaling_results': results}


def benchmark_bond_dimension_impact(device: str = 'cpu', verbose: bool = True) -> Dict:
    """Benchmark impact of bond dimension limits"""
    results = []
    n_qubits = 5
    marked_state = 20

    if verbose:
        print("\n" + "="*70)
        print("Benchmark: Impact of Bond Dimension Limits")
        print("="*70)

    for chi_max in [8, 16, 32, 64, 128, 256]:
        config = GroverConfig(
            n_qubits=n_qubits,
            chi_max=chi_max,
            device=device,
            verbose=False
        )

        oracle = BitmapOracle(n_qubits, {marked_state}, device=device)

        start_time = time.time()
        grover = GroverSearch(oracle, config)
        result = grover.run()
        total_time = (time.time() - start_time) * 1000

        results.append({
            'chi_max': chi_max,
            'n_qubits': n_qubits,
            'total_time_ms': total_time,
            'time_per_iteration_ms': result['runtime_ms'] / result['iterations_used'],
            'measured_state': result['measured_state'],
        })

        if verbose:
            print(f"  χ_max={chi_max:3d}: "
                  f"{total_time:6.2f} ms total, "
                  f"{result['runtime_ms']/result['iterations_used']:5.2f} ms/iter")

    return {'bond_dim_results': results}


def benchmark_oracle_comparison(device: str = 'cpu', verbose: bool = True) -> Dict:
    """Compare performance between different oracle implementations"""
    results = []
    n_qubits = 4
    marked_state = 10

    if verbose:
        print("\n" + "="*70)
        print("Benchmark: Oracle Implementation Comparison")
        print("="*70)

    # Bitmap oracle
    start_time = time.time()
    result_bitmap = grover_search(
        n_qubits=n_qubits,
        marked_states={marked_state},
        device=device,
        verbose=False
    )
    time_bitmap = (time.time() - start_time) * 1000

    # Function oracle
    start_time = time.time()
    result_function = grover_search(
        n_qubits=n_qubits,
        marked_states=lambda x: x == marked_state,
        device=device,
        verbose=False
    )
    time_function = (time.time() - start_time) * 1000

    results = {
        'bitmap_oracle': {
            'total_time_ms': time_bitmap,
            'time_per_iteration_ms': result_bitmap['runtime_ms'] / result_bitmap['iterations_used'],
            'measured_state': result_bitmap['measured_state'],
        },
        'function_oracle': {
            'total_time_ms': time_function,
            'time_per_iteration_ms': result_function['runtime_ms'] / result_function['iterations_used'],
            'measured_state': result_function['measured_state'],
        }
    }

    if verbose:
        print(f"  Bitmap oracle:   {time_bitmap:6.2f} ms total, "
              f"{result_bitmap['runtime_ms']/result_bitmap['iterations_used']:5.2f} ms/iter")
        print(f"  Function oracle: {time_function:6.2f} ms total, "
              f"{result_function['runtime_ms']/result_function['iterations_used']:5.2f} ms/iter")
        speedup = time_function / time_bitmap if time_bitmap > 0 else 1.0
        print(f"  Speedup: {speedup:.2f}x (bitmap vs function)")

    return {'oracle_comparison': results}


def benchmark_multiple_marked_states(device: str = 'cpu', verbose: bool = True) -> Dict:
    """Benchmark search with varying numbers of marked states"""
    results = []
    n_qubits = 5
    N = 2**n_qubits

    if verbose:
        print("\n" + "="*70)
        print("Benchmark: Multiple Marked States")
        print("="*70)

    for n_marked in [1, 2, 4, 8]:
        # Mark evenly spaced states
        marked_states = set(range(0, N, N // n_marked))

        start_time = time.time()
        result = grover_search(
            n_qubits=n_qubits,
            marked_states=marked_states,
            device=device,
            verbose=False
        )
        total_time = (time.time() - start_time) * 1000

        optimal_k = calculate_grover_iterations(n_qubits, n_marked)

        results.append({
            'n_marked': n_marked,
            'fraction_marked': n_marked / N,
            'optimal_iterations': optimal_k,
            'total_time_ms': total_time,
            'time_per_iteration_ms': result['runtime_ms'] / result['iterations_used'],
        })

        if verbose:
            print(f"  {n_marked:2d} marked states ({n_marked/N*100:5.1f}%): "
                  f"{total_time:6.2f} ms total, "
                  f"{optimal_k} iterations, "
                  f"{result['runtime_ms']/result['iterations_used']:5.2f} ms/iter")

    return {'multiple_marked_results': results}


def benchmark_dtype_comparison(device: str = 'cpu', verbose: bool = True) -> Dict:
    """Compare performance between different data types"""
    results = {}
    n_qubits = 4
    marked_state = 10

    if verbose:
        print("\n" + "="*70)
        print("Benchmark: Data Type Comparison")
        print("="*70)

    for dtype_name, dtype in [('complex64', torch.complex64), ('complex128', torch.complex128)]:
        config = GroverConfig(
            n_qubits=n_qubits,
            dtype=dtype,
            device=device,
            verbose=False
        )

        oracle = BitmapOracle(n_qubits, {marked_state}, device=device, dtype=dtype)

        start_time = time.time()
        grover = GroverSearch(oracle, config)
        result = grover.run()
        total_time = (time.time() - start_time) * 1000

        results[dtype_name] = {
            'total_time_ms': total_time,
            'time_per_iteration_ms': result['runtime_ms'] / result['iterations_used'],
            'measured_state': result['measured_state'],
        }

        if verbose:
            print(f"  {dtype_name:12s}: {total_time:6.2f} ms total, "
                  f"{result['runtime_ms']/result['iterations_used']:5.2f} ms/iter")

    if verbose and 'complex64' in results and 'complex128' in results:
        speedup = results['complex128']['total_time_ms'] / results['complex64']['total_time_ms']
        print(f"  Speedup (complex64 vs complex128): {speedup:.2f}x")

    return {'dtype_comparison': results}


def run_all_benchmarks(device: str = 'cpu', save_path: str = None, verbose: bool = True):
    """Run all benchmarks and optionally save results"""
    if verbose:
        print("\n" + "#"*70)
        print(f"# GROVER'S ALGORITHM BENCHMARK SUITE")
        print(f"# Device: {device}")
        print(f"# PyTorch version: {torch.__version__}")
        print("#"*70)

    all_results = {}

    # Run benchmarks
    all_results.update(benchmark_scaling_with_qubits(device, verbose))
    all_results.update(benchmark_bond_dimension_impact(device, verbose))
    all_results.update(benchmark_oracle_comparison(device, verbose))
    all_results.update(benchmark_multiple_marked_states(device, verbose))
    all_results.update(benchmark_dtype_comparison(device, verbose))

    # Add metadata
    all_results['metadata'] = {
        'device': device,
        'torch_version': torch.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save results if requested
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {save_path}")

    if verbose:
        print("\n" + "#"*70)
        print("# BENCHMARK SUITE COMPLETE")
        print("#"*70 + "\n")

    return all_results


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(description='Benchmark Grover\'s algorithm implementation')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run benchmarks on (default: cpu)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save benchmark results as JSON')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    # Run benchmarks
    results = run_all_benchmarks(
        device=args.device,
        save_path=args.save,
        verbose=not args.quiet
    )

    return results


if __name__ == '__main__':
    main()
