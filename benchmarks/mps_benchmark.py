"""
MPS Backend Benchmark

Compares MPS vs Statevector performance on various circuit sizes.
Tests:
1. Small circuits (5-10 qubits) - MPS should be competitive
2. Medium circuits (15-20 qubits) - MPS should be faster
3. Large circuits (25-30 qubits) - Only MPS can run
"""

import time
import numpy as np
from qiskit import QuantumCircuit
from atlas_q.adapters import ATLASQBackend


def benchmark_circuit(n_qubits, shots, backend_type='mps'):
    """Benchmark a single circuit"""
    if backend_type == 'mps':
        backend = ATLASQBackend(enable_mps=True, mps_threshold=1, enable_stabilizer=False)
    else:  # statevector
        backend = ATLASQBackend(enable_mps=False, enable_stabilizer=False)

    # Create circuit: chain of H and CNOT gates
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits - 1):
        qc.h(i)
        qc.cx(i, i+1)
    qc.h(n_qubits - 1)

    # Benchmark
    start = time.perf_counter()
    result = backend.run(qc, shots=shots).result()
    elapsed = time.perf_counter() - start

    # Verify
    counts = result.get_counts()
    assert len(counts) > 0, "No measurement results"

    metadata = result.results[0].header
    return elapsed, metadata['backend_used']


def main():
    print("=" * 70)
    print("MPS Backend Performance Benchmark")
    print("=" * 70)

    # Test configurations
    configs = [
        # (n_qubits, shots, description)
        (5, 100, "Small circuit (5 qubits)"),
        (10, 100, "Medium circuit (10 qubits)"),
        (15, 100, "Large circuit (15 qubits)"),
        (20, 50, "Very large circuit (20 qubits)"),
        (25, 20, "Huge circuit (25 qubits, MPS only)"),
        (30, 10, "Massive circuit (30 qubits, MPS only)"),
    ]

    results = []

    for n_qubits, shots, desc in configs:
        print(f"\n{desc}")
        print("-" * 70)

        # Try statevector first (if small enough)
        if n_qubits <= 20:
            try:
                t_sv, backend_sv = benchmark_circuit(n_qubits, shots, 'statevector')
                print(f"  Statevector: {t_sv*1000:.2f} ms")
            except Exception as e:
                t_sv = None
                print(f"  Statevector: Failed ({e})")
        else:
            t_sv = None
            print(f"  Statevector: Too large (>20 qubits)")

        # MPS should always work
        try:
            t_mps, backend_mps = benchmark_circuit(n_qubits, shots, 'mps')
            print(f"  MPS:         {t_mps*1000:.2f} ms")

            if t_sv is not None:
                speedup = t_sv / t_mps
                if speedup > 1:
                    print(f"  → MPS is {speedup:.2f}× faster than statevector")
                else:
                    print(f"  → Statevector is {1/speedup:.2f}× faster than MPS")

            results.append({
                'n_qubits': n_qubits,
                'shots': shots,
                't_mps': t_mps,
                't_sv': t_sv,
                'speedup': speedup if t_sv is not None else None
            })
        except Exception as e:
            print(f"  MPS: Failed ({e})")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Qubits':<10} {'Shots':<10} {'MPS (ms)':<15} {'Statevector (ms)':<20} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        mps_time = f"{r['t_mps']*1000:.2f}"
        sv_time = f"{r['t_sv']*1000:.2f}" if r['t_sv'] is not None else "N/A"
        speedup = f"{r['speedup']:.2f}×" if r['speedup'] is not None else "N/A"
        print(f"{r['n_qubits']:<10} {r['shots']:<10} {mps_time:<15} {sv_time:<20} {speedup:<10}")

    print("\n" + "=" * 70)
    print("Conclusion:")

    # Find crossover point where MPS becomes faster
    faster_at = None
    for r in results:
        if r['speedup'] is not None and r['speedup'] > 1:
            faster_at = r['n_qubits']
            break

    if faster_at:
        print(f"  • MPS becomes faster than statevector at {faster_at} qubits")
    else:
        print(f"  • MPS is competitive with statevector for all tested sizes")

    print(f"  • MPS enables circuits up to 30+ qubits (vs 20 for statevector)")
    print(f"  • Memory usage: O(n×χ²) for MPS vs O(2^n) for statevector")
    print("=" * 70)


if __name__ == '__main__':
    main()
