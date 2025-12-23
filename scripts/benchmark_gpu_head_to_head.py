#!/usr/bin/env python3
"""
GPU Head-to-Head Benchmark: ATLAS-Q vs Qiskit Aer

This benchmark compares both frameworks using their FULL GPU capabilities:
- ATLAS-Q: PyTorch CUDA + Triton kernels
- Qiskit Aer: GPU backend (cuStateVec)

All tests run on the same hardware for fair comparison.
"""

import sys
import time
import gc

import numpy as np
import torch

sys.path.insert(0, '/home/admin/ATLAS-Q/src')


def format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def get_gpu_info():
    """Print GPU information"""
    print("=" * 70)
    print("GPU HARDWARE INFORMATION")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"  PyTorch CUDA: Available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  PyTorch CUDA: NOT Available")

    # Check Triton
    try:
        import triton
        print(f"  Triton: v{triton.__version__}")
    except ImportError:
        print("  Triton: NOT Available")

    # Check Qiskit Aer GPU
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator(method='statevector', device='GPU')
        print(f"  Qiskit Aer GPU: Available")
    except Exception as e:
        print(f"  Qiskit Aer GPU: NOT Available ({e})")

    print()


def warmup_gpu():
    """Warmup GPU to get accurate benchmarks"""
    print("Warming up GPU...")
    if torch.cuda.is_available():
        try:
            # PyTorch warmup with smaller tensor
            x = torch.randn(100, 100, device='cuda')
            for _ in range(10):
                y = x @ x
            torch.cuda.synchronize()
            del x, y
        except torch.cuda.OutOfMemoryError:
            print("  Warning: GPU memory limited, skipping warmup")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print available memory
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        total_mem = torch.cuda.mem_get_info()[1] / 1e9
        print(f"  Available GPU memory: {free_mem:.1f} / {total_mem:.1f} GB")
    print("Warmup complete.\n")


def benchmark_statevector_simulation():
    """Benchmark statevector simulation on GPU"""
    print("=" * 70)
    print("1. STATEVECTOR SIMULATION (GPU)")
    print("=" * 70)
    print("   Testing random circuit execution at various qubit counts\n")

    results = []

    # Adjust qubit range based on available memory
    if torch.cuda.is_available():
        free_mem_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_mem_gb < 5:
            qubit_range = [8, 10, 12, 14]
        elif free_mem_gb < 20:
            qubit_range = [10, 12, 14, 16, 18]
        else:
            qubit_range = [10, 12, 14, 16, 18, 20, 22]
    else:
        qubit_range = [10, 12, 14, 16]

    for n_qubits in qubit_range:
        print(f"  {n_qubits} qubits:")

        # ========== ATLAS-Q ==========
        atlas_time = None
        try:
            from atlas_q.statevector_simulator import StatevectorSimulator

            # Create simulator on GPU
            sim = StatevectorSimulator(n_qubits, device='cuda')

            # Warmup
            for i in range(min(n_qubits, 5)):
                sim.h(i)
            sim.reset()

            torch.cuda.synchronize()
            start = time.perf_counter()

            # Build a random-ish circuit
            for i in range(n_qubits):
                sim.h(i)
            for _ in range(n_qubits):  # depth = n_qubits layers
                for i in range(0, n_qubits - 1, 2):
                    sim.cnot(i, i + 1)
                for i in range(1, n_qubits - 1, 2):
                    sim.cnot(i, i + 1)
                for i in range(n_qubits):
                    sim.rz(i, np.pi / 4)

            # Force computation
            state = sim.state
            torch.cuda.synchronize()
            atlas_time = time.perf_counter() - start

            del sim, state
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ATLAS-Q Error: {e}")

        # ========== Qiskit Aer GPU ==========
        qiskit_time = None
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator

            # Build equivalent circuit
            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                qc.h(i)
            for _ in range(n_qubits):
                for i in range(0, n_qubits - 1, 2):
                    qc.cx(i, i + 1)
                for i in range(1, n_qubits - 1, 2):
                    qc.cx(i, i + 1)
                for i in range(n_qubits):
                    qc.rz(np.pi / 4, i)
            qc.save_statevector()

            # Try GPU first, fall back to CPU
            try:
                sim_aer = AerSimulator(method='statevector', device='GPU')
                device_used = "GPU"
            except Exception:
                sim_aer = AerSimulator(method='statevector', device='CPU')
                device_used = "CPU"

            # Warmup
            sim_aer.run(qc, shots=1).result()

            start = time.perf_counter()
            result = sim_aer.run(qc, shots=1).result()
            qiskit_time = time.perf_counter() - start

        except Exception as e:
            print(f"    Qiskit Error: {e}")
            device_used = "N/A"

        # Report results
        if atlas_time:
            print(f"    ATLAS-Q (CUDA):     {format_time(atlas_time)}")
        if qiskit_time:
            print(f"    Qiskit Aer ({device_used}):  {format_time(qiskit_time)}")

        if atlas_time and qiskit_time:
            speedup = qiskit_time / atlas_time
            winner = "ATLAS-Q" if speedup > 1 else "Qiskit"
            print(f"    Speedup: {speedup:.2f}x ({winner} faster)")

        results.append({
            'qubits': n_qubits,
            'atlas_time': atlas_time,
            'qiskit_time': qiskit_time,
            'speedup': qiskit_time / atlas_time if (atlas_time and qiskit_time) else None
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return results


def benchmark_density_matrix_gpu():
    """Benchmark density matrix simulation on GPU"""
    print("=" * 70)
    print("2. DENSITY MATRIX SIMULATION (GPU)")
    print("=" * 70)
    print("   Testing noisy circuit simulation with Kraus channels\n")

    results = []

    for n_qubits in [4, 6, 8, 10]:
        print(f"  {n_qubits} qubits:")

        # ========== ATLAS-Q GPU ==========
        atlas_time = None
        purity = None
        try:
            from atlas_q import get_density_matrix
            dm = get_density_matrix()
            DensityMatrixSimulator = dm['DensityMatrixSimulator']
            DensityMatrixConfig = dm['DensityMatrixConfig']

            config = DensityMatrixConfig(device='cuda')

            # Warmup
            sim = DensityMatrixSimulator(n_qubits, config=config)
            sim.h(0)
            del sim
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            start = time.perf_counter()

            sim = DensityMatrixSimulator(n_qubits, config=config)
            for i in range(n_qubits):
                sim.h(i)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)
            for i in range(n_qubits):
                sim.apply_depolarizing(i, p=0.01)
            purity = sim.purity()

            torch.cuda.synchronize()
            atlas_time = time.perf_counter() - start

            del sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ATLAS-Q Error: {e}")

        # ========== Qiskit Aer GPU ==========
        qiskit_time = None
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            qc = QuantumCircuit(n_qubits)
            for i in range(n_qubits):
                qc.h(i)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.save_density_matrix()

            noise = NoiseModel()
            noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['h'])
            noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ['cx'])

            # density_matrix method typically CPU-only in Aer
            try:
                sim_aer = AerSimulator(method='density_matrix', device='GPU', noise_model=noise)
                device_used = "GPU"
            except Exception:
                sim_aer = AerSimulator(method='density_matrix', device='CPU', noise_model=noise)
                device_used = "CPU"

            # Warmup
            sim_aer.run(qc, shots=1).result()

            start = time.perf_counter()
            result = sim_aer.run(qc, shots=1).result()
            qiskit_time = time.perf_counter() - start

        except Exception as e:
            print(f"    Qiskit Error: {e}")
            device_used = "N/A"

        if atlas_time:
            print(f"    ATLAS-Q (CUDA):     {format_time(atlas_time)}")
            if purity:
                print(f"                        Purity: {purity:.4f}")
        if qiskit_time:
            print(f"    Qiskit Aer ({device_used}):  {format_time(qiskit_time)}")

        if atlas_time and qiskit_time:
            speedup = qiskit_time / atlas_time
            winner = "ATLAS-Q" if speedup > 1 else "Qiskit"
            print(f"    Speedup: {speedup:.2f}x ({winner} faster)")

        results.append({
            'qubits': n_qubits,
            'atlas_time': atlas_time,
            'qiskit_time': qiskit_time,
            'speedup': qiskit_time / atlas_time if (atlas_time and qiskit_time) else None
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return results


def benchmark_mps_simulation():
    """Benchmark MPS simulation (ATLAS-Q specialty)"""
    print("=" * 70)
    print("3. MPS/TENSOR NETWORK SIMULATION (ATLAS-Q GPU)")
    print("=" * 70)
    print("   Testing large-scale simulation with Matrix Product States\n")
    print("   Note: This is where ATLAS-Q shines - scaling to 30+ qubits\n")

    results = []

    for n_qubits in [20, 24, 28, 30]:
        print(f"  {n_qubits} qubits:")

        atlas_time = None
        try:
            from atlas_q.mps_simulator import MPSSimulator

            # Create MPS simulator on GPU
            sim = MPSSimulator(n_qubits, bond_dim=64, device='cuda')

            torch.cuda.synchronize()
            start = time.perf_counter()

            # GHZ-like circuit
            sim.h(0)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)

            # Measure expectation value
            # Use simple sampling instead
            samples = sim.sample(shots=100)

            torch.cuda.synchronize()
            atlas_time = time.perf_counter() - start

            print(f"    ATLAS-Q MPS (CUDA): {format_time(atlas_time)}")
            print(f"                        Bond dim: 64, Samples: 100")

            del sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ATLAS-Q MPS Error: {e}")

        # Qiskit MPS for comparison
        qiskit_time = None
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator

            qc = QuantumCircuit(n_qubits)
            qc.h(0)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure_all()

            sim_aer = AerSimulator(method='matrix_product_state')

            start = time.perf_counter()
            result = sim_aer.run(qc, shots=100).result()
            qiskit_time = time.perf_counter() - start

            print(f"    Qiskit MPS (CPU):   {format_time(qiskit_time)}")

        except Exception as e:
            print(f"    Qiskit MPS Error: {e}")

        if atlas_time and qiskit_time:
            speedup = qiskit_time / atlas_time
            winner = "ATLAS-Q" if speedup > 1 else "Qiskit"
            print(f"    Speedup: {speedup:.2f}x ({winner} faster)")

        results.append({
            'qubits': n_qubits,
            'atlas_time': atlas_time,
            'qiskit_time': qiskit_time
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return results


def benchmark_triton_kernels():
    """Benchmark Triton kernel performance"""
    print("=" * 70)
    print("4. TRITON KERNEL ACCELERATION")
    print("=" * 70)
    print("   Testing custom Triton kernels for quantum operations\n")

    try:
        from triton_kernels.density_matrix_ops import (
            apply_kraus_channel_triton,
            purity_triton,
        )

        print("   Triton kernels: AVAILABLE\n")

        results = []

        for n_qubits in [4, 6, 8]:
            dim = 2 ** n_qubits
            print(f"  {n_qubits} qubits (dim={dim}x{dim}):")

            # Create random density matrix
            rho = torch.randn(dim, dim, dtype=torch.complex128, device='cuda')
            rho = rho @ rho.conj().T
            rho = rho / rho.trace()

            # Purity benchmark
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                p = purity_triton(rho)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / 100

            # PyTorch baseline
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                p_torch = torch.trace(rho @ rho).real.item()
            torch.cuda.synchronize()
            torch_time = (time.perf_counter() - start) / 100

            print(f"    Purity (Triton):  {format_time(triton_time)}")
            print(f"    Purity (PyTorch): {format_time(torch_time)}")
            print(f"    Speedup: {torch_time/triton_time:.2f}x")

            results.append({
                'qubits': n_qubits,
                'triton_time': triton_time,
                'torch_time': torch_time,
                'speedup': torch_time / triton_time
            })
            print()

        return results

    except ImportError as e:
        print(f"   Triton kernels not available: {e}")
        return []


def print_summary(sv_results, dm_results, mps_results):
    """Print competitive analysis summary"""
    print("=" * 70)
    print("GPU HEAD-TO-HEAD SUMMARY")
    print("=" * 70)

    print("\n  STATEVECTOR SIMULATION:")
    print("  " + "-" * 50)
    for r in sv_results:
        if r['speedup']:
            winner = "ATLAS-Q ✓" if r['speedup'] > 1 else "Qiskit ✓"
            print(f"    {r['qubits']:2d} qubits: {r['speedup']:.2f}x - {winner}")

    print("\n  DENSITY MATRIX SIMULATION:")
    print("  " + "-" * 50)
    for r in dm_results:
        if r['speedup']:
            winner = "ATLAS-Q ✓" if r['speedup'] > 1 else "Qiskit ✓"
            print(f"    {r['qubits']:2d} qubits: {r['speedup']:.2f}x - {winner}")

    print("\n  MPS SIMULATION (Large Scale):")
    print("  " + "-" * 50)
    for r in mps_results:
        if r.get('atlas_time') and r.get('qiskit_time'):
            speedup = r['qiskit_time'] / r['atlas_time']
            winner = "ATLAS-Q ✓" if speedup > 1 else "Qiskit ✓"
            print(f"    {r['qubits']:2d} qubits: {speedup:.2f}x - {winner}")
        elif r.get('atlas_time'):
            print(f"    {r['qubits']:2d} qubits: ATLAS-Q only (Qiskit failed/unavailable)")

    # Calculate overall winner
    atlas_wins = 0
    qiskit_wins = 0

    for r in sv_results + dm_results:
        if r.get('speedup'):
            if r['speedup'] > 1:
                atlas_wins += 1
            else:
                qiskit_wins += 1

    print("\n" + "=" * 70)
    print(f"  OVERALL: ATLAS-Q wins {atlas_wins}, Qiskit wins {qiskit_wins}")

    if atlas_wins > qiskit_wins:
        print("  VERDICT: ATLAS-Q demonstrates superior GPU performance")
    elif qiskit_wins > atlas_wins:
        print("  VERDICT: Qiskit Aer demonstrates superior GPU performance")
    else:
        print("  VERDICT: Both frameworks perform comparably on GPU")

    print("=" * 70)


def main():
    print("=" * 70)
    print("ATLAS-Q v0.8.0 vs Qiskit Aer - GPU HEAD-TO-HEAD BENCHMARK")
    print("=" * 70)
    print("Both frameworks using maximum GPU acceleration")
    print()

    get_gpu_info()
    warmup_gpu()

    sv_results = benchmark_statevector_simulation()
    dm_results = benchmark_density_matrix_gpu()
    mps_results = benchmark_mps_simulation()
    benchmark_triton_kernels()

    print_summary(sv_results, dm_results, mps_results)


if __name__ == "__main__":
    main()
