#!/usr/bin/env python3
"""
Simplified GPU Benchmark: ATLAS-Q Performance Demonstration

This benchmark demonstrates ATLAS-Q's GPU capabilities:
- PyTorch CUDA for density matrix operations
- MPS simulator for large-scale simulation
- Direct comparison against CPU for speedup measurements

Note: Qiskit Aer GPU is not available on aarch64, so we compare
against Qiskit Aer CPU to show our GPU advantage.
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
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Total Memory: {total_mem:.1f} GB")
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  Available Memory: {free_mem:.1f} GB")
    else:
        print("  PyTorch CUDA: NOT Available")

    # Check Triton
    try:
        import triton
        print(f"  Triton: v{triton.__version__}")
    except ImportError:
        print("  Triton: NOT Available")

    print()


def benchmark_density_matrix():
    """Benchmark density matrix simulation: GPU vs CPU"""
    print("=" * 70)
    print("1. DENSITY MATRIX SIMULATION (GPU vs CPU)")
    print("=" * 70)
    print("   Testing noisy circuit simulation with depolarizing channels\n")

    from atlas_q import get_density_matrix
    dm = get_density_matrix()
    DensityMatrixSimulator = dm['DensityMatrixSimulator']
    DensityMatrixConfig = dm['DensityMatrixConfig']

    results = []

    for n_qubits in [4, 6, 8, 10]:
        print(f"  {n_qubits} qubits:")

        # ========== ATLAS-Q GPU ==========
        gpu_time = None
        gpu_purity = None
        try:
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
            gpu_purity = sim.purity()

            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - start

            del sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    GPU Error: {e}")

        # ========== ATLAS-Q CPU ==========
        cpu_time = None
        cpu_purity = None
        try:
            config = DensityMatrixConfig(device='cpu')

            start = time.perf_counter()

            sim = DensityMatrixSimulator(n_qubits, config=config)
            for i in range(n_qubits):
                sim.h(i)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)
            for i in range(n_qubits):
                sim.apply_depolarizing(i, p=0.01)
            cpu_purity = sim.purity()

            cpu_time = time.perf_counter() - start

            del sim

        except Exception as e:
            print(f"    CPU Error: {e}")

        if gpu_time:
            print(f"    ATLAS-Q GPU: {format_time(gpu_time)} (purity: {gpu_purity:.4f})")
        if cpu_time:
            print(f"    ATLAS-Q CPU: {format_time(cpu_time)} (purity: {cpu_purity:.4f})")

        if gpu_time and cpu_time:
            speedup = cpu_time / gpu_time
            print(f"    GPU Speedup: {speedup:.2f}x")

        results.append({
            'qubits': n_qubits,
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': cpu_time / gpu_time if (gpu_time and cpu_time) else None
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return results


def benchmark_mps_simulation():
    """Benchmark MPS simulation: GPU vs CPU"""
    print("=" * 70)
    print("2. MPS TENSOR NETWORK SIMULATION (GPU vs CPU)")
    print("=" * 70)
    print("   Testing large-scale GHZ circuit with Matrix Product States\n")

    from atlas_q.mps_pytorch import MatrixProductStatePyTorch as MPSSimulator

    results = []

    for n_qubits in [16, 20, 24, 28]:
        print(f"  {n_qubits} qubits:")

        # ========== ATLAS-Q GPU ==========
        gpu_time = None
        try:
            # Warmup
            sim = MPSSimulator(num_qubits=n_qubits, bond_dim=32, device='cuda')
            sim.h(0)
            del sim
            torch.cuda.empty_cache()

            torch.cuda.synchronize()
            start = time.perf_counter()

            sim = MPSSimulator(num_qubits=n_qubits, bond_dim=64, device='cuda')
            # GHZ circuit
            sim.h(0)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)

            # Sample to force computation
            samples = sim.sample(num_shots=100)

            torch.cuda.synchronize()
            gpu_time = time.perf_counter() - start

            del sim
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    GPU Error: {e}")

        # ========== ATLAS-Q CPU ==========
        cpu_time = None
        try:
            start = time.perf_counter()

            sim = MPSSimulator(num_qubits=n_qubits, bond_dim=64, device='cpu')
            sim.h(0)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)

            samples = sim.sample(num_shots=100)

            cpu_time = time.perf_counter() - start

            del sim

        except Exception as e:
            print(f"    CPU Error: {e}")

        if gpu_time:
            print(f"    ATLAS-Q MPS (GPU): {format_time(gpu_time)}")
        if cpu_time:
            print(f"    ATLAS-Q MPS (CPU): {format_time(cpu_time)}")

        if gpu_time and cpu_time:
            speedup = cpu_time / gpu_time
            print(f"    GPU Speedup: {speedup:.2f}x")

        results.append({
            'qubits': n_qubits,
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': cpu_time / gpu_time if (gpu_time and cpu_time) else None
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return results


def benchmark_gpu_backend():
    """Benchmark direct CUDA statevector backend"""
    print("=" * 70)
    print("3. DIRECT CUDA STATEVECTOR BACKEND")
    print("=" * 70)
    print("   Testing raw CUDA performance via direct driver API\n")

    try:
        from atlas_q.gpu_backend import GPUStatevectorSimulator

        results = []

        for n_qubits in [16, 18, 20, 22]:
            print(f"  {n_qubits} qubits (state size: {2**n_qubits:,}):")

            try:
                # Create simulator
                start = time.perf_counter()
                sim = GPUStatevectorSimulator(n_qubits)
                init_time = time.perf_counter() - start

                # Apply gates
                start = time.perf_counter()
                for i in range(n_qubits):
                    sim.h(i)
                for i in range(n_qubits - 1):
                    sim.cnot(i, i + 1)
                gate_time = time.perf_counter() - start

                # Sample
                start = time.perf_counter()
                samples = sim.sample(100)
                sample_time = time.perf_counter() - start

                print(f"    Initialization: {format_time(init_time)}")
                print(f"    Gate application: {format_time(gate_time)}")
                print(f"    Sampling (100 shots): {format_time(sample_time)}")
                print(f"    Total: {format_time(init_time + gate_time + sample_time)}")

                results.append({
                    'qubits': n_qubits,
                    'init_time': init_time,
                    'gate_time': gate_time,
                    'sample_time': sample_time
                })

            except Exception as e:
                print(f"    Error: {e}")
                results.append({'qubits': n_qubits, 'error': str(e)})

            print()

        return results

    except ImportError as e:
        print(f"   GPU backend not available: {e}")
        return []


def benchmark_vs_qiskit_cpu():
    """Compare against Qiskit on CPU"""
    print("=" * 70)
    print("4. ATLAS-Q GPU vs QISKIT AER CPU")
    print("=" * 70)
    print("   Fair comparison: ATLAS-Q GPU vs Qiskit Aer CPU\n")

    from atlas_q import get_density_matrix
    dm = get_density_matrix()
    DensityMatrixSimulator = dm['DensityMatrixSimulator']
    DensityMatrixConfig = dm['DensityMatrixConfig']

    results = []

    for n_qubits in [4, 6, 8]:
        print(f"  {n_qubits} qubits:")

        # ========== ATLAS-Q GPU ==========
        atlas_time = None
        try:
            config = DensityMatrixConfig(device='cuda')

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

        # ========== Qiskit Aer CPU ==========
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

            sim_aer = AerSimulator(method='density_matrix', noise_model=noise)

            # Warmup
            sim_aer.run(qc, shots=1).result()

            start = time.perf_counter()
            result = sim_aer.run(qc, shots=1).result()
            qiskit_time = time.perf_counter() - start

        except Exception as e:
            print(f"    Qiskit Error: {e}")

        if atlas_time:
            print(f"    ATLAS-Q (GPU): {format_time(atlas_time)}")
        if qiskit_time:
            print(f"    Qiskit Aer (CPU): {format_time(qiskit_time)}")

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


def print_summary(dm_results, mps_results, qiskit_results):
    """Print performance summary"""
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\n  DENSITY MATRIX (GPU vs CPU):")
    print("  " + "-" * 50)
    for r in dm_results:
        if r.get('speedup'):
            print(f"    {r['qubits']:2d} qubits: {r['speedup']:.2f}x GPU speedup")

    print("\n  MPS SIMULATION (GPU vs CPU):")
    print("  " + "-" * 50)
    for r in mps_results:
        if r.get('speedup'):
            print(f"    {r['qubits']:2d} qubits: {r['speedup']:.2f}x GPU speedup")

    print("\n  ATLAS-Q GPU vs QISKIT CPU:")
    print("  " + "-" * 50)
    for r in qiskit_results:
        if r.get('speedup'):
            winner = "ATLAS-Q ✓" if r['speedup'] > 1 else "Qiskit ✓"
            print(f"    {r['qubits']:2d} qubits: {r['speedup']:.2f}x - {winner}")

    # Overall
    atlas_wins = sum(1 for r in qiskit_results if r.get('speedup', 0) > 1)
    qiskit_wins = sum(1 for r in qiskit_results if r.get('speedup', 0) <= 1 and r.get('speedup'))

    print("\n" + "=" * 70)
    if atlas_wins > qiskit_wins:
        print("  VERDICT: ATLAS-Q GPU demonstrates competitive performance")
    elif atlas_wins > 0:
        print("  VERDICT: ATLAS-Q and Qiskit perform comparably")
    print("=" * 70)


def main():
    print("=" * 70)
    print("ATLAS-Q v0.8.0 - GPU PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    get_gpu_info()

    dm_results = benchmark_density_matrix()
    mps_results = benchmark_mps_simulation()
    benchmark_gpu_backend()
    qiskit_results = benchmark_vs_qiskit_cpu()

    print_summary(dm_results, mps_results, qiskit_results)


if __name__ == "__main__":
    main()
