#!/usr/bin/env python3
"""
Benchmark: Rust Statevector vs Python/NumPy Statevector

Tests performance on various quantum circuits
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/admin/ATLAS-Q')
import atlas_q_core


class PythonStatevector:
    """Simple Python statevector implementation for comparison"""

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # |00...0⟩

    def apply_single_gate(self, qubit, gate):
        """Apply single-qubit gate"""
        size = len(self.state)
        qubit_mask = 1 << qubit

        for i in range(size):
            if (i & qubit_mask) == 0:
                j = i | qubit_mask
                amp0 = self.state[i]
                amp1 = self.state[j]

                self.state[i] = gate[0, 0] * amp0 + gate[0, 1] * amp1
                self.state[j] = gate[1, 0] * amp0 + gate[1, 1] * amp1

    def h(self, qubit):
        """Hadamard gate"""
        sqrt2_inv = 1.0 / np.sqrt(2)
        gate = np.array([[sqrt2_inv, sqrt2_inv],
                        [sqrt2_inv, -sqrt2_inv]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def x(self, qubit):
        """Pauli X gate"""
        gate = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def y(self, qubit):
        """Pauli Y gate"""
        gate = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def z(self, qubit):
        """Pauli Z gate"""
        gate = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def t(self, qubit):
        """T gate"""
        gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def rx(self, qubit, theta):
        """RX rotation"""
        half = theta / 2
        gate = np.array([[np.cos(half), -1j * np.sin(half)],
                        [-1j * np.sin(half), np.cos(half)]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def ry(self, qubit, theta):
        """RY rotation"""
        half = theta / 2
        gate = np.array([[np.cos(half), -np.sin(half)],
                        [np.sin(half), np.cos(half)]], dtype=np.complex128)
        self.apply_single_gate(qubit, gate)

    def cnot(self, control, target):
        """CNOT gate"""
        size = len(self.state)
        mask_control = 1 << control
        mask_target = 1 << target

        for i in range(size):
            if (i & mask_control) and (i & mask_target) == 0:
                j = i | mask_target
                self.state[i], self.state[j] = self.state[j], self.state[i]

    def sample(self, num_shots):
        """Sample measurement outcomes"""
        probs = np.abs(self.state) ** 2
        probs /= np.sum(probs)
        return np.random.choice(len(self.state), size=num_shots, p=probs)


def bench_rust_statevector(n_qubits, circuit_type, n_trials=10):
    """Benchmark Rust statevector"""
    times = []

    for _ in range(n_trials):
        sim = atlas_q_core.StatevectorSimulatorRust(n_qubits)

        start = time.perf_counter()

        if circuit_type == 'ghz':
            # GHZ state
            sim.h(0)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)

        elif circuit_type == 'random_clifford':
            # Random Clifford gates
            for _ in range(n_qubits * 5):
                gate = np.random.randint(0, 4)
                qubit = np.random.randint(0, n_qubits)
                if gate == 0:
                    sim.h(qubit)
                elif gate == 1:
                    sim.x(qubit)
                elif gate == 2:
                    sim.z(qubit)
                else:
                    if n_qubits > 1:
                        target = (qubit + 1) % n_qubits
                        sim.cnot(qubit, target)

        elif circuit_type == 'rotation':
            # Rotation gates (non-Clifford)
            for i in range(n_qubits):
                theta = np.random.uniform(0, 2 * np.pi)
                sim.rx(i, theta)
                sim.ry(i, theta / 2)

        elif circuit_type == 'grover_like':
            # Grover-like circuit
            for i in range(n_qubits):
                sim.h(i)
            for _ in range(3):
                for i in range(n_qubits - 1):
                    sim.cnot(i, i + 1)
                for i in range(n_qubits):
                    sim.t(i)

        # Sample
        samples = sim.sample(100)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times) * 1000, np.std(times) * 1000  # Convert to ms


def bench_python_statevector(n_qubits, circuit_type, n_trials=10):
    """Benchmark Python statevector"""
    times = []

    for _ in range(n_trials):
        sim = PythonStatevector(n_qubits)

        start = time.perf_counter()

        if circuit_type == 'ghz':
            sim.h(0)
            for i in range(n_qubits - 1):
                sim.cnot(i, i + 1)

        elif circuit_type == 'random_clifford':
            for _ in range(n_qubits * 5):
                gate = np.random.randint(0, 4)
                qubit = np.random.randint(0, n_qubits)
                if gate == 0:
                    sim.h(qubit)
                elif gate == 1:
                    sim.x(qubit)
                elif gate == 2:
                    sim.z(qubit)
                else:
                    if n_qubits > 1:
                        target = (qubit + 1) % n_qubits
                        sim.cnot(qubit, target)

        elif circuit_type == 'rotation':
            for i in range(n_qubits):
                theta = np.random.uniform(0, 2 * np.pi)
                sim.rx(i, theta)
                sim.ry(i, theta / 2)

        elif circuit_type == 'grover_like':
            for i in range(n_qubits):
                sim.h(i)
            for _ in range(3):
                for i in range(n_qubits - 1):
                    sim.cnot(i, i + 1)
                for i in range(n_qubits):
                    sim.t(i)

        # Sample
        samples = sim.sample(100)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times) * 1000, np.std(times) * 1000


print("=" * 90)
print("STATEVECTOR BENCHMARK: Rust vs Python/NumPy")
print("=" * 90)
print()

# Test different circuit types and sizes
circuits = ['ghz', 'random_clifford', 'rotation', 'grover_like']
qubit_counts = [5, 10, 15, 18]

for circuit_type in circuits:
    print(f"\n{'='*90}")
    print(f"Circuit Type: {circuit_type.upper()}")
    print(f"{'='*90}")
    print(f"{'Qubits':<8} {'Rust (ms)':<15} {'Python (ms)':<15} {'Speedup':<12}")
    print("-" * 90)

    for n_qubits in qubit_counts:
        # Skip large circuits for Python (too slow)
        if n_qubits > 15 and circuit_type in ['grover_like', 'rotation']:
            print(f"{n_qubits:<8} {'✓ Rust only':<15} {'Too slow':<15} {'-':<12}")
            continue

        if n_qubits > 12:
            # Reduce trials for larger circuits
            n_trials = 3
        else:
            n_trials = 10

        try:
            rust_mean, rust_std = bench_rust_statevector(n_qubits, circuit_type, n_trials)

            # Skip Python for very large circuits
            if n_qubits <= 15:
                python_mean, python_std = bench_python_statevector(n_qubits, circuit_type, n_trials)
                speedup = python_mean / rust_mean
                print(f"{n_qubits:<8} {rust_mean:>8.2f}±{rust_std:>4.1f}  {python_mean:>8.2f}±{python_std:>4.1f}  {speedup:>8.1f}×")
            else:
                print(f"{n_qubits:<8} {rust_mean:>8.2f}±{rust_std:>4.1f}  {'N/A':<15} {'-':<12}")

        except Exception as e:
            print(f"{n_qubits:<8} Error: {e}")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print("""
Rust Statevector Performance:
  ✓ Handles circuits up to 18 qubits (Python struggles at 15)
  ✓ Parallel execution for large states
  ✓ SIMD-optimized complex arithmetic
  ✓ Zero-copy operations where possible

Python Statevector (NumPy):
  - Pure Python loops (slow)
  - No parallelization
  - Good for prototyping, not production

Use Cases:
  - Rust: Production workloads, Grover's, QFT, VQE (small), general circuits
  - Python: Quick prototyping, educational purposes
""")

print("\n✓ Benchmark complete!")
