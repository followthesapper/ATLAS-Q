"""
Demonstration of Adaptive MPS for Moderate-to-High Entanglement

Showcases:
- Bell pair creation
- GHZ state generation
- Moderate entanglement quantum circuits
- Adaptive truncation in action
- Error tracking and bond dimension evolution

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import time
from atlas_q.adaptive_mps import AdaptiveMPS, DTypePolicy


def demo_bell_pair():
    """Demonstrate Bell pair creation with minimal truncation"""
    print("=" * 70)
    print("DEMO 1: Bell Pair Creation")
    print("=" * 70)
    print("Creating |Φ⁺⟩ = (|00⟩ + |11⟩)/√2\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mps = AdaptiveMPS(
        num_qubits=2,
        bond_dim=2,
        eps_bond=1e-10,
        chi_max_per_bond=4,
        device=device
    )

    # Define gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CNOT = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex64)

    # Create Bell pair
    print("Step 1: Apply Hadamard to qubit 0")
    mps.apply_single_qubit_gate(0, H)

    print("Step 2: Apply CNOT(0→1)")
    mps.apply_two_site_gate(0, CNOT)

    # Check results
    stats = mps.stats_summary()
    print(f"\nResults:")
    print(f"  Bond dimension: {mps.bond_dims[0]}")
    print(f"  Entanglement entropy: {stats['mean_entropy']:.4f} bits")
    print(f"  Global error bound: {mps.global_error_bound():.2e}")
    print(f"  Memory usage: {mps.get_memory_usage() / 1024:.2f} KB")

    # Verify correct Bell state
    psi = mps.to_statevector()
    print(f"\nState vector amplitudes:")
    for i, amp in enumerate(psi):
        if abs(amp) > 1e-10:
            z = complex(amp.real.item(), amp.imag.item())
            print(f"  |{i:02b}⟩: {z.real:+.4f}{z.imag:+.4f}i")

    print()


def demo_ghz_state(n=5):
    """Demonstrate GHZ state |000...0⟩ + |111...1⟩"""
    print("=" * 70)
    print(f"DEMO 2: GHZ State ({n} qubits)")
    print("=" * 70)
    print(f"Creating |{'0'*n}⟩ + |{'1'*n}⟩\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mps = AdaptiveMPS(
        num_qubits=n,
        bond_dim=2,
        eps_bond=1e-10,
        chi_max_per_bond=4,
        device=device
    )

    # Define gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CNOT = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex64)

    # Create GHZ state
    print("Step 1: H on qubit 0")
    mps.apply_single_qubit_gate(0, H)

    print("Step 2: CNOT chain 0→1→2→...→n-1")
    for i in range(n - 1):
        mps.apply_two_site_gate(i, CNOT)

    # Check results
    stats = mps.stats_summary()
    print(f"\nResults:")
    print(f"  Bond dimensions: {mps.bond_dims}")
    print(f"  Max χ: {stats['max_chi']}")
    print(f"  Mean entropy: {stats['mean_entropy']:.4f} bits")
    print(f"  Global error: {mps.global_error_bound():.2e}")
    print(f"  Total time: {stats['total_time_ms']:.2f} ms")

    print()


def demo_moderate_entanglement(n=10):
    """Demonstrate moderate entanglement with adaptive truncation"""
    print("=" * 70)
    print(f"DEMO 3: Moderate Entanglement Circuit ({n} qubits)")
    print("=" * 70)
    print("Alternating layers of Hadamards and CZ gates\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mps = AdaptiveMPS(
        num_qubits=n,
        bond_dim=4,
        eps_bond=1e-6,
        chi_max_per_bond=32,
        device=device
    )

    # Define gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

    layers = 4
    print(f"Applying {layers} layers of gates...\n")

    for layer in range(layers):
        print(f"Layer {layer + 1}:")

        # Hadamard layer
        print("  - Hadamards on all qubits")
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)

        # Entangling layer (even/odd pattern)
        offset = layer % 2
        print(f"  - CZ gates (offset={offset})")
        for i in range(offset, n - 1, 2):
            mps.apply_two_site_gate(i, CZ)

        # Check bond dimensions after each layer
        max_chi = max(mps.bond_dims)
        print(f"  → Max χ after layer: {max_chi}")

    # Final statistics
    print("\n" + "-" * 70)
    stats = mps.stats_summary()
    print(f"Final Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Max bond dimension: {stats['max_chi']}")
    print(f"  Mean bond dimension: {stats['mean_chi']:.2f}")
    print(f"  Mean entropy: {stats['mean_entropy']:.4f} bits")
    print(f"  Max entropy: {stats['p95_entropy']:.4f} bits (95th percentile)")
    print(f"  Global error bound: {mps.global_error_bound():.2e}")
    print(f"  Total computation time: {stats['total_time_ms']:.2f} ms")
    print(f"  CUDA SVD usage: {stats['cuda_svd_pct']:.1f}%")
    print(f"  Memory usage: {mps.get_memory_usage() / (1024**2):.2f} MB")

    print()


def demo_adaptive_truncation():
    """Demonstrate adaptive truncation with different tolerances"""
    print("=" * 70)
    print("DEMO 4: Adaptive Truncation Comparison")
    print("=" * 70)
    print("Same circuit with different truncation tolerances\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 8
    tolerances = [1e-3, 1e-6, 1e-9]

    # Define gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

    print(f"{'Tolerance':<12} {'Max χ':<10} {'Mean χ':<10} {'Error':<12} {'Memory (KB)'}")
    print("-" * 70)

    for eps in tolerances:
        mps = AdaptiveMPS(
            num_qubits=n,
            bond_dim=4,
            eps_bond=eps,
            chi_max_per_bond=64,
            device=device
        )

        # Apply circuit
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)
        for i in range(n - 1):
            mps.apply_two_site_gate(i, CZ)
        for i in range(n):
            mps.apply_single_qubit_gate(i, H)

        stats = mps.stats_summary()
        mem_kb = mps.get_memory_usage() / 1024

        print(f"{eps:<12.0e} {stats['max_chi']:<10.0f} {stats['mean_chi']:<10.2f} "
              f"{mps.global_error_bound():<12.2e} {mem_kb:<.2f}")

    print()


def demo_large_scale(n=50):
    """Demonstrate large-scale simulation with moderate entanglement"""
    print("=" * 70)
    print(f"DEMO 5: Large-Scale Simulation ({n} qubits)")
    print("=" * 70)
    print("Showcasing GPU acceleration and memory efficiency\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping large-scale demo")
        return

    device = 'cuda'
    mps = AdaptiveMPS(
        num_qubits=n,
        bond_dim=8,
        eps_bond=1e-6,
        chi_max_per_bond=64,
        device=device
    )

    # Define gates
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64))

    print(f"Initial memory: {mps.get_memory_usage() / (1024**2):.2f} MB")
    print()

    start_time = time.time()

    # Layer 1: Hadamards
    print("Layer 1: Hadamards on all qubits...")
    for i in range(n):
        mps.apply_single_qubit_gate(i, H)
    print(f"  → Max χ: {max(mps.bond_dims)}, Memory: {mps.get_memory_usage() / (1024**2):.2f} MB")

    # Layer 2: Entangling (even bonds)
    print("Layer 2: CZ gates on even bonds...")
    for i in range(0, n - 1, 2):
        mps.apply_two_site_gate(i, CZ)
    print(f"  → Max χ: {max(mps.bond_dims)}, Memory: {mps.get_memory_usage() / (1024**2):.2f} MB")

    # Layer 3: More Hadamards
    print("Layer 3: Hadamards on all qubits...")
    for i in range(n):
        mps.apply_single_qubit_gate(i, H)
    print(f"  → Max χ: {max(mps.bond_dims)}, Memory: {mps.get_memory_usage() / (1024**2):.2f} MB")

    # Layer 4: Entangling (odd bonds)
    print("Layer 4: CZ gates on odd bonds...")
    for i in range(1, n - 1, 2):
        mps.apply_two_site_gate(i, CZ)
    print(f"  → Max χ: {max(mps.bond_dims)}, Memory: {mps.get_memory_usage() / (1024**2):.2f} MB")

    elapsed = time.time() - start_time

    stats = mps.stats_summary()
    print(f"\nFinal Results:")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Operations/sec: {stats['total_operations'] / elapsed:.1f}")
    print(f"  Max χ: {stats['max_chi']}")
    print(f"  Mean χ: {stats['mean_chi']:.2f}")
    print(f"  Global error: {mps.global_error_bound():.2e}")
    print(f"  Final memory: {mps.get_memory_usage() / (1024**2):.2f} MB")
    print(f"  CUDA SVD: {stats['cuda_svd_pct']:.1f}%")

    print()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("ADAPTIVE MPS DEMONSTRATION")
    print("Moderate-to-High Entanglement Quantum Simulation")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print()

    # Run demos
    demo_bell_pair()
    demo_ghz_state(n=5)
    demo_moderate_entanglement(n=10)
    demo_adaptive_truncation()
    demo_large_scale(n=50)

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ Adaptive MPS handles moderate entanglement efficiently")
    print("✓ Bond dimensions grow only where needed")
    print("✓ Error bounds are rigorously tracked")
    print("✓ GPU acceleration enables large-scale simulations")
    print("✓ Memory usage scales linearly with n·χ²")
    print()


if __name__ == '__main__':
    main()
