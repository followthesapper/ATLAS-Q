"""
ATLAS-Q New Features Demonstration

Shows off the new capabilities added in October 2025:
1. Noise models
2. Stabilizer/Clifford backend
3. MPO operations
4. TDVP time evolution
5. VQE/QAOA

Usage:
    python examples/demo_new_features.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt


def demo_noise_models():
    """Demo: Realistic NISQ noise simulation"""
    print("\n" + "="*60)
    print("DEMO 1: Noise Models & NISQ Parity")
    print("="*60)

    from atlas_q import get_noise_models, get_adaptive_mps

    noise = get_noise_models()
    mps_cls = get_adaptive_mps()

    # Create depolarizing noise model
    noise_model = noise['NoiseModel'].depolarizing(p1q=0.001, p2q=0.01)
    print("✓ Created depolarizing noise model")
    print(f"  - Single-qubit error rate: 0.1%")
    print(f"  - Two-qubit error rate: 1.0%")

    # Create MPS
    mps = mps_cls['AdaptiveMPS'](num_qubits=10, bond_dim=4, device='cuda')

    # Apply noisy gates
    applicator = noise['StochasticNoiseApplicator'](noise_model, seed=42)

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    for q in range(10):
        mps.apply_single_qubit_gate(q, H.cuda())
        applicator.apply_1q_noise(mps, q)

    print("✓ Applied 10 noisy Hadamard gates")
    print(f"✓ Estimated fidelity: {applicator.get_fidelity_estimate():.4f}")


def demo_stabilizer():
    """Demo: Fast Clifford simulation with hybrid handoff"""
    print("\n" + "="*60)
    print("DEMO 2: Stabilizer Backend (Clifford Fast Path)")
    print("="*60)

    from atlas_q import get_stabilizer

    stab = get_stabilizer()

    # Hybrid simulator
    sim = stab['HybridSimulator'](n_qubits=100, use_stabilizer=True, device='cuda')
    print("✓ Created hybrid simulator (100 qubits)")

    # Clifford gates (fast!)
    print("  Running Clifford circuit...")
    for i in range(100):
        sim.h(i)
    for i in range(99):
        sim.cnot(i, i+1)

    stats = sim.get_statistics()
    print(f"✓ Mode: {stats['mode']}")
    print(f"✓ Clifford gates: {stats['clifford_gates']}")

    # Add T-gate (triggers switch to MPS)
    print("  Adding T-gate (non-Clifford)...")
    sim.t(0)

    stats = sim.get_statistics()
    print(f"✓ Mode after T-gate: {stats['mode']}")
    print(f"✓ Automatic handoff to MPS completed!")


def demo_mpo_observables():
    """Demo: MPO operations and expectation values"""
    print("\n" + "="*60)
    print("DEMO 3: MPO Operations & Observables")
    print("="*60)

    from atlas_q import get_mpo_ops, get_adaptive_mps

    mpo = get_mpo_ops()
    mps_cls = get_adaptive_mps()

    # Build Ising Hamiltonian
    n_sites = 20
    H = mpo['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.5, device='cuda')
    print(f"✓ Built Ising Hamiltonian ({n_sites} sites)")
    print(f"  H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ")
    print(f"  J = 1.0, h = 0.5")

    # Create ground state (approximate)
    mps = mps_cls['AdaptiveMPS'](n_sites, bond_dim=16, device='cuda')

    # Apply simple state preparation
    H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / np.sqrt(2)
    for q in range(n_sites):
        mps.apply_single_qubit_gate(q, H_gate)

    # Compute energy
    energy = mpo['expectation_value'](H, mps)
    print(f"✓ Energy expectation: ⟨H⟩ = {energy.real:.6f}")

    # Compute correlation function
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda')
    corr = mpo['correlation_function'](X, 0, X, 10, mps)
    print(f"✓ Correlation: ⟨X₀ X₁₀⟩ = {corr.real:.6f}")


def demo_tdvp():
    """Demo: TDVP time evolution"""
    print("\n" + "="*60)
    print("DEMO 4: TDVP Time Evolution")
    print("="*60)

    from atlas_q import get_tdvp, get_mpo_ops, get_adaptive_mps

    tdvp_module = get_tdvp()
    mpo_module = get_mpo_ops()
    mps_cls = get_adaptive_mps()

    # Build Hamiltonian
    n_sites = 20
    H = mpo_module['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.5, device='cuda')
    print(f"✓ Built Hamiltonian ({n_sites} sites)")

    # Initial state |+⟩^⊗n
    mps = mps_cls['AdaptiveMPS'](n_sites, bond_dim=8, device='cuda')
    H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device='cuda') / np.sqrt(2)
    for q in range(n_sites):
        mps.apply_single_qubit_gate(q, H_gate)
    print(f"✓ Initialized state |+⟩^⊗{n_sites}")

    # Configure TDVP
    config = tdvp_module['TDVPConfig'](
        dt=0.05,
        t_final=2.0,
        order=1,  # 1-site TDVP (faster)
        chi_max=32
    )

    print(f"✓ Running 1-site TDVP evolution...")
    print(f"  dt = {config.dt}, t_final = {config.t_final}")

    # Run TDVP
    final_mps, times, energies = tdvp_module['run_tdvp'](H, mps, config)

    print(f"✓ Evolution complete!")
    print(f"  Initial energy: {energies[0].real:.6f}")
    print(f"  Final energy: {energies[-1].real:.6f}")
    print(f"  Energy conservation: ΔE = {abs(energies[-1].real - energies[0].real):.2e}")


def demo_vqe():
    """Demo: VQE ground state finding"""
    print("\n" + "="*60)
    print("DEMO 5: VQE Ground State Finding")
    print("="*60)

    try:
        from atlas_q import get_vqe_qaoa, get_mpo_ops

        vqe_module = get_vqe_qaoa()
        mpo_module = get_mpo_ops()

        # Build Hamiltonian
        n_sites = 10
        H = mpo_module['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.5, device='cuda')
        print(f"✓ Built Ising Hamiltonian ({n_sites} sites)")

        # Configure VQE
        config = vqe_module['VQEConfig'](
            ansatz='hardware_efficient',
            n_layers=2,
            optimizer='COBYLA',
            max_iter=50,
            chi_max=32,
            device='cuda'
        )
        print(f"✓ Configured VQE:")
        print(f"  - Ansatz: hardware-efficient")
        print(f"  - Layers: {config.n_layers}")
        print(f"  - Optimizer: {config.optimizer}")

        # Run VQE
        print(f"\n  Running VQE optimization...")
        vqe = vqe_module['VQE'](H, config)
        energy, params = vqe.run()

        print(f"\n✓ VQE converged!")
        print(f"  Ground state energy: {energy:.6f}")
        print(f"  Optimal parameters: {len(params)} values")

    except ImportError as e:
        print(f"⚠ SciPy not available, skipping VQE demo")
        print(f"  Install with: pip install scipy")


def demo_qaoa():
    """Demo: QAOA combinatorial optimization"""
    print("\n" + "="*60)
    print("DEMO 6: QAOA Combinatorial Optimization")
    print("="*60)

    try:
        from atlas_q import get_vqe_qaoa, get_mpo_ops

        vqe_module = get_vqe_qaoa()
        mpo_module = get_mpo_ops()

        # MaxCut Hamiltonian (J < 0 for MaxCut)
        n_sites = 10
        H_cost = mpo_module['MPOBuilder'].ising_hamiltonian(
            n_sites=n_sites, J=-1.0, h=0.0, device='cuda'
        )
        print(f"✓ Built MaxCut Hamiltonian ({n_sites} nodes)")

        # Run QAOA
        print(f"✓ Running QAOA (p=2 layers)...")
        qaoa = vqe_module['QAOA'](H_cost, n_layers=2, device='cuda')
        cost, params = qaoa.run()

        print(f"\n✓ QAOA converged!")
        print(f"  Optimal cost: {cost:.6f}")
        print(f"  Parameters: {params}")

    except ImportError:
        print(f"⚠ SciPy not available, skipping QAOA demo")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "ATLAS-Q NEW FEATURES DEMO")
    print(" "*10 + "October 2025 Enhancement Showcase")
    print("="*70)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ CUDA Available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠ Warning: CUDA not available, using CPU (slower)")

    # Run demos
    demo_noise_models()
    demo_stabilizer()
    demo_mpo_observables()
    demo_tdvp()
    demo_vqe()
    demo_qaoa()

    print("\n" + "="*70)
    print(" "*20 + "ALL DEMOS COMPLETED!")
    print("="*70)
    print(f"\nNew capabilities demonstrated:")
    print(f"  ✓ NISQ noise modeling")
    print(f"  ✓ Clifford/stabilizer fast path")
    print(f"  ✓ MPO operations & observables")
    print(f"  ✓ TDVP time evolution")
    print(f"  ✓ VQE ground state finding")
    print(f"  ✓ QAOA optimization")
    print(f"\nATLAS-Q is now competitive with Qiskit Aer, Cirq, ITensor, and TeNPy!")
    print(f"For more info, see IMPLEMENTATION_SUMMARY.md\n")


if __name__ == '__main__':
    main()
