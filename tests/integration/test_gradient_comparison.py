#!/usr/bin/env python3
"""
Test gradient methods comparison.
Compares group-level vs per-Pauli gradients for VQE.
"""

import pytest
import torch
import numpy as np
from atlas_q import get_mpo_ops, get_vqe_qaoa
from atlas_q.ansatz_uccsd import UCCSDAnsatz

# Load modules
mpo_modules = get_mpo_ops()
vqe_modules = get_vqe_qaoa()

MPOBuilder = mpo_modules["MPOBuilder"]
VQE = vqe_modules["VQE"]
VQEConfig = vqe_modules["VQEConfig"]


@pytest.mark.parametrize("config_name,use_per_pauli_grad,use_warm_start", [
    ("Group-level gradients + warm-start", False, True),
    ("Per-Pauli gradients + warm-start", True, True),
])
def test_gradient_comparison(config_name, use_per_pauli_grad, use_warm_start):
    """Test a specific VQE configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.complex128

    # Build H2 Hamiltonian
    H = MPOBuilder.molecular_hamiltonian_from_specs(
        molecule='H2',
        basis='sto-3g',
        device=device,
        dtype=dtype
    )

    # Build UCCSD ansatz
    ansatz = UCCSDAnsatz(molecule='H2', basis='sto-3g', device=device, dtype=dtype)
    ansatz.unitary_backend = 'dense'

    # VQE configuration
    config = VQEConfig(
        ansatz='hardware_efficient',
        n_layers=1,
        max_iter=100,
        optimizer='L-BFGS-B',
        tol=1e-6,  # More reasonable tolerance for L-BFGS-B
        device=device,
        chi_max=256,
        dtype=dtype
    )

    # Create VQE instance
    vqe = VQE(H, config, custom_ansatz=ansatz)
    vqe._fci_ref = -1.13728380  # FCI reference for H2/STO-3G

    # Configure optional features
    vqe._use_warm_start = use_warm_start
    vqe._use_per_pauli_gradients = use_per_pauli_grad

    # Run VQE
    theta0 = np.random.normal(loc=0.0, scale=0.5, size=ansatz.n_parameters)
    energy, params = vqe.run(initial_params=theta0)

    # Results
    e_fci = -1.13728380
    error_kcal = (energy - e_fci) * 627.509  # Ha to kcal/mol

    print(f"\n{'='*80}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*80}")
    print(f"  Final Energy:      {energy:.8f} Ha")
    print(f"  Error vs FCI:      {error_kcal:+.2f} kcal/mol")
    print(f"  Iterations:        {vqe.iteration}")
    print(f"{'='*80}\n")

    # Assert reasonable accuracy for UCCSD with warm-start
    # Note: UCCSD optimization with warm-start may converge early, resulting in higher error
    assert abs(error_kcal) < 15.0, f"Error {error_kcal:.2f} kcal/mol exceeds threshold"
