"""
Comprehensive ATLAS-Q Benchmarking Suite

Tests all v0.5.0 features against:
- Baseline expectations
- Analytical solutions where available
- Performance targets

Benchmarks:
1. Noise models (fidelity tracking, Kraus completeness)
2. Stabilizer backend (Clifford speed vs MPS)
3. MPO operations (energy calculations)
4. TDVP (energy conservation, accuracy)
5. VQE/QAOA (convergence to known ground states)
6. 2D circuits (SWAP overhead)
7. Overall integration tests

Author: ATLAS-Q Contributors
Date: October 26, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import time
from typing import Dict, List, Tuple


class BenchmarkRunner:
    """Runs comprehensive benchmarks for ATLAS-Q"""

    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {}

        print(f"\n{'='*70}")
        print(f"  ATLAS-Q v0.5.0 Comprehensive Benchmark Suite")
        print(f"{'='*70}")
        print(f"\nDevice: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"\n{'='*70}\n")

    def run_all_benchmarks(self):
        """Run all benchmark suites"""
        self.benchmark_noise_models()
        self.benchmark_stabilizer()
        self.benchmark_mpo_ops()
        self.benchmark_tdvp()
        self.benchmark_vqe()
        self.benchmark_2d_circuits()
        self.benchmark_integration()

        self.print_summary()

    def benchmark_noise_models(self):
        """Test noise model correctness and performance"""
        print("\n" + "="*70)
        print("BENCHMARK 1: Noise Models & NISQ Parity")
        print("="*70)

        from atlas_q import get_noise_models, get_adaptive_mps

        noise = get_noise_models()
        mps_cls = get_adaptive_mps()

        tests = {}

        # Test 1: Kraus operator completeness
        print("\n[1.1] Testing Kraus operator completeness...")
        noise_model = noise['NoiseModel'].depolarizing(p1q=0.1, device=self.device)
        channel = noise_model.channels_1q['depolarizing']

        completeness = sum(K.conj().T @ K for K in channel.kraus_ops)
        expected = torch.eye(2, dtype=torch.complex64, device=self.device)
        error = torch.norm(completeness - expected).item()

        tests['kraus_completeness_error'] = error
        print(f"  Completeness error: {error:.2e} (target: <1e-10)")
        print(f"  ✅ PASS" if error < 1e-10 else f"  ❌ FAIL")

        # Test 2: Fidelity tracking
        print("\n[1.2] Testing fidelity tracking...")
        applicator = noise['StochasticNoiseApplicator'](noise_model, seed=42)
        mps = mps_cls['AdaptiveMPS'](5, bond_dim=4, device=self.device)

        # Apply 20 noisy gates
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        for _ in range(20):
            mps.apply_single_qubit_gate(0, H)
            applicator.apply_1q_noise(mps, 0)

        fidelity = applicator.get_fidelity_estimate()
        expected_fidelity = (1 - 0.1) ** 20  # Rough estimate

        tests['fidelity_after_20_gates'] = fidelity
        tests['fidelity_matches_expectation'] = abs(fidelity - expected_fidelity) < 0.15

        print(f"  Fidelity after 20 gates: {fidelity:.4f}")
        print(f"  Expected (rough): {expected_fidelity:.4f}")
        print(f"  ✅ PASS" if tests['fidelity_matches_expectation'] else f"  ⚠️  WARNING")

        # Test 3: Performance
        print("\n[1.3] Testing noise application performance...")
        n_trials = 100
        start = time.time()
        for _ in range(n_trials):
            applicator.apply_1q_noise(mps, 0)
        elapsed = time.time() - start

        tests['noise_ops_per_sec'] = n_trials / elapsed
        print(f"  Noise ops/sec: {tests['noise_ops_per_sec']:.1f}")
        print(f"  ✅ PASS (target: >1000 ops/sec)" if tests['noise_ops_per_sec'] > 1000 else f"  ⚠️  SLOW")

        self.results['noise_models'] = tests

    def benchmark_stabilizer(self):
        """Test stabilizer backend speed vs MPS"""
        print("\n" + "="*70)
        print("BENCHMARK 2: Stabilizer Backend (Clifford Fast Path)")
        print("="*70)

        from atlas_q import get_stabilizer, get_adaptive_mps

        stab = get_stabilizer()
        mps_cls = get_adaptive_mps()

        tests = {}

        # Test 1: Correctness (Bell state)
        print("\n[2.1] Testing Bell state creation...")

        outcomes = []
        for _ in range(100):
            sim_copy = stab['StabilizerSimulator'](n_qubits=2)
            sim_copy.h(0)
            sim_copy.cnot(0, 1)
            m0 = sim_copy.measure(0)
            m1 = sim_copy.measure(1)
            outcomes.append(m0 == m1)

        correlation = sum(outcomes) / len(outcomes)
        tests['bell_state_correlation'] = correlation
        print(f"  Correlation: {correlation:.2f} (target: 1.0)")
        print(f"  ✅ PASS" if correlation > 0.95 else f"  ❌ FAIL")

        # Test 2: Speed comparison
        print("\n[2.2] Speed test: 50-qubit Clifford circuit...")

        n_qubits = 50
        n_layers = 10

        # Stabilizer time
        start = time.time()
        sim_stab = stab['HybridSimulator'](n_qubits=n_qubits, use_stabilizer=True, device=self.device)
        for layer in range(n_layers):
            for i in range(n_qubits):
                sim_stab.h(i)
            for i in range(n_qubits - 1):
                sim_stab.cnot(i, i + 1)
        stab_time = time.time() - start

        # MPS time (smaller circuit for fairness)
        start = time.time()
        mps = mps_cls['AdaptiveMPS'](20, bond_dim=8, device=self.device)
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex64, device=self.device))

        for layer in range(5):  # Fewer layers
            for i in range(20):
                mps.apply_single_qubit_gate(i, H)
            for i in range(19):
                mps.apply_two_site_gate(i, CZ)
        mps_time = time.time() - start

        tests['stabilizer_time'] = stab_time
        tests['mps_time'] = mps_time
        tests['speedup_estimate'] = mps_time / stab_time if stab_time > 0 else 0

        print(f"  Stabilizer ({n_qubits}q, {n_layers} layers): {stab_time:.3f}s")
        print(f"  MPS (20q, 5 layers): {mps_time:.3f}s")
        print(f"  Estimated speedup: {tests['speedup_estimate']:.1f}× (target: >10×)")
        print(f"  ✅ PASS" if tests['speedup_estimate'] > 10 else f"  ⚠️  SLOWER THAN EXPECTED")

        # Test 3: Handoff mechanism
        print("\n[2.3] Testing Clifford → MPS handoff...")
        hybrid = stab['HybridSimulator'](n_qubits=10, use_stabilizer=True, device=self.device)

        for i in range(10):
            hybrid.h(i)

        stats_before = hybrid.get_statistics()
        hybrid.t(0)  # Trigger handoff
        stats_after = hybrid.get_statistics()

        tests['handoff_successful'] = (stats_before['mode'] == 'stabilizer' and
                                       stats_after['mode'] == 'mps')

        print(f"  Mode before T-gate: {stats_before['mode']}")
        print(f"  Mode after T-gate: {stats_after['mode']}")
        print(f"  ✅ PASS" if tests['handoff_successful'] else f"  ❌ FAIL")

        self.results['stabilizer'] = tests

    def benchmark_mpo_ops(self):
        """Test MPO operations accuracy"""
        print("\n" + "="*70)
        print("BENCHMARK 3: MPO Operations & Observables")
        print("="*70)

        from atlas_q import get_mpo_ops, get_adaptive_mps

        mpo = get_mpo_ops()
        mps_cls = get_adaptive_mps()

        tests = {}

        # Test 1: Ising Hamiltonian energy
        print("\n[3.1] Testing Ising Hamiltonian energy...")
        n_sites = 5
        H = mpo['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.0, device=self.device)

        # Ground state: |00000⟩, E = -J*(n-1) = -4
        mps = mps_cls['AdaptiveMPS'](n_sites, bond_dim=2, device=self.device)
        energy = mpo['expectation_value'](H, mps).real

        exact_energy = -1.0 * (n_sites - 1)
        error = abs(energy - exact_energy)

        tests['ising_energy_error'] = error
        print(f"  Computed energy: {energy:.6f}")
        print(f"  Exact energy: {exact_energy:.6f}")
        print(f"  Error: {error:.2e} (target: <1e-5)")
        print(f"  ✅ PASS" if error < 1e-5 else f"  ❌ FAIL")

        # Test 2: Identity operator
        print("\n[3.2] Testing identity operator...")
        I_mpo = mpo['MPOBuilder'].identity_mpo(n_sites=n_sites, device=self.device)
        norm = mpo['expectation_value'](I_mpo, mps).real

        tests['identity_norm_error'] = abs(norm - 1.0)
        print(f"  ⟨ψ|I|ψ⟩ = {norm:.6f} (target: 1.0)")
        print(f"  ✅ PASS" if tests['identity_norm_error'] < 1e-5 else f"  ❌ FAIL")

        # Test 3: Performance
        print("\n[3.3] Testing MPO expectation value performance...")
        n_trials = 50
        start = time.time()
        for _ in range(n_trials):
            _ = mpo['expectation_value'](H, mps)
        elapsed = time.time() - start

        tests['mpo_evals_per_sec'] = n_trials / elapsed
        print(f"  MPO evaluations/sec: {tests['mpo_evals_per_sec']:.1f}")
        print(f"  ✅ PASS (target: >10/sec)" if tests['mpo_evals_per_sec'] > 10 else f"  ⚠️  SLOW")

        self.results['mpo_ops'] = tests

    def benchmark_tdvp(self):
        """Test TDVP energy conservation and accuracy"""
        print("\n" + "="*70)
        print("BENCHMARK 4: TDVP Time Evolution")
        print("="*70)

        from atlas_q import get_tdvp, get_mpo_ops, get_adaptive_mps

        tdvp_module = get_tdvp()
        mpo_module = get_mpo_ops()
        mps_cls = get_adaptive_mps()

        tests = {}

        # Test 1: Energy conservation (1-site TDVP)
        print("\n[4.1] Testing energy conservation (1-site TDVP)...")
        n_sites = 5
        H = mpo_module['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.5, device=self.device)

        mps = mps_cls['AdaptiveMPS'](n_sites, bond_dim=8, device=self.device)
        H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / np.sqrt(2)
        for q in range(n_sites):
            mps.apply_single_qubit_gate(q, H_gate)

        config = tdvp_module['TDVPConfig'](dt=0.02, t_final=1.0, order=1, chi_max=16)

        start = time.time()
        final_mps, times, energies = tdvp_module['run_tdvp'](H, mps, config)
        tdvp_time = time.time() - start

        energy_drift = abs(energies[-1].real - energies[0].real)
        tests['tdvp_energy_drift'] = energy_drift
        tests['tdvp_runtime'] = tdvp_time

        print(f"  Initial energy: {energies[0].real:.6f}")
        print(f"  Final energy: {energies[-1].real:.6f}")
        print(f"  Energy drift: {energy_drift:.2e} (target: <0.01)")
        print(f"  Runtime: {tdvp_time:.3f}s")
        print(f"  ✅ PASS" if energy_drift < 0.01 else f"  ⚠️  DRIFT")

        # Test 2: 2-site TDVP
        print("\n[4.2] Testing 2-site TDVP...")
        mps2 = mps_cls['AdaptiveMPS'](n_sites, bond_dim=4, device=self.device)
        config2 = tdvp_module['TDVPConfig'](dt=0.02, t_final=0.5, order=2, chi_max=16)

        final_mps2, times2, energies2 = tdvp_module['run_tdvp'](H, mps2, config2)

        energy_drift2 = abs(energies2[-1].real - energies2[0].real)
        tests['tdvp2_energy_drift'] = energy_drift2

        print(f"  Energy drift (2-site): {energy_drift2:.2e}")
        print(f"  ✅ PASS" if energy_drift2 < 0.01 else f"  ⚠️  DRIFT")

        self.results['tdvp'] = tests

    def benchmark_vqe(self):
        """Test VQE convergence to known ground states"""
        print("\n" + "="*70)
        print("BENCHMARK 5: VQE/QAOA Suite")
        print("="*70)

        try:
            from atlas_q import get_vqe_qaoa, get_mpo_ops

            vqe_module = get_vqe_qaoa()
            mpo_module = get_mpo_ops()

            tests = {}

            # Test 1: VQE on simple Ising
            print("\n[5.1] Testing VQE ground state finding...")
            n_sites = 4
            H = mpo_module['MPOBuilder'].ising_hamiltonian(n_sites=n_sites, J=1.0, h=0.0, device=self.device)

            exact_ground = -1.0 * (n_sites - 1)

            config = vqe_module['VQEConfig'](
                n_layers=2,
                optimizer='COBYLA',
                max_iter=50,
                device=self.device,
                chi_max=16
            )

            start = time.time()
            vqe = vqe_module['VQE'](H, config)
            energy, params = vqe.run()
            vqe_time = time.time() - start

            error = abs(energy - exact_ground)
            tests['vqe_energy_error'] = error
            tests['vqe_runtime'] = vqe_time

            print(f"  VQE energy: {energy:.6f}")
            print(f"  Exact ground: {exact_ground:.6f}")
            print(f"  Error: {error:.6f} (target: <0.5)")
            print(f"  Runtime: {vqe_time:.3f}s")
            print(f"  ✅ PASS" if error < 0.5 else f"  ⚠️  ACCURACY")

            # Test 2: QAOA
            print("\n[5.2] Testing QAOA optimization...")
            H_cost = mpo_module['MPOBuilder'].ising_hamiltonian(n_sites=4, J=-1.0, h=0.0, device=self.device)

            qaoa = vqe_module['QAOA'](H_cost, n_layers=2, device=self.device)

            start = time.time()
            cost, params = qaoa.run()
            qaoa_time = time.time() - start

            tests['qaoa_cost'] = cost
            tests['qaoa_runtime'] = qaoa_time

            print(f"  QAOA cost: {cost:.6f}")
            print(f"  Runtime: {qaoa_time:.3f}s")
            print(f"  ✅ COMPLETE")

            self.results['vqe_qaoa'] = tests

        except ImportError:
            print("\n⚠️  SciPy not available - skipping VQE/QAOA benchmarks")
            self.results['vqe_qaoa'] = {'skipped': True, 'reason': 'scipy not installed'}

    def benchmark_2d_circuits(self):
        """Test 2D circuit compilation overhead"""
        print("\n" + "="*70)
        print("BENCHMARK 6: 2D/Planar Circuits")
        print("="*70)

        from atlas_q import get_planar_2d

        planar = get_planar_2d()

        tests = {}

        # Test 1: Snake mapping correctness
        print("\n[6.1] Testing snake mapping...")
        mapper = planar['SnakeMapper'](rows=4, cols=4)

        # Check mapping consistency
        errors = 0
        for r in range(4):
            for c in range(4):
                idx = mapper.map_2d_to_1d(r, c)
                r_back, c_back = mapper.map_1d_to_2d(idx)
                if r_back != r or c_back != c:
                    errors += 1

        tests['mapping_errors'] = errors
        print(f"  Mapping errors: {errors} (target: 0)")
        print(f"  ✅ PASS" if errors == 0 else f"  ❌ FAIL")

        # Test 2: SWAP overhead
        print("\n[6.2] Testing SWAP synthesis overhead...")
        circuit_2d = planar['Planar2DCircuit'](rows=4, cols=4)

        gates_2d = [
            ('H', [(i // 4, i % 4)], []) for i in range(16)
        ] + [
            ('CNOT', [(0, 0), (3, 3)], []),  # Long-range
            ('CNOT', [(0, 3), (3, 0)], []),  # Long-range
        ]

        gates_1d = circuit_2d.compile_circuit(gates_2d)

        swap_count = sum(1 for g, _, _ in gates_1d if g == 'SWAP')
        overhead = len(gates_1d) / len(gates_2d)

        tests['swap_count'] = swap_count
        tests['overhead_factor'] = overhead

        print(f"  Original gates: {len(gates_2d)}")
        print(f"  With SWAPs: {len(gates_1d)}")
        print(f"  SWAPs inserted: {swap_count}")
        print(f"  Overhead factor: {overhead:.2f}×")
        print(f"  ✅ ACCEPTABLE" if overhead < 5.0 else f"  ⚠️  HIGH OVERHEAD")

        self.results['2d_circuits'] = tests

    def benchmark_integration(self):
        """Integration tests combining multiple features"""
        print("\n" + "="*70)
        print("BENCHMARK 7: Integration Tests")
        print("="*70)

        from atlas_q import (
            get_noise_models, get_stabilizer, get_mpo_ops, get_adaptive_mps
        )

        noise = get_noise_models()
        stab = get_stabilizer()
        mpo = get_mpo_ops()
        mps_cls = get_adaptive_mps()

        tests = {}

        # Test 1: Noisy + Hybrid + Measurement
        print("\n[7.1] Integration: Noisy hybrid simulation...")

        noise_model = noise['NoiseModel'].depolarizing(p1q=0.01, device=self.device)
        applicator = noise['StochasticNoiseApplicator'](noise_model, seed=42)

        hybrid = stab['HybridSimulator'](n_qubits=10, use_stabilizer=True, device=self.device)

        # Clifford part
        for i in range(10):
            hybrid.h(i)
        for i in range(9):
            hybrid.cnot(i, i + 1)

        # Non-Clifford triggers handoff
        hybrid.t(0)

        # Check mode switched
        stats = hybrid.get_statistics()
        tests['integration_handoff'] = (stats['mode'] == 'mps')

        print(f"  Mode after mixed circuit: {stats['mode']}")
        print(f"  Clifford gates: {stats['clifford_gates']}")
        print(f"  Non-Clifford gates: {stats['non_clifford_gates']}")
        print(f"  ✅ PASS" if tests['integration_handoff'] else f"  ❌ FAIL")

        # Test 2: Full workflow
        print("\n[7.2] Integration: Full workflow test...")
        success = True
        try:
            # MPS creation
            mps = mps_cls['AdaptiveMPS'](8, bond_dim=8, device=self.device)

            # MPO measurement
            H = mpo['MPOBuilder'].ising_hamiltonian(n_sites=8, J=1.0, h=0.5, device=self.device)
            energy = mpo['expectation_value'](H, mps)

            # Noise application
            H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / np.sqrt(2)
            for q in range(8):
                mps.apply_single_qubit_gate(q, H_gate)
                applicator.apply_1q_noise(mps, q)

            fidelity = applicator.get_fidelity_estimate()

            print(f"  Energy: {energy.real:.6f}")
            print(f"  Fidelity: {fidelity:.4f}")
            print(f"  ✅ PASS")

        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            success = False

        tests['full_workflow'] = success

        self.results['integration'] = tests

    def print_summary(self):
        """Print final benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        total_tests = 0
        passed_tests = 0

        for category, tests in self.results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")

            if isinstance(tests, dict) and 'skipped' in tests:
                print(f"  ⚠️  SKIPPED: {tests['reason']}")
                continue

            for test_name, value in tests.items():
                total_tests += 1

                if isinstance(value, bool):
                    status = "✅ PASS" if value else "❌ FAIL"
                    if value:
                        passed_tests += 1
                elif isinstance(value, (int, float)):
                    status = f"{value:.4g}"
                else:
                    status = str(value)

                print(f"  {test_name}: {status}")

        print("\n" + "="*70)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed ({100*passed_tests/total_tests:.1f}%)")
        print("="*70)

        # Save results
        import json
        with open('benchmark_results.json', 'w') as f:
            # Convert to JSON-serializable format
            results_clean = {}
            for k, v in self.results.items():
                if isinstance(v, dict):
                    results_clean[k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, torch.Tensor)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    results_clean[k] = v

            json.dump(results_clean, f, indent=2)

        print(f"\nResults saved to: benchmark_results.json\n")


def main():
    """Run comprehensive benchmarks"""
    runner = BenchmarkRunner(device='cuda')
    runner.run_all_benchmarks()


if __name__ == '__main__':
    main()
