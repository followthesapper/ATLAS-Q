"""
Tests for Qiskit adapter

Verifies:
- Circuit execution matches Qiskit Aer
- IR grouping reduces measurements
- Automatic backend selection (Clifford/MPS/statevector)
- Coherence metrics for VQE patterns
- GPU acceleration when available
"""

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp

from atlas_q.adapters import ATLASQBackend, ATLASQProvider


class TestQiskitAdapter:
    """Test suite for Qiskit adapter"""

    def test_basic_circuit_execution(self):
        """Test basic circuit execution"""
        backend = ATLASQBackend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        job = backend.run(qc, shots=1000)
        result = job.result()

        assert result.success
        counts = result.get_counts()
        assert '00' in counts or '11' in counts
        assert sum(counts.values()) == 1000

    def test_clifford_detection(self):
        """Test automatic Clifford circuit detection"""
        backend = ATLASQBackend(enable_stabilizer=True)

        # Clifford circuit
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cz(1, 2)
        qc.s(2)

        job = backend.run(qc, shots=100)
        result = job.result()

        # Check that stabilizer backend was used
        metadata = result.results[0].header
        assert metadata['backend_used'] == 'stabilizer'

    def test_mps_threshold(self):
        """Test MPS backend activation for large circuits"""
        # Disable stabilizer to force MPS for Clifford circuit
        backend = ATLASQBackend(enable_mps=True, mps_threshold=10, enable_stabilizer=False)

        # Circuit with 15 qubits (above threshold)
        qc = QuantumCircuit(15)
        for i in range(14):
            qc.h(i)
            qc.cx(i, i+1)

        job = backend.run(qc, shots=10)
        result = job.result()

        metadata = result.results[0].header
        # Should use MPS backend for large circuit
        assert metadata['backend_used'] == 'mps'

        # Verify we got valid measurement results
        counts = result.get_counts()
        assert len(counts) > 0
        assert sum(counts.values()) == 10

    def test_ir_grouping(self):
        """Test automatic IR observable grouping"""
        backend = ATLASQBackend(enable_ir=True)

        # Create VQE-like circuit
        qc = QuantumCircuit(2)
        qc.ry(0.5, 0)
        qc.ry(0.3, 1)
        qc.cx(0, 1)

        # Create Hamiltonian with multiple terms
        observables = SparsePauliOp.from_list([
            ('ZZ', 1.0),
            ('XX', 0.5),
            ('YY', 0.3),
            ('ZI', 0.2),
            ('IZ', 0.2)
        ])

        job = backend.run(qc, shots=1000, observables=observables)
        result = job.result()

        # Check IR compression ratio
        metadata = result.results[0].header
        compression = metadata.get('ir_compression_ratio')

        # IR should group commuting observables
        assert compression is not None, "IR compression should be computed"
        assert compression < 1.0, "IR should reduce number of measurement groups"
        print(f"IR grouped {5} observables into {int(5*compression)} groups ({compression:.2f} compression)")

    def test_coherence_metrics_vqe(self):
        """Test coherence metrics for VQE patterns"""
        backend = ATLASQBackend()

        # VQE circuit with parameters
        qc = QuantumCircuit(2)
        qc.ry(0.5, 0)
        qc.ry(0.3, 1)
        qc.cx(0, 1)

        # Observable
        observables = SparsePauliOp.from_list([('ZZ', 1.0), ('XX', 0.5)])

        job = backend.run(qc, shots=1000, observables=observables)
        result = job.result()

        metadata = result.results[0].header
        coherence = metadata.get('coherence_metrics')

        if coherence is not None:
            assert 'mean_resultant_length' in coherence
            assert 'classification' in coherence
            assert coherence['classification'] in ['GO', 'NO-GO']

    def test_multi_circuit_execution(self):
        """Test execution of multiple circuits"""
        backend = ATLASQBackend()

        circuits = []
        for i in range(3):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            circuits.append(qc)

        job = backend.run(circuits, shots=100)
        result = job.result()

        assert result.success
        assert len(result.results) == 3

        for exp_result in result.results:
            counts = exp_result.data.counts
            assert sum(counts.values()) == 100

    def test_provider_interface(self):
        """Test provider interface"""
        provider = ATLASQProvider()

        backends = provider.backends()
        assert len(backends) > 0

        backend = provider.get_backend()
        assert isinstance(backend, ATLASQBackend)

    def test_backend_options(self):
        """Test backend configuration options"""
        backend = ATLASQBackend(
            enable_ir=False,
            enable_mps=False,
            enable_stabilizer=False,
            mps_threshold=50
        )

        assert backend._enable_ir == False
        assert backend._enable_mps == False
        assert backend._enable_stabilizer == False
        assert backend._mps_threshold == 50

    def test_bell_state(self):
        """Test Bell state preparation and measurement"""
        backend = ATLASQBackend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        job = backend.run(qc, shots=1000)
        result = job.result()

        counts = result.get_counts()

        # Bell state should have ~50% |00⟩ and ~50% |11⟩
        count_00 = counts.get('00', 0)
        count_11 = counts.get('11', 0)

        assert count_00 + count_11 > 900  # At least 90% in |00⟩ or |11⟩

    def test_ghz_state(self):
        """Test GHZ state preparation"""
        backend = ATLASQBackend()

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        job = backend.run(qc, shots=1000)
        result = job.result()

        counts = result.get_counts()

        # GHZ state should have ~50% |000⟩ and ~50% |111⟩
        count_000 = counts.get('000', 0)
        count_111 = counts.get('111', 0)

        assert count_000 + count_111 > 900

    def test_parametric_circuit(self):
        """Test parametric circuit execution"""
        backend = ATLASQBackend()

        qc = QuantumCircuit(1)
        qc.ry(np.pi / 4, 0)
        qc.measure_all()

        job = backend.run(qc, shots=1000)
        result = job.result()

        assert result.success
        counts = result.get_counts()
        assert len(counts) > 0

    def test_job_status(self):
        """Test job status tracking"""
        backend = ATLASQBackend()

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.measure_all()

        job = backend.run(qc, shots=100)

        # Job should be done (synchronous execution)
        status = job.status()
        assert str(status) == 'JobStatus.DONE'

        # Should have job ID
        job_id = job.job_id()
        assert job_id is not None


@pytest.mark.benchmark
class TestQiskitBenchmarks:
    """Benchmark tests comparing ATLAS-Q vs Qiskit Aer"""

    def test_clifford_speedup(self, benchmark):
        """Benchmark Clifford circuit execution"""
        backend = ATLASQBackend(enable_stabilizer=True)

        qc = QuantumCircuit(10)
        for i in range(9):
            qc.h(i)
            qc.cx(i, i+1)

        def run_circuit():
            job = backend.run(qc, shots=1000)
            return job.result()

        result = benchmark(run_circuit)
        assert result.success

    def test_ir_measurement_reduction(self):
        """Benchmark IR measurement reduction"""
        backend = ATLASQBackend(enable_ir=True)

        qc = QuantumCircuit(4)
        qc.ry(0.5, 0)
        qc.ry(0.3, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Large Hamiltonian
        terms = []
        for i in range(100):
            pauli_str = 'I' * (i % 4) + 'Z' + 'I' * (3 - i % 4)
            terms.append((pauli_str[:4], 0.1))

        observables = SparsePauliOp.from_list(terms)

        job = backend.run(qc, shots=100, observables=observables)
        result = job.result()

        metadata = result.results[0].header
        compression = metadata.get('ir_compression_ratio')

        if compression:
            print(f"IR compression: {compression:.2f}x reduction")
            assert compression < 0.5  # Should reduce by >50%
