"""
IBM Quantum Hardware Backend Adapter

Provides integration with IBM Quantum hardware through:
- Job submission to IBM Quantum systems
- Backend selection and properties retrieval
- Circuit transpilation for IBM native gates
- Error mitigation options
- Result post-processing

Requires: qiskit-ibm-runtime or qiskit-ibm-provider

Author: ATLAS-Q Contributors
Date: December 2025
License: MIT
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Check for IBM Quantum dependencies
IBM_RUNTIME_AVAILABLE = False
IBM_PROVIDER_AVAILABLE = False
QISKIT_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    transpile = None

try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Sampler,
        Estimator,
        Session,
        Options,
    )
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    QiskitRuntimeService = None
    Sampler = None
    Estimator = None

try:
    from qiskit_ibm_provider import IBMProvider
    IBM_PROVIDER_AVAILABLE = True
except ImportError:
    IBMProvider = None


class BackendType(Enum):
    """IBM Quantum backend types"""
    SIMULATOR = auto()
    REAL_HARDWARE = auto()
    FAKE_BACKEND = auto()


@dataclass
class BackendProperties:
    """Properties of an IBM Quantum backend"""
    name: str
    backend_type: BackendType
    n_qubits: int
    basis_gates: List[str]
    coupling_map: List[Tuple[int, int]]
    t1_times: Dict[int, float]  # Qubit -> T1 in microseconds
    t2_times: Dict[int, float]  # Qubit -> T2 in microseconds
    gate_errors: Dict[str, Dict[Tuple[int, ...], float]]  # Gate -> (qubits) -> error
    readout_errors: Dict[int, float]  # Qubit -> readout error
    max_shots: int = 100000
    max_circuits: int = 300
    pending_jobs: int = 0
    status: str = "online"
    last_calibration: Optional[datetime] = None


@dataclass
class JobResult:
    """Result from an IBM Quantum job"""
    job_id: str
    backend_name: str
    counts: Dict[str, int]
    shots: int
    success: bool
    execution_time: Optional[float] = None
    raw_data: Optional[Dict] = None
    error_message: Optional[str] = None


@dataclass
class IBMQuantumConfig:
    """Configuration for IBM Quantum connection"""
    # Authentication
    token: Optional[str] = None
    channel: str = "ibm_quantum"  # "ibm_quantum" or "ibm_cloud"
    instance: Optional[str] = None  # hub/group/project

    # Execution options
    shots: int = 4096
    optimization_level: int = 1
    resilience_level: int = 0  # 0-2 for error mitigation

    # Job options
    max_execution_time: Optional[int] = None
    session_mode: bool = False

    # Backend preferences
    preferred_backend: Optional[str] = None
    min_qubits: int = 1
    simulator_fallback: bool = True


class IBMQuantumBackend:
    """
    IBM Quantum hardware backend adapter for ATLAS-Q.

    Provides:
    - Connection to IBM Quantum services
    - Backend discovery and selection
    - Circuit submission and result retrieval
    - Automatic transpilation for hardware
    - Error mitigation options

    Example:
        >>> config = IBMQuantumConfig(token="YOUR_TOKEN")
        >>> backend = IBMQuantumBackend(config)
        >>> backends = backend.list_backends()
        >>> backend.select_backend("ibm_brisbane")
        >>> result = backend.run_circuit(circuit, shots=1000)
    """

    def __init__(self, config: Optional[IBMQuantumConfig] = None):
        """
        Initialize IBM Quantum backend.

        Args:
            config: Configuration options (token, shots, etc.)
        """
        self.config = config or IBMQuantumConfig()
        self.service = None
        self.provider = None
        self.backend = None
        self.session = None
        self._connected = False

        # Check dependencies
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for IBM Quantum integration. "
                "Install with: pip install qiskit"
            )

        if not (IBM_RUNTIME_AVAILABLE or IBM_PROVIDER_AVAILABLE):
            raise ImportError(
                "IBM Quantum provider not found. Install with: "
                "pip install qiskit-ibm-runtime"
            )

    def connect(self, token: Optional[str] = None) -> bool:
        """
        Connect to IBM Quantum services.

        Args:
            token: IBM Quantum API token (or use saved credentials)

        Returns:
            True if connection successful
        """
        token = token or self.config.token

        try:
            if IBM_RUNTIME_AVAILABLE:
                # Use qiskit-ibm-runtime (preferred)
                if token:
                    self.service = QiskitRuntimeService(
                        channel=self.config.channel,
                        token=token,
                        instance=self.config.instance
                    )
                else:
                    # Try saved credentials
                    self.service = QiskitRuntimeService(
                        channel=self.config.channel,
                        instance=self.config.instance
                    )
                self._connected = True
                return True

            elif IBM_PROVIDER_AVAILABLE:
                # Fallback to qiskit-ibm-provider
                if token:
                    self.provider = IBMProvider(token=token)
                else:
                    self.provider = IBMProvider()
                self._connected = True
                return True

        except Exception as e:
            warnings.warn(f"Failed to connect to IBM Quantum: {e}")
            return False

    def list_backends(
        self,
        min_qubits: Optional[int] = None,
        simulator: bool = False,
        operational: bool = True
    ) -> List[str]:
        """
        List available IBM Quantum backends.

        Args:
            min_qubits: Minimum number of qubits required
            simulator: Include simulators
            operational: Only include operational backends

        Returns:
            List of backend names
        """
        if not self._connected:
            self.connect()

        min_qubits = min_qubits or self.config.min_qubits

        try:
            if self.service:
                backends = self.service.backends(
                    min_num_qubits=min_qubits,
                    simulator=simulator,
                    operational=operational
                )
                return [b.name for b in backends]

            elif self.provider:
                backends = self.provider.backends(
                    min_num_qubits=min_qubits,
                    simulator=simulator,
                    operational=operational
                )
                return [b.name for b in backends]

        except Exception as e:
            warnings.warn(f"Failed to list backends: {e}")
            return []

        return []

    def get_backend_properties(self, backend_name: str) -> Optional[BackendProperties]:
        """
        Get detailed properties of a backend.

        Args:
            backend_name: Name of the backend

        Returns:
            BackendProperties object or None
        """
        if not self._connected:
            self.connect()

        try:
            if self.service:
                backend = self.service.backend(backend_name)
            elif self.provider:
                backend = self.provider.get_backend(backend_name)
            else:
                return None

            # Extract properties
            config = backend.configuration()
            props = backend.properties() if hasattr(backend, 'properties') else None

            # Determine backend type
            if config.simulator:
                backend_type = BackendType.SIMULATOR
            else:
                backend_type = BackendType.REAL_HARDWARE

            # Get coupling map
            coupling_map = []
            if hasattr(config, 'coupling_map') and config.coupling_map:
                coupling_map = [tuple(c) for c in config.coupling_map]

            # Get T1/T2 times and errors
            t1_times = {}
            t2_times = {}
            readout_errors = {}
            gate_errors = {}

            if props:
                for qubit in range(config.n_qubits):
                    try:
                        t1_times[qubit] = props.t1(qubit) * 1e6  # Convert to µs
                        t2_times[qubit] = props.t2(qubit) * 1e6
                        readout_errors[qubit] = props.readout_error(qubit)
                    except Exception:
                        pass

                # Get gate errors
                for gate in config.basis_gates:
                    gate_errors[gate] = {}
                    try:
                        for qubit in range(config.n_qubits):
                            if gate in ['cx', 'ecr', 'cz']:
                                # Two-qubit gate
                                for neighbor in coupling_map:
                                    if neighbor[0] == qubit:
                                        try:
                                            err = props.gate_error(gate, neighbor)
                                            gate_errors[gate][tuple(neighbor)] = err
                                        except Exception:
                                            pass
                            else:
                                # Single-qubit gate
                                try:
                                    err = props.gate_error(gate, qubit)
                                    gate_errors[gate][(qubit,)] = err
                                except Exception:
                                    pass
                    except Exception:
                        pass

            return BackendProperties(
                name=backend_name,
                backend_type=backend_type,
                n_qubits=config.n_qubits,
                basis_gates=config.basis_gates,
                coupling_map=coupling_map,
                t1_times=t1_times,
                t2_times=t2_times,
                gate_errors=gate_errors,
                readout_errors=readout_errors,
                max_shots=config.max_shots if hasattr(config, 'max_shots') else 100000,
                max_circuits=config.max_experiments if hasattr(config, 'max_experiments') else 300,
                status=backend.status().status_msg if hasattr(backend, 'status') else "unknown"
            )

        except Exception as e:
            warnings.warn(f"Failed to get backend properties: {e}")
            return None

    def select_backend(self, backend_name: str) -> bool:
        """
        Select a backend for execution.

        Args:
            backend_name: Name of the backend

        Returns:
            True if selection successful
        """
        if not self._connected:
            self.connect()

        try:
            if self.service:
                self.backend = self.service.backend(backend_name)
            elif self.provider:
                self.backend = self.provider.get_backend(backend_name)
            return self.backend is not None

        except Exception as e:
            warnings.warn(f"Failed to select backend: {e}")
            return False

    def select_least_busy(
        self,
        min_qubits: int = 1,
        simulator: bool = False
    ) -> Optional[str]:
        """
        Select the least busy operational backend.

        Args:
            min_qubits: Minimum number of qubits
            simulator: Include simulators

        Returns:
            Selected backend name or None
        """
        if not self._connected:
            self.connect()

        try:
            if self.service:
                backend = self.service.least_busy(
                    min_num_qubits=min_qubits,
                    simulator=simulator,
                    operational=True
                )
                self.backend = backend
                return backend.name

            elif self.provider:
                from qiskit_ibm_provider import least_busy
                backends = self.provider.backends(
                    min_num_qubits=min_qubits,
                    simulator=simulator,
                    operational=True
                )
                if backends:
                    backend = least_busy(backends)
                    self.backend = backend
                    return backend.name

        except Exception as e:
            warnings.warn(f"Failed to find least busy backend: {e}")

        return None

    def _create_qiskit_circuit(
        self,
        gates: List[Tuple[str, List[int], List[float]]],
        n_qubits: int
    ) -> "QuantumCircuit":
        """
        Convert ATLAS-Q gate list to Qiskit QuantumCircuit.

        Args:
            gates: List of (gate_name, qubits, params)
            n_qubits: Number of qubits

        Returns:
            Qiskit QuantumCircuit
        """
        qc = QuantumCircuit(n_qubits, n_qubits)

        for gate_name, qubits, params in gates:
            gate_name = gate_name.lower()

            if gate_name in ['h', 'hadamard']:
                qc.h(qubits[0])
            elif gate_name == 'x':
                qc.x(qubits[0])
            elif gate_name == 'y':
                qc.y(qubits[0])
            elif gate_name == 'z':
                qc.z(qubits[0])
            elif gate_name == 's':
                qc.s(qubits[0])
            elif gate_name == 't':
                qc.t(qubits[0])
            elif gate_name == 'rx':
                qc.rx(params[0], qubits[0])
            elif gate_name == 'ry':
                qc.ry(params[0], qubits[0])
            elif gate_name == 'rz':
                qc.rz(params[0], qubits[0])
            elif gate_name in ['cnot', 'cx']:
                qc.cx(qubits[0], qubits[1])
            elif gate_name == 'cz':
                qc.cz(qubits[0], qubits[1])
            elif gate_name == 'swap':
                qc.swap(qubits[0], qubits[1])
            elif gate_name in ['ccx', 'toffoli']:
                qc.ccx(qubits[0], qubits[1], qubits[2])
            elif gate_name == 'u':
                qc.u(params[0], params[1], params[2], qubits[0])
            else:
                warnings.warn(f"Unknown gate: {gate_name}")

        # Add measurements
        qc.measure_all()

        return qc

    def run_circuit(
        self,
        circuit: Union["QuantumCircuit", List[Tuple[str, List[int], List[float]]]],
        n_qubits: Optional[int] = None,
        shots: Optional[int] = None,
        wait_for_result: bool = True
    ) -> JobResult:
        """
        Run a circuit on IBM Quantum hardware.

        Args:
            circuit: Qiskit QuantumCircuit or ATLAS-Q gate list
            n_qubits: Number of qubits (required if using gate list)
            shots: Number of shots
            wait_for_result: If True, wait for job completion

        Returns:
            JobResult with counts and metadata
        """
        if not self.backend:
            if self.config.preferred_backend:
                self.select_backend(self.config.preferred_backend)
            else:
                self.select_least_busy(min_qubits=n_qubits or 1)

        if not self.backend:
            return JobResult(
                job_id="",
                backend_name="none",
                counts={},
                shots=0,
                success=False,
                error_message="No backend available"
            )

        shots = shots or self.config.shots

        # Convert to Qiskit circuit if needed
        if isinstance(circuit, list):
            if n_qubits is None:
                # Infer from gate list
                n_qubits = max(max(g[1]) for g in circuit) + 1 if circuit else 1
            qc = self._create_qiskit_circuit(circuit, n_qubits)
        else:
            qc = circuit

        # Transpile for target backend
        transpiled = transpile(
            qc,
            backend=self.backend,
            optimization_level=self.config.optimization_level
        )

        try:
            if IBM_RUNTIME_AVAILABLE and self.service:
                # Use Sampler primitive
                options = Options()
                options.execution.shots = shots
                options.resilience_level = self.config.resilience_level

                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session, options=options)
                    job = sampler.run([transpiled])

                    if wait_for_result:
                        result = job.result()
                        # Extract counts from quasi-distribution
                        quasi_dist = result.quasi_dists[0]
                        counts = {}
                        n_bits = transpiled.num_clbits
                        for outcome, prob in quasi_dist.items():
                            bitstring = format(outcome, f'0{n_bits}b')
                            counts[bitstring] = int(prob * shots)

                        return JobResult(
                            job_id=job.job_id(),
                            backend_name=self.backend.name,
                            counts=counts,
                            shots=shots,
                            success=True
                        )
                    else:
                        return JobResult(
                            job_id=job.job_id(),
                            backend_name=self.backend.name,
                            counts={},
                            shots=shots,
                            success=True
                        )

            else:
                # Fallback to direct backend execution
                job = self.backend.run(transpiled, shots=shots)

                if wait_for_result:
                    result = job.result()
                    counts = result.get_counts()

                    return JobResult(
                        job_id=job.job_id(),
                        backend_name=self.backend.name,
                        counts=counts,
                        shots=shots,
                        success=True,
                        execution_time=result.time_taken if hasattr(result, 'time_taken') else None
                    )
                else:
                    return JobResult(
                        job_id=job.job_id(),
                        backend_name=self.backend.name,
                        counts={},
                        shots=shots,
                        success=True
                    )

        except Exception as e:
            return JobResult(
                job_id="",
                backend_name=self.backend.name if self.backend else "none",
                counts={},
                shots=shots,
                success=False,
                error_message=str(e)
            )

    def get_job_result(self, job_id: str) -> Optional[JobResult]:
        """
        Retrieve results for a previously submitted job.

        Args:
            job_id: Job ID from run_circuit

        Returns:
            JobResult or None if not found
        """
        try:
            if self.service:
                job = self.service.job(job_id)
            elif self.provider:
                job = self.provider.retrieve_job(job_id)
            else:
                return None

            result = job.result()
            counts = result.get_counts() if hasattr(result, 'get_counts') else {}

            return JobResult(
                job_id=job_id,
                backend_name=job.backend().name if hasattr(job, 'backend') else "unknown",
                counts=counts,
                shots=sum(counts.values()) if counts else 0,
                success=True
            )

        except Exception as e:
            return JobResult(
                job_id=job_id,
                backend_name="unknown",
                counts={},
                shots=0,
                success=False,
                error_message=str(e)
            )

    def estimate_cost(
        self,
        circuit: "QuantumCircuit",
        shots: int = 4096
    ) -> Dict[str, Any]:
        """
        Estimate execution time and queue position.

        Args:
            circuit: Circuit to estimate
            shots: Number of shots

        Returns:
            Dictionary with estimates
        """
        if not self.backend:
            return {"error": "No backend selected"}

        try:
            status = self.backend.status()

            return {
                "backend": self.backend.name,
                "shots": shots,
                "pending_jobs": status.pending_jobs,
                "operational": status.operational,
                "circuit_depth": circuit.depth(),
                "circuit_gates": sum(circuit.count_ops().values()),
                "estimated_queue_time_minutes": status.pending_jobs * 2  # Rough estimate
            }

        except Exception as e:
            return {"error": str(e)}


class IBMQuantumSimulator:
    """
    IBM Quantum simulator backend (local or cloud).

    For testing circuits before running on real hardware.
    """

    def __init__(self, use_cloud: bool = False, config: Optional[IBMQuantumConfig] = None):
        """
        Initialize simulator.

        Args:
            use_cloud: If True, use IBM's cloud simulator
            config: Configuration options
        """
        self.config = config or IBMQuantumConfig()
        self.use_cloud = use_cloud

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required: pip install qiskit")

        # Try to import Aer for local simulation
        self.aer_available = False
        try:
            from qiskit_aer import AerSimulator
            self.simulator = AerSimulator()
            self.aer_available = True
        except ImportError:
            if not use_cloud:
                warnings.warn(
                    "qiskit-aer not available. Install with: pip install qiskit-aer"
                )

    def run(
        self,
        circuit: "QuantumCircuit",
        shots: int = 4096,
        noise_model: Optional[Any] = None
    ) -> JobResult:
        """
        Run circuit on simulator.

        Args:
            circuit: Qiskit circuit
            shots: Number of shots
            noise_model: Optional noise model

        Returns:
            JobResult with counts
        """
        if not self.aer_available and not self.use_cloud:
            return JobResult(
                job_id="local",
                backend_name="unavailable",
                counts={},
                shots=0,
                success=False,
                error_message="No simulator available"
            )

        try:
            if self.aer_available and not self.use_cloud:
                job = self.simulator.run(
                    circuit,
                    shots=shots,
                    noise_model=noise_model
                )
                result = job.result()
                counts = result.get_counts()

                return JobResult(
                    job_id="local_sim",
                    backend_name="aer_simulator",
                    counts=counts,
                    shots=shots,
                    success=True
                )

        except Exception as e:
            return JobResult(
                job_id="",
                backend_name="simulator",
                counts={},
                shots=0,
                success=False,
                error_message=str(e)
            )


def get_ibm_quantum():
    """Get IBM Quantum backend classes"""
    return {
        'IBMQuantumBackend': IBMQuantumBackend,
        'IBMQuantumSimulator': IBMQuantumSimulator,
        'IBMQuantumConfig': IBMQuantumConfig,
        'BackendProperties': BackendProperties,
        'BackendType': BackendType,
        'JobResult': JobResult,
        'IBM_RUNTIME_AVAILABLE': IBM_RUNTIME_AVAILABLE,
        'IBM_PROVIDER_AVAILABLE': IBM_PROVIDER_AVAILABLE,
        'QISKIT_AVAILABLE': QISKIT_AVAILABLE,
    }


if __name__ == "__main__":
    print("IBM Quantum Backend Demo")
    print("=" * 50)

    print(f"\nDependency status:")
    print(f"  Qiskit available: {QISKIT_AVAILABLE}")
    print(f"  IBM Runtime available: {IBM_RUNTIME_AVAILABLE}")
    print(f"  IBM Provider available: {IBM_PROVIDER_AVAILABLE}")

    if QISKIT_AVAILABLE:
        # Create a simple circuit
        print("\nCreating test circuit...")
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        print(qc)

        # Try local simulator
        print("\nTrying local simulator...")
        try:
            sim = IBMQuantumSimulator(use_cloud=False)
            result = sim.run(qc, shots=1000)
            if result.success:
                print(f"Results: {result.counts}")
            else:
                print(f"Simulation failed: {result.error_message}")
        except ImportError as e:
            print(f"Simulator not available: {e}")

        # Show how to connect to IBM Quantum
        print("\nTo connect to IBM Quantum hardware:")
        print("  1. Get token from https://quantum.ibm.com/")
        print("  2. config = IBMQuantumConfig(token='YOUR_TOKEN')")
        print("  3. backend = IBMQuantumBackend(config)")
        print("  4. backend.connect()")
        print("  5. backends = backend.list_backends()")
    else:
        print("\nInstall Qiskit to enable IBM Quantum integration:")
        print("  pip install qiskit qiskit-ibm-runtime")
