"""
GPU Statevector Backend (Direct CUDA Driver API)

Uses pre-compiled PTX files and CUDA Driver API directly via ctypes.
Version-independent - works with any CUDA runtime version.
"""

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# Find CUDA driver library
def find_cuda_library():
    """Find libcuda.so on the system"""
    # Common locations
    locations = [
        '/usr/lib/aarch64-linux-gnu/libcuda.so',  # ARM
        '/usr/local/cuda/lib64/libcuda.so',  # x86_64
        '/usr/lib/x86_64-linux-gnu/libcuda.so',  # x86_64
        '/usr/lib64/libcuda.so',
    ]

    for loc in locations:
        if os.path.exists(loc):
            return loc

    # Try ctypes.util
    lib = ctypes.util.find_library('cuda')
    if lib:
        return lib

    raise RuntimeError("Could not find CUDA driver library (libcuda.so)")


class GPUStatevectorSimulator:
    """
    GPU-accelerated statevector simulator using CUDA Driver API directly.

    No compile-time CUDA version dependency - loads pre-compiled PTX at runtime.
    """

    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if n_qubits > 28:
            raise ValueError("n_qubits > 28 exceeds GPU memory")

        self.n_qubits = n_qubits
        self.state_size = 1 << n_qubits

        # Load CUDA driver
        cuda_lib = find_cuda_library()
        self.cuda = ctypes.CDLL(cuda_lib)

        # Initialize CUDA
        self._check(self.cuda.cuInit(0))

        # Get device
        self.device = ctypes.c_int()
        self._check(self.cuda.cuDeviceGet(ctypes.byref(self.device), 0))

        # Create context
        self.context = ctypes.c_void_p()
        self._check(self.cuda.cuCtxCreate_v2(
            ctypes.byref(self.context),
            0,
            self.device
        ))

        # Allocate device memory (interleaved real/imag)
        buffer_size = self.state_size * 2 * 8  # 2 f64s per amplitude
        self.d_state = ctypes.c_void_p()
        self._check(self.cuda.cuMemAlloc_v2(
            ctypes.byref(self.d_state),
            buffer_size
        ))

        # Initialize to |0...0⟩
        init_state = np.zeros(self.state_size * 2, dtype=np.float64)
        init_state[0] = 1.0  # First amplitude = 1+0i
        self._check(self.cuda.cuMemcpyHtoD_v2(
            self.d_state,
            init_state.ctypes.data,
            buffer_size
        ))

        # Load PTX modules
        self.modules = {}
        self.functions = {}
        self._load_kernels()

    def _check(self, result):
        """Check CUDA call result"""
        if result != 0:
            raise RuntimeError(f"CUDA error: {result}")

    def _load_kernels(self):
        """Load pre-compiled PTX kernels"""
        # Find PTX files
        kernel_dir = Path(__file__).parent.parent.parent / 'atlas_q_core' / 'cuda_kernels'

        # Load single-qubit kernels
        single_ptx = kernel_dir / 'single_qubit_gates_f64.ptx'
        if not single_ptx.exists():
            raise FileNotFoundError(f"PTX file not found: {single_ptx}")

        module1 = ctypes.c_void_p()
        self._check(self.cuda.cuModuleLoad(
            ctypes.byref(module1),
            str(single_ptx).encode()
        ))
        self.modules['single'] = module1

        # Load kernel functions
        for name in ['apply_hadamard', 'apply_pauli_x', 'apply_pauli_z', 'apply_rx', 'apply_ry']:
            func = ctypes.c_void_p()
            self._check(self.cuda.cuModuleGetFunction(
                ctypes.byref(func),
                module1,
                name.encode()
            ))
            self.functions[name] = func

        # Load two-qubit kernels
        two_ptx = kernel_dir / 'two_qubit_gates_f64.ptx'
        if two_ptx.exists():
            module2 = ctypes.c_void_p()
            self._check(self.cuda.cuModuleLoad(
                ctypes.byref(module2),
                str(two_ptx).encode()
            ))
            self.modules['two'] = module2

            for name in ['apply_cnot', 'apply_cz']:
                func = ctypes.c_void_p()
                self._check(self.cuda.cuModuleGetFunction(
                    ctypes.byref(func),
                    module2,
                    name.encode()
                ))
                self.functions[name] = func

    def _launch_kernel(self, kernel_name: str, grid_dim: tuple, block_dim: tuple, args: list):
        """Launch a CUDA kernel"""
        func = self.functions[kernel_name]

        # Convert args to ctypes pointers
        arg_ptrs = []
        arg_vals = []
        for arg in args:
            if isinstance(arg, ctypes.c_void_p):
                arg_vals.append(arg)
                arg_ptrs.append(ctypes.addressof(arg))
            elif isinstance(arg, int):
                if arg > 2**32:
                    val = ctypes.c_ulonglong(arg)
                else:
                    val = ctypes.c_int(arg)
                arg_vals.append(val)
                arg_ptrs.append(ctypes.addressof(val))
            elif isinstance(arg, float):
                val = ctypes.c_double(arg)
                arg_vals.append(val)
                arg_ptrs.append(ctypes.addressof(val))

        # Create array of pointers
        arg_array = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)

        # Launch
        self._check(self.cuda.cuLaunchKernel(
            func,
            *grid_dim, 1,  # Grid dimensions
            *block_dim, 1,  # Block dimensions
            0,  # Shared memory
            None,  # Stream
            arg_array,  # Kernel args
            None  # Extra
        ))

        # Synchronize
        self._check(self.cuda.cuCtxSynchronize())

    def h(self, qubit: int):
        """Apply Hadamard gate"""
        if qubit >= self.n_qubits:
            raise ValueError("Qubit index out of bounds")

        num_pairs = self.state_size // 2
        threads_per_block = 256
        num_blocks = (num_pairs + threads_per_block - 1) // threads_per_block

        self._launch_kernel(
            'apply_hadamard',
            (num_blocks, 1),
            (threads_per_block, 1),
            [self.d_state, self.state_size, qubit]
        )

    def x(self, qubit: int):
        """Apply Pauli-X gate"""
        if qubit >= self.n_qubits:
            raise ValueError("Qubit index out of bounds")

        num_pairs = self.state_size // 2
        threads_per_block = 256
        num_blocks = (num_pairs + threads_per_block - 1) // threads_per_block

        self._launch_kernel(
            'apply_pauli_x',
            (num_blocks, 1),
            (threads_per_block, 1),
            [self.d_state, self.state_size, qubit]
        )

    def z(self, qubit: int):
        """Apply Pauli-Z gate"""
        if qubit >= self.n_qubits:
            raise ValueError("Qubit index out of bounds")

        num_pairs = self.state_size // 2
        threads_per_block = 256
        num_blocks = (num_pairs + threads_per_block - 1) // threads_per_block

        self._launch_kernel(
            'apply_pauli_z',
            (num_blocks, 1),
            (threads_per_block, 1),
            [self.d_state, self.state_size, qubit]
        )

    def cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        if control >= self.n_qubits or target >= self.n_qubits:
            raise ValueError("Qubit index out of bounds")
        if control == target:
            raise ValueError("Control and target must be different")

        num_pairs = self.state_size // 4
        threads_per_block = 256
        num_blocks = (num_pairs + threads_per_block - 1) // threads_per_block

        self._launch_kernel(
            'apply_cnot',
            (num_blocks, 1),
            (threads_per_block, 1),
            [self.d_state, self.state_size, control, target]
        )

    def cx(self, control: int, target: int):
        """Alias for cnot"""
        self.cnot(control, target)

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        """Sample measurement outcomes"""
        # Copy state to host
        state_data = np.zeros(self.state_size * 2, dtype=np.float64)
        buffer_size = self.state_size * 2 * 8

        self._check(self.cuda.cuMemcpyDtoH_v2(
            state_data.ctypes.data,
            self.d_state,
            buffer_size
        ))

        # Convert to complex
        state = state_data[::2] + 1j * state_data[1::2]

        # Compute probabilities
        probs = np.abs(state) ** 2
        probs = probs / probs.sum()  # Normalize

        # Sample
        outcomes = np.random.choice(self.state_size, size=shots, p=probs)

        # Convert to bitstring counts
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def reset(self):
        """Reset to |0...0⟩ state"""
        buffer_size = self.state_size * 2 * 8
        init_state = np.zeros(self.state_size * 2, dtype=np.float64)
        init_state[0] = 1.0

        self._check(self.cuda.cuMemcpyHtoD_v2(
            self.d_state,
            init_state.ctypes.data,
            buffer_size
        ))

    def __del__(self):
        """Cleanup GPU resources"""
        try:
            if hasattr(self, 'd_state'):
                self.cuda.cuMemFree_v2(self.d_state)
            if hasattr(self, 'context'):
                self.cuda.cuCtxDestroy_v2(self.context)
        except Exception:
            pass  # Cleanup errors are non-critical


def is_gpu_available() -> bool:
    """Check if GPU backend is available"""
    try:
        find_cuda_library()
        return True
    except (RuntimeError, OSError):
        return False


def get_gpu_info() -> Optional[Dict]:
    """Get GPU information"""
    if not is_gpu_available():
        return None

    try:
        cuda = ctypes.CDLL(find_cuda_library())
        cuda.cuInit(0)

        device = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(device), 0)

        # Get device name
        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, device)

        return {
            'available': True,
            'name': name.value.decode(),
            'cuda_version': 'Runtime detection',
        }
    except (RuntimeError, OSError, AttributeError):
        return None


if __name__ == "__main__":
    import time

    print("Testing GPU Backend (Direct CUDA Driver API)")
    print("=" * 60)

    if not is_gpu_available():
        print("GPU not available!")
        exit(1)

    info = get_gpu_info()
    print(f"GPU: {info['name']}")
    print()

    # Test Bell state
    print("Creating Bell state on 2 qubits...")
    sim = GPUStatevectorSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    counts = sim.sample(1000)
    print(f"Measurement results: {counts}")
    print()

    # Test larger circuit
    n_qubits = 20
    print(f"Benchmarking {n_qubits}-qubit GHZ circuit...")
    sim = GPUStatevectorSimulator(n_qubits)

    start = time.perf_counter()
    sim.h(0)
    for i in range(n_qubits - 1):
        sim.cnot(i, i + 1)
    elapsed = time.perf_counter() - start

    print(f"Circuit time: {elapsed * 1000:.2f} ms")
    print(f"Per-gate time: {elapsed * 1000 / n_qubits:.3f} ms")
