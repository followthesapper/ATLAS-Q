/**
 * Single-Qubit Gate CUDA Kernels
 *
 * High-performance GPU kernels for single-qubit quantum gate operations.
 * Optimized for:
 * - Coalesced memory access (128-byte aligned)
 * - Warp-level parallelism (32 threads per warp)
 * - Minimal shared memory usage
 *
 * Target performance: 100× faster than CPU for 20+ qubits
 */

#include <cuComplex.h>

// Helper functions for complex arithmetic
__device__ __forceinline__ cuDoubleComplex complex_add(
    cuDoubleComplex a,
    cuDoubleComplex b
) {
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b),
        cuCimag(a) + cuCimag(b)
    );
}

__device__ __forceinline__ cuDoubleComplex complex_mul(
    cuDoubleComplex a,
    cuDoubleComplex b
) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

/**
 * Apply arbitrary single-qubit gate
 *
 * Gate matrix: [g00, g01]
 *              [g10, g11]
 *
 * Each thread handles one amplitude pair (i, j) where:
 * - i has target qubit = 0
 * - j has target qubit = 1
 *
 * Memory access pattern is coalesced for optimal bandwidth.
 */
__global__ void apply_single_qubit_gate(
    cuDoubleComplex* state,         // State vector (size 2^n)
    unsigned long long size,        // State vector size
    int target_qubit,               // Target qubit index
    const cuDoubleComplex* gate     // Gate matrix [g00, g01, g10, g11]
) {
    // Global thread index
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one amplitude pair
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    // Calculate indices for amplitude pair
    // Convert flat index to (i, j) where i has target qubit = 0
    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    // Index with target qubit = 0
    unsigned long long i = (tid & lower_mask) | ((tid & upper_mask) << 1);
    // Index with target qubit = 1
    unsigned long long j = i | mask;

    // Load amplitudes (coalesced access)
    cuDoubleComplex amp0 = state[i];
    cuDoubleComplex amp1 = state[j];

    // Load gate matrix from global memory
    // (will be cached in L1 after first access)
    cuDoubleComplex g00 = gate[0];
    cuDoubleComplex g01 = gate[1];
    cuDoubleComplex g10 = gate[2];
    cuDoubleComplex g11 = gate[3];

    // Apply gate: [new0, new1] = gate * [amp0, amp1]
    cuDoubleComplex new0 = complex_add(
        complex_mul(g00, amp0),
        complex_mul(g01, amp1)
    );
    cuDoubleComplex new1 = complex_add(
        complex_mul(g10, amp0),
        complex_mul(g11, amp1)
    );

    // Store results (coalesced access)
    state[i] = new0;
    state[j] = new1;
}

/**
 * Optimized Hadamard gate kernel
 *
 * H = 1/√2 * [1   1]
 *            [1  -1]
 *
 * Specialized version that avoids loading gate matrix from memory.
 */
__global__ void apply_hadamard(
    cuDoubleComplex* state,
    unsigned long long size,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    unsigned long long i = (tid & lower_mask) | ((tid & upper_mask) << 1);
    unsigned long long j = i | mask;

    cuDoubleComplex amp0 = state[i];
    cuDoubleComplex amp1 = state[j];

    // H gate with 1/√2 factor
    const double inv_sqrt2 = 0.7071067811865475; // 1/√2

    cuDoubleComplex new0 = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(amp0) + cuCreal(amp1)),
        inv_sqrt2 * (cuCimag(amp0) + cuCimag(amp1))
    );
    cuDoubleComplex new1 = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(amp0) - cuCreal(amp1)),
        inv_sqrt2 * (cuCimag(amp0) - cuCimag(amp1))
    );

    state[i] = new0;
    state[j] = new1;
}

/**
 * Optimized Pauli-X gate kernel
 *
 * X = [0  1]
 *     [1  0]
 *
 * Just swaps amplitudes (no multiplication needed).
 */
__global__ void apply_pauli_x(
    cuDoubleComplex* state,
    unsigned long long size,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    unsigned long long i = (tid & lower_mask) | ((tid & upper_mask) << 1);
    unsigned long long j = i | mask;

    // Swap amplitudes
    cuDoubleComplex temp = state[i];
    state[i] = state[j];
    state[j] = temp;
}

/**
 * Optimized Pauli-Z gate kernel
 *
 * Z = [1   0]
 *     [0  -1]
 *
 * Only modifies amplitude with target qubit = 1.
 */
__global__ void apply_pauli_z(
    cuDoubleComplex* state,
    unsigned long long size,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    unsigned long long j = ((tid & lower_mask) | ((tid & upper_mask) << 1)) | mask;

    // Negate amplitude at j
    state[j] = make_cuDoubleComplex(-cuCreal(state[j]), -cuCimag(state[j]));
}

/**
 * Rotation gate kernels
 *
 * These gates have parameters (theta) and need to compute sin/cos.
 * We compute the gate matrix on host and pass it in for efficiency.
 */

// RX(θ) = [cos(θ/2)      -i*sin(θ/2)]
//         [-i*sin(θ/2)    cos(θ/2)   ]
__global__ void apply_rx(
    cuDoubleComplex* state,
    unsigned long long size,
    int target_qubit,
    double theta
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    unsigned long long i = (tid & lower_mask) | ((tid & upper_mask) << 1);
    unsigned long long j = i | mask;

    cuDoubleComplex amp0 = state[i];
    cuDoubleComplex amp1 = state[j];

    double half_theta = theta * 0.5;
    double cos_val = cos(half_theta);
    double sin_val = sin(half_theta);

    // RX matrix application
    cuDoubleComplex new0 = make_cuDoubleComplex(
        cos_val * cuCreal(amp0) + sin_val * cuCimag(amp1),
        cos_val * cuCimag(amp0) - sin_val * cuCreal(amp1)
    );
    cuDoubleComplex new1 = make_cuDoubleComplex(
        cos_val * cuCreal(amp1) + sin_val * cuCimag(amp0),
        cos_val * cuCimag(amp1) - sin_val * cuCreal(amp0)
    );

    state[i] = new0;
    state[j] = new1;
}

// RY(θ) = [cos(θ/2)   -sin(θ/2)]
//         [sin(θ/2)    cos(θ/2)]
__global__ void apply_ry(
    cuDoubleComplex* state,
    unsigned long long size,
    int target_qubit,
    double theta
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 1;
    if (tid >= num_pairs) return;

    unsigned long long mask = 1ULL << target_qubit;
    unsigned long long lower_mask = mask - 1;
    unsigned long long upper_mask = ~lower_mask;

    unsigned long long i = (tid & lower_mask) | ((tid & upper_mask) << 1);
    unsigned long long j = i | mask;

    cuDoubleComplex amp0 = state[i];
    cuDoubleComplex amp1 = state[j];

    double half_theta = theta * 0.5;
    double cos_val = cos(half_theta);
    double sin_val = sin(half_theta);

    // RY matrix application
    cuDoubleComplex new0 = make_cuDoubleComplex(
        cos_val * cuCreal(amp0) - sin_val * cuCreal(amp1),
        cos_val * cuCimag(amp0) - sin_val * cuCimag(amp1)
    );
    cuDoubleComplex new1 = make_cuDoubleComplex(
        sin_val * cuCreal(amp0) + cos_val * cuCreal(amp1),
        sin_val * cuCimag(amp0) + cos_val * cuCimag(amp1)
    );

    state[i] = new0;
    state[j] = new1;
}

/**
 * Compute probabilities for all basis states
 *
 * Used for measurement and sampling.
 * Each thread computes |amplitude|^2 for one state.
 */
__global__ void compute_probabilities(
    const cuDoubleComplex* state,
    double* probabilities,
    unsigned long long size
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        cuDoubleComplex amp = state[idx];
        double re = cuCreal(amp);
        double im = cuCimag(amp);
        probabilities[idx] = re * re + im * im;
    }
}
