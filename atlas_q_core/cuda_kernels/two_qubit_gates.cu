/**
 * Two-Qubit Gate CUDA Kernels
 *
 * High-performance GPU kernels for two-qubit quantum gate operations.
 * Optimized for:
 * - Efficient control/target qubit indexing
 * - Minimized thread divergence
 * - Coalesced memory access where possible
 *
 * Performance target: 100× faster than CPU Rust for 20+ qubits
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
 * Apply CNOT gate (Controlled-NOT)
 *
 * CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
 *
 * Only flips target qubit when control qubit is 1.
 * This is the most important two-qubit gate for quantum algorithms.
 *
 * Each thread handles one amplitude pair in the control=1 subspace.
 */
__global__ void apply_cnot(
    cuDoubleComplex* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 2;  // size / 4 pairs
    if (tid >= num_pairs) return;

    // Bit masks for control and target qubits
    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;

    // Calculate indices with control=1, target=0 and target=1
    // We only process the control=1 subspace (control=0 is identity)

    // Create masks for index calculation
    unsigned long long lower_mask = (1ULL << (control_qubit < target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask = (1ULL << (control_qubit > target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    // Calculate base index with both qubits = 0
    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Index with control=1, target=0
    unsigned long long idx_c1_t0 = base_idx | control_mask;
    // Index with control=1, target=1
    unsigned long long idx_c1_t1 = idx_c1_t0 | target_mask;

    // CNOT: Swap amplitudes when control=1
    cuDoubleComplex temp = state[idx_c1_t0];
    state[idx_c1_t0] = state[idx_c1_t1];
    state[idx_c1_t1] = temp;
}

/**
 * Apply CZ gate (Controlled-Z)
 *
 * CZ = diag(1, 1, 1, -1) in computational basis
 *
 * Only applies phase flip when both control and target are 1.
 * Symmetric gate (control and target are interchangeable).
 */
__global__ void apply_cz(
    cuDoubleComplex* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_states = size >> 2;  // size / 4
    if (tid >= num_states) return;

    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;

    // Calculate index with both qubits variable
    unsigned long long lower_mask = (1ULL << (control_qubit < target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask = (1ULL << (control_qubit > target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Index with both control=1 and target=1
    unsigned long long idx_11 = base_idx | control_mask | target_mask;

    // CZ: Negate amplitude when both qubits are 1
    state[idx_11] = make_cuDoubleComplex(
        -cuCreal(state[idx_11]),
        -cuCimag(state[idx_11])
    );
}

/**
 * Apply SWAP gate
 *
 * SWAP exchanges the states of two qubits.
 *
 * SWAP = [1 0 0 0]
 *        [0 0 1 0]
 *        [0 1 0 0]
 *        [0 0 0 1]
 *
 * Swaps amplitudes: |01⟩ ↔ |10⟩
 */
__global__ void apply_swap(
    cuDoubleComplex* state,
    unsigned long long size,
    int qubit1,
    int qubit2
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 2;  // size / 4 pairs
    if (tid >= num_pairs) return;

    unsigned long long mask1 = 1ULL << qubit1;
    unsigned long long mask2 = 1ULL << qubit2;

    // Calculate base index with both qubits = 0
    unsigned long long lower_mask = (1ULL << (qubit1 < qubit2 ? qubit1 : qubit2)) - 1;
    unsigned long long mid_mask = (1ULL << (qubit1 > qubit2 ? qubit1 : qubit2)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Indices for |01⟩ and |10⟩ configurations
    unsigned long long idx_01 = base_idx | mask2;  // qubit1=0, qubit2=1
    unsigned long long idx_10 = base_idx | mask1;  // qubit1=1, qubit2=0

    // Swap amplitudes
    cuDoubleComplex temp = state[idx_01];
    state[idx_01] = state[idx_10];
    state[idx_10] = temp;
}

/**
 * Apply arbitrary two-qubit gate
 *
 * Gate matrix: 4×4 complex matrix in computational basis
 * [g00 g01 g02 g03]
 * [g10 g11 g12 g13]
 * [g20 g21 g22 g23]
 * [g30 g31 g32 g33]
 *
 * Maps basis: |00⟩, |01⟩, |10⟩, |11⟩
 *
 * This is the general form for any two-qubit gate.
 */
__global__ void apply_two_qubit_gate(
    cuDoubleComplex* state,
    unsigned long long size,
    int qubit1,
    int qubit2,
    const cuDoubleComplex* gate  // 4×4 matrix (16 elements)
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_quartets = size >> 2;
    if (tid >= num_quartets) return;

    unsigned long long mask1 = 1ULL << qubit1;
    unsigned long long mask2 = 1ULL << qubit2;

    // Calculate indices for amplitude quartet
    unsigned long long lower_mask = (1ULL << (qubit1 < qubit2 ? qubit1 : qubit2)) - 1;
    unsigned long long mid_mask = (1ULL << (qubit1 > qubit2 ? qubit1 : qubit2)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Four amplitude indices: |00⟩, |01⟩, |10⟩, |11⟩
    unsigned long long idx_00 = base_idx;
    unsigned long long idx_01 = base_idx | mask2;
    unsigned long long idx_10 = base_idx | mask1;
    unsigned long long idx_11 = base_idx | mask1 | mask2;

    // Load amplitudes
    cuDoubleComplex amp00 = state[idx_00];
    cuDoubleComplex amp01 = state[idx_01];
    cuDoubleComplex amp10 = state[idx_10];
    cuDoubleComplex amp11 = state[idx_11];

    // Apply gate matrix: new_amp = gate * old_amp
    cuDoubleComplex new00 = complex_add(
        complex_add(
            complex_mul(gate[0], amp00),
            complex_mul(gate[1], amp01)
        ),
        complex_add(
            complex_mul(gate[2], amp10),
            complex_mul(gate[3], amp11)
        )
    );

    cuDoubleComplex new01 = complex_add(
        complex_add(
            complex_mul(gate[4], amp00),
            complex_mul(gate[5], amp01)
        ),
        complex_add(
            complex_mul(gate[6], amp10),
            complex_mul(gate[7], amp11)
        )
    );

    cuDoubleComplex new10 = complex_add(
        complex_add(
            complex_mul(gate[8], amp00),
            complex_mul(gate[9], amp01)
        ),
        complex_add(
            complex_mul(gate[10], amp10),
            complex_mul(gate[11], amp11)
        )
    );

    cuDoubleComplex new11 = complex_add(
        complex_add(
            complex_mul(gate[12], amp00),
            complex_mul(gate[13], amp01)
        ),
        complex_add(
            complex_mul(gate[14], amp10),
            complex_mul(gate[15], amp11)
        )
    );

    // Store results
    state[idx_00] = new00;
    state[idx_01] = new01;
    state[idx_10] = new10;
    state[idx_11] = new11;
}

/**
 * Apply controlled rotation RY gate
 *
 * CRY(θ) applies RY(θ) to target when control=1
 *
 * Useful for variational algorithms and chemistry simulations.
 */
__global__ void apply_cry(
    cuDoubleComplex* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit,
    double theta
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 2;
    if (tid >= num_pairs) return;

    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;

    unsigned long long lower_mask = (1ULL << (control_qubit < target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask = (1ULL << (control_qubit > target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Only apply when control=1
    unsigned long long idx_c1_t0 = base_idx | control_mask;
    unsigned long long idx_c1_t1 = idx_c1_t0 | target_mask;

    cuDoubleComplex amp0 = state[idx_c1_t0];
    cuDoubleComplex amp1 = state[idx_c1_t1];

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

    state[idx_c1_t0] = new0;
    state[idx_c1_t1] = new1;
}

/**
 * Apply controlled phase gate
 *
 * CPhase(θ) = diag(1, 1, 1, e^(iθ))
 *
 * Applies phase when both qubits are 1.
 */
__global__ void apply_cphase(
    cuDoubleComplex* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit,
    double theta
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_states = size >> 2;
    if (tid >= num_states) return;

    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;

    unsigned long long lower_mask = (1ULL << (control_qubit < target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask = (1ULL << (control_qubit > target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    unsigned long long idx_11 = base_idx | control_mask | target_mask;

    // Apply phase: e^(iθ) = cos(θ) + i*sin(θ)
    cuDoubleComplex amp = state[idx_11];
    double cos_theta = cos(theta);
    double sin_theta = sin(theta);

    state[idx_11] = make_cuDoubleComplex(
        cos_theta * cuCreal(amp) - sin_theta * cuCimag(amp),
        cos_theta * cuCimag(amp) + sin_theta * cuCreal(amp)
    );
}
