/**
 * Two-Qubit Gate CUDA Kernels (raw f64 version)
 *
 * State stored as interleaved f64: [re0, im0, re1, im1, ...]
 * Optimized for efficient control/target qubit indexing.
 */

extern "C" {

/**
 * Apply CNOT gate (Controlled-NOT)
 *
 * CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
 */
__global__ void apply_cnot(
    double* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = size >> 2;
    if (tid >= num_pairs) return;

    unsigned long long control_mask = 1ULL << control_qubit;
    unsigned long long target_mask = 1ULL << target_qubit;

    // Calculate indices
    unsigned long long lower_mask = (1ULL << (control_qubit < target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask = (1ULL << (control_qubit > target_qubit ? control_qubit : target_qubit)) - 1;
    unsigned long long mid_mask_adjusted = mid_mask & ~lower_mask;

    unsigned long long base_idx = (tid & lower_mask) |
                                  ((tid & mid_mask_adjusted) << 1) |
                                  ((tid & ~mid_mask) << 2);

    // Indices with control=1, target=0 and target=1
    unsigned long long idx_c1_t0 = base_idx | control_mask;
    unsigned long long idx_c1_t1 = idx_c1_t0 | target_mask;

    // Convert to f64 indices
    unsigned long long f64_idx_t0 = idx_c1_t0 * 2;
    unsigned long long f64_idx_t1 = idx_c1_t1 * 2;

    // Swap amplitudes when control=1
    double temp_re = state[f64_idx_t0];
    double temp_im = state[f64_idx_t0 + 1];

    state[f64_idx_t0] = state[f64_idx_t1];
    state[f64_idx_t0 + 1] = state[f64_idx_t1 + 1];

    state[f64_idx_t1] = temp_re;
    state[f64_idx_t1 + 1] = temp_im;
}

/**
 * Apply CZ gate (Controlled-Z)
 *
 * CZ = diag(1, 1, 1, -1) in computational basis
 */
__global__ void apply_cz(
    double* state,
    unsigned long long size,
    int control_qubit,
    int target_qubit
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

    // Index with both control=1 and target=1
    unsigned long long idx_11 = base_idx | control_mask | target_mask;
    unsigned long long f64_idx = idx_11 * 2;

    // Negate amplitude when both qubits are 1
    state[f64_idx] = -state[f64_idx];
    state[f64_idx + 1] = -state[f64_idx + 1];
}

}
