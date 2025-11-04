/**
 * Single-Qubit Gate CUDA Kernels (raw f64 version)
 *
 * State stored as interleaved f64: [re0, im0, re1, im1, ...]
 * Optimized for coalesced memory access and warp-level parallelism.
 */

extern "C" {

/**
 * Apply Hadamard gate
 *
 * H = 1/√2 * [1   1]
 *            [1  -1]
 */
__global__ void apply_hadamard(
    double* state,              // Interleaved: [re, im, re, im, ...]
    unsigned long long size,    // Number of amplitudes (not f64s)
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

    // Convert to f64 indices (×2 for real/imag)
    unsigned long long idx_i = i * 2;
    unsigned long long idx_j = j * 2;

    // Load amplitudes
    double re0 = state[idx_i];
    double im0 = state[idx_i + 1];
    double re1 = state[idx_j];
    double im1 = state[idx_j + 1];

    // H gate with 1/√2 factor
    const double inv_sqrt2 = 0.7071067811865475;

    double new_re0 = inv_sqrt2 * (re0 + re1);
    double new_im0 = inv_sqrt2 * (im0 + im1);
    double new_re1 = inv_sqrt2 * (re0 - re1);
    double new_im1 = inv_sqrt2 * (im0 - im1);

    // Store results
    state[idx_i] = new_re0;
    state[idx_i + 1] = new_im0;
    state[idx_j] = new_re1;
    state[idx_j + 1] = new_im1;
}

/**
 * Apply Pauli-X gate (bit flip)
 *
 * X = [0  1]
 *     [1  0]
 */
__global__ void apply_pauli_x(
    double* state,
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

    unsigned long long idx_i = i * 2;
    unsigned long long idx_j = j * 2;

    // Swap amplitudes
    double temp_re = state[idx_i];
    double temp_im = state[idx_i + 1];

    state[idx_i] = state[idx_j];
    state[idx_i + 1] = state[idx_j + 1];

    state[idx_j] = temp_re;
    state[idx_j + 1] = temp_im;
}

/**
 * Apply Pauli-Z gate (phase flip)
 *
 * Z = [1   0]
 *     [0  -1]
 */
__global__ void apply_pauli_z(
    double* state,
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
    unsigned long long idx_j = j * 2;

    // Negate amplitude at j
    state[idx_j] = -state[idx_j];
    state[idx_j + 1] = -state[idx_j + 1];
}

/**
 * Apply RX rotation gate
 *
 * RX(θ) = [cos(θ/2)      -i*sin(θ/2)]
 *         [-i*sin(θ/2)    cos(θ/2)   ]
 */
__global__ void apply_rx(
    double* state,
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

    unsigned long long idx_i = i * 2;
    unsigned long long idx_j = j * 2;

    double re0 = state[idx_i];
    double im0 = state[idx_i + 1];
    double re1 = state[idx_j];
    double im1 = state[idx_j + 1];

    double half_theta = theta * 0.5;
    double cos_val = cos(half_theta);
    double sin_val = sin(half_theta);

    // RX matrix: [[cos, -i*sin], [-i*sin, cos]]
    // (a + bi) * (-i) = b - ai
    double new_re0 = cos_val * re0 + sin_val * im1;
    double new_im0 = cos_val * im0 - sin_val * re1;
    double new_re1 = cos_val * re1 + sin_val * im0;
    double new_im1 = cos_val * im1 - sin_val * re0;

    state[idx_i] = new_re0;
    state[idx_i + 1] = new_im0;
    state[idx_j] = new_re1;
    state[idx_j + 1] = new_im1;
}

/**
 * Apply RY rotation gate
 *
 * RY(θ) = [cos(θ/2)   -sin(θ/2)]
 *         [sin(θ/2)    cos(θ/2)]
 */
__global__ void apply_ry(
    double* state,
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

    unsigned long long idx_i = i * 2;
    unsigned long long idx_j = j * 2;

    double re0 = state[idx_i];
    double im0 = state[idx_i + 1];
    double re1 = state[idx_j];
    double im1 = state[idx_j + 1];

    double half_theta = theta * 0.5;
    double cos_val = cos(half_theta);
    double sin_val = sin(half_theta);

    // RY matrix application
    double new_re0 = cos_val * re0 - sin_val * re1;
    double new_im0 = cos_val * im0 - sin_val * im1;
    double new_re1 = sin_val * re0 + cos_val * re1;
    double new_im1 = sin_val * im0 + cos_val * im1;

    state[idx_i] = new_re0;
    state[idx_i + 1] = new_im0;
    state[idx_j] = new_re1;
    state[idx_j + 1] = new_im1;
}

}
