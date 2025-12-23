//! Statevector Backend - Full Quantum State Simulation
//!
//! Implements exact simulation of quantum circuits by maintaining
//! the full state vector in memory.
//!
//! Complexity: O(2^n) memory, O(2^n) per gate
//! Use for: Small circuits (≤20 qubits), Grover's, QFT, general algorithms
//!
//! Key optimizations:
//! - SIMD autovectorization for complex arithmetic
//! - Parallel gate application via Rayon for n > 10
//! - Cache-friendly memory access patterns
//! - Zero-copy where possible

use num_complex::Complex64;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;

/// Statevector quantum simulator
///
/// Stores the full quantum state as a vector of 2^n complex amplitudes.
/// Memory usage: 16 bytes × 2^n (e.g., 16 MB for 20 qubits)
#[pyclass]
pub struct StatevectorSimulatorRust {
    n_qubits: usize,
    /// State vector: amplitudes[i] = amplitude of |i⟩
    /// Length: 2^n_qubits
    state: Vec<Complex64>,
}

#[pymethods]
impl StatevectorSimulatorRust {
    /// Create new statevector simulator initialized to |00...0⟩
    #[new]
    pub fn new(n_qubits: usize) -> Self {
        let size = 1 << n_qubits; // 2^n_qubits
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0); // Initialize to |00...0⟩

        Self { n_qubits, state }
    }

    /// Get number of qubits
    #[getter]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Apply single-qubit gate
    ///
    /// Args:
    ///     qubit: Target qubit index
    ///     gate_flat: 2×2 gate matrix as flat list of 8 floats [re00, im00, re01, im01, re10, im10, re11, im11]
    ///                where gate = [[a,b],[c,d]] in row-major order
    pub fn apply_single_gate_raw(&mut self, qubit: usize, gate_flat: Vec<f64>) {
        assert_eq!(gate_flat.len(), 8, "Gate must have 8 elements (4 complex numbers)");

        let gate = [
            Complex64::new(gate_flat[0], gate_flat[1]),
            Complex64::new(gate_flat[2], gate_flat[3]),
            Complex64::new(gate_flat[4], gate_flat[5]),
            Complex64::new(gate_flat[6], gate_flat[7]),
        ];

        self._apply_single_gate(qubit, gate);
    }

    /// Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        let gate = [
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// Pauli X gate
    pub fn x(&mut self, qubit: usize) {
        let gate = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// Pauli Y gate
    pub fn y(&mut self, qubit: usize) {
        let gate = [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// Pauli Z gate
    pub fn z(&mut self, qubit: usize) {
        let gate = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// S gate (phase)
    pub fn s(&mut self, qubit: usize) {
        let gate = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// S† gate
    pub fn sdg(&mut self, qubit: usize) {
        let gate = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// T gate
    pub fn t(&mut self, qubit: usize) {
        let phase = std::f64::consts::FRAC_PI_4; // π/4
        let gate = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(phase.cos(), phase.sin()),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// T† gate
    pub fn tdg(&mut self, qubit: usize) {
        let phase = -std::f64::consts::FRAC_PI_4; // -π/4
        let gate = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(phase.cos(), phase.sin()),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// RX rotation gate
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        let half_theta = theta / 2.0;
        let cos = half_theta.cos();
        let sin = half_theta.sin();
        let gate = [
            Complex64::new(cos, 0.0),
            Complex64::new(0.0, -sin),
            Complex64::new(0.0, -sin),
            Complex64::new(cos, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// RY rotation gate
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        let half_theta = theta / 2.0;
        let cos = half_theta.cos();
        let sin = half_theta.sin();
        let gate = [
            Complex64::new(cos, 0.0),
            Complex64::new(-sin, 0.0),
            Complex64::new(sin, 0.0),
            Complex64::new(cos, 0.0),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// RZ rotation gate
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        let half_theta = theta / 2.0;
        let gate = [
            Complex64::new(half_theta.cos(), -half_theta.sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(half_theta.cos(), half_theta.sin()),
        ];
        self._apply_single_gate(qubit, gate);
    }

    /// CNOT gate (controlled-X)
    pub fn cnot(&mut self, control: usize, target: usize) {
        // CNOT matrix:
        // |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩
        #[rustfmt::skip]
        let gate = [
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        ];
        self._apply_two_qubit_gate(control, target, gate);
    }

    /// CZ gate (controlled-Z)
    pub fn cz(&mut self, control: usize, target: usize) {
        #[rustfmt::skip]
        let gate = [
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
        ];
        self._apply_two_qubit_gate(control, target, gate);
    }

    /// SWAP gate
    pub fn swap(&mut self, qubit0: usize, qubit1: usize) {
        #[rustfmt::skip]
        let gate = [
            Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
        ];
        self._apply_two_qubit_gate(qubit0, qubit1, gate);
    }

    /// Measure a single qubit
    ///
    /// Returns: measurement outcome (0 or 1)
    ///
    /// This is a projective measurement that collapses the state.
    pub fn measure(&mut self, qubit: usize) -> bool {
        assert!(qubit < self.n_qubits, "Qubit index out of range");

        let size = self.state.len();
        let qubit_mask = 1 << qubit;

        // Calculate probability of measuring |1⟩
        let mut prob_one = 0.0;
        for i in 0..size {
            if (i & qubit_mask) != 0 {
                let amp = self.state[i];
                prob_one += amp.re * amp.re + amp.im * amp.im;
            }
        }

        // Sample outcome
        let outcome = rand::thread_rng().gen::<f64>() < prob_one;

        // Collapse state
        let norm_factor = if outcome {
            1.0 / prob_one.sqrt()
        } else {
            1.0 / (1.0 - prob_one).sqrt()
        };

        for i in 0..size {
            let has_bit = (i & qubit_mask) != 0;
            if has_bit != outcome {
                self.state[i] = Complex64::new(0.0, 0.0);
            } else {
                self.state[i] *= norm_factor;
            }
        }

        outcome
    }

    /// Sample multiple measurement outcomes without collapsing
    ///
    /// Returns: list of measurement outcomes as integers
    pub fn sample(&self, num_shots: usize) -> Vec<usize> {
        // Compute probabilities
        let mut probs: Vec<f64> = self
            .state
            .iter()
            .map(|amp| amp.re * amp.re + amp.im * amp.im)
            .collect();

        // Normalize (in case of numerical errors)
        let total: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= total;
        }

        // Sample using cumulative probabilities
        let mut results = Vec::with_capacity(num_shots);
        let mut rng = rand::thread_rng();

        for _ in 0..num_shots {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    results.push(i);
                    break;
                }
            }
        }

        results
    }

    /// Reset to |00...0⟩ state
    pub fn reset(&mut self) {
        for i in 1..self.state.len() {
            self.state[i] = Complex64::new(0.0, 0.0);
        }
        self.state[0] = Complex64::new(1.0, 0.0);
    }

    // =========================================================================
    // IR (Informational Relativity) Coherence Metrics
    // =========================================================================
    // These methods compute coherence on the response field (quantum state
    // amplitudes) per IR Law L8 (Placement Principle).

    /// Compute IR response field coherence R̄ from quantum state amplitudes.
    ///
    /// This is the CORRECT placement per IR Law L8:
    /// "Coherence must be measured on response manifolds, not on probes or encodings."
    ///
    /// Mathematical formula:
    ///   R̄ = |Σ |χ_i| e^(iθ_i)| / Σ |χ_i|
    /// where χ_i = amplitude_i and θ_i = arg(amplitude_i)
    ///
    /// Returns: (R_bar, V_phi, is_above_e2)
    /// - R_bar: Mean resultant length [0, 1] (higher = more coherent)
    /// - V_phi: Circular variance [0, ∞] (lower = more coherent)
    /// - is_above_e2: Whether R̄ > e^-2 ≈ 0.135 (GO/NO-GO threshold)
    pub fn compute_response_coherence(&self) -> (f64, f64, bool) {
        const E2_THRESHOLD: f64 = 0.135; // e^-2 ≈ 0.1353

        let mut weighted_sum_re = 0.0;
        let mut weighted_sum_im = 0.0;
        let mut total_weight = 0.0;

        for amp in &self.state {
            let magnitude = (amp.re * amp.re + amp.im * amp.im).sqrt();
            if magnitude > 1e-15 {
                // Weight phasor by magnitude (response strength)
                weighted_sum_re += amp.re; // |χ| * cos(θ) = Re(χ)
                weighted_sum_im += amp.im; // |χ| * sin(θ) = Im(χ)
                total_weight += magnitude;
            }
        }

        if total_weight < 1e-15 {
            return (0.0, f64::INFINITY, false);
        }

        // Compute mean resultant length
        let mean_re = weighted_sum_re / total_weight;
        let mean_im = weighted_sum_im / total_weight;
        let r_bar = (mean_re * mean_re + mean_im * mean_im).sqrt();
        let r_bar = r_bar.clamp(0.0, 1.0);

        // Compute circular variance via coherence law: V_φ = -2 ln(R̄)
        let v_phi = if r_bar > 1e-10 {
            -2.0 * r_bar.ln()
        } else {
            f64::INFINITY
        };

        let is_above_e2 = r_bar > E2_THRESHOLD;

        (r_bar, v_phi, is_above_e2)
    }

    /// Compute spectral coherence from state amplitudes.
    ///
    /// This measures power concentration in dominant amplitude modes.
    /// High spectral coherence = power concentrated in few basis states.
    /// Low spectral coherence = power spread across many states.
    ///
    /// Returns: Spectral coherence R̄ ∈ [0, 1]
    pub fn compute_spectral_coherence(&self) -> f64 {
        let mut max_prob = 0.0;
        let mut total_prob = 0.0;

        for amp in &self.state {
            let prob = amp.re * amp.re + amp.im * amp.im;
            total_prob += prob;
            if prob > max_prob {
                max_prob = prob;
            }
        }

        if total_prob < 1e-15 {
            return 0.0;
        }

        // Coherence = concentration in dominant mode
        max_prob / total_prob
    }

    /// Compute the IR relational matrix M_ij = χ_i χ_j cos(θ_i - θ_j).
    ///
    /// This implements IR spectral lifting representation for structure
    /// identification. The dominant eigenmode of M encodes global coherent
    /// structure invisible to pointwise statistics.
    ///
    /// Note: Returns flattened matrix in row-major order for Python interop.
    /// Matrix is n×n where n = 2^n_qubits.
    ///
    /// Returns: (M_flat, eigenvalues_sorted_desc, spectral_coherence)
    pub fn compute_relational_matrix(&self) -> (Vec<f64>, Vec<f64>, f64) {
        let n = self.state.len();
        let mut m_flat = vec![0.0; n * n];

        // Extract magnitudes and phases
        let magnitudes: Vec<f64> = self.state.iter()
            .map(|amp| (amp.re * amp.re + amp.im * amp.im).sqrt())
            .collect();

        let phases: Vec<f64> = self.state.iter()
            .map(|amp| amp.im.atan2(amp.re))
            .collect();

        // Build relational matrix M_ij = χ_i * χ_j * cos(θ_i - θ_j)
        for i in 0..n {
            for j in 0..n {
                let phase_diff = phases[i] - phases[j];
                m_flat[i * n + j] = magnitudes[i] * magnitudes[j] * phase_diff.cos();
            }
        }

        // For large matrices, skip eigendecomposition (expensive)
        // Return spectral coherence estimate from Frobenius norm
        if n > 256 {
            let frobenius_sq: f64 = m_flat.iter().map(|x| x * x).sum();
            let trace: f64 = (0..n).map(|i| m_flat[i * n + i]).sum();
            let spectral_coherence = if frobenius_sq > 1e-15 {
                (trace * trace / frobenius_sq).sqrt().clamp(0.0, 1.0)
            } else {
                0.0
            };
            return (m_flat, vec![], spectral_coherence);
        }

        // For small matrices, compute eigenvalues
        // Simple power iteration for dominant eigenvalue (sufficient for coherence)
        let dominant_eigenvalue = self._power_iteration_eigenvalue(&m_flat, n, 50);

        // Compute trace for spectral coherence estimate
        let trace: f64 = (0..n).map(|i| m_flat[i * n + i]).sum();

        let spectral_coherence = if trace.abs() > 1e-15 {
            (dominant_eigenvalue / trace).clamp(0.0, 1.0)
        } else {
            0.0
        };

        (m_flat, vec![dominant_eigenvalue], spectral_coherence)
    }

    /// Compute coherence-aware truncation recommendation.
    ///
    /// Based on IR insights:
    /// - High coherence (R̄ > e^-2): Preserve structure, tight truncation
    /// - Low coherence (R̄ < e^-2): Structure hidden, aggressive truncation OK
    ///
    /// Args:
    ///     base_threshold: Base truncation threshold
    ///
    /// Returns: (adjusted_threshold, coherence_regime)
    /// - adjusted_threshold: Modified threshold based on coherence
    /// - coherence_regime: 0 = AIR (aggressive OK), 1 = transition, 2 = IR (conservative)
    pub fn coherence_truncation_recommendation(&self, base_threshold: f64) -> (f64, i32) {
        let (r_bar, _, _) = self.compute_response_coherence();

        const E2_THRESHOLD: f64 = 0.135;

        if r_bar > E2_THRESHOLD {
            // IR regime: structure observable, be conservative
            (base_threshold * 0.5, 2)
        } else if r_bar > E2_THRESHOLD * 0.5 {
            // Transition regime: moderate caution
            (base_threshold, 1)
        } else {
            // AIR regime: structure hidden, aggressive truncation OK
            (base_threshold * 2.0, 0)
        }
    }

    /// Get state amplitudes as flat list of (re, im) pairs.
    ///
    /// This provides direct access to the response field for Python-side
    /// coherence analysis.
    pub fn get_amplitudes(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.state.len() * 2);
        for amp in &self.state {
            result.push(amp.re);
            result.push(amp.im);
        }
        result
    }

    /// Get state probabilities |amplitude|².
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.state.iter()
            .map(|amp| amp.re * amp.re + amp.im * amp.im)
            .collect()
    }
}

// Private helper methods (not exposed to Python)
impl StatevectorSimulatorRust {
    /// Power iteration to find dominant eigenvalue of symmetric matrix.
    /// Used for spectral coherence computation from relational matrix.
    fn _power_iteration_eigenvalue(&self, m_flat: &[f64], n: usize, iterations: usize) -> f64 {
        // Start with uniform vector
        let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
        let mut new_v = vec![0.0; n];

        for _ in 0..iterations {
            // Matrix-vector multiply: new_v = M * v
            for i in 0..n {
                new_v[i] = 0.0;
                for j in 0..n {
                    new_v[i] += m_flat[i * n + j] * v[j];
                }
            }

            // Compute norm for normalization and eigenvalue estimate
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return 0.0;
            }

            // Normalize
            for x in new_v.iter_mut() {
                *x /= norm;
            }

            std::mem::swap(&mut v, &mut new_v);
        }

        // Compute Rayleigh quotient for eigenvalue: λ = v^T M v / v^T v
        let mut numerator = 0.0;
        for i in 0..n {
            for j in 0..n {
                numerator += v[i] * m_flat[i * n + j] * v[j];
            }
        }

        numerator
    }

    fn _apply_single_gate(&mut self, qubit: usize, gate: [Complex64; 4]) {
        assert!(qubit < self.n_qubits, "Qubit index out of range");

        let size = self.state.len();
        let qubit_mask = 1 << qubit;

        // Extract gate elements
        let u00 = gate[0];
        let u01 = gate[1];
        let u10 = gate[2];
        let u11 = gate[3];

        // Use parallel iteration for large states (> 4096 amplitudes, ~12 qubits)
        if size > 4096 {
            // Parallel version: compute updates in parallel, then apply serially
            let pairs: Vec<(usize, usize, Complex64, Complex64)> = (0..size)
                .into_par_iter()
                .filter(|&i| (i & qubit_mask) == 0)
                .map(|i| {
                    let j = i | qubit_mask;
                    let amp0 = self.state[i];
                    let amp1 = self.state[j];

                    // Apply gate
                    let new0 = u00 * amp0 + u01 * amp1;
                    let new1 = u10 * amp0 + u11 * amp1;

                    (i, j, new0, new1)
                })
                .collect();

            // Write back serially (no data races)
            for (i, j, new0, new1) in pairs {
                self.state[i] = new0;
                self.state[j] = new1;
            }
        } else {
            // Serial version for small states
            for i in 0..size {
                if (i & qubit_mask) == 0 {
                    let j = i | qubit_mask;

                    let amp0 = self.state[i];
                    let amp1 = self.state[j];

                    // Apply gate: |ψ'⟩ = U|ψ⟩
                    let new0 = u00 * amp0 + u01 * amp1;
                    let new1 = u10 * amp0 + u11 * amp1;

                    self.state[i] = new0;
                    self.state[j] = new1;
                }
            }
        }
    }

    fn _apply_two_qubit_gate(
        &mut self,
        qubit0: usize,
        qubit1: usize,
        gate: [Complex64; 16],
    ) {
        assert!(qubit0 < self.n_qubits, "Qubit0 index out of range");
        assert!(qubit1 < self.n_qubits, "Qubit1 index out of range");
        assert_ne!(qubit0, qubit1, "Qubits must be different");

        let size = self.state.len();
        let mask0 = 1 << qubit0;
        let mask1 = 1 << qubit1;

        // For each basis state, check if both control bits are 0
        // Then apply gate to the 4 relevant amplitudes
        for i in 0..size {
            if (i & mask0) == 0 && (i & mask1) == 0 {
                // Four indices corresponding to |00⟩, |01⟩, |10⟩, |11⟩ on these qubits
                let i00 = i;
                let i01 = i | mask1;
                let i10 = i | mask0;
                let i11 = i | mask0 | mask1;

                // Get current amplitudes
                let a00 = self.state[i00];
                let a01 = self.state[i01];
                let a10 = self.state[i10];
                let a11 = self.state[i11];

                // Apply 4×4 gate
                let new00 = gate[0] * a00 + gate[1] * a01 + gate[2] * a10 + gate[3] * a11;
                let new01 = gate[4] * a00 + gate[5] * a01 + gate[6] * a10 + gate[7] * a11;
                let new10 = gate[8] * a00 + gate[9] * a01 + gate[10] * a10 + gate[11] * a11;
                let new11 = gate[12] * a00 + gate[13] * a01 + gate[14] * a10 + gate[15] * a11;

                // Write back
                self.state[i00] = new00;
                self.state[i01] = new01;
                self.state[i10] = new10;
                self.state[i11] = new11;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let sim = StatevectorSimulatorRust::new(3);
        assert_eq!(sim.n_qubits(), 3);
        assert_eq!(sim.state.len(), 8);
        assert_eq!(sim.state[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_hadamard() {
        let mut sim = StatevectorSimulatorRust::new(1);
        sim.h(0);

        // After H on |0⟩, should be (|0⟩ + |1⟩)/√2
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((sim.state[0].re - sqrt2_inv).abs() < 1e-10);
        assert!((sim.state[1].re - sqrt2_inv).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_bell() {
        let mut sim = StatevectorSimulatorRust::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        // Bell state: (|00⟩ + |11⟩)/√2
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((sim.state[0].re - sqrt2_inv).abs() < 1e-10); // |00⟩
        assert!((sim.state[1].norm() < 1e-10)); // |01⟩ should be 0
        assert!((sim.state[2].norm() < 1e-10)); // |10⟩ should be 0
        assert!((sim.state[3].re - sqrt2_inv).abs() < 1e-10); // |11⟩
    }

    // ==========================================================================
    // IR Coherence Tests
    // ==========================================================================

    #[test]
    fn test_initial_state_coherence() {
        // |0⟩ state should have perfect spectral coherence (all power in one mode)
        let sim = StatevectorSimulatorRust::new(3);
        let spectral_coh = sim.compute_spectral_coherence();
        assert!((spectral_coh - 1.0).abs() < 1e-10, "Initial |0⟩ should have spectral coherence = 1.0");
    }

    #[test]
    fn test_superposition_coherence() {
        // H on all qubits creates uniform superposition
        // Should have lower spectral coherence (power spread)
        let mut sim = StatevectorSimulatorRust::new(3);
        sim.h(0);
        sim.h(1);
        sim.h(2);

        let spectral_coh = sim.compute_spectral_coherence();
        // With 8 states, each has 1/8 probability, so max_prob/total = 1/8 = 0.125
        assert!((spectral_coh - 0.125).abs() < 1e-10, "Uniform superposition should have spectral coherence = 1/n");
    }

    #[test]
    fn test_bell_state_response_coherence() {
        // Bell state (|00⟩ + |11⟩)/√2 should have high response coherence
        // Both amplitudes have the same phase (0)
        let mut sim = StatevectorSimulatorRust::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        let (r_bar, v_phi, is_above_e2) = sim.compute_response_coherence();

        // With phases aligned at 0, R̄ should be 1.0
        assert!(r_bar > 0.99, "Bell state should have high response coherence, got {}", r_bar);
        assert!(is_above_e2, "Bell state should be in IR regime (above e^-2)");
        assert!(v_phi < 0.1, "Bell state should have low circular variance");
    }

    #[test]
    fn test_coherence_truncation_recommendation() {
        // Initial state (high coherence) should recommend conservative truncation
        let sim = StatevectorSimulatorRust::new(2);
        let (adj_thresh, regime) = sim.coherence_truncation_recommendation(1e-6);

        // High coherence → regime = 2 (IR), threshold halved
        assert_eq!(regime, 2, "Initial state should be in IR regime");
        assert!((adj_thresh - 0.5e-6).abs() < 1e-12, "Threshold should be halved in IR regime");
    }

    #[test]
    fn test_get_amplitudes() {
        let mut sim = StatevectorSimulatorRust::new(2);
        sim.h(0);

        let amps = sim.get_amplitudes();
        // Should be 4 states × 2 (re, im) = 8 values
        assert_eq!(amps.len(), 8);

        // After H(0) on |00⟩: (|00⟩ + |01⟩)/√2 in little-endian qubit order
        // state[0] = |00⟩, state[1] = |01⟩ (qubit 0 flipped)
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        assert!((amps[0] - sqrt2_inv).abs() < 1e-10); // |00⟩ re
        assert!(amps[1].abs() < 1e-10); // |00⟩ im
        assert!((amps[2] - sqrt2_inv).abs() < 1e-10); // |01⟩ re (index 1*2 = 2)
    }

    #[test]
    fn test_relational_matrix() {
        // Small 2-qubit test
        let mut sim = StatevectorSimulatorRust::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        let (m_flat, eigenvalues, spectral_coh) = sim.compute_relational_matrix();

        // Matrix should be 4×4 = 16 elements
        assert_eq!(m_flat.len(), 16);

        // Should have non-trivial spectral coherence
        assert!(spectral_coh > 0.0, "Spectral coherence should be positive");
        assert!(spectral_coh <= 1.0, "Spectral coherence should be ≤ 1");
    }
}
