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
                prob_one += (amp.re * amp.re + amp.im * amp.im);
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
}

// Private helper methods (not exposed to Python)
impl StatevectorSimulatorRust {
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
}
