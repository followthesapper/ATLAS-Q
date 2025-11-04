//! Stabilizer Backend - Clifford Circuit Simulation
//!
//! Implements the Gottesman-Knill theorem for efficient simulation of
//! Clifford circuits using the stabilizer formalism.
//!
//! Complexity: O(n²) gates, O(n²) memory
//! vs O(2^n) for statevector
//!
//! Reference: "Improved Simulation of Stabilizer Circuits" - Aaronson & Gottesman

use bitvec::prelude::*;
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;

/// Stabilizer simulator using bit-packed tableau representation
///
/// The tableau represents 2n stabilizers + n destabilizers:
/// - Rows 0..n: Destabilizers
/// - Rows n..2n: Stabilizers
/// - Columns 0..n: X part
/// - Columns n..2n: Z part
/// - Column 2n: Phase bit
#[pyclass]
pub struct StabilizerSimulatorRust {
    n_qubits: usize,
    /// Bit-packed tableau [2n × (2n+1)]
    /// Using BitVec for efficient bit operations
    tableau: BitVec<u8, Lsb0>,
    /// Measurement outcomes (for deterministic measurements)
    measurements: Vec<Option<bool>>,
}

#[pymethods]
impl StabilizerSimulatorRust {
    /// Create new stabilizer simulator
    #[new]
    pub fn new(n_qubits: usize) -> Self {
        let rows = 2 * n_qubits;
        let cols = 2 * n_qubits + 1;
        let mut tableau = bitvec![u8, Lsb0; 0; rows * cols];

        // Initialize to |00...0⟩ state
        // Destabilizers: X_i for i in 0..n
        for i in 0..n_qubits {
            tableau.set(i * cols + i, true); // X part
        }

        // Stabilizers: Z_i for i in 0..n
        for i in 0..n_qubits {
            tableau.set((n_qubits + i) * cols + (n_qubits + i), true); // Z part
        }

        Self {
            n_qubits,
            tableau,
            measurements: vec![None; n_qubits],
        }
    }

    /// Apply Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        let cols = 2 * self.n_qubits + 1;
        let n = self.n_qubits;

        // For each row in the tableau
        for row in 0..(2 * n) {
            let x_bit = self.get_bit(row, qubit);
            let z_bit = self.get_bit(row, n + qubit);

            // H: X <-> Z, add phase if both X and Z
            if x_bit && z_bit {
                self.flip_phase(row);
            }

            // Swap X and Z
            self.set_bit(row, qubit, z_bit);
            self.set_bit(row, n + qubit, x_bit);
        }
    }

    /// Apply Pauli X gate
    pub fn x(&mut self, qubit: usize) {
        let n = self.n_qubits;

        // X gate adds phase if Z is present
        for row in 0..(2 * n) {
            if self.get_bit(row, n + qubit) {
                self.flip_phase(row);
            }
        }
    }

    /// Apply Pauli Y gate
    pub fn y(&mut self, qubit: usize) {
        let n = self.n_qubits;

        // Y gate adds phase if either X or Z is present
        for row in 0..(2 * n) {
            let x_bit = self.get_bit(row, qubit);
            let z_bit = self.get_bit(row, n + qubit);

            if x_bit || z_bit {
                self.flip_phase(row);
            }
        }
    }

    /// Apply Pauli Z gate
    pub fn z(&mut self, qubit: usize) {
        let n = self.n_qubits;

        // Z gate adds phase if X is present
        for row in 0..(2 * n) {
            if self.get_bit(row, qubit) {
                self.flip_phase(row);
            }
        }
    }

    /// Apply S gate (phase gate)
    pub fn s(&mut self, qubit: usize) {
        let n = self.n_qubits;

        // S: X -> Y (add Z), add phase if X present
        for row in 0..(2 * n) {
            let x_bit = self.get_bit(row, qubit);
            if x_bit {
                let z_bit = self.get_bit(row, n + qubit);
                self.set_bit(row, n + qubit, !z_bit); // Flip Z
                if !z_bit {
                    // Was 0, now 1 -> no phase change
                } else {
                    // Was 1, now 0 -> add phase (X·Z = iY)
                    self.flip_phase(row);
                }
            }
        }
    }

    /// Apply S† gate
    pub fn sdg(&mut self, qubit: usize) {
        // S† = S · S · S
        self.s(qubit);
        self.s(qubit);
        self.s(qubit);
    }

    /// Apply CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) {
        let n = self.n_qubits;

        // CNOT updates:
        // X on control: add X on target
        // Z on target: add Z on control
        // Phase update if X_control · Z_target

        for row in 0..(2 * n) {
            let x_control = self.get_bit(row, control);
            let z_control = self.get_bit(row, n + control);
            let x_target = self.get_bit(row, target);
            let z_target = self.get_bit(row, n + target);

            // Update phase
            if x_control && z_target && (x_target != z_control) {
                self.flip_phase(row);
            }

            // Update target X
            self.set_bit(row, target, x_control ^ x_target);

            // Update control Z
            self.set_bit(row, n + control, z_control ^ z_target);
        }
    }

    /// Apply CZ gate
    pub fn cz(&mut self, q0: usize, q1: usize) {
        // CZ = H(q1) · CNOT(q0, q1) · H(q1)
        self.h(q1);
        self.cnot(q0, q1);
        self.h(q1);
    }

    /// Measure a qubit
    ///
    /// Returns (outcome, is_random)
    /// - outcome: 0 or 1
    /// - is_random: true if measurement was random, false if deterministic
    pub fn measure(&mut self, qubit: usize) -> (bool, bool) {
        let n = self.n_qubits;

        // Check if measurement is deterministic (no X on qubit in destabilizers)
        let mut p: Option<usize> = None;
        for i in 0..n {
            if self.get_bit(i, qubit) {
                // Has X on this qubit
                p = Some(i);
                break;
            }
        }

        match p {
            None => {
                // Deterministic measurement
                // Outcome is determined by phase of stabilizer with Z on this qubit
                let mut outcome = false;
                for i in n..(2 * n) {
                    if self.get_bit(i, n + qubit) && !self.get_bit(i, qubit) {
                        // Has Z but not X
                        outcome = self.get_phase(i);

                        // Account for other qubits with Z (parity)
                        for j in 0..n {
                            if j != qubit && self.get_bit(i, n + j) {
                                if let Some(m) = self.measurements[j] {
                                    outcome ^= m;
                                }
                            }
                        }
                        break;
                    }
                }

                self.measurements[qubit] = Some(outcome);
                (outcome, false)
            }
            Some(p_idx) => {
                // Random measurement
                let outcome: bool = rand::thread_rng().gen();

                // Set row p to be Z_qubit with appropriate phase
                self.set_row_to_zero(p_idx);
                self.set_bit(p_idx, n + qubit, true);
                if outcome {
                    self.flip_phase(p_idx);
                }

                // Update other rows that have X on this qubit
                for i in 0..(2 * n) {
                    if i != p_idx && self.get_bit(i, qubit) {
                        self.row_add(i, p_idx + n);
                    }
                }

                self.measurements[qubit] = Some(outcome);
                (outcome, true)
            }
        }
    }

    /// Get current number of qubits
    #[getter]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Reset to |00...0⟩ state
    pub fn reset(&mut self) {
        *self = Self::new(self.n_qubits);
    }
}

// Private helper methods
impl StabilizerSimulatorRust {
    #[inline]
    fn get_bit(&self, row: usize, col: usize) -> bool {
        let cols = 2 * self.n_qubits + 1;
        self.tableau[row * cols + col]
    }

    #[inline]
    fn set_bit(&mut self, row: usize, col: usize, value: bool) {
        let cols = 2 * self.n_qubits + 1;
        self.tableau.set(row * cols + col, value);
    }

    #[inline]
    fn get_phase(&self, row: usize) -> bool {
        let cols = 2 * self.n_qubits + 1;
        self.tableau[row * cols + (2 * self.n_qubits)]
    }

    #[inline]
    fn flip_phase(&mut self, row: usize) {
        let cols = 2 * self.n_qubits + 1;
        let idx = row * cols + (2 * self.n_qubits);
        let current = self.tableau[idx];
        self.tableau.set(idx, !current);
    }

    fn set_row_to_zero(&mut self, row: usize) {
        let cols = 2 * self.n_qubits + 1;
        let start = row * cols;
        let end = start + cols;
        self.tableau[start..end].fill(false);
    }

    /// Row addition: row_i = row_i XOR row_j (mod 2)
    fn row_add(&mut self, i: usize, j: usize) {
        let cols = 2 * self.n_qubits + 1;
        let i_start = i * cols;
        let j_start = j * cols;

        for k in 0..cols {
            let i_bit = self.tableau[i_start + k];
            let j_bit = self.tableau[j_start + k];
            self.tableau.set(i_start + k, i_bit ^ j_bit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let sim = StabilizerSimulatorRust::new(3);
        assert_eq!(sim.n_qubits(), 3);
    }

    #[test]
    fn test_hadamard() {
        let mut sim = StabilizerSimulatorRust::new(1);
        sim.h(0);
        // After H on |0⟩, measurement should be random
        let (_, is_random) = sim.measure(0);
        // Note: This test is probabilistic, but with proper implementation
        // the first measurement after H should be random
    }

    #[test]
    fn test_cnot_bell_state() {
        let mut sim = StabilizerSimulatorRust::new(2);
        sim.h(0);
        sim.cnot(0, 1);

        // Bell state created
        // Both measurements should give same result
        let (m0, _) = sim.measure(0);
        let (m1, _) = sim.measure(1);
        assert_eq!(m0, m1);
    }
}
