"""
Variational Quantum Eigensolver (VQE) and QAOA

Implements variational algorithms for optimization and ground state finding:
- VQE for chemistry and physics
- QAOA for combinatorial optimization
- Hardware-efficient ansätze
- Parameter optimization

Author: ATLAS-Q Contributors
Date: October 2025
License: MIT
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. VQE/QAOA optimization will be limited.")

# Matplotlib headless for saving plots
import matplotlib

from .adaptive_mps import AdaptiveMPS
from .mpo_ops import MPO, expectation_value

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class VQEConfig:
    """Configuration for VQE"""
    ansatz: str = "hardware_efficient"  # 'hardware_efficient', 'uccsd', 'custom'
    n_layers: int = 3
    optimizer: str = "L-BFGS-B"         # 'COBYLA', 'L-BFGS-B', 'BFGS', 'Adam' (SciPy)
    max_iter: int = 200
    tol: float = 1e-9
    chi_max: int = 256
    gradient_method: str = "group"      # 'group', 'per_pauli', None (defaults to 'group' for best performance)
    device: str = "cuda"
    dtype: torch.dtype = torch.complex128  # complex128 for precision


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz for VQE.

    Layer = [Ry(θ) on all qubits] + [CZ on neighboring pairs]
    """

    def __init__(self, n_qubits: int, n_layers: int, device: str = "cuda", dtype: torch.dtype = torch.complex128):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        self.n_params = n_qubits * n_layers

    def apply(self, mps: AdaptiveMPS, params: np.ndarray):
        assert len(params) == self.n_params
        param_idx = 0
        for _ in range(self.n_layers):
            # single-qubit RY
            for q in range(self.n_qubits):
                theta = params[param_idx]; param_idx += 1
                mps.apply_single_qubit_gate(q, self._ry_gate(theta))
            # CZ entanglers (even then odd pairs)
            CZ = torch.diag(torch.tensor([1, 1, 1, -1], dtype=self.dtype, device=self.device))
            for q in range(0, self.n_qubits - 1, 2):
                mps.apply_two_site_gate(q, CZ)
            for q in range(1, self.n_qubits - 1, 2):
                mps.apply_two_site_gate(q, CZ)

    def _ry_gate(self, theta: float) -> torch.Tensor:
        c = np.cos(theta / 2.0); s = np.sin(theta / 2.0)
        return torch.tensor([[c, -s], [s, c]], dtype=self.dtype, device=self.device)

    def get_param_shift(self, k: int) -> float:
        # Standard shift for Ry parameters
        return np.pi / 2.0

    # For chemistry drivers that expect it
    def prepare_hf_state(self, chi_max: int = 64) -> AdaptiveMPS:
        return AdaptiveMPS(
            num_qubits=self.n_qubits, bond_dim=2, chi_max_per_bond=chi_max,
            device=self.device, dtype=self.dtype
        )


class VQE:
    """
    VQE driver with:
      - Lightweight warm-start (1D/2D)
      - Optional per-Pauli parameter-shift gradients
      - Incremental JSONL progress + PNG plots in molecular_results/
      - Quiet-by-default logging with heartbeat
    """

    def __init__(self, hamiltonian: MPO, config: VQEConfig, custom_ansatz=None, output_dir: Optional[str] = None):
        self.H = hamiltonian
        self.config = config

        # Init ansatz
        if custom_ansatz is not None:
            self.ansatz = custom_ansatz
        elif config.ansatz == "hardware_efficient":
            self.ansatz = HardwareEfficientAnsatz(
                n_qubits=self.H.n_sites, n_layers=config.n_layers, device=config.device, dtype=config.dtype
            )
        elif config.ansatz == "custom":
            self.ansatz = None
        else:
            raise ValueError(f"Unknown ansatz: {config.ansatz}")

        # Tracking
        self.energies: List[float] = []
        self.param_history: List[np.ndarray] = []
        self.iteration: int = 0

        # Behavior flags (tweak from caller)
        self._fci_ref: Optional[float] = None
        self._use_warm_start: bool = True
        self._use_per_pauli_gradients: bool = False

        # Logging/outputs
        self.quiet: bool = True          # minimal console chatter
        self.status_every: int = 10      # heartbeat cadence

        # Warm-start defaults (tuned to be light for >10 params)
        self.ws_points_1d: int = 21
        self.ws_theta_max_1d: float = 1.5
        self.ws_points_2d: int = 11
        self.ws_theta_max_2d: float = 1.5

        # Incremental outputs (only enabled if output_dir explicitly provided)
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self._progress_path: Optional[str] = None
        self._run_tag: Optional[str] = None

    # ---------------- Utilities: minimal logging & progress I/O ----------------

    def _p(self, msg: str):
        if not self.quiet:
            print(msg, flush=True)

    def _start_progress_log(self, label: str):
        if not self.output_dir:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._run_tag = f"{label}_{ts}"
        self._progress_path = os.path.join(self.output_dir, f"{self._run_tag}.jsonl")
        with open(self._progress_path, "a") as f:
            f.write(json.dumps({"event": "run_start", "label": label, "time": time.time()}) + "\n")

    def _write_progress(self, payload: Dict):
        if not self._progress_path:
            return
        row = dict(payload); row.setdefault("time", time.time())
        with open(self._progress_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ---------------- Core evals ----------------

    def _energy_at_params(self, params: np.ndarray) -> float:
        # Prepare HF state if available, otherwise |0…0⟩
        if hasattr(self.ansatz, "prepare_hf_state"):
            mps = self.ansatz.prepare_hf_state(chi_max=self.config.chi_max)
        else:
            mps = AdaptiveMPS(
                num_qubits=self.H.n_sites, bond_dim=2, chi_max_per_bond=self.config.chi_max,
                device=self.config.device, dtype=self.config.dtype,
            )
        # Apply ansatz (UCCSD expects chi_max in apply)
        try:
            self.ansatz.apply(mps, params, chi_max=self.config.chi_max)  # UCCSD signature
        except TypeError:
            self.ansatz.apply(mps, params)                                # HEA signature
        E = expectation_value(self.H, mps)
        return float(E.real)

    def _cost_function(self, params: np.ndarray) -> float:
        E = self._energy_at_params(params)
        self.energies.append(E)
        self.param_history.append(params.copy())
        self.iteration += 1

        if (self.iteration % self.status_every == 0) or (self.iteration == 1):
            self._write_progress({"event": "iter", "iter": self.iteration, "E": E, "best_E": float(min(self.energies))})
            self._p(f"[VQE] iter {self.iteration:4d}  E = {E:.8f}")

        return E

    # --------- Gradients ---------

    def _gradient_at_params_parameter_shift(self, params: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(params)
        for k in range(self.ansatz.n_params):
            s = self.ansatz.get_param_shift(k)
            p_plus = params.copy(); p_plus[k] += s
            p_minus = params.copy(); p_minus[k] -= s
            grad[k] = 0.5 * (self._energy_at_params(p_plus) - self._energy_at_params(p_minus))
        return grad

    def _gradient_at_params_per_pauli_shift(self, params: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(params)
        for k in range(self.ansatz.n_params):
            if not hasattr(self.ansatz, "param_groups"):
                # fallback to group-level shift if ansatz doesn't expose groups
                s = self.ansatz.get_param_shift(k)
                p_plus = params.copy(); p_plus[k] += s
                p_minus = params.copy(); p_minus[k] -= s
                grad[k] = 0.5 * (self._energy_at_params(p_plus) - self._energy_at_params(p_minus))
                continue

            gk = 0.0
            for coeff, _P in self.ansatz.param_groups[k]:
                a = abs(coeff.imag) if abs(coeff.imag) > 1e-14 else abs(coeff.real)
                if a < 1e-14:  # skip zeros
                    continue
                s = np.pi / (2.0 * a)
                p_plus = params.copy(); p_plus[k] += s
                p_minus = params.copy(); p_minus[k] -= s
                gk += a * (self._energy_at_params(p_plus) - self._energy_at_params(p_minus)) / 2.0
            grad[k] = gk
        return grad

    # --------- Warm-start (light) ---------

    def _grid_search_1d(self, k: int, theta_max: float, points: int) -> float:
        thetas = np.linspace(-theta_max, theta_max, points)
        base = np.zeros(self.ansatz.n_params)
        best_E = float("+inf"); best_t = 0.0
        self._write_progress({"event": "warm1d_start", "k": k, "points": points})
        for i, t in enumerate(thetas, 1):
            p = base.copy(); p[k] = t
            E = self._energy_at_params(p)
            if E < best_E:
                best_E, best_t = E, t
            if i % max(1, points // 5) == 0:
                self._write_progress({"event": "warm1d_step", "k": k, "i": i, "E": E})
        self._write_progress({"event": "warm1d_done", "k": k, "theta": best_t, "E": best_E})
        return best_t

    def _grid_search_2d(self, i0: int, i1: int, theta_max: float, points: int) -> np.ndarray:
        grid = np.linspace(-theta_max, theta_max, points)
        base = np.zeros(self.ansatz.n_params)
        best = base.copy(); best_E = float("+inf")
        self._write_progress({"event": "warm2d_start", "i0": i0, "i1": i1, "points": points})
        for t0 in grid:
            row_best = float("+inf")
            for t1 in grid:
                p = base.copy(); p[i0] = t0; p[i1] = t1
                E = self._energy_at_params(p)
                if E < best_E:
                    best_E, best = E, p
                if E < row_best:
                    row_best = E
            self._write_progress({"event": "warm2d_row", "i0_val": float(t0), "row_best_E": row_best})
        self._write_progress({"event": "warm2d_done", "best_E": best_E})
        return best

    # --------- Plotting ---------

    def _plot_convergence(self, label: str):
        if not self.energies or not self.output_dir:
            return
        plt.figure(figsize=(8, 5))
        xs = list(range(1, len(self.energies) + 1))
        plt.plot(xs, self.energies, marker='o', linewidth=1.5)
        plt.xlabel("Iteration"); plt.ylabel("Energy (Hartree)")
        title = f"VQE Convergence: {label}"
        if self._run_tag:
            title += f"  [{self._run_tag}]"
        plt.title(title); plt.grid(alpha=0.3); plt.tight_layout()
        path = os.path.join(self.output_dir, f"{self._run_tag}_convergence.png")
        plt.savefig(path, dpi=200); plt.close()
        self._write_progress({"event": "plot_saved", "path": path})

    # --------- Public: run() ---------

    def run(self, initial_params: Optional[np.ndarray] = None, label: str = "molecule") -> Tuple[float, np.ndarray]:
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for VQE optimization")

        self._start_progress_log(label)
        t0 = time.time()

        # Sanity HF
        zero = np.zeros(self.ansatz.n_params, dtype=float)
        hf_energy = self._energy_at_params(zero)
        self._write_progress({"event": "sanity", "hf_energy": hf_energy})
        self._p(f"[VQE] <HF|H|HF> = {hf_energy:.8f} Ha")

        # Initial params
        if initial_params is None:
            initial_params = np.zeros(self.ansatz.n_params, dtype=float)

        # Warm-start (lightweight)
        if self._use_warm_start:
            n = self.ansatz.n_params
            pts1 = self.ws_points_1d if n > 10 else 41
            tmax1 = self.ws_theta_max_1d
            warm = initial_params.copy()
            self._p(f"[Warm-Start] 1D scans on {min(n, 8)} params ({pts1} pts, ±{tmax1})")
            for k in range(min(n, 8)):
                warm[k] = self._grid_search_1d(k, theta_max=tmax1, points=pts1)

            if n >= 2:
                pts2 = self.ws_points_2d if n > 10 else 21
                tmax2 = self.ws_theta_max_2d
                seed2d = self._grid_search_2d(0, 1, theta_max=tmax2, points=pts2)
                if self._energy_at_params(seed2d) < self._energy_at_params(warm):
                    warm = seed2d
            initial_params = warm
            self._write_progress({"event": "warm_done", "start_E": self._energy_at_params(initial_params)})

        # Optimizer
        method = self.config.optimizer
        options = {"maxiter": self.config.max_iter}
        if method.upper() in {"BFGS", "L-BFGS-B"}:
            options["ftol"] = self.config.tol
            options["gtol"] = self.config.tol

        use_jac = method.upper() in {"BFGS", "L-BFGS-B"}
        extra = {}
        if use_jac:
            if self._use_per_pauli_gradients:
                extra["jac"] = lambda x: self._gradient_at_params_per_pauli_shift(x)
            else:
                extra["jac"] = lambda x: self._gradient_at_params_parameter_shift(x)
        if method.upper() == "L-BFGS-B":
            extra["bounds"] = [(-np.pi, np.pi)] * self.ansatz.n_params

        # Reset traces
        self.iteration = 0
        self.energies = []
        self.param_history = []

        self._p(f"[VQE] Starting {method} with {self.ansatz.n_params} params…")
        self._write_progress({"event": "opt_start", "method": method})

        result = minimize(self._cost_function, initial_params, method=method, options=options, **extra)

        elapsed = time.time() - t0
        self._write_progress({"event": "opt_done", "final_E": float(result.fun), "iters": self.iteration, "elapsed_s": elapsed})
        self._plot_convergence(label)
        self._p(f"VQE converged: E = {result.fun:.6f} after {self.iteration} iterations ({elapsed:.2f}s)")
        return float(result.fun), np.array(result.x, dtype=float)


class QAOAAnsatz:
    """QAOA ansatz for combinatorial optimization"""
    def __init__(self, cost_hamiltonian: MPO, n_layers: int, device: str = "cuda", dtype: torch.dtype = torch.complex128):
        self.H_cost = cost_hamiltonian
        self.n_qubits = cost_hamiltonian.n_sites
        self.n_layers = n_layers
        self.device = device
        self.dtype = dtype
        X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
        ops = [-X] * self.n_qubits
        self.H_mixer = MPO.from_local_ops(ops, device=device)
        self.n_params = 2 * n_layers

    def apply(self, mps: AdaptiveMPS, params: np.ndarray):
        assert len(params) == self.n_params
        for layer in range(self.n_layers):
            gamma = params[2 * layer]; beta = params[2 * layer + 1]
            self._apply_cost_layer(mps, gamma)
            self._apply_mixer_layer(mps, beta)

    def _apply_cost_layer(self, mps: AdaptiveMPS, gamma: float):
        for q in range(self.n_qubits):
            mps.apply_single_qubit_gate(q, self._rz_gate(2 * gamma))

    def _apply_mixer_layer(self, mps: AdaptiveMPS, beta: float):
        for q in range(self.n_qubits):
            mps.apply_single_qubit_gate(q, self._rx_gate(2 * beta))

    def _rz_gate(self, theta: float) -> torch.Tensor:
        return torch.tensor(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=self.dtype, device=self.device,
        )

    def _rx_gate(self, theta: float) -> torch.Tensor:
        c = np.cos(theta / 2); s = np.sin(theta / 2)
        return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=self.dtype, device=self.device)


class QAOA:
    """Quantum Approximate Optimization Algorithm"""
    def __init__(self, cost_hamiltonian: MPO, n_layers: int = 3, optimizer: str = "COBYLA",
                 device: str = "cuda", dtype: torch.dtype = torch.complex128):
        self.H_cost = cost_hamiltonian
        self.n_layers = n_layers
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.ansatz = QAOAAnsatz(cost_hamiltonian, n_layers, device, dtype)
        self.energies = []; self.iteration = 0

    @property
    def n_params(self) -> int:
        return self.ansatz.n_params

    def _cost_function(self, params: np.ndarray) -> float:
        mps = AdaptiveMPS(num_qubits=self.H_cost.n_sites, bond_dim=2, chi_max_per_bond=64,
                          device=self.device, dtype=self.dtype)
        H = torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / np.sqrt(2)
        for q in range(mps.num_qubits):
            mps.apply_single_qubit_gate(q, H)
        self.ansatz.apply(mps, params)
        E = expectation_value(self.H_cost, mps)
        self.energies.append(float(E.real)); self.iteration += 1
        if self.iteration % 10 == 0:
            print(f"QAOA Iter {self.iteration}: Cost = {E.real:.6f}")
        return float(E.real)

    def run(self, initial_params: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy required for QAOA")
        if initial_params is None:
            initial_params = np.random.randn(self.ansatz.n_params) * 0.1
        result = minimize(self._cost_function, initial_params, method=self.optimizer, options={"maxiter": 200})
        print(f"\nQAOA converged: Cost = {result.fun:.6f}")
        return float(result.fun), np.array(result.x, dtype=float)

    def _prepare_initial_state(self) -> AdaptiveMPS:
        """Prepare initial |+⟩^⊗n state for QAOA"""
        mps = AdaptiveMPS(num_qubits=self.H_cost.n_sites, bond_dim=2, chi_max_per_bond=64,
                          device=self.device, dtype=self.dtype)
        # Apply Hadamard to all qubits to create |+⟩ state
        H = torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / np.sqrt(2)
        for q in range(mps.num_qubits):
            mps.apply_single_qubit_gate(q, H)
        return mps

    def _build_mixer_hamiltonian(self) -> MPO:
        """Build the QAOA mixer Hamiltonian (X mixer)"""
        return self.ansatz.H_mixer


# Chemistry-specific utilities

def build_molecular_hamiltonian(
    h1: np.ndarray, h2: np.ndarray, mapping: str = "jordan_wigner", device: str = "cuda"
) -> MPO:
    """
    Build molecular Hamiltonian MPO from 1- and 2-electron integrals

    Args:
        h1: One-electron integrals [n_orb, n_orb]
        h2: Two-electron integrals [n_orb, n_orb, n_orb, n_orb]
        mapping: 'jordan_wigner' or 'bravyi_kitaev'
        device: torch device

    Returns:
        MPO representation of electronic Hamiltonian
    """
    # Placeholder - proper implementation requires:
    # 1. Second quantization operators
    # 2. Fermion-to-qubit mapping
    # 3. Pauli string collection
    # 4. MPO compression

    n_qubits = h1.shape[0] * 2  # Spin-orbitals

    # For now, return identity
    return MPO.identity(n_qubits, device=device)
