#!/usr/bin/env python3
"""
QAOA MaxCut on a grid graph using tensor-network simulation.
Supports adaptive or ML-guided truncation via tn_core.mps_apply_2q.
"""

import math
import time
import sys
import numpy as np
import torch
from pathlib import Path
import argparse

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.quantum_hybrid_system.tools_qih import tn_core


def _qh_load_ml_predictor(model_path: str, device: str):
    """Load either the original RankPredictorWrapper or the fine-tuned/compiled wrapper.
    Returns None if model_path is falsy."""
    if not model_path:
        return None
    try:
        # Prefer fine-tuned wrapper if name suggests it (ft/compiled)
        if any(tok in str(model_path) for tok in ("compiled", "_ft", "ft")):
            from src.quantum_hybrid_system.tools_qih.finetuned_predictor import FinetunedPredictorWrapper
            return FinetunedPredictorWrapper(model_path=str(model_path), device=device)
        from src.quantum_hybrid_system.tools_qih.ai_rank_predictor import RankPredictorWrapper
        return RankPredictorWrapper(model_path=str(model_path), device=device)
    except Exception:
        # Fallback in reverse order
        try:
            from src.quantum_hybrid_system.tools_qih.ai_rank_predictor import RankPredictorWrapper
            return RankPredictorWrapper(model_path=str(model_path), device=device)
        except Exception:
            from src.quantum_hybrid_system.tools_qih.finetuned_predictor import FinetunedPredictorWrapper
            return FinetunedPredictorWrapper(model_path=str(model_path), device=device)


# ---------------- Gates ----------------
def rzz_gate(theta, device, dtype):
    """
    RZZ(θ) = exp(-i θ/2 Z⊗Z) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2})
    Returns a (4,4) complex matrix on the given device/dtype.
    """
    cdtype = dtype  # complex dtype (torch.complex64 or complex128)
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
    phi = torch.tensor(0.5 * float(theta), device=device, dtype=rdtype)
    e_minus = torch.exp((-1j * phi).to(cdtype))
    e_plus  = torch.exp(( 1j * phi).to(cdtype))
    return torch.diag(torch.stack([e_minus, e_plus, e_plus, e_minus]))


def rx_gate(theta, device, dtype):
    """RX(θ) = exp(-i θ/2 X)"""
    I = torch.eye(2, device=device, dtype=dtype)
    X = torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype)
    return torch.matrix_exp(-0.5j * theta * X)


# ---------------- SWAP routing helpers ----------------
def _qh_swap_gate(device, dtype):
    """2-qubit SWAP gate"""
    SW = torch.zeros((4, 4), dtype=dtype, device=device)
    SW[0, 0] = SW[3, 3] = 1
    SW[1, 2] = SW[2, 1] = 1
    return SW


def _qh_apply_swap_neighbors(cores, k, device, dtype, chi_max, svd_driver, adaptive, tol, ai):
    """Apply SWAP between adjacent qubits k and k+1"""
    SW = _qh_swap_gate(device, dtype)
    return tn_core.mps_apply_2q(cores, k, k+1, SW,
        chi_max=chi_max, svd_driver=svd_driver,
        adaptive=adaptive, tol=tol, ai_predictor=ai)


def _qh_apply_2q_any(cores, i, j, U, device, dtype, chi_max, svd_driver, adaptive, tol, ai):
    """Apply 2-qubit gate between arbitrary qubits i,j using SWAP routing"""
    if j < i:
        i, j = j, i
    peak, total_err = 1, 0.0
    # Bubble j left to i+1
    for k in range(j-1, i, -1):
        cores, chi, err = _qh_apply_swap_neighbors(cores, k-1, device, dtype, chi_max, svd_driver, adaptive, tol, ai)
        peak = max(peak, chi)
        total_err += float(err or 0.0)
    # Apply gate at (i, i+1)
    cores, chi, err = tn_core.mps_apply_2q(
        cores, i, i+1, U, chi_max=chi_max, svd_driver=svd_driver,
        adaptive=adaptive, tol=tol, ai_predictor=ai)
    peak = max(peak, chi)
    total_err += float(err or 0.0)
    # Bubble back to restore order
    for k in range(i+1, j):
        cores, chi, err = _qh_apply_swap_neighbors(cores, k, device, dtype, chi_max, svd_driver, adaptive, tol, ai)
        peak = max(peak, chi)
        total_err += float(err or 0.0)
    return cores, peak, total_err


# ---------------- Layers ----------------
def apply_cost_layer(cores, edges, gamma, chi_max, svd_driver, adaptive, tol, ai):
    """Apply cost layer for all edges"""
    device, dtype = cores[0].device, cores[0].dtype
    peak, total_err = 1, 0.0
    U = rzz_gate(gamma, device, dtype)
    for (i, j) in edges:
        cores, pk, e = _qh_apply_2q_any(cores, i, j, U, device, dtype, chi_max, svd_driver, adaptive, tol, ai)
        peak = max(peak, pk)
        total_err += e
    return cores, peak, total_err


def apply_mixer_layer(cores, beta, device, dtype):
    """Apply mixer layer (RX rotations on all qubits)"""
    RX = rx_gate(2*beta, device, dtype)
    for q in range(len(cores)):
        cores = tn_core.mps_apply_1q(cores, q, RX)
    return cores


# ---------------- Sampling ----------------
def sweep_sample(cores, shots=256):
    """Sample from MPS using sweep sampling"""
    n = len(cores)
    device, dtype = cores[0].device, cores[0].dtype
    out = np.empty((shots, n), dtype=np.int8)
    with torch.no_grad():
        for t in range(shots):
            L = torch.ones((1, 1), device=device, dtype=dtype)
            bits = []
            for q in range(n):
                A = cores[q]
                M0, M1 = A[:, 0, :], A[:, 1, :]
                p0 = (M0.conj().T @ L @ M0).real.sum().item()
                p1 = (M1.conj().T @ L @ M1).real.sum().item()
                p0 = max(p0, 0)
                p1 = max(p1, 0)
                norm = p0 + p1
                if norm <= 0:
                    p0, p1 = 0.5, 0.5
                else:
                    p0 /= norm
                    p1 /= norm
                b = np.random.choice([0, 1], p=[p0, p1])
                bits.append(b)
                M = M0 if b == 0 else M1
                L = (M.conj().T @ L @ M) / (p0 if b == 0 else p1 + 1e-12)
            out[t] = bits
    return out


def estimate_cut_from_samples(samples, edges):
    """Estimate cut value from samples"""
    arr = np.asarray(samples, dtype=np.int8)
    cuts = np.zeros((arr.shape[0],), dtype=np.float32)
    for (i, j) in edges:
        cuts += (arr[:, i] ^ arr[:, j]).astype(np.float32)
    mean = float(cuts.mean())
    std = float(cuts.std(ddof=1)) if arr.shape[0] > 1 else 0.0
    return mean, std


# ---------------- Main ----------------
def run(args):
    """Main QAOA execution"""
    n = args.rows * args.cols
    edges = []
    for r in range(args.rows):
        for c in range(args.cols):
            i = r * args.cols + c
            if c + 1 < args.cols:
                edges.append((i, i + 1))
            if r + 1 < args.rows:
                edges.append((i, i + args.cols))
    
    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    dtype = torch.complex64 if args.dtype == 'c64' else torch.complex128
    
    # ML predictor object (None unless --ai-compression)
    ai = None
    if args.ai_compression and args.ai_model:
        try:
            ai = _qh_load_ml_predictor(args.ai_model, device)
            if args.verbose:
                print(f"✅ Loaded ML predictor from {args.ai_model}")
        except Exception as e:
            print(f'⚠️  ML predictor load failed: {e}')

    # Initialize state
    cores = tn_core.mps_init_plus(n, device=device, dtype=dtype)
    
    # Default parameter schedules
    if args.gamma is not None:
        gammas = [args.gamma] * args.p
    else:
        gammas = [0.6 * (1 + k / args.p) for k in range(args.p)]
    
    if args.beta is not None:
        betas = [args.beta] * args.p
    else:
        betas = [0.8 * (1 - k / args.p) for k in range(args.p)]

    t0 = time.time()
    total_err, peak_chi = 0.0, 1
    
    for layer in range(args.p):
        # Cost layer
        cores, pk, e = apply_cost_layer(cores, edges, gammas[layer],
            args.chi, args.svd_driver, args.adaptive, args.tol, ai)
        peak_chi = max(peak_chi, pk)
        total_err += e
        
        # Mixer layer
        cores = apply_mixer_layer(cores, betas[layer], device, dtype)
        
        if args.verbose:
            print(f"  layer {layer+1}/{args.p}: χ_peak={pk}, cum_trunc_err={total_err:.2e}")

    elapsed = time.time() - t0
    
    # Sample and estimate cut
    samples = sweep_sample(cores, shots=args.shots)
    mean_cut, std_cut = estimate_cut_from_samples(samples, edges)
    
    print("\n================ RESULT ================")
    print(f"Graph: {args.rows}x{args.cols} (n={n}, |E|={len(edges)})")
    print(f"QAOA depth p={args.p}   (svd={args.svd_driver}, χ_max={args.chi}, tol={args.tol}, adaptive={args.adaptive}, ML={bool(ai)})")
    print(f"Time: {elapsed:.2f}s   Peak χ: {peak_chi}   Sum trunc. err: {total_err:.3e}")
    print(f"Estimated cut: {mean_cut:.1f} / {len(edges)} (ratio={mean_cut/len(edges):.3f}, ±{std_cut:.1f})")
    print("========================================")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--chi", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--svd-driver", type=str, default="gesvda", choices=["gesvda", "gesvdj", "gesvd"])
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="c64", choices=["c64", "c128"])
    ap.add_argument("--verbose", action="store_true")
    # AI options
    ap.add_argument("--ai-compression", action="store_true")
    ap.add_argument("--ai-model", type=str, default="")
    args = ap.parse_args()
    run(args)
