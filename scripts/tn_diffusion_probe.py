import torch
try:
    torch.backends.cuda.matmul.allow_tf32=True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

#!/usr/bin/env python3
import argparse, os, time, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Import your simulator’s tensor-network core ---
from atlas_q.tools_qih.tn_core import (
    mps_init_plus, mps_apply_2q, build_entangler
)

def _assert_device_dtype(t, device, dtype):
    if t.device.type != device:
        t = t.to(device)
    if t.dtype != dtype:
        t = t.to(dtype)
    return t

def _blend_entangler(alpha, theta, device, dtype):
    """
    Blend a structured CZ entangler with a random SU(4) and re-orthonormalize.
    U_alpha = (1-alpha)*Ucz + alpha*Usu, then QR -> unitary-like.
    """
    Ucz = build_entangler('cz', theta, device, dtype)
    Usu = build_entangler('su4', theta, device, dtype)
    Ucz = _assert_device_dtype(Ucz, device, dtype)
    Usu = _assert_device_dtype(Usu, device, dtype)
    U = (1.0 - alpha) * Ucz + alpha * Usu
    # Re-orthonormalize the linear blend
    Q, _ = torch.linalg.qr(U)
    return Q

def _brickwork_pairs(nsites, layer_parity):
    """
    Return list of (i, j) pairs for brickwork pattern.
    layer_parity = 0 -> even bonds (0,1),(2,3),...
    layer_parity = 1 -> odd bonds (1,2),(3,4),...
    """
    start = 0 if layer_parity == 0 else 1
    pairs = []
    # Stop one before the end so j=i+1 is in range
    for i in range(start, nsites - 1, 2):
        pairs.append((i, i + 1))
    return pairs

def _safe_writer(csv_path):
    """
    Return (writer, fh). If csv_path is None, (None, None).
    If file doesn't exist, write header. We append rows with an 'alpha' column.
    """
    if csv_path is None:
        return None, None

    new_file = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    fieldnames = ["alpha", "layer", "chi_mean", "chi_max", "trunc_error_sum"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        w.writeheader()
    return w, f

def quantum_diffusion_probe(rows=8, cols=8, depth=20,
                            chi=1024, alpha=0.5, tol=1e-4,
                            adaptive=True, device="cuda", dtype=torch.complex64,
                            verbose=False, csv_path=None):
    """
    Run one AQED experiment at fixed alpha.
    Returns: (chi_mean, chi_max, trunc_sum, D_E) where each of the first three are arrays of length 'depth',
             and D_E is a scalar proxy from the growth rate of Var[χ(layer)].
    """
    nqubits = rows * cols
    cores = mps_init_plus(nqubits, device=device, dtype=dtype)

    # CSV writer (optional)
    writer, fh = _safe_writer(csv_path)

    chi_mean_hist, chi_max_hist, trunc_sum_hist = [], [], []
    var_hist = []

    # Main layers
    for layer in range(depth):
        # A small random jitter of theta promotes mixing but stays reproducible-ish if you set seeds outside.
        theta = 0.05 + 0.05 * np.random.rand()
        Ublend = _blend_entangler(alpha=alpha, theta=float(theta), device=device, dtype=dtype)

        # Even/odd brickwork alternation
        pairs = _brickwork_pairs(nqubits, layer_parity=layer % 2)

        local_chi, local_err = [], []

        for (i, j) in pairs:
            try:
                out = mps_apply_2q(
                    cores, i, j, Ublend,
                    chi_max=chi, tol=tol, adaptive=adaptive
                )
                # Expected return signature: (cores_updated, chi_val, trunc_error)
                if isinstance(out, tuple) and len(out) == 3:
                    cores, chi_val, err_val = out
                else:
                    # Some older variants might return (cores, chi) or just cores; be defensive.
                    if isinstance(out, tuple) and len(out) >= 2:
                        cores, chi_val = out[0], out[1]
                        err_val = 0.0
                    else:
                        cores, chi_val, err_val = out, 0, 0.0
                local_chi.append(float(chi_val))
                local_err.append(float(err_val))
            except Exception as e:
                # Keep going if a particular pair fails
                if verbose:
                    print(f"[warn] skipping pair ({i},{j}) layer={layer}: {e}")
                continue

        # If no gates ran this layer (shouldn't happen for n>2), record zeros
        if len(local_chi) == 0:
            m, M, S = 0.0, 0.0, 0.0
            v = 0.0
        else:
            m = float(np.mean(local_chi))
            M = float(np.max(local_chi))
            S = float(np.sum(local_err))
            v = float(np.var(local_chi))

        chi_mean_hist.append(m)
        chi_max_hist.append(M)
        trunc_sum_hist.append(S)
        var_hist.append(v)

        if verbose:
            print(f"Layer {layer:02d}: mean χ={m:.2f}, max χ={M:.2f}")

        # Per-layer CSV row
        if writer is not None:
            writer.writerow({
                "alpha": alpha, "layer": layer,
                "chi_mean": m, "chi_max": M, "trunc_error_sum": S
            })

    # Close CSV handle if opened
    if fh is not None:
        fh.close()

    # Diffusion proxy: slope of variance vs time (finite difference / linear fit)
    # Ignore early burn-in if desired; for simplicity we use all layers here.
    t = np.arange(len(var_hist), dtype=float)
    if len(t) >= 2:
        A = np.vstack([t, np.ones_like(t)]).T
        # Least squares fit var ~ a*t + b
        a, b = np.linalg.lstsq(A, np.array(var_hist), rcond=None)[0]
        D_E = float(a)
    else:
        D_E = 0.0

    return (np.array(chi_mean_hist),
            np.array(chi_max_hist),
            np.array(trunc_sum_hist),
            D_E)

def main():
    p = argparse.ArgumentParser(description="AQED: Adaptive Quantum Entanglement Diffusion probe")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--chi", type=int, default=1024)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--adaptive", action="store_true", help="enable adaptive truncation in TN core")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="complex64", choices=["complex64", "complex128"])
    p.add_argument("--save", type=str, default=None, help="save plot to this path (PNG)")
    p.add_argument("--csv", type=str, default=None, help="append per-layer rows to this CSV")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dtype = torch.complex128 if args.dtype == "complex128" else torch.complex64
    alphas = [float(a) for a in args.alphas.split(",")]

    # Plot setup
    plt.figure(figsize=(7, 5))
    results = []

    for a in alphas:
        print(f"\nRunning α={a:.2f} ...")
        t0 = time.time()
        chi_mean, chi_max, trunc_sum, D_E = quantum_diffusion_probe(
            rows=args.rows, cols=args.cols, depth=args.depth,
            chi=args.chi, alpha=a, tol=args.tol,
            adaptive=args.adaptive, device=args.device, dtype=dtype,
            verbose=args.verbose, csv_path=args.csv
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  ⟨χ⟩_final={chi_mean[-1]:.1f}")
        print(f"  Entanglement diffusion proxy D_E ≈ {D_E:.3e}")

        results.append((a, chi_mean, D_E))
        plt.plot(chi_mean, label=f"α={a:.2f}")

    plt.xlabel("Layer t")
    plt.ylabel("Mean bond dimension ⟨χ⟩")
    plt.title("AQED: Entanglement Diffusion vs Chaos Index α")
    plt.legend()
    plt.tight_layout()
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=200)
    plt.show()

    # Summary
    print("\n=== Diffusion Summary ===")
    for a, chi_curve, D_E in results:
        print(f"α={a:.2f}  D_E≈{D_E:.3e}  χ_final={chi_curve[-1]:.1f}")

if __name__ == "__main__":
    main()
