import argparse
import time
import sys
from pathlib import Path

# Direct import to avoid __init__.py issues
import torch
import numpy as np

# Import tn_core directly
tn_core_path = Path(__file__).parent.parent / 'src' / 'atlas_q' / 'tools_qih' / 'tn_core.py'
import importlib.util
spec = importlib.util.spec_from_file_location("tn_core", tn_core_path)
tn_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tn_core)

# Import ML predictor
ai_predictor_path = Path(__file__).parent.parent / 'src' / 'atlas_q' / 'tools_qih' / 'ai_rank_predictor.py'
spec_ai = importlib.util.spec_from_file_location("ai_rank_predictor", ai_predictor_path)
ai_predictor_module = importlib.util.module_from_spec(spec_ai)
spec_ai.loader.exec_module(ai_predictor_module)

# Now we can use the functions
svd_robust = tn_core.svd_robust
build_entangler = tn_core.build_entangler
mps_init_plus = tn_core.mps_init_plus
mps_apply_1q = tn_core.mps_apply_1q
mps_apply_2q = tn_core.mps_apply_2q
RankPredictorWrapper = ai_predictor_module.RankPredictorWrapper

# Fine-tuned predictor loader
from pathlib import Path as _PTH
_finetuned_path = _PTH(__file__).parent.parent / 'src' / 'atlas_q' / 'tools_qih' / 'finetuned_predictor.py'
_spec_ft = importlib.util.spec_from_file_location("finetuned_predictor", _finetuned_path)
_mod_ft = importlib.util.module_from_spec(_spec_ft)
_spec_ft.loader.exec_module(_mod_ft)
FinetunedPredictorWrapper = _mod_ft.FinetunedPredictorWrapper

def main():
    import random, numpy as _np, torch as _torch
    try:
        args  # noqa
    except NameError:
        pass
    # 'args' defined below; seed after parse

    ap = argparse.ArgumentParser(description="Quantum-inspired tensor network simulator with ML compression")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4, help="number of brickwork layers")
    ap.add_argument("--chi", type=int, default=256, help="maximum bond dimension")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="c64", choices=["c64", "c128"])
    ap.add_argument("--svd-driver", choices=["gesvdj", "gesvda", "gesvd"], default="gesvda",
                    help="SVD driver: gesvdj (accurate), gesvda (fast approximate), gesvd (default)")
    ap.add_argument("--entangler", default="cz", choices=["cz", "rzz", "cnot", "none", "su4"],
                    help="Entangling gate: cz, rzz, cnot, su4 (random), or none")
    ap.add_argument("--adaptive", action="store_true",
                    help="Enable adaptive bond dimension truncation")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Adaptive truncation tolerance")
    ap.add_argument("--ai-compression", action="store_true",
                    help="Use ML predictor for rank selection")
    ap.add_argument("--ai-model", type=str, default="models/rank_predictor.pt",
                    help="Path to trained ML rank predictor model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed per-layer statistics")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.complex64 if args.dtype == "c64" else torch.complex128
    n = args.rows * args.cols

    # Load ML predictor if requested
    ai_predictor = None
    if args.ai_compression:
        model_path = Path(args.ai_model) if hasattr(args, 'ai_model') and args.ai_model else Path(args.ai_model)
        if not model_path.exists():
            model_path = Path("models/rank_predictor.pt")

        if model_path.exists():
            try:
                # Check if it's a fine-tuned model (simpler architecture)
                if 'ft' in str(model_path).lower() or '_ft' in str(model_path):
                    print(f"Loading FINE-TUNED model from {model_path}")
                    ai_predictor = FinetunedPredictorWrapper(model_path=str(model_path), device=args.device)
                    print(f"✅ Loaded FINE-TUNED ML rank predictor")
                else:
                    print(f"Loading standard model from {model_path}")
                    ai_predictor = RankPredictorWrapper(model_path=str(model_path), device=args.device)
                    print(f"✅ Loaded ML rank predictor")
            except Exception as e:
                print(f"⚠️  Failed to load ML predictor: {e}")
                print("   Falling back to standard adaptive truncation")
        else:
            print(f"⚠️  ML model not found at {model_path}")
            print("   Run: python scripts/train_rank_predictor.py")
            print("   Falling back to standard adaptive truncation")

    # Linearize 2D grid in snake order (keeps many nearest-neighbors adjacent in 1D)
    def snake_index(r, c):
        return r * args.cols + (c if r % 2 == 0 else (args.cols - 1 - c))

    pairs_even = []
    pairs_odd = []

    # Build horizontal neighbors first (more likely to be adjacent in snake)
    for r in range(args.rows):
        for c in range(args.cols - 1):
            i = snake_index(r, c)
            j = snake_index(r, c + 1)
            if min(i, j) % 2 == 0:
                pairs_even.append((min(i, j), max(i, j)))
            else:
                pairs_odd.append((min(i, j), max(i, j)))

    # Add vertical couplings that are adjacent in snake ordering
    for r in range(args.rows - 1):
        for c in range(args.cols):
            i = snake_index(r, c)
            j = snake_index(r + 1, c)
            if abs(i - j) == 1:
                if min(i, j) % 2 == 0:
                    pairs_even.append((min(i, j), max(i, j)))
                else:
                    pairs_odd.append((min(i, j), max(i, j)))

    # Initialize MPS
    cores = mps_init_plus(n, device=args.device, dtype=dtype)
    peak_chi = 1
    total_truncation_error = 0.0
    t0 = time.time()

    compression_mode = "ML" if ai_predictor else ("adaptive" if args.adaptive else "fixed")
    print(f"\n{'='*70}")
    print(f"Quantum-Inspired Tensor Network Simulator")
    print(f"{'='*70}")
    print(f"Grid: {args.rows}×{args.cols} ({n} qubits)")
    print(f"Depth: {args.depth} layers")
    print(f"Max χ: {args.chi}")
    print(f"Entangler: {args.entangler}")
    print(f"SVD driver: {args.svd_driver}")
    print(f"Compression: {compression_mode}")
    if args.adaptive:
        print(f"Tolerance: {args.tol}")
    print(f"{'='*70}\n")

    for layer in range(args.depth):
        theta = 0.3 + 0.1 * layer
        U2 = build_entangler(args.entangler, theta, args.device, dtype)

        layer_truncation_error = 0.0
        layer_gates = 0

        for i, j in pairs_even:
            cores, chi, trunc_err = mps_apply_2q(
                cores, i, j, U2, chi_max=args.chi,
                svd_driver=args.svd_driver,
                adaptive=args.adaptive, tol=args.tol,
                ai_predictor=ai_predictor
            )
            peak_chi = max(peak_chi, chi)
            layer_truncation_error += trunc_err
            layer_gates += 1

        for i, j in pairs_odd:
            cores, chi, trunc_err = mps_apply_2q(
                cores, i, j, U2, chi_max=args.chi,
                svd_driver=args.svd_driver,
                adaptive=args.adaptive, tol=args.tol,
                ai_predictor=ai_predictor
            )
            peak_chi = max(peak_chi, chi)
            layer_truncation_error += trunc_err
            layer_gates += 1

        avg_layer_error = layer_truncation_error / layer_gates if layer_gates > 0 else 0.0
        total_truncation_error += layer_truncation_error

        if args.verbose:
            print(f"  Layer {layer + 1}/{args.depth}: χ={peak_chi}, "
                  f"avg_trunc_err={avg_layer_error:.2e}, gates={layer_gates}")
        else:
            print(f"  Layer {layer + 1}/{args.depth}: peak_chi={peak_chi}")

    dt = time.time() - t0
    avg_total_error = total_truncation_error / (args.depth * (len(pairs_even) + len(pairs_odd)))

    print(f"\n{'='*70}")
    print(f"Simulation Complete")
    print(f"{'='*70}")
    print(f"Time: {dt:.2f}s")
    print(f"Peak χ: {peak_chi}")
    print(f"Avg truncation error: {avg_total_error:.2e}")
    print(f"Total gates: {args.depth * (len(pairs_even) + len(pairs_odd))}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
