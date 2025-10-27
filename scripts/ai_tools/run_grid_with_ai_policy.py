#!/usr/bin/env python
import os, sys, argparse, time
from pathlib import Path

# --- Import the base runner (your existing script) ---
sys.path.insert(0, str(Path(__file__).parent))
import tn_grid8x8_shallow as base

# Optional adapter for clamp variants
from atlas_q.tools_qih.ai_rank_policy import CalibratedPolicyAdapter, PolicyConfig
from atlas_q.tools_qih.ai_rank_predictor import RankPredictorWrapper

def parse_args():
    p = argparse.ArgumentParser(description="Run grid TN with switchable AI policy")
    # Core sim args (keep names aligned with tn_grid8x8_shallow)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--chi", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["c64","c128"], default="c64")
    p.add_argument("--svd-driver", type=str, choices=["gesvdj","gesvda","gesvd"], default="gesvda")
    p.add_argument("--entangler", type=str, choices=["cz","rzz","cnot","none","su4"], default="cz")
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--ai-compression", action="store_true")
    p.add_argument("--ai-model", type=str, default=str(Path(__file__).parent.parent / "models" / "rank_predictor.pt"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    # Policy flags
    p.add_argument("--ai-policy", type=str, default="guardrail",
                   choices=["baseline","guardrail","blend","clamp","clamp+blend"],
                   help="AI rank policy")
    p.add_argument("--ai-alpha", type=float, default=0.7,
                   help="Blend factor for 'blend' or 'clamp+blend' (0..1)")
    p.add_argument("--ai-fmin", type=float, default=0.50, help="Min keep fraction for clamp* policies")
    p.add_argument("--ai-fmax", type=float, default=0.995, help="Max keep fraction for clamp* policies")
    p.add_argument("--ai-log", action="store_true", help="Verbose per-layer AI rank logs")
    return p.parse_args()

def maybe_build_external_predictor(args):
    """
    If the policy needs clamping, we wrap the base predictor with CalibratedPolicyAdapter
    and inject it into tn_grid8x8_shallow via its _EXTERNAL_AI_PREDICTOR hook.
    """
    if not args.ai_compression:
        return None

    needs_clamp = args.ai_policy in ("clamp","clamp+blend")
    if not needs_clamp:
        return None

    rp = RankPredictorWrapper(model_path=args.ai_model, device=args.device)
    cfg = PolicyConfig(f_min=args.ai_fmin, f_max=args.ai_fmax, chi_budget=args.chi)
    adapter = CalibratedPolicyAdapter(rp, cfg, chi_max=args.chi)
    return adapter

def main():
    args = parse_args()

    # Normalize alpha
    if args.ai_alpha < 0.0: args.ai_alpha = 0.0
    if args.ai_alpha > 1.0: args.ai_alpha = 1.0

    # Option A: tell tn_core which policy to use (your earlier env-var patch)
    os.environ["QIH_AI_POLICY"] = args.ai_policy
    os.environ["QIH_AI_ALPHA"]  = str(args.ai_alpha)
    if args.ai_log:
        os.environ["QIH_AI_LOG"] = "1"

    # Option B: inject external predictor when clamping is requested
    ext_pred = maybe_build_external_predictor(args)
    if ext_pred is not None:
        # tn_grid8x8_shallow reads this global if present (we added a hook earlier)
        base._EXTERNAL_AI_PREDICTOR = ext_pred
        if args.verbose:
            print("✅ Using external AI predictor (CalibratedPolicyAdapter)")

    # Build a minimal argv for base.main() (it parses sys.argv itself)
    argv = [
        "tn_grid8x8_shallow.py",
        "--rows", str(args.rows),
        "--cols", str(args.cols),
        "--depth", str(args.depth),
        "--chi", str(args.chi),
        "--device", args.device,
        "--svd-driver", args.svd_driver,
        "--entangler", args.entangler,
        "--tol", str(args.tol),
    ]
    if args.dtype == "c128":
        argv += ["--dtype", "c128"]
    if args.adaptive:
        argv += ["--adaptive"]
    if args.ai_compression:
        argv += ["--ai-compression"]
        argv += ["--ai-model", args.ai_model]
    if args.verbose:
        argv += ["--verbose"]

    # Hand off to the existing entry point
    sys.argv = argv
    return base.main()

if __name__ == "__main__":
    raise SystemExit(main())
