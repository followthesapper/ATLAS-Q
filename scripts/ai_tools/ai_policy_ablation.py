#!/usr/bin/env python
import subprocess, sys, re, json

POLICIES = ["guardrail","blend","clamp","clamp+blend","baseline"]

BASE_CMD = [
    sys.executable, "scripts/run_grid_with_ai_policy.py",
    "--rows","8","--cols","8","--depth","12","--chi","512",
    "--entangler","cz","--svd-driver","gesvda","--adaptive","--tol","1e-4",
    "--ai-compression"
]

def run_one(policy):
    cmd = BASE_CMD + ["--ai-policy", policy]
    if policy in ("blend","clamp+blend"):
        cmd += ["--ai-alpha","0.7"]
    if policy in ("clamp","clamp+blend"):
        cmd += ["--ai-fmin","0.50","--ai-fmax","0.995"]

    print("\n==> Running:", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr, file=sys.stderr)
        return {"policy": policy, "error": f"exit {out.returncode}"}

    text = out.stdout
    time_m = re.search(r"Time:\s*([\d\.]+)s", text)
    peak_m = re.search(r"Peak\s*χ:\s*(\d+)", text)
    avgerr_m = re.search(r"Avg truncation error:\s*([0-9.eE+-]+)", text)

    return {
        "policy": policy,
        "time_s": float(time_m.group(1)) if time_m else None,
        "peak_chi": int(peak_m.group(1)) if peak_m else None,
        "avg_err": float(avgerr_m.group(1)) if avgerr_m else None
    }

def main():
    results = [run_one(p) for p in POLICIES]

    print("\n=== AI Policy Ablation (8x8, depth=12, chi=512) ===")
    print(f"{'Policy':<12} {'Time (s)':>10} {'Peak χ':>8} {'Avg Err':>12}")
    print("-"*46)
    for r in results:
        if "error" in r:
            print(f"{r['policy']:<12} {'ERR':>10} {'-':>8} {r['error']:>12}")
        else:
            t = f"{r['time_s']:.2f}" if r['time_s'] is not None else "-"
            p = f"{r['peak_chi']}" if r['peak_chi'] is not None else "-"
            e = f"{r['avg_err']:.3e}" if r['avg_err'] is not None else "-"
            print(f"{r['policy']:<12} {t:>10} {p:>8} {e:>12}")

    # Also dump JSON (handy for plotting later)
    print("\nJSON:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
