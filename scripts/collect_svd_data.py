#!/usr/bin/env python
import os, subprocess, itertools, sys
from pathlib import Path

ROWS = [8,10,12]
COLS = [8,10,12]
DEPTHS = [8,12,15]
ENTANGLERS = ["cz","cnot","rzz"]
TOLS = [1e-3,5e-4,1e-4]

def run_one(r,c,d,e,tol, logdir, seed=123):
    env = os.environ.copy()
    env["QIH_SVD_LOG_DIR"] = str(Path(logdir).resolve())
    env["QIH_ENTANGLER"] = e
    cmd = [
        sys.executable, "scripts/tn_grid8x8_shallow.py",
        "--rows", str(r), "--cols", str(c), "--depth", str(d),
        "--chi", "1024", "--entangler", e, "--svd-driver", "gesvda",
        "--adaptive", "--tol", str(tol), "--seed", str(seed),
    ]
    print("==>"," ".join(cmd))
    return subprocess.run(cmd, env=env).returncode

def main():
    out = Path("runs/svd_logs"); out.mkdir(parents=True, exist_ok=True)
    for r,c,d,e,t in itertools.product(ROWS, COLS, DEPTHS, ENTANGLERS, TOLS):
        rc = run_one(r,c,d,e,t, out)
        if rc != 0:
            print(f"!! exit {rc} on ({r},{c},d={d},{e},tol={t})", file=sys.stderr)

if __name__ == "__main__":
    main()
