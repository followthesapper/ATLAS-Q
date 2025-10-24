#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot benchmark:
- Runs a FAST AQED diffusion probe (small 6x6xdepth) to produce controller CSV
- Trains baseline transformer
- Trains AQED-fixed (low & med) mixers
- Trains AQED-controller using the probe CSV
- Collects/compares stats, writes tables + plots into ../runs/

Run from inside transformers/:
    python bench_aqed_vs_baseline.py --quick   # fewer batches (fast sanity)
    python bench_aqed_vs_baseline.py           # default batches like your examples
"""

import os, sys, subprocess, shlex, argparse, textwrap
from pathlib import Path

R = Path(__file__).resolve().parent  # transformers/
ROOT = R.parent                      # project root
RUNS = ROOT / "runs"
SCRIPTS = ROOT / "scripts"

def run(cmd, env=None):
    print(f"\n$ {cmd}")
    proc = subprocess.run(cmd, shell=True, env=env)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def ensure_deps():
    try:
        import pandas as pd  # noqa
        import matplotlib.pyplot as plt  # noqa
    except Exception:
        print("\n[!] This script needs pandas and matplotlib.")
        print("    pip install pandas matplotlib\n")
        sys.exit(1)

def fast_probe(csv_out, png_out, depth=24, chi=512, tol=1e-6):
    env = os.environ.copy()
    # so probe can import your package
    env["PYTHONPATH"] = f"{ROOT}:{ROOT/'src'}:" + env.get("PYTHONPATH","")
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = f"""python {shlex.quote(str(SCRIPTS/'tn_diffusion_probe.py'))} \
      --rows 6 --cols 6 --depth {depth} \
      --chi {chi} --tol {tol} --adaptive \
      --alphas "0.0,0.5,1.0" \
      --save {shlex.quote(str(png_out))} \
      --csv  {shlex.quote(str(csv_out))}
    """
    run(cmd, env=env)

def train_baseline(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs, out_csv):
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = f"""python train_baseline_transformer.py \
      --seq_len {seq_len} --batch_size {bs} \
      --train_batches {train_batches} --val_batches {val_batches} \
      --d_model {d_model} --n_layers {n_layers} --n_heads {n_heads} \
      --d_ff {d_ff} --epochs {epochs} \
      --log_csv {shlex.quote(str(out_csv))}
    """
    run(cmd)

def train_aqed_fixed(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs,
                     mixer_depth, pair_frac, out_csv):
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = f"""python train_transformer_aqed_mixer.py \
      --seq_len {seq_len} --batch_size {bs} \
      --train_batches {train_batches} --val_batches {val_batches} \
      --d_model {d_model} --n_layers {n_layers} --n_heads {n_heads} \
      --d_ff {d_ff} --epochs {epochs} \
      --mixer_depth {mixer_depth} --pair_frac {pair_frac} \
      --log_csv {shlex.quote(str(out_csv))}
    """
    run(cmd)

def train_aqed_controller(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs,
                          controller_csv, warmup, update, pair_min, pair_max, mixer_max_depth, out_csv):
    RUNS.mkdir(parents=True, exist_ok=True)
    cmd = f"""python train_transformer_aqed_controller.py \
      --seq_len {seq_len} --batch_size {bs} \
      --train_batches {train_batches} --val_batches {val_batches} \
      --d_model {d_model} --n_layers {n_layers} --n_heads {n_heads} \
      --d_ff {d_ff} --epochs {epochs} \
      --controller_csv {shlex.quote(str(controller_csv))} \
      --ctrl_warmup_steps {warmup} --ctrl_update_steps {update} \
      --pair_frac_min {pair_min} --pair_frac_max {pair_max} \
      --mixer_max_depth {mixer_max_depth} \
      --log_csv {shlex.quote(str(out_csv))}
    """
    run(cmd)

def summarize(paths):
    py = rf"""
import os, pandas as pd, matplotlib.pyplot as plt, numpy as np

runs = {paths!r}
rows = []
for path in runs:
    if not os.path.exists(path): 
        print(f"[skip] missing {{path}}"); 
        continue
    df = pd.read_csv(path)
    last_val = df[df["split"]=="val"].tail(1).copy()
    if last_val.empty: 
        print(f"[skip] no 'val' rows in {{path}}"); 
        continue
    last_val["run"] = os.path.basename(path)
    rows.append(last_val[["run","loss","ppl","tok_per_s","max_mem_mib","elapsed_s"]])

if not rows:
    print("No results to summarize."); raise SystemExit

out = pd.concat(rows, ignore_index=True)
print(out.to_string(index=False))
out.to_csv("{RUNS.as_posix()}/summary_compare.csv", index=False)

# rel metrics vs baseline
base = out[out["run"].str.contains("baseline", case=False)].head(1)
base = base.iloc[0]
out2 = out.copy()
out2["tok_per_s_rel"] = out2["tok_per_s"]/base["tok_per_s"]
out2["loss_delta"] = out2["loss"] - base["loss"]
out2.to_csv("{RUNS.as_posix()}/summary_compare_with_rel.csv", index=False)
print("\\nWrote summary_compare.csv & summary_compare_with_rel.csv")

# plots
plt.figure()
plt.bar(out["run"], out["tok_per_s"])
plt.xticks(rotation=45, ha="right"); plt.ylabel("tokens/sec"); plt.title("Throughput by run")
plt.tight_layout(); plt.savefig("{RUNS.as_posix()}/summary_tokps.png")

plt.figure()
plt.bar(out["run"], out["loss"])
plt.xticks(rotation=45, ha="right"); plt.ylabel("val loss"); plt.title("Validation loss by run")
plt.tight_layout(); plt.savefig("{RUNS.as_posix()}/summary_loss.png")

# Pareto (loss vs tok/s)
def pareto_mask(df):
    pts = df[["loss","tok_per_s"]].to_numpy()
    mask = np.ones(len(pts), dtype=bool)
    for i,(L,T) in enumerate(pts):
        if not mask[i]: continue
        # dominated if exists j with loss<=L and tok>=T and at least one strict
        dom = (df["loss"]<=L) & (df["tok_per_s"]>=T) & ((df["loss"]<L)|(df["tok_per_s"]>T))
        if dom.any(): mask[i] = False
    return mask

mask = pareto_mask(out)
plt.figure()
for _,r in out.iterrows():
    plt.scatter(r["loss"], r["tok_per_s"])
    plt.text(r["loss"]+1e-3, r["tok_per_s"], r["run"], fontsize=8)
plt.scatter(out.loc[mask,"loss"], out.loc[mask,"tok_per_s"], s=100, facecolors='none', edgecolors='red', label="Pareto")
plt.xlabel("val loss (lower better)"); plt.ylabel("tokens/sec (higher better)")
plt.title("Loss vs Throughput (Pareto-highlighted)")
plt.legend(); plt.tight_layout()
plt.savefig("{RUNS.as_posix()}/summary_pareto.png")

print("Saved plots: summary_tokps.png, summary_loss.png, summary_pareto.png in runs/")
"""
    run(f'python - <<\'PY\'\n{py}\nPY')

def main():
    ensure_deps()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Shorter run (fewer batches) for fast sanity.")
    args = ap.parse_args()

    # Default configs (match your examples)
    seq_len, bs = 512, 64
    d_model, n_layers, n_heads, d_ff = 512, 8, 8, 2048
    epochs = 1
    train_batches, val_batches = (1000, 100) if args.quick else (3000, 200)

    # 1) FAST diffusion probe → controller CSV
    probe_csv = RUNS / ("aqed_alpha_sweep_small_quick.csv" if args.quick else "aqed_alpha_sweep_small_d32.csv")
    probe_png = RUNS / ("aqed_alpha_sweep_small_quick.png" if args.quick else "aqed_alpha_sweep_small_d32.png")
    depth = 16 if args.quick else 32
    fast_probe(probe_csv, probe_png, depth=depth, chi=512, tol=1e-6)

    # 2) Baseline
    baseline_csv = RUNS / ("baseline_quick_stats.csv" if args.quick else "baseline_small_stats.csv")
    train_baseline(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs, baseline_csv)

    # 3) AQED fixed (low + med)
    aqed_low_csv = RUNS / ("aqed_fixed_low_quick.csv" if args.quick else "aqed_fixed_low.csv")
    train_aqed_fixed(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs,
                     mixer_depth=1, pair_frac=0.15, out_csv=aqed_low_csv)

    aqed_med_csv = RUNS / ("aqed_fixed_med_quick.csv" if args.quick else "aqed_fixed_med.csv")
    train_aqed_fixed(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs,
                     mixer_depth=2, pair_frac=0.30, out_csv=aqed_med_csv)

    # 4) AQED controller
    aqed_ctrl_csv = RUNS / ("aqed_controller_quick_stats.csv" if args.quick else "aqed_controller_stats_d32.csv")
    warmup = 50 if args.quick else 100
    update = 50 if args.quick else 100
    train_aqed_controller(seq_len, bs, train_batches, val_batches, d_model, n_layers, n_heads, d_ff, epochs,
                          controller_csv=probe_csv, warmup=warmup, update=update,
                          pair_min=0.05, pair_max=0.6, mixer_max_depth=3,
                          out_csv=aqed_ctrl_csv)

    # 5) Summaries + plots
    summarize([
        str(baseline_csv),
        str(aqed_low_csv),
        str(aqed_med_csv),
        str(aqed_ctrl_csv),
    ])

    print("\nAll done. See ../runs/ for CSVs and plots.")

if __name__ == "__main__":
    main()
