import os, json, gzip, glob, math, numpy as np
from pathlib import Path

OUT_DIR = Path("runs/ftdata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_load_events(paths):
    rows = []
    bad = 0
    for p in paths:
        try:
            with gzip.open(p, "rt") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        if ev.get('kind') == 'svd_spectrum':  # Filter for SVD events
                            rows.append((p, ev))
                    except Exception:
                        bad += 1
        except Exception:
            bad += 1
    return rows, bad

def sv_to_features_and_target(ev):
    """
    Extract features from SVD event.
    ev should have: S_top128, adaptive_keep, tol, chi_max
    """
    # Get singular values (already in S_top128)
    sv = np.array(ev.get('S_top128', []), dtype=np.float32)
    
    # Ensure length 128
    if sv.size < 128:
        sv = np.pad(sv, (0, 128 - sv.size))
    elif sv.size > 128:
        sv = sv[:128]
    
    # Remove padding zeros for analysis
    sv_nonzero = sv[sv > 0]
    n_sv = max(1, len(sv_nonzero))
    
    # Compute features
    sv2 = sv**2
    sw = sv2.sum() + 1e-12
    mass = sv2 / sw
    
    # Spectral entropy
    H = float(-(mass[mass>0] * np.log(mass[mass>0] + 1e-12)).sum())
    
    # Log-sv features (shifted so max=0)
    logs = np.log(sv + 1e-12)
    logs -= logs.max()
    
    # Cumulative mass landmarks
    cuts = [8, 16, 32, 64, 96, 128]
    cm = [float(mass[:k].sum()) for k in cuts]
    
    # Features: 128 (log SVs) + 1 (entropy) + 6 (cumulative mass) = 135
    x = np.concatenate([logs, np.array([H], np.float32), np.array(cm, np.float32)], axis=0)
    
    # Target: adaptive_keep as fraction of total SVs
    adaptive_keep = int(ev.get('adaptive_keep', n_sv))
    frac = min(1.0, adaptive_keep / float(n_sv))
    
    return x.astype(np.float32), np.float32(frac)

def main():
    # Look in the correct directory
    paths = sorted(glob.glob("runs/svd_logs/svd_events_*.jsonl.gz"))
    if not paths:
        raise SystemExit("No runs/svd_logs/svd_events_*.jsonl.gz found.")
    
    print(f"Found {len(paths)} log files")
    rows, bad = safe_load_events(paths)
    print(f"Loaded {len(rows)} SVD events; skipped {bad} corrupt lines.")
    
    if not rows:
        raise SystemExit("No valid SVD events found!")
    
    X, y, file_ids = [], [], []
    skipped = 0
    
    for p, ev in rows:
        try:
            xi, yi = sv_to_features_and_target(ev)
            X.append(xi)
            y.append(yi)
            file_ids.append(p)
        except Exception as e:
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} events due to feature extraction errors")
    
    X = np.stack(X)
    y = np.stack(y)
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
    
    # File-level split: keep all events from a file together
    uniq = sorted(set(file_ids))
    rng = np.random.default_rng(123)
    rng.shuffle(uniq)
    cut = int(0.9 * len(uniq))
    train_files, val_files = set(uniq[:cut]), set(uniq[cut:])
    split = np.array([0 if f in train_files else 1 for f in file_ids], dtype=np.int8)
    
    train_count = (split == 0).sum()
    val_count = (split == 1).sum()
    print(f"\nTrain: {train_count} events ({100*train_count/len(split):.1f}%)")
    print(f"Val:   {val_count} events ({100*val_count/len(split):.1f}%)")
    
    np.savez_compressed(OUT_DIR / "dataset_real_sv.npz", X=X, y=y, split=split)
    
    with open(OUT_DIR / "file_split.json", "w") as f:
        import json
        json.dump({"train": list(train_files), "val": list(val_files)}, f, indent=2)
    
    print(f"\n✅ Saved {OUT_DIR / 'dataset_real_sv.npz'}")
    print(f"✅ Saved {OUT_DIR / 'file_split.json'}")

if __name__ == "__main__":
    main()
