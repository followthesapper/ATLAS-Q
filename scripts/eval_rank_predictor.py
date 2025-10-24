import numpy as np, torch, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finetune_rank_predictor import MLP

NPZ = Path("runs/ftdata/dataset_real_sv.npz")
OUT = Path("runs/ftdata/eval")
OUT.mkdir(parents=True, exist_ok=True)

def retained_mass_from_frac(fracs, sv):
    sv2 = sv**2
    cm = np.cumsum(sv2, axis=1)
    denom = sv2.sum(axis=1, keepdims=True) + 1e-12
    idx = np.clip(np.rint(fracs * (sv.shape[1]-1)).astype(int), 0, sv.shape[1]-1)
    rm = cm[np.arange(cm.shape[0]), idx] / denom.squeeze(1)
    return rm

def main(model_path="models/rank_predictor_ft.pt"):
    d = np.load(NPZ, allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split"]
    va = split == 1
    Xv, yv = torch.tensor(X[va], dtype=torch.float32), torch.tensor(y[va], dtype=torch.float32)
    # reconstruct a dummy sv matrix from features (we don't have raw SVs here),
    # so we evaluate only calibration of yhat vs y (fraction), which is still informative.
    model = MLP(d_in=X.shape[1]); model.load_state_dict(torch.load(model_path, map_location="cpu")); model.eval()
    with torch.no_grad():
        yhat = model(Xv).numpy()

    # Reliability curve
    bins = np.linspace(0,1,21)
    which = np.digitize(yhat, bins)-1
    pts_x, pts_y = [], []
    for b in range(len(bins)-1):
        mask = which == b
        if mask.any():
            pts_x.append(yhat[mask].mean())
            pts_y.append(yv.numpy()[mask].mean())
    plt.figure()
    plt.plot([0,1],[0,1], linestyle='--')
    plt.scatter(pts_x, pts_y)
    plt.xlabel("Predicted fraction"); plt.ylabel("Observed fraction")
    plt.title("Calibration (val split)")
    plt.savefig(OUT/"calibration_fraction.png", dpi=140)
    print("✅ Wrote", OUT/"calibration_fraction.png")

if __name__ == "__main__":
    main()
