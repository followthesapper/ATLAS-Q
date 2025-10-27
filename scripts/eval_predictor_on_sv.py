#!/usr/bin/env python
import argparse, numpy as np, torch
from pathlib import Path
from atlas_q.tools_qih.ai_rank_predictor import RankPredictorWrapper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="models/rank_predictor.pt")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    z = np.load(args.data)
    sig = torch.tensor(z["sigmas"], dtype=torch.float32, device="cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    tgt = torch.tensor(z["target_rank"], dtype=torch.float32, device=sig.device)
    n = sig.shape[0]

    rp = RankPredictorWrapper(model_path=args.model, device="cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")

    preds = []
    for i in range(n):
        preds.append(float(rp.predict(sig[i])))
    preds = torch.tensor(preds, device=sig.device)
    mse = torch.mean((preds - tgt)**2).item()
    mae = torch.mean(torch.abs(preds - tgt)).item()
    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")

    # simple calibration: fraction bins (pred/len(sig_nonzero))
    lengths = (sig>1e-10).sum(dim=1).float()
    pred_f = (preds/lengths).clamp(0,1).cpu().numpy()
    tgt_f  = (tgt/lengths).clamp(0,1).cpu().numpy()
    bins = np.linspace(0,1,11)
    print("Calibration (bin, mean target fraction):")
    for b0,b1 in zip(bins[:-1], bins[1:]):
        m = (pred_f>=b0) & (pred_f<b1)
        if m.any():
            print(f"[{b0:.1f},{b1:.1f}): {tgt_f[m].mean():.3f} (n={m.sum()})")
