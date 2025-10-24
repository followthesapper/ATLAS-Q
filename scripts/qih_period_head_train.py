
"""
Train a tiny learned PeriodHead for windowed period classification.
Saves model to /mnt/data/period_head.pt if torch is available.
Falls back to FFT-only if torch is missing.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from learned_period_head import PeriodHead, window_to_signal

def synth_wave(period, length=256, noise=0.2, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.arange(length)
    x = np.sin(2*np.pi*t/period) + 0.2*np.sin(2*np.pi*t/(period*2))
    x += noise * rng.normal(size=length)
    return x

def make_dataset(periods=range(4,65), n_per=128, length=256, noise=0.2):
    rng = np.random.default_rng(0)
    X, y = [], []
    for r in periods:
        for _ in range(n_per):
            x = synth_wave(r, length=length, noise=noise, rng=rng)
            X.append(x); y.append(r)
    X = np.stack(X, axis=0); y = np.array(y, dtype=int)
    return X, y

def train_period_head():
    X, y = make_dataset()
    # map labels to 0..C-1 with rmin..rmax
    rmin, rmax = 4, 64
    y_idx = y - rmin
    if HAVE_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PeriodHead(win=X.shape[1], rmin=rmin, rmax=rmax)
        model.to(device)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y_idx, dtype=torch.long))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for step, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = lossf(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                print(f"step={step} loss={loss.item():.4f}")
        out_path = "/mnt/data/period_head.pt"
        torch.save(model.state_dict(), out_path)
        print("Saved:", out_path)
        return out_path
    else:
        print("Torch not available; training skipped. Falling back to FFT in downstream code.")
        return None

if __name__ == "__main__":
    train_period_head()
