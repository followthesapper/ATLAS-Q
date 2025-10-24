import numpy as np, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path

class MLP(nn.Module):
    def __init__(self, d_in=135, d_h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def load_npz(path):
    d = np.load(path)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    split = d["split"].astype(np.int64)
    tr = (split == 0); va = (split == 1)
    return X[tr], y[tr], X[va], y[va]

def train(npz="runs/ftdata/dataset_real_sv.npz",
          out="models/rank_predictor_ft.pt",
          epochs=6, batch=8192, lr=1e-3, wd=1e-4, device="cuda"):
    Xtr, ytr, Xva, yva = load_npz(npz)
    model = MLP(d_in=Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = (1e9, None)
    for ep in range(1, epochs+1):
        model.train()
        p = torch.randperm(Xtr.size(0))
        loss_sum = 0.0
        for i in range(0, Xtr.size(0), batch):
            idx = p[i:i+batch]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * idx.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva.to(device))
            val_mse = ((pv - yva.to(device))**2).mean().item()
        tr_loss = loss_sum / Xtr.size(0)
        print(f"epoch {ep}  train_mse={tr_loss:.6f}  val_mse={val_mse:.6f}")
        if val_mse < best[0]:
            best = (val_mse, model.state_dict())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], out)
    print("✅ Saved", out, " (best val_mse=", f"{best[0]:.6f}", ")")

if __name__ == "__main__":
    train()
