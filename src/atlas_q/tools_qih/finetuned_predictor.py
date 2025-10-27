"""Adapter for fine-tuned MLP predictor."""

import numpy as np
import torch
import torch.nn as nn


class FinetunedMLP(nn.Module):
    """Simple MLP matching the fine-tuned architecture."""

    def __init__(self, d_in=135, d_h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FinetunedPredictorWrapper:
    """Wrapper to match RankPredictorWrapper interface."""

    def __init__(self, model_path, device="cuda"):
        self.device = device

        # Try to load as compiled model first
        if "compiled" in str(model_path):
            try:
                self.model = torch.jit.load(str(model_path), map_location=device)
                print("   Using TorchScript compiled model (faster)")
            except:
                # Fallback to regular model
                self.model = FinetunedMLP().to(device)
                self.model.load_state_dict(
                    torch.load(str(model_path).replace("_compiled", ""), map_location=device)
                )
        else:
            self.model = FinetunedMLP().to(device)
            self.model.load_state_dict(torch.load(str(model_path), map_location=device))

        self.model.eval()

    def _features(self, sigmas: torch.Tensor):
        """Compute features from singular values."""
        sv = sigmas.detach().float().cpu().numpy()
        if sv.size < 128:
            sv = np.pad(sv, (0, 128 - sv.size))
        elif sv.size > 128:
            sv = sv[:128]

        sv2 = sv * sv
        sw = float(sv2.sum() + 1e-12)
        mass = sv2 / sw

        # Spectral entropy
        H = float(-(mass[mass > 0] * np.log(mass[mass > 0] + 1e-12)).sum())

        # Log-sv features
        logs = np.log(sv + 1e-12)
        logs -= logs.max()

        # Cumulative mass landmarks
        cuts = [8, 16, 32, 64, 96, 128]
        cm = [float(mass[:k].sum()) for k in cuts]

        x = np.concatenate([logs, np.array([H], np.float32), np.array(cm, np.float32)])
        return torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict(self, sigmas: torch.Tensor) -> int:
        """Predict rank for given singular values."""
        with torch.no_grad():
            x = self._features(sigmas)
            frac = float(self.model(x).item())
            n_sv = int((sigmas > 1e-10).sum().item()) or 1
            r = max(1, min(int(frac * n_sv), n_sv))
            return r
