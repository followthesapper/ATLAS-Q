"""
Learned Period Head + Mixture QIH features.

If PyTorch is available, provides a small classifier over candidate periods.
NumPy fallback returns top-k FFT periods.
"""

import numpy as np

try:
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

from atlas_q.tools_qih.periodic_mixture_features import estimate_topk_periods_fft
from atlas_q.tools_qih.qih_pat import qft_histogram_for_period


def window_to_signal(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean() if len(x) else x
    s = x.std()
    if s > 1e-12:
        x = x / s
    return x


def mixture_qih_from_periods(periods, n=10, shots=512, bins=64):
    Hs = [qft_histogram_for_period(r, n=n, shots=shots, bins=bins) for r in periods]
    return np.concatenate(Hs, axis=0)


if _HAVE_TORCH:

    class PeriodHead(nn.Module):
        """
        Classify the dominant period among discrete classes [rmin..rmax].
        """

        def __init__(self, win, rmin=2, rmax=128, hidden=64):
            super().__init__()
            self.rmin, self.rmax = rmin, rmax
            self.win = win
            self.net = nn.Sequential(
                nn.Linear(win, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, rmax - rmin + 1),
            )

        def forward(self, xw):
            # xw: [B, W] float tensor
            # normalize per window
            mu = xw.mean(dim=1, keepdim=True)
            sd = xw.std(dim=1, keepdim=True) + 1e-6
            z = (xw - mu) / sd
            return self.net(z)  # logits

        def classes(self):
            return list(range(self.rmin, self.rmax + 1))

    def learned_periods_or_fft(x_window, model: PeriodHead, topk=3):
        """
        If model is None, return top-k FFT periods. Else, use softmax to pick top-k classes.
        """
        if model is None:
            return estimate_topk_periods_fft(x_window, k=topk)
        import torch

        with torch.no_grad():
            xx = torch.tensor(window_to_signal(x_window)[None, :], dtype=torch.float32)
            logits = model(xx)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        classes = model.classes()
        idx = np.argsort(probs)[::-1][:topk]
        return [classes[i] for i in idx]

else:

    def learned_periods_or_fft(x_window, model=None, topk=3):
        return estimate_topk_periods_fft(x_window, k=topk)
