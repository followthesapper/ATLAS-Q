import numpy as np

# Optional torch
try:
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

try:
    from atlas_q import PeriodicState
except Exception:
    PeriodicState = None


def _estimate_period_fft(x, rmin=2, rmax=128):
    """Crude period estimate via FFT peak (ignoring DC)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    # real fft
    yf = np.fft.rfft(x)
    mag = np.abs(yf)
    mag[0] = 0.0
    # frequencies -> periods (in samples)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Avoid div-by-zero: mask freqs too small
    eps = 1e-12
    periods = np.where(freqs > eps, 1.0 / freqs, np.inf)
    # mask out-of-range periods
    mask = (periods >= rmin) & (periods <= rmax)
    if not np.any(mask):
        return max(rmin, min(rmax, n // 4 if n >= 8 else rmin))
    k = np.argmax(mag[mask])
    return int(np.clip(np.round(periods[mask][k]), rmin, rmax))


def qft_histogram_for_period(r, n=10, shots=1024, bins=64):
    """Return a normalized histogram of analytic QFT samples for a PeriodicState with period r.
    n: number of qubits for the frequency register (2**n bins)
    """
    if PeriodicState is None:
        raise RuntimeError("PeriodicState not available")
    st = PeriodicState(num_qubits=n, period=int(r))
    samples = st.measure(num_shots=int(shots), use_qft=True)
    N = 2**n
    h = np.zeros(int(bins), dtype=float)
    for s in samples:
        h[(s * bins) // N] += 1.0
    tot = h.sum()
    if tot > 0:
        h /= tot
    return h


def qih_pat_window_features(x_window, n=10, shots=1024, bins=64, rmin=2, rmax=128):
    """Compute QIH-PAT side-channel features for one window of a univariate sequence.
    Returns (histogram, r_hat).
    """
    r_hat = _estimate_period_fft(x_window, rmin=rmin, rmax=rmax)
    h = qft_histogram_for_period(r_hat, n=n, shots=shots, bins=bins)
    return h, r_hat


def qih_pat_sequence_features(x, win=256, stride=128, n=10, shots=1024, bins=64):
    """Slide over sequence x to compute per-window QIH-PAT features.
    Returns array of shape [num_windows, bins] and list of periods.
    """
    x = np.asarray(x, dtype=float)
    feats, periods = [], []
    for start in range(0, max(1, len(x) - win + 1), stride):
        w = x[start : start + win]
        if len(w) < win:
            # pad with mean to keep FFT behavior stable
            pad = np.full(win - len(w), w.mean() if len(w) else 0.0, dtype=float)
            w = np.concatenate([w, pad], axis=0)
        h, r = qih_pat_window_features(w, n=n, shots=shots, bins=bins)
        feats.append(h)
        periods.append(r)
    return np.stack(feats, axis=0), periods


if _HAVE_TORCH:

    class QIHPATSideChannel(nn.Module):
        """Torch module that turns a window of scalar values into a fused feature with QIH-PAT histogram.
        Usage:
            z = module(x_window_tensor)  # x_window_tensor: [B, W]
        """

        def __init__(self, win, bins=64, proj_dim=32):
            super().__init__()
            self.win = win
            self.bins = bins
            self.fc = nn.Linear(bins, proj_dim)

        def forward(self, xw):
            # xw: [B, W] CPU tensor. We run the numpy-based QFT histogram per item.
            xs = xw.detach().cpu().numpy()
            H = []
            for i in range(xs.shape[0]):
                h, _ = qih_pat_window_features(xs[i], bins=self.bins)
                H.append(h)
            H = torch.tensor(np.stack(H, axis=0), dtype=xw.dtype, device=xw.device)
            return self.fc(H)
