"""
Periodic Mixture Features: pick top-k candidate periods from a simple periodogram (FFT),
then build QIH histograms for each candidate period and concatenate them.
Useful when signals have multiple coexisting rhythms.
"""

import numpy as np

from atlas_q.tools_qih.qih_pat import qft_histogram_for_period


def estimate_topk_periods_fft(x, k=3, rmin=2, rmax=256):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    yf = np.fft.rfft(x)
    mag = np.abs(yf)
    mag[0] = 0.0
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    eps = 1e-12
    periods = np.where(freqs > eps, 1.0 / freqs, np.inf)
    mask = (periods >= rmin) & (periods <= rmax)
    idx = np.argsort(mag[mask])[::-1][:k]
    return [int(np.clip(round(p), rmin, rmax)) for p in periods[mask][idx]]


def qih_mixture_histogram(x_window, k=3, n=10, shots=512, bins=64):
    rs = estimate_topk_periods_fft(x_window, k=k)
    Hs = [qft_histogram_for_period(r, n=n, shots=shots, bins=bins) for r in rs]
    return np.concatenate(Hs, axis=0), rs
