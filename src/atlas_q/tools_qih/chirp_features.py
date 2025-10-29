"""
Chirp family utilities: generate chirps and track period via sliding QIH features.
"""

import numpy as np

from .learned_period_head import learned_periods_or_fft


def synth_chirp(T=4096, r_start=20, r_end=80, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # phase integral for changing period: frequency ~ 1/r(t)
    r_t = np.linspace(r_start, r_end, T)
    phi = 2 * np.pi * np.cumsum(1.0 / np.maximum(r_t, 1e-6))
    x = np.sin(phi) + noise * rng.normal(size=T)
    return x, r_t


def track_periods(x, win=256, stride=128, topk=1):
    est = []
    for start in range(0, max(1, len(x) - win + 1), stride):
        w = x[start : start + win]
        rs = learned_periods_or_fft(w, model=None, topk=topk)
        est.append(rs[0])
    return np.array(est, dtype=int)
