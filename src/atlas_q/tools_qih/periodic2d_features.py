"""
2D lattice periodicity features using 2D DFT/QFT analogy.
Given a 2D image/patch, estimate top-(kx, ky) frequencies and produce coarse histograms.
"""

import numpy as np


def topk_2d_freqs(img, k=3):
    f = np.fft.rfft2(img - img.mean())
    mag = np.abs(f)
    mag[0, 0] = 0.0
    # flatten indices
    idx = np.unravel_index(np.argsort(mag, axis=None)[::-1][:k], mag.shape)
    coords = list(zip(idx[0].tolist(), idx[1].tolist()))
    return coords, mag


def hist2d_from_freqs(freq_coords, shape, bins=32):
    H = np.zeros((bins, bins), dtype=float)
    H = H.ravel()
    Ny, Nx2 = shape  # note rfft2 width differs; treat abstractly
    for ky, kx in freq_coords:
        # map to bins
        bx = int(np.clip(round((kx / max(1, Nx2 - 1)) * (bins - 1)), 0, bins - 1))
        by = int(np.clip(round((ky / max(1, Ny - 1)) * (bins - 1)), 0, bins - 1))
        H[by * bins + bx] += 1.0
    if H.sum() > 0:
        H /= H.sum()
    return H
