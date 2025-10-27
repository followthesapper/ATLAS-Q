
import numpy as np

def coarse_hist(samples, n, bins=64):
    N = 2**n
    h = np.zeros(bins, dtype=float)
    for s in samples:
        h[(s * bins)//N] += 1.0
    if h.sum() > 0:
        h /= h.sum()
    return h

def class_sep(h1, h2):
    # simple L1 distance
    return float(np.abs(h1 - h2).sum())

def test_qft_histogram_separability():
    from atlas_q import PeriodicState
    n = 9
    r1, r2 = 4, 6
    s1 = PeriodicState(num_qubits=n, period=r1).measure(num_shots=2000, use_qft=True)
    s2 = PeriodicState(num_qubits=n, period=r2).measure(num_shots=2000, use_qft=True)
    h1 = coarse_hist(s1, n)
    h2 = coarse_hist(s2, n)
    # Expect some separability between different periods
    assert class_sep(h1, h2) > 0.5
