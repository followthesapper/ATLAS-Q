
def test_qih_pat_basic_histogram():
    from qih_pat import qft_histogram_for_period
    h = qft_histogram_for_period(8, n=9, shots=256, bins=32)
    assert h.shape == (32,)
    assert abs(h.sum() - 1.0) < 1e-6

def test_qih_pat_sequence_features():
    import numpy as np
    from qih_pat import qih_pat_sequence_features
    x = np.sin(2*np.pi*np.arange(512)/8) + 0.1*np.random.default_rng(0).normal(size=512)
    H, periods = qih_pat_sequence_features(x, win=128, stride=64, bins=32)
    assert H.ndim == 2 and H.shape[1] == 32
    assert len(periods) == H.shape[0]
