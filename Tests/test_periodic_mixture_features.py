
def test_qih_mixture_histogram_shapes():
    import numpy as np
    from periodic_mixture_features import qih_mixture_histogram
    x = np.sin(2*np.pi*np.arange(512)/8) + 0.3*np.sin(2*np.pi*np.arange(512)/16)
    H, rs = qih_mixture_histogram(x, k=2, bins=32)
    assert H.shape == (64,)
    assert len(rs) == 2
