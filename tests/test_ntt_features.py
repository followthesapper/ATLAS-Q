
def test_ntt_period_estimation():
    import numpy as np
    from ntt_features import top_period_ntt
    N = 128
    r = 16
    x = (5*np.sin(2*np.pi*np.arange(N)/r)).astype(int)
    p, meta, X = top_period_ntt(x)
    assert p > 0
