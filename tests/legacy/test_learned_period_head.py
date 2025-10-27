
def test_learned_periods_or_fft_fallback():
    import numpy as np
    from learned_period_head import learned_periods_or_fft
    x = np.sin(2*np.pi*np.arange(512)/12)
    rs = learned_periods_or_fft(x, model=None, topk=2)
    assert isinstance(rs, list) and len(rs)==2
    assert all(isinstance(r, int) for r in rs)
