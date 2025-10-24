
def test_chirp_track_runs():
    from chirp_features import synth_chirp, track_periods
    x, r = synth_chirp(T=2048, r_start=12, r_end=40, noise=0.1, seed=2)
    est = track_periods(x, win=256, stride=128, topk=1)
    assert len(est) > 0
