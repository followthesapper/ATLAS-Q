
def test_2d_histogram_shape():
    import numpy as np
    from periodic2d_features import topk_2d_freqs, hist2d_from_freqs
    img = np.zeros((32,32))
    coords, mag = topk_2d_freqs(img, k=2)
    H = hist2d_from_freqs(coords, mag.shape, bins=16)
    assert H.shape == (16*16,)
    assert abs(H.sum() - 1.0) < 1e-6 or H.sum()==0.0
