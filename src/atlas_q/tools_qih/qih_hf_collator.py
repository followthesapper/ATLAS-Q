"""
Example collator that computes QIH mixture histograms per window for token batches.
This is illustrative and does not import Hugging Face directly to keep dependencies light.
"""

import numpy as np

from learned_period_head import mixture_qih_from_periods
from periodic_mixture_features import estimate_topk_periods_fft


class QIHCollator:
    def __init__(self, win=256, stride=256, bins=64, topk=2):
        self.win = win
        self.stride = stride
        self.bins = bins
        self.topk = topk

    def __call__(self, batch_token_ids):
        # batch_token_ids: List[List[int]]
        H_batch = []
        for ids in batch_token_ids:
            x = np.array(ids[: self.win], dtype=float)
            x = x - x.mean() if len(x) else x
            per = estimate_topk_periods_fft(x, k=self.topk)
            H = mixture_qih_from_periods(per, bins=self.bins)
            H_batch.append(H)
        return np.stack(H_batch, axis=0)
