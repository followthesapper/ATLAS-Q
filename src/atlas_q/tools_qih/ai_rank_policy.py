from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PolicyConfig:
    f_min: float = 0.70  # minimum keep fraction
    f_max: float = 0.98  # maximum keep fraction
    chi_budget: Optional[int] = None
    cal_x: Optional[np.ndarray] = None
    cal_y: Optional[np.ndarray] = None


class CalibratedPolicyAdapter:
    """
    Backward-compatible adapter: exposes .predict(sigmas)->rank
    - optional monotone calibration (isotonic-like)
    - clamps to [f_min, f_max]
    - enforces chi_budget/chi_max at the call site
    """

    def __init__(self, base_predictor, cfg: PolicyConfig, chi_max: int):
        self.base = base_predictor
        self.cfg = cfg
        self.chi_max = int(chi_max)

    def _calibrate(self, f: float) -> float:
        if self.cfg.cal_x is None or self.cfg.cal_y is None:
            return float(np.clip(f, 0.0, 1.0))
        return float(np.interp(f, self.cfg.cal_x, self.cfg.cal_y))

    def _fraction_to_rank(self, f: float, n_sv: int) -> int:
        return max(1, int(math.floor(f * n_sv + 1e-6)))

    def predict(self, sigmas: torch.Tensor) -> int:
        with torch.no_grad():
            n_sv = int((sigmas > 1e-10).sum().item()) or 1
            if hasattr(self.base, "predict_fraction"):
                f = float(self.base.predict_fraction(sigmas))
            else:
                r = int(self.base.predict(sigmas))
                f = r / float(n_sv)
            f = self._calibrate(f)
            f = float(np.clip(f, self.cfg.f_min, self.cfg.f_max))
            r = self._fraction_to_rank(f, n_sv)
            r = min(r, self.chi_max)
            if self.cfg.chi_budget is not None:
                r = min(r, int(self.cfg.chi_budget))
            return max(1, int(r))
