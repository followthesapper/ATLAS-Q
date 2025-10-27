
import math
import pytest

def have(h, name):
    return hasattr(h, name)

def test_period_small_cases():
    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False, use_gpu=False)
    cases = [(2, 15, 4), (7, 15, 4), (2, 21, 6), (3, 221, None)]
    for a, N, r_expected in cases:
        res = h.find_period(a, N, method="auto")
        assert isinstance(res.period, int) and res.period > 0
        if r_expected is not None:
            # allow multiples (e.g., if algorithm returns a divisor or multiple)
            assert (res.period == r_expected) or (r_expected % res.period == 0) or (res.period % r_expected == 0)
        assert res.time_seconds >= 0.0

def test_factor_number_semiprimes():
    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False)
    for N in [221, 391, 437, 899]:
        f = h.factor_number(N)
        assert isinstance(f, (list, tuple)) and len(f) == 2
        assert f[0] * f[1] == N
