
import pytest

pytestmark = pytest.mark.gpu

def test_gpu_find_period_fallback():
    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False, use_gpu=True)
    a, N = 7, 899
    res_gpu = None
    if hasattr(h, "find_period_gpu"):
        res_gpu = h.find_period_gpu(a, N)
        assert res_gpu.period > 0
    # CPU always available
    res_cpu = h.find_period(a, N)
    assert res_cpu.period > 0
