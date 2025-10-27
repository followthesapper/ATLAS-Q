
import pytest

@pytest.mark.gpu
def test_cpu_gpu_period_consistency():
    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False, use_gpu=True)
    a, N = 7, 899
    cpu = h.find_period(a, N)
    # GPU may not exist; skip if API missing
    if not hasattr(h, "find_period_gpu"):
        pytest.skip("GPU path not available")
    gpu = h.find_period_gpu(a, N)
    assert cpu.period == gpu.period, "CPU and GPU should agree on the period"
