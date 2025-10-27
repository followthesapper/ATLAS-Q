
import numpy as np

def near(sample, center, N, tol):
    d = min((sample-center) % N, (center-sample) % N)
    return d <= tol

def test_qft_peaks_for_periodic_state():
    from atlas_q import PeriodicState
    n = 10
    r = 5
    N = 2**n
    peaks = [k * N // r for k in range(r)]
    st = PeriodicState(num_qubits=n, period=r)
    samples = st.measure(num_shots=2000, use_qft=True)
    # At least 60% should land within ±2 bins of expected peaks
    tol = 2
    good = sum(1 for s in samples if any(near(s, p, N, tol) for p in peaks))
    assert good / len(samples) >= 0.6
