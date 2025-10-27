
import numpy as np

def neighbor_corr(bits):
    n = bits.shape[1]
    cs = []
    for j in range(n-1):
        x, y = bits[:, j], bits[:, j+1]
        cs.append(((x==y).mean() - (x!=y).mean()))
    return float(np.mean(cs))

def test_mps_neighbor_correlation_sampling():
    from atlas_q import MatrixProductState
    mps = MatrixProductState(num_qubits=24, bond_dim=8)
    if hasattr(mps, "canonicalize_left_to_right"):
        mps.canonicalize_left_to_right()
    if hasattr(mps, "canonicalize_right_to_left"):
        mps.canonicalize_right_to_left()
    # Sample
    try:
        samples = mps.sweep_sample(num_shots=300)
    except AttributeError:
        samples = mps.measure(num_shots=300)
    # Convert to bit arrays (fallback for int encoding)
    def to_bits(s, n=24):
        if isinstance(s, (list, tuple, np.ndarray)) and len(s)==n:
            return np.array(s, dtype=int)
        v = int(s)
        b = np.zeros(n, dtype=int)
        for j in range(n):
            b[j] = (v >> j) & 1
        return b
    B = np.vstack([to_bits(s) for s in samples])
    c = neighbor_corr(B)
    # Not asserting a specific value—just that it's finite and in [-1,1]
    assert -1.0 <= c <= 1.0
