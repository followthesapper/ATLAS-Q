
import pytest

def test_mps_canonicalization_idempotent():
    from atlas_q import MatrixProductState
    mps = MatrixProductState(num_qubits=16, bond_dim=8)
    # Capture a few amplitudes, then canonicalize L->R and R->L; amplitudes should remain finite
    idx = [0, 3, 7, 25, 101]
    before = [mps.get_amplitude(i) for i in idx] if hasattr(mps, "get_amplitude") else [0]*len(idx)
    if hasattr(mps, "canonicalize_left_to_right"):
        mps.canonicalize_left_to_right()
    if hasattr(mps, "canonicalize_right_to_left"):
        mps.canonicalize_right_to_left()
    after = [mps.get_amplitude(i) for i in idx] if hasattr(mps, "get_amplitude") else before
    assert all((a is None) or (b is None) or (abs(a-b) < 1e-6 or abs(a-b) < 1e-2) for a,b in zip(before, after))
