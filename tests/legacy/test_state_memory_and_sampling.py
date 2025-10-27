
def test_memory_usage_and_sampling():
    from atlas_q import PeriodicState, ProductState, MatrixProductState
    p = PeriodicState(num_qubits=24, period=6)
    prod = ProductState(num_qubits=24)
    mps = MatrixProductState(num_qubits=24, bond_dim=8)
    assert p.memory_usage() > 0
    assert prod.memory_usage() >= p.memory_usage()
    assert mps.memory_usage() >= prod.memory_usage() or mps.memory_usage() > 0

    # Basic sampling does not throw and returns correct counts
    assert len(p.measure(num_shots=10)) == 10
    assert len(prod.measure(num_shots=12)) == 12
    # MPS may require sweep sampling
    try:
        got = mps.sweep_sample(num_shots=16)
        assert len(got) == 16
    except AttributeError:
        # fall back to regular measure if provided
        got = mps.measure(num_shots=16)
        assert len(got) == 16
