
def test_circuit_build_and_execute():
    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False)
    c = h.create_circuit(num_qubits=3)
    # Gates should chain (method returns circuit)
    c.h(0).h(1).rz(2, 1.0)
    # Execute with auto backend
    state = h.execute_circuit(c, backend="auto")
    # Should be able to draw some samples
    samples = state.measure(num_shots=50)
    assert len(samples) == 50
