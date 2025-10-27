
def test_public_api_surface():
    import atlas_q as qhs
    # Check key classes and methods exist (or skip gracefully)
    need = [
        "QuantumClassicalHybrid",
        "PeriodicState",
        "ProductState",
        "MatrixProductState",
    ]
    for name in need:
        assert hasattr(qhs, name), f"Missing symbol: {name}"
    # Spot-check common methods
    Hybrid = qhs.QuantumClassicalHybrid
    h = Hybrid(verbose=False)
    assert hasattr(h, "find_period")
    assert hasattr(h, "factor_number")
    assert hasattr(h, "create_circuit")
    assert hasattr(h, "execute_circuit")
