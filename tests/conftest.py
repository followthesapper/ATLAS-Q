
import os
import random
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def _set_seed():
    random.seed(0)
    try:
        np.random.seed(0)
    except Exception:
        pass
    yield

def pytest_report_header(config):
    return "Quantum Hybrid Simulator — extra tests"
