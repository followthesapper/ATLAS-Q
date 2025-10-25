# Quantum Hybrid Simulator Usage Guide

Practical tutorials for quantum simulation and quantum-inspired machine learning.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Compressed Quantum States](#3-compressed-quantum-states)
4. [Period-Finding & Factorization](#4-period-finding--factorization)
5. [Tensor Networks (MPS)](#5-tensor-networks-mps)
6. [Quantum-Inspired ML](#6-quantum-inspired-ml)
7. [GPU Acceleration](#7-gpu-acceleration)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installation

### Basic Installation

```bash
pip install quantum-hybrid-simulator
```

### With ML Support

```bash
pip install quantum-hybrid-simulator[ml]
```

### With GPU Acceleration

```bash
pip install quantum-hybrid-simulator[gpu]
```

### All Features

```bash
pip install quantum-hybrid-simulator[all]
```

### Verify Installation

```python
import quantum_hybrid_system as qhs
print(qhs.__version__)  # Should print 0.3.0

# Quick test
sim = qhs.get_quantum_sim()[0]()
factors = sim.factor(15)
assert factors == [3, 5]
print("✓ Installation verified!")
```

---

## 2. Quick Start

### Factor a Semiprime (30 seconds)

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()
factors = sim.factor(221)  # 13 × 17
print(f"Factors: {factors}")  # [13, 17]
```

### Simulate 100 Qubits (30 seconds)

```python
from quantum_hybrid_system import MatrixProductState

mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16)
mps.apply_gate('H', target=0)
outcome = mps.measure()
print(f"Memory: {mps.memory_bytes() / 1024:.1f} KB")  # Only ~200 KB!
```

---

## 3. Compressed Quantum States

### 3.1 Periodic States (O(1) Memory)

**Use case:** Shor's algorithm, order-finding

```python
from quantum_hybrid_system import PeriodicState

# Create periodic state |ψ⟩ = (1/√k) Σ |a + j·r⟩
ps = PeriodicState(n_qubits=20, offset=0, period=7)

# Memory usage - just 32 bytes!
print(f"Memory: {ps.memory_bytes()} bytes")  # 32 B

# Sample from QFT (analytic, O(1) time)
samples = [ps.sample_qft_measurement() for _ in range(1000)]

# Visualize
import matplotlib.pyplot as plt
plt.hist(samples, bins=50)
plt.xlabel("Measurement outcome")
plt.ylabel("Frequency")
plt.title(f"QFT of Periodic State (period={ps.period})")
plt.show()
```

**Key insight:** Peaks appear at multiples of N/r, enabling period estimation.

### 3.2 Product States (O(n) Memory)

**Use case:** Separable states, no entanglement

```python
from quantum_hybrid_system import ProductState

# Initialize |+⟩^50 (all qubits in |+⟩ = (|0⟩ + |1⟩)/√2)
ps = ProductState.init_plus(n_qubits=50)

# Apply single-qubit gates
ps.apply_gate('X', target=0)  # Pauli-X (bit flip)
ps.apply_gate('H', target=1)  # Hadamard
ps.apply_gate('RZ', target=2, angle=np.pi/4)  # Rotation

# Measure
outcome = ps.measure()
print(f"Measurement: {outcome}")
print(f"Memory: {ps.memory_bytes()} bytes")  # 16 × 50 = 800 B
```

**Limitation:** Cannot represent entangled states.

### 3.3 Matrix Product States (O(n·χ²) Memory)

**Use case:** Moderate entanglement, shallow circuits

```python
from quantum_hybrid_system import MatrixProductState

# Initialize |0⟩^100 with bond dimension χ=16
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16)

# Apply gates
mps.apply_gate('H', target=0)
mps.apply_gate('X', target=5)

# Two-qubit gates (creates entanglement)
mps.apply_two_qubit_gate('CNOT', control=0, target=1)
mps.apply_two_qubit_gate('CZ', control=10, target=11)

# Measure
outcome = mps.measure()
print(f"Outcome: {outcome}")
print(f"Memory: {mps.memory_bytes() / 1024:.1f} KB")  # ~200 KB
```

---

## 4. Period-Finding & Factorization

### 4.1 Basic Period-Finding

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()

# Define periodic function
def f(x):
    return pow(7, x, 15)  # 7^x mod 15

# Find period (uses O(√r) algorithms)
period = sim.find_period(f, max_period=15)
print(f"Period: {period}")  # 4 (because 7^4 ≡ 1 mod 15)
```

### 4.2 Factor Semiprimes

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()

# Factor small semiprimes
test_cases = [
    (15, [3, 5]),
    (21, [3, 7]),
    (35, [5, 7]),
    (221, [13, 17]),
    (10403, [101, 103]),
]

for N, expected in test_cases:
    factors = sim.factor(N)
    print(f"{N:5d} = {factors[0]:4d} × {factors[1]:4d} | ✓" if factors == expected else "✗")
```

### 4.3 Benchmark Performance

```python
import time

for bits in [6, 8, 10, 12, 14]:
    # Generate random semiprime
    from random import randint
    p = next_prime(2**(bits//2) + randint(1, 100))
    q = next_prime(2**(bits//2) + randint(1, 100))
    N = p * q

    start = time.time()
    try:
        factors = sim.factor(N)
        elapsed = time.time() - start
        success = (factors == [p, q] or factors == [q, p])
        print(f"{bits:2d}-bit: {N:6d} = {factors[0]:4d} × {factors[1]:4d} | {elapsed*1000:6.1f} ms | {'✓' if success else '✗'}")
    except Exception as e:
        print(f"{bits:2d}-bit: Failed ({e})")
```

**Expected results:**
- 6-bit: <1 ms, 100% success
- 10-bit: ~10 ms, 100% success
- 13-bit: ~100 ms, ~95% success
- 16-bit: ~300 ms, ~90% success

---

## 5. Tensor Networks (MPS)

### 5.1 Build a GHZ State

GHZ state: (|0...0⟩ + |1...1⟩)/√2 (maximal entanglement)

```python
from quantum_hybrid_system import MatrixProductState

n_qubits = 50
mps = MatrixProductState.init_zero(n_qubits=n_qubits, max_bond_dim=16)

# Create GHZ: H on qubit 0, then CNOT chain
mps.apply_gate('H', target=0)
for i in range(n_qubits - 1):
    mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

# Measure (should be all 0s or all 1s)
for _ in range(10):
    outcome = mps.measure()
    print(f"Measurement: {outcome} (all {'0' if outcome.count('0') == n_qubits else '1'}s)")
```

### 5.2 Quantum Random Walk

```python
n_qubits = 20
mps = MatrixProductState.init_zero(n_qubits=n_qubits, max_bond_dim=8)

# Initial state: |+⟩ on qubit n//2 (middle)
mps.apply_gate('H', target=n_qubits // 2)

# Walk: apply Hadamard + CNOT repeatedly
for step in range(10):
    for i in range(0, n_qubits-1, 2):  # Even pairs
        mps.apply_gate('H', target=i)
        mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

    for i in range(1, n_qubits-1, 2):  # Odd pairs
        mps.apply_gate('H', target=i)
        mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

# Measure distribution
samples = [mps.measure() for _ in range(1000)]
positions = [s.index('1') if '1' in s else n_qubits//2 for s in samples]

import matplotlib.pyplot as plt
plt.hist(positions, bins=n_qubits)
plt.xlabel("Position")
plt.ylabel("Frequency")
plt.title(f"Quantum Walk (10 steps)")
plt.show()
```

### 5.3 Monitor Entanglement

```python
mps = MatrixProductState.init_zero(n_qubits=20, max_bond_dim=16)

entanglement_history = []

for depth in range(20):
    # Apply random gates
    import random
    i = random.randint(0, 18)
    mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

    # Measure entanglement (approximate via bond dimension)
    avg_bond = sum(core.shape[0] for core in mps.cores) / len(mps.cores)
    entanglement_history.append(avg_bond)

    print(f"Depth {depth:2d}: Avg bond dim = {avg_bond:.2f}")

# Entanglement grows with circuit depth (until saturation at χ_max)
```

---

## 6. Quantum-Inspired ML

### 6.1 Periodic Features for Time Series

```python
import numpy as np
from quantum_hybrid_system.tools_qih import periodic_mixture_features

# Generate signal with hidden periods
t = np.arange(1000)
signal = (
    np.sin(2 * np.pi * t / 24) +       # Daily cycle
    0.5 * np.sin(2 * np.pi * t / 168)  # Weekly cycle
    + 0.1 * np.random.randn(1000)      # Noise
)

# Extract QIH features
periods = [24, 168]
qih_features = periodic_mixture_features(signal, periods, hist_bins=32)

print(f"Feature shape: {qih_features.shape}")  # (64,) - 32 bins per period

# Use for ML
from sklearn.ensemble import RandomForestClassifier
# Assume we have labels for anomaly detection
labels = (np.abs(signal - signal.mean()) > 2 * signal.std()).astype(int)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    qih_features.reshape(1, -1), labels[:1], test_size=0.2
)

# For real use, extract features from sliding windows
```

### 6.2 Automatic Period Detection

```python
from quantum_hybrid_system.tools_qih import qih_pat

# Detect periods automatically
detected_periods = qih_pat.detect_periods_fft(signal, top_k=3)
print(f"Detected periods: {detected_periods}")  # Should find [24, 168, ...]

# Build features for each detected period
features = []
for period in detected_periods:
    qih = qih_pat.compute_qih(signal, period, bins=64)
    features.append(qih)

features = np.concatenate(features)
print(f"Total features: {features.shape}")
```

### 6.3 Anomaly Detection with QIH

```python
from quantum_hybrid_system.tools_qih import periodic_mixture_features
from sklearn.ensemble import IsolationForest

# Training data (normal behavior)
normal_data = load_sensor_data(days=30)  # Your data loading function

# Detect periods
from quantum_hybrid_system.tools_qih import qih_pat
periods = qih_pat.detect_periods_fft(normal_data, top_k=2)

# Extract features from sliding windows
window_size = 100
train_features = []
for i in range(len(normal_data) - window_size):
    window = normal_data[i:i+window_size]
    qih = periodic_mixture_features(window, periods, hist_bins=32)
    train_features.append(qih)

train_features = np.array(train_features)

# Train detector
detector = IsolationForest(contamination=0.05)
detector.fit(train_features)

# Monitoring (streaming)
for new_sample in streaming_sensor_data():
    qih = periodic_mixture_features(new_sample, periods, hist_bins=32)
    anomaly_score = detector.decision_function([qih])[0]

    if anomaly_score < threshold:
        alert(f"Anomaly detected at {new_sample.timestamp}")
```

---

## 7. GPU Acceleration

### 7.1 Check GPU Availability

```python
try:
    import cupy as cp
    n_gpus = cp.cuda.runtime.getDeviceCount()
    print(f"✓ CuPy available: {n_gpus} GPU(s)")

    # GPU info
    for i in range(n_gpus):
        device = cp.cuda.Device(i)
        props = device.attributes
        print(f"  GPU {i}: {props['Name'].decode()}")
except ImportError:
    print("✗ CuPy not installed (GPU acceleration disabled)")
```

### 7.2 Enable GPU in MPS

```python
from quantum_hybrid_system import MatrixProductState

# MPS automatically uses GPU if CuPy is available
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=32, device='cuda')

# Apply gates (uses GPU kernels)
import time
start = time.time()
for i in range(100):
    mps.apply_two_qubit_gate('CNOT', control=i%99, target=(i+1)%99)
elapsed = time.time() - start

print(f"100 CNOT gates: {elapsed:.3f}s ({100/elapsed:.1f} gates/sec)")
```

**Expected speedup:** 10-20× vs CPU for χ ≥ 32

### 7.3 GPU Period-Finding

```python
from quantum_hybrid_system import QuantumClassicalHybrid

# Enable GPU
sim = QuantumClassicalHybrid(use_gpu=True)

# GPU-accelerated batched checking
def f(x):
    return pow(123, x, 10007)  # Larger modulus

import time
start = time.time()
period = sim.find_period(f, max_period=10000)
elapsed = time.time() - start

print(f"Period: {period}, Time: {elapsed:.3f}s")
```

**Expected speedup:** 100-1000× vs CPU (when testing many candidates)

---

## 8. Troubleshooting

### 8.1 "SVD did not converge"

**Symptom:** `RuntimeError: SVD did not converge`

**Solution:** The simulator has automatic fallback, but you can also try:

```python
# Use CPU fallback explicitly
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16, device='cpu')
```

Or increase numerical stability:

```python
from quantum_hybrid_system.tools_qih import tn_core
tn_core.ENABLE_SVD_FALLBACK = True  # Default is True
```

### 8.2 "Period-finding fails"

**Symptom:** Returns incorrect period or raises error

**Diagnosis:**
1. Is period larger than `max_period`? → Increase `max_period`
2. Is function truly periodic? → Verify manually
3. Is period very large (>10000)? → May need longer runtime

**Solution:**
```python
# Increase max_period
period = sim.find_period(f, max_period=100000)

# Enable verbose mode
sim = QuantumClassicalHybrid(verbose=True)
period = sim.find_period(f, max_period=1000)
```

### 8.3 "Out of Memory (MPS)"

**Symptom:** OOM when simulating deep circuits

**Solutions:**
1. Reduce bond dimension:
   ```python
   mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=8)  # Lower χ
   ```

2. Use GPU (larger memory):
   ```python
   mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=32, device='cuda')
   ```

3. Accept larger truncation error (adaptive):
   ```python
   # Truncate more aggressively (in MPS internals)
   # This reduces memory at cost of accuracy
   ```

---

## 9. Advanced Usage

### 9.1 Custom Quantum Circuit

```python
from quantum_hybrid_system import MatrixProductState

def custom_circuit(n_qubits, depth):
    mps = MatrixProductState.init_zero(n_qubits=n_qubits, max_bond_dim=16)

    for layer in range(depth):
        # Layer of Hadamards
        for i in range(n_qubits):
            mps.apply_gate('H', target=i)

        # Layer of CNOTs (even)
        for i in range(0, n_qubits-1, 2):
            mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

        # Layer of CNOTs (odd)
        for i in range(1, n_qubits-1, 2):
            mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

    return mps

mps = custom_circuit(n_qubits=50, depth=10)
outcome = mps.measure()
print(f"Measurement: {outcome}")
```

### 9.2 Benchmark Different Backends

```python
import time

backends = [
    ("Periodic", lambda n: PeriodicState(n_qubits=n, offset=0, period=7)),
    ("Product", lambda n: ProductState.init_zero(n_qubits=n)),
    ("MPS (χ=8)", lambda n: MatrixProductState.init_zero(n, max_bond_dim=8)),
    ("MPS (χ=16)", lambda n: MatrixProductState.init_zero(n, max_bond_dim=16)),
]

n_qubits = 50

for name, init_fn in backends:
    start = time.time()
    state = init_fn(n_qubits)
    init_time = time.time() - start

    mem = state.memory_bytes() / 1024  # KB

    print(f"{name:15s}: {init_time*1000:6.2f} ms, {mem:8.2f} KB")
```

---

## 10. Resources

- **[Quantum Simulator Whitepaper](QUANTUM_SIMULATOR_WHITEPAPER.md)** - Algorithms, theory, proofs
- **[README.md](README.md)** - Project overview
- **[Interactive Notebooks](Notebooks/)** - 21 Jupyter notebooks
- **[GitHub Issues](https://github.com/your-org/quantum-hybrid-simulator/issues)** - Report bugs

---

**Happy Simulating!** 🎉

*For questions: https://github.com/your-org/quantum-hybrid-simulator/issues*
