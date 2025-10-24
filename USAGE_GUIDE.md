# Usage Guide

Practical tutorials for common tasks with the Quantum Hybrid Simulator.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Quantum State Compression](#2-quantum-state-compression)
3. [Period-Finding and Factorization](#3-period-finding-and-factorization)
4. [Time Series Analysis with QIH](#4-time-series-analysis-with-qih)
5. [Training Transformers with AQED](#5-training-transformers-with-aqed)
6. [GPU Acceleration](#6-gpu-acceleration)
7. [Custom Experiments](#7-custom-experiments)

---

## 1. Getting Started

### Installation

```bash
# Basic installation
pip install quantum-hybrid-simulator

# With all features
pip install quantum-hybrid-simulator[all]

# Development mode (from source)
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[dev]
```

### Verify Installation

```python
import quantum_hybrid_system as qhs
print(qhs.__version__)  # Should print 0.2.0

# Quick test
sim = qhs.QuantumClassicalHybrid()
factors = sim.factor(15)
assert factors == [3, 5], "Installation test failed"
print("✓ Installation verified!")
```

---

## 2. Quantum State Compression

### 2.1 Periodic States (O(1) Memory)

**Use case:** Shor's algorithm, hidden subgroup problems

```python
from quantum_hybrid_system import PeriodicState

# Create periodic state: |ψ⟩ = (1/√k) Σ |a + j*r⟩
ps = PeriodicState(n_qubits=20, offset=3, period=7)

# Memory usage - just 32 bytes!
print(f"Memory: {ps.memory_bytes()} bytes")

# Sample from QFT (analytic, no FFT needed)
qft_samples = [ps.sample_qft_measurement() for _ in range(100)]

# Plot QFT histogram
import matplotlib.pyplot as plt
plt.hist(qft_samples, bins=50)
plt.xlabel("Measurement outcome")
plt.ylabel("Frequency")
plt.title(f"QFT of Periodic State (period={ps.period})")
plt.show()
```

### 2.2 Product States (O(n) Memory)

**Use case:** Separable states, initial conditions

```python
from quantum_hybrid_system import ProductState

# Initialize |+⟩^n state
ps = ProductState.init_plus(n_qubits=50)

# Apply single-qubit gates (O(1) each)
ps.apply_gate('X', target=0)  # Pauli-X on qubit 0
ps.apply_gate('H', target=1)  # Hadamard on qubit 1

# Measure (O(n))
outcome = ps.measure()
print(f"Measurement: {outcome}")
```

### 2.3 Matrix Product States (O(n×χ²) Memory)

**Use case:** Moderate entanglement, shallow circuits

```python
from quantum_hybrid_system import MatrixProductState

# Initialize |0⟩^n with bond dimension χ=16
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16)

# Apply two-qubit gates (entangle qubits)
for i in range(0, 99, 2):
    mps.apply_two_qubit_gate('CNOT', control=i, target=i+1)

# Measure
outcome = mps.measure()
print(f"Entangled measurement: {outcome}")
print(f"Memory: {mps.memory_bytes() / 1024:.1f} KB")
```

---

## 3. Period-Finding and Factorization

### 3.1 Basic Period-Finding

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()

# Define periodic function
def f(x):
    return pow(7, x, 15)  # 7^x mod 15

# Find period (uses O(√r) algorithms)
period = sim.find_period(f, max_period=15)
print(f"Period: {period}")  # Should be 4
```

### 3.2 Factor Semiprimes

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid()

# Factor small semiprimes
factors = sim.factor(221)  # 13 × 17
print(f"Factors of 221: {factors}")

# Benchmark performance
import time
for bits in [6, 8, 10, 12]:
    N = 2**bits + 1  # Not a semiprime, but for demo
    start = time.time()
    try:
        factors = sim.factor(N)
        elapsed = time.time() - start
        print(f"{bits}-bit: {factors} in {elapsed:.3f}s")
    except:
        print(f"{bits}-bit: Failed")
```

---

## 4. Time Series Analysis with QIH

### 4.1 Hidden Period Detection

```python
import numpy as np
from quantum_hybrid_system.tools_qih import periodic_mixture_features

# Generate signal with hidden periods
t = np.arange(1000)
signal = (
    np.sin(2 * np.pi * t / 24) +      # Daily cycle
    0.5 * np.sin(2 * np.pi * t / 168) # Weekly cycle
    + 0.1 * np.random.randn(1000)     # Noise
)

# Extract QIH features
periods = [24, 168]
qih_feats = periodic_mixture_features(signal, periods, hist_bins=32)

print(f"Feature shape: {qih_feats.shape}")  # (64,) - 32 bins per period

# Use for ML
from sklearn.ensemble import RandomForestClassifier
# ... train model with qih_feats as input
```

### 4.2 Multi-Period Forecasting

```python
from quantum_hybrid_system.tools_qih import qih_pat
import numpy as np

# Load time series data
data = np.loadtxt("sensor_data.csv")

# Detect periods automatically
detected_periods = qih_pat.detect_periods_fft(data, top_k=3)
print(f"Detected periods: {detected_periods}")

# Build forecasting model per period
forecasts = {}
for period in detected_periods:
    # Extract period-specific features
    features = qih_pat.compute_qih(data, period, bins=64)
    # Train model (e.g., ARIMA, Prophet, or ML)
    # forecast = model.predict(features)
    # forecasts[period] = forecast
```

---

## 5. Training Transformers with AQED

### 5.1 Quick Start with AQED

**New unified package (recommended):**

```bash
# Setup environment (for NVIDIA GB10/DGX Spark)
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

# Quick test
python scripts/train_aqed.py --seq_len 2048 --epochs 1

# Full training with all optimizations
python scripts/train_aqed.py \
  --seq_len 4096 --batch_size 8 --epochs 3 \
  --attn_keep_every 8 --compile \
  --log_csv runs/aqed_training.csv
```

### 5.2 Using AQED in Python Code

```python
import os
# Setup for torch.compile on GB10
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'

from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig
import torch

# Configure AQED (6-10× faster than traditional transformers!)
config = AQEDConfig(
    vocab_size=32000,
    seq_len=4096,
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,

    # AQED optimizations
    attn_keep_every=8,  # Use full attention every 8 layers
    use_flash=True,     # Memory-efficient attention (PyTorch SDPA)
    compile=True,       # torch.compile for additional speedup

    # Training settings
    batch_size=16,
    epochs=3,
    lr=3e-4,
    amp=True,           # Automatic mixed precision (FP16)
)

# Create model
model = AQEDTransformerLM(config).cuda()

# torch.compile (first epoch slow for compilation, subsequent epochs fast!)
if config.compile:
    model = torch.compile(model, mode="max-autotune")

# Train your model
# ... (see scripts/train_aqed.py for complete example)
```

### 5.3 Legacy Training Scripts (Preserved)

```bash
# Baseline transformer (traditional, no optimizations)
cd transformers/
python train_baseline_transformer_fast.py \
  --seq_len 4096 --batch_size 8 --epochs 1 \
  --log_csv ../runs/baseline.csv

# Ultra-fast transformer (all optimizations)
python train_transformer_ultra_fast.py \
  --seq_len 4096 --batch_size 8 --epochs 3 \
  --attn_keep_every 8 --compile \
  --log_csv ../runs/ultra_fast.csv
```

### 5.3 Compare Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
baseline = pd.read_csv("../runs/baseline.csv")
aqed = pd.read_csv("../runs/aqed_routed.csv")

# Plot loss curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(baseline[baseline.split == "train"].loss, label="Baseline")
plt.plot(aqed[aqed.split == "train"].loss, label="AQED")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(baseline[baseline.split == "train"].tok_per_s, label="Baseline")
plt.plot(aqed[aqed.split == "train"].tok_per_s, label="AQED")
plt.xlabel("Step")
plt.ylabel("Tokens/sec")
plt.legend()
plt.title("Throughput")

plt.tight_layout()
plt.savefig("../runs/comparison.png")
plt.show()
```

### 5.4 Hyperparameter Tuning

**Key knobs to tune:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `attn_keep_every` | 2–16 | Use full attention every N layers (higher = faster, slightly lower quality) |
| `seq_len` | 1024–16384 | Sequence length (longer = bigger speedup) |
| `batch_size` | 4–32 | Batch size (adjust for GPU memory) |
| `compile` | true/false | Enable torch.compile (1.2-1.4× additional speedup) |

**Recommended configurations:**

```bash
# Conservative (best quality, ~2× speedup)
python scripts/train_aqed.py --seq_len 2048 --attn_keep_every 4 --compile

# Balanced (good quality, ~6× speedup) ⭐ RECOMMENDED
python scripts/train_aqed.py --seq_len 4096 --attn_keep_every 8 --compile

# Aggressive (max speed, ~10× speedup)
python scripts/train_aqed.py --seq_len 8192 --attn_keep_every 16 --compile
```

**Quick tuning script:**
```python
import os

for seq_len in [2048, 4096, 8192]:
    for attn_every in [4, 8, 16]:
        cmd = f"""python scripts/train_aqed.py \
          --seq_len {seq_len} --batch_size 8 --epochs 2 \
          --attn_keep_every {attn_every} --compile \
          --log_csv runs/sweep_L{seq_len}_skip{attn_every}.csv"""
        os.system(cmd)
```

---

## 6. GPU Acceleration

### 6.1 Check GPU Availability

```python
try:
    import cupy as cp
    print(f"✓ CuPy available: {cp.cuda.runtime.getDeviceCount()} GPU(s)")
except ImportError:
    print("✗ CuPy not available (GPU acceleration disabled)")
```

### 6.2 Enable GPU in MPS

```python
from quantum_hybrid_system import MatrixProductState

# MPS automatically uses GPU if CuPy is available
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=32, device='cuda')

# Apply gates (uses GPU kernels)
mps.apply_two_qubit_gate('CNOT', control=0, target=1)
```

### 6.3 GPU Period-Finding

```python
from quantum_hybrid_system import QuantumClassicalHybrid

sim = QuantumClassicalHybrid(use_gpu=True)

# GPU-accelerated batched checking
def f(x):
    return pow(123, x, 10007)

period = sim.find_period(f, max_period=10000)
print(f"Period: {period}")
```

---

## 7. Custom Experiments

### 7.1 Custom Quantum Circuit

```python
from quantum_hybrid_system import MatrixProductState

mps = MatrixProductState.init_zero(n_qubits=10, max_bond_dim=16)

# Build custom circuit
circuit = [
    ('H', 0),          # Hadamard on qubit 0
    ('CNOT', 0, 1),    # CNOT from 0 to 1
    ('RY', 2, 0.5),    # RY(0.5) on qubit 2
    ('CZ', 1, 2),      # CZ from 1 to 2
]

# Execute
for gate in circuit:
    if len(gate) == 2:
        gate_name, target = gate
        mps.apply_single_qubit_gate(gate_name, target)
    elif len(gate) == 3:
        if gate[0] in ['RX', 'RY', 'RZ']:
            gate_name, target, angle = gate
            mps.apply_single_qubit_gate(gate_name, target, angle=angle)
        else:
            gate_name, control, target = gate
            mps.apply_two_qubit_gate(gate_name, control, target)

# Measure
outcome = mps.measure()
print(f"Circuit output: {outcome}")
```

### 7.2 Benchmark Different Backends

```python
from quantum_hybrid_system import PeriodicState, ProductState, MatrixProductState
import time

backends = [
    ("Periodic", PeriodicState.init_zero),
    ("Product", ProductState.init_zero),
    ("MPS (χ=8)", lambda n: MatrixProductState.init_zero(n, max_bond_dim=8)),
    ("MPS (χ=16)", lambda n: MatrixProductState.init_zero(n, max_bond_dim=16)),
]

for name, init_fn in backends:
    start = time.time()
    state = init_fn(n_qubits=50)
    init_time = time.time() - start

    mem = state.memory_bytes() / 1024  # KB

    print(f"{name:15s}: {init_time:.4f}s, {mem:.2f} KB")
```

---

## 8. Troubleshooting

### Issue: SVD Fails

**Symptom:** `RuntimeError: SVD did not converge`

**Solution:**
```python
# Enable robust multi-driver SVD
from quantum_hybrid_system.tools_qih import tn_core
tn_core.ENABLE_SVD_FALLBACK = True  # Default is True

# If still failing, try CPU fallback
mps = MatrixProductState.init_zero(n_qubits=100, max_bond_dim=16, device='cpu')
```

### Issue: AQED Not Faster than Baseline

**Symptom:** Expected speedup not observed

**Diagnosis:**
- At L≤1024, attention is already cheap; try L≥2048
- First epoch is slow (torch.compile compilation); check epoch 2+
- Not using torch.compile; add `--compile` flag

**Solution:**
```bash
# Longer sequences for bigger speedup
python scripts/train_aqed.py \
  --seq_len 4096 --batch_size 8 --epochs 3 \
  --attn_keep_every 8 --compile

# Check epoch 2+ performance, not epoch 1!
```

### Issue: torch.compile Fails on GB10

**Symptom:** `PTXASError: ptxas fatal: Value 'sm_121a' is not defined`

**Solution:** Set environment variables:
```bash
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
```

Add to `~/.bashrc` for permanent fix.

### Issue: GPU Out of Memory

**Symptom:** `CUDA error: out of memory`

**Solution:**
```bash
# Reduce batch size
--batch_size 32

# Reduce bond dimension (for MPS)
--max_bond_dim 8

# Use gradient checkpointing (for transformers)
--gradient_checkpointing

# Monitor memory usage
nvidia-smi --query-gpu=memory.used --format=csv --loop=1
```

---

## 9. Advanced Topics

### 9.1 Custom Period-Finding Algorithm

```python
from quantum_hybrid_system import QuantumClassicalHybrid

class MyPeriodFinder:
    def find_period(self, f, max_period):
        # Your custom algorithm here
        # ...
        return period

sim = QuantumClassicalHybrid()
sim.period_finder = MyPeriodFinder()
```

### 9.2 Integrate with Qiskit

```python
# Convert MPS to Qiskit StateVector (expensive!)
from quantum_hybrid_system import MatrixProductState
import numpy as np
from qiskit.quantum_info import Statevector

mps = MatrixProductState.init_zero(n_qubits=10, max_bond_dim=8)
# ... apply gates ...

# Extract full state vector (warning: exponential memory!)
full_vector = mps.to_full_statevector()  # 2^10 = 1024 elements
qiskit_state = Statevector(full_vector)

# Now use Qiskit methods
print(qiskit_state.probabilities())
```

---

## 10. Best Practices

### Performance

1. **Use the right backend:**
   - Periodic → structured periodic problems (Shor's)
   - Product → separable states only
   - MPS → moderate entanglement (shallow circuits)

2. **GPU acceleration:**
   - Enable for large bond dimensions (χ ≥ 32)
   - Enable for period-finding on large ranges

3. **AQED tuning:**
   - Start with baseline, measure throughput/loss
   - Try `route_frac=0.15, mixer_depth=2` first
   - Tune based on profiling (see below)

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose mode
sim = QuantumClassicalHybrid(verbose=True)
```

### Profiling

```bash
# Python profiler
python -m cProfile -o profile.stats train_transformer_routed_hybrid.py ...

# View results
python -m pstats profile.stats
> sort cumtime
> stats 20
```

---

## 11. Resources

- **[Technical Whitepaper](WHITEPAPER.md)**: Deep dive into algorithms, complexity analysis, and performance benchmarks
- **[README.md](README.md)**: Project overview, quick start, and installation guide
- **[Notebooks](Notebooks/)**: 21 interactive examples
- **[GitHub Issues](https://github.com/your-org/quantum-hybrid-simulator/issues)**: Report bugs
- **[Archived Documentation](docs/archive/)**: Historical docs, detailed guides, and testing results

---

**Happy simulating!** 🎉
