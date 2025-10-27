# ATLAS-Q Scripts

Demonstration scripts, benchmarks, AI/ML tools, and performance probes for ATLAS-Q.

## Directory Structure

```
scripts/
├── demos/           # 7 demonstration scripts
├── benchmarks/      # 6 benchmark scripts
├── ai_tools/        # 13 AI/ML training and evaluation scripts
└── probes/          # 2 capacity/performance probes
```

---

## Demonstrations (`demos/`)

Interactive demonstrations of ATLAS-Q features:

### Tensor Network Simulations
- **`demo_adaptive_mps.py`** - Adaptive MPS with GPU acceleration
- **`demo_groundbreaking_features.py`** - Robust SVD and AI-guided truncation
- **`demo_new_features.py`** - v0.5.0 feature showcase
- **`tn_grid8x8_shallow.py`** - Tensor network on 8×8 grid
- **`tn_diffusion_probe.py`** - Entanglement diffusion in circuits

### Quantum Algorithms
- **`qaoa_maxcut_grid.py`** - QAOA MaxCut on grid graphs
- **`tn_qaoa_line64.py`** - QAOA on 64-qubit line topology (standalone)

**Usage:**
```bash
# Run adaptive MPS demo
python demos/demo_adaptive_mps.py

# Run QAOA demo
python demos/qaoa_maxcut_grid.py --n_qubits 8

# Show all v0.5.0 features
python demos/demo_new_features.py
```

---

## Benchmarks (`benchmarks/`)

Performance benchmarking and competitive comparisons:

### Core Benchmarks
- **`comprehensive_benchmark.py`** - All 7 ATLAS-Q benchmarks (noise, stabilizer, MPO, TDVP, VQE, 2D, integration)
- **`competitive_comparison.py`** - Compare against Qiskit Aer, Cirq, ITensor
- **`benchmark_simulator.py`** - Quantum simulator performance metrics
- **`bench_moderate_capacity.py`** - Moderate-scale capacity testing

### Specialized Benchmarks
- **`benchmark_compiled.py`** - Compiled vs uncompiled AI model performance
- **`gpu_benchmark.py`** - Standalone GPU performance (matmul, conv, FFT)

**Usage:**
```bash
# Run comprehensive test suite
python benchmarks/comprehensive_benchmark.py

# Compare against competition
python benchmarks/competitive_comparison.py

# GPU-only benchmarks
python benchmarks/gpu_benchmark.py --fp16 --size 4096
```

---

## AI/ML Tools (`ai_tools/`)

Tools for training and evaluating AI-guided MPS truncation models:

### Training & Fine-Tuning
- **`finetune_rank_predictor.py`** - Fine-tune rank predictor on specific algorithms
- **`optimize_ai_model.py`** - Optimize hyperparameters for AI models
- **`qih_period_head_train.py`** - Train quantum-inspired ML heads

### Dataset Generation
- **`build_ft_dataset.py`** - Build fine-tuning datasets from simulations
- **`capture_sv_dataset.py`** - Capture state vector data for training
- **`collect_svd_data.py`** - Collect SVD statistics from simulations
- **`compare_real_vs_synthetic.py`** - Compare real vs synthetic training data

### Evaluation & Analysis
- **`eval_predictor_on_sv.py`** - Evaluate predictor on state vectors
- **`eval_rank_predictor.py`** - Evaluate rank predictor accuracy
- **`analyze_svd_logs.py`** - Analyze SVD event logs
- **`count_svd_events.py`** - Count SVD truncation events

### Policy Optimization
- **`ai_policy_ablation.py`** - Ablation studies for AI truncation policies
- **`run_grid_with_ai_policy.py`** - Run grid simulations with AI policies

**Usage:**
```bash
# Collect training data
python ai_tools/collect_svd_data.py --n_qubits 20 --depth 10 --output data.pt

# Fine-tune a model
python ai_tools/finetune_rank_predictor.py \
    --base_model ../models/rank_predictor.pt \
    --dataset data.pt \
    --epochs 20

# Evaluate model
python ai_tools/eval_rank_predictor.py --model ../models/rank_predictor_ft.pt

# Run ablation study
python ai_tools/ai_policy_ablation.py --n_qubits 30 --depth 50
```

---

## Capacity Probes (`probes/`)

Test maximum capacity and performance limits:

- **`qubit_capacity_probe.py`** - Test maximum qubit capacity (full state vs MPS)
- **`plot_qubit_probe.py`** - Parse and visualize capacity probe results

**Usage:**
```bash
# Probe capacity with 70% GPU memory safety margin
python probes/qubit_capacity_probe.py --safety_frac 0.7 --verbose > probe.log

# Visualize results
python probes/plot_qubit_probe.py probe.log
```

---

## Quick Reference

### Run All Benchmarks
```bash
python benchmarks/comprehensive_benchmark.py
```

### Demo ATLAS-Q Features
```bash
python demos/demo_adaptive_mps.py
python demos/qaoa_maxcut_grid.py
```

### Train AI Models
```bash
# Full workflow
python ai_tools/collect_svd_data.py --output svd_data.pt
python ai_tools/finetune_rank_predictor.py --dataset svd_data.pt
python ai_tools/eval_rank_predictor.py --model ../models/rank_predictor_ft.pt
```

### Test Capacity
```bash
python probes/qubit_capacity_probe.py --verbose
```

---

## Dependencies

All scripts require:
- **Core**: PyTorch 2.10+, NumPy, SciPy, matplotlib
- **GPU**: CUDA-enabled GPU with Triton kernels (run `../setup_triton.sh`)
- **AI Tools**: scikit-learn (for training scripts)

Install all dependencies:
```bash
cd ..
pip install -r requirements.txt
pip install -e .
./setup_triton.sh  # For GPU acceleration
```

---

## Archived Scripts

Old/broken scripts moved to `../archive/old_scripts/`:
- Scripts using deleted AQED modules (mps_mixer, hybrid_aqed, etc.)
- Benchmark scripts with old import paths
- Experimental AQED optimization tests

---

## Integration with Package

All scripts import from the main `atlas_q` package:

```python
# Correct imports (v0.5.0)
from atlas_q import get_quantum_sim, get_adaptive_mps
from atlas_q.adaptive_mps import AdaptiveMPS
from atlas_q.vqe_qaoa import VQE, QAOA
from atlas_q.tools_qih.tn_core import mps_init_plus

# OLD imports (don't use)
# from quantum_hybrid_system import ...  # Package renamed!
```

---

## Adding New Scripts

Place scripts in appropriate subdirectory:
- **demos/** - Interactive demonstrations, tutorials
- **benchmarks/** - Performance testing, comparisons
- **ai_tools/** - AI model training, evaluation, analysis
- **probes/** - Capacity/performance testing

---

## Support

- **Package docs**: `../docs/`
- **API reference**: `../docs/README_QUANTUM.md`
- **Models**: `../models/README.md`
- **Tests**: `../tests/README.md`

---

**Total Scripts**: 28 working scripts (7 demos, 6 benchmarks, 13 AI tools, 2 probes)
**Version**: 0.5.0
**Last Updated**: October 26, 2025
