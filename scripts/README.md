# ATLAS-Q Scripts

This directory contains demonstration scripts, benchmarks, and AI/ML tools for ATLAS-Q.

## Core Demonstrations (v0.5.0)

**Tensor Network Demos:**
- `demo_adaptive_mps.py` - Demonstrates adaptive MPS with GPU acceleration
- `demo_groundbreaking_features.py` - Shows robust SVD and AI-guided truncation
- `demo_new_features.py` - Demonstrates v0.5.0 features
- `qaoa_maxcut_grid.py` - QAOA MaxCut demonstration on grid graphs
- `tn_grid8x8_shallow.py` - Tensor network simulation on 8×8 grid
- `tn_qaoa_line64.py` - QAOA on 64-qubit line topology (standalone)
- `tn_diffusion_probe.py` - Tests entanglement diffusion in circuits

**Benchmarks:**
- `benchmark_simulator.py` - Comprehensive quantum simulator benchmarks
- `benchmark_compiled.py` - Compare compiled vs uncompiled AI models
- `gpu_benchmark.py` - Standalone GPU performance testing (matmul, conv, FFT)
- `qubit_capacity_probe.py` - Test maximum qubit capacity (full state vs MPS)
- `plot_qubit_probe.py` - Parse and visualize capacity probe results

## AI/ML Tools

**Rank Predictor Training:**
- `finetune_rank_predictor.py` - Fine-tune AI rank predictor models
- `optimize_ai_model.py` - Optimize AI model hyperparameters
- `qih_period_head_train.py` - Train quantum-inspired ML heads

**Dataset Generation:**
- `build_ft_dataset.py` - Build fine-tuning datasets from simulations
- `capture_sv_dataset.py` - Capture state vector data for training
- `collect_svd_data.py` - Collect SVD statistics from simulations
- `compare_real_vs_synthetic.py` - Compare real vs synthetic training data

**Evaluation & Analysis:**
- `eval_predictor_on_sv.py` - Evaluate predictor on state vectors
- `eval_rank_predictor.py` - Evaluate rank predictor accuracy
- `analyze_svd_logs.py` - Analyze SVD event logs
- `count_svd_events.py` - Count SVD truncation events

**Policy Optimization:**
- `ai_policy_ablation.py` - Ablation studies for AI truncation policies
- `run_grid_with_ai_policy.py` - Run grid simulations with AI policies

## Usage Examples

### Run Quantum Benchmarks
```bash
python scripts/benchmark_simulator.py
```

### Demo Adaptive MPS
```bash
python scripts/demo_adaptive_mps.py
```

### Test QAOA
```bash
python scripts/qaoa_maxcut_grid.py --n_qubits 8
```

### Probe Maximum Capacity
```bash
python scripts/qubit_capacity_probe.py --safety_frac 0.7 --verbose
python scripts/plot_qubit_probe.py probe_results.log
```

### Train AI Models
```bash
# Collect training data
python scripts/collect_svd_data.py --n_qubits 20 --depth 10

# Fine-tune predictor
python scripts/finetune_rank_predictor.py --dataset svd_data.pt

# Evaluate
python scripts/eval_rank_predictor.py --model models/rank_predictor.pt
```

## Dependencies

Core demos require:
- PyTorch 2.10+ with CUDA
- ATLAS-Q package (`pip install -e .`)
- Triton (for GPU kernels)

AI/ML tools additionally require:
- scikit-learn (for training)
- matplotlib (for plotting)

## Archived Scripts

Old/broken scripts have been moved to `archive/old_scripts/`:
- `benchmark_mps_pytorch.py` - Used deleted MatrixProductState class
- `benchmark_tensorcore_mps.py` - Used deleted AQED mps_mixer modules
- `benchmark_scan_optimizations.py` - Used deleted AQED optimization modules
- `benchmark_simulator_envs.py` - Used old import paths

These scripts tested experimental AQED components that were removed in v0.5.0.
