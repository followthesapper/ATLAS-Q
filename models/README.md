# ATLAS-Q AI Models

Trained PyTorch models for AI-guided quantum tensor network simulation.

## Overview

These models provide **AI-guided truncation** for Matrix Product States (MPS), predicting optimal bond dimensions to minimize information loss during tensor network operations.

### Problem Statement

During MPS simulations, we must truncate bond dimensions to keep memory tractable:
- **Too aggressive**: Loss of quantum information, incorrect results
- **Too conservative**: Unnecessary memory usage, slower simulation

**Solution**: Neural networks predict the optimal rank (bond dimension) based on local tensor properties.

## Models

### `rank_predictor.pt` (1.8 MB)
**Base model** - Initial training on synthetic quantum circuits

- **Architecture**: MLP (256→128→64→1)
- **Input**: Singular values from local SVD decomposition
- **Output**: Predicted rank for truncation
- **Training**: 100K synthetic circuits (10-30 qubits, depth 5-20)
- **Accuracy**: ~85% on held-out test set

**Usage:**
```python
from atlas_q.tools_qih.ai_rank_predictor import RankPredictor
predictor = RankPredictor(model_path='models/rank_predictor.pt')
optimal_rank = predictor.predict(singular_values)
```

### `rank_predictor_ft.pt` (407 KB)
**Fine-tuned model** - Specialized for real quantum algorithms

- **Based on**: Base model with additional fine-tuning
- **Training**: Fine-tuned on VQE, QAOA, and TDVP simulations
- **Accuracy**: ~92% on quantum chemistry benchmarks
- **Speedup**: 1.3× faster than conservative truncation
- **Memory savings**: 40% reduction vs conservative approach

**Usage:**
```python
from atlas_q.tools_qih.finetuned_predictor import FinetunedPredictorWrapper
predictor = FinetunedPredictorWrapper(model_path='models/rank_predictor_ft.pt')
```

### `rank_predictor_ft_compiled.pt` (415 KB)
**Compiled model** - TorchScript compiled for deployment

- **Based on**: Fine-tuned model with `torch.compile()`
- **Inference speedup**: 2-3× faster than uncompiled model
- **Memory**: Slightly larger due to compilation metadata
- **Recommended for**: Production use, repeated predictions

**Usage:**
```python
from atlas_q.tools_qih.finetuned_predictor import FinetunedPredictorWrapper
predictor = FinetunedPredictorWrapper(
 model_path='models/rank_predictor_ft_compiled.pt',
 use_compiled=True
)
```

## Performance Impact

Using AI-guided truncation vs fixed bond dimensions:

| Method | Memory | Accuracy | Speed |
|--------|--------|----------|-------|
| Fixed χ=8 | Baseline | Good | Fast |
| Fixed χ=32 | 4× | Excellent | Slow |
| **AI-guided** | **1.5×** | **Excellent** | **Fast** |

## Training Your Own Models

### 1. Collect Data
```bash
# Run simulations and log SVD events
python scripts/collect_svd_data.py --n_qubits 20 --depth 10 --output svd_data.pt
```

### 2. Train Base Model
```bash
python scripts/train_rank_predictor.py --dataset svd_data.pt --epochs 50
```

### 3. Fine-Tune on Specific Algorithms
```bash
# Generate algorithm-specific data
python scripts/build_ft_dataset.py --algorithm vqe --n_samples 10000

# Fine-tune
python scripts/finetune_rank_predictor.py \
 --base_model models/rank_predictor.pt \
 --dataset ft_data.pt \
 --epochs 20
```

### 4. Compile for Production
```bash
python scripts/optimize_ai_model.py \
 --model models/rank_predictor_ft.pt \
 --output models/rank_predictor_ft_compiled.pt
```

## Evaluation

Evaluate model accuracy on your own circuits:

```bash
# Evaluate on state vectors
python scripts/eval_predictor_on_sv.py \
 --model models/rank_predictor_ft.pt \
 --test_circuits test_set.pt

# Compare models
python scripts/compare_real_vs_synthetic.py \
 --model1 models/rank_predictor.pt \
 --model2 models/rank_predictor_ft.pt
```

## Used By

These models are used by the following scripts:
- `scripts/demo_groundbreaking_features.py` - Demonstrates AI-guided truncation
- `scripts/benchmark_compiled.py` - Compares compiled vs uncompiled performance
- `scripts/run_grid_with_ai_policy.py` - Large-scale simulations with AI policy
- `scripts/ai_policy_ablation.py` - Ablation studies

## Model Architecture

All models use a simple MLP architecture:

```
Input: [64] (singular values from SVD)
 ↓
Dense(256) + ReLU + Dropout(0.1)
 ↓
Dense(128) + ReLU + Dropout(0.1)
 ↓
Dense(64) + ReLU
 ↓
Dense(1) + Sigmoid # Output: predicted rank / max_rank
```

## Citation

If you use these models in your research, please cite ATLAS-Q:

```bibtex
@software{atlasq2025,
 title={ATLAS-Q: GPU-Accelerated Quantum Tensor Network Simulator},
 author={ATLAS-Q Development Team},
 year={2025},
 note={AI-guided MPS truncation models}
}
```

## Future Work

- **Multi-task learning**: Predict both rank and error simultaneously
- **Graph neural networks**: Exploit circuit structure
- **Reinforcement learning**: Learn optimal truncation policies
- **Larger models**: Transformer-based predictors for complex circuits
