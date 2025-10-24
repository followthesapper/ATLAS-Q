#!/bin/bash
# Quick Benchmark Script
# Compares baseline vs ultra-fast+compile performance
#
# Usage: ./quick_benchmark.sh
# Time: ~5 minutes

set -e

export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

cd "$(dirname "$0")"
source venv/bin/activate

echo "========================================"
echo "Quick Benchmark: Baseline vs Compiled"
echo "========================================"
echo ""

# Create results directory
mkdir -p runs/benchmarks

# Baseline (no compile, skip=1 = full attention every layer)
echo "Running baseline (2 epochs, ~2 min)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len 4096 \
  --batch_size 8 \
  --epochs 2 \
  --attn_keep_every 1 \
  --train_batches 100 \
  --val_batches 20 \
  --log_csv runs/benchmarks/baseline.csv \
  > /dev/null 2>&1

# Ultra-fast + compile
echo "Running ultra-fast + compile (2 epochs, ~3 min)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len 4096 \
  --batch_size 8 \
  --epochs 2 \
  --attn_keep_every 8 \
  --compile \
  --train_batches 100 \
  --val_batches 20 \
  --log_csv runs/benchmarks/compiled.csv \
  > /dev/null 2>&1

echo ""
echo "========================================"
echo "Results (Epoch 2 validation):"
echo "========================================"

python3 - <<'EOF'
import pandas as pd

try:
    base = pd.read_csv('runs/benchmarks/baseline.csv')
    comp = pd.read_csv('runs/benchmarks/compiled.csv')

    # Get epoch 2 validation results
    base_val = base[(base['split'] == 'val') & (base['step'] > 100)].iloc[-1]
    comp_val = comp[(comp['split'] == 'val') & (comp['step'] > 100)].iloc[-1]

    print(f"Baseline:        {base_val['tok_per_s']:>10,.0f} tok/s  (loss: {base_val['loss']:.4f})")
    print(f"Ultra + compile: {comp_val['tok_per_s']:>10,.0f} tok/s  (loss: {comp_val['loss']:.4f})")
    print(f"Speedup:         {comp_val['tok_per_s'] / base_val['tok_per_s']:>10.2f}×")
    print("")
    print("✓ Benchmark complete!")
    print(f"✓ Full results: runs/benchmarks/")

except Exception as e:
    print(f"Error reading results: {e}")
    print("Check runs/benchmarks/*.csv for raw data")
EOF

echo "========================================"
