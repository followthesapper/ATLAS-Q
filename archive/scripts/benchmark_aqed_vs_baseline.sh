#!/bin/bash
# Comprehensive Benchmark: AQED Algorithm vs Traditional Baseline
#
# This answers: Does the AQED algorithm provide meaningful speedup?
#
# Tests 4 configurations:
# 1. Baseline (full attention, no optimizations)
# 2. Baseline + GPU optimizations (torch.compile + SDPA)
# 3. AQED (attention skipping, no optimizations)
# 4. AQED + GPU optimizations (our full stack)

set -e

# GB10 workaround
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

cd "$(dirname "$0")"
source venv/bin/activate

# Configuration
SEQ_LEN=4096
BATCH_SIZE=8
EPOCHS=3
TRAIN_BATCHES=150
VAL_BATCHES=30

mkdir -p runs/aqed_benchmark

echo "================================================================"
echo "AQED Algorithm vs Traditional Baseline Benchmark"
echo "================================================================"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS (first is compilation, judge by epoch 2-3)"
echo ""
echo "Tests 1-4: AQED v1 (attention skipping)"
echo "Tests 5-6: AQED LowRank (real-valued Linformer + routing)"
echo ""
echo "This will take ~15 minutes total"
echo "================================================================"
echo ""

# Test 1: Baseline (traditional transformer - full attention every layer)
echo "[1/6] Running BASELINE (traditional, no optimizations)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model baseline \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --no_flash \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/1_baseline_noopt.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# Test 2: Baseline + GPU optimizations
echo "[2/6] Running BASELINE + GPU optimizations (compile + SDPA)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model baseline \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --compile \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/2_baseline_optimized.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# Test 3: AQED v1 (attention skipping, no GPU optimizations)
echo "[3/6] Running AQED v1 (attention skipping, no optimizations)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_old \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --no_flash \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/3_aqed_v1_noopt.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# Test 4: AQED v1 + GPU optimizations (full stack)
echo "[4/6] Running AQED v1 + GPU optimizations (full stack)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_old \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --compile \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/4_aqed_v1_optimized.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# NEW: AQED LowRank (real-valued with Triton-fused projections, no compile)
echo "[5/6] Running AQED LowRank (real-valued + Triton, no compile)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_lowrank \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --no_flash \
  --rank 64 \
  --route_frac 0.10 \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/5_aqed_lowrank_noopt.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# NEW: AQED LowRank + compile
echo "[6/6] Running AQED LowRank + compile..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_lowrank \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --compile \
  --rank 64 \
  --route_frac 0.10 \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/6_aqed_lowrank_optimized.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# Analyze results
echo "================================================================"
echo "RESULTS (Epoch 3 validation - steady state performance)"
echo "================================================================"

python3 - <<'EOF'
import pandas as pd
import sys

configs = [
    ("1. Baseline (traditional)", "runs/aqed_benchmark/1_baseline_noopt.csv"),
    ("2. Baseline + GPU opt", "runs/aqed_benchmark/2_baseline_optimized.csv"),
    ("3. AQED v1 (skip-attn)", "runs/aqed_benchmark/3_aqed_v1_noopt.csv"),
    ("4. AQED v1 + compile", "runs/aqed_benchmark/4_aqed_v1_optimized.csv"),
    ("5. AQED LowRank (Triton)", "runs/aqed_benchmark/5_aqed_lowrank_noopt.csv"),
    ("6. AQED LowRank + compile", "runs/aqed_benchmark/6_aqed_lowrank_optimized.csv"),
]

results = []
baseline_tps = None

print("")
print(f"{'Configuration':<28} {'Throughput':>15} {'Loss':>8} {'Speedup':>10}")
print("-" * 75)

for name, csv_path in configs:
    try:
        df = pd.read_csv(csv_path)
        # Get epoch 3 validation (steady state, after compilation)
        epoch3_val = df[(df['split'] == 'val') & (df['step'] > 200)].iloc[-1]

        tps = epoch3_val['tok_per_s']
        loss = epoch3_val['loss']

        if baseline_tps is None:
            baseline_tps = tps
            speedup_str = "1.00× (baseline)"
        else:
            speedup = tps / baseline_tps
            speedup_str = f"{speedup:.2f}×"

        print(f"{name:<28} {tps:>12,.0f} tok/s {loss:>7.4f} {speedup_str:>10}")
        results.append((name, tps, loss, speedup_str))

    except Exception as e:
        print(f"{name:<28} ERROR: {e}")

print("-" * 75)
print("")

# Key insights (relative to 1. Baseline noopt)
if len(results) >= 6:
    baseline_noopt = results[0][1]
    baseline_opt = results[1][1]
    aqed_v1_noopt = results[2][1]
    aqed_v1_opt = results[3][1]
    aqed_lowrank_noopt = results[4][1]
    aqed_lowrank_opt = results[5][1]

    print("KEY INSIGHTS:")
    print("-" * 75)
    print("1. AQED v1 vs Baseline (no opts):")
    print(f"   {aqed_v1_noopt / baseline_noopt:.2f}×")
    print("")
    print("2. AQED LowRank vs Baseline (no opts):")
    print(f"   {aqed_lowrank_noopt / baseline_noopt:.2f}× (real-valued Triton-fused)")
    print("")
    print(f"3. GPU Optimization Benefit (for baseline):")
    print(f"   {baseline_opt / baseline_noopt:.2f}× faster")
    print("")
    print("4. Total Benefit (AQED LowRank + compile vs Baseline noopt):")
    print(f"   {aqed_lowrank_opt / baseline_noopt:.2f}×")
    print("")

    if aqed_lowrank_opt / baseline_noopt >= 2.0:
        print("✓ RESULT: AQED LowRank provides SIGNIFICANT speedup (>2×)!")
    elif aqed_lowrank_opt / baseline_noopt >= 1.5:
        print("✓ RESULT: AQED LowRank provides MEANINGFUL speedup (1.5-2×)")
    elif aqed_lowrank_opt / baseline_noopt >= 1.2:
        print("ℹ RESULT: AQED LowRank provides MODEST speedup (1.2-1.5×)")
    else:
        print("⚠ RESULT: AQED LowRank speedup is MINIMAL (<1.2×)")
        print("   Consider testing at longer sequences (L=8192+)")

print("")
print("Full results saved to: runs/aqed_benchmark/")
print("================================================================")
EOF

echo ""
echo "Benchmark complete!"
