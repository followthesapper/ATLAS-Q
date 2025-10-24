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
echo "This will take ~15 minutes total"
echo "================================================================"
echo ""

# Test 1: Baseline (traditional transformer - full attention every layer)
echo "[1/4] Running BASELINE (traditional, no optimizations)..."
python3 transformers/train_transformer_ultra_fast.py \
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
echo "[2/4] Running BASELINE + GPU optimizations (compile + SDPA)..."
python3 transformers/train_transformer_ultra_fast.py \
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

# Test 3: AQED algorithm (attention skipping, no GPU optimizations)
echo "[3/4] Running AQED algorithm (attention skipping, no optimizations)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --no_flash \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/3_aqed_noopt.csv \
  > /dev/null 2>&1

echo "   ✓ Complete"
echo ""

# Test 4: AQED + GPU optimizations (full stack)
echo "[4/4] Running AQED + GPU optimizations (full stack)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --compile \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/4_aqed_optimized.csv \
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
    ("3. AQED algorithm", "runs/aqed_benchmark/3_aqed_noopt.csv"),
    ("4. AQED + GPU opt", "runs/aqed_benchmark/4_aqed_optimized.csv"),
]

results = []
baseline_tps = None

print("")
print(f"{'Configuration':<25} {'Throughput':>15} {'Loss':>8} {'Speedup':>10}")
print("-" * 70)

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

        print(f"{name:<25} {tps:>12,.0f} tok/s {loss:>7.4f} {speedup_str:>10}")
        results.append((name, tps, loss, speedup_str))

    except Exception as e:
        print(f"{name:<25} ERROR: {e}")

print("-" * 70)
print("")

# Key insights
if len(results) >= 4:
    baseline_noopt = results[0][1]
    baseline_opt = results[1][1]
    aqed_noopt = results[2][1]
    aqed_opt = results[3][1]

    print("KEY INSIGHTS:")
    print("-" * 70)
    print(f"1. Algorithm Benefit (AQED vs Baseline, no opts):")
    print(f"   {aqed_noopt / baseline_noopt:.2f}× faster")
    print(f"   This is the PURE AQED algorithm speedup!")
    print("")
    print(f"2. GPU Optimization Benefit (for baseline):")
    print(f"   {baseline_opt / baseline_noopt:.2f}× faster")
    print(f"   Standard optimizations help traditional training")
    print("")
    print(f"3. Total Benefit (AQED + opts vs Baseline):")
    print(f"   {aqed_opt / baseline_noopt:.2f}× faster")
    print(f"   This is your complete system advantage!")
    print("")

    if aqed_opt / baseline_noopt >= 2.0:
        print("✓ RESULT: AQED provides SIGNIFICANT speedup (>2×)!")
    elif aqed_opt / baseline_noopt >= 1.5:
        print("✓ RESULT: AQED provides MEANINGFUL speedup (1.5-2×)")
    elif aqed_opt / baseline_noopt >= 1.2:
        print("ℹ RESULT: AQED provides MODEST speedup (1.2-1.5×)")
    else:
        print("⚠ RESULT: AQED speedup is MINIMAL (<1.2×)")
        print("   Consider testing at longer sequences (L=8192+)")

print("")
print("Full results saved to: runs/aqed_benchmark/")
print("================================================================")
EOF

echo ""
echo "Benchmark complete!"
