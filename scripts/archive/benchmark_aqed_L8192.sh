#!/bin/bash
# AQED Benchmark at L=8192 - Testing for 10× speedup target

set -e

export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

cd "$(dirname "$0")"
source venv/bin/activate

SEQ_LEN=8192
BATCH_SIZE=4
EPOCHS=3
TRAIN_BATCHES=100
VAL_BATCHES=20

mkdir -p runs/aqed_benchmark

echo "================================================================"
echo "AQED at L=8192 - Targeting 10× Speedup"
echo "================================================================"
echo ""

# Baseline
echo "[1/4] Baseline (traditional, no opts)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --attn_keep_every 1 --no_flash \
  --train_batches $TRAIN_BATCHES --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/1_baseline_L8192_noopt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"

# Baseline + opts
echo "[2/4] Baseline + GPU opts..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --attn_keep_every 1 --compile \
  --train_batches $TRAIN_BATCHES --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/2_baseline_L8192_opt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"

# AQED
echo "[3/4] AQED algorithm (skip=16)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --attn_keep_every 16 --no_flash \
  --train_batches $TRAIN_BATCHES --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/3_aqed_L8192_noopt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"

# AQED + opts
echo "[4/4] AQED + GPU opts (full stack)..."
python3 transformers/train_transformer_ultra_fast.py \
  --seq_len $SEQ_LEN --batch_size $BATCH_SIZE --epochs $EPOCHS \
  --attn_keep_every 16 --compile \
  --train_batches $TRAIN_BATCHES --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_benchmark/4_aqed_L8192_opt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"

echo ""
echo "================================================================"
echo "RESULTS at L=8192"
echo "================================================================"

python3 - <<'EOF'
import pandas as pd

configs = [
    ("1. Baseline (traditional)", "runs/aqed_benchmark/1_baseline_L8192_noopt.csv"),
    ("2. Baseline + GPU opt", "runs/aqed_benchmark/2_baseline_L8192_opt.csv"),
    ("3. AQED algorithm", "runs/aqed_benchmark/3_aqed_L8192_noopt.csv"),
    ("4. AQED + GPU opt", "runs/aqed_benchmark/4_aqed_L8192_opt.csv"),
]

results = []
baseline_tps = None

print("")
print(f"{'Configuration':<25} {'Throughput':>15} {'Loss':>8} {'Speedup':>10}")
print("-" * 70)

for name, csv_path in configs:
    try:
        df = pd.read_csv(csv_path)
        epoch3_val = df[(df['split'] == 'val') & (df['step'] > 150)].iloc[-1]

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

if len(results) >= 4:
    baseline_noopt = results[0][1]
    aqed_opt = results[3][1]
    total_speedup = aqed_opt / baseline_noopt

    print("KEY RESULT:")
    print(f"  AQED + optimizations: {total_speedup:.2f}× faster than traditional baseline")
    print("")

    if total_speedup >= 10.0:
        print("🎯 SUCCESS! Achieved 10× speedup target!")
    elif total_speedup >= 7.0:
        print("✓ EXCELLENT! Close to 10× target")
    elif total_speedup >= 5.0:
        print("✓ VERY GOOD! Significant speedup achieved")
    else:
        print("ℹ Good progress, may need L=16384+ for 10×")

print("")
print("================================================================")
EOF
