#!/bin/bash
# Benchmark at L=8192 to see if AQED LowRank shines at longer sequences
# Reduce batch size to fit in memory

set -e

# GB10 workaround
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"

cd "$(dirname "$0")"
source venv/bin/activate

# Configuration for L=8192
SEQ_LEN=8192
BATCH_SIZE=4  # Reduced from 8 to fit in memory
EPOCHS=3
TRAIN_BATCHES=100  # Reduced from 150 (fewer steps, longer sequences)
VAL_BATCHES=20     # Reduced from 30

mkdir -p runs/aqed_L8192

echo "========================================================================"
echo "AQED Benchmark at L=8192 (Low-Rank Should Shine Here!)"
echo "========================================================================"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo ""
echo "Low-rank KV is O(L·r) vs O(L²), so the crossover favors LowRank at L≥8k"
echo "========================================================================"
echo ""

# Test 1: Baseline
echo "[1/6] Baseline (traditional, no opts)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model baseline \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --no_flash \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_L8192/1_baseline_noopt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Test 2: Baseline + GPU opt
echo "[2/6] Baseline + GPU optimizations..."
python3 transformers/train_transformer_ultra_fast.py \
  --model baseline \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 1 \
  --compile \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_L8192/2_baseline_optimized.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Test 3: AQED v1 (no opts)
echo "[3/6] AQED v1 (attention skipping, no opts)..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_old \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --no_flash \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_L8192/3_aqed_v1_noopt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Test 4: AQED v1 + compile
echo "[4/6] AQED v1 + compile..."
python3 transformers/train_transformer_ultra_fast.py \
  --model aqed_old \
  --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --attn_keep_every 8 \
  --compile \
  --train_batches $TRAIN_BATCHES \
  --val_batches $VAL_BATCHES \
  --log_csv runs/aqed_L8192/4_aqed_v1_optimized.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Test 5: AQED LowRank (Triton)
echo "[5/6] AQED LowRank (Triton, no compile)..."
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
  --log_csv runs/aqed_L8192/5_aqed_lowrank_noopt.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Test 6: AQED LowRank + compile
echo "[6/6] AQED LowRank + compile..."
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
  --log_csv runs/aqed_L8192/6_aqed_lowrank_optimized.csv \
  > /dev/null 2>&1
echo "   ✓ Complete"
echo ""

# Analyze results
echo "========================================================================"
echo "RESULTS (Epoch 3 validation - steady state, L=8192)"
echo "========================================================================"

python3 - <<'EOF'
import pandas as pd

configs = [
    ("1. Baseline (traditional)", "runs/aqed_L8192/1_baseline_noopt.csv"),
    ("2. Baseline + GPU opt", "runs/aqed_L8192/2_baseline_optimized.csv"),
    ("3. AQED v1 (skip-attn)", "runs/aqed_L8192/3_aqed_v1_noopt.csv"),
    ("4. AQED v1 + compile", "runs/aqed_L8192/4_aqed_v1_optimized.csv"),
    ("5. AQED LowRank (Triton)", "runs/aqed_L8192/5_aqed_lowrank_noopt.csv"),
    ("6. AQED LowRank + compile", "runs/aqed_L8192/6_aqed_lowrank_optimized.csv"),
]

results = []
baseline_tps = None

print("")
print(f"{'Configuration':<30} {'Throughput':>15} {'Loss':>8} {'Speedup':>12}")
print("-" * 80)

for name, csv_path in configs:
    try:
        df = pd.read_csv(csv_path)
        # Get final validation (last val entry)
        final_val = df[df['split'] == 'val'].iloc[-1]

        tps = final_val['tok_per_s']
        loss = final_val['loss']

        if baseline_tps is None:
            baseline_tps = tps
            speedup_str = "1.00× (baseline)"
        else:
            speedup = tps / baseline_tps
            speedup_str = f"{speedup:.2f}×"

        print(f"{name:<30} {tps:>12,.0f} tok/s {loss:>7.4f} {speedup_str:>12}")
        results.append((name, tps, loss, speedup_str))

    except Exception as e:
        print(f"{name:<30} ERROR: {e}")

print("-" * 80)
print("")

# Analysis
if len(results) >= 6:
    baseline_noopt = results[0][1]
    baseline_opt = results[1][1]
    aqed_v1_noopt = results[2][1]
    aqed_v1_opt = results[3][1]
    aqed_lowrank_noopt = results[4][1]
    aqed_lowrank_opt = results[5][1]

    print("KEY INSIGHTS at L=8192:")
    print("-" * 80)
    print("")
    print("1. Does LowRank catch up at longer sequences?")
    print(f"   AQED LowRank + compile: {aqed_lowrank_opt:,.0f} tok/s ({aqed_lowrank_opt / baseline_noopt:.2f}× vs baseline)")
    print(f"   AQED v1 + compile:      {aqed_v1_opt:,.0f} tok/s ({aqed_v1_opt / baseline_noopt:.2f}× vs baseline)")
    print(f"   LowRank vs v1 ratio:    {aqed_lowrank_opt / aqed_v1_opt:.2f}×")
    print("")

    if aqed_lowrank_opt > aqed_v1_opt:
        print("   ✓ YES! LowRank WINS at L=8192 (O(L·r) < O(L²) crossover!)")
    else:
        gap = (1 - aqed_lowrank_opt / aqed_v1_opt) * 100
        print(f"   LowRank is {gap:.1f}% slower but uses less memory")
        print(f"   (May win at L≥16k or with memory constraints)")

    print("")
    print("2. Memory-Quality Trade-off:")
    print(f"   LowRank: O(L·r) = O(8192·64) memory for KV")
    print(f"   Full:    O(L²)  = O(8192²)  memory for KV")
    print(f"   Memory savings: ~{8192 / 64:.0f}× less KV memory")

print("")
print("========================================================================"
)
print("Full results: runs/aqed_L8192/")
print("========================================================================"
)
EOF

echo ""
echo "Benchmark complete!"
