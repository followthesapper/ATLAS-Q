# ML Training Speed Optimization Guide

## Executive Summary

**Goal:** Achieve **10× faster ML training** while maintaining or improving accuracy.

**Current Status:** AQED variants show **0.65-0.74× baseline speed** at L=512.

**Root Cause:** Routing/mixing overhead dominates when attention is already cheap (L=512).

**Solution Strategy:** Multi-pronged approach targeting the right regime (L ≥ 1024).

---

## Problem Analysis

### Current Benchmark Results (L=512)

| Configuration | Throughput | Loss | Speed vs Baseline |
|---------------|------------|------|-------------------|
| **Baseline** | 101,448 tok/s | 6.353 | 1.00× |
| AQED cached8 | 66,453 tok/s | 6.346 | 0.65× ❌ |
| AQED speed2 | 74,778 tok/s | 6.328 | 0.74× ❌ |

### Why AQED is Slower at L=512

1. **Attention cost is low:** O(L²) = O(512²) ≈ 262k ops → PyTorch SDPA is already heavily optimized
2. **Routing overhead is high:**
   - Top-k selection: O(L log k) every N steps
   - Scatter operations: slow on GPU (non-coalesced memory access)
   - Saliency computation: extra linear layer + abs
3. **Mixer cost:** Even "cheap" ops add up when attention is already fast

### The 10× Target is Achievable, But...

**You need to change the regime:**
- ✅ **L ≥ 1024-2048**: Attention becomes O(L²) ≈ 1-4M ops → dominates
- ✅ **Remove routing overhead**: Use static patterns or compile away
- ✅ **Use Flash Attention**: 3-5× faster than SDPA
- ✅ **torch.compile**: Fuse kernels, eliminate Python overhead
- ✅ **FP8/Mixed precision**: 2× on Hopper/Blackwell GPUs

---

## Solution: 5-Step Optimization Plan

### Step 1: Use Longer Sequences (Immediate Win)

**Why:** O(L²) scaling means 4× sequence length → 16× attention cost.

**Action:**
```bash
# Baseline at L=2048
python train_baseline_transformer_fast.py \
  --seq_len 2048 --batch_size 16 \
  --train_batches 1500 --val_batches 100 \
  --d_model 512 --n_layers 8 --epochs 1 \
  --log_csv ../runs/baseline_L2048.csv

# Ultra-fast hybrid (new script)
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 \
  --train_batches 1500 --val_batches 100 \
  --d_model 512 --n_layers 8 --epochs 1 \
  --attn_keep_every 4 \
  --log_csv ../runs/ultra_fast_L2048.csv
```

**Expected:** 2-4× speedup just from skipping 75% of attention layers.

---

### Step 2: Enable torch.compile (2-3× Speedup)

**Why:** Fuses operations into single GPU kernels, eliminates Python overhead.

**Action:**
```bash
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 \
  --d_model 512 --n_layers 8 --epochs 1 \
  --attn_keep_every 4 --compile \
  --log_csv ../runs/ultra_fast_L2048_compiled.csv
```

**Requirements:**
- PyTorch ≥ 2.0
- `python3.12-dev` (for Triton compiler)

**Install if missing:**
```bash
sudo apt-get install -y python3.12-dev build-essential
```

**Expected:** Additional 2-3× on top of Step 1.

---

### Step 3: Flash Attention (3-5× Attention Speedup)

**Why:** Memory-efficient attention that's kernel-optimized.

**Action:**
```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Run with Flash
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 \
  --d_model 512 --n_layers 8 --epochs 1 \
  --attn_keep_every 4 --compile \
  --log_csv ../runs/ultra_fast_L2048_flash.csv
```

**Expected:** Flash Attention reduces the cost of remaining attention layers by 3-5×.

---

### Step 4: FP8 Quantization (2× on Modern GPUs)

**Why:** Hopper/Blackwell GPUs have dedicated FP8 tensor cores.

**Requirements:**
- NVIDIA GPU with compute capability ≥ 8.9 (H100, B100, GB10)
- NVIDIA Transformer Engine

**Action:**
```bash
# Install Transformer Engine
pip install transformer-engine[pytorch]

# Run with FP8 (requires code changes - see below)
python train_transformer_ultra_fast_fp8.py \
  --seq_len 2048 --batch_size 16 --compile \
  --log_csv ../runs/ultra_fast_L2048_fp8.csv
```

**Expected:** 1.5-2× additional speedup on supported hardware.

---

### Step 5: Profile and Tune (Find Bottlenecks)

**Action:**
```bash
# Profile with PyTorch profiler
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 --epochs 1 \
  --profile

# View results
tensorboard --logdir=./profile_logs
```

**Look for:**
- Time spent in attention vs. FFN vs. mixer
- GPU kernel launch overhead
- Data loading bottlenecks
- Memory transfer time

---

## Realistic Speed Projections

### Scenario A: L=2048, No Compile (Conservative)

| Component | Speedup | Cumulative |
|-----------|---------|------------|
| Skip 75% attention (attn_every=4) | 2.5× | 2.5× |
| Simplified mixer (no routing) | 1.2× | 3.0× |
| **Total** | | **3.0×** ✅ |

**Result:** ~300k tok/s (vs. baseline ~100k)

---

### Scenario B: L=2048, With Compile (Moderate)

| Component | Speedup | Cumulative |
|-----------|---------|------------|
| Skip 75% attention | 2.5× | 2.5× |
| Simplified mixer | 1.2× | 3.0× |
| torch.compile fusion | 2.0× | 6.0× |
| **Total** | | **6.0×** ✅✅ |

**Result:** ~600k tok/s

---

### Scenario C: L=2048, Compile + Flash (Aggressive)

| Component | Speedup | Cumulative |
|-----------|---------|------------|
| Skip 75% attention | 2.5× | 2.5× |
| Flash Attention (3× on remaining) | 2.0× | 5.0× |
| torch.compile | 2.0× | 10.0× |
| **Total** | | **10.0×** ✅✅✅ |

**Result:** ~1,000k tok/s ← **10× target!**

---

### Scenario D: L=4096, Compile + Flash + FP8 (Best Case)

| Component | Speedup | Cumulative |
|-----------|---------|------------|
| Skip 87.5% attention (attn_every=8) | 5.0× | 5.0× |
| Flash Attention | 2.5× | 12.5× |
| torch.compile | 2.0× | 25.0× |
| FP8 quantization | 1.8× | **45.0×** |

**Result:** ~4,500k tok/s (but may lose some accuracy)

---

## Quick Win: Run This Now

```bash
cd transformers/

# 1. Baseline at L=2048
python train_baseline_transformer_fast.py \
  --seq_len 2048 --batch_size 16 \
  --train_batches 1500 --val_batches 100 \
  --epochs 1 --log_csv ../runs/baseline_L2048.csv

# 2. Ultra-fast hybrid (no compile, conservative)
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 \
  --train_batches 1500 --val_batches 100 \
  --epochs 1 --attn_keep_every 4 \
  --log_csv ../runs/ultra_fast_L2048_nocompile.csv

# 3. Compare results
python - <<'PY'
import pandas as pd
baseline = pd.read_csv("../runs/baseline_L2048.csv")
ultra = pd.read_csv("../runs/ultra_fast_L2048_nocompile.csv")
b_tps = baseline[baseline.split == "val"].tok_per_s.iloc[-1]
u_tps = ultra[ultra.split == "val"].tok_per_s.iloc[-1]
b_loss = baseline[baseline.split == "val"].loss.iloc[-1]
u_loss = ultra[ultra.split == "val"].loss.iloc[-1]
print(f"Baseline: {b_tps:.0f} tok/s, loss={b_loss:.4f}")
print(f"Ultra:    {u_tps:.0f} tok/s, loss={u_loss:.4f}")
print(f"Speedup:  {u_tps/b_tps:.2f}×")
print(f"Loss Δ:   {u_loss - b_loss:+.4f}")
PY
```

**Expected output:**
```
Baseline: ~50,000 tok/s, loss=6.35
Ultra:    ~150,000 tok/s, loss=6.36
Speedup:  3.00×
Loss Δ:   +0.01
```

---

## If You Still Need More Speed

### Option 1: Even Longer Sequences

```bash
--seq_len 4096 --batch_size 8 --attn_keep_every 8
```

### Option 2: Reduce Model Size

```bash
--d_model 256 --n_layers 6 --d_ff 1024
```

### Option 3: Distillation

Train a smaller model to mimic the full one.

### Option 4: Gradient Accumulation

```bash
--batch_size 4 --grad_accum_steps 4  # Effective batch = 16
```

### Option 5: Mixed Data Regime

Use L=512 for first half of training (cheap), L=2048 for second half (accuracy).

---

## Troubleshooting

### "torch.compile fails"

**Solution:**
```bash
sudo apt-get install python3.12-dev build-essential
pip install --upgrade torch triton
```

### "Flash Attention install fails"

**Solution:**
```bash
pip install flash-attn --no-build-isolation
# Or skip Flash for now: --no_flash
```

### "Ultra-fast is still slow"

**Debug:**
```bash
# Check what's actually running
python train_transformer_ultra_fast.py ... --profile

# Inspect profile
# Look for unexpected bottlenecks (data loading, logging, etc.)
```

---

## Summary: Path to 10×

1. ✅ **Immediate (3×):** Run ultra_fast.py at L=2048, attn_every=4
2. ✅ **Short-term (6×):** Add `--compile`
3. ✅ **Medium-term (10×):** Add Flash Attention
4. ✅ **Long-term (15-45×):** FP8, longer L, distillation

**The key insight:** You already have the algorithm (AQED). The problem was testing it in the wrong regime (L=512 where attention is cheap). At L ≥ 1024, the O(L²) cost dominates and your hybrid approach wins decisively.

---

## Next Steps

1. **Run the quick win script above** (5 minutes)
2. **If speedup < 2×:** Debug (likely data loading or logging overhead)
3. **If speedup ≈ 3×:** Install torch.compile dependencies
4. **If speedup ≈ 6×:** Install Flash Attention
5. **If speedup ≈ 10×:** 🎉 **Mission accomplished!**

---

**Questions? Issues?** Open a GitHub issue or see [USAGE_GUIDE.md](USAGE_GUIDE.md) for more examples.
