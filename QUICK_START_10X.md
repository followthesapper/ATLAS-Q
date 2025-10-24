# Quick Start: Get to 10× Speed in 1 Week

**Goal:** Achieve 6-15× ML training speedup in 7 days.

---

## Day 1: Setup & Baseline (30 minutes)

### Step 1: Install Dependencies

```bash
cd ~/quantum-hybrid-simulator

# Activate venv
source venv/bin/activate

# Core dependencies (already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check version (need 2.0+)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Step 2: Run Baseline Benchmark

```bash
cd transformers/

# Quick test (5 minutes)
python benchmark_speedup.py --quick

# OR full test (20 minutes)
python benchmark_speedup.py --full
```

**Expected output:**
```
Baseline:             Loss=6.35    50,000 tok/s   1.00×
Ultra (no compile):   Loss=6.36    150,000 tok/s  3.00× 🎉
```

**If you see 3×+:** ✅ Move to Day 2
**If you see <2×:** ⚠️ See Troubleshooting section below

---

## Day 2: Add torch.compile (5 minutes) ✅ WORKING!

### Step 1: Set Environment Variables

**GB10/DGX Spark users**: Add to `~/.bashrc`:

```bash
# torch.compile workaround for GB10
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
```

Then: `source ~/.bashrc`

### Step 2: Benchmark with Compile

```bash
# Use the production script
./run_training.sh \
  --seq_len 4096 --batch_size 8 --epochs 3 \
  --attn_keep_every 8 --compile \
  --log_csv runs/ultra_compile.csv
```

**Expected speedup:** 1.25-1.37× (epoch 2+ vs baseline)

**Important:** Epoch 1 will be slow (compiling kernels). Judge by Epoch 2+!

---

## Day 3: Add Flash Attention (2 hours)

### Step 1: Install Flash Attention

```bash
# This can take 10-20 minutes to compile
pip install flash-attn --no-build-isolation

# Verify
python -c "import flash_attn; print(f'Flash Attn: {flash_attn.__version__}')"
```

### Step 2: Benchmark with Flash

```bash
cd transformers/

# Run with Flash Attention
python train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 --epochs 1 \
  --attn_keep_every 4 --compile \
  --log_csv ../runs/ultra_flash_test.csv
```

**Expected speedup:** 1.5-2× additional (total 9-12× vs baseline)

---

## Day 4: Optimize for Your Task (3 hours)

### Tune Hyperparameters

```bash
cd transformers/

# Sweep attention skipping
for k in 2 4 6 8; do
    python train_transformer_ultra_fast.py \
      --seq_len 2048 --batch_size 16 --epochs 1 \
      --attn_keep_every $k --compile \
      --log_csv ../runs/sweep_k${k}.csv
done

# Compare results
python - <<'PY'
import pandas as pd
import glob

for csv in sorted(glob.glob("../runs/sweep_k*.csv")):
    df = pd.read_csv(csv)
    val = df[df.split == "val"].iloc[-1]
    print(f"{csv:30s}: loss={val.loss:.4f}, tok/s={val.tok_per_s:.0f}")
PY
```

### Find Optimal Balance

- `attn_keep_every=2`: Best loss, moderate speed
- `attn_keep_every=4`: Good loss, good speed ✅ **RECOMMENDED**
- `attn_keep_every=8`: Okay loss, best speed

---

## Day 5: Add Triton Kernels (4 hours)

### Step 1: Install Triton

```bash
pip install triton
```

### Step 2: Test Triton Kernels

```bash
cd transformers/

# Test that kernels work
python -c "
from triton_kernels import TRITON_AVAILABLE, fused_aqed_mixer
print(f'Triton available: {TRITON_AVAILABLE}')
"
```

### Step 3: Benchmark with Triton

```bash
# Coming soon: train_transformer_triton.py
# For now, Triton kernels are integrated into ultra_fast.py
```

**Expected speedup:** 1.2-1.5× additional (total 11-18× vs baseline)

---

## Day 6-7: Production Testing & Tuning

### Run on Real Data

Replace Zipf synthetic data with your actual dataset:

```python
# In train_transformer_ultra_fast.py, replace ZipfDataset with:
from torch.utils.data import DataLoader
from your_dataset import YourDataset

train_ds = DataLoader(YourDataset('train'), batch_size=cfg.batch_size)
val_ds = DataLoader(YourDataset('val'), batch_size=cfg.batch_size)
```

### Profile & Optimize

```bash
# Profile to find remaining bottlenecks
python -m torch.profiler train_transformer_ultra_fast.py \
  --seq_len 2048 --batch_size 16 --epochs 1 \
  --compile --attn_keep_every 4

# View results
tensorboard --logdir=./profile_logs
```

### Longer Sequences

```bash
# Try L=4096 for even bigger wins
python train_transformer_ultra_fast.py \
  --seq_len 4096 --batch_size 8 --epochs 1 \
  --attn_keep_every 8 --compile \
  --log_csv ../runs/ultra_L4096.csv
```

**Expected at L=4096:** 15-20× vs baseline

---

## Troubleshooting

### "Speedup is only 1.5×"

**Diagnosis:**
1. Check sequence length: `L=512` is too short. Use `L ≥ 1024`.
2. Check GPU utilization: Run `nvidia-smi` during training. Should be 90%+.
3. Check that compile actually ran: Look for "Compiling model..." in output.

**Solution:**
```bash
# Longer sequence
python train_transformer_ultra_fast.py --seq_len 2048 ...

# Check GPU
watch -n 1 nvidia-smi
```

### "torch.compile fails"

**Error:** `Python.h: No such file or directory`

**Solution:**
```bash
sudo apt-get install python3.12-dev build-essential
```

### "Flash Attention install fails"

**Error:** Various CUDA compilation errors

**Solution:**
```bash
# Skip Flash for now
python train_transformer_ultra_fast.py --no_flash ...

# OR try pre-built wheels (if available)
pip install flash-attn --no-build-isolation --no-cache-dir
```

### "Out of memory"

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
python train_transformer_ultra_fast.py --batch_size 8 ...

# OR reduce sequence length
python train_transformer_ultra_fast.py --seq_len 1024 ...
```

### "Speedup plateaus at 3×"

**Likely cause:** Attention cost is still low at current L.

**Solution:**
```bash
# Longer sequences
python train_transformer_ultra_fast.py --seq_len 4096 --batch_size 4 ...

# More aggressive skipping
python train_transformer_ultra_fast.py --attn_keep_every 8 ...
```

---

## Expected Timeline

| Day | Task | Time | Cumulative Speedup |
|-----|------|------|--------------------|
| 1 | Baseline + Ultra (no compile) | 30 min | 3× |
| 2 | Add torch.compile | 1 hr | 6× |
| 3 | Add Flash Attention | 2 hr | 9-12× |
| 4 | Hyperparameter tuning | 3 hr | 10-15× |
| 5 | Triton kernels | 4 hr | 12-18× |
| 6-7 | Production testing | 8 hr | Validated |

---

## Success Criteria

By end of Week 1, you should have:

- ✅ **6× minimum speedup** (realistic for most setups)
- ✅ **10× target speedup** (achievable with all optimizations)
- ✅ **Loss within 0.02** of baseline (maintained quality)
- ✅ **Reproducible benchmark** (can re-run anytime)

---

## Next Steps (Week 2+)

Once you hit 10×:

1. **Integrate into your research:** Use ultra_fast.py as template for your models
2. **Scale up:** Multi-GPU training (PyTorch DDP)
3. **Publish results:** Write blog post / research paper
4. **Contribute back:** Submit benchmarks to GitHub
5. **Build applications:** Use the speedup for real-world problems

---

## Getting Help

**If stuck:**

1. Check `ML_OPTIMIZATION_GUIDE.md` for detailed explanations
2. Open GitHub issue with benchmark results
3. Join discussions: https://github.com/followthesapper/quantum-hybrid-simulator/discussions

**Good luck reaching 10×!** 🚀

---

## Quick Commands Reference

```bash
# Day 1: Baseline
python benchmark_speedup.py --quick

# Day 2: + Compile
python train_transformer_ultra_fast.py --seq_len 2048 --batch_size 16 --epochs 1 --compile --log_csv ../runs/test.csv

# Day 3: + Flash
python train_transformer_ultra_fast.py --seq_len 2048 --batch_size 16 --epochs 1 --compile --log_csv ../runs/test_flash.csv

# Day 4: Tune
for k in 2 4 6 8; do
    python train_transformer_ultra_fast.py --seq_len 2048 --batch_size 16 --epochs 1 --attn_keep_every $k --compile --log_csv ../runs/sweep_k${k}.csv
done

# Day 5: Longer context
python train_transformer_ultra_fast.py --seq_len 4096 --batch_size 8 --epochs 1 --attn_keep_every 8 --compile --log_csv ../runs/L4096.csv

# Compare all
python -c "
import pandas as pd
import glob
for csv in sorted(glob.glob('../runs/*.csv')):
    try:
        df = pd.read_csv(csv)
        val = df[df.split == 'val'].iloc[-1]
        print(f'{csv:40s}: {val.tok_per_s:8.0f} tok/s, loss={val.loss:.4f}')
    except: pass
"
```

---

**Now go run `python benchmark_speedup.py --quick` and see your first speedup!** 🎉
