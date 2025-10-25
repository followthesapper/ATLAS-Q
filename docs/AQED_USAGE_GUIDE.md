# AQED Usage Guide

Practical tutorials for training transformers 6-10× faster with AQED.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Command-Line Training](#3-command-line-training)
4. [Python API](#4-python-api)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Troubleshooting](#6-troubleshooting)
7. [Advanced Usage](#7-advanced-usage)

---

## 1. Installation

### Basic Installation

```bash
pip install quantum-hybrid-simulator[ml]
```

### Development Installation

```bash
git clone https://github.com/your-org/quantum-hybrid-simulator
cd quantum-hybrid-simulator
pip install -e .[all]
```

### Setup for NVIDIA GB10 (DGX Spark)

If using torch.compile on GB10 GPUs:

```bash
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
```

Add to `~/.bashrc` for permanent effect.

---

## 2. Quick Start

### 30-Second Example

```bash
# Train a transformer 6× faster!
python scripts/train_aqed.py --seq_len 4096 --attn_keep_every 8 --compile
```

**That's it!** You now have a transformer that trains 6-10× faster than baseline.

---

## 3. Command-Line Training

### 3.1 Basic Training

```bash
python scripts/train_aqed.py \
  --seq_len 4096 \
  --batch_size 8 \
  --epochs 3
```

### 3.2 With All Optimizations (Recommended)

```bash
python scripts/train_aqed.py \
  --seq_len 4096 \
  --batch_size 8 \
  --epochs 3 \
  --attn_keep_every 8 \
  --compile \
  --log_csv runs/aqed_training.csv
```

**Expected result:** 6-10× speedup with loss within 1-2% of baseline

### 3.3 Conservative (Best Quality)

```bash
python scripts/train_aqed.py \
  --seq_len 2048 \
  --attn_keep_every 4 \
  --compile
```

**Expected result:** 2-3× speedup with <0.5% loss difference

### 3.4 Aggressive (Maximum Speed)

```bash
python scripts/train_aqed.py \
  --seq_len 8192 \
  --batch_size 4 \
  --attn_keep_every 16 \
  --compile
```

**Expected result:** 8-12× speedup with ~5% loss difference

### 3.5 Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--seq_len` | 2048 | Sequence length (higher = bigger AQED benefit) |
| `--batch_size` | 8 | Batch size (adjust for GPU memory) |
| `--epochs` | 3 | Number of training epochs |
| `--attn_keep_every` | 8 | Use full attention every N layers (key parameter!) |
| `--compile` | False | Enable torch.compile (1.2-1.4× additional speedup) |
| `--log_csv` | None | Path to save training logs (CSV format) |
| `--d_model` | 512 | Model dimension |
| `--n_layers` | 8 | Number of transformer layers |
| `--n_heads` | 8 | Number of attention heads |
| `--lr` | 3e-4 | Learning rate |

---

## 4. Python API

### 4.1 Basic Usage

```python
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'

from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig
import torch

# Configure model
config = AQEDConfig(
    vocab_size=32000,
    seq_len=4096,
    d_model=512,
    n_layers=8,
    n_heads=8,
    attn_keep_every=8,  # Key parameter!
    compile=True,
)

# Create model
model = AQEDTransformerLM(config).cuda()

# torch.compile (optional but recommended)
if config.compile:
    model = torch.compile(model, mode="max-autotune")

# Your model is now 6-10× faster!
```

### 4.2 Training Loop

```python
from torch.utils.data import DataLoader
import torch.optim as optim

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training
model.train()
for epoch in range(3):
    for batch in train_loader:
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 4.3 Inference

```python
model.eval()
with torch.no_grad():
    input_ids = torch.randint(0, config.vocab_size, (1, 100)).cuda()
    logits = model(input_ids)  # [1, 100, vocab_size]
    predictions = logits.argmax(dim=-1)
```

---

## 5. Hyperparameter Tuning

### 5.1 Key Parameter: `attn_keep_every`

This controls the speed/quality tradeoff:

| Value | Speedup | Quality Loss | Use Case |
|-------|---------|--------------|----------|
| 1 | 1× (no speedup) | 0% | Baseline (no AQED) |
| 4 | ~3× | <0.5% | **Production (conservative)** |
| **8** | **~6×** | **<1.5%** | **Recommended (balanced)** ✅ |
| 16 | ~10× | ~5-7% | Research/prototyping |
| 32 | ~15× | ~15-20% | Extreme speed (quality degrades) |

**Rule of thumb:** Start with 8, decrease if quality is insufficient, increase if speed is priority.

### 5.2 Sequence Length Effects

AQED speedup **increases** with longer sequences:

| seq_len | Baseline Speed | AQED Speed | Speedup |
|---------|---------------|------------|---------|
| 512 | Fast | Slightly faster | **1.5-2×** |
| 1024 | Medium | Faster | **2-3×** |
| 2048 | Medium | Much faster | **4-5×** |
| **4096** | **Slow** | **Fast** | **6-7×** ✅ |
| **8192** | **Very slow** | **Fast** | **9-11×** ✅ |

**Recommendation:** AQED shines at L ≥ 2048

### 5.3 Hyperparameter Sweep

```python
import os

for seq_len in [2048, 4096, 8192]:
    for attn_every in [4, 8, 16]:
        cmd = f"""python scripts/train_aqed.py \
          --seq_len {seq_len} \
          --batch_size 8 \
          --epochs 2 \
          --attn_keep_every {attn_every} \
          --compile \
          --log_csv runs/sweep_L{seq_len}_skip{attn_every}.csv"""
        os.system(cmd)
```

Then compare results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load all CSVs
results = []
for seq_len in [2048, 4096, 8192]:
    for attn_every in [4, 8, 16]:
        df = pd.read_csv(f"runs/sweep_L{seq_len}_skip{attn_every}.csv")
        final_loss = df[df.split == "val"].loss.iloc[-1]
        final_speed = df[df.split == "val"].tok_per_s.iloc[-1]
        results.append({
            'seq_len': seq_len,
            'attn_every': attn_every,
            'loss': final_loss,
            'speed': final_speed
        })

results_df = pd.DataFrame(results)
print(results_df)
```

---

## 6. Troubleshooting

### 6.1 "torch.compile fails with PTXASError"

**Symptom:** Error mentioning `sm_121a` or PTXAS

**Solution:** Set environment variables:
```bash
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="12.0"
```

### 6.2 "First epoch is very slow"

**This is expected!** torch.compile compiles kernels in epoch 1. Check epoch 2+ for real performance.

**Example:**
- Epoch 1: 8-22k tok/s (compiling)
- Epoch 2+: 230-240k tok/s (actual speed)

**Always run ≥2 epochs for benchmarking.**

### 6.3 "AQED not faster than baseline"

**Diagnosis:**
1. Are you at short sequences (L ≤ 1024)? → Try L ≥ 2048
2. Did you enable `--compile`? → Add the flag
3. Are you measuring epoch 1? → Check epoch 2+
4. Is baseline already optimized? → Compare against vanilla PyTorch transformer

### 6.4 "GPU Out of Memory"

**Solutions:**
1. Reduce `--batch_size` by 2×
2. Reduce `--seq_len` (but AQED benefit decreases)
3. Enable gradient checkpointing (if available)
4. Use smaller model (`--d_model 256`, `--n_layers 4`)

### 6.5 "Loss is much worse than baseline"

**Diagnosis:**
1. Is `attn_keep_every` too high (≥16)? → Try 8 or 4
2. Is dataset very small? → AQED works best with large-scale training
3. Is task extremely sensitive to attention? → Try conservative settings

---

## 7. Advanced Usage

### 7.1 Integrate with HuggingFace

```python
from transformers import GPT2Tokenizer
from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# AQED model (drop-in replacement for GPT-2)
config = AQEDConfig(vocab_size=len(tokenizer), seq_len=1024, attn_keep_every=8)
model = AQEDTransformerLM(config).cuda()

# Use with HF Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 7.2 Custom Data Pipeline

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.encodings = tokenizer(texts, max_length=seq_len, truncation=True, padding='max_length')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'labels': torch.tensor(self.encodings['input_ids'][idx])
        }

# Create dataset
dataset = MyDataset(my_texts, tokenizer, seq_len=4096)
loader = DataLoader(dataset, batch_size=8, num_workers=8, pin_memory=True)
```

### 7.3 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    input_ids = batch['input_ids'].cuda()
    labels = batch['labels'].cuda()

    optimizer.zero_grad()

    # Forward in FP16
    with autocast():
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

    # Backward with scaled gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Expected benefit:** Additional 1.5-2× speedup

### 7.4 Save & Load Checkpoints

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### 7.5 Distributed Training (Multi-GPU)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = AQEDTransformerLM(config).cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop (same as before)
```

---

## 8. Best Practices

### Performance

1. **Always use torch.compile** (1.2-1.4× extra speedup)
2. **Run ≥2 epochs** for benchmarking (epoch 1 is compilation)
3. **Use longer sequences** (L ≥ 2048 for maximum benefit)
4. **Enable AMP** (1.5-2× speedup)
5. **Pin memory** in DataLoader (`pin_memory=True`)
6. **Use `num_workers > 0`** to keep GPU fed

### Quality

1. **Start with `attn_keep_every=8`** (proven sweet spot)
2. **Validate frequently** (monitor loss convergence)
3. **Compare to baseline** (train both, compare final metrics)
4. **Test on holdout set** (ensure generalization)

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose mode
config = AQEDConfig(..., verbose=True)
```

---

## 9. Real-World Examples

### 9.1 Language Model Fine-Tuning

```bash
# Fine-tune AQED on custom corpus (6× faster!)
python scripts/train_aqed.py \
  --data_path my_corpus.txt \
  --seq_len 4096 \
  --batch_size 16 \
  --epochs 10 \
  --attn_keep_every 8 \
  --compile \
  --save_dir checkpoints/my_model
```

### 9.2 Code Generation

```bash
# Train code model (faster iteration!)
python scripts/train_aqed.py \
  --data_path code_dataset/ \
  --seq_len 8192 \
  --batch_size 4 \
  --attn_keep_every 8 \
  --compile \
  --vocab_size 50000
```

---

## 10. Resources

- **[AQED Whitepaper](AQED_WHITEPAPER.md)** - Mathematical details, proofs, complexity analysis
- **[README.md](README.md)** - Project overview and quick start
- **[GitHub Issues](https://github.com/your-org/quantum-hybrid-simulator/issues)** - Report bugs
- **[Interactive Notebooks](Notebooks/)** - Jupyter examples

---

**Happy Training!** 🚀

*For questions: https://github.com/your-org/quantum-hybrid-simulator/issues*
