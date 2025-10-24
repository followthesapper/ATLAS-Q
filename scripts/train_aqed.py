#!/usr/bin/env python3
"""
AQED Transformer Training Script

Unified training script using the quantum_hybrid_system.aqed package.

Usage:
    # Quick test
    python scripts/train_aqed.py --seq_len 2048 --epochs 1

    # Full training with all optimizations
    python scripts/train_aqed.py --seq_len 4096 --batch_size 8 --epochs 3 \
        --attn_keep_every 8 --compile --log_csv runs/my_training.csv

    # Or use the wrapper script:
    ./run_training.sh --seq_len 4096 --epochs 3 --compile
"""

import os
import sys
import time
import math
import csv
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import AQED from package
from quantum_hybrid_system.aqed import AQEDTransformerLM, AQEDConfig

# AMP compatibility
try:
    from torch.amp import GradScaler, autocast
except:
    from torch.cuda.amp import GradScaler, autocast


class ZipfDataset:
    """Synthetic Zipf-distributed dataset for language modeling"""
    def __init__(self, vocab, seq_len, batches, alpha=1.1, device="cuda"):
        self.vocab, self.seq_len, self.batches = vocab, seq_len, batches
        self.device = device
        ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
        probs = ranks.pow(-alpha)
        probs = probs / probs.sum()
        self.probs = probs.to(device)

    @torch.no_grad()
    def __iter__(self):
        for _ in range(self.batches):
            x = torch.multinomial(self.probs, self.seq_len * 2, replacement=True)
            x = x.reshape(2, self.seq_len)
            yield x[0].to(torch.long), x[1].to(torch.long)

    def __len__(self):
        return self.batches


def run_epoch(model, dataset, optimizer, scaler, device, config, phase="train", step0=0, logf=None):
    """Run one epoch of training or validation"""
    model.train(phase == "train")
    total_loss, total_tok, max_mem = 0.0, 0, 0.0
    start_t = time.time()
    step = step0

    for inp, tgt in dataset:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=config.amp):
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs_tokens = inp.numel()
        total_tok += bs_tokens
        total_loss += float(loss.detach()) * bs_tokens

        try:
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            max_mem = max(max_mem, mem)
        except:
            pass

        if logf and (step % config.log_every == 0):
            elapsed = time.time() - start_t
            tok_per_s = total_tok / max(1e-9, elapsed)
            logf.writerow({
                "step": step,
                "split": phase,
                "loss": total_loss / max(1, total_tok),
                "ppl": math.exp(min(20.0, total_loss / max(1, total_tok))),
                "tok_per_s": tok_per_s,
                "max_mem_mib": max_mem,
                "elapsed_s": elapsed,
            })

        step += 1

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem, elapsed, step


def main():
    ap = argparse.ArgumentParser(description="Train AQED Transformer")

    # Model architecture
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train_batches", type=int, default=1500)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--no_amp", action="store_true")

    # AQED optimizations
    ap.add_argument("--attn_keep_every", type=int, default=4,
                    help="Attention every N layers (higher = faster)")
    ap.add_argument("--no_flash", action="store_true",
                    help="Disable memory-efficient attention")
    ap.add_argument("--compile", action="store_true",
                    help="Use torch.compile for 2-3× additional speedup")

    # Logging
    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()

    # Create config
    config = AQEDConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        attn_keep_every=args.attn_keep_every,
        use_flash=not args.no_flash,
        compile=args.compile,
        log_csv=args.log_csv,
        log_every=args.log_every,
    )

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    random.seed(1337)

    # Data
    train_ds = ZipfDataset(config.vocab_size, config.seq_len, args.train_batches, device=device)
    val_ds = ZipfDataset(config.vocab_size, config.seq_len, args.val_batches, device=device)

    # Model
    print("=" * 60)
    print("AQED Transformer Training")
    print("=" * 60)
    print(f"Config: L={config.seq_len}, B={config.batch_size}, "
          f"layers={config.n_layers}, attn_every={config.attn_keep_every}")

    model = AQEDTransformerLM(config).to(device)

    # torch.compile
    if config.compile:
        print("⚡ Compiling model with torch.compile()...")
        print("   (First epoch will be slow - compilation overhead)")
        model = torch.compile(model, mode="max-autotune")

    # Check Flash/SDPA
    if config.use_flash:
        if hasattr(F, 'scaled_dot_product_attention'):
            print("✓ Using PyTorch memory-efficient attention (SDPA)")
        else:
            print("ℹ Memory-efficient attention not available")

    print("=" * 60)
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.amp)

    # Logging
    fobj, logf = None, None
    if config.log_csv:
        os.makedirs(os.path.dirname(config.log_csv) or ".", exist_ok=True)
        fobj = open(config.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step", "split", "loss", "ppl", "tok_per_s", "max_mem_mib", "elapsed_s"
        ])
        logf.writeheader()

    # Training loop
    global_step = 0
    best_val = float("inf")

    for epoch in range(config.epochs):
        tr = run_epoch(model, train_ds, optimizer, scaler, device, config,
                       "train", step0=global_step, logf=logf)
        train_loss, train_ppl, train_tps, train_mem, train_time, global_step = tr

        with torch.no_grad():
            va = run_epoch(model, val_ds, optimizer, scaler, device, config,
                           "val", step0=global_step, logf=logf)
        val_loss, val_ppl, val_tps, val_mem, val_time, global_step = va

        if config.log_csv and logf:
            logf.writerow({
                "step": global_step, "split": "val",
                "loss": val_loss, "ppl": val_ppl,
                "tok_per_s": val_tps, "max_mem_mib": val_mem, "elapsed_s": val_time
            })

        best_val = min(best_val, val_loss)
        print(f"Epoch {epoch+1}/{config.epochs}  "
              f"train loss {train_loss:.4f} ppl {train_ppl:.2f} tps {train_tps:.1f} | "
              f"val loss {val_loss:.4f} ppl {val_ppl:.2f} tps {val_tps:.1f}")

    if fobj:
        fobj.close()

    print()
    print("=" * 60)
    print(f"✓ Training complete! Best val loss={best_val:.4f}")
    if config.log_csv:
        print(f"✓ Results saved to: {config.log_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
