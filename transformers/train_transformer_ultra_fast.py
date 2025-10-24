#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Fast AQED Transformer with Maximum Optimizations

Key optimizations:
1. torch.compile() for all critical paths
2. Flash Attention for any full attention (if available)
3. Minimal routing overhead (precomputed, static patterns)
4. Simplified mixer (pure linear ops, no scatter)
5. FP16 AMP throughout
6. Fused kernels where possible

Target: 10× speedup over baseline at L ≥ 1024

Usage:
  python train_transformer_ultra_fast.py \
    --seq_len 2048 --batch_size 16 --epochs 1 \
    --attn_keep_every 4 --compile \
    --log_csv ../runs/ultra_fast.csv
"""
import os, time, math, csv, argparse, random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- AMP compatibility ----------------
try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP = False

# Try Flash Attention
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

# ---------------- Config ----------------
@dataclass
class Config:
    vocab_size: int = 32000
    seq_len: int = 2048
    batch_size: int = 16
    train_batches: int = 1500
    val_batches: int = 100

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    amp: bool = True

    # AQED knobs
    attn_keep_every: int = 4  # Full attention every N layers
    use_flash: bool = _HAS_FLASH  # Use Flash Attention if available
    compile: bool = False  # torch.compile everything

    # Logging
    log_csv: Optional[str] = None
    log_every: int = 50

# ---------------- Synthetic Zipf data ----------------
class ZipfDataset:
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
            inp, tgt = x[0], x[1]
            yield inp.to(torch.long), tgt.to(torch.long)

    def __len__(self):
        return self.batches

# ---------------- Ultra-Fast AQED Mixer ----------------
class UltraFastMixer(nn.Module):
    """
    Simplified mixer: just two linear layers + gating, no scatter.
    Compiles to a single fused kernel.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        mix = torch.tanh(self.lin(x))
        g = torch.sigmoid(self.gate(x))
        out = g * mix + (1.0 - g) * x
        return self.drop(out)

# ---------------- Hybrid Block ----------------
class UltraHybridBlock(nn.Module):
    """
    Alternates between full attention (every attn_keep_every layers) and mixer.
    Uses Flash Attention if available.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, layer_idx, attn_keep_every, use_flash):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_attention = (layer_idx % attn_keep_every == 0)
        self.use_flash = use_flash and _HAS_FLASH

        self.ln1 = nn.LayerNorm(d_model)
        if self.use_attention:
            if self.use_flash:
                # Flash Attention path (requires specific input format)
                self.attn = None  # We'll call flash_attn_func directly
                self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
                self.out_proj = nn.Linear(d_model, d_model, bias=False)
            else:
                self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        else:
            self.mixer = UltraFastMixer(d_model, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        # Fused FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        h = self.ln1(x)

        if self.use_attention:
            if self.use_flash and _HAS_FLASH:
                # Flash Attention path
                B, L, D = h.shape
                qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.d_head)
                # flash_attn_func expects [B, L, 3, H, D_head]
                attn_out = flash_attn_func(qkv, dropout_p=0.0 if not self.training else 0.1)
                attn_out = attn_out.reshape(B, L, D)
                attn_out = self.out_proj(attn_out)
            else:
                # Standard attention
                attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
        else:
            # Mixer path
            x = x + self.mixer(h)

        # FFN
        y = self.ffn(self.ln2(x))
        x = x + y
        return x

# ---------------- Model ----------------
class UltraFastTransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([
            UltraHybridBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                layer_idx=i,
                attn_keep_every=cfg.attn_keep_every,
                use_flash=cfg.use_flash
            ) for i in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.long)
        B, L = x.shape
        h = self.tok(x) + self.pos[:, :L, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

# ---------------- Train / Eval ----------------
def _ensure_batch(inp, tgt):
    if inp.dim() == 1: inp = inp.unsqueeze(0)
    if tgt.dim() == 1: tgt = tgt.unsqueeze(0)
    return inp.to(torch.long), tgt.to(torch.long)

def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config, phase="train", step0=0, logf=None):
    model.train(phase == "train")
    total_loss, total_tok, max_mem_mib = 0.0, 0, 0.0
    start_t = time.time()
    step = step0

    for it, (inp, tgt) in enumerate(dataset):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        inp, tgt = _ensure_batch(inp, tgt)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=cfg.amp):
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
            max_mem_mib = max(max_mem_mib, mem)
        except Exception:
            pass

        if logf and (step % cfg.log_every == 0):
            elapsed = time.time() - start_t
            tok_per_s = total_tok / max(1e-9, elapsed)
            logf.writerow({
                "step": step,
                "split": phase,
                "loss": total_loss / max(1, total_tok),
                "ppl": math.exp(min(20.0, total_loss / max(1, total_tok))),
                "tok_per_s": tok_per_s,
                "max_mem_mib": max_mem_mib,
                "elapsed_s": elapsed,
            })

        step += 1

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem_mib, elapsed, step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--train_batches", type=int, default=1500)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--attn_keep_every", type=int, default=4)
    ap.add_argument("--no_flash", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=50)

    args = ap.parse_args()
    cfg = Config(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
        attn_keep_every=args.attn_keep_every,
        use_flash=_HAS_FLASH and not args.no_flash,
        compile=args.compile,
        log_csv=args.log_csv,
        log_every=args.log_every,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); random.seed(1337)

    # Data
    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds   = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches,   device=device)

    # Model
    model = UltraFastTransformerLM(cfg).to(device)

    # torch.compile if requested
    if cfg.compile:
        print("⚡ Compiling model with torch.compile()...")
        model = torch.compile(model, mode="max-autotune")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp)

    # Logging
    fobj = None; logf = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj = open(cfg.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s"
        ])
        logf.writeheader()

    global_step = 0
    best_val = float("inf")

    print(f"Config: L={cfg.seq_len}, B={cfg.batch_size}, attn_every={cfg.attn_keep_every}")
    if cfg.use_flash:
        print("✓ Using Flash Attention")
    if cfg.compile:
        print("✓ Using torch.compile")

    for epoch in range(cfg.epochs):
        tr = run_epoch(model, train_ds, optimizer, scaler, device, cfg,
                       "train", step0=global_step, logf=logf)
        train_loss, train_ppl, train_tps, train_mem, train_time, global_step = tr

        with torch.no_grad():
            va = run_epoch(model, val_ds, optimizer, scaler, device, cfg,
                           "val", step0=global_step, logf=logf)
        val_loss, val_ppl, val_tps, val_mem, val_time, global_step = va

        if cfg.log_csv:
            logf.writerow({
                "step": global_step, "split": "val",
                "loss": val_loss, "ppl": val_ppl,
                "tok_per_s": val_tps, "max_mem_mib": val_mem, "elapsed_s": val_time
            })
        best_val = min(best_val, val_loss)
        print(f"Epoch {epoch+1}/{cfg.epochs}  "
              f"train loss {train_loss:.4f} ppl {train_ppl:.2f} tps {train_tps:.1f} | "
              f"val loss {val_loss:.4f} ppl {val_ppl:.2f} tps {val_tps:.1f}")

    if fobj:
        fobj.close()

    print(f"✓ Done. Best val loss={best_val:.4f}. CSV -> {cfg.log_csv or '(none)'}")

if __name__ == "__main__":
    main()
