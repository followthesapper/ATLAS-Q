#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer LM + Vectorized AQED mixer (no per-pair Python loops),
SDPA/FlashAttention, fused AdamW, optional torch.compile.
"""

import os, time, math, argparse, random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- fastest matmul kernels ---
try:
    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ gives big speedups
except Exception:
    pass

# --- AMP compatibility (new/old API) ---
try:
    from torch.amp import GradScaler as NewGradScaler
    from torch.amp import autocast as new_autocast
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler as OldGradScaler
    from torch.cuda.amp import autocast as old_autocast
    _USE_NEW_AMP = False

def make_scaler(enabled: bool):
    return (NewGradScaler(enabled=enabled) if _USE_NEW_AMP else OldGradScaler(enabled=enabled))

class AutocastCtx:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        try:
            from torch.amp import autocast as _new_autocast
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.ctx = _new_autocast(device_type, enabled=self.enabled)
        except Exception:
            from torch.cuda.amp import autocast as _old_autocast
            self.ctx = _old_autocast(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)

# --- SDPA/FlashAttention wrapper ---
from contextlib import contextmanager
from torch.nn.functional import scaled_dot_product_attention as sdpa

@contextmanager
def flash_attn_on():
    """Prefer flash attention where available; else mem-efficient."""
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            yield
    except Exception:
        yield

# ----------------- Config & Data -----------------
@dataclass
class Config:
    vocab_size: int = 32000
    seq_len:   int = 512
    batch_size:int = 64
    train_batches: int = 3000
    val_batches:   int = 200
    d_model: int = 512
    n_layers:int = 8
    n_heads: int = 8
    d_ff:    int = 2048
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 1
    amp: bool = True
    compile: bool = False
    # AQED mixer knobs (fixed here; controller version is separate)
    mixer_depth: int = 1
    pair_frac: float  = 0.15
    log_csv: str | None = None
    log_every: int = 50

class ZipfDataset:
    def __init__(self, vocab, seq_len, batches, alpha=1.1, device="cuda"):
        self.vocab, self.seq_len, self.batches = vocab, seq_len, batches
        self.device = device
        ranks = torch.arange(1, vocab + 1, dtype=torch.float64)
        probs = ranks.pow(-alpha); probs = probs / probs.sum()
        self.probs = probs.to(device)
    @torch.no_grad()
    def __iter__(self):
        for _ in range(self.batches):
            x = torch.multinomial(self.probs, self.seq_len * 2, replacement=True)
            x = x.reshape(2, self.seq_len)
            yield x[0].to(torch.long), x[1].to(torch.long)
    def __len__(self): return self.batches

# ----------------- Model -----------------
class SDPATransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, is_causal=True):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        B, L, D = q.shape
        q = q.view(B, L, self.nh, self.dh).transpose(1, 2)  # [B,H,L,dh]
        k = k.view(B, L, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dh).transpose(1, 2)
        with flash_attn_on():
            out = sdpa(q, k, v, attn_mask=None, is_causal=is_causal)  # [B,H,L,dh]
        out = out.transpose(1, 2).contiguous().view(B, L, self.nh * self.dh)
        x = x + self.proj(out)
        x = x + self.ff(self.ln2(x))
        return x

class AQEDMixLiteFast(nn.Module):
    """Vectorized pairwise mixer (no Python loop over pairs)."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model, bias=False)
        self.lin2 = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.gate.weight)

    @torch.no_grad()
    def _pair_indices(self, L: int, n_pairs: int, device):
        # sample 2*n_pairs unique positions, pair them
        idx = torch.randperm(L, device=device)[: 2 * n_pairs]
        i = idx[0::2]; j = idx[1::2]
        return i, j

    def forward(self, x: torch.Tensor, pair_frac: float, depth: int) -> torch.Tensor:
        if depth <= 0 or pair_frac <= 0:
            return x
        B, L, D = x.shape
        n_pairs = max(1, int((L // 2) * pair_frac))
        out = x
        for _ in range(depth):
            i, j = self._pair_indices(L, n_pairs, x.device)         # [P], [P]
            xi = out.index_select(1, i)                              # [B,P,D]
            xj = out.index_select(1, j)                              # [B,P,D]
            mix = torch.tanh(self.lin1(xi) + self.lin2(xj))          # [B,P,D]
            g = torch.sigmoid(self.gate(torch.cat([xi, xj], dim=-1)))# [B,P,D]
            mixed = self.drop(g * mix + (1.0 - g) * xi)              # [B,P,D]
            out = out.clone()
            out.scatter_(1, i.view(1, -1, 1).expand(B, -1, D), mixed)
        return out

class TinyTransformerLM_AQED(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([SDPATransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                     for _ in range(cfg.n_layers)])
        self.mixer = AQEDMixLiteFast(cfg.d_model, cfg.dropout)
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x, pair_frac=None, mixer_depth=None):
        if x.dim() == 1: x = x.unsqueeze(0)
        B, L = x.shape
        h = self.tok(x) + self.pos[:, :L, :]
        for blk in self.blocks:
            h = blk(h, is_causal=True)
        pf = float(self.cfg.pair_frac if pair_frac is None else pair_frac)
        md = int(self.cfg.mixer_depth if mixer_depth is None else mixer_depth)
        h = self.mixer(h, pair_frac=pf, depth=md)
        h = self.lnf(h)
        return self.head(h)

# ----------------- Train/Eval -----------------
def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config, phase="train"):
    model.train(phase == "train")
    tot_loss, tot_tok, max_mem = 0.0, 0, 0.0
    t0 = time.time()
    for step, (inp, tgt) in enumerate(dataset):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with AutocastCtx(enabled=cfg.amp):
            logits = model(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bs_tokens = int(inp.numel())
        tot_tok += bs_tokens
        tot_loss += float(loss.detach()) * bs_tokens
        try:
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        except Exception:
            pass
    elapsed = time.time() - t0
    tps = tot_tok / max(1e-9, elapsed)
    avg_loss = tot_loss / max(1, tot_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tps, max_mem, elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--train_batches", type=int, default=3000)
    ap.add_argument("--val_batches", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    # AQED mixer (fixed here)
    ap.add_argument("--mixer_depth", type=int, default=1)
    ap.add_argument("--pair_frac", type=float, default=0.15)
    ap.add_argument("--log_csv", type=str, default=None)
    args = ap.parse_args()

    cfg = Config(
        vocab_size=args.vocab_size, seq_len=args.seq_len, batch_size=args.batch_size,
        train_batches=args.train_batches, val_batches=args.val_batches,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, amp=not args.no_amp, compile=args.compile,
        mixer_depth=args.mixer_depth, pair_frac=args.pair_frac,
        log_csv=args.log_csv
    )

    torch.manual_seed(1337); random.seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds   = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches,   device=device)

    model = TinyTransformerLM_AQED(cfg).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay, fused=True)
    scaler = make_scaler(enabled=cfg.amp)

    # logging
    writer = None; fobj = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        import csv
        fobj = open(cfg.log_csv, "w", newline="")
        writer = csv.DictWriter(fobj, fieldnames=["step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s"])
        writer.writeheader()

    step = 0
    best = float("inf")
    for epoch in range(cfg.epochs):
        tr = run_epoch(model, train_ds, optimizer, scaler, device, cfg, "train")
        va = run_epoch(model, val_ds, optimizer, scaler, device, cfg, "val")
        (trL, trP, trT, trM, trS), (vaL, vaP, vaT, vaM, vaS) = tr, va
        if writer:
            writer.writerow({"step": step, "split": "train", "loss": trL, "ppl": trP, "tok_per_s": trT, "max_mem_mib": trM, "elapsed_s": trS})
            writer.writerow({"step": step, "split": "val",   "loss": vaL, "ppl": vaP, "tok_per_s": vaT, "max_mem_mib": vaM, "elapsed_s": vaS})
        step += 1
        best = min(best, vaL)
        print(f"Epoch {epoch+1}/{cfg.epochs}  train loss {trL:.4f} ppl {trP:.2f} tps {trT:.1f} | val loss {vaL:.4f} ppl {vaP:.2f} tps {vaT:.1f}")

    if fobj: fobj.close()
    print(f"Done. Best val loss={best:.4f}. CSV -> {cfg.log_csv or '(none)'}")

if __name__ == "__main__":
    main()
