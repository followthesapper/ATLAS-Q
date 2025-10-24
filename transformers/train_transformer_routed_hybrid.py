#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routed Hybrid Transformer (faster edition)
- A fraction of tokens (route_frac) use full attention; the rest use a cheap mixer.
- Faster merge via scatter (no boolean index writes).
- Optional cached routing updated every N steps to avoid per-iter topk cost.
- Robust to inputs shaped [L] or [B,L].
- AMP-safe dtype handling when merging branches.

Usage (from transformers/):
  python train_transformer_routed_hybrid.py \
    --seq_len 512 --batch_size 64 \
    --train_batches 3000 --val_batches 200 \
    --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 --epochs 1 \
    --route_frac 0.15 --mixer_depth 2 --mixer_stride 1 \
    --route_update_every 8 \
    --log_csv ../runs/aqed_routed_hybrid.csv
"""
import os, time, math, csv, argparse, random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- AMP compatibility ----------------
try:
    from torch.amp import GradScaler as _NewGradScaler
    from torch.amp import autocast as _new_autocast
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler as _OldGradScaler
    from torch.cuda.amp import autocast as _old_autocast
    _USE_NEW_AMP = False

def make_scaler(enabled: bool):
    return (_NewGradScaler(enabled=enabled) if _USE_NEW_AMP else _OldGradScaler(enabled=enabled))

class Autocast:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        if _USE_NEW_AMP:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.ctx = _new_autocast(device_type, enabled=self.enabled)
        else:
            self.ctx = _old_autocast(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, et, ev, tb):
        return self.ctx.__exit__(et, ev, tb)

# ---------------- Config ----------------
@dataclass
class Config:
    vocab_size: int = 32000
    seq_len: int = 512
    batch_size: int = 64
    train_batches: int = 3000
    val_batches: int = 200

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    amp: bool = True

    # Routing/mixing knobs
    route_frac: float = 0.15
    mixer_depth: int = 2
    mixer_stride: int = 1  # >1: only attempt to mix every Nth position

    # Recompute routing every N steps (reduce topk overhead)
    route_update_every: int = 1

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
            inp, tgt = x[0], x[1]  # each is [L]
            yield inp.to(torch.long), tgt.to(torch.long)

    def __len__(self):
        return self.batches

# ---------------- AQED-style mixer (vectorized & cheap) ----------------
class AQEDMixVec(nn.Module):
    """
    Cheap pairwise mixing on a fraction of positions.
    Vectorized: fixed strided pairing (deterministic & fast).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model, bias=False)
        self.lin2 = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor, depth: int = 1, stride: int = 1) -> torch.Tensor:
        # x: [B, L, D]
        if depth <= 0:
            return x
        B, L, D = x.shape
        out = x
        # Create fixed, strided pairs: (0,1), (2,3), ... shifted by stride each layer
        for d in range(depth):
            offset = (d * stride) % max(1, L // 2)
            i = torch.arange(0, L - 1, 2, device=x.device)
            j = i + 1
            if offset > 0:
                i = (i + 2 * offset) % L
                j = (j + 2 * offset) % L

            xi = out[:, i, :]  # [B, P, D]
            xj = out[:, j, :]

            mix = torch.tanh(self.lin1(xi) + self.lin2(xj))
            g = torch.sigmoid(self.gate(torch.cat([xi, xj], dim=-1)))
            mixed = g * mix + (1.0 - g) * xi

            out = out.clone()
            out[:, i, :] = self.drop(mixed)
        return out

# ---------------- Transformer blocks ----------------
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class HybridRoutedBlock(nn.Module):
    """
    LayerNorm -> (Attention on routed subset) + (AQED mixer on the rest) -> scatter-merge -> Residual -> FFN -> Residual
    Speed tricks:
      - scatter merge with integer indices (no boolean masks).
      - cached routing indices updated every N steps.
      - AMP dtype alignment when merging.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout,
                 route_frac=0.15, mixer_depth=2, mixer_stride=1, route_update_every=1):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mixer = AQEDMixVec(d_model, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.route_frac = route_frac
        self.mixer_depth = mixer_depth
        self.mixer_stride = mixer_stride
        self.route_update_every = max(1, route_update_every)

        # light saliency scorer (linear + abs)
        self.score = nn.Linear(d_model, 1, bias=False)
        nn.init.xavier_uniform_(self.score.weight)

        # cache
        self._cached_idx = None   # [B, k]
        self._last_step = -1

    def _saliency(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, L]
        s = self.score(x).squeeze(-1).abs()
        return s

    def _route_indices(self, h: torch.Tensor, step: int) -> torch.Tensor:
        # recompute only if needed
        if (self._cached_idx is None) or (step < 0) or (step % self.route_update_every == 0):
            B, L, D = h.shape
            k = max(1, int(self.route_frac * L))
            sal = self._saliency(h)                # [B, L]
            topk_idx = torch.topk(sal, k=k, dim=1, largest=True, sorted=False).indices  # [B, k]
            self._cached_idx = topk_idx
            self._last_step = step
        return self._cached_idx

    def forward(self, x: torch.Tensor, step: int = -1) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        h = self.ln_in(x)
        ref_dtype = h.dtype

        # --- Routing decision (possibly cached) ---
        topk_idx = self._route_indices(h, step)    # [B, k]
        k = topk_idx.size(1)

        # --- Attention branch (queries = selected tokens) ---
        # Gather queries
        idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, D)    # [B, k, D]
        q_sel = h.gather(dim=1, index=idx_expand)                # [B, k, D]
        attn_out, _ = self.attn(q_sel, h, h, need_weights=False) # [B, k, D]
        attn_out = attn_out.to(ref_dtype)

        # --- Mixer branch for everyone (cheap) ---
        mixed = self.mixer(h, depth=self.mixer_depth, stride=self.mixer_stride)
        mixed = mixed.to(ref_dtype)

        # --- Scatter-merge (write attention results back into mixed) ---
        out = mixed.scatter(dim=1, index=idx_expand, src=attn_out)

        # Residual + FFN
        x = x + out
        y = self.ffn(self.ln1(x))
        x = self.ln2(x + y)
        return x

class RoutedHybridTransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([
            HybridRoutedBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                              route_frac=cfg.route_frac,
                              mixer_depth=cfg.mixer_depth,
                              mixer_stride=cfg.mixer_stride,
                              route_update_every=cfg.route_update_every)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, step: int = -1) -> torch.Tensor:
        # Accept [L] or [B,L]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.long)

        B, L = x.shape
        h = self.tok(x) + self.pos[:, :L, :]
        for blk in self.blocks:
            h = blk(h, step=step)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

# ---------------- Train / Eval ----------------
def _ensure_batch(inp: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Accept [L] or [B,L]; cast to long for embeddings
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

        with Autocast(enabled=cfg.amp):
            logits = model(inp, step=step)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # accounting
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
                "route_frac": cfg.route_frac,
                "mixer_depth": cfg.mixer_depth
            })

        step += 1

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem_mib, elapsed, step

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
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--route_frac", type=float, default=0.15)
    ap.add_argument("--mixer_depth", type=int, default=2)
    ap.add_argument("--mixer_stride", type=int, default=1)
    ap.add_argument("--route_update_every", type=int, default=1)

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
        route_frac=args.route_frac,
        mixer_depth=args.mixer_depth,
        mixer_stride=args.mixer_stride,
        route_update_every=args.route_update_every,
        log_csv=args.log_csv,
        log_every=args.log_every,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); random.seed(1337)

    # Data
    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds   = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches,   device=device)

    # Model/opt
    model = RoutedHybridTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(enabled=cfg.amp)

    # Logging
    fobj = None; logf = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj = open(cfg.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s",
            "route_frac","mixer_depth"
        ])
        logf.writeheader()

    global_step = 0
    best_val = float("inf")

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
                "tok_per_s": val_tps, "max_mem_mib": val_mem, "elapsed_s": val_time,
                "route_frac": cfg.route_frac, "mixer_depth": cfg.mixer_depth
            })
        best_val = min(best_val, val_loss)
        print(f"Epoch {epoch+1}/{cfg.epochs}  "
              f"train loss {train_loss:.4f} ppl {train_ppl:.2f} tps {train_tps:.1f} | "
              f"val loss {val_loss:.4f} ppl {val_ppl:.2f} tps {val_tps:.1f}")

    if fobj:
        fobj.close()

    print(f"Done. Best val loss={best_val:.4f}. CSV -> {cfg.log_csv or '(none)'}")

if __name__ == "__main__":
    main()
