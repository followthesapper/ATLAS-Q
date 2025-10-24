#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a small Transformer (LM toy) where a differentiable mixer is
controlled by your AQED probe in *real time* via a CSV produced by the
AQED diffusion script.

Usage (example):
  python train_transformer_aqed_controller.py \
    --seq_len 512 --batch_size 64 \
    --train_batches 3000 --val_batches 200 \
    --d_model 512 --n_layers 8 --n_heads 8 --d_ff 2048 --epochs 1 \
    --controller_csv ../runs/aqed_alpha_sweep.csv \
    --ctrl_update_steps 200 --ctrl_warmup_steps 200 \
    --pair_frac_min 0.05 --pair_frac_max 0.6 \
    --mixer_max_depth 3 \
    --log_csv ../runs/aqed_controller_stats.csv
"""
import os, time, math, csv, argparse, random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- AMP compatibility (works on old/new PyTorch) ----------
# Prefer new torch.amp.* if available; else fall back to torch.cuda.amp.*
try:
    from torch.amp import GradScaler as _NewGradScaler
    from torch.amp import autocast as _new_autocast
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler as _OldGradScaler
    from torch.cuda.amp import autocast as _old_autocast
    _USE_NEW_AMP = False

def make_scaler(enabled: bool):
    if _USE_NEW_AMP:
        # Older point releases may not accept device_type kw;
        # using the simplest ctor ensures compatibility.
        return _NewGradScaler(enabled=enabled)
    else:
        return _OldGradScaler(enabled=enabled)

class AutocastCtx:
    """Context manager that works across torch versions."""
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        try:
            # New API
            from torch.amp import autocast as _new_autocast  # noqa
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.ctx = _new_autocast(device_type, enabled=self.enabled)
        except Exception:
            # Old API
            from torch.cuda.amp import autocast as _old_autocast  # noqa
            self.ctx = _old_autocast(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)


# ---------- Config ----------
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

    # Controller / AQED CSV
    controller_csv: Optional[str] = None
    ctrl_warmup_steps: int = 200
    ctrl_update_steps: int = 200
    ema: float = 0.8

    # Map AQED → mixer knobs
    pair_frac_min: float = 0.05
    pair_frac_max: float = 0.6
    mixer_max_depth: int = 3

    # Logging
    log_csv: Optional[str] = None
    log_every: int = 50


# ---------- Synthetic dataset ----------
class ZipfDataset:
    """
    Yields (inp, tgt) pairs shaped [L] (1D). We normalize to [B,L] later.
    """
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


# ---------- AQED controller ----------
class AQEDController:
    """
    Watches an AQED CSV file with rows like:
      layer,chi_mean,chi_max,(optional)D_E,trunc_error_sum
    If D_E is absent per-row, we derive a slope-based proxy from chi_mean.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ema_chi = None
        self.ema_DE = None

    def _safe_float(self, s) -> Optional[float]:
        try:
            return float(s)
        except Exception:
            return None

    def _read_tail_rows(self, path, max_rows=128) -> List[dict]:
        if not path or not os.path.exists(path):
            return []
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        def is_numeric_layer(r):
            v = r.get("layer")
            return v is not None and str(v).strip().isdigit()
        rows = [r for r in rows if is_numeric_layer(r)]
        return rows[-max_rows:]

    def _derive_DE(self, chi_series: List[float]) -> float:
        if len(chi_series) < 4:
            return 0.0
        x = torch.arange(len(chi_series), dtype=torch.float64)
        y = torch.tensor(chi_series, dtype=torch.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x*x).sum().clamp_min(1e-8)
        slope = (x*y).sum() / denom
        norm = (y.abs().mean() + 1e-6)
        return float((slope / norm).clamp(-10, 10))

    def poll(self) -> Tuple[float, int]:
        base_pair = 0.15
        base_depth = 1
        path = self.cfg.controller_csv
        if not path:
            return base_pair, base_depth

        rows = self._read_tail_rows(path, max_rows=256)
        if not rows:
            return base_pair, base_depth

        chi_means, DE_vals = [], []
        for r in rows:
            cm = self._safe_float(r.get("chi_mean"))
            if cm is not None: chi_means.append(cm)
            de = self._safe_float(r.get("D_E"))
            if de is not None: DE_vals.append(de)

        if chi_means:
            cur = chi_means[-1]
            self.ema_chi = cur if self.ema_chi is None else self.cfg.ema * self.ema_chi + (1 - self.cfg.ema) * cur
        if DE_vals:
            cur = DE_vals[-1]
            self.ema_DE = cur if self.ema_DE is None else self.cfg.ema * self.ema_DE + (1 - self.cfg.ema) * cur
        else:
            if len(chi_means) >= 4:
                de_proxy = self._derive_DE(chi_means[-16:])
                self.ema_DE = de_proxy if self.ema_DE is None else self.cfg.ema * self.ema_DE + (1 - self.cfg.ema) * de_proxy

        # Map to knobs
        # pf from absolute chi level (more stable on short/flat probes)
        if self.ema_chi is None:
            pf = base_pair
        else:
            # normalize chi to [0,1] with a soft cap; 256–512 is a decent scale for your 6x6, chi=512 runs
            scale = 512.0
            chi_norm = max(0.0, min(1.0, self.ema_chi / scale))
            lo, hi = self.cfg.pair_frac_min, self.cfg.pair_frac_max
            pf = float(lo + (hi - lo) * chi_norm)
        if self.ema_chi is None:
            depth = base_depth
        else:
            # crude scale: encourage deeper mixing as χ grows
            scale = max(16.0, self.cfg.d_model / 32.0)
            frac = max(0.0, min(1.0, self.ema_chi / scale))
            depth = int(round(frac * self.cfg.mixer_max_depth))

        depth = max(0, min(self.cfg.mixer_max_depth, depth))
        pf = float(max(self.cfg.pair_frac_min, min(self.cfg.pair_frac_max, pf)))
        return pf, depth


# ---------- Mixer & Transformer ----------
class AQEDMixLite(nn.Module):
    """
    Lightweight differentiable pairwise mixer (does not call into TN code).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model, bias=False)
        self.lin2 = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor, pair_frac: float, depth: int) -> torch.Tensor:
        if depth <= 0 or pair_frac <= 0:
            return x
        B, L, D = x.shape
        n_pairs = max(1, int((L // 2) * pair_frac))
        out = x
        for _ in range(depth):
            perm = torch.randperm(L, device=x.device)
            pairs = perm.view(-1, 2)[:n_pairs]
            i = pairs[:, 0]
            j = pairs[:, 1]
            xi = out[:, i, :]
            xj = out[:, j, :]
            mix = torch.tanh(self.lin1(xi) + self.lin2(xj))
            g = torch.sigmoid(self.gate(torch.cat([xi, xj], dim=-1)))
            mixed = g * mix + (1 - g) * xi
            out = out.clone()
            out[:, i, :] = self.dropout(mixed)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + a)
        y = self.ff(x)
        x = self.ln2(x + y)
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.mixer = AQEDMixLite(cfg.d_model, cfg.dropout)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x, pair_frac=0.2, mixer_depth=1):
        # Accept [L] or [B, L]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.long)

        B, L = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :L, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.mixer(h, pair_frac=pair_frac, depth=mixer_depth)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


# ---------- Train/Eval ----------
def _ensure_batch(inp, tgt):
    """Normalize to [B,L] long tensors."""
    if getattr(inp, "dim", lambda: 2)() == 1:
        inp = inp.unsqueeze(0)
    if getattr(tgt, "dim", lambda: 2)() == 1:
        tgt = tgt.unsqueeze(0)
    return inp.to(torch.long), tgt.to(torch.long)

def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config,
              controller: Optional[AQEDController], phase="train",
              global_step_start=0, logf=None):
    model.train(phase == "train")
    total_loss, total_tok, max_mem_mib = 0.0, 0, 0.0
    global_step = global_step_start
    start_t = time.time()

    pair_frac, mixer_depth = 0.15, 1

    for _, (inp, tgt) in enumerate(dataset):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        if phase == "train":
            if controller and global_step >= cfg.ctrl_warmup_steps:
                if (global_step - cfg.ctrl_warmup_steps) % cfg.ctrl_update_steps == 0:
                    pf, md = controller.poll()
                    pair_frac, mixer_depth = pf, md

        with AutocastCtx(enabled=cfg.amp):
            # Ensure batch dimension and dtype for embedding lookup
            inp, tgt = _ensure_batch(inp, tgt)
            logits = model(inp, pair_frac=pair_frac, mixer_depth=mixer_depth)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs_tokens = inp.numel()
        total_tok += bs_tokens
        total_loss += float(loss.detach()) * bs_tokens

        try:
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            max_mem_mib = max(max_mem_mib, mem)
        except Exception:
            pass

        if logf and (global_step % cfg.log_every == 0):
            elapsed = time.time() - start_t
            tok_per_s = total_tok / max(1e-9, elapsed)
            logf.writerow({
                "step": global_step,
                "split": phase,
                "loss": total_loss / max(1, total_tok),
                "ppl": math.exp(min(20.0, total_loss / max(1, total_tok))),
                "tok_per_s": tok_per_s,
                "max_mem_mib": max_mem_mib,
                "elapsed_s": elapsed,
                "pair_frac": pair_frac,
                "mixer_depth": mixer_depth
            })

        global_step += 1

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem_mib, elapsed, global_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--train_batches", type=int, default=3000)
    p.add_argument("--val_batches", type=int, default=200)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--no_amp", action="store_true")

    # Controller options
    p.add_argument("--controller_csv", type=str, default=None)
    p.add_argument("--ctrl_warmup_steps", type=int, default=200)
    p.add_argument("--ctrl_update_steps", type=int, default=200)
    p.add_argument("--ema", type=float, default=0.8)
    p.add_argument("--pair_frac_min", type=float, default=0.05)
    p.add_argument("--pair_frac_max", type=float, default=0.6)
    p.add_argument("--mixer_max_depth", type=int, default=3)

    p.add_argument("--log_csv", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)

    args = p.parse_args()
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
        controller_csv=args.controller_csv,
        ctrl_warmup_steps=args.ctrl_warmup_steps,
        ctrl_update_steps=args.ctrl_update_steps,
        ema=args.ema,
        pair_frac_min=args.pair_frac_min,
        pair_frac_max=args.pair_frac_max,
        mixer_max_depth=args.mixer_max_depth,
        log_csv=args.log_csv,
        log_every=args.log_every
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)
    random.seed(1337)

    model = TinyTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(enabled=cfg.amp)

    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds   = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches,   device=device)

    controller = AQEDController(cfg) if cfg.controller_csv else None

    logf = None
    fobj = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj = open(cfg.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s","pair_frac","mixer_depth"
        ])
        logf.writeheader()

    global_step = 0
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        tr = run_epoch(model, train_ds, optimizer, scaler, device, cfg, controller,
                       "train", global_step_start=global_step, logf=logf)
        train_loss, train_ppl, train_tps, train_mem, train_time, global_step = tr

        with torch.no_grad():
            va = run_epoch(model, val_ds, optimizer, scaler, device, cfg, controller,
                           "val", global_step_start=global_step, logf=logf)
        val_loss, val_ppl, val_tps, val_mem, val_time, global_step = va

        if cfg.log_csv:
            logf.writerow({
                "step": global_step, "split": "val",
                "loss": val_loss, "ppl": val_ppl,
                "tok_per_s": val_tps, "max_mem_mib": val_mem, "elapsed_s": val_time,
                "pair_frac": float("nan"), "mixer_depth": float("nan")
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
