#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Transformer with loss-aware AQED controller:
- Attention in every K-th block (arg: --attn_keep_every), mixer in others
- Cheap 1D AQED probe every N steps (arg: --probe_every)
- Controller outputs (pair_frac, mixer_depth, attn_rank) each step
- Optional low-rank projection of K,V to reduce attention cost

Logs: step, split, loss, ppl, tok_per_s, max_mem_mib, elapsed_s, pair_frac, mixer_depth, attn_rank
"""

import os, time, math, csv, argparse, random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Our adaptive bits
from adaptive_components import (
    AQEDProbe1D,
    LossAwareController,
    low_rank_project_attention,
)

# ---------- AMP compatibility (works on old/new PyTorch) ----------
try:
    from torch.amp import GradScaler as _NewGradScaler
    from torch.amp import autocast as _new_autocast
    _USE_NEW_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler as _OldGradScaler
    from torch.cuda.amp import autocast as _old_autocast
    _USE_NEW_AMP = False

def make_scaler(enabled: bool):
    return (_NewGradScaler if _USE_NEW_AMP else _OldGradScaler)(enabled=enabled)

class AutocastCtx:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        try:
            from torch.amp import autocast as _new_autocast  # new API
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.ctx = _new_autocast(device_type, enabled=self.enabled)
        except Exception:
            from torch.cuda.amp import autocast as _old_autocast           # old API
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

    # Hybrid pattern
    attn_keep_every: int = 2

    # Probe/Controller wiring (forwarded to adaptive_components)
    probe_every: int = 50
    probe_window: int = 32
    probe_hist: int = 64
    probe_stride: int = 8
    probe_max_tokens: int = 4096

    pair_frac_min: float = 0.05
    pair_frac_max: float = 0.6
    mixer_depth_max: int = 3
    rank_min: int = 64
    rank_max: int = 256
    entropy_gate: float = 2.0
    skip_patience: int = 200
    revert_steps: int = 100
    revert_threshold: float = 0.05  # 5% loss jump
    bandit_beta: float = 0.2
    cost_weight: float = 0.5
    ema: float = 0.9

    # Logging
    log_csv: Optional[str] = None
    log_every: int = 50
    power_csv: Optional[str] = None   # optional (best-effort)

# ---------- Dataset ----------
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

# ---------- Mixer ----------
class AQEDMixLite(nn.Module):
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
            # deterministic block pairing (vectorizable, avoids per-step randomness cost)
            idx = torch.arange(L, device=x.device)
            pairs = torch.stack([idx[0::2], idx[1::2]], dim=1)[:n_pairs]  # [n_pairs, 2]
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

# ---------- Blocks ----------
class AttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: [B, L, D] -> [B, H, L, Dh]
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head
        return x.view(B, L, H, Dh).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x):
        # x: [B, H, L, Dh] -> [B, L, D]
        B, H, L, Dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

    def forward(self, x, attn_rank=None):
        # Pre-norm
        h = self.ln1(x)
        B, L, D = h.shape
        qkv = self.qkv(h)                          # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)                   # [B, H, L, Dh]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Optional low-rank projection of K,V along sequence dim
        if (attn_rank is not None) and (attn_rank < L):
            q, k, v = low_rank_project_attention(q, k, v, rank=int(attn_rank))  # k,v: [B,H,r,Dh]

        # Scaled dot-product attention (uses Flash/SDPA kernels when available)
        # Q,K,V: [B,H,L,Dh], [B,H,L or r, Dh]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        attn = self._merge_heads(attn)            # [B, L, D]
        x = x + self.dropout(self.out(attn))
        y = self.ff(self.ln2(x))
        x = x + y
        return x

class MixerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mixer = AQEDMixLite(d_model, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pair_frac=0.2, mixer_depth=1):
        h = self.ln(x)
        h = self.mixer(h, pair_frac=pair_frac, depth=mixer_depth)
        x = x + h
        y = self.ff(self.ln2(x))
        x = x + y
        return x

# ---------- Hybrid Transformer ----------
class HybridTransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            if i % max(1, cfg.attn_keep_every) == 0:
                self.blocks.append(AttnBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout))
            else:
                self.blocks.append(MixerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x, pair_frac=0.2, mixer_depth=1, attn_rank=None, return_embeddings=False):
        B, L = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :L, :]
        for blk in self.blocks:
            if isinstance(blk, AttnBlock):
                h = blk(h, attn_rank=attn_rank)
            else:
                h = blk(h, pair_frac=pair_frac, mixer_depth=mixer_depth)
        out = self.ln_f(h)
        logits = self.head(out)
        if return_embeddings:
            return logits, h
        return logits

# ---------- Train/Eval ----------
def _ensure_batch(inp, tgt):
    if getattr(inp, "dim", lambda: 2)() == 1: inp = inp.unsqueeze(0)
    if getattr(tgt, "dim", lambda: 2)() == 1: tgt = tgt.unsqueeze(0)
    return inp.to(torch.long), tgt.to(torch.long)

def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config,
              controller: LossAwareController, probe: AQEDProbe1D,
              phase="train", global_step_start=0, logf=None):
    model.train(phase == "train")
    total_loss, total_tok, max_mem_mib = 0.0, 0, 0.0
    global_step = global_step_start
    start_t = time.time()

    # default knobs before controller warms
    pair_frac, mixer_depth, attn_rank = 0.15, 1, cfg.rank_max

    last_signals = None
    last_time = float("nan")

    for bi, (inp, tgt) in enumerate(dataset):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        inp, tgt = _ensure_batch(inp, tgt)

        # Build signal dict for controller (pre-loss)
        signals = {
            "step": float(global_step),
            "lr": float(getattr(optimizer, "param_groups", [{}])[0].get("lr", cfg.lr)),
        }

        # Maybe run probe on embeddings (cheap)
        with torch.no_grad():
            # Grab token embeddings only (no blocks) to keep probe light
            B, L = inp.shape
            emb = model.tok_emb(inp) + model.pos_emb[:, :L, :]
            probe_out = probe.maybe_probe(global_step, embeddings=emb)
        signals.update(probe_out)

        # Controller decides knobs
        pf, md, rk, _ = controller.decide({**signals, "loss": total_loss / max(1, total_tok) if total_tok > 0 else 0.0})
        pair_frac, mixer_depth, attn_rank = pf, md, rk

        # Forward + loss
        with AutocastCtx(enabled=cfg.amp):
            logits = model(inp, pair_frac=pair_frac, mixer_depth=mixer_depth, attn_rank=attn_rank)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs_tokens = inp.numel()
        total_tok += bs_tokens
        total_loss += float(loss.detach()) * bs_tokens

        # power sample (optional) – best-effort
        # (Kept minimal: you can wire a background sampler writing to cfg.power_csv)
        # -- omitted here to avoid shell calls during training loop --

        # mem
        try:
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            max_mem_mib = max(max_mem_mib, mem)
        except Exception:
            pass

        # reward update for controller (loss-aware, time-cost aware)
        cur_time = time.time() - start_t
        cur_signals = {**signals, "loss": float(loss.detach())}
        if last_signals is not None:
            if phase == "train":

                controller.update_policy(last_signals, cur_signals,
                                     last_time if math.isfinite(last_time) else cur_time,
                                     cur_time)
        last_signals, last_time = cur_signals, cur_time

        # logging
        if logf and (global_step % cfg.log_every == 0):
            elapsed = time.time() - start_t
            tok_per_s = total_tok / max(1e-9, elapsed)
            logf.writerow({
                "step": global_step, "split": phase,
                "loss": total_loss / max(1, total_tok),
                "ppl": math.exp(min(20.0, total_loss / max(1, total_tok))),
                "tok_per_s": tok_per_s, "max_mem_mib": max_mem_mib, "elapsed_s": elapsed,
                "pair_frac": pair_frac, "mixer_depth": mixer_depth, "attn_rank": attn_rank
            })

        global_step += 1

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem_mib, elapsed, global_step

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    # model/optim
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

    # hybrid/probe/controller
    p.add_argument("--attn_keep_every", type=int, default=2)
    p.add_argument("--probe_every", type=int, default=50)
    p.add_argument("--pair_frac_min", type=float, default=0.05)
    p.add_argument("--pair_frac_max", type=float, default=0.6)
    p.add_argument("--mixer_depth_max", type=int, default=3)
    p.add_argument("--rank_min", type=int, default=64)
    p.add_argument("--rank_max", type=int, default=256)
    p.add_argument("--entropy_gate", type=float, default=2.0)
    p.add_argument("--skip_patience", type=int, default=200)
    p.add_argument("--revert_steps", type=int, default=100)
    p.add_argument("--revert_threshold", type=float, default=0.05)
    p.add_argument("--bandit_beta", type=float, default=0.2)
    p.add_argument("--cost_weight", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.9)

    # logging
    p.add_argument("--log_csv", type=str, default=None)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--power_csv", type=str, default=None)

    args = p.parse_args()
    cfg = Config(
        vocab_size=args.vocab_size, seq_len=args.seq_len, batch_size=args.batch_size,
        train_batches=args.train_batches, val_batches=args.val_batches,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        amp=not args.no_amp, attn_keep_every=args.attn_keep_every, probe_every=args.probe_every,
        pair_frac_min=args.pair_frac_min, pair_frac_max=args.pair_frac_max,
        mixer_depth_max=args.mixer_depth_max, rank_min=args.rank_min, rank_max=args.rank_max,
        entropy_gate=args.entropy_gate, skip_patience=args.skip_patience,
        revert_steps=args.revert_steps, revert_threshold=args.revert_threshold,
        bandit_beta=args.bandit_beta, cost_weight=args.cost_weight, ema=args.ema,
        log_csv=args.log_csv, log_every=args.log_every, power_csv=args.power_csv
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); random.seed(1337)

    model = HybridTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(enabled=cfg.amp)

    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds   = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches,   device=device)

    # adaptive pieces
    controller = LossAwareController(cfg)
    probe = AQEDProbe1D(cfg)

    # logging
    logf = None; fobj = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj = open(cfg.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s",
            "pair_frac","mixer_depth","attn_rank"
        ])
        logf.writeheader()

    global_step = 0
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        tr = run_epoch(model, train_ds, optimizer, scaler, device, cfg, controller, probe,
                       "train", global_step_start=global_step, logf=logf)
        train_loss, train_ppl, train_tps, train_mem, train_time, global_step = tr

        with torch.no_grad():
            va = run_epoch(model, val_ds, optimizer, scaler, device, cfg, controller, probe,
                           "val", global_step_start=global_step, logf=logf)
        val_loss, val_ppl, val_tps, val_mem, val_time, global_step = va

        if cfg.log_csv:
            logf.writerow({
                "step": global_step, "split": "val", "loss": val_loss, "ppl": val_ppl,
                "tok_per_s": val_tps, "max_mem_mib": val_mem, "elapsed_s": val_time,
                "pair_frac": float("nan"), "mixer_depth": float("nan"), "attn_rank": float("nan"),
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
