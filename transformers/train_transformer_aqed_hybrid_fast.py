#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Transformer (fast): keep every K-th attention layer; replace the others
with a vectorized AQED mixer. Supports torch.compile, AMP (both old/new APIs),
TF32 knobs, and optional precomputed pair patterns.

Usage examples are at the bottom or see: -h
"""

import os, time, math, csv, argparse, random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------- AMP compatibility (new torch.amp vs old torch.cuda.amp) ----------
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
        return _NewGradScaler(enabled=enabled)
    else:
        return _OldGradScaler(enabled=enabled)

class AutocastCtx:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        try:
            # new API
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            self.ctx = _new_autocast(device_type, enabled=self.enabled)
        except Exception:
            self.ctx = _old_autocast(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)


# ------------------------------- Config --------------------------------------
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

    # Hybrid controls
    attn_keep_every: int = 2    # keep 1 out of K attention layers
    mixer_depth: int = 1        # number of mixer iterations per replaced layer
    pair_frac: float = 0.10     # fraction of L//2 pairs per iteration

    # Optional: precompute pair patterns to reduce RNG/perm overhead
    precompute_pairs: bool = False
    pair_seed: int = 1337
    precompute_steps: int = 1024   # length of the pair pattern cycle

    # Torch compile
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"   # or "max-autotune"
    compile_dynamic: bool = False

    # Logging
    log_csv: Optional[str] = None
    log_every: int = 50


# ----------------------------- Synthetic data --------------------------------
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


# ----------------------------- AQED Mixer (fast) -----------------------------
class AQEDMixLiteFast(nn.Module):
    """
    Vectorized 2-token mixing using two linear maps + gated blend.
    Avoids full-tensor clone() by index_copy_ on token dim.
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

        # pair pattern buffers populated optionally at runtime
        self.register_buffer("_pairs_cycle", None, persistent=False)   # [T, n_pairs, 2]
        self._cycle_len = 0
        self._cycle_pos = 0

    def precompute_pairs(self, L: int, pair_frac: float, steps: int, device: str, seed: int = 1337):
        torch.manual_seed(seed)
        n_pairs = max(1, int((L // 2) * max(0.0, min(1.0, pair_frac))))
        if n_pairs < 1:
            self._pairs_cycle = None
            self._cycle_len = 0
            return
        pairs = []
        for _ in range(steps):
            perm = torch.randperm(L, device=device)
            pairs_t = perm.view(-1, 2)[:n_pairs]  # [n_pairs, 2]
            pairs.append(pairs_t)
        self._pairs_cycle = torch.stack(pairs, dim=0)  # [steps, n_pairs, 2]
        self._cycle_len = steps
        self._cycle_pos = 0

    def _next_pairs(self, L: int, pair_frac: float, device: str) -> torch.Tensor:
        if self._pairs_cycle is not None and self._cycle_len > 0:
            out = self._pairs_cycle[self._cycle_pos]
            self._cycle_pos = (self._cycle_pos + 1) % self._cycle_len
            return out
        # fallback: per-call permutation
        n_pairs = max(1, int((L // 2) * max(0.0, min(1.0, pair_frac))))
        perm = torch.randperm(L, device=device)
        return perm.view(-1, 2)[:n_pairs]

    @torch.no_grad()
    def _index_copy(self, base: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        base: [B,L,D], idx: [n_pairs], src: [B,n_pairs,D]
        Performs base[:, idx, :] = src with no full clone, returns modified base.
        """
        # index_copy_ wants the index to be 1D; we write all pairs at once on dim=1
        # We ensure contiguous shapes for best performance
        return base.index_copy(1, idx, src)

    def forward(self, x: torch.Tensor, pair_frac: float, depth: int) -> torch.Tensor:
        if depth <= 0 or pair_frac <= 0:
            return x
        B, L, D = x.shape
        out = x

        for _ in range(depth):
            pairs = self._next_pairs(L, pair_frac, x.device)  # [n_pairs, 2]
            i = pairs[:, 0]
            j = pairs[:, 1]

            # Gather slices
            xi = out[:, i, :]  # [B, n_pairs, D]
            xj = out[:, j, :]

            # Vectorized mixing + gate
            mix = torch.tanh(self.lin1(xi) + self.lin2(xj))  # [B, n_pairs, D]
            g = torch.sigmoid(self.gate(torch.cat([xi, xj], dim=-1)))  # [B, n_pairs, D]
            mixed = g * mix + (1 - g) * xi  # [B, n_pairs, D]
            mixed = self.dropout(mixed)

            # In-place index write on dim=1 for positions i
            out = self._index_copy(out, i, mixed)

        return out


# ----------------------------- Transformer blocks ----------------------------
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


class TinyHybridTransformerLM(nn.Module):
    """
    Keep every K-th attention layer; other layers use AQED mixer in the residual
    slot (so shape & stability match standard block ordering).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                     for _ in range(cfg.n_layers)])
        self.mixer = AQEDMixLiteFast(cfg.d_model, cfg.dropout)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def maybe_precompute_pairs(self, L: int, device: str):
        if self.cfg.precompute_pairs:
            self.mixer.precompute_pairs(
                L=L,
                pair_frac=self.cfg.pair_frac,
                steps=self.cfg.precompute_steps,
                device=device,
                seed=self.cfg.pair_seed
            )

    def forward(self, x, pair_frac=None, mixer_depth=None, attn_keep_every=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.long)

        pair_frac = self.cfg.pair_frac if pair_frac is None else pair_frac
        mixer_depth = self.cfg.mixer_depth if mixer_depth is None else mixer_depth
        keep_k = self.cfg.attn_keep_every if attn_keep_every is None else attn_keep_every

        B, L = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :L, :]

        # one-time optional precompute
        if (self.mixer._pairs_cycle is None) and self.cfg.precompute_pairs:
            self.maybe_precompute_pairs(L, h.device)

        for li, blk in enumerate(self.blocks):
            if keep_k > 0 and (li % keep_k) == 0:
                # full attention block
                h = blk(h)
            else:
                # replace attention with mixer in the residual slot:
                # baseline does: a = Attn(h); h = ln1(h + a)
                # we do: a = Mixer(h) - h; h = ln1(h + a) = ln1(Mixer(h))
                mixed = self.mixer(h, pair_frac=pair_frac, depth=mixer_depth)
                h = blk.ln1(mixed)          # mimic ln1(h + a) with h'==mixed
                y = blk.ff(h)
                h = blk.ln2(h + y)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


# -------------------------------- Train/Eval ---------------------------------
def _ensure_batch(inp, tgt):
    if getattr(inp, "dim", lambda: 2)() == 1: inp = inp.unsqueeze(0)
    if getattr(tgt, "dim", lambda: 2)() == 1: tgt = tgt.unsqueeze(0)
    return inp.to(torch.long), tgt.to(torch.long)

def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config,
              phase="train", logf=None, sdpa_ctx=None):
    model.train(phase == "train")
    total_loss, total_tok, max_mem_mib = 0.0, 0, 0.0
    start_t = time.time()

    for step, (inp, tgt) in enumerate(dataset):
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with sdpa_ctx if sdpa_ctx is not None else torch.no_grad():  # contextmanager even in eval
            with AutocastCtx(enabled=cfg.amp):
                inp, tgt = _ensure_batch(inp, tgt)
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs_tokens = inp.numel()
        total_tok += bs_tokens
        total_loss += float(loss.detach()) * bs_tokens

        # track max mem (best-effort)
        try:
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
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
                "attn_keep_every": cfg.attn_keep_every,
                "mixer_depth": cfg.mixer_depth,
                "pair_frac": cfg.pair_frac
            })

    elapsed = time.time() - start_t
    tok_per_s = total_tok / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, total_tok)
    return avg_loss, math.exp(min(20.0, avg_loss)), tok_per_s, max_mem_mib, elapsed


def main():
    p = argparse.ArgumentParser()
    # Model/data
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

    # Hybrid knobs
    p.add_argument("--attn_keep_every", type=int, default=2)
    p.add_argument("--mixer_depth", type=int, default=1)
    p.add_argument("--pair_frac", type=float, default=0.10)

    # Pair precompute
    p.add_argument("--precompute_pairs", action="store_true")
    p.add_argument("--pair_seed", type=int, default=1337)
    p.add_argument("--precompute_steps", type=int, default=1024)

    # Compile
    p.add_argument("--compile", dest="use_compile", action="store_true")
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")
    p.add_argument("--compile_dynamic", action="store_true")

    # Logging
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
        attn_keep_every=args.attn_keep_every,
        mixer_depth=args.mixer_depth,
        pair_frac=args.pair_frac,
        precompute_pairs=args.precompute_pairs,
        pair_seed=args.pair_seed,
        precompute_steps=args.precompute_steps,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        compile_dynamic=args.compile_dynamic,
        log_csv=args.log_csv,
        log_every=args.log_every
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Enable fast matmul (TF32) where supported
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Prefer Flash or Math kernels for SDPA
    sdpa_ctx = None
    try:
        from torch.nn.attention import sdpa_kernel
        sdpa_ctx = sdpa_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
    except Exception:
        pass

    torch.manual_seed(1337)
    random.seed(1337)

    # Model
    model = TinyHybridTransformerLM(cfg).to(device)

    # Optionally compile (helpful once mixer is lightweight)
    if cfg.use_compile:
        try:
            model = torch.compile(model, mode=cfg.compile_mode, dynamic=cfg.compile_dynamic)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}. Continuing without compile.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(enabled=cfg.amp)

    # Datasets
    train_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.train_batches, device=device)
    val_ds = ZipfDataset(cfg.vocab_size, cfg.seq_len, cfg.val_batches, device=device)

    # Logging CSV
    logf = None
    fobj = None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj = open(cfg.log_csv, "w", newline="")
        logf = csv.DictWriter(fobj, fieldnames=[
            "step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s",
            "attn_keep_every","mixer_depth","pair_frac"
        ])
        logf.writeheader()

    best_val = float("inf")
    for epoch in range(cfg.epochs):
        tr_loss, tr_ppl, tr_tps, tr_mem, tr_time = run_epoch(
            model, train_ds, optimizer, scaler, device, cfg, "train", logf=logf, sdpa_ctx=sdpa_ctx
        )
        with torch.no_grad():
            va_loss, va_ppl, va_tps, va_mem, va_time = run_epoch(
                model, val_ds, optimizer, scaler, device, cfg, "val", logf=logf, sdpa_ctx=sdpa_ctx
            )

        if cfg.log_csv and logf:
            logf.writerow({
                "step": cfg.train_batches, "split": "val",
                "loss": va_loss, "ppl": va_ppl, "tok_per_s": va_tps,
                "max_mem_mib": va_mem, "elapsed_s": va_time,
                "attn_keep_every": cfg.attn_keep_every,
                "mixer_depth": cfg.mixer_depth,
                "pair_frac": cfg.pair_frac
            })

        best_val = min(best_val, va_loss)
        print(f"Epoch {epoch+1}/{cfg.epochs}  "
              f"train loss {tr_loss:.4f} ppl {tr_ppl:.2f} tps {tr_tps:.1f} | "
              f"val loss {va_loss:.4f} ppl {va_ppl:.2f} tps {va_tps:.1f}")

    if fobj:
        fobj.close()

    print(f"Done. Best val loss={best_val:.4f}. CSV -> {cfg.log_csv or '(none)'}")


if __name__ == "__main__":
    main()

