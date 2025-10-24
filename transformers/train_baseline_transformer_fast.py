#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, math, csv, argparse, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import GradScaler as _NewGradScaler
    from torch.amp import autocast as _new_autocast
    _NEW = True
except Exception:
    from torch.cuda.amp import GradScaler as _OldGradScaler
    from torch.cuda.amp import autocast as _old_autocast
    _NEW = False

def make_scaler(enabled: bool):
    return (_NewGradScaler(enabled=enabled) if _NEW else _OldGradScaler(enabled=enabled))

class AutocastCtx:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None
    def __enter__(self):
        try:
            from torch.amp import autocast as _ac
            self.ctx = _ac("cuda" if torch.cuda.is_available() else "cpu", enabled=self.enabled)
        except Exception:
            from torch.cuda.amp import autocast as _ac_old
            self.ctx = _ac_old(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, et, ex, tb):
        return self.ctx.__exit__(et, ex, tb)

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
    def forward(self, x):
        a,_ = self.attn(x,x,x,need_weights=False)
        x = self.ln1(x+a)
        y = self.ff(x)
        x = self.ln2(x+y)
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    def forward(self,x):
        B,L = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :L, :]
        for blk in self.blocks: h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

def _ensure_batch(inp, tgt):
    if inp.dim()==1: inp=inp.unsqueeze(0)
    if tgt.dim()==1: tgt=tgt.unsqueeze(0)
    return inp.to(torch.long), tgt.to(torch.long)

def run_epoch(model, dataset, optimizer, scaler, device, cfg: Config, phase="train", step0=0, logf=None):
    is_train = (phase == "train")
    model.train(is_train)
    total_loss,total_tok,max_mem = 0.0,0,0.0
    start_t=time.time(); step=step0
    for it,(inp,tgt) in enumerate(dataset):
        inp,tgt=_ensure_batch(inp,tgt)
        inp=inp.to(device, non_blocking=True); tgt=tgt.to(device, non_blocking=True)
        t0=time.time()
        with AutocastCtx(enabled=cfg.amp):
            logits=model(inp)
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), tgt.reshape(-1))
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bs=inp.numel()
        total_tok+=bs
        total_loss+=float(loss.detach())*bs
        try:
            mem=torch.cuda.max_memory_allocated(device)/(1024**2)
            max_mem=max(max_mem,mem)
        except Exception:
            pass
        if logf and (step%cfg.log_every==0):
            elapsed=time.time()-start_t
            tps=total_tok/max(1e-9,elapsed)
            logf.writerow({
                "step": step, "split": phase, "loss": total_loss/max(1,total_tok),
                "ppl": math.exp(min(20.0,total_loss/max(1,total_tok))),
                "tok_per_s": tps, "max_mem_mib": max_mem, "elapsed_s": elapsed
            })
        step+=1
    elapsed=time.time()-start_t
    tps=total_tok/max(1e-9,elapsed)
    avg=total_loss/max(1,total_tok)
    return avg, math.exp(min(20.0,avg)), tps, max_mem, elapsed, step

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--vocab_size", type=int, default=32000)
    a.add_argument("--seq_len", type=int, default=512)
    a.add_argument("--batch_size", type=int, default=64)
    a.add_argument("--train_batches", type=int, default=3000)
    a.add_argument("--val_batches", type=int, default=200)
    a.add_argument("--d_model", type=int, default=512)
    a.add_argument("--n_layers", type=int, default=8)
    a.add_argument("--n_heads", type=int, default=8)
    a.add_argument("--d_ff", type=int, default=2048)
    a.add_argument("--dropout", type=float, default=0.1)
    a.add_argument("--epochs", type=int, default=1)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--weight_decay", type=float, default=0.01)
    a.add_argument("--no_amp", action="store_true")
    a.add_argument("--log_csv", type=str, default=None)
    a.add_argument("--log_every", type=int, default=50)
    args=a.parse_args()
    cfg=Config(
        vocab_size=args.vocab_size, seq_len=args.seq_len, batch_size=args.batch_size,
        train_batches=args.train_batches, val_batches=args.val_batches,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=args.dropout, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        amp=not args.no_amp, log_csv=args.log_csv, log_every=args.log_every
    )
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337); random.seed(1337)
    model=TinyTransformerLM(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler=make_scaler(enabled=cfg.amp)
    train_ds=ZipfDataset(cfg.vocab_size,cfg.seq_len,cfg.train_batches,device=device)
    val_ds=ZipfDataset(cfg.vocab_size,cfg.seq_len,cfg.val_batches,device=device)
    logf=None; fobj=None
    if cfg.log_csv:
        os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
        fobj=open(cfg.log_csv,"w",newline="")
        logf=csv.DictWriter(fobj, fieldnames=["step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s"])
        logf.writeheader()
    best=float("inf"); step=0
    for epoch in range(cfg.epochs):
        tr=run_epoch(model,train_ds,opt,scaler,device,cfg,"train",step0=step,logf=logf)
        train_loss, train_ppl, train_tps, train_mem, train_time, step = tr
        with torch.no_grad():
            va=run_epoch(model,val_ds,opt,scaler,device,cfg,"val",step0=step,logf=logf)
        val_loss, val_ppl, val_tps, val_mem, val_time, step = va
        if cfg.log_csv:
            logf.writerow({"step": step,"split":"val","loss":val_loss,"ppl":val_ppl,
                           "tok_per_s":val_tps,"max_mem_mib":val_mem,"elapsed_s":val_time})
        best=min(best,val_loss)
        print(f"Epoch {epoch+1}/{cfg.epochs}  train loss {train_loss:.4f} ppl {train_ppl:.2f} tps {train_tps:.1f} | val loss {val_loss:.4f} ppl {val_ppl:.2f} tps {val_tps:.1f}")
    if fobj: fobj.close()
    print(f"Done. Best val loss={best:.4f}. CSV -> {cfg.log_csv or '(none)'}")

if __name__=="__main__":
    main()
