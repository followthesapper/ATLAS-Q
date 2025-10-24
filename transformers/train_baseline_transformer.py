#!/usr/bin/env python3
import os, time, math, csv, argparse, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities & Repro
# -------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def tokens_per_second(total_tokens, elapsed_sec):
    return float(total_tokens) / max(1e-9, float(elapsed_sec))

# -------------------------
# Synthetic dataset
# -------------------------
class SyntheticLM(torch.utils.data.IterableDataset):
    """
    Generates sequences with local + medium range structure that a small Transformer can learn.
    Alphabet size = vocab_size. Sequence length = seq_len. Y is next-token prediction.
    """
    def __init__(self, vocab_size=256, seq_len=256, batches=1000, seed=1337):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batches = batches
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        g = self.generator
        for _ in range(self.batches):
            # Base random stream
            x = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
            # Inject a few patterns (n-gram style repeats with offsets)
            for k in range(4, 16, 4):
                x[k:] = (x[k:] + x[:-k]) % self.vocab_size
            # Targets are next tokens
            y = torch.roll(x, shifts=-1)
            yield x, y

# -------------------------
# Model
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:L, :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.drop(x) + residual

        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.drop(x) + residual
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6, d_ff=1536, dropout=0.0, max_len=8192):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, attn_mask=None):
        # idx: [B, L]
        x = self.tok_emb(idx)  # [B, L, D]
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, L, V]
        return logits

# -------------------------
# Training
# -------------------------
@dataclass
class Config:
    vocab_size: int = 256
    seq_len: int = 256
    train_batches: int = 2000
    val_batches: int = 100
    batch_size: int = 32
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    amp: bool = True
    epochs: int = 1
    seed: int = 1337
    device: str = "cuda"
    log_csv: str = "baseline_stats.csv"

def make_mask(L: int, device):
    # Causal mask for MultiheadAttention: shape [L, L], True=mask
    m = torch.ones(L, L, dtype=torch.bool, device=device).triu(1)
    return m

def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_ds = SyntheticLM(cfg.vocab_size, cfg.seq_len, cfg.train_batches, seed=cfg.seed)
    val_ds = SyntheticLM(cfg.vocab_size, cfg.seq_len, cfg.val_batches, seed=cfg.seed+1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size)

    model = TinyTransformerLM(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.dropout)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_tokens = 0
    t0 = time.time()
    best_val = float("inf")

    # CSV log
    os.makedirs(os.path.dirname(cfg.log_csv) or ".", exist_ok=True)
    fcsv = open(cfg.log_csv, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=["epoch","step","split","loss","ppl","tok_per_s","max_mem_mib","elapsed_s"])
    writer.writeheader()

    for epoch in range(1, cfg.epochs+1):
        model.train()
        step = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            attn_mask = make_mask(cfg.seq_len, device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(xb, attn_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

            total_tokens += xb.numel()
            step += 1
            if step % 50 == 0:
                elapsed = time.time() - t0
                tok_s = tokens_per_second(total_tokens, elapsed)
                max_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                writer.writerow({
                    "epoch": epoch, "step": step, "split": "train",
                    "loss": loss.item(), "ppl": math.exp(loss.item()),
                    "tok_per_s": tok_s, "max_mem_mib": round(max_mem,1),
                    "elapsed_s": round(elapsed,2)
                })
                fcsv.flush()

        # Validation
        model.eval()
        val_loss_sum = 0.0; val_tokens = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                attn_mask = make_mask(cfg.seq_len, device)
                logits = model(xb, attn_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))
                val_loss_sum += loss.item() * xb.size(0)
                val_tokens += xb.size(0)

        val_loss = val_loss_sum / max(1, val_tokens)
        elapsed = time.time() - t0
        tok_s = tokens_per_second(total_tokens, elapsed)
        max_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        writer.writerow({
            "epoch": epoch, "step": step, "split": "val",
            "loss": val_loss, "ppl": math.exp(val_loss),
            "tok_per_s": tok_s, "max_mem_mib": round(max_mem,1),
            "elapsed_s": round(elapsed,2)
        }); fcsv.flush()
        best_val = min(best_val, val_loss)

    fcsv.close()
    print(f"Done. Best val loss={best_val:.4f}. CSV -> {cfg.log_csv}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--train_batches", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1536)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--log_csv", type=str, default="baseline_stats.csv")
    args = p.parse_args()
    cfg = Config(
        vocab_size=args.vocab_size, seq_len=args.seq_len,
        train_batches=args.train_batches, val_batches=args.val_batches,
        batch_size=args.batch_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
        seed=args.seed, device=args.device, amp=not args.no_amp,
        log_csv=args.log_csv
    )
    train(cfg)

if __name__ == "__main__":
    main()
