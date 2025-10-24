# adaptive_components.py
# -*- coding: utf-8 -*-
"""
Adaptive control components for AQED-hybrid training.

Exposes:
  - LossAwareController: maps training/probe signals -> mixer knobs (pair_frac, mixer_depth),
                         attention rank (attn_rank), and (optional) per-layer skip mask.
                         Also updates a tiny bandit-style policy with a reward that trades
                         val-loss improvement vs. extra time (cost).

  - AQEDProbe1D: a lightweight, GPU-friendly 1D proxy for “entanglement” that you can run
                 every N steps on token embeddings (or layer activations). It maintains a
                 short history and emits a slope-based diffusion proxy D_E.

Utilities:
  - attention_entropy_from_logits / attention_entropy_from_weights
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _safe_get(d: Dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default


def attention_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Approximate attention entropy from attention logits (before softmax).
    logits: [B, nH, L, L] or [B, L, L]
    Returns mean entropy per head averaged to a scalar tensor.
    """
    if logits is None:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

    if logits.dim() == 3:  # [B, L, L]
        probs = logits.softmax(dim=-1).clamp_min(1e-12)
        ent = -(probs * probs.log()).sum(dim=-1).mean()  # mean over (B, L)
        return ent

    if logits.dim() == 4:  # [B, nH, L, L]
        probs = logits.softmax(dim=-1).clamp_min(1e-12)
        ent = -(probs * probs.log()).sum(dim=-1).mean()  # mean over (B, nH, L)
        return ent

    return torch.tensor(0.0, device=logits.device)


def attention_entropy_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Exact entropy from attention weights (after softmax).
    weights: [B, nH, L, L] or [B, L, L]
    """
    if weights is None:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

    p = weights.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1).mean()
    return ent


# -----------------------------------------------------------------------------
# AQEDProbe1D: cheap proxy for “entanglement/diffusion”
# -----------------------------------------------------------------------------
class AQEDProbe1D:
    """
    A lightweight proxy that runs on embeddings/activations:
      - chi_proxy: magnitude of local Gram spectrum tail (low-rankness inverse)
      - entropy_est: optional entropy proxy from attention logits/weights
      - D_E: slope of chi_proxy over recent window (diffusion-like trend)

    Use: call probe.maybe_probe(step, embeddings, attn_logits_or_weights)
    """

    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.every = int(getattr(cfg, "probe_every", 50))
        self.window = int(getattr(cfg, "probe_window", 32))
        self.hist_len = int(getattr(cfg, "probe_hist", 64))
        self._chi_hist: List[float] = []

        # pooling stride along sequence to keep it cheap
        self.pool_stride = int(getattr(cfg, "probe_stride", 8))
        self.max_tokens = int(getattr(cfg, "probe_max_tokens", 4096))

    @torch.no_grad()
    def _chi_from_embeddings(self, x: torch.Tensor) -> float:
        """
        x: [B, L, D] (float). We subsample tokens then estimate a rank-ness proxy
        from the spectrum of (X X^T). Returns a scalar (higher ~ more “mixed”).
        """
        if x is None or x.numel() == 0:
            return 0.0
        B, L, D = x.shape
        dev = x.device

        # Subsample along sequence
        x = x[:, ::self.pool_stride, :]           # [B, L', D]
        x = x.reshape(-1, x.shape[-1])            # [B*L', D]
        if x.shape[0] > self.max_tokens:
            idx = torch.randperm(x.shape[0], device=dev)[: self.max_tokens]
            x = x.index_select(0, idx)

        # Normalize tokens to unit length to stabilize Gram
        x = F.normalize(x, dim=-1, eps=1e-6)

        # Compute small Gram via skinny SVD (x = U S V^T, S: singular vals of x)
        # Spectrum of Gram ~ S^2. We approximate how "spread" it is.
        try:
            # svd_lowrank adapts rank automatically; cap at 64 for speed
            U, S, V = torch.linalg.svd_lowrank(x, q=min(64, min(x.shape) - 1))
            if S.numel() == 0:
                return 0.0
            p = (S ** 2)
            p = p / (p.sum() + 1e-12)
            # entropy of spectrum: high = more mixed (higher "chi")
            spec_ent = -(p * (p + 1e-12).log()).sum()
            # rescale roughly into [0, ~log(64)] -> normalize by log cap
            chi = float(spec_ent / math.log(max(2, p.numel())))
            return chi
        except Exception:
            # Fallback: trace ratio
            G_trace = (x * x).sum()
            G_fro2 = (x @ x.T).pow(2).sum()
            # if perfectly rank-1, ratio ~ 1; if uniform, lower
            r = float((G_trace ** 2) / (G_fro2 + 1e-12))
            chi = 1.0 - r
            return max(0.0, min(1.0, chi))

    @torch.no_grad()
    def maybe_probe(
        self,
        step: int,
        embeddings: Optional[torch.Tensor] = None,
        attn_logits: Optional[torch.Tensor] = None,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict of signals (possibly empty if not probing this step):
          { "chi_mean": float, "entropy": float, "D_E": float }
        """
        if step % max(1, self.every) != 0:
            return {}

        sig: Dict[str, float] = {}
        chi = 0.0
        if embeddings is not None:
            chi = self._chi_from_embeddings(embeddings)
            self._chi_hist.append(float(chi))
            if len(self._chi_hist) > self.hist_len:
                self._chi_hist = self._chi_hist[-self.hist_len :]
            sig["chi_mean"] = float(chi)

        # entropy (optional)
        ent = 0.0
        if attn_weights is not None:
            ent = float(attention_entropy_from_weights(attn_weights).item())
        elif attn_logits is not None:
            ent = float(attention_entropy_from_logits(attn_logits).item())
        sig["entropy"] = ent

        # D_E proxy from chi series slope over recent window
        if len(self._chi_hist) >= 4:
            w = min(self.window, len(self._chi_hist))
            y = torch.tensor(self._chi_hist[-w:], dtype=torch.float64, device=self.device)
            x = torch.arange(w, dtype=torch.float64, device=self.device)
            x = x - x.mean()
            y = y - y.mean()
            denom = (x * x).sum().clamp_min(1e-8)
            slope = (x * y).sum() / denom
            # Normalize slope by mean abs to limit range
            norm = (y.abs().mean() + 1e-6)
            de = float((slope / norm).clamp(-10, 10))
            sig["D_E"] = de
        else:
            sig["D_E"] = 0.0

        return sig


# -----------------------------------------------------------------------------
# LossAwareController: maps signals -> actions, and learns a tiny policy
# -----------------------------------------------------------------------------
class LossAwareController:
    """
    A tiny, online, loss-aware controller.

    Inputs (signals dict expected keys, but optional):
      - "loss": scalar (training or EMA validation)
      - "chi_mean": from probe (0..~1 or normalized)
      - "D_E": probe slope proxy (negative/positive trend)
      - "entropy": attention entropy proxy
      - "grad_norm": gradient norm (optional)
      - "lr": learning rate (optional)
      - "step": global step (optional)
      - "tok_per_s": throughput (optional)

    Outputs (actions):
      - pair_frac   in [pair_frac_min, pair_frac_max]
      - mixer_depth in {0..mixer_depth_max}
      - attn_rank   in [rank_min, rank_max]
      - skip_mask   None for now (placeholder if you add per-layer skipping)

    Learning:
      - On update_policy(), we compute reward:
            R = -(loss_t - loss_{t-1}) - cost_weight * ((time_t - time_{t-1}) / time_{t-1})
        and regress a linear predictor on a decayed feature EMA.

      - A small beta scales predicted deltas on top of a heuristic base mapping.

    Safety:
      - Warm-up guard: if we never called decide() (no features yet),
        update_policy() returns early (prevents your crash).
      - Revert window: if loss gets worse by > revert_threshold, force baseline-ish actions
        for revert_steps.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Action ranges
        self.pair_lo = float(getattr(cfg, "pair_frac_min", 0.05))
        self.pair_hi = float(getattr(cfg, "pair_frac_max", 0.6))
        self.depth_max = int(getattr(cfg, "mixer_depth_max", 3))
        self.rank_lo = int(getattr(cfg, "rank_min", 64))
        self.rank_hi = int(getattr(cfg, "rank_max", 256))

        # Bandit/policy hyperparams
        self.beta = float(getattr(cfg, "bandit_beta", 0.2))
        self.cost_w = float(getattr(cfg, "cost_weight", 0.5))
        self.ema_decay = float(getattr(cfg, "ema", 0.9))

        # Safety knobs
        self.entropy_gate = float(getattr(cfg, "entropy_gate", 2.0))
        self.skip_patience = int(getattr(cfg, "skip_patience", 200))
        self.revert_steps = int(getattr(cfg, "revert_steps", 100))
        self.revert_threshold = float(getattr(cfg, "revert_threshold", 0.05))  # 5% loss jump

        # Internal state
        self.state_ema: Optional[torch.Tensor] = None  # feature EMA vector
        self.last_loss: Optional[float] = None
        self.revert_until: int = -1
        self._feat_layout = [
            "loss", "chi_mean", "D_E", "entropy",
            "grad_norm", "lr", "tok_per_s", "step_norm"
        ]
        self._feat_norm = torch.tensor(
            [10.0, 1.0, 1.0, 5.0, 10.0, 1e-3, 1e5, 1.0],  # rough scales
            device=self.dev, dtype=torch.float32
        )

        # Tiny linear ensemble for robustness
        H = len(self._feat_layout)
        E = 4  # ensemble width
        self.W = nn.Parameter(torch.zeros(E, H, device=self.dev))
        self.b = nn.Parameter(torch.zeros(E, device=self.dev))
        nn.init.normal_(self.W, std=0.05)
        nn.init.normal_(self.b, std=0.05)
        self.optW = torch.optim.Adam([self.W, self.b], lr=1e-3)

        # Step counter (for normalization)
        self._decide_calls = 0

    # ---- feature extraction ---------------------------------------------------
    def _feat_vector(self, signals: Dict[str, float]) -> torch.Tensor:
        step = _safe_get(signals, "step", 0.0)
        x = [
            _safe_get(signals, "loss", 0.0),
            _safe_get(signals, "chi_mean", 0.0),
            _safe_get(signals, "D_E", 0.0),
            _safe_get(signals, "entropy", 0.0),
            _safe_get(signals, "grad_norm", 0.0),
            _safe_get(signals, "lr", 0.0),
            _safe_get(signals, "tok_per_s", 0.0),
            float(step) / max(1.0, float(getattr(self.cfg, "train_batches", 1000))),  # ~[0,1]
        ]
        v = torch.tensor(x, device=self.dev, dtype=torch.float32)
        v = v / (self._feat_norm + 1e-8)
        return v

    def _update_feat_ema(self, v: torch.Tensor) -> torch.Tensor:
        if self.state_ema is None:
            self.state_ema = v.detach()
        else:
            self.state_ema = self.ema_decay * self.state_ema + (1 - self.ema_decay) * v.detach()
        return self.state_ema

    # ---- base heuristic mapping ----------------------------------------------
    def _base_map(self, chi: float, de: float, ent: float) -> Tuple[float, int, int]:
        """
        Map probes to sane defaults. Higher chi/entropy -> allow more mixing & rank.
        Negative DE (mixing falling) -> nudge up slightly; positive -> keep moderate.
        """
        # pair frac
        t = max(0.0, min(1.0, chi))  # normalize-ish
        pair = self.pair_lo + (self.pair_hi - self.pair_lo) * t

        # depth increases with entropy & chi
        depth_f = (0.5 * t + 0.5 * max(0.0, min(1.0, ent / max(1.0, self.entropy_gate))))
        depth = int(round(depth_f * self.depth_max))
        depth = max(0, min(self.depth_max, depth))

        # rank schedule: chi & entropy
        r_t = max(0.0, min(1.0, 0.5 * t + 0.5 * min(1.0, ent / max(1.0, self.entropy_gate))))
        rank = int(round(self.rank_lo + (self.rank_hi - self.rank_lo) * r_t))
        rank = max(self.rank_lo, min(self.rank_hi, rank))

        # DE adjustment: if slope negative (losing mixing), bump a bit
        if de < 0:
            pair = min(self.pair_hi, pair * (1.0 - 0.25 * de))  # de<0 => multiply >1
            depth = min(self.depth_max, max(depth, 1))

        return pair, depth, rank

    # ---- public: decide() -----------------------------------------------------
    @torch.no_grad()
    def decide(self, signals: Dict[str, float]) -> Tuple[float, int, int, Optional[torch.Tensor]]:
        """
        Returns:
          pair_frac, mixer_depth, attn_rank, skip_mask(None placeholder)

        Use every step (both train/val) so the feature EMA stays current.
        """
        self._decide_calls += 1
        step = int(_safe_get(signals, "step", 0))
        loss = _safe_get(signals, "loss", float("nan"))
        chi = _safe_get(signals, "chi_mean", 0.0)
        de = _safe_get(signals, "D_E", 0.0)
        ent = _safe_get(signals, "entropy", 0.0)

        # Safety: revert window if loss jumped
        if self.last_loss is not None and math.isfinite(loss):
            if loss > self.last_loss * (1.0 + self.revert_threshold):
                self.revert_until = step + self.revert_steps
        if math.isfinite(loss):
            self.last_loss = loss

        v = self._feat_vector(signals)
        v_ema = self._update_feat_ema(v)

        # Base heuristic knobs
        base_pair, base_depth, base_rank = self._base_map(chi, de, ent)

        # Learned delta from tiny ensemble
        # (Use mean head to produce a small additive adjustment.)
        pred = (self.W.mean(dim=0) @ v_ema + self.b.mean()).tanh()  # [-1,1]
        # Scale deltas
        d_pair = float(self.beta * pred.item() * 0.1 * (self.pair_hi - self.pair_lo))
        d_depth = int(round(self.beta * pred.item() * 0.5 * self.depth_max))
        d_rank = int(round(self.beta * pred.item() * 0.2 * (self.rank_hi - self.rank_lo)))

        pair = max(self.pair_lo, min(self.pair_hi, base_pair + d_pair))
        depth = int(max(0, min(self.depth_max, base_depth + d_depth)))
        rank = int(max(self.rank_lo, min(self.rank_hi, base_rank + d_rank)))

        # Entropy gating: if attention entropy is very low (very peaky),
        # mixing may hurt — reduce depth and pair.
        if ent < 0.5 * self.entropy_gate:
            pair = max(self.pair_lo, 0.5 * pair)
            depth = max(0, depth - 1)

        # Revert window: force conservative settings
        if step <= self.revert_until:
            pair = self.pair_lo
            depth = 0  # turn off mixer briefly
            # keep rank moderate to preserve grad signal
            rank = max(self.rank_lo, min(self.rank_hi, (self.rank_lo + self.rank_hi) // 2))

        skip_mask = None  # placeholder if you implement per-layer/head skipping

        return float(pair), int(depth), int(rank), skip_mask

    # ---- public: update_policy() ---------------------------------------------
    def update_policy(
        self,
        prev_signals: Dict[str, float],
        cur_signals: Dict[str, float],
        prev_time: float,
        cur_time: float,
    ):
        """
        Bandit-like update:
          reward R = -(Δloss) - cost_weight * Δtime_rel

        We predict R from the *previous* state_ema (feature vector).
        """
        # --- guard: don't update until we've formed a feature EMA via decide() ---
        if self.state_ema is None:
            return

        prev_loss = _safe_get(prev_signals, "loss", float("nan"))
        cur_loss = _safe_get(cur_signals, "loss", float("nan"))
        if not (math.isfinite(prev_loss) and math.isfinite(cur_loss)):
            return

        dloss = cur_loss - prev_loss
        dtime_rel = 0.0
        if prev_time > 1e-6 and math.isfinite(prev_time) and math.isfinite(cur_time):
            dtime_rel = (cur_time - prev_time) / max(prev_time, 1e-6)
        R = -dloss - self.cost_w * dtime_rel

        x_prev = self.state_ema.detach()
        target = torch.tensor([R], dtype=torch.float32, device=self.dev)
        pred = (self.W.mean(dim=0) @ x_prev + self.b.mean()).unsqueeze(0)

        loss = F.mse_loss(pred, target)
        self.optW.zero_grad()
        loss.backward()
        self.optW.step()


# -----------------------------------------------------------------------------
# (Optional) simple rank projection helper
# -----------------------------------------------------------------------------
def low_rank_project_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Very simple low-rank projection of K,V via top-k SVD on K (token dimension).
    Shapes: Q,K,V: [B, nH, L, Dh]
    Returns projected (Q, K_r, V_r). Keep Q to compute attention with K_r,V_r.
    """
    B, H, L, Dh = K.shape
    r = max(1, min(rank, L))
    dev = K.device

    # Flatten heads/batch for per-head SVD; keep it tiny (and batched by loop).
    K_r_list, V_r_list = [], []
    for b in range(B):
        Kb = K[b]  # [H, L, Dh]
        Vb = V[b]
        Kb_ = Kb.reshape(H * Dh, L).T  # [L, H*Dh]
        try:
            # Use lowrank SVD; q=r ensures only top-r vectors
            U, S, Vt = torch.linalg.svd_lowrank(Kb_, q=min(r, min(Kb_.shape) - 1))
            Ur = U[:, :r]  # [L, r]
        except Exception:
            # Fallback: random projection
            Ur = torch.randn(L, r, device=dev)
            Ur, _ = torch.linalg.qr(Ur, mode="reduced")
        # Project K,V into r basis along L
        # K: [H, L, Dh] -> [H, r, Dh] via Ur^T @ (L)
        Kr = torch.einsum("lr,hld->hrd", Ur, Kb)
        Vr = torch.einsum("lr,hld->hrd", Ur, Vb)
        K_r_list.append(Kr)
        V_r_list.append(Vr)

    K_r = torch.stack(K_r_list, dim=0)  # [B, H, r, Dh]
    V_r = torch.stack(V_r_list, dim=0)  # [B, H, r, Dh]
    return Q, K_r, V_r
