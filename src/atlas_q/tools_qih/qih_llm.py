"""
QIH-LLM: Quantum-Inspired Hybrid features for language modeling.

This module extracts QIH-PAT side-channel features from token sequences and offers a
simple PyTorch fusion layer that injects these features into a Transformer embedding stack.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

from .qih_pat import qih_pat_sequence_features


def tokens_to_signal(token_ids, vocab_size=None, normalize=True):
    """
    Map a token ID sequence to a scalar signal for spectral analysis.
    Heuristic: use centered token ids (id - mean_id); optionally normalize to unit variance.
    """
    x = np.asarray(token_ids, dtype=float)
    if vocab_size is not None and vocab_size > 0:
        mean_id = (vocab_size - 1) / 2.0
        x = x - mean_id
    else:
        x = x - x.mean() if len(x) else x
    if normalize:
        s = x.std()
        if s > 1e-8:
            x = x / s
    return x


def qih_features_for_tokens(token_ids, win=256, stride=128, n=10, shots=1024, bins=64):
    """
    Compute per-window QIH-PAT histograms for a token sequence (list/array of ids).
    Returns: H [num_windows, bins], periods [num_windows]
    """
    sig = tokens_to_signal(token_ids)
    H, pers = qih_pat_sequence_features(sig, win=win, stride=stride, n=n, shots=shots, bins=bins)
    return H, pers


try:
    import torch
    import torch.nn as nn

    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

if _HAVE_TORCH:

    class QIHFusionEmbedding(nn.Module):
        """
        Fuse QIH-PAT side-channel into token embeddings.
        Given token embeddings E [B, T, d], windows slide along T; each window gets a QIH histogram -> proj
        and is broadcast-add to token embeddings in that window.
        """

        def __init__(
            self, win: int, stride: int, bins: int = 64, proj_dim: int = None, embed_dim: int = 128
        ):
            super().__init__()
            self.win = win
            self.stride = stride
            self.bins = bins
            self.embed_dim = embed_dim
            self.proj = nn.Linear(bins, proj_dim or embed_dim)

        def forward(self, token_ids: torch.Tensor, token_embeds: torch.Tensor):
            """
            token_ids: [B, T] integer tensor
            token_embeds: [B, T, d]
            returns: fused_embeds [B, T, d]
            """
            B, T = token_ids.shape
            device = token_embeds.device
            fused = token_embeds
            for b in range(B):
                ids = token_ids[b].detach().cpu().numpy().astype(int)
                # Compute QIH per-window histograms for this sequence
                H, _ = qih_pat_sequence_features(
                    tokens_to_signal(ids), win=self.win, stride=self.stride, bins=self.bins
                )
                import numpy as _np
                import torch as _torch

                H = _torch.tensor(
                    _np.stack(H, axis=0), dtype=token_embeds.dtype, device=device
                )  # [Wn, bins]
                Z = self.proj(H)  # [Wn, d]
                # Broadcast-add over the corresponding token ranges
                w_idx = 0
                for start in range(0, max(1, T - self.win + 1), self.stride):
                    end = min(start + self.win, T)
                    fused[b, start:end, :] = fused[b, start:end, :] + Z[w_idx].unsqueeze(0).expand(
                        end - start, -1
                    )
                    w_idx += 1
            return fused
