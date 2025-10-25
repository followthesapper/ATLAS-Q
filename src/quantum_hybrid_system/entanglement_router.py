"""
EntanglementRouter: Adaptive Token Routing for Hybrid Attention
===============================================================

Selects tokens for full vs. MPS-compressed attention based on entanglement measures.

Key Insight: High-entanglement tokens require precise modeling (full attention),
while low-entanglement tokens can be efficiently mixed via MPS operations.

Routing Strategies:
- Entropy-based: Select top-k by Von Neumann entropy
- Learned: Train gating network to predict importance
- Hybrid: Combine entropy + learned scores

Author: Claude Code (Quantum-AQED Integration)
Date: October 24, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


class EntanglementRouter(nn.Module):
    """
    Routes tokens to full attention vs MPS mixing based on entanglement.

    Args:
        route_frac: Fraction of tokens to route to full attention (e.g., 0.15 = 15%)
        d_model: Model dimension (for learned routing)
        strategy: Routing strategy ('entropy', 'learned', 'hybrid')
        temperature: Softmax temperature for learned routing (lower = more decisive)

    Example:
        >>> router = EntanglementRouter(route_frac=0.15, d_model=1024, strategy='entropy')
        >>> entropy = mps_memory.local_entropy_tokens()  # [L]
        >>> indices, mask = router(entropy, hidden_states)
        >>> # indices: [K] where K ≈ 0.15 * L (tokens for full attention)
        >>> # mask: [L] boolean (True = full attention, False = MPS)
    """

    def __init__(
        self,
        route_frac: float = 0.15,
        d_model: Optional[int] = None,
        strategy: Literal['entropy', 'learned', 'hybrid'] = 'entropy',
        temperature: float = 1.0,
        min_tokens: int = 1,
    ):
        super().__init__()

        self.route_frac = route_frac
        self.d_model = d_model
        self.strategy = strategy
        self.temperature = temperature
        self.min_tokens = min_tokens

        # Learned routing components (if needed)
        if strategy in ['learned', 'hybrid']:
            if d_model is None:
                raise ValueError("d_model required for learned/hybrid routing")

            # Simple gating network: [D] -> [1] importance score
            self.gate_net = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
            )
        else:
            self.gate_net = None

        # Load balancing loss weight (for training)
        self.load_balance_weight = 0.01

        # Statistics tracking (keep on CPU to avoid device transfer overhead)
        self.register_buffer('_total_routed', torch.tensor(0, dtype=torch.long, device='cpu'), persistent=True)
        self.register_buffer('_total_tokens', torch.tensor(0, dtype=torch.long, device='cpu'), persistent=True)

        # Telemetry: per-forward routing fractions for monitoring
        self._routing_history = []
        self._max_history = 1000  # Keep last 1000 forwards

    def forward(
        self,
        entropy: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        routing_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to full attention vs MPS mixing.

        Args:
            entropy: Entanglement entropy [L] or [B, L] from MPSMemory
            hidden_states: Token embeddings [B, L, D] (for learned routing)
            routing_override: Optional explicit routing scores [B, L]

        Returns:
            indices: [B, K] indices of tokens for full attention
            mask: [B, L] boolean mask (True = full attention)
            aux_loss: dict with load balancing loss and stats
        """
        if routing_override is not None:
            scores = routing_override
        elif self.strategy == 'entropy':
            if entropy is None:
                # Fallback to learned routing if entropy not available and gate_net exists
                if hidden_states is not None and self.gate_net is not None:
                    scores = self._learned_routing(hidden_states)
                elif hidden_states is not None:
                    # Fallback to random routing if no gate_net
                    B, L, D = hidden_states.shape
                    scores = torch.rand(B, L, device=hidden_states.device)
                else:
                    raise ValueError("entropy or hidden_states required for routing")
            else:
                scores = entropy
        elif self.strategy == 'learned':
            if hidden_states is None:
                raise ValueError("hidden_states required for learned routing")
            scores = self._learned_routing(hidden_states)
        elif self.strategy == 'hybrid':
            if entropy is None and hidden_states is None:
                raise ValueError("entropy or hidden_states required for hybrid")
            elif entropy is None:
                # Fallback to learned only
                scores = self._learned_routing(hidden_states)
            elif hidden_states is None:
                # Fallback to entropy only
                scores = entropy
            else:
                scores = self._hybrid_routing(entropy, hidden_states)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Handle batch dimension
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)  # [1, L]
        if hidden_states is not None and scores.shape[0] == 1 and hidden_states.shape[0] > 1:
            scores = scores.expand(hidden_states.shape[0], -1)  # [B, L]

        B, L = scores.shape

        # Determine number of tokens to route
        K = max(self.min_tokens, int(self.route_frac * L))
        K = min(K, L)  # Can't route more than we have

        # Top-K selection
        if K == L:
            # Route all to full attention
            indices = torch.arange(L, device=scores.device).unsqueeze(0).expand(B, -1)
            mask = torch.ones(B, L, dtype=torch.bool, device=scores.device)
        else:
            # Select top K
            topk_scores, indices = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
            mask = torch.zeros(B, L, dtype=torch.bool, device=scores.device)
            mask.scatter_(1, indices, True)

        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_load_balance_loss(scores, mask, K, L)

        # Update statistics (buffers are already on CPU)
        num_routed = mask.sum().item()
        self._total_routed += num_routed
        self._total_tokens += B * L

        # Telemetry: track routing fraction history
        route_frac = num_routed / (B * L)
        if len(self._routing_history) >= self._max_history:
            self._routing_history.pop(0)
        self._routing_history.append({
            'route_frac': route_frac,
            'num_routed': num_routed,
            'strategy': self.strategy,
        })

        return indices, mask, aux_loss

    def _learned_routing(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Learned routing via gating network.

        Args:
            hidden_states: [B, L, D]

        Returns:
            scores: [B, L] importance scores
        """
        B, L, D = hidden_states.shape

        # Apply gate network
        gate_logits = self.gate_net(hidden_states).squeeze(-1)  # [B, L]

        # Temperature scaling
        scores = gate_logits / self.temperature

        return scores

    def _hybrid_routing(
        self,
        entropy: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hybrid routing: combine entropy + learned scores.

        Args:
            entropy: [L] or [B, L]
            hidden_states: [B, L, D]

        Returns:
            scores: [B, L] combined importance scores
        """
        # Normalize entropy to [0, 1]
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(0)

        entropy_norm = (entropy - entropy.min(dim=-1, keepdim=True)[0]) / (
            entropy.max(dim=-1, keepdim=True)[0] - entropy.min(dim=-1, keepdim=True)[0] + 1e-8
        )

        # Learned scores
        learned_scores = self._learned_routing(hidden_states)
        learned_norm = torch.sigmoid(learned_scores)  # Also normalize to [0, 1]

        # Combine: 50% entropy, 50% learned (can be made learnable)
        scores = 0.5 * entropy_norm + 0.5 * learned_norm

        return scores

    def _compute_load_balance_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        K: int,
        L: int,
    ) -> dict:
        """
        Compute load balancing loss to encourage even token distribution.

        This helps prevent the router from always selecting the same tokens.

        Args:
            scores: [B, L] routing scores
            mask: [B, L] routing mask
            K: Number of routed tokens
            L: Total tokens

        Returns:
            dict with 'loss' and statistics
        """
        # Expected routing fraction
        target_frac = K / L

        # Actual routing per position (averaged over batch)
        actual_frac = mask.float().mean(dim=0)  # [L]

        # Encourage uniform distribution (minimize variance)
        # L2 loss: sum((actual - target)^2)
        balance_loss = ((actual_frac - target_frac) ** 2).mean()

        # Scale by weight
        loss = self.load_balance_weight * balance_loss

        return {
            'loss': loss,
            'balance_loss': balance_loss.item(),
            'target_frac': target_frac,
            'actual_frac_mean': actual_frac.mean().item(),
            'actual_frac_std': actual_frac.std().item(),
            'num_routed': K,
        }

    def get_routing_stats(self) -> dict:
        """Return cumulative routing statistics."""
        total_tokens = max(self._total_tokens.item(), 1)
        avg_route_frac = self._total_routed.item() / total_tokens

        return {
            'total_routed': self._total_routed.item(),
            'total_tokens': total_tokens,
            'avg_route_frac': avg_route_frac,
            'target_route_frac': self.route_frac,
            'routing_efficiency': avg_route_frac / max(self.route_frac, 1e-8),
        }

    def reset_stats(self):
        """Reset routing statistics and telemetry history."""
        self._total_routed.zero_()
        self._total_tokens.zero_()
        self._routing_history.clear()

    def get_telemetry(self) -> dict:
        """
        Get detailed telemetry data for monitoring.

        Returns:
            dict with routing history and statistics
        """
        if not self._routing_history:
            return {
                'num_samples': 0,
                'mean_route_frac': 0.0,
                'std_route_frac': 0.0,
                'min_route_frac': 0.0,
                'max_route_frac': 0.0,
            }

        route_fracs = [h['route_frac'] for h in self._routing_history]
        import statistics

        return {
            'num_samples': len(route_fracs),
            'mean_route_frac': statistics.mean(route_fracs),
            'std_route_frac': statistics.stdev(route_fracs) if len(route_fracs) > 1 else 0.0,
            'min_route_frac': min(route_fracs),
            'max_route_frac': max(route_fracs),
            'target_route_frac': self.route_frac,
            'strategy': self.strategy,
            'recent_history': self._routing_history[-10:],  # Last 10 forwards
        }


class AutomatonRouter(nn.Module):
    """
    Finite automaton-based router for structured sequences (e.g., code, math).

    Uses pattern matching to identify tokens that require full attention:
    - Opening/closing brackets
    - Function definitions
    - Control flow keywords
    - Mathematical operators

    This is a specialized router for deterministic patterns.
    """

    def __init__(
        self,
        route_frac: float = 0.15,
        patterns: Optional[list] = None,
    ):
        super().__init__()

        self.route_frac = route_frac

        # Default patterns for code/math
        if patterns is None:
            patterns = [
                'def ', 'class ', 'if ', 'for ', 'while ',  # Code keywords
                '(', ')', '{', '}', '[', ']',  # Brackets
                '+', '-', '*', '/', '=',  # Operators
            ]

        self.patterns = patterns

    def forward(
        self,
        tokens: torch.Tensor,
        token_to_str: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route based on pattern matching.

        Args:
            tokens: [B, L] token IDs
            token_to_str: Function to convert token ID to string

        Returns:
            indices: [B, K] indices for full attention
            mask: [B, L] routing mask
        """
        B, L = tokens.shape
        K = max(1, int(self.route_frac * L))

        # Pattern matching (simplified - in practice use tokenizer)
        mask = torch.zeros(B, L, dtype=torch.bool, device=tokens.device)

        for b in range(B):
            for l in range(L):
                token_str = token_to_str(tokens[b, l].item())
                if any(pat in token_str for pat in self.patterns):
                    mask[b, l] = True

        # If we found more than K, take first K
        # If we found fewer, fill with high-entropy positions
        num_matched = mask.sum(dim=-1)  # [B]

        indices_list = []
        for b in range(B):
            matched_indices = mask[b].nonzero(as_tuple=True)[0]
            if len(matched_indices) >= K:
                indices_list.append(matched_indices[:K])
            else:
                # Fill remaining with random
                remaining = K - len(matched_indices)
                unmatched = (~mask[b]).nonzero(as_tuple=True)[0]
                if len(unmatched) > 0:
                    extra = unmatched[torch.randperm(len(unmatched))[:remaining]]
                    indices_list.append(torch.cat([matched_indices, extra]))
                else:
                    indices_list.append(matched_indices)

        # Pad to same length
        max_len = max(len(idx) for idx in indices_list)
        indices = torch.zeros(B, max_len, dtype=torch.long, device=tokens.device)
        for b, idx in enumerate(indices_list):
            indices[b, :len(idx)] = idx

        # Update mask
        mask.zero_()
        for b in range(B):
            mask[b, indices[b]] = True

        return indices, mask


__all__ = [
    'EntanglementRouter',
    'AutomatonRouter',
]
