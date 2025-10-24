"""
Triton GPU kernels for AQED operations.

These kernels fuse multiple operations to reduce kernel launch overhead
and improve memory locality.
"""

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

from .fused_mixer import fused_aqed_mixer, aqed_mixer_forward
from .fast_routing import fast_topk_routing, compute_routing_indices
from .packed_attention import pack_tokens, unpack_tokens, packed_attention_forward

__all__ = [
    'TRITON_AVAILABLE',
    'fused_aqed_mixer',
    'aqed_mixer_forward',
    'fast_topk_routing',
    'compute_routing_indices',
    'pack_tokens',
    'unpack_tokens',
    'packed_attention_forward',
]
