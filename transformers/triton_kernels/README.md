# Triton Kernels (AQED Transformers)

Custom Triton kernels for AQED transformer optimizations.

## Kernels

### `fast_routing.py` - Fast Token Routing
Efficient top-k routing for hybrid attention.

**Used by:** Routed AQED variants
**Purpose:** Select important tokens for full attention

```python
from transformers.triton_kernels.fast_routing import fast_topk_routing
indices = fast_topk_routing(saliency_scores, k=route_frac*L)
```

### `fused_mixer.py` - Fused Mixer Operations
Fused gated mixing for AQED mixer layers.

**Used by:** AQEDMixer (quantum-inspired diffusion)
**Speedup:** 1.5-2× vs separate PyTorch ops

```python
from transformers.triton_kernels.fused_mixer import fused_gated_mix
output = fused_gated_mix(h_i, h_j, W1, W2, W_gate)
```

### `packed_attention.py` - Packed Attention
Efficient attention for variable-length sequences.

**Used by:** Batch processing with padding
**Benefit:** Reduced memory for padded sequences

## Note

For **quantum simulator kernels** (modpow, MPS ops), see:
`triton_kernels/README.md` (root level)

## Legacy

These kernels were developed during AQED optimization experiments.
Current AQED uses PyTorch SDPA (Flash Attention) by default for better compatibility.

Custom kernels can be enabled for further optimization if needed.
