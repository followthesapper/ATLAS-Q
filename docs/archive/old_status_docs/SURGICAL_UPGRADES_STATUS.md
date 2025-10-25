# Surgical Upgrades Status

**Date:** October 24, 2025
**Based on:** ChatGPT's 9-point surgical upgrade plan

---

## Progress Summary

### ✅ Completed (5/9)

1. **Wire up persistent learnable MPS cores** - DONE
   - Created `mps_mixer.py` with nn.Parameter cores
   - 485,888 parameters for L=64, D=128, χ_max=32
   - Gradients flow correctly
   - No per-forward rebuilds

2. **Integrate MPS mixer into HybridAQEDLayer** - DONE
   - Created `hybrid_aqed_layer_v2.py`
   - Uses vectorized two-einsum approach (fixed per ChatGPT feedback)
   - Mathematically correct: `left × core × right` contraction
   - Removed obsolete `_mix_proj` layer

3. **Router hardening + telemetry** - DONE
   - All 3 strategies tested (entropy, learned, hybrid)
   - CPU buffers enforced for statistics
   - Telemetry tracking with `.get_telemetry()`
   - Fallback behaviors robust

4. **Backend policy lock-in** - DONE
   - Phase-2 (modular arithmetic): Triton by default
   - Phase-3 (MPS/SVD): PyTorch/cuBLAS by default (`prefer_triton=False`)
   - Created `BACKEND_POLICY.md` documentation
   - Verified across all components

5. **Minimal benchmark harness** - DONE
   - Created `scripts/benchmark_hybrid_aqed_v2.py`
   - Compares Hybrid AQED v2 vs Full Attention
   - Measures latency and memory
   - Results: Currently slower due to MPS scan overhead (see below)

### 🔄 In Progress (0/9)

None currently.

### ⏳ Pending (4/9)

6. **Optimize MPS prefix/suffix scans** - TODO (bottleneck identified)
7. **Training loop integration** - TODO
8. **Profiling & hotspots** - TODO
9. **Automaton-MPO for Phase-2** - TODO
10. **Documentation pass** - TODO

---

## Current Performance

### Benchmark Results (as of latest run)

| L    | d_model | Full Attn (ms) | Hybrid AQED v2 (ms) | Speedup |
|------|---------|----------------|---------------------|---------|
| 128  | 256     | 6.01           | 401.65              | 0.01×   |
| 256  | 512     | 1.91           | 407.07              | 0.00×   |
| 512  | 512     | 3.50           | 1009.03             | 0.00×   |
| 1024 | 1024    | 20.75          | 2017.87             | 0.01×   |

**Status:** ❌ Slower than baseline (expected 5-15× speedup)

### Progress on Mixing Performance

- **Broken concat version:** ~495 ms (mathematically wrong)
- **Current vectorized einsum:** ~127-407 ms (mathematically correct)
- **Improvement:** 3.9× faster than broken version

### Identified Bottleneck

The **MPS prefix/suffix scans** are still using Python loops:

```python
# In MPSMixer.forward() - BOTTLENECK
for k in range(L):
    G = self.cores[k]
    x = H[:, k]
    Gx = torch.einsum('ldh,bd->blh', G, x)
    y = torch.einsum('bl,blh->bh', left, Gx)
    left = y
    left_states.append(left)
```

**Problem:** O(L) Python loop with per-iteration tensor operations

**Solution:** Vectorize the scan itself (use associative scan or batched recurrence)

---

## What Works

### ✅ Mathematically Correct

The mixing layer now does the right math:

```python
# Vectorized contractions (NO Python loops in mixing)
tmp = torch.einsum('blh,lhdm->bldm', prefix, cores)  # [B, L, D, chi_max]
mix = torch.einsum('bldm,blm->bld', tmp, suffix)     # [B, L, D]
x_mixed = x + mix.real
```

### ✅ Persistent Parameters

MPS cores are `nn.Parameter` objects - trained end-to-end, no rebuilds:

```python
>>> layer = HybridAQEDLayer(...)
>>> layer.count_mps_parameters()
485888
>>> out1, _ = layer(x1)
>>> out2, _ = layer(x2)  # Same cores reused
```

### ✅ Router Robustness

All routing strategies work:
- Entropy-based: ✅
- Learned (with gating network): ✅
- Hybrid (entropy + learned): ✅
- Fallbacks: ✅
- Telemetry: ✅

### ✅ Backend Policy

- Phase-2 operations → Triton (15-20× speedup)
- Phase-3 operations → PyTorch/cuBLAS (2-20× speedup)
- Verified with signature inspection

---

## Next Steps

### Priority 1: Optimize MPS Scans (Critical Bottleneck)

**Current Issue:** Python loops in prefix/suffix scans

**Options:**

1. **Associative Scan** (preferred)
   - Use `torch.cumsum` or custom associative scan
   - Fully parallel on GPU
   - O(log L) depth instead of O(L)

2. **Batched Recurrence**
   - Stack all operations and use batched matmuls
   - Still sequential but faster than Python loop

3. **Cache Scans** (if inputs are similar)
   - Pre-compute scans for common inputs
   - Not applicable for training

**Recommended:** Implement associative scan for prefix/suffix computations

### Priority 2: Training Loop Integration

Once scans are optimized:
- Integrate with PyTorch DataLoader
- Add loss functions (cross-entropy for LM)
- Verify end-to-end gradient flow
- Test on small dataset (WikiText-2)

### Priority 3: Profiling

Use PyTorch profiler to identify remaining hotspots:
```python
with torch.profiler.profile(...) as prof:
    out, stats = layer(x)
print(prof.key_averages().table())
```

---

## Files Modified

### Core Implementation

- `src/quantum_hybrid_system/mps_mixer.py`
  - Added `scan=True` parameter
  - Returns padded tensors for vectorization
  - 485k-720k learnable parameters

- `src/quantum_hybrid_system/hybrid_aqed_layer_v2.py`
  - Vectorized two-einsum mixing (ChatGPT's fix)
  - Removed obsolete `_mix_proj`
  - Mathematically correct contractions

- `src/quantum_hybrid_system/entanglement_router.py`
  - CPU buffer enforcement
  - Telemetry tracking (`.get_telemetry()`)
  - All strategies tested

### Backend Policy

- `src/quantum_hybrid_system/mps_triton_integration.py`
  - `prefer_triton=False` (default to PyTorch)
- `src/quantum_hybrid_system/batched_mps_gates.py`
  - `prefer_triton=False`
- `src/quantum_hybrid_system/hybrid_aqed_layer_v2.py`
  - `prefer_triton=False`

### Documentation

- `BACKEND_POLICY.md` - Comprehensive policy document
- `SURGICAL_UPGRADES_STATUS.md` - This file
- `FIXES_APPLIED.md` - Previous fixes from ChatGPT

### Benchmarking

- `scripts/benchmark_hybrid_aqed_v2.py` - Minimal harness

---

## Lessons Learned

### ❌ Mistake 1: Threw Out the Math

**What I did wrong:** Replaced `left × core × right` with `concat(prefix, suffix) → Linear`

**Impact:** Lost model power, mathematically incorrect, slower

**Fix:** Restored vectorized einsums per ChatGPT feedback

### ❌ Mistake 2: Kept Python Loops

**What I did wrong:** "Optimized" by removing einsum but kept `for k in range(L)`

**Impact:** Python loop overhead dominates performance

**Fix:** Vectorized mixing layer, but scans still need work

### ✅ Success 1: Backend Policy

Locked in Phase-2 (Triton) vs Phase-3 (PyTorch) based on empirical benchmarks

### ✅ Success 2: Persistent Cores

MPS cores as `nn.Parameter` - no rebuilds, clean integration with PyTorch

---

## Performance Target

**Goal:** 5-15× speedup over full attention at L≥512

**Current:** 0.01× (100× slower) due to scan overhead

**Gap:** Need to optimize MPS scans to close the ~100-200× gap

---

## Contact

- **Implementation:** Claude Code (Quantum-AQED Integration)
- **Architecture Guidance:** ChatGPT (Surgical Upgrades)
- **Last Updated:** October 24, 2025
