# Tensor Core Experiment Results

**Date:** October 24, 2025
**Objective:** Achieve 1.5-2.5× speedup on MPS scan using Tensor Cores
**Result:** ❌ **Failed** - 0.91× speedup (9% slowdown)

---

## What We Tried

Implemented complex→real 2×2 block conversion to leverage FP16/TF32 Tensor Cores:

```python
# Complex: z = a + ib
# Real 2×2: [[a, -b],
#            [b,  a]]
```

**Implementation:**
- Created `MPSMixerTensorCore` with `complex_to_real_block()` conversion
- Modified `_tile_inclusive_scan()` to use real batched matmuls
- Added `torch.cuda.amp.autocast()` for BF16/FP16
- Enabled TF32: `torch.backends.cuda.matmul.allow_tf32 = True`

---

## Performance Results

| Configuration | Original (ms) | TensorCore (ms) | Speedup |
|---------------|---------------|-----------------|---------|
| L=256, D=512, χ=32 | 34.47 | 38.79 | 0.89× |
| L=512, D=512, χ=32 | 67.15 | 76.68 | 0.88× |
| L=1024, D=1024, χ=32 | 255.79 | 265.04 | 0.97× |
| **Average** | - | - | **0.91×** |

**Expected:** 1.5-2.5× speedup
**Actual:** 0.91× speedup (9% **slowdown**)

---

## Root Cause Analysis

### 1. Dimension Doubling Overhead

Complex→real 2×2 blocks **doubles** matrix dimensions:
- Original: [B, m, chi, chi] complex64 → 8·B·m·chi² bytes
- Real 2×2: [B, m, 2·chi, 2·chi] float32 → 16·B·m·chi² bytes

**Result:** 2× memory, 4× operations (matrix mult scales as O(n³))

### 2. Small Matrix Size

For chi_max=32:
- Original: 32×32 complex matrices
- Real 2×2: 64×64 real matrices

**Problem:** Too small to amortize Tensor Core setup overhead

### 3. Complex Operations Already Optimized

PyTorch's complex matmul might already use efficient kernels for these sizes.

### 4. Reshape/Conversion Overhead

Converting between formats has non-trivial cost:
```python
# Pack: [B, m, chi, chi] → [B, m, 2·chi, 2·chi]
# Unpack: [B, m, 2·chi, 2·chi] → [B, m, chi, chi]
```

---

## Why This Doesn't Work

ChatGPT's recommendation assumes:
1. **Large matrices** where Tensor Core throughput dominates
2. **Memory-bound operations** where FP16 helps bandwidth
3. **Operations that PyTorch doesn't optimize well**

**Reality for our case:**
1. **Small matrices** (32×32) - setup overhead dominates
2. **Compute-bound** - sequential dependencies in scan
3. **PyTorch complex ops are already good**

---

## Alternative Approaches (Priority Order)

### 1. Associative Scan Primitive ⭐ (Recommended)

Replace sequential scan with parallel associative scan:
- Use PyTorch's native scan if available (torch 2.x+)
- Or implement parallel prefix scan on GPU
- Expected: 2-4× speedup from parallelism

**Why better:**
- Reduces depth from O(tile) to O(log tile) **in parallel**
- No format conversion overhead
- Works well for small matrices

### 2. Larger Bond Dimension

Test with chi_max=64 or 128:
- Larger matrices → better Tensor Core utilization
- May amortize conversion overhead

### 3. Custom CUDA Kernel

Fuse operations: transfer matrix build + scan
- Eliminate intermediate buffers
- Better memory locality
- Expected: 2-3× speedup

### 4. Optimize Tile Size

Current: tile=64 (hardcoded)
- Tune per GPU architecture
- Larger tiles → fewer inter-tile operations
- Trade-off: memory vs parallelism

---

## Lessons Learned

### ❌ Tensor Cores Not Always Faster

- Small matrices (< 64×64) have too much overhead
- Format conversion can dominate performance
- Need to profile, not assume

### ✅ Measure Before Optimizing

- Empirical testing revealed the issue
- Assumptions from large-scale ML don't always apply
- Benchmark early and often

### 🔍 Root Cause Matters

The bottleneck is **sequential depth**, not **FLOPS**:
- MPS scan has O(L) sequential dependencies
- Making each step faster (Tensor Cores) doesn't help much
- Need to **reduce depth** (parallel scan) instead

---

## Next Steps

### Immediate Priority: Associative Scan

Implement parallel prefix scan to reduce sequential depth:

```python
# Instead of:
for k in range(m):
    acc = acc @ cum[k]  # O(m) sequential

# Do:
parallel_prefix_scan(cum)  # O(log m) depth, parallel
```

**Expected gain:** 2-4× from reduced depth + GPU parallelism

### If That Doesn't Work: Kernel Fusion

Custom CUDA kernel combining:
1. Transfer matrix build: `T[k] = cores[k] @ H[:,k]`
2. Prefix scan: Parallel associative scan
3. State extraction: Prefix/suffix vectors

**Expected gain:** 2-3× from fusion + memory locality

---

## Code Artifacts

**Files Created:**
- `src/quantum_hybrid_system/mps_mixer_tensorcore.py` (357 lines)
- `scripts/benchmark_tensorcore_mps.py` (136 lines)

**Status:** Experimental, not used in production

**Recommendation:** Archive for future reference, focus on associative scan

---

## Acknowledgments

- **Optimization Strategy:** ChatGPT
- **Implementation:** Claude Code
- **Finding:** Tensor Cores don't help for small matrices with sequential bottlenecks

---

## Conclusion

Tensor Core acceleration via complex→real 2×2 blocks resulted in a **9% slowdown** instead of the expected 1.5-2.5× speedup. Root cause is dimension doubling overhead for small matrices (chi_max=32).

**Path forward:** Focus on **associative scan** to reduce sequential depth (O(L) → O(log L) in parallel) rather than making sequential steps faster.

**Status:** Experiment concluded ❌ → Pivot to associative scan ✅
