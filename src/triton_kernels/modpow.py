"""
Triton Kernel for Batched Modular Exponentiation

Optimized GPU kernel for computing a^x mod N for many values of x in parallel.
This is the critical operation for period-finding in Shor's algorithm.

Key optimizations:
1. Binary exponentiation (square-and-multiply)
2. Memory coalescing for better bandwidth utilization
3. Efficient modular arithmetic
4. Batched operation for parallelism

Expected speedup: 2-3× over CuPy CUDA kernel

Author: ATLAS-Q Contributors
Date: October 2025
"""

import time
from typing import List, Optional

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def modpow_kernel(
    # Input pointers
    bases_ptr,      # [N] - base values (typically all same value 'a')
    exponents_ptr,  # [N] - exponent values to compute
    moduli_ptr,     # [N] - modulus values (typically all same value 'N')
    # Output pointer
    results_ptr,    # [N] - output: base^exponent mod modulus
    # Sizes
    n_elements,     # Total number of elements
    # Block size (compile-time constant)
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for batched modular exponentiation

    Computes: results[i] = bases[i]^exponents[i] mod moduli[i]

    Uses binary exponentiation algorithm:
    - If exponent bit is 1: multiply result by current base
    - Square the base and shift exponent right
    - Repeat until exponent is 0
    """
    # Get program ID and compute offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Compute offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid elements
    mask = offsets < n_elements

    # Load inputs
    base = tl.load(bases_ptr + offsets, mask=mask, other=0)
    exponent = tl.load(exponents_ptr + offsets, mask=mask, other=0)
    modulus = tl.load(moduli_ptr + offsets, mask=mask, other=1)

    # Initialize result to 1
    result = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)

    # Reduce base modulo modulus first
    base = base % modulus

    # Binary exponentiation
    # We need to loop, but Triton doesn't support while loops well
    # So we use a fixed number of iterations (64 bits max for int64)
    # Note: Triton doesn't support break, so we iterate all 64 times
    for _ in range(64):
        # If exponent is odd, multiply result by base
        is_odd = (exponent & 1) == 1
        result = tl.where(is_odd, (result * base) % modulus, result)

        # Square base and halve exponent
        base = (base * base) % modulus
        exponent = exponent >> 1

    # Store results
    tl.store(results_ptr + offsets, result, mask=mask)


@triton.jit
def modpow_check_kernel(
    # Input pointers
    bases_ptr,      # [N] - base values
    exponents_ptr,  # [N] - exponent values (period candidates)
    moduli_ptr,     # [N] - modulus values
    # Output pointer
    is_period_ptr,  # [N] - output: 1 if base^exponent mod modulus == 1, else 0
    # Sizes
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel that computes modpow AND checks if result == 1

    This is optimized for period finding where we want to find the
    first exponent r such that a^r ≡ 1 (mod N)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    base = tl.load(bases_ptr + offsets, mask=mask, other=0)
    exponent = tl.load(exponents_ptr + offsets, mask=mask, other=0)
    modulus = tl.load(moduli_ptr + offsets, mask=mask, other=1)

    # Initialize result
    result = tl.full([BLOCK_SIZE], 1, dtype=tl.int64)
    base = base % modulus

    # Binary exponentiation
    for _ in range(64):
        is_odd = (exponent & 1) == 1
        result = tl.where(is_odd, (result * base) % modulus, result)
        base = (base * base) % modulus
        exponent = exponent >> 1

    # Check if result == 1 (this is a period!)
    is_one = (result == 1).to(tl.int64)

    # Store result
    tl.store(is_period_ptr + offsets, is_one, mask=mask)


def batched_modpow_triton(a: int, exponents: List[int], N: int,
                          device: str = 'cuda') -> torch.Tensor:
    """
    Batched modular exponentiation using Triton

    Computes a^x mod N for each x in exponents

    Args:
        a: Base value
        exponents: List of exponent values
        N: Modulus
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Tensor of results: [a^e mod N for e in exponents]
    """
    n = len(exponents)

    # Convert inputs to tensors
    bases = torch.full((n,), a, dtype=torch.int64, device=device)
    exps = torch.tensor(exponents, dtype=torch.int64, device=device)
    mods = torch.full((n,), N, dtype=torch.int64, device=device)
    results = torch.zeros(n, dtype=torch.int64, device=device)

    # Choose block size based on problem size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    # Launch kernel
    modpow_kernel[grid](
        bases, exps, mods,
        results,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return results


def batched_modpow_check_triton(a: int, candidates: List[int], N: int,
                                device: str = 'cuda') -> Optional[int]:
    """
    Find first candidate r where a^r ≡ 1 (mod N)

    This is optimized for period finding in Shor's algorithm.

    Args:
        a: Base value
        candidates: List of period candidates to check
        N: Modulus
        device: Device to use

    Returns:
        First r where a^r ≡ 1 (mod N), or None if not found
    """
    n = len(candidates)

    # Convert inputs to tensors
    bases = torch.full((n,), a, dtype=torch.int64, device=device)
    cands = torch.tensor(candidates, dtype=torch.int64, device=device)
    mods = torch.full((n,), N, dtype=torch.int64, device=device)
    is_period = torch.zeros(n, dtype=torch.int64, device=device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)

    modpow_check_kernel[grid](
        bases, cands, mods,
        is_period,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Find first match
    matches = torch.nonzero(is_period)
    if len(matches) > 0:
        idx = matches[0].item()
        return candidates[idx]

    return None


def batched_modpow_numpy(a: int, exponents: List[int], N: int) -> List[int]:
    """NumPy baseline: compute using Python's built-in pow()"""
    return [pow(a, e, N) for e in exponents]


def batched_modpow_torch(a: int, exponents: List[int], N: int,
                         device: str = 'cuda') -> List[int]:
    """PyTorch baseline: compute using torch.pow() with modulo"""
    # Note: PyTorch doesn't have built-in modular exponentiation
    # So this is just for comparison and uses Python's pow()
    return batched_modpow_numpy(a, exponents, N)


def benchmark_modpow_implementations(
    a: int = 3,
    N: int = 15,
    n_exponents: int = 10000,
    n_trials: int = 10
):
    """
    Benchmark different modular exponentiation implementations

    Args:
        a: Base value
        N: Modulus
        n_exponents: Number of exponents to test
        n_trials: Number of trials for timing

    Returns:
        Dictionary with timing results
    """
    print(f"\n{'='*70}")
    print(f"Modular Exponentiation Benchmark")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Base (a): {a}")
    print(f"  Modulus (N): {N}")
    print(f"  Number of exponents: {n_exponents:,}")
    print(f"  Trials: {n_trials}")

    # Generate random exponents
    np.random.seed(42)
    exponents = np.random.randint(1, N, size=n_exponents).tolist()

    results = {}

    # 1. NumPy baseline (CPU)
    print(f"\n[1/3] NumPy (CPU) baseline...")
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result_numpy = batched_modpow_numpy(a, exponents, N)
        times.append(time.perf_counter() - start)

    numpy_time = np.mean(times)
    results['numpy'] = {
        'time': numpy_time,
        'throughput': n_exponents / numpy_time,
        'result': result_numpy
    }
    print(f"  Time: {numpy_time*1000:.2f} ms")
    print(f"  Throughput: {n_exponents/numpy_time:,.0f} ops/sec")

    # 2. Triton (GPU)
    print(f"\n[2/3] Triton (GPU)...")
    # Warmup
    for _ in range(3):
        _ = batched_modpow_triton(a, exponents, N)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result_triton = batched_modpow_triton(a, exponents, N)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    triton_time = np.mean(times)
    results['triton'] = {
        'time': triton_time,
        'throughput': n_exponents / triton_time,
        'result': result_triton.cpu().tolist()
    }
    print(f"  Time: {triton_time*1000:.2f} ms")
    print(f"  Throughput: {n_exponents/triton_time:,.0f} ops/sec")
    print(f"  Speedup vs NumPy: {numpy_time/triton_time:.2f}×")

    # 3. CuPy (if available)
    try:
        import cupy as cp
        print(f"\n[3/3] CuPy (GPU)...")

        # Compile CuPy kernel
        cupy_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void batched_modpow(const long long* bases,
                           const long long* exponents,
                           const long long* moduli,
                           long long* results,
                           const int n) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;

            long long base = bases[idx];
            long long exp = exponents[idx];
            long long mod = moduli[idx];
            long long result = 1;

            base = base % mod;

            while (exp > 0) {
                if (exp % 2 == 1) {
                    result = (result * base) % mod;
                }
                exp = exp >> 1;
                base = (base * base) % mod;
            }

            results[idx] = result;
        }
        ''', 'batched_modpow')

        # Prepare arrays
        bases_cp = cp.full(n_exponents, a, dtype=cp.int64)
        exps_cp = cp.array(exponents, dtype=cp.int64)
        mods_cp = cp.full(n_exponents, N, dtype=cp.int64)
        results_cp = cp.zeros(n_exponents, dtype=cp.int64)

        # Warmup
        for _ in range(3):
            cupy_kernel((n_exponents // 256 + 1,), (256,),
                       (bases_cp, exps_cp, mods_cp, results_cp, n_exponents))
        cp.cuda.Stream.null.synchronize()

        times = []
        for _ in range(n_trials):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            cupy_kernel((n_exponents // 256 + 1,), (256,),
                       (bases_cp, exps_cp, mods_cp, results_cp, n_exponents))
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        cupy_time = np.mean(times)
        results['cupy'] = {
            'time': cupy_time,
            'throughput': n_exponents / cupy_time,
            'result': cp.asnumpy(results_cp).tolist()
        }
        print(f"  Time: {cupy_time*1000:.2f} ms")
        print(f"  Throughput: {n_exponents/cupy_time:,.0f} ops/sec")
        print(f"  Speedup vs NumPy: {numpy_time/cupy_time:.2f}×")
        print(f"  Triton vs CuPy: {cupy_time/triton_time:.2f}× (Triton advantage)")

    except ImportError:
        print(f"\n[3/3] CuPy not available (skipping)")
        results['cupy'] = None

    # Verify correctness
    print(f"\n{'='*70}")
    print("Correctness Verification:")

    # Check first 10 results
    print(f"  First 10 results (NumPy):  {results['numpy']['result'][:10]}")
    print(f"  First 10 results (Triton): {results['triton']['result'][:10]}")

    # Verify all match
    numpy_results = results['numpy']['result']
    triton_results = results['triton']['result']

    matches = sum(1 for n, t in zip(numpy_results, triton_results) if n == t)
    print(f"  Matching results: {matches}/{n_exponents} ({100*matches/n_exponents:.2f}%)")

    if matches == n_exponents:
        print(f"  ✓ All results match!")
    else:
        print(f"  ✗ Results don't match!")
        # Show first mismatch
        for i, (n, t) in enumerate(zip(numpy_results, triton_results)):
            if n != t:
                print(f"    First mismatch at index {i}:")
                print(f"      NumPy: {a}^{exponents[i]} mod {N} = {n}")
                print(f"      Triton: {a}^{exponents[i]} mod {N} = {t}")
                break

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  NumPy (CPU):  {numpy_time*1000:.2f} ms")
    print(f"  Triton (GPU): {triton_time*1000:.2f} ms ({numpy_time/triton_time:.2f}× speedup)")
    if results['cupy']:
        print(f"  CuPy (GPU):   {results['cupy']['time']*1000:.2f} ms ({numpy_time/results['cupy']['time']:.2f}× speedup)")
        print(f"  Triton vs CuPy: {results['cupy']['time']/triton_time:.2f}×")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_modpow_implementations(
        a=7,
        N=15,
        n_exponents=100000,
        n_trials=20
    )
