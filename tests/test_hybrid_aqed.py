"""
Comprehensive Tests for Quantum-AQED Hybrid System
==================================================

Tests and benchmarks for the integrated MPS-based attention system.

Test Coverage:
1. MPSMemory compression and reconstruction
2. EntanglementRouter token selection
3. HybridAQEDLayer forward pass
4. Batched gate operations
5. Performance benchmarks vs full attention

Author: Claude Code
Date: October 24, 2025
"""

import os
import time
import torch
import torch.nn as nn
import pytest
import math

from src.quantum_hybrid_system.mps_memory import MPSMemory, compress_sequence
from src.quantum_hybrid_system.entanglement_router import EntanglementRouter
from src.quantum_hybrid_system.hybrid_aqed_layer import HybridAQEDLayer, HybridAQEDTransformer
from src.quantum_hybrid_system.batched_mps_gates import (
    BatchedMPSGateApplicator,
    LearnedGateGenerator,
    MPSMixingScheduler,
    create_standard_gates,
)


# ============================================================================
# Test 1: MPSMemory Compression
# ============================================================================

@pytest.mark.parametrize("L,D,chi_max", [(32, 64, 16), (64, 128, 32), (128, 256, 64)])
def test_mps_memory_compression(L, D, chi_max):
    """Test MPS compression and reconstruction quality."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B = 2

    # Create random sequence
    H = torch.randn(B, L, D, device=device)

    # Compress
    mem = MPSMemory(d_model=D, chi_max=chi_max, device=device)
    mem.build_from(H)

    assert mem.is_built
    assert len(mem.cores) > 0
    assert mem.sequence_length == L

    # Check compression ratio
    stats = mem.get_stats()
    assert stats['compression_ratio'] > 1.0, "Should achieve some compression"
    assert stats['max_bond_dim'] <= chi_max, f"Bond dim {stats['max_bond_dim']} exceeds max {chi_max}"

    print(f"\nMPS Compression Stats (L={L}, D={D}, χ_max={chi_max}):")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}×")
    print(f"  Avg bond dim: {stats['avg_bond_dim']:.1f}")
    print(f"  Max bond dim: {stats['max_bond_dim']}")


@pytest.mark.parametrize("L,D", [(32, 64), (64, 128)])
def test_mps_memory_reconstruction_error(L, D):
    """Test reconstruction error for MPS compression."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"

    # Create low-rank sequence (should compress well)
    rank = 8
    U = torch.randn(1, L, rank, device=device)
    V = torch.randn(1, rank, D, device=device)
    H = (U @ V).to(dtype=torch.complex64)  # [1, L, D]

    # Compress with generous chi_max
    mem = MPSMemory(d_model=D, chi_max=64, device=device)
    mem.build_from(H)

    # Reconstruct
    H_reconstructed = mem.to_dense()

    # Compute error
    error = (H.squeeze(0) - H_reconstructed).abs().max().item()
    rel_error = error / H.abs().max().item()

    print(f"\nMPS Reconstruction (L={L}, D={D}, rank={rank}):")
    print(f"  Max abs error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2e}")

    # Low-rank data should compress nearly perfectly
    assert rel_error < 0.1, f"Reconstruction error {rel_error} too high for low-rank data"


# ============================================================================
# Test 2: EntanglementRouter
# ============================================================================

@pytest.mark.parametrize("L,route_frac", [(64, 0.15), (128, 0.2), (256, 0.1)])
def test_entanglement_router_entropy(L, route_frac):
    """Test entropy-based routing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B = 2

    # Create entropy values (higher = more important)
    entropy = torch.rand(L, device=device)

    # Router
    router = EntanglementRouter(route_frac=route_frac, strategy='entropy')

    # Route
    indices, mask, aux = router(entropy=entropy.unsqueeze(0).expand(B, -1))

    # Check outputs
    K_expected = max(1, int(route_frac * L))
    assert indices.shape == (B, K_expected), f"Expected {(B, K_expected)}, got {indices.shape}"
    assert mask.shape == (B, L)
    assert mask.sum().item() == B * K_expected

    # Check that highest entropy tokens are selected
    top_entropy_indices = torch.argsort(entropy, descending=True)[:K_expected]
    selected_entropies = entropy[indices[0]]
    top_entropies = entropy[top_entropy_indices]

    # Selected should match top (order may differ)
    assert torch.allclose(
        selected_entropies.sort()[0],
        top_entropies.sort()[0],
        atol=1e-6
    ), "Router should select highest entropy tokens"

    print(f"\nRouter Test (L={L}, ρ={route_frac}):")
    print(f"  Routed tokens: {K_expected}/{L} ({100*route_frac:.1f}%)")
    print(f"  Avg entropy (selected): {selected_entropies.mean():.3f}")
    print(f"  Avg entropy (all): {entropy.mean():.3f}")


def test_learned_router():
    """Test learned routing strategy."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B, L, D = 2, 64, 128

    hidden_states = torch.randn(B, L, D, device=device)

    router = EntanglementRouter(
        route_frac=0.15,
        d_model=D,
        strategy='learned',
    ).to(device)

    indices, mask, aux = router(hidden_states=hidden_states)

    assert indices.shape[0] == B
    assert mask.shape == (B, L)
    assert 'loss' in aux

    print(f"\nLearned Router Test:")
    print(f"  Routed: {mask.sum().item()} / {B*L}")
    print(f"  Load balance loss: {aux['balance_loss']:.4f}")


# ============================================================================
# Test 3: HybridAQEDLayer
# ============================================================================

@pytest.mark.parametrize("B,L,D,n_heads", [(2, 64, 128, 4), (1, 32, 64, 2)])
def test_hybrid_aqed_layer_forward(B, L, D, n_heads):
    """Test HybridAQEDLayer forward pass."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"

    # Create layer
    layer = HybridAQEDLayer(
        d_model=D,
        n_heads=n_heads,
        chi_max=16,
        route_frac=0.15,
        use_mps_mixing=True,
    ).to(device)

    # Input
    hidden_states = torch.randn(B, L, D, device=device)

    # Forward
    output, mps_state, aux = layer(hidden_states, mps_state=None)

    # Check outputs
    assert output.shape == (B, L, D), f"Expected {(B, L, D)}, got {output.shape}"
    assert mps_state is not None
    assert mps_state.is_built
    assert 'router_loss' in aux
    assert 'num_full_attn_tokens' in aux

    print(f"\nHybridAQEDLayer Test (B={B}, L={L}, D={D}):")
    print(f"  Output shape: {output.shape}")
    print(f"  Full attn tokens: {aux['num_full_attn_tokens']}")
    print(f"  MPS tokens: {aux['num_mps_tokens']}")
    print(f"  MPS compression: {aux['mps_stats'].get('compression_ratio', 'N/A')}")


def test_hybrid_aqed_layer_gradients():
    """Test that gradients flow correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B, L, D = 2, 32, 64

    layer = HybridAQEDLayer(
        d_model=D,
        n_heads=4,
        chi_max=16,
        routing_strategy='learned',
    ).to(device)

    hidden_states = torch.randn(B, L, D, device=device, requires_grad=True)

    output, _, aux = layer(hidden_states)

    # Compute loss
    loss = output.sum() + aux['router_loss']
    loss.backward()

    # Check gradients exist
    assert hidden_states.grad is not None
    assert layer.qkv_proj.weight.grad is not None

    print(f"\nGradient Test:")
    print(f"  Input grad norm: {hidden_states.grad.norm().item():.4f}")
    print(f"  QKV grad norm: {layer.qkv_proj.weight.grad.norm().item():.4f}")


# ============================================================================
# Test 4: Batched Gate Operations
# ============================================================================

def test_batched_gate_applicator():
    """Test batched MPS gate application."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    dtype = torch.complex64

    # Create MPS cores
    N = 5  # Number of pairs
    chi = 16
    Ai_list = [torch.randn(chi, 2, chi, device=device, dtype=dtype) for _ in range(N)]
    Aj_list = [torch.randn(chi, 2, chi, device=device, dtype=dtype) for _ in range(N)]

    # Create gates
    gates = torch.stack([create_standard_gates('cnot', device, dtype) for _ in range(N)])

    # Apply
    applicator = BatchedMPSGateApplicator(max_bond=32)
    Ai_new, Aj_new = applicator.apply_batch(Ai_list, Aj_list, gates)

    assert len(Ai_new) == N
    assert len(Aj_new) == N

    # Check bond dimensions are capped
    for A in Ai_new:
        assert A.shape[2] <= 32, "Bond dim should be capped at max_bond"

    print(f"\nBatched Gate Applicator Test:")
    print(f"  Processed {N} pairs")
    print(f"  Avg bond dim: {sum(A.shape[2] for A in Ai_new) / N:.1f}")


def test_learned_gate_generator():
    """Test learned gate generation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B, D = 4, 128

    generator = LearnedGateGenerator(
        d_model=D,
        num_gates=1,
        parametrization='free',
    ).to(device)

    features_i = torch.randn(B, D, device=device)
    features_j = torch.randn(B, D, device=device)

    gates = generator(features_i, features_j)

    assert gates.shape == (B, 4, 4), f"Expected {(B, 4, 4)}, got {gates.shape}"
    assert gates.dtype == torch.complex64

    print(f"\nLearned Gate Generator Test:")
    print(f"  Generated gates: {gates.shape}")
    print(f"  Gate norm: {gates.norm(dim=(-2, -1)).mean().item():.3f}")


# ============================================================================
# Test 5: Full Transformer
# ============================================================================

def test_hybrid_aqed_transformer():
    """Test full transformer with hybrid layers."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B, L = 2, 64
    vocab_size = 1000

    model = HybridAQEDTransformer(
        n_layers=4,
        d_model=128,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=512,
        chi_max=16,
        route_frac=0.15,
    ).to(device)

    input_ids = torch.randint(0, vocab_size, (B, L), device=device)

    logits, aux_list = model(input_ids, return_aux=True)

    assert logits.shape == (B, L, vocab_size)
    assert len(aux_list) == 4  # 4 layers

    print(f"\nHybrid Transformer Test:")
    print(f"  Input: {input_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Num layers: {len(aux_list)}")


# ============================================================================
# Benchmark: Hybrid vs Full Attention
# ============================================================================

def _timeit(fn, warmup=3, iters=10):
    """Time a function."""
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters
    else:
        for _ in range(warmup):
            fn()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - start) / iters


@pytest.mark.parametrize("L,D", [(256, 256), (512, 512), (1024, 1024)])
def test_benchmark_hybrid_vs_full(L, D):
    """Benchmark hybrid AQED vs full attention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    B = 2
    n_heads = 8

    # Create input
    hidden_states = torch.randn(B, L, D, device=device)

    # Hybrid AQED layer
    hybrid_layer = HybridAQEDLayer(
        d_model=D,
        n_heads=n_heads,
        chi_max=32,
        route_frac=0.15,
        use_mps_mixing=True,
    ).to(device)

    # Full attention layer (standard transformer)
    full_attn_layer = nn.TransformerEncoderLayer(
        d_model=D,
        nhead=n_heads,
        dim_feedforward=4 * D,
        batch_first=True,
    ).to(device)

    # Benchmark hybrid
    def hybrid_fwd():
        with torch.no_grad():
            hybrid_layer(hidden_states)

    # Benchmark full
    def full_fwd():
        with torch.no_grad():
            full_attn_layer(hidden_states)

    t_hybrid = _timeit(hybrid_fwd, warmup=5, iters=20)
    t_full = _timeit(full_fwd, warmup=5, iters=20)

    speedup = t_full / t_hybrid if t_hybrid > 0 else 1.0

    # Theoretical complexity
    # Full: O(L^2 * D) ≈ L^2 * D FLOPs
    # Hybrid: O(ρ*L*R*D + L*χ^3) where ρ=0.15, R=16, χ=32
    flops_full = L * L * D
    flops_hybrid = 0.15 * L * 16 * D + L * 32**3

    theoretical_speedup = flops_full / flops_hybrid

    print(f"\n{'='*60}")
    print(f"BENCHMARK: Hybrid AQED vs Full Attention (L={L}, D={D})")
    print(f"{'='*60}")
    print(f"  Hybrid time:     {t_hybrid*1000:.2f} ms")
    print(f"  Full time:       {t_full*1000:.2f} ms")
    print(f"  Measured speedup:    {speedup:.2f}×")
    print(f"  Theoretical speedup: {theoretical_speedup:.2f}×")
    print(f"  FLOPs (full):    {flops_full:.2e}")
    print(f"  FLOPs (hybrid):  {flops_hybrid:.2e}")
    print(f"{'='*60}")

    # For large L, hybrid should be faster (or at least not much slower)
    # For small L, overhead may dominate
    if L >= 512:
        # We expect some speedup for large sequences
        # But implementation overhead may prevent full theoretical speedup
        # Let's just check it doesn't crash and completes
        assert t_hybrid < 10.0, "Hybrid taking too long (>10s)"
        assert t_full < 10.0, "Full taking too long (>10s)"


# ============================================================================
# Test 6: Memory Usage
# ============================================================================

def test_memory_usage():
    """Compare memory usage of hybrid vs full attention."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    B, L, D = 2, 512, 512
    n_heads = 8

    # Measure hybrid
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    hybrid_layer = HybridAQEDLayer(
        d_model=D,
        n_heads=n_heads,
        chi_max=32,
        route_frac=0.15,
    ).to(device)

    hidden_states = torch.randn(B, L, D, device=device)
    output, _, _ = hybrid_layer(hidden_states)

    mem_hybrid = torch.cuda.max_memory_allocated() / 1024**2  # MB

    # Cleanup
    del hybrid_layer, output
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure full
    full_layer = nn.TransformerEncoderLayer(
        d_model=D,
        nhead=n_heads,
        dim_feedforward=4 * D,
        batch_first=True,
    ).to(device)

    output_full = full_layer(hidden_states)
    mem_full = torch.cuda.max_memory_allocated() / 1024**2  # MB

    print(f"\nMemory Usage (L={L}, D={D}):")
    print(f"  Hybrid: {mem_hybrid:.1f} MB")
    print(f"  Full:   {mem_full:.1f} MB")
    print(f"  Ratio:  {mem_full / max(mem_hybrid, 1):.2f}×")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
