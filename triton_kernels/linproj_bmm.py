# triton_kernels/linproj_bmm.py
# Fused Linformer-style sequence projection for K and V:
#   K_proj = E^T @ K,  V_proj = E^T @ V
# Shapes:
#   E:        [L, r]    (we use E_T = E^T: [r, L] internally)
#   K, V:     [B, H, L, Dh]
#   K_proj,V: [B, H, r, Dh]
#
# All tensors are real-valued, CUDA-contiguous recommended.
# Accumulate in fp32, output in input dtype.

import torch
import triton
import triton.language as tl

def _ceildiv(a, b): return (a + b - 1) // b

@triton.jit
def _linproj_one_kernel(
    E_T_ptr,         # *const T  [r, L]
    X_ptr,           # *const T  [L, Dh]   (this is per (B,H) slice)
    Y_ptr,           # *      T  [r, Dh]
    r: tl.constexpr,
    L: tl.constexpr,
    Dh: tl.constexpr,
    stride_e_r, stride_e_l,     # strides for E_T
    stride_x_l, stride_x_d,     # strides for X
    stride_y_r, stride_y_d,     # strides for Y
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    r_idx = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    d_idx = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)

    # masks for bounds
    r_mask = r_idx < r
    d_mask = d_idx < Dh

    # accumulator
    acc = tl.zeros((BLOCK_R, BLOCK_D), dtype=ACC_DTYPE)

    # reduction over L (sequence length)
    for l0 in range(0, L, BLOCK_L):
        l_idx = l0 + tl.arange(0, BLOCK_L)
        l_mask = l_idx < L

        # load E_T tile: [BLOCK_R, BLOCK_L]
        e_ptrs = E_T_ptr + (r_idx[:, None] * stride_e_r + l_idx[None, :] * stride_e_l)
        e_tile = tl.load(e_ptrs, mask=r_mask[:, None] & l_mask[None, :], other=0.0)

        # load X tile: [BLOCK_L, BLOCK_D]
        x_ptrs = X_ptr + (l_idx[:, None] * stride_x_l + d_idx[None, :] * stride_x_d)
        x_tile = tl.load(x_ptrs, mask=l_mask[:, None] & d_mask[None, :], other=0.0)

        # Cast both to accumulator dtype for dot product
        e_tile_acc = e_tile.to(ACC_DTYPE)
        x_tile_acc = x_tile.to(ACC_DTYPE)

        # acc += e_tile @ x_tile
        acc += tl.dot(e_tile_acc, x_tile_acc)

    # write back
    y_ptrs = Y_ptr + (r_idx[:, None] * stride_y_r + d_idx[None, :] * stride_y_d)
    # cast to output dtype
    acc_out = acc
    tl.store(y_ptrs, acc_out, mask=r_mask[:, None] & d_mask[None, :])

def _pick_blocks(r, Dh, L):
    # Tuned heuristics for better GPU utilization
    # Larger blocks = fewer launches, better fusion
    if Dh >= 128:
        BLOCK_D = 128
    elif Dh >= 64:
        BLOCK_D = 64
    else:
        BLOCK_D = 64  # Increased from 32 for better occupancy

    if r >= 128:
        BLOCK_R = 128
    elif r >= 64:
        BLOCK_R = 64
    else:
        BLOCK_R = 64  # Increased from 32 for better occupancy

    # Larger reduction tiles for big sequences (L=4096)
    if L >= 4096:
        BLOCK_L = 256  # Increased from 128 for better memory coalescing
    elif L >= 1024:
        BLOCK_L = 128
    elif L >= 512:
        BLOCK_L = 64
    else:
        BLOCK_L = 64  # Increased from 32

    return BLOCK_R, BLOCK_D, BLOCK_L

@torch.no_grad()
def linproj_bmm_fused(E, K, V):
    """
    Fused projection for K and V with shared E:
      E: [L, r] (float16/bfloat16/float32)
      K: [B, H, L, Dh]
      V: [B, H, L, Dh]
    Returns:
      Kp, Vp: [B, H, r, Dh] with same dtype as inputs
    """
    assert E.is_cuda and K.is_cuda and V.is_cuda, "All tensors must be CUDA"
    assert E.dim() == 2 and K.dim() == 4 and V.dim() == 4, \
        f"Expected E.dim=2, K.dim=4, V.dim=4, got {E.dim()}, {K.dim()}, {V.dim()}"
    L, r = E.shape
    B, H, Lk, Dh = K.shape
    B2, H2, Lv, Dh2 = V.shape
    assert Lk == L and Lv == L and Dh == Dh2 and B == B2 and H == H2, \
        f"Shape mismatch: E=[{L},{r}], K=[{B},{H},{Lk},{Dh}], V=[{B2},{H2},{Lv},{Dh2}]"

    device = K.device
    dtype = K.dtype
    # Map torch dtype to triton dtype for accumulator
    acc_dtype_triton = tl.float32

    # Pretranspose E to [r, L] for friendly memory access
    E_T = E.t().contiguous()

    # Output buffers
    Kp = torch.empty((B, H, r, Dh), device=device, dtype=dtype)
    Vp = torch.empty((B, H, r, Dh), device=device, dtype=dtype)

    # Strides (row-major-like assumptions; still computed generically)
    stride_e_r, stride_e_l = E_T.stride()
    # one (B,H) slice is [L, Dh]
    stride_x_l = K.stride(-2)   # along L
    stride_x_d = K.stride(-1)   # along Dh
    stride_y_r = Kp.stride(-2)  # along r
    stride_y_d = Kp.stride(-1)  # along Dh

    # Tiling
    BLOCK_R, BLOCK_D, BLOCK_L = _pick_blocks(r, Dh, L)
    grid = ( _ceildiv(r, BLOCK_R), _ceildiv(Dh, BLOCK_D) )

    # Launch per (B,H) slice
    for b in range(B):
        for h in range(H):
            K_bh = K[b, h]
            V_bh = V[b, h]
            Kp_bh = Kp[b, h]
            Vp_bh = Vp[b, h]

            # Use 8 warps for better occupancy on larger blocks
            num_warps = 8 if (BLOCK_R >= 64 and BLOCK_D >= 64) else 4

            _linproj_one_kernel[grid](
                E_T, K_bh, Kp_bh,
                r, L, Dh,
                stride_e_r, stride_e_l,
                stride_x_l, stride_x_d,
                stride_y_r, stride_y_d,
                BLOCK_R=BLOCK_R,
                BLOCK_D=BLOCK_D,
                BLOCK_L=BLOCK_L,
                ACC_DTYPE=acc_dtype_triton,
                num_warps=num_warps, num_stages=2
            )

            _linproj_one_kernel[grid](
                E_T, V_bh, Vp_bh,
                r, L, Dh,
                stride_e_r, stride_e_l,
                stride_x_l, stride_x_d,
                stride_y_r, stride_y_d,
                BLOCK_R=BLOCK_R,
                BLOCK_D=BLOCK_D,
                BLOCK_L=BLOCK_L,
                ACC_DTYPE=acc_dtype_triton,
                num_warps=num_warps, num_stages=2
            )

    return Kp, Vp


__all__ = ['linproj_bmm_fused']
