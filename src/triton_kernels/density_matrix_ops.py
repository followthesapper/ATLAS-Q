"""
Triton Kernels for Density Matrix Operations
=============================================

GPU-accelerated operations for mixed-state quantum simulation:
- Kraus operator application (noise channels)
- Partial trace operations
- Purity/entropy calculations

Author: ATLAS-Q Contributors
Date: December 2025
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernel: Kraus Operator Application
# ============================================================================

@triton.jit
def _kraus_apply_kernel(
    # Inputs
    rho_r_ptr, rho_i_ptr,  # Input density matrix [dim, dim] split real/imag
    K_r_ptr, K_i_ptr,      # Kraus operator [d, d] split real/imag
    out_r_ptr, out_i_ptr,  # Output accumulator [dim, dim] split real/imag
    # Dimensions
    dim: tl.constexpr,
    d: tl.constexpr,       # Kraus operator dimension (2 for single-qubit)
    qubit: tl.constexpr,   # Target qubit index
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Apply single Kraus operator to density matrix on specified qubit.

    Computes: out += (I ⊗ K ⊗ I) @ rho @ (I ⊗ K† ⊗ I)

    Uses tiled matrix multiplication for efficiency.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute output indices
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offs < dim
    mask_n = n_offs < dim
    mask = mask_m[:, None] & mask_n[None, :]

    # Accumulator for this tile
    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)

    # For each (m, n) compute: sum over k1, k2 of K[m_q, k1] * rho[m', n'] * K†[k2, n_q]
    # where m' has qubit q replaced by k1, n' has qubit q replaced by k2
    # This is O(d^2) per output element

    # qubit_mask is used for bit manipulation (qubit is a constexpr)
    qubit_mask = 1 << qubit

    for k1 in range(d):
        for k2 in range(d):
            # Compute modified indices
            # m' = (m with qubit q set to k1)
            # n' = (n with qubit q set to k2)
            m_mod = (m_offs & ~qubit_mask) | (k1 << qubit)
            n_mod = (n_offs & ~qubit_mask) | (k2 << qubit)

            # Extract original qubit values
            m_q = (m_offs >> qubit) & 1
            n_q = (n_offs >> qubit) & 1

            # Load K[m_q, k1] and K†[k2, n_q] = conj(K[n_q, k2])
            K_idx1 = m_q * d + k1
            K_idx2 = n_q * d + k2

            K1_r = tl.load(K_r_ptr + K_idx1, mask=mask_m)
            K1_i = tl.load(K_i_ptr + K_idx1, mask=mask_m)
            K2_r = tl.load(K_r_ptr + K_idx2, mask=mask_n)
            K2_i = tl.load(K_i_ptr + K_idx2, mask=mask_n)

            # K† = conj(K), so K2 becomes (K2_r, -K2_i)
            K2_i = -K2_i

            # Load rho[m_mod, n_mod]
            rho_idx = m_mod[:, None] * dim + n_mod[None, :]
            rho_r = tl.load(rho_r_ptr + rho_idx, mask=mask)
            rho_i = tl.load(rho_i_ptr + rho_idx, mask=mask)

            # Complex multiplication: (K1 * rho * K2†)
            # K1 * rho
            tmp_r = K1_r[:, None] * rho_r - K1_i[:, None] * rho_i
            tmp_i = K1_r[:, None] * rho_i + K1_i[:, None] * rho_r

            # tmp * K2†
            acc_r += tmp_r * K2_r[None, :] - tmp_i * K2_i[None, :]
            acc_i += tmp_r * K2_i[None, :] + tmp_i * K2_r[None, :]

    # Atomic add to output
    out_idx = m_offs[:, None] * dim + n_offs[None, :]
    tl.atomic_add(out_r_ptr + out_idx, acc_r, mask=mask)
    tl.atomic_add(out_i_ptr + out_idx, acc_i, mask=mask)


@triton.jit
def _purity_kernel(
    # Input density matrix
    rho_r_ptr, rho_i_ptr,
    # Output purity (scalar)
    purity_ptr,
    # Dimension
    dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Compute purity = Tr(ρ²) using parallel reduction.

    Purity = sum_ij |ρ_ij|²
    """
    pid = tl.program_id(0)

    # Each block handles a range of elements
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < dim * dim

    # Load elements
    rho_r = tl.load(rho_r_ptr + offs, mask=mask, other=0.0)
    rho_i = tl.load(rho_i_ptr + offs, mask=mask, other=0.0)

    # Compute |ρ_ij|²
    mag_sq = rho_r * rho_r + rho_i * rho_i

    # Sum within block
    block_sum = tl.sum(mag_sq)

    # Atomic add to global purity
    tl.atomic_add(purity_ptr, block_sum)


# ============================================================================
# Python Wrappers
# ============================================================================

def apply_kraus_channel_triton(
    rho: torch.Tensor,
    kraus_ops: list,
    qubit: int,
    n_qubits: int
) -> torch.Tensor:
    """
    Apply Kraus channel to density matrix using Triton kernel.

    Args:
        rho: Density matrix [dim, dim] complex tensor
        kraus_ops: List of Kraus operators [d, d] each
        qubit: Target qubit index
        n_qubits: Total number of qubits

    Returns:
        Updated density matrix
    """
    if not rho.is_cuda:
        # Fallback to CPU implementation
        return _apply_kraus_channel_cpu(rho, kraus_ops, qubit, n_qubits)

    dim = rho.shape[0]

    # Split complex into real/imag
    rho_r = rho.real.contiguous()
    rho_i = rho.imag.contiguous()

    # Output accumulator
    out_r = torch.zeros_like(rho_r)
    out_i = torch.zeros_like(rho_i)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(dim, BLOCK_M), triton.cdiv(dim, BLOCK_N))

    for K in kraus_ops:
        K = K.to(rho.device, dtype=rho.dtype)
        d = K.shape[0]
        K_r = K.real.contiguous()
        K_i = K.imag.contiguous()

        _kraus_apply_kernel[grid](
            rho_r, rho_i,
            K_r, K_i,
            out_r, out_i,
            dim=dim,
            d=d,
            qubit=qubit,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    return torch.complex(out_r, out_i)


def purity_triton(rho: torch.Tensor) -> float:
    """
    Compute purity Tr(ρ²) using Triton kernel.

    Args:
        rho: Density matrix [dim, dim] complex tensor

    Returns:
        Purity value in [0, 1]
    """
    if not rho.is_cuda:
        # CPU fallback
        return torch.trace(rho @ rho).real.item()

    dim = rho.shape[0]

    rho_r = rho.real.contiguous()
    rho_i = rho.imag.contiguous()

    purity = torch.zeros(1, device=rho.device, dtype=torch.float64)

    BLOCK = 1024
    grid = (triton.cdiv(dim * dim, BLOCK),)

    _purity_kernel[grid](
        rho_r, rho_i,
        purity,
        dim=dim,
        BLOCK=BLOCK,
    )

    return purity.item()


def _apply_kraus_channel_cpu(
    rho: torch.Tensor,
    kraus_ops: list,
    qubit: int,
    n_qubits: int
) -> torch.Tensor:
    """CPU fallback for Kraus channel application."""
    dim = rho.shape[0]
    result = torch.zeros_like(rho)

    for K in kraus_ops:
        # Build full Kraus operator via Kronecker product
        K_full = K
        for i in range(n_qubits):
            if i < qubit:
                K_full = torch.kron(torch.eye(2, dtype=K.dtype, device=K.device), K_full)
            elif i > qubit:
                K_full = torch.kron(K_full, torch.eye(2, dtype=K.dtype, device=K.device))

        result += K_full @ rho @ K_full.conj().T

    return result


# ============================================================================
# Optimized Gate Application using Rust Backend
# ============================================================================

def try_import_rust_backend():
    """Try to import the Rust backend for faster operations."""
    try:
        import atlas_q_core
        return atlas_q_core
    except ImportError:
        return None


def apply_gate_rust(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit: int
) -> torch.Tensor:
    """
    Apply gate using Rust backend if available.

    Falls back to PyTorch if Rust not compiled.
    """
    rust = try_import_rust_backend()

    if rust is not None and state.is_cpu and gate.shape == (2, 2):
        # Use Rust statevector backend for single-qubit gates
        n_qubits = int(torch.log2(torch.tensor(state.shape[0])).item())
        sim = rust.StatevectorSimulatorRust(n_qubits)

        # Set state
        state_flat = state.flatten().numpy()
        sim.set_state([s.real for s in state_flat], [s.imag for s in state_flat])

        # Apply gate
        gate_flat = [
            gate[0, 0].real, gate[0, 0].imag,
            gate[0, 1].real, gate[0, 1].imag,
            gate[1, 0].real, gate[1, 0].imag,
            gate[1, 1].real, gate[1, 1].imag,
        ]
        sim.apply_single_gate_raw(qubit, gate_flat)

        # Get result
        real, imag = sim.get_state()
        return torch.tensor([complex(r, i) for r, i in zip(real, imag)], dtype=state.dtype)

    # PyTorch fallback
    n_qubits = int(torch.log2(torch.tensor(state.shape[0])).item())
    U = gate
    for i in range(n_qubits):
        if i < qubit:
            U = torch.kron(torch.eye(2, dtype=gate.dtype, device=gate.device), U)
        elif i > qubit:
            U = torch.kron(U, torch.eye(2, dtype=gate.dtype, device=gate.device))

    return U @ state
