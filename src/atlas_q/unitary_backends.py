"""
Unitary-preserving backends for applying Pauli string rotations to MPS.

These implementations guarantee near-perfect norm preservation for exp(i*lambda*P) gates.
Opt-in dense path is safest for small N; includes robust SVD fallback.
"""

from __future__ import annotations

import numpy as np
import torch


def _pauli_matrix(letter: str, dtype, device):
    if letter == 'I':
        return torch.eye(2, dtype=dtype, device=device)
    if letter == 'X':
        return torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    if letter == 'Y':
        return torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    if letter == 'Z':
        return torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    raise ValueError(f"Invalid Pauli letter: {letter}")


def _safe_svd(A: torch.Tensor, add_eps: float = 1e-10):
    """
    Robust SVD that avoids GPU cusolver stalls:
      1) try torch.linalg.svd on current device
      2) tiny Tikhonov (add_eps * I) if square to stabilize
      3) CPU fallback SVD if needed, then move factors back
    """
    dev = A.device
    dtype = A.dtype
    try:
        A_use = A
        if add_eps > 0 and A.shape[-2] == A.shape[-1]:
            A_use = A + add_eps * torch.eye(A.shape[-1], dtype=dtype, device=dev)
        return torch.linalg.svd(A_use, full_matrices=False)
    except RuntimeError:
        A_cpu = A.detach().cpu()
        if add_eps > 0 and A_cpu.shape[-2] == A_cpu.shape[-1]:
            A_cpu = A_cpu + add_eps * torch.eye(A_cpu.shape[-1], dtype=A_cpu.dtype, device='cpu')
        Uc, S, Vhc = torch.linalg.svd(A_cpu, full_matrices=False)
        return Uc.to(dev), S.to(dev), Vhc.to(dev)


def apply_pauli_string_rotation_dense(mps, pauli_string: str, lam: float, chi_max: int = 256):
    """
    Apply U = exp(i * lam * P) where P is a Pauli string, using dense statevector.

    SAFE path for N <= ~16. Preserves norm (re-normalizes if tiny drift).
    """
    dtype = mps.tensors[0].dtype
    device = mps.tensors[0].device
    N = mps.num_qubits
    if len(pauli_string) != N:
        raise ValueError(f"Pauli string length {len(pauli_string)} != num_qubits {N}")

    # Statevector
    psi = mps.to_statevector()  # (2**N,)

    # Build P
    P = _pauli_matrix(pauli_string[0], dtype, device)
    for letter in pauli_string[1:]:
        P = torch.kron(P, _pauli_matrix(letter, dtype, device))

    # U = cos(l) I + i sin(l) P
    dim = 2 ** N
    I = torch.eye(dim, dtype=dtype, device=device)
    c = torch.cos(torch.tensor(lam, dtype=torch.float64))
    s = torch.sin(torch.tensor(lam, dtype=torch.float64))
    U = c.to(dtype) * I + (1j * s).to(dtype) * P

    psi_new = U @ psi
    # Renormalize (paranoid)
    norm = torch.linalg.norm(psi_new)
    if abs(norm - 1.0) > 1e-12:
        print(f"[WARNING] Dense unitary norm error: {abs(norm - 1.0):.2e} (renormalizing)")
    psi_new = psi_new / norm

    # statevector -> MPS via sequential SVD
    state_tensor = psi_new.reshape([2] * N)
    new_cores = []
    tensor = state_tensor
    chi_left = 1

    for i in range(N - 1):
        tensor = tensor.reshape(chi_left * 2, -1)
        U_svd, S, Vh = _safe_svd(tensor, add_eps=1e-10)

        chi_right = min(chi_max, len(S))
        U_svd = U_svd[:, :chi_right]
        S = S[:chi_right]
        Vh = Vh[:chi_right, :]

        core = U_svd.reshape(chi_left, 2, chi_right).contiguous()
        new_cores.append(core)

        S_diag = torch.diag(S.to(dtype))
        tensor = S_diag @ Vh
        tensor = tensor.reshape([chi_right] + [2] * (N - i - 1))
        chi_left = chi_right

    new_cores.append(tensor.reshape(chi_left, 2, 1).contiguous())
    mps.tensors = new_cores
    mps.to_left_canonical()


def apply_pauli_string_rotation_circuit(mps, pauli_string: str, lam: float, chi_max: int = 256):
    """
    FUTURE: scalable decomposition with basis change + parity ladder + Rz + uncompute.
    """
    raise NotImplementedError("Circuit backend not yet implemented; use 'dense' backend.")
