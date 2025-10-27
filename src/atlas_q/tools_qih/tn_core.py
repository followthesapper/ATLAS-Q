import numpy as np
import torch
from torch.linalg import svd

try:
    from . import svd_logger
except ImportError:
    # Direct execution fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import svd_logger
import os


# -------- Robust SVD (multi-driver + jitter + CPU fallback) --------
def svd_robust(T: torch.Tensor, driver: str = "gesvdj"):
    """
    Robust SVD with automatic fallback through multiple drivers and strategies.
    Tries: requested driver -> gesvda -> gesvdj -> gesvd, each with/without jitter,
    finally CPU fallback.
    """
    tried = []

    def _try(drv, jitter=False, cpu=False):
        X = T
        if jitter:
            eps = (T.norm() + 1e-12) * 1e-6
            noise = (torch.randn_like(T.real) + 1j * torch.randn_like(T.real)) * eps
            X = T + noise.to(T.dtype)
        scale = X.norm()
        Xn = X / scale if float(scale) > 0 else X
        if cpu:
            U, S, Vh = torch.linalg.svd(Xn.to("cpu"), full_matrices=False)
            U, S, Vh = U.to(T.device), S.to(T.device), Vh.to(T.device)
        else:
            U, S, Vh = svd(Xn, full_matrices=False, driver=drv)
        if float(scale) > 0:
            S = S * scale
        return U, S, Vh

    # Try requested driver first, then fallback sequence
    for drv in [driver, "gesvda", "gesvdj", "gesvd"]:
        if drv in tried:
            continue
        for jitter in (False, True):
            try:
                return _try(drv, jitter=jitter, cpu=False)
            except Exception as e:
                tried.append(f"{drv}{'+j' if jitter else ''}:{type(e).__name__}")

    # Final fallback: CPU
    try:
        return _try("gesvdj", jitter=True, cpu=True)
    except Exception as e:
        tried.append(f"cpu:{type(e).__name__}")
        raise RuntimeError("All SVD attempts failed: " + " ; ".join(tried))


# -------- Single-qubit rotations --------
def Rx(t, device, dtype):
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    M = torch.empty((2, 2), device=device, dtype=dtype)
    M[0, 0] = c
    M[0, 1] = -1j * s
    M[1, 0] = -1j * s
    M[1, 1] = c
    return M


def Rz(t, device, dtype):
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    M = torch.zeros((2, 2), device=device, dtype=dtype)
    M[0, 0] = (c - 1j * s).to(dtype)
    M[1, 1] = (c + 1j * s).to(dtype)
    return M


def Ry(t, device, dtype):
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    M = torch.empty((2, 2), device=device, dtype=dtype)
    M[0, 0] = c
    M[0, 1] = -s
    M[1, 0] = s
    M[1, 1] = c
    return M


# -------- Entangling gates --------
def CZ_gate(device, dtype):
    """Controlled-Z gate"""
    return torch.diag(torch.tensor([1, 1, 1, -1], device=device, dtype=dtype))


def CNOT_gate(device, dtype):
    """CNOT gate"""
    M = torch.zeros((4, 4), device=device, dtype=dtype)
    M[0, 0] = M[1, 1] = M[2, 3] = M[3, 2] = 1
    return M


def RZZ_gate(theta, device, dtype):
    """RZZ(theta) = exp(-i theta/2 Z⊗Z)"""
    t = torch.as_tensor(theta, device=device, dtype=torch.float32)
    c, s = torch.cos(t / 2).to(dtype), torch.sin(t / 2).to(dtype)
    return torch.diag(torch.stack([c - 1j * s, c + 1j * s, c + 1j * s, c - 1j * s]))


def rand_su4_gate(device, dtype):
    """Random SU(4) via QR decomposition of complex Gaussian"""
    A = torch.randn(4, 4, device=device) + 1j * torch.randn(4, 4, device=device)
    Q, R = torch.linalg.qr(A)
    # Normalize to SU(4)
    phase = torch.angle(torch.linalg.det(Q))
    Q = Q * torch.exp(-1j * phase / 4)
    return Q.to(dtype)


# -------- Build structured entangling gates --------
def build_entangler(entangler: str, theta: float, device, dtype):
    """
    Build a 4x4 two-qubit gate based on entangler type.

    Args:
        entangler: One of 'cz', 'rzz', 'cnot', 'su4', 'none'
        theta: Parameter for structured gates
        device: torch device
        dtype: torch dtype

    Returns:
        4x4 unitary tensor
    """
    if entangler == "none":
        return torch.eye(4, device=device, dtype=dtype)
    elif entangler == "su4":
        return rand_su4_gate(device, dtype)

    # For structured gates: build U_right @ E @ U_left
    angles = [theta * (k + 1) for k in range(12)]
    a, b, c, d, e, f, g, h, i, j, k, l = angles

    # Left local unitaries
    U1_left = Rz(a, device, dtype) @ Rx(b, device, dtype) @ Rz(c, device, dtype)
    U2_left = Rz(d, device, dtype) @ Rx(e, device, dtype) @ Rz(f, device, dtype)
    U_left = torch.kron(U1_left, U2_left)

    # Entangling gate
    if entangler == "cz":
        E = CZ_gate(device, dtype)
    elif entangler == "cnot":
        E = CNOT_gate(device, dtype)
    elif entangler == "rzz":
        E = RZZ_gate(theta, device, dtype)
    else:
        raise ValueError(f"Unknown entangler: {entangler}")

    # Right local unitaries
    U1_right = Rz(g, device, dtype) @ Rx(h, device, dtype) @ Rz(i, device, dtype)
    U2_right = Rz(j, device, dtype) @ Rx(k, device, dtype) @ Rz(l, device, dtype)
    U_right = torch.kron(U1_right, U2_right)

    return U_right @ E @ U_left


# -------- MPS operations --------
def mps_init_plus(n, device="cuda", dtype=torch.complex64):
    """Initialize MPS in |+>^n state"""
    v = torch.tensor([1 / np.sqrt(2), 1 / np.sqrt(2)], device=device, dtype=dtype)
    return [v.view(1, 2, 1).clone() for _ in range(n)]


def mps_apply_1q(cores, i, U):
    """Apply single-qubit gate U to site i"""
    A = cores[i]
    cores[i] = torch.einsum("ldr,du->lur", A, U)
    return cores


SVD_EVENT_LOGGER = svd_logger.get_logger()


def mps_apply_2q(
    cores,
    i,
    j,
    U,
    chi_max=None,
    eps=1e-10,
    svd_driver="gesvdj",
    adaptive=False,
    tol=1e-4,
    ai_predictor=None,
):
    """
    Apply two-qubit gate U to adjacent sites i, j=i+1 with SVD compression.

    Args:
        cores: List of MPS tensors
        i, j: Site indices (must be adjacent, j = i+1)
        U: 4x4 unitary gate
        chi_max: Maximum bond dimension (hard cap)
        eps: Truncation threshold for singular values
        svd_driver: SVD algorithm ('gesvdj', 'gesvda', 'gesvd')
        adaptive: If True, use adaptive truncation based on tol
        tol: Tolerance for adaptive truncation (keep SVs until (1-tol)*total weight)
        ai_predictor: Optional AI model for rank prediction

    Returns:
        cores: Updated MPS cores
        chi: Resulting bond dimension
        truncation_error: Discarded spectral weight
    """
    assert j == i + 1, "Sites must be adjacent"

    Ai, Aj = cores[i], cores[j]
    li, _, ri = Ai.shape
    r_i, _, rj = Aj.shape
    assert ri == r_i, "Bond dimensions must match"

    # Contract and apply gate
    T = torch.einsum("abc,cde->abde", Ai, Aj)
    T = T.reshape(li, 4, rj)
    T = torch.einsum("ldr,du->lur", T, U).reshape(li, 2, 2, rj)
    # Use reshape instead of view for non-contiguous tensors
    T = T.reshape(li * 2, 2 * rj)

    # Robust SVD
    U1, S, Vh = svd_robust(T, driver=svd_driver)
    # Prepare logging ingredients
    S_squared = S**2
    total_weight = S_squared.sum()
    # Adaptive cap implied by tol (used both for real truncation and logging)
    adaptive_keep = S.numel()
    if adaptive and tol > 0:
        cumsum = torch.cumsum(S_squared, dim=0)
        keep_mask = cumsum <= (1 - tol) * total_weight
        if keep_mask.any():
            adaptive_keep = min(keep_mask.sum().item() + 1, S.numel())
        else:
            adaptive_keep = 1
        if chi_max is not None:
            adaptive_keep = min(adaptive_keep, chi_max)
    # Emit a compact record (top-128 SVs, zero-padded)
    try:
        top = min(128, S.numel())
        Sv = S[:top].detach().abs().double().tolist()
        if top < 128:
            Sv = Sv + [0.0] * (128 - top)
        # pre/post info that we have locally
        pre_rank = int(Ai.shape[2])  # right dim of Ai equals bond before gate
        meta_entangler = os.getenv("QIH_ENTANGLER", "unknown")
        SVD_EVENT_LOGGER.log(
            kind="svd_spectrum",
            i=int(i),
            j=int(j),
            pre_rank=pre_rank,
            chi_max=int(chi_max) if chi_max is not None else None,
            tol=float(tol),
            adaptive=bool(adaptive),
            entangler=meta_entangler,
            total_weight=float(
                total_weight.detach().cpu().item()
                if hasattr(total_weight, "detach")
                else float(total_weight)
            ),
            S_top128=Sv,
            adaptive_keep=int(adaptive_keep),
        )
    except Exception:
        pass

    # Calculate total spectral weight for error tracking
    S_squared = S**2
    total_weight = S_squared.sum()

    # Determine truncation strategy

    if ai_predictor is not None and S.numel() > 10:
        # ---------- AI-assisted rank prediction with policy control ----------
        # Policies (env var QIH_AI_POLICY): baseline | guardrail | blend | clamp | clamp+blend
        # Optional: QIH_AI_ALPHA (for 'blend'/'clamp+blend'), default 0.7
        policy = os.getenv("QIH_AI_POLICY", "guardrail").lower()
        try:
            alpha = float(os.getenv("QIH_AI_ALPHA", "0.7"))
        except Exception:
            alpha = 0.7
        alpha = max(0.0, min(1.0, alpha))

        # Proposed AI rank
        ai_keep = int(ai_predictor.predict(S))
        if chi_max is not None:
            ai_keep = min(ai_keep, chi_max)

        # Adaptive cap (how many SVs to satisfy tolerance)
        adaptive_keep = S.numel()
        if adaptive and tol > 0:
            cumsum = torch.cumsum(S_squared, dim=0)
            keep_mask = cumsum <= (1 - tol) * total_weight
            if keep_mask.any():
                adaptive_keep = min(keep_mask.sum().item() + 1, S.numel())
            else:
                adaptive_keep = 1
            if chi_max is not None:
                adaptive_keep = min(adaptive_keep, chi_max)

        if policy == "baseline":
            keep_count = max(1, ai_keep)

        elif policy == "guardrail":
            keep_count = max(1, min(ai_keep, adaptive_keep))

        elif policy == "blend":
            blend = int(round(alpha * ai_keep + (1 - alpha) * adaptive_keep))
            keep_count = max(1, min(blend, adaptive_keep))

        elif policy == "clamp":
            # Assume adapter already clamped AI fraction to [f_min, f_max]
            keep_count = max(1, ai_keep)

        elif policy == "clamp+blend":
            blend = int(round(alpha * ai_keep + (1 - alpha) * adaptive_keep))
            keep_count = max(1, min(blend, adaptive_keep))
        else:
            # Fallback to guardrail
            keep_count = max(1, min(ai_keep, adaptive_keep))

        S_trunc = S[:keep_count]
        U1_trunc = U1[:, :keep_count]
        Vh_trunc = Vh[:keep_count, :]
        # --------------------------------------------------------------------

    elif adaptive and tol > 0:
        # Adaptive truncation based on spectral weight
        cumsum = torch.cumsum(S_squared, dim=0)
        # Keep enough SVs to retain (1-tol) of total weight
        keep_mask = cumsum <= (1 - tol) * total_weight
        if keep_mask.any():
            keep_count = keep_mask.sum().item() + 1  # +1 to ensure we exceed threshold
            keep_count = min(keep_count, S.numel())
        else:
            keep_count = 1  # Keep at least one

        if chi_max is not None:
            keep_count = min(keep_count, chi_max)

        S_trunc = S[:keep_count]
        U1_trunc = U1[:, :keep_count]
        Vh_trunc = Vh[:keep_count, :]
    else:
        # Fixed truncation
        if chi_max is not None and S.numel() > chi_max:
            S_trunc = S[:chi_max]
            U1_trunc = U1[:, :chi_max]
            Vh_trunc = Vh[:chi_max, :]
        else:
            S_trunc = S
            U1_trunc = U1
            Vh_trunc = Vh

    # Calculate truncation error
    kept_weight = (S_trunc**2).sum()
    truncation_error = 1.0 - (kept_weight / total_weight).item() if total_weight > 0 else 0.0

    # Remove very small singular values
    keep = S_trunc > eps
    S_final = S_trunc[keep]
    U1_final = U1_trunc[:, keep]
    Vh_final = Vh_trunc[keep, :]

    chi = S_final.numel()

    # Reshape back to MPS form with singular values distributed
    U1_final = U1_final.reshape(li, 2, chi)
    Vh_final = Vh_final.reshape(chi, 2, rj)
    Ssqrt = torch.sqrt(S_final)

    cores[i] = torch.einsum("ldr,r->ldr", U1_final, Ssqrt)
    cores[j] = torch.einsum("r,rde->rde", Ssqrt, Vh_final)

    return cores, chi, truncation_error
