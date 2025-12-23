"""
VRA Core Functions
==================

Core spectral analysis functions from Vaca Resonance Analysis (VRA).
Ported from VRA project for integration with ATLAS-Q.

Mathematical Foundation:
- Treats modular sequences as resonant phase lattices
- Coherent averaging across multiplicative group Z_N*
- Universal coherence law: C = exp(-V_φ/2)
- Threshold at e^-2 ≈ 0.1353 marks coherence collapse

Author: Adapted from VRA by Dylan Vaca
"""

from typing import List, Optional, Tuple

import numpy as np


def multiplicative_order(a: int, N: int, max_iter: int = 10000) -> Optional[int]:
    """
    Compute multiplicative order of a modulo N.

    Finds smallest r > 0 such that a^r ≡ 1 (mod N).

    Parameters
    ----------
    a : int
        Base element
    N : int
        Modulus
    max_iter : int, optional
        Maximum iterations (default: 10000)

    Returns
    -------
    int or None
        Order r, or None if gcd(a, N) != 1

    Examples
    --------
    >>> multiplicative_order(7, 15)
    4
    >>> multiplicative_order(2, 15)
    4
    """
    if np.gcd(a, N) != 1:
        return None

    x = a
    for r in range(1, min(max_iter, N)):
        if x == 1:
            return r
        x = (x * a) % N

    return None


def modular_sequence(N: int, a: int, x0: int, length: int) -> np.ndarray:
    """
    Generate modular iteration sequence.

    Creates sequence x_i where x_{i+1} = a * x_i mod N.

    Parameters
    ----------
    N : int
        Modulus
    a : int
        Base (multiplier)
    x0 : int
        Starting seed
    length : int
        Sequence length

    Returns
    -------
    np.ndarray
        Integer sequence of shape (length,)
    """
    xs = np.zeros(length, dtype=np.int64)
    xs[0] = x0
    for i in range(1, length):
        xs[i] = (a * xs[i-1]) % N
    return xs


def phase_embed(xs: np.ndarray, N: int) -> np.ndarray:
    """
    Phase embedding into complex unit circle.

    Maps modular sequence to complex exponentials:
    u_i = exp(2πj * x_i / N)

    Parameters
    ----------
    xs : np.ndarray
        Modular sequence
    N : int
        Modulus (phase normalization)

    Returns
    -------
    np.ndarray
        Complex signal on unit circle
    """
    phases = 2.0 * np.pi * xs / N
    return np.exp(1j * phases)


def compute_averaged_spectrum(
    N: int,
    bases: List[int],
    x0: int,
    length: int,
    zp: int = 16,
    window: str = "hann"
) -> np.ndarray:
    """
    Compute coherently averaged power spectrum across multiple bases.

    CRITICAL: Performs coherent averaging by summing complex FFTs before
    squaring. This preserves phase relationships and enables SNR scaling.

    Coherent: |Σ U_m / M|² → SNR scales as √M (validated)
    Incoherent: Σ|U_m|²/M → No SNR gain

    Parameters
    ----------
    N : int
        Modulus
    bases : List[int]
        List of M bases with same multiplicative order
    x0 : int
        Starting seed (typically 1)
    length : int
        Sequence length before zero-padding
    zp : int, optional
        Zero-padding factor (default: 16)
    window : str, optional
        Window function: 'hann', 'hamming', 'blackman', 'none' (default: 'hann')

    Returns
    -------
    np.ndarray
        Coherently averaged power spectrum of shape (length * zp,)

    Notes
    -----
    VRA achieves:
    - +5.87 dB per doubling of length L (validated in E16)
    - +3.0 dB per doubling of bases M (limited by phase incoherence)
    - Professional-grade SNR: 36-58 dB
    """
    M = len(bases)
    L = length * zp
    U_sum = None

    for a in bases:
        # Generate modular sequence
        xs = modular_sequence(N, a, x0, length)

        # Phase embed
        us = phase_embed(xs, N)

        # Apply window
        if window == "hann":
            w = np.hanning(length)
        elif window == "hamming":
            w = np.hamming(length)
        elif window == "blackman":
            w = np.blackman(length)
        else:
            w = np.ones(length)

        us_windowed = us * w

        # Zero-pad
        us_padded = np.zeros(L, dtype=np.complex128)
        us_padded[:length] = us_windowed

        # FFT (keep complex!)
        U = np.fft.fft(us_padded)

        # Sum complex FFTs
        if U_sum is None:
            U_sum = np.zeros_like(U, dtype=np.complex128)
        U_sum += U

    # Average THEN square (coherent)
    U_mean = U_sum / M
    mag2_avg = np.abs(U_mean) ** 2

    return mag2_avg


def find_period_candidates(
    spectrum: np.ndarray,
    N: int,
    top_k: int = 10,
    min_snr_db: float = 10.0
) -> List[Tuple[int, float]]:
    """
    Extract period candidates from VRA spectrum.

    Identifies harmonic peaks in the spectrum and infers potential periods.
    Uses CFAR detection with validated threshold α=4.0.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum from compute_averaged_spectrum
    N : int
        Modulus (upper bound on period)
    top_k : int, optional
        Number of top candidates to return (default: 10)
    min_snr_db : float, optional
        Minimum SNR threshold in dB (default: 10.0)

    Returns
    -------
    List[Tuple[int, float]]
        List of (period, confidence) tuples, sorted by confidence

    Notes
    -----
    Uses CFAR (Constant False Alarm Rate) detection validated in VRA:
    - Optimal α=4.0 (99.9% precision, 100% recall)
    - Sub-bin accuracy: μ=0.28 bins
    - False alarm rate: P_FA ≤ 0.01
    """
    L = len(spectrum)

    # CFAR threshold (α=4.0 validated in VRA experiment G1)
    alpha = 4.0
    threshold = np.percentile(spectrum, 100 * (1 - 1/alpha))

    # Additional SNR filter
    noise_floor = np.median(spectrum)
    min_power = noise_floor * 10**(min_snr_db / 10)

    # Find peaks
    peak_indices = np.where(spectrum > max(threshold, min_power))[0]

    if len(peak_indices) == 0:
        return []

    # Extract peak powers
    peak_powers = spectrum[peak_indices]

    # Infer periods from harmonic spacing
    # If we see peaks at k, 2k, 3k, ... then period r = L/k
    candidates = {}

    for idx in peak_indices:
        if idx == 0:
            continue

        # Infer period from this peak
        period = L // idx

        if period > 1 and period < N:
            confidence = spectrum[idx] / noise_floor  # SNR as confidence

            if period in candidates:
                candidates[period] = max(candidates[period], confidence)
            else:
                candidates[period] = confidence

    # Sort by confidence
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_candidates[:top_k]


def compute_coherence(spectrum: np.ndarray) -> float:
    """
    Compute coherence metric from spectrum.

    Measures concentration of power, related to VRA's coherence law:
    C = exp(-V_φ/2)

    Threshold at e^-2 ≈ 0.1353 indicates coherence collapse.

    Parameters
    ----------
    spectrum : np.ndarray
        Power spectrum

    Returns
    -------
    float
        Coherence metric C ∈ [0, 1]
    """
    if spectrum.sum() == 0:
        return 0.0

    # Concentration ratio
    C = np.max(spectrum) / np.sum(spectrum)

    return float(C)
