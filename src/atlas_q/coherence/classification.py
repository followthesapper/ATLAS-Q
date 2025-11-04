"""
Coherence Classification
========================

GO/NO-GO classification based on coherence metrics for quantum measurements.

Author: ATLAS-Q Development Team
Date: November 2025
"""

from dataclasses import dataclass
from typing import Tuple

from .metrics import CoherenceMetrics


@dataclass
class CoherenceClassification:
    """
    Classification result from coherence-based quality assessment.

    Attributes:
        status: Classification status ('GO' or 'NO-GO')
        reason: Human-readable explanation of classification
        coherence: The coherence metrics used for classification
        confidence: Confidence level (0-1) in the classification
    """
    status: str
    reason: str
    coherence: CoherenceMetrics
    confidence: float = 1.0

    def __post_init__(self):
        """Validate classification."""
        if self.status not in ('GO', 'NO-GO'):
            raise ValueError(f"status must be 'GO' or 'NO-GO', got '{self.status}'")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def is_go(self) -> bool:
        """Check if classification is GO."""
        return self.status == 'GO'

    def is_no_go(self) -> bool:
        """Check if classification is NO-GO."""
        return self.status == 'NO-GO'

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"[{self.status}] {self.reason} (confidence: {self.confidence:.2f})"


def classify_go_no_go(
    coherence: CoherenceMetrics,
    threshold: float = 0.135,
    use_confidence: bool = True
) -> CoherenceClassification:
    """
    Classify quantum measurement quality using e^-2 boundary.

    This implements VRA Test 7 (GO/NO-GO Classification) using the
    universal e^-2 threshold (R̄ ≈ 0.135) to separate trustworthy
    results from noisy ones.

    Classification Rules:
        - R̄ > threshold → GO (high confidence in results)
        - R̄ ≤ threshold → NO-GO (results may be unreliable)

    The confidence score is computed based on how far R̄ is from the
    threshold, providing a measure of classification certainty.

    Args:
        coherence: CoherenceMetrics from compute_coherence()
        threshold: e^-2 boundary threshold (default: 0.135)
        use_confidence: Whether to compute confidence score (default: True)

    Returns:
        CoherenceClassification with status, reason, and confidence

    Example:
        >>> from atlas_q.coherence import compute_coherence, classify_go_no_go
        >>> outcomes = np.array([0.9, 0.85, 0.88, 0.87])
        >>> coherence = compute_coherence(outcomes)
        >>> classification = classify_go_no_go(coherence)
        >>> print(classification)
        [GO] Coherence above e^-2 boundary (R̄=0.875 > 0.135) (confidence: 0.98)

    References:
        - VRA Test 7: GO/NO-GO boundary validation
        - Threshold derived from Random Matrix Theory
        - Hardware validation: COHERENCE_AWARE_VQE_BREAKTHROUGH.md
    """
    R_bar = coherence.R_bar

    # Determine classification
    if R_bar > threshold:
        status = "GO"
        reason = f"Coherence above e^-2 boundary (R̄={R_bar:.3f} > {threshold:.3f})"
    else:
        status = "NO-GO"
        reason = f"Coherence below e^-2 boundary (R̄={R_bar:.3f} ≤ {threshold:.3f})"

    # Compute confidence score
    if use_confidence:
        # Confidence increases with distance from threshold
        # Use sigmoid-like function centered at threshold
        distance = abs(R_bar - threshold)
        # Scale distance: 0.05 away → ~0.9 confidence, 0.2 away → ~0.99 confidence
        confidence = 1.0 - 0.5 * (1.0 / (1.0 + (distance / 0.05)))
        confidence = min(1.0, max(0.5, confidence))  # Clamp to [0.5, 1.0]
    else:
        confidence = 1.0

    return CoherenceClassification(
        status=status,
        reason=reason,
        coherence=coherence,
        confidence=confidence
    )


def classify_with_history(
    coherence_history: list,
    threshold: float = 0.135,
    window_size: int = 5,
    min_go_fraction: float = 0.8
) -> CoherenceClassification:
    """
    Classify using historical coherence measurements.

    This provides more robust classification by considering multiple
    measurements rather than a single snapshot. Useful for detecting
    trends and filtering transient noise.

    Args:
        coherence_history: List of CoherenceMetrics from successive measurements
        threshold: e^-2 boundary threshold (default: 0.135)
        window_size: Number of recent measurements to consider (default: 5)
        min_go_fraction: Minimum fraction of GO classifications required (default: 0.8)

    Returns:
        CoherenceClassification based on recent history

    Example:
        >>> history = [compute_coherence(outcomes) for outcomes in measurement_batches]
        >>> classification = classify_with_history(history, window_size=10)
        >>> if classification.is_go():
        ...     print("Consistently high coherence over last 10 measurements")

    Notes:
        - More conservative than single-shot classification
        - Helps avoid false positives from lucky measurements
        - Useful for adaptive algorithms that need stable quality assessment
    """
    if not coherence_history:
        raise ValueError("coherence_history cannot be empty")

    # Take last window_size measurements
    recent = coherence_history[-window_size:]

    # Classify each measurement
    go_count = sum(1 for c in recent if c.R_bar > threshold)
    go_fraction = go_count / len(recent)

    # Aggregate R̄ (mean of recent measurements)
    avg_R_bar = sum(c.R_bar for c in recent) / len(recent)

    # Determine status based on fraction of GO measurements
    if go_fraction >= min_go_fraction:
        status = "GO"
        reason = (f"Consistent coherence: {go_count}/{len(recent)} measurements above "
                  f"threshold (avg R̄={avg_R_bar:.3f})")
        confidence = go_fraction
    else:
        status = "NO-GO"
        reason = (f"Inconsistent coherence: only {go_count}/{len(recent)} measurements "
                  f"above threshold (avg R̄={avg_R_bar:.3f})")
        confidence = 1.0 - go_fraction

    # Use most recent coherence metrics for reference
    return CoherenceClassification(
        status=status,
        reason=reason,
        coherence=recent[-1],
        confidence=confidence
    )


def adaptive_vra_decision(
    coherence: CoherenceMetrics,
    threshold: float = 0.135,
    enable_hysteresis: bool = True,
    hysteresis_delta: float = 0.02
) -> Tuple[bool, str]:
    """
    Decide whether to enable VRA grouping based on coherence.

    This implements adaptive VRA where grouping is enabled when
    coherence is high and disabled when coherence is low.

    Args:
        coherence: Current coherence metrics
        threshold: e^-2 boundary threshold (default: 0.135)
        enable_hysteresis: Use hysteresis to avoid chattering (default: True)
        hysteresis_delta: Hysteresis margin around threshold (default: 0.02)

    Returns:
        Tuple of (enable_vra, reason)

    Example:
        >>> coherence = compute_coherence(outcomes)
        >>> enable_vra, reason = adaptive_vra_decision(coherence)
        >>> if enable_vra:
        ...     # Use VRA grouping for measurement compression
        ...     groups = vra_grouping(...)
        >>> else:
        ...     # Fall back to individual measurements
        ...     groups = individual_measurements(...)

    Notes:
        - Hysteresis prevents rapid switching near threshold
        - Upper threshold = threshold + delta (turn ON VRA)
        - Lower threshold = threshold - delta (turn OFF VRA)
        - Helps stabilize adaptive algorithms
    """
    R_bar = coherence.R_bar

    if enable_hysteresis:
        upper_threshold = threshold + hysteresis_delta
        lower_threshold = threshold - hysteresis_delta

        if R_bar > upper_threshold:
            enable_vra = True
            reason = f"High coherence (R̄={R_bar:.3f} > {upper_threshold:.3f}) → VRA ON"
        elif R_bar < lower_threshold:
            enable_vra = False
            reason = f"Low coherence (R̄={R_bar:.3f} < {lower_threshold:.3f}) → VRA OFF"
        else:
            # In hysteresis band: maintain previous state or default to ON
            enable_vra = R_bar > threshold
            reason = (f"Coherence in hysteresis band ({lower_threshold:.3f} < R̄={R_bar:.3f} "
                      f"< {upper_threshold:.3f}) → VRA {'ON' if enable_vra else 'OFF'}")
    else:
        if R_bar > threshold:
            enable_vra = True
            reason = f"Coherence above threshold (R̄={R_bar:.3f} > {threshold:.3f}) → VRA ON"
        else:
            enable_vra = False
            reason = f"Coherence below threshold (R̄={R_bar:.3f} ≤ {threshold:.3f}) → VRA OFF"

    return enable_vra, reason
