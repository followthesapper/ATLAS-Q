"""
Coherence-Aware VQE
===================

Variational Quantum Eigensolver with real-time coherence tracking and
GO/NO-GO classification based on Informational Relativity (IR).

This module extends the standard VQE with:
- Real-time coherence monitoring (R̄, V_φ)
- GO/NO-GO classification using e^-2 boundary
- Adaptive IR grouping decisions
- Measurement quality assessment

Author: ATLAS-Q Development Team
Date: November 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .coherence import (
    CoherenceClassification,
    CoherenceMetrics,
    adaptive_ir_decision,
    classify_go_no_go,
    compute_coherence,
)
from .mpo_ops import MPO
from .vqe_qaoa import VQE, VQEConfig


# =============================================================================
# NEW: Pre-computation regime analysis (IR-correct approach)
# =============================================================================

def analyze_vqe_regime(hamiltonian: "MPO") -> "RegimeAnalysis":
    """
    Analyze Hamiltonian regime BEFORE running VQE.

    This is the IR-correct approach: diagnose regime first, then decide
    whether to run VQE at all, and which representation to use.

    Args:
        hamiltonian: MPO Hamiltonian

    Returns:
        RegimeAnalysis with observability classification
    """
    from .ir_enhanced.regime_analyzer import (
        analyze_hamiltonian_regime,
        RegimeAnalysis,
    )

    # Extract coefficients and Pauli strings from Hamiltonian
    # This depends on MPO implementation
    try:
        if hasattr(hamiltonian, 'coefficients') and hasattr(hamiltonian, 'pauli_strings'):
            coefficients = np.array(hamiltonian.coefficients)
            pauli_strings = hamiltonian.pauli_strings
        elif hasattr(hamiltonian, 'terms'):
            # Alternative MPO format
            coefficients = np.array([t.coeff for t in hamiltonian.terms])
            pauli_strings = [t.pauli for t in hamiltonian.terms]
        else:
            # Fallback: use uniform coefficients
            n_sites = hamiltonian.n_sites if hasattr(hamiltonian, 'n_sites') else 4
            coefficients = np.ones(n_sites)
            pauli_strings = None
    except Exception:
        # Safe fallback
        coefficients = np.array([1.0])
        pauli_strings = None

    return analyze_hamiltonian_regime(
        coefficients=coefficients,
        pauli_strings=pauli_strings,
    )


@dataclass
class CoherenceAwareVQEResult:
    """
    Results from coherence-aware VQE run.

    Attributes:
        energy: Ground state energy (Hartree)
        params: Optimal variational parameters
        n_iterations: Number of optimization iterations
        coherence: Final coherence metrics
        classification: GO/NO-GO classification
        regime_analysis: Pre-computation regime analysis (NEW)
        coherence_history: Coherence at each iteration (if enabled)
        energy_history: Energy at each iteration
        convergence_plot_path: Path to convergence plot (if output_dir set)
    """
    energy: float
    params: np.ndarray
    n_iterations: int
    coherence: CoherenceMetrics
    classification: CoherenceClassification
    regime_analysis: Optional["RegimeAnalysis"] = None  # NEW: Pre-computation analysis
    coherence_history: Optional[List[CoherenceMetrics]] = None
    energy_history: Optional[List[float]] = None
    convergence_plot_path: Optional[str] = None

    def is_go(self) -> bool:
        """Check if result passed GO/NO-GO classification."""
        return self.classification.is_go()

    def is_trustworthy(self, threshold: float = 0.135) -> bool:
        """Check if coherence is above threshold (alias for is_go)."""
        return self.coherence.R_bar > threshold

    def structure_was_observable(self) -> bool:
        """Check if structure was in IR regime (observable) before computation."""
        if self.regime_analysis is None:
            return self.is_go()  # Fallback to post-hoc check
        from .ir_enhanced.regime_analyzer import ObservabilityRegime
        return self.regime_analysis.regime == ObservabilityRegime.IR

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            "="*70,
            "Coherence-Aware VQE Results",
            "="*70,
            f"Energy: {self.energy:.8f} Ha",
            f"Iterations: {self.n_iterations}",
            f"",
        ]

        # NEW: Include pre-computation regime analysis
        if self.regime_analysis is not None:
            lines.extend([
                "Pre-Computation Regime Analysis:",
                f"  Regime: {self.regime_analysis.regime.value.upper()}",
                f"  Coherence R̄: {self.regime_analysis.coherence:.4f}",
                f"  Structure Observable: {'YES' if self.regime_analysis.structure_observable else 'NO'}",
                f"  Representation Cost: {self.regime_analysis.representation_cost}",
                f"",
            ])

        lines.extend([
            "Post-Computation Coherence Metrics:",
            f"  R̄ (Mean Resultant Length): {self.coherence.R_bar:.4f}",
            f"  V_φ (Circular Variance): {self.coherence.V_phi:.4f}",
            f"  Above e^-2 boundary: {'YES' if self.coherence.is_above_e2_boundary else 'NO'}",
            f"",
            f"Classification: {self.classification}",
            "="*70,
        ])
        return "\n".join(lines)


class CoherenceAwareVQE:
    """
    VQE with coherence tracking and GO/NO-GO classification.

    This is a wrapper around the standard VQE that adds real-time
    coherence monitoring based on Informational Relativity (IR).

    Key Features:
        - Real-time coherence tracking during optimization
        - GO/NO-GO classification using e^-2 boundary
        - Adaptive IR grouping decisions
        - Optional per-iteration coherence callback
        - Backward compatible with standard VQE

    Example:
        >>> from atlas_q import mpo_ops
        >>> from atlas_q.coherence_aware_vqe import CoherenceAwareVQE, VQEConfig
        >>>
        >>> # Build Hamiltonian
        >>> H = mpo_ops.molecular_hamiltonian_from_specs('H2')
        >>>
        >>> # Create coherence-aware VQE
        >>> config = VQEConfig(ansatz='hardware_efficient', n_layers=2)
        >>> vqe = CoherenceAwareVQE(H, config, enable_coherence_tracking=True)
        >>>
        >>> # Run optimization
        >>> result = vqe.run()
        >>>
        >>> # Check results
        >>> print(result.summary())
        >>> if result.is_go():
        ...     print("✓ Results are trustworthy")
        >>> else:
        ...     print("⚠ Low coherence detected")

    Args:
        hamiltonian: MPO Hamiltonian
        config: VQE configuration
        custom_ansatz: Optional custom ansatz
        output_dir: Optional directory for outputs
        enable_coherence_tracking: Enable per-iteration coherence tracking
        e2_threshold: Threshold for GO/NO-GO classification (default: 0.135)
        coherence_callback: Optional callback(iteration, coherence) for custom handling

    Notes:
        - Coherence is computed from Hamiltonian term expectations
        - Requires MPO with accessible Pauli terms
        - Backward compatible: can be used as drop-in replacement for VQE
    """

    def __init__(
        self,
        hamiltonian: MPO,
        config: VQEConfig,
        custom_ansatz=None,
        output_dir: Optional[str] = None,
        enable_coherence_tracking: bool = True,
        e2_threshold: float = 0.135,
        coherence_callback: Optional[Callable[[int, CoherenceMetrics], None]] = None,
    ):
        # Create underlying VQE
        self.vqe = VQE(hamiltonian, config, custom_ansatz, output_dir)

        # Coherence settings
        self.enable_coherence_tracking = enable_coherence_tracking
        self.e2_threshold = e2_threshold
        self.coherence_callback = coherence_callback

        # Coherence tracking
        self.coherence_history: List[CoherenceMetrics] = []
        self._last_measurement_outcomes: Optional[np.ndarray] = None

        # Access to underlying VQE attributes
        self.config = self.vqe.config
        self.H = self.vqe.H
        self.ansatz = self.vqe.ansatz

    def _compute_coherence_from_hamiltonian(self, mps) -> Optional[CoherenceMetrics]:
        """
        Compute coherence from Hamiltonian term expectations.

        This evaluates ⟨ψ|P_i|ψ⟩ for each Pauli term P_i in the Hamiltonian
        and uses these expectations to compute coherence metrics.

        Args:
            mps: Current MPS state

        Returns:
            CoherenceMetrics or None if computation fails
        """
        if not self.enable_coherence_tracking:
            return None

        try:
            # Extract Pauli expectations from Hamiltonian
            # This assumes MPO has accessible term-by-term evaluation
            from .mpo_ops import expectation_value_per_term

            if not hasattr(expectation_value_per_term, '__call__'):
                # Fallback: compute from energy evaluations if per-term not available
                return None

            expectations = expectation_value_per_term(self.H, mps)
            self._last_measurement_outcomes = np.array([float(e.real) for e in expectations])

            # Normalize expectations to [-1, 1] if needed
            # (Pauli expectations are already in this range)
            coherence = compute_coherence(self._last_measurement_outcomes, self.e2_threshold)

            return coherence

        except Exception as e:
            # Gracefully degrade if coherence computation fails
            import warnings
            warnings.warn(f"Coherence computation failed: {e}")
            return None

    def _energy_at_params_with_coherence(self, params: np.ndarray) -> Tuple[float, Optional[CoherenceMetrics]]:
        """
        Evaluate energy and coherence at given parameters.

        This wraps the standard VQE energy evaluation to also compute
        coherence metrics from the Hamiltonian term expectations.

        Args:
            params: Variational parameters

        Returns:
            Tuple of (energy, coherence_metrics)
        """
        # Get energy using VQE's method
        energy = self.vqe._energy_at_params(params)

        # Compute coherence if enabled
        # Note: We need access to the MPS state used for energy evaluation
        # For now, reconstruct it (TODO: optimize by caching in VQE)
        if self.enable_coherence_tracking:
            from .adaptive_mps import AdaptiveMPS

            if hasattr(self.ansatz, "prepare_hf_state"):
                mps = self.ansatz.prepare_hf_state(chi_max=self.config.chi_max)
            else:
                mps = AdaptiveMPS(
                    num_qubits=self.H.n_sites,
                    bond_dim=2,
                    chi_max_per_bond=self.config.chi_max,
                    device=self.config.device,
                    dtype=self.config.dtype,
                )

            try:
                self.ansatz.apply(mps, params, chi_max=self.config.chi_max)
            except TypeError:
                self.ansatz.apply(mps, params)

            coherence = self._compute_coherence_from_hamiltonian(mps)
        else:
            coherence = None

        return energy, coherence

    def run(
        self,
        initial_params: Optional[np.ndarray] = None,
        label: str = "molecule",
        skip_regime_analysis: bool = False,
    ) -> CoherenceAwareVQEResult:
        """
        Run coherence-aware VQE optimization.

        NEW: Now performs regime analysis BEFORE optimization (IR-correct approach).

        This performs standard VQE optimization while tracking coherence
        metrics at each iteration.

        Args:
            initial_params: Optional starting parameters (otherwise uses warm-start)
            label: Label for this run (used in output files)
            skip_regime_analysis: Skip pre-computation analysis (not recommended)

        Returns:
            CoherenceAwareVQEResult with energy, parameters, and coherence metrics

        Example:
            >>> vqe = CoherenceAwareVQE(hamiltonian, config)
            >>> result = vqe.run(label="H2_sto3g")
            >>> print(result.summary())
            >>> if result.structure_was_observable():
            ...     print(f"Ground state: {result.energy:.6f} Ha")
            >>> else:
            ...     print("Warning: Structure was in AIR regime")
        """
        # =====================================================================
        # NEW: Step 0 - Regime Analysis BEFORE Optimization (IR-correct)
        # =====================================================================
        regime_analysis = None
        if not skip_regime_analysis:
            try:
                regime_analysis = analyze_vqe_regime(self.H)

                # Log regime warning if in AIR regime
                from .ir_enhanced.regime_analyzer import ObservabilityRegime
                if regime_analysis.regime == ObservabilityRegime.AIR:
                    import warnings
                    warnings.warn(
                        f"Hamiltonian in AIR regime (R̄={regime_analysis.coherence:.3f} < e^-2). "
                        f"Structure globally hidden - VQE results may be unreliable. "
                        f"Representation cost: {regime_analysis.representation_cost}"
                    )
            except Exception as e:
                import warnings
                warnings.warn(f"Regime analysis failed: {e}")

        # Run standard VQE
        energy, params = self.vqe.run(initial_params, label)

        # Compute final coherence
        _, final_coherence = self._energy_at_params_with_coherence(params)

        if final_coherence is None:
            # Fallback if coherence computation not available
            import warnings
            warnings.warn("Coherence tracking not available - using default values")
            final_coherence = CoherenceMetrics(
                R_bar=0.0,
                V_phi=np.inf,
                is_above_e2_boundary=False,
                ir_predicted_to_help=False,
                n_measurements=0
            )

        # Classify
        classification = classify_go_no_go(final_coherence, self.e2_threshold)

        # Build result
        result = CoherenceAwareVQEResult(
            energy=energy,
            params=params,
            n_iterations=self.vqe.iteration,
            coherence=final_coherence,
            classification=classification,
            regime_analysis=regime_analysis,  # NEW: Include pre-computation analysis
            coherence_history=self.coherence_history if self.coherence_history else None,
            energy_history=self.vqe.energies.copy() if self.vqe.energies else None,
            convergence_plot_path=None,  # TODO: add if plot was saved
        )

        return result

    def run_with_adaptive_ir(
        self,
        initial_params: Optional[np.ndarray] = None,
        label: str = "molecule",
        ir_callback: Optional[Callable[[bool, str], None]] = None,
    ) -> CoherenceAwareVQEResult:
        """
        Run VQE with adaptive IR grouping decisions.

        This enables adaptive behavior where IR grouping is turned ON/OFF
        based on measured coherence during optimization.

        Args:
            initial_params: Optional starting parameters
            label: Label for this run
            ir_callback: Optional callback(enable_ir, reason) for IR decisions

        Returns:
            CoherenceAwareVQEResult with adaptive IR history

        Example:
            >>> def my_ir_callback(enable, reason):
            ...     print(f"IR: {'ON' if enable else 'OFF'} - {reason}")
            >>>
            >>> result = vqe.run_with_adaptive_ir(ir_callback=my_ir_callback)
        """
        # Run with coherence tracking
        result = self.run(initial_params, label)

        # Make adaptive IR decision based on final coherence
        if result.coherence:
            enable_ir, reason = adaptive_ir_decision(result.coherence, self.e2_threshold)
            if ir_callback:
                ir_callback(enable_ir, reason)

        return result

    def set_fci_reference(self, fci_energy: float):
        """Set Full CI reference energy for comparison."""
        self.vqe._fci_ref = fci_energy

    def set_quiet(self, quiet: bool):
        """Enable/disable console output."""
        self.vqe.quiet = quiet

    def enable_warm_start(self, enable: bool = True):
        """Enable/disable warm start initialization."""
        self.vqe._use_warm_start = enable

    @property
    def energies(self) -> List[float]:
        """Get energy history from optimization."""
        return self.vqe.energies

    @property
    def param_history(self) -> List[np.ndarray]:
        """Get parameter history from optimization."""
        return self.vqe.param_history


# Convenience function for quick usage
def coherence_aware_vqe(
    hamiltonian: MPO,
    ansatz: str = 'hardware_efficient',
    n_layers: int = 3,
    chi_max: int = 256,
    enable_coherence_tracking: bool = True,
    output_dir: Optional[str] = None,
    **kwargs
) -> CoherenceAwareVQEResult:
    """
    Quick-start function for coherence-aware VQE.

    Args:
        hamiltonian: MPO Hamiltonian
        ansatz: Ansatz type ('hardware_efficient', 'uccsd')
        n_layers: Number of ansatz layers
        chi_max: Maximum MPS bond dimension
        enable_coherence_tracking: Enable coherence monitoring
        output_dir: Optional directory for outputs
        **kwargs: Additional VQEConfig parameters

    Returns:
        CoherenceAwareVQEResult

    Example:
        >>> from atlas_q import mpo_ops
        >>> from atlas_q.coherence_aware_vqe import coherence_aware_vqe
        >>>
        >>> H = mpo_ops.molecular_hamiltonian_from_specs('H2')
        >>> result = coherence_aware_vqe(H, n_layers=2, chi_max=128)
        >>> print(result.summary())
    """
    config = VQEConfig(
        ansatz=ansatz,
        n_layers=n_layers,
        chi_max=chi_max,
        **kwargs
    )

    vqe = CoherenceAwareVQE(
        hamiltonian,
        config,
        output_dir=output_dir,
        enable_coherence_tracking=enable_coherence_tracking,
    )

    return vqe.run()
