IR Module (Informational Relativity)
====================================

.. module:: atlas_q.ir_enhanced

The IR module provides Informational Relativity integration for quantum simulation,
enabling pre-computation regime diagnosis and coherence-based optimization.

**Version:** 1.1.0

Overview
--------

The IR module implements the key insight from Informational Relativity: **regime determines representation,
not the other way around**. Before running quantum algorithms, analyze the problem structure to:

1. Determine if the structure is observable (IR regime) or hidden (AIR regime)
2. Decide if grouping/optimization will help
3. Choose appropriate representation and algorithm parameters

Key Concepts
------------

Observability Regimes
^^^^^^^^^^^^^^^^^^^^^

The module classifies problems into three regimes based on coherence R̄:

- **IR (Informational Relativity)**: R̄ > e^-2 ≈ 0.135 - Structure is observable
- **TRANSITION**: R̄ ≈ e^-2 - Near the GO/NO-GO boundary
- **AIR (Anti-IR)**: R̄ < e^-2 - Structure is globally hidden

GO/NO-GO Classification
^^^^^^^^^^^^^^^^^^^^^^^

The physics-derived e^-2 threshold (≈0.135) provides a universal boundary:

- **GO**: Coherence above threshold → Quantum methods likely effective
- **NO-GO**: Coherence below threshold → Classical methods may be better

Regime Analyzer
---------------

Pre-computation diagnosis for quantum problems.

.. autofunction:: atlas_q.ir_enhanced.analyze_state_regime

.. autofunction:: atlas_q.ir_enhanced.analyze_hamiltonian_regime

.. autofunction:: atlas_q.ir_enhanced.analyze_mps_bond_regime

.. autofunction:: atlas_q.ir_enhanced.predict_quantum_advantage

.. autofunction:: atlas_q.ir_enhanced.should_use_ir_grouping

Classes
^^^^^^^

.. autoclass:: atlas_q.ir_enhanced.regime_analyzer.ObservabilityRegime
   :members:

.. autoclass:: atlas_q.ir_enhanced.regime_analyzer.RegimeAnalysis
   :members:

.. autoclass:: atlas_q.ir_enhanced.regime_analyzer.RepresentationType
   :members:

Spectral Lifting
----------------

Full M_ij relational matrix analysis for structure detection.

.. autofunction:: atlas_q.ir_enhanced.compute_relational_matrix

.. autofunction:: atlas_q.ir_enhanced.compute_relational_matrix_from_amplitudes

.. autofunction:: atlas_q.ir_enhanced.extract_structure_modes

.. autofunction:: atlas_q.ir_enhanced.spectral_grouping

.. autofunction:: atlas_q.ir_enhanced.spectral_lifting_analysis

.. autofunction:: atlas_q.ir_enhanced.spectral_lifting_from_amplitudes

.. autofunction:: atlas_q.ir_enhanced.coherent_structure_score

.. autoclass:: atlas_q.ir_enhanced.SpectralLiftingResult
   :members:

VQE Grouping
------------

Coherence-based Pauli term grouping for VQE (4× circuit reduction).

.. autofunction:: atlas_q.ir_enhanced.ir_hamiltonian_grouping

.. autofunction:: atlas_q.ir_enhanced.estimate_pauli_coherence_matrix

.. autofunction:: atlas_q.ir_enhanced.group_by_variance_minimization

.. autofunction:: atlas_q.ir_enhanced.allocate_shots_neyman

.. autofunction:: atlas_q.ir_enhanced.compute_Q_GLS

.. autofunction:: atlas_q.ir_enhanced.pauli_commutes

.. autofunction:: atlas_q.ir_enhanced.check_group_commutativity

.. autoclass:: atlas_q.ir_enhanced.GroupingResult
   :members:

Period Finding Enhancement
--------------------------

IR preprocessing for quantum period estimation (42% shot reduction).

.. autofunction:: atlas_q.ir_enhanced.ir_enhanced_period_finding

.. autofunction:: atlas_q.ir_enhanced.ir_preprocess_period

.. autofunction:: atlas_q.ir_enhanced.estimate_shot_reduction

Core Functions
^^^^^^^^^^^^^^

.. autofunction:: atlas_q.ir_enhanced.multiplicative_order

.. autofunction:: atlas_q.ir_enhanced.compute_averaged_spectrum

.. autofunction:: atlas_q.ir_enhanced.find_period_candidates

QAOA Grouping
-------------

Edge-based cost operator grouping for QAOA.

.. autofunction:: atlas_q.ir_enhanced.ir_qaoa_grouping

.. autofunction:: atlas_q.ir_enhanced.edges_commute

.. autofunction:: atlas_q.ir_enhanced.check_group_commutativity_edges

.. autofunction:: atlas_q.ir_enhanced.estimate_edge_coherence_matrix

.. autofunction:: atlas_q.ir_enhanced.group_edges_by_commutativity

.. autoclass:: atlas_q.ir_enhanced.QAOAGroupingResult
   :members:

Gradient Grouping
-----------------

Parameter shift optimization with coherence.

.. autofunction:: atlas_q.ir_enhanced.ir_gradient_grouping

.. autofunction:: atlas_q.ir_enhanced.parameter_shift_gradient_ir

.. autofunction:: atlas_q.ir_enhanced.estimate_gradient_coherence_matrix

.. autofunction:: atlas_q.ir_enhanced.group_parameters_by_variance

.. autoclass:: atlas_q.ir_enhanced.GradientGroupingResult
   :members:

TDVP Observables
----------------

Real-time coherence tracking for time evolution.

.. autofunction:: atlas_q.ir_enhanced.ir_tdvp_observable_grouping

.. autoclass:: atlas_q.ir_enhanced.TDVPObservableGroupingResult
   :members:

Shadow Tomography
-----------------

Adaptive classical shadows with IR.

.. autofunction:: atlas_q.ir_enhanced.ir_shadow_sampling

.. autoclass:: atlas_q.ir_enhanced.ShadowSamplingResult
   :members:

State Tomography
----------------

IR-enhanced state reconstruction.

.. autofunction:: atlas_q.ir_enhanced.ir_state_tomography

.. autofunction:: atlas_q.ir_enhanced.generate_pauli_basis

.. autofunction:: atlas_q.ir_enhanced.tomography_measurement_groups

.. autoclass:: atlas_q.ir_enhanced.TomographyStrategy
   :members:

Usage Examples
--------------

Pre-Computation Regime Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.ir_enhanced import (
       analyze_state_regime,
       predict_quantum_advantage,
       ObservabilityRegime,
   )
   import numpy as np

   # Analyze state amplitudes
   amplitudes = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
   phases = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

   regime = analyze_state_regime(amplitudes, phases)

   print(f"Regime: {regime.regime.value}")  # 'ir', 'transition', or 'air'
   print(f"Coherence R̄: {regime.coherence:.4f}")
   print(f"Structure Observable: {regime.structure_observable}")

   if regime.regime == ObservabilityRegime.IR:
       print("GO: Quantum methods will be effective")
   elif regime.regime == ObservabilityRegime.AIR:
       print("NO-GO: Consider classical methods")

VQE with IR Grouping
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.ir_enhanced import ir_hamiltonian_grouping
   import numpy as np

   # Hamiltonian coefficients and Pauli strings
   coefficients = np.array([1.5, -0.8, 0.3, -0.2, 0.1, -0.05])
   pauli_strings = ["ZZII", "IZZI", "IIZZ", "XXII", "IXXI", "IIXX"]

   # Get optimal grouping
   result = ir_hamiltonian_grouping(
       coefficients=coefficients,
       pauli_strings=pauli_strings,
       total_shots=10000,
       max_group_size=3,
   )

   print(f"Number of groups: {len(result.groups)}")
   print(f"Variance reduction: {result.variance_reduction:.1f}×")
   print(f"Method: {result.method}")  # Includes regime info

Period Finding with IR
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from atlas_q.ir_enhanced import (
       ir_enhanced_period_finding,
       estimate_shot_reduction,
   )

   # Estimate shot reduction
   reduction = estimate_shot_reduction(N=15, coherence=0.8, num_candidates=3)
   print(f"Expected shot reduction: {(1-reduction)*100:.0f}%")

   # Run IR-enhanced period finding
   result = ir_enhanced_period_finding(a=7, N=15)
   print(f"Period: {result.period}")

Performance Numbers
-------------------

+---------------------+------------------------+
| Feature             | Improvement            |
+=====================+========================+
| VQE Grouping        | 4× circuit reduction   |
+---------------------+------------------------+
| Period Finding      | 42% shot reduction     |
+---------------------+------------------------+
| Variance Reduction  | 2-60× (VQE)            |
+---------------------+------------------------+
| MPS Scalability     | 100 qubits in 1.56s    |
+---------------------+------------------------+
| Memory Compression  | 10^25× (100 qubits)    |
+---------------------+------------------------+

See Also
--------

- :doc:`../user_guide/coherence_aware_vqe` - Coherence-aware VQE tutorial
- :doc:`vqe_qaoa` - VQE/QAOA reference
- :doc:`truncation` - Regime-aware truncation
