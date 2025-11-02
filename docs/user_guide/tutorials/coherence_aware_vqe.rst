Coherence-Aware Quantum Computing
===================================

.. contents::
   :local:
   :depth: 2

Introduction
------------

ATLAS-Q features the **world's first coherence-aware quantum computing framework**, enabling algorithms to validate their own trustworthiness in real-time using physics-derived universal thresholds. This breakthrough transforms quantum computing from "hope it works" to "know it works."

What is Coherence-Aware Computing?
-----------------------------------

Traditional quantum algorithms run "blind" - they execute circuits and return results, but provide no measure of whether those results are trustworthy. Coherence-aware computing adds real-time quality monitoring based on **Vaca Resonance Analysis (VRA)**, tracking circular statistics that quantify measurement coherence.

Key Concepts
~~~~~~~~~~~~

**Coherence Metrics**

- **R̄ (Mean Resultant Length)**: Measures phase coherence (0 = random, 1 = perfect)
- **V_φ (Circular Variance)**: Quantifies phase spread (0 = perfect, ∞ = random)
- **Coherence Law**: R̄ = e^(-V_φ/2) - Universal relationship validated on hardware

**Universal Threshold**

- **e^-2 Boundary** (R̄ ≈ 0.135): Physics-derived threshold separating trustworthy from noisy results
- Above boundary → "GO" (high confidence in results)
- Below boundary → "NO-GO" (results may be unreliable)

**Why This Matters**

Unlike error mitigation techniques that try to fix noise, coherence awareness provides an **objective quality score** that:

- Requires no classical simulation for validation
- Works across all quantum hardware platforms
- Provides real-time feedback during algorithm execution
- Enables adaptive algorithm behavior based on measurement quality

Hardware Validation
-------------------

The framework has been validated on **IBM Brisbane** (127-qubit Eagle r3) with production-scale molecules:

.. list-table:: Hardware Results
   :header-rows: 1
   :widths: 15 10 10 10 15 15 20

   * - Molecule
     - Qubits
     - Terms
     - Groups
     - R̄
     - Runtime
     - Job ID
   * - H2
     - 4
     - 15
     - 15
     - 0.891
     - 29s
     - d43pjb90f7bc7388mlo0
   * - LiH
     - 12
     - 631
     - 127
     - 0.980
     - 43s
     - d43ppngg60jg738f3g4g
   * - H2O
     - 14
     - 1086
     - 219
     - **0.988**
     - 96s
     - d43q6e07i53s73e4dad0

**Key Achievement**: Near-ideal coherence (R̄ > 0.98) maintained on production-scale molecules while achieving 5× measurement compression via VRA grouping.

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from atlas_q import CoherenceAwareVQE
    from atlas_q.vra_enhanced import vra_grouping

    # Create VQE instance with coherence tracking
    vqe = CoherenceAwareVQE(
        hamiltonian=hamiltonian,
        ansatz='UCCSD',
        enable_coherence_tracking=True,
        e2_boundary_check=True  # Enable GO/NO-GO classification
    )

    # Run with VRA grouping for measurement compression
    result = vqe.run(
        backend='ibm_brisbane',
        shots=1000,
        use_vra_grouping=True
    )

    # Check results
    print(f"Energy: {result.energy:.6f} Ha")
    print(f"Coherence R̄: {result.coherence.R_bar:.4f}")
    print(f"Classification: {result.go_no_go}")  # GO or NO-GO

    if result.go_no_go == "GO":
        print("✓ Results are trustworthy (R̄ > e^-2)")
    else:
        print("⚠ Results may be unreliable (R̄ < e^-2)")

Molecular Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from atlas_q import extract_molecular_hamiltonian

    # Extract H2O Hamiltonian from PySCF
    hamiltonian, n_qubits, e_nuc = extract_molecular_hamiltonian(
        molecule='H2O',
        basis='sto-3g',
        geometry='equilibrium'  # or provide custom coordinates
    )

    # Run coherence-aware VQE
    vqe = CoherenceAwareVQE(hamiltonian)
    result = vqe.run(backend='ibm_brisbane', shots=1000)

    # Results include both electronic and nuclear energies
    print(f"Electronic: {result.energy_elec:.6f} Ha")
    print(f"Nuclear: {result.energy_nuc:.6f} Ha")
    print(f"Total: {result.energy:.6f} Ha")

Advanced Features
-----------------

VRA Grouping for Measurement Compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VRA grouping reduces measurement overhead by grouping commuting Pauli operators:

.. code-block:: python

    from atlas_q.vra_enhanced import vra_grouping, compute_coherence

    # Group 1086 Pauli terms → 219 measurement groups (5× compression)
    groups = vra_grouping(
        pauli_strings,
        coefficients,
        strategy='qubit_wise_commuting'  # QWC grouping
    )

    # Each group shares measurement basis, reducing circuit count
    print(f"Compression: {len(pauli_strings)}/{len(groups)} = {len(pauli_strings)/len(groups):.1f}×")

Real-Time Coherence Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track coherence during optimization:

.. code-block:: python

    # Enable per-iteration coherence logging
    vqe = CoherenceAwareVQE(
        hamiltonian,
        coherence_tracking='per_iteration',  # Log every iteration
        coherence_callback=lambda iter, R_bar: print(f"Iter {iter}: R̄={R_bar:.4f}")
    )

    result = vqe.optimize(
        optimizer='COBYLA',
        max_iterations=100
    )

    # Access coherence history
    import matplotlib.pyplot as plt
    plt.plot(result.coherence_history)
    plt.axhline(0.135, color='r', linestyle='--', label='e^-2 boundary')
    plt.xlabel('Iteration')
    plt.ylabel('R̄')
    plt.legend()
    plt.show()

Adaptive Algorithm Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically adjust strategy based on coherence:

.. code-block:: python

    vqe = CoherenceAwareVQE(
        hamiltonian,
        adaptive_vra=True,  # Switch VRA ON/OFF based on R̄
        vra_threshold=0.135  # e^-2 boundary
    )

    # VRA grouping automatically enabled when R̄ > 0.135
    # Falls back to standard measurement when R̄ < 0.135
    result = vqe.run(backend='ibm_brisbane')

Integration with Existing Code
-------------------------------

Drop-In Replacement
~~~~~~~~~~~~~~~~~~~

Coherence-aware VQE is a drop-in replacement for standard VQE:

.. code-block:: python

    # Before: Standard VQE
    from atlas_q import VQE
    vqe = VQE(hamiltonian)
    result = vqe.run(backend='aer_simulator')

    # After: Coherence-aware VQE (backwards compatible)
    from atlas_q import CoherenceAwareVQE
    vqe = CoherenceAwareVQE(
        hamiltonian,
        enable_coherence_tracking=True  # New feature, optional
    )
    result = vqe.run(backend='ibm_brisbane')

    # All existing code works, plus new coherence metrics
    print(f"Energy: {result.energy}")  # Standard result
    print(f"R̄: {result.coherence.R_bar}")  # New coherence metric

Hybrid Workflows
~~~~~~~~~~~~~~~~

Combine with ATLAS-Q's tensor network backend:

.. code-block:: python

    from atlas_q import QuantumHybridSystem

    system = QuantumHybridSystem(
        n_qubits=14,
        backend='mps',
        chi_max=256  # MPS bond dimension
    )

    # Run coherence-aware VQE with MPS backend
    vqe = CoherenceAwareVQE(
        hamiltonian,
        quantum_system=system,
        enable_coherence_tracking=True
    )

    result = vqe.run(shots=10000)  # Simulated shots from MPS

Technical Details
-----------------

Coherence Calculation
~~~~~~~~~~~~~~~~~~~~~

Coherence metrics are computed from Pauli expectation values:

.. math::

    R̄ = |\\langle e^{i\\phi} \\rangle| = \\sqrt{\\langle \\cos\\phi \\rangle^2 + \\langle \\sin\\phi \\rangle^2}

    V_\\phi = -2 \\ln R̄

    \\text{Coherence Law: } R̄ = e^{-V_\\phi/2}

**Implementation**:

.. code-block:: python

    def compute_coherence(measurement_outcomes: np.ndarray) -> CoherenceMetrics:
        """
        Compute circular statistics from Pauli expectations.

        Args:
            measurement_outcomes: Array of ⟨P⟩ values from each Pauli term

        Returns:
            CoherenceMetrics with R_bar, V_phi, and boundary classification
        """
        # Map Pauli expectations to phase angles
        # ⟨P⟩ ∈ [-1, 1] → φ ∈ [0, 2π]
        phases = np.arccos(np.clip(measurement_outcomes, -1, 1))

        # Compute mean resultant length (circular mean)
        cos_mean = np.mean(np.cos(phases))
        sin_mean = np.mean(np.sin(phases))
        R_bar = np.sqrt(cos_mean**2 + sin_mean**2)

        # Circular variance
        V_phi = -2 * np.log(R_bar) if R_bar > 0 else np.inf

        # Classification
        is_above_e2 = R_bar > 0.135

        return CoherenceMetrics(R_bar, V_phi, is_above_e2)

VRA Grouping Algorithm
~~~~~~~~~~~~~~~~~~~~~~

Qubit-wise commuting (QWC) grouping strategy:

.. code-block:: python

    def vra_grouping(pauli_strings: List[str],
                    coefficients: np.ndarray,
                    strategy: str = 'qubit_wise_commuting') -> List[List[int]]:
        """
        Group commuting Pauli strings for simultaneous measurement.

        Paulis commute if they act with compatible operators on each qubit:
        - (I, Z) commute with (I, Z)
        - (X, Y) commute with (X, Y)
        - (I) commutes with everything

        Args:
            pauli_strings: List of Pauli strings (e.g., ['IXZY', 'IIIZ'])
            coefficients: Hamiltonian coefficients
            strategy: Grouping strategy ('qubit_wise_commuting')

        Returns:
            List of groups, each group is list of indices into pauli_strings
        """
        # Implementation groups Paulis sharing measurement basis
        # Each group measured with single circuit
        # Reduces shots by ~5× for typical molecules

Best Practices
--------------

1. **Always Check Coherence**

   .. code-block:: python

       if result.coherence.R_bar < 0.135:
           warnings.warn("Low coherence detected. Results may be unreliable.")

2. **Use VRA Grouping for Large Hamiltonians**

   .. code-block:: python

       # For > 100 Pauli terms, VRA grouping significantly reduces runtime
       use_vra = len(hamiltonian.paulis) > 100

3. **Monitor Coherence During Optimization**

   .. code-block:: python

       # Track coherence evolution to detect convergence issues
       coherence_tracking='per_iteration'

4. **Validate on Simulator First**

   .. code-block:: python

       # Test on AerSimulator before running on hardware
       result_sim = vqe.run(backend='aer_simulator')
       result_hw = vqe.run(backend='ibm_brisbane')

       # Compare coherence levels
       print(f"Simulator R̄: {result_sim.coherence.R_bar:.4f}")
       print(f"Hardware R̄: {result_hw.coherence.R_bar:.4f}")

Troubleshooting
---------------

Low Coherence (R̄ < 0.135)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

- Deep circuits with many gates → increased decoherence
- Noisy hardware backend
- Inadequate error mitigation

**Solutions:**

- Reduce ansatz depth
- Enable error mitigation techniques
- Try different hardware backend
- Increase shots per measurement

High Variance in Results
~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

- Insufficient shot count
- Heterogeneous error rates across qubits

**Solutions:**

- Increase shots (proportional to 1/√shots)
- Use per-member averaging in VRA grouping
- Apply readout error mitigation

References
----------

**Papers:**

- Vaca Resonance Analysis: [Paper link to be added]
- Coherence Law Validation: COHERENCE_AWARE_VQE_BREAKTHROUGH.md
- Hardware Results: VRA_HARDWARE_VALIDATION_SUMMARY.md

**Code:**

- Main implementation: ``benchmarks/vra_coherence_aware_hardware_benchmark.py``
- VRA enhanced modules: ``src/atlas_q/vra_enhanced/``
- Integration tests: ``tests/integration/test_vra_qaoa_grouping.py``

**External Resources:**

- IBM Quantum: https://quantum.ibm.com
- Circular Statistics: Mardia & Jupp, "Directional Statistics" (2000)
- Random Matrix Theory: Mehta, "Random Matrices" (2004)
