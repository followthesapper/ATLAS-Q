<style>
 figure.half img { width: 50%; height: auto; }
</style>


# ATLAS-Q Engineering Guide (System Design, Not Code)

*A technical blueprint for how ATLAS-Q turns theory (stabilizers, tensor networks, error bounds) into a robust, GPU-accelerated system. This document is intentionally **code-free** and focuses on **architecture, components, dataflows, decision logic, and operational qualities**.*

---

## 1. What We're Building (in Systems Terms)

**ATLAS-Q** is a *workload-aware quantum simulation platform* that **routes work** to the most efficient computational model:

* **Stabilizer backend** for Clifford-only subcircuits (O(n²) updates)
* **MPS/Tensor-network backend** for general circuits with *bounded entanglement* (O(n·χ²) state, adaptive truncation)
* **Hybrid decision engine** that *switches* between these models at runtime based on circuit content and entanglement signals
* **GPU acceleration** (Triton + cuBLAS) to remove bandwidth/launch overheads in the hottest paths

**Engineering outcome:** A system that stays fast by *exploiting structure*, remains *stable* via guarded numerics, and stays *predictable* through budgeting and telemetry.

---

## 2. Design Principles (Guiding Constraints → System Behaviors)

1. **Exploit structure first**: Route to stabilizer or MPS based on circuit semantics; never brute-force a 2^n state unless unavoidable
2. **Locality over generality**: Target 1D/2D layouts and nearest-neighbor mappings to keep memory and comms bounded
3. **Bounded inaccuracy, bounded memory**: Every approximation is budgeted and explainable (tolerances, χ caps, global error bound)
4. **GPU-first data motion**: Fuse operations and keep tensors hot in device memory; avoid host↔device churn
5. **Fail predictably**: If accuracy or memory budgets cannot be met, refuse early with actionable diagnostics—not silent slowdowns

---

## 3. Theory → System Mapping (Conceptual Translation)

| Theory Concept | System Artifact | Operational Guarantee |
|----------------|-----------------|----------------------|
| Gottesman-Knill (Clifford circuits) | **Stabilizer Tableau Engine** | O(n²) updates, deterministic, large-n possible |
| Matrix Product States (MPS) | **Adaptive MPS Engine** | O(n·χ²) memory; χ grows only when entanglement demands |
| Truncation errors | **Truncation Controller** | Per-bond tolerance → global error bound tracked & reported |
| Two-qubit gate locality | **Fused Gate Path** | Minimized DRAM traffic; single pass merge→apply→reshape |
| Hamiltonians as MPOs | **MPO Layer** | Contracted observables and TDVP with planned χ growth |
| Noise via Kraus maps | **Noise Application Hooks** | Stochastic channels with reproducible seeds |

---

## 4. System Architecture (Macro View)

```

 Python API Layer
 • Public classes, configs, lazy import, error surfaces



 Backend Switch (Hybrid)
 • Circuit inspection (Clifford/non-Clifford)
 • Entanglement & χ telemetry
 • Handoff policies, budgets, error bounds



 Stabilizer Core MPS/TN Core (PEPS v0.6.0)
 (tableau, O(n²)) (adaptive χ, SVD
 ctrl, MPO/TDVP)



 GPU Acceleration Subsystem
 • Kernel fusion (Triton) • cuBLAS contractions
 • Memory pools & streams • Precision policy

```

**Figure 1.** System architecture overview.
**Key property:** The **Hybrid** layer mediates *all* handoffs so that numerical state, error guarantees, and performance invariants remain coherent across models.

### Figure 2: MPS Data Structure Representation

```
Classical State Vector (2^n complex numbers):

 c000 c001 c010 c011 c100 c101 c110 c111 2^3 = 8 amplitudes


MPS Representation (chain of tensors):

Qubit 1 Qubit 2 Qubit 3
 χ χ
 A¹ A² A³


 i₁ i₂ i₃ (physical indices 0/1)

Each site k has two matrices:
A₀^k: (χₖ₋₁ × χₖ) when physical index = 0
A₁^k: (χₖ₋₁ × χₖ) when physical index = 1

Parameters: ~2 n χ² (vs 2^n amplitudes)

Example (n=50, χ=32): ~102,400 params → ~0.78 MB (complex64) or ~1.56 MB (complex128)
```

**Figure 2.** MPS structure vs full statevector. *Operational takeaway:* MPS memory is **linear in n** for fixed χ; only χ growth is dangerous.

### Figure 3: Stabilizer Tableau Representation

```
Standard State Vector for 3-qubit GHZ:
|ψ = 1/√2(|000 + |111)
→ Needs 2^3 = 8 complex amplitudes

Stabilizer Tableau (compact representation):


 X operators Z operators Phase

 1 0 0 0 0 0 +1 → X₁X₂
 0 1 0 0 0 0 +1 → X₂X₃
 0 0 0 1 1 1 +1 → Z₁Z₂Z₃

```

**Figure 3.** Stabilizer tableau. *Storage note:* A standard tableau is a \((2n)\times(2n+1)\) binary matrix (~\(4n^2+2n\) bits) plus phases. Example \(n=1000\) → ~0.5 MB for the tableau itself; add implementation overhead as needed.

---

## 5. Primary Dataflows (Figures)

### Figure 4: Two-Qubit Gate on MPS (System Sequence)

```
 API Layer Backend Switch MPS Core GPU Subsystem

 apply_gate(i,i+1)


 request resources


 fuse & apply



 matrix Θ ready


 SVD + truncate
 (ε_bond, χ_max)


 cores + ε_local


 accumulate
 ε_global


 stats (χ, ε, MB)

```

### Figure 5: Hybrid Handoff (Clifford → MPS)

```

 Clifford Gates
 H, S, CNOT




 Non-Clifford?
 (e.g., T, Toffoli,
 arbitrary-angle R)




 NO YES



 Stay in Materialize
 Stabilizer Tableau → MPS




 Canonicalize
 & Normalize




 Continue in MPS
 (adaptive χ)

```

### Figure 6: TDVP Sweep (Operational)

```

 START



 L→R Sweep:
 Build Left Envs



 Update Sites via
 Effective H



 R→L Sweep:
 Build Right Envs



 Update &
 Recanonicalize



 Energy
 Drift OK?


 YES NO


 DONE Adjust:
 • dt
 • χ
 • iterations




 (loop back)
```

---

## 6. Component Responsibilities

### 6.1 Backend Switch (Hybrid)
* **Classifies** gate sequences (Clifford/not) and **routes** operations
* **Tracks** entanglement proxies (χ growth, truncation frequency) and **enforces budgets** (χ caps, global MB cap, accuracy targets)
* **Guarantees** that a handoff preserves canonical form and numerical invariants

### 6.2 Stabilizer Core
* **Tableau engine** with bit-packed matrices; O(n²) updates
* **Ideal for** long Clifford prefixes, error-correction code studies, and large-n Clifford-heavy workloads

### 6.3 MPS/Tensor Core
* **Adaptive χ controller** reacts to entanglement; **truncation** bounded by per-bond tolerance; **global error** is accumulated and surfaced
* **MPO** layer for observables; **TDVP** for time evolution; **VQE/QAOA** orchestration for variational loops

### 6.4 GPU Acceleration Subsystem
* **Kernel fusion** to reduce global memory traffic and kernel launch overhead
* **cuBLAS-aligned contractions** (tensor cores) and **precision policy** (TF32, complex64→128 promotion when needed)
* **Memory pools/streams** tuned for reuse and predictable latency

---

## 7. Control Policies (How We Keep It Predictable)

### 7.1 Accuracy Policy
* **Per-bond tolerance** (\(\varepsilon_\text{bond}\)) selects SVD keep-rank → **provable global bound** \(\varepsilon_\text{global} = \sqrt{\sum \varepsilon_\text{local}^2}\)
* **Energy/observable monitors** can trigger *adaptive tightening* when drift is detected (TDVP/VQE loops)

### 7.2 Memory Policy
* **Per-bond χ cap** and **global memory budget** enforced at the Hybrid layer
* Requests that would exceed the budget are **refused early** with remediation hints (e.g., lower χ or raise \(\varepsilon_\text{bond}\))

### 7.3 Routing Policy
* Prefer **Stabilizer** while the circuit is strictly Clifford
* **Switch once** on first non-Clifford (conservative)
* Optional return is disabled by default to avoid thrash

---

## 8. Reliability & Numerical Guardrails
* **SVD cascade** with stability boosters (jitter injection, CPU fallback) → prevents hard failures on ill-conditioned splits
* **Canonicalization** (left/right/mixed) after updates → well-conditioned cores and stable successive operations
* **Precision promotion** on demand → guard against catastrophic cancellation
* **Deterministic modes** for tests/benchmarks (fixed RNG, fixed contraction order) → reproducible artifacts

---

## 9. Performance Model (Where the Speed Comes From)

1. **Work avoidance** via routing (Clifford → stabilizer: O(n²) vs O(2^n))
2. **Representation efficiency** (MPS: O(n·χ²) memory; χ grows only where needed)
3. **Data movement minimization** (fused hot path: merge→apply→reshape in one device pass)
4. **Hardware utilization** (tensor cores on contractions; tuned tile sizes; contiguous layouts)

### Figure 7: Latency Budget (Conceptual)

```
Two-Qubit Gate Operation Timeline (χ = 64)


0ms 10ms 20ms 25ms


 Contractions SVD Overheads
 (35%) (40%) (25%)

 Merge tensors Decomposition Memory
 Apply gate Truncation Transfers
 Reshape Recanonical. Bookkeeping


With GPU Fusion:

 Fused (25%) SVD (55%) (20%) ← Overhead reduced

```

<figure class="half">
 <img src="./assets/atlasq_figure5_memory_scaling.png" alt="Memory scaling: statevector (exponential) vs MPS (linear)">
 <figcaption><strong>Figure 8.</strong> Memory usage vs qubits. Statevector grows exponentially; MPS is linear in n for fixed χ. <em>Operational takeaway:</em> only χ growth is dangerous.</figcaption>
</figure>

<figure class="half">
 <img src="./assets/atlasq_figure6_chi_growth.png" alt="Bond dimension χ evolution over gate index with χmax reference">
 <figcaption><strong>Figure 9.</strong> Bond dimension χ growth after the first non-Clifford region; saturation under χ<sub>max</sub>. <em>Operational takeaway:</em> watch for χ spikes post-handoff; consider SWAP/layout changes or χ caps.</figcaption>
</figure>

### Figure 10: Hybrid Execution Profile

```
Execution Mode Over Time

Stabilizer
 ↑
 First T-gate

MPS

Timeline: |----Clifford----|-------Non-Clifford-------|
 0 500 gates 1500 gates

Performance Profile:

 Phase 1: Fast Phase 2: Adaptive
 • O(n²) ops • O(nχ³) ops
 • Fixed memory • Growing χ
 • No truncation • Active truncation

```

---

## 10. Error Accumulation & Truncation Strategy

### Figure 11(a): Local truncation rule of thumb
```
Singular Values at Bond (sorted) Truncation Decision

σ₁ 0.89 Keep
σ₂ 0.31 Keep } Cumulative: 99.7%
σ₃ 0.15 Keep } exceeds (1-ε²)
σ₄ 0.08 Keep
 Truncation threshold
σ₅ 0.03 Drop
σ₆ 0.01 Drop } Local error:
σ₇ 0.005 Drop } ε_local² = Σ (dropped σᵢ)²
```

<figure class="half">
 <img src="./assets/atlasq_figure10_error_evolution.png" alt="εglobal vs number of truncations with tolerance line (log scale)">
 <figcaption><strong>Figure 11(b).</strong> Tracked global error bound (log scale) vs number of truncations with target tolerance. <em>Operational takeaway:</em> tighten \(\varepsilon_\text{bond}\) or cap χ if the tolerance line is crossed.</figcaption>
</figure>

**Global bound:** \(\varepsilon_\text{global} \le \sqrt{\sum \varepsilon_{\text{local}}^2}\)

---

## 11. Observability & Telemetry (Explain Every Tradeoff)

### Metrics
* **Performance**: ops/sec, gate throughput, GPU utilization
* **Accuracy**: truncation counts, \(\varepsilon_\text{global}\) evolution, drift indicators
* **Resources**: χ histogram, memory bytes, precision promotions

### Events
* `mode_switch`: Stabilizer ↔ MPS transitions
* `svd_fallback`: Numerical stability triggers
* `oom_warning`: Memory pressure indicators
* `truncation_applied`: Per-bond truncation events
* `noise_applied`: Stochastic channel applications

### Traces
* Per-iteration records enabling flame-chart correlation with χ growth and error bound
* Gate-by-gate telemetry for bottleneck identification

### User Surfaces
* Single **Stats Summary** object after runs with actionable deltas
* Example: *"χ grew on bonds 12–14; suggest higher ε_bond or lower depth"*

### Figure 12: Real-Time Telemetry Dashboard (Conceptual)

```

 ATLAS-Q Runtime Monitor


 Circuit: VQE_H2O Gates: 1247/2000 Time: 1.3s

 Mode: [Stabilizer]500[MPS]747
 ↑
 T-gate @500

 χ Distribution: Memory Usage: GPU Utilization:



 8 16 32 64 128 23MB / 100MB 87%

 Bonds with χ>32: [12,13,14,27,28]

 Error Budget:
 Local: 8.2e-7
 Global: 6.1e-6 / 1e-5

 Recent Events:
 [1.23s] truncation_applied bond=14 dropped=3.1e-7
 [1.21s] chi_growth bond=14 old=32 new=64
 [0.50s] mode_switch stabilizer→mps

 Recommendations:
 • Consider SWAP at qubits (12,15) to reduce χ
 • ε_bond could be relaxed to 1e-5 for 2× speedup


```

---

## 12. Operational Scenarios (Runbooks without Code)

### Scenario A: Circuit with Long Clifford Prefix
**Expectation:** Stabilizer residency for most of the depth; handoff only at first T-gate.
**Operational tip:** If χ spikes immediately after handoff, consider SWAP synthesis to keep entanglement local.

### Scenario B: TDVP Time Evolution
**Expectation:** Gradual χ growth; memory policy must allow growth while keeping \(\varepsilon_\text{global}\) within target.
**Operational tip:** When drift triggers, tighten \(\varepsilon_\text{bond}\) or reduce dt; observe χ trend before and after adjustment.

### Scenario C: Noisy NISQ Simulation
**Expectation:** Noise hooks may force MPS mode; more frequent truncations.
**Operational tip:** Budget headroom for extra χ; prioritize fused path since two-qubit gates dominate.

---

## 13. Risks & Mitigations (Engineering View)

| Risk | Impact | Mitigation |
|------|--------|------------|
| χ explosion on dense entanglement | OOM / slowdowns | χ caps, early refusal, route redesign (layout/ordering) |
| SVD non-convergence | Run failure | SVD cascade + precision promotion + canonicalization |
| Thrashing at hybrid boundary | Performance variance | One-way handoff default; hysteresis if return enabled |
| DRAM bandwidth bottleneck | Low GPU efficiency | Kernel fusion, contiguous layouts, tuned tiles |
| Silent accuracy loss | Misleading results | Global error accounting + drift monitors + user-visible bounds |

---

## 14. What Makes This "Engineering" (not just Math)

* We **choose** the representation *dynamically* based on live signals (gates, χ, errors)
* We **shape data** for hardware (contiguity, fused passes) rather than perfect algebraic symmetry
* We **commit to contracts** (accuracy, memory, determinism) and build **policy engines** to uphold them
* We **instrument** every tradeoff so users can make informed decisions

---

## 15. Roadmap (System-Level)

### Near Term
* **Multi-GPU**: Shard MPS bonds across devices; overlap comms with contractions; NCCL collectives for SVD factors
* **Adapters**: Import/export from Qiskit/Cirq with layout heuristics and SWAP synthesis policies

### Medium Term
* **PEPS enhancements**: Advanced tiled 2D contractions with adaptive planner (base PEPS implemented in v0.6.0)
* **Adaptive solvers**: Randomized/Nyström SVD for very large χ regimes (policy-driven)

### Long Term
* **Multi-node distributed simulation**: Network-aware bond partitioning across nodes (single-node multi-GPU implemented in v0.6.0)
* **Hardware-specific optimizations**: Custom kernels for specific GPU architectures
* **Automatic circuit optimization**: ML-driven layout and gate ordering

---

## Appendix A: Symbols & Signals (for Operators)

| Symbol/Term | Definition | Operational Significance |
|-------------|------------|--------------------------|
| **n** | Number of qubits | Determines tableau size (n²) or MPS length |
| **χ (chi)** | Bond dimension | Entanglement capacity; larger χ = richer correlations, more memory/compute |
| **ε_bond** | Per-bond truncation tolerance | Smaller = higher accuracy, slower |
| **ε_global** | Total tracked error bound | \(\sqrt{\sum \varepsilon_{\text{local}}^2}\); displayed after runs |
| **MPO** | Matrix product operator | Encodes Hamiltonians/observables as local tensors |
| **TDVP** | Time-dependent variational principle | Time evolution algorithm over MPS; often allows χ growth |
| **Clifford** | H, S, CNOT, etc. | Gates that preserve stabilizer states; enable O(n²) simulation |
| **non-Clifford** | T, Toffoli, arbitrary-angle rotations | Often force MPS mode |

---

## Appendix B: Performance Benchmarks (Reference)

*Benchmarks on NVIDIA A100, complex64 precision*

| Circuit Type | Qubits | Gates | Bond Dim | Memory | Time | Backend |
|--------------|--------|-------|----------|--------|------|---------|
| QFT | 100 | 5,000 | χ=32 | 12 MB | 0.8s | MPS |
| Random Clifford | 1,000 | 10,000 | N/A | 0.5–2 MB (tableau) | 0.3s | Stabilizer |
| QAOA (p=4) | 50 | 400 | χ=64 | 25 MB | 1.2s | MPS |
| Shor's (15) | 18 | 8,000 | χ=16 | 1 MB | 0.5s | Hybrid |
| VQE (H₂O) | 14 | 2,000 | χ=128 | 8 MB | 2.1s | MPS |

---

## Appendix C: Debugging Checklist

### When Simulation is Slow
1. Check χ growth pattern (sudden spikes indicate non-local entanglement)
2. Verify gate layout (long-range gates cause χ inflation)
3. Review truncation tolerance (too strict = unnecessary precision)
4. Monitor GPU utilization (low % suggests CPU fallback active)

### When Results Seem Wrong
1. Check \(\varepsilon_\text{global}\) against problem tolerance requirements
2. Verify canonicalization (numerical drift in long circuits)
3. Review precision policy (complex64 may be insufficient)
4. Validate circuit translation (gate decomposition artifacts)

### When Memory Explodes
1. Monitor χ histogram (exponential growth = wrong algorithm choice)
2. Check for unnecessary state copies (in-place updates preferred)
3. Review batch size (parallel shots multiply memory)
4. Verify GPU memory pooling (fragmentation issues)

---

## Appendix D: Configuration Templates

### High-Accuracy Scientific Computing
```yaml
mode: mps
chi_max: 256
eps_bond: 1e-10
precision: complex128
canonicalization: mixed
deterministic: true
```

### Fast NISQ Emulation
```yaml
mode: hybrid
chi_max: 64
eps_bond: 1e-6
precision: complex64
gpu_fusion: aggressive
noise_model: depolarizing(0.001)
```

### Large Clifford Circuits
```yaml
mode: stabilizer
fallback: mps
chi_max: 32
tableau_compression: true
gpu_offload: false
```

### Variational Algorithms (VQE/QAOA)
```yaml
mode: mps
chi_max: 128
eps_bond: 1e-8
tdvp_order: 2-site
adaptive_chi: true
energy_tolerance: 1e-7
```

---

*End of ATLAS-Q Engineering Guide (System Design)*
