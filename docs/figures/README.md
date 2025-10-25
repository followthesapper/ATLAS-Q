# Figures and Diagrams

Visual documentation for AQED transformers and Quantum Hybrid Simulator.

## AQED Transformer Diagrams

### `aqed_architecture_comparison.svg`
Side-by-side architectural comparison of three transformer variants:
- **Traditional Transformer**: Full self-attention at every layer (O(N·L²·d))
- **AQED v1** ⭐: Selective attention skipping (`attn_keep_every=8`) achieving 6-10× speedup
- **AQED LowRank**: Dual-path architecture with architectural bottlenecks (not recommended)

**Key Insights:**
- Shows attention computation patterns across layers
- Highlights AQED's selective skipping strategy
- Explains why LowRank variant underperforms (101 kernels/step)

**Referenced in:** `AQED_WHITEPAPER.md` Section 3 (Architecture)

---

### `aqed_performance_comparison.svg`
Bar chart visualization of Tests 1-6 benchmark results:
- Test 1: Baseline (no optimization) - 34k tokens/sec
- Test 2: Baseline (optimized) - 49k tokens/sec
- Test 3: AQED v1 (no optimization) - 145k tokens/sec
- Test 4: AQED v1 (optimized) ⭐ - 211k tokens/sec (6.18× speedup)
- Test 5: AQED LowRank (no optimization) - 99k tokens/sec
- Test 6: AQED LowRank (optimized) - 113k tokens/sec

**Includes:**
- L=8192 inset showing 10.59× speedup
- Key findings summary
- Visual comparison of performance scaling

**Referenced in:** `AQED_WHITEPAPER.md` Section 5 (Benchmarks), `README.md`

---

## Quantum Simulator Diagrams

### `quantum_simulator_comparison.svg`
Comprehensive comparison of quantum simulation approaches:

**Traditional Classical Simulators:**
- Full State Vector: O(2ⁿ) memory - impractical beyond 30-40 qubits
- Tensor Networks (MPS): O(n·χ²·d²) memory - limited by entanglement

**Our Hybrid System:**
- Compressed Periodic State: O(1) memory - constant regardless of qubit count
- O(√r) Period-Finding: 100-1000× faster than exhaustive search
- GPU-accelerated modular exponentiation

**Example:** Factor N=221 (Shor's algorithm)
- Traditional: 4 MB memory, 2¹⁵ amplitudes
- Hybrid: 24 bytes memory, O(√r) time

**Referenced in:** `QUANTUM_SIMULATOR_WHITEPAPER.md` Section 2 (State Compression), `README.md`

---

## Usage

All diagrams are in SVG format for:
- ✅ Scalable vector graphics (zoom without pixelation)
- ✅ Small file sizes (~10-15 KB each)
- ✅ Easy embedding in markdown and HTML
- ✅ Professional publication quality

### Viewing Diagrams

**In Browser:**
```bash
open docs/figures/aqed_architecture_comparison.svg
```

**In Markdown:**
```markdown
![AQED Architecture](docs/figures/aqed_architecture_comparison.svg)
```

**In HTML:**
```html
<img src="docs/figures/aqed_architecture_comparison.svg" alt="AQED Architecture" width="100%">
```

---

## Diagram Sources

All diagrams created: **2025-10-25** (v0.3.0)

Based on:
- Benchmark results from `runs/aqed_benchmark/` and `runs/aqed_L8192/`
- Technical specifications from `AQED_WHITEPAPER.md`
- Complexity analysis from `QUANTUM_SIMULATOR_WHITEPAPER.md`

---

## Version

Version: **0.3.0**
Last updated: **2025-10-25**

See `CHANGELOG.md` for diagram history and updates.
