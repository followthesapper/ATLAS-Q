# ATLAS-Q: What It Is and Why It Matters
**Adaptive Tensor Learning And Simulation – Quantum**

*A complete guide for everyone - from family and friends to quantum researchers*

---

## How to Read This Document

- **For friends and family:** Read the first 3-4 sections for a simple explanation
- **For potential users:** Read through "Real-World Use Cases" to see if this fits your needs
- **For technical evaluation:** Read the entire document including comparisons and technical details
- **For researchers:** Jump to "Technical Approach" and "Performance Numbers"

### Related Documentation

- **[ Try It Now](../ATLAS_Q_Demo.ipynb)** - Interactive Jupyter notebook (no install needed with Colab!)
- **[Complete Guide](COMPLETE_GUIDE.md)** - Detailed tutorials and API reference
- **[Feature Status](FEATURE_STATUS.md)** - What's working right now
- **[Whitepaper](WHITEPAPER.md)** - Technical architecture
- **[Research Paper](RESEARCH_PAPER.md)** - Theory and algorithms

---

## The 30-Second Version

ATLAS-Q is software that lets regular computers (with good graphics cards) simulate quantum computers. Instead of needing an actual billion-dollar quantum computer, you can run quantum algorithms on your desktop or laptop.

**The cool part:** It can simulate systems with 100,000+ quantum bits (qubits) when most similar software maxes out at 30-40 qubits. That's like being able to edit an 8K video on your phone when everyone else needs a supercomputer.

**How it works:** Instead of storing every possible quantum state (which explodes exponentially), ATLAS-Q stores patterns and rules. When quantum systems have structure, this compression works incredibly well.

---

## What's a Quantum Computer? (Quick Refresher)

Think of regular computer bits like light switches - they're either ON (1) or OFF (0). Quantum bits (qubits) are like dimmer switches - they can be partially on AND partially off at the same time. This lets quantum computers explore many solutions simultaneously.

**The problem with simulation:** Simulating quantum computers on regular computers gets exponentially harder:
- 30 qubits = 16 GB of memory needed
- 40 qubits = 16 TERABYTES of memory needed
- 50 qubits = 16 PETABYTES of memory needed (basically impossible)

Most simulators hit a wall at 30-40 qubits because they run out of memory.

---

## What ATLAS-Q Does Differently

### **The Core Insight: Exploit Patterns, Not Brute Force**

Instead of storing every possible quantum state (like storing every word in the dictionary), ATLAS-Q stores the *patterns* and *rules* (like storing grammar rules and key words). When quantum states have structure, this compression trick works incredibly well.

**Think of it like:**
- **Regular approach:** Store every email address individually
- **ATLAS-Q approach:** Store the pattern (regex) instead: `[a-z]+@gmail\.com`

Or:
- **Traditional:** Recording every word spoken in a conversation
- **ATLAS-Q:** Recording the grammar rules and key words to regenerate the conversation

This is exactly how ATLAS-Q represents quantum states - as compact patterns that can regenerate the full information when needed.

### **The Results:**
- Compressed 16 GB down to 0.03 MB (626,000× smaller!)
- Can simulate 100,000 qubits for low-to-moderate entanglement scenarios
- 20× faster for certain types of quantum operations (Clifford circuits)
- 1.5-3× faster than standard approaches through custom GPU code
- Runs on regular gaming GPUs (not just supercomputers)

### **Important Caveat:**
The 100,000 qubit number applies to specific types of quantum circuits - particularly those with **low-to-moderate entanglement** or structured patterns (like nearest-neighbor coupled systems, molecular simulations, or shallow circuits).

It's not a general-purpose 100,000 qubit quantum computer simulator. Think of it like data compression: you can compress a text file to 1% of its size, but random data won't compress at all. ATLAS-Q works great for "structured" quantum states but struggles with maximally entangled random states (just like every other classical simulator).

---

## Why This Matters

### **For Scientific Research**
Before ATLAS-Q, testing a new quantum algorithm meant:
1. Writing the algorithm
2. Waiting months for access to a real quantum computer
3. Paying thousands of dollars per hour to run it
4. Limited to small tests (50-100 qubits on real hardware)

Now researchers can:
1. Test algorithms on their laptop
2. Simulate 100,000+ qubits for certain problems
3. Iterate and debug quickly
4. Save months and thousands of dollars

### **For Education**
University quantum computing courses were limited:
- Students could only run tiny 10-20 qubit examples
- Had to share expensive cloud computing credits
- Couldn't experiment freely

With ATLAS-Q:
- Students can run 100-qubit examples on their laptops
- Experiment and learn without cost limits
- See quantum algorithms at realistic scales

### **For Algorithm Development**
Quantum algorithm developers can prototype and test at scales not available on current quantum hardware, enabling faster iteration and better algorithms before deploying to expensive real quantum computers.

---

## Real-World Examples

### **Example 1: Drug Discovery**
Pharmaceutical companies want to simulate how molecules behave to find new medicines. Each atom in a molecule might need several qubits to simulate:
- **Traditional simulators:** ~30 atoms maximum
- **ATLAS-Q:** 100+ atoms for molecules with typical structures
- **Why it works:** Molecules naturally have low-to-moderate entanglement because atoms interact locally (nearest neighbors), not all-to-all
- **Result:** Can test 10× more drug candidates in the same time

### **Example 2: Financial Optimization**
Running quantum optimization algorithms (QAOA) for portfolio optimization or risk analysis:
- **Traditional simulators:** 20-30 assets maximum
- **ATLAS-Q:** 100+ assets with complex constraints
- **Why it works:** QAOA circuits typically have moderate depth and local structure
- **Result:** Test sophisticated strategies that wouldn't fit in traditional simulators

### **Example 3: Materials Science**
Simulating quantum spin systems to understand magnetism or superconductivity:
- **Traditional simulators:** ~40 spins
- **ATLAS-Q:** 1,000+ spins in 1D or 2D lattices
- **Why it works:** Condensed matter systems have local interactions (nearest-neighbor coupling)
- **Result:** Study phase transitions and quantum behavior at realistic scales

### **Example 4: Quantum Algorithm Research**
PhD student working on new quantum error correction codes:
- **Problem:** Need to test 50-100 qubit error correction circuits
- **ATLAS-Q solution:** Run thousands of test cases locally, iterate quickly
- **Why it works:** Error correction circuits often have structure (stabilizer codes)
- **Result:** Faster research progress without waiting for cloud quantum computer access

---

## What Makes ATLAS-Q Special?

### **1. Extreme Memory Efficiency**
By storing patterns instead of full quantum states, ATLAS-Q achieves compression ratios that make the impossible possible:
- 30 qubits: 0.03 MB instead of 16 GB
- 100,000 qubits with low-to-moderate entanglement: ~120 GB (vs impossible for traditional simulators)

### **2. Custom GPU Acceleration**
Hand-written GPU kernels in Triton specifically optimized for quantum tensor operations:
- 1.5-3× faster than using PyTorch alone
- Optimized for the exact math quantum simulation needs
- Like the difference between a calculator and a purpose-built scientific instrument

### **3. Smart Automatic Optimization**
ATLAS-Q automatically detects when quantum operations can use faster algorithms:
- Switches to ultra-fast Stabilizer formalism when possible (20× speedup)
- Falls back to general tensor networks when needed
- You don't need to know which is running - it just works

### **4. Production-Grade Robustness**
Built to actually work, not just for research demos:
- Multiple fallback strategies when numerical issues arise
- Comprehensive error tracking and diagnostics
- Mixed precision support for numerical stability
- 75+ tests across multiple categories

---

## Honest Questions, Honest Answers

### **Q: Does ATLAS-Q scale better than any other simulator?**

**Honest answer:** For the right types of problems, it scales as well as the best tensor network simulators and better than traditional statevector simulators.

**Details:**
- **Other tensor network simulators** (ITensor, quimb, CUDA-Q with MPS) can ALSO simulate 100K+ qubits with low entanglement
- **What makes ATLAS-Q special:** The engineering - custom GPU kernels, hybrid backend switching, demonstrated 626,000× compression
- **Fundamental scaling:** All tensor network methods scale the same way - by entanglement, not by qubit count
- **ATLAS-Q's advantage:** Faster execution (1.5-3× from Triton kernels, 20× from Clifford optimization), better integration

**Bottom line:** ATLAS-Q scales as well as modern tensor network simulators, with optimizations that make it faster and more efficient for many scenarios. It's not a new scaling breakthrough, just excellent engineering of proven tensor network theory.

### **Q: Could ATLAS-Q run on a real quantum computer? Would it be insanely better?**

**Honest answer:** No, ATLAS-Q is a classical simulator - but the techniques could be adapted for hybrid systems.

**Details:**
- **What ATLAS-Q does:** Simulates quantum computers on classical hardware (GPUs)
- **Could it run on quantum computers?** Not as-is. It's a classical program running on GPUs.
- **Hybrid classical-quantum:** Research (2024) shows tensor network techniques CAN be combined with real quantum computers:
 - Classical computer (ATLAS-Q-like) handles low-entanglement parts
 - Quantum computer handles highly entangled parts
 - They work together in a hybrid system
- **What this would require:** Significant redesign - not something ATLAS-Q does today

**Think of it like:** ATLAS-Q is a flight simulator for training pilots. You don't fly the simulator IN a plane - you use it to practice before flying a real plane. But someday, flight simulators could be integrated into plane cockpits for training during flights (hybrid approach).

**Current reality:** You use ATLAS-Q INSTEAD of a quantum computer for testing, research, and education. It's not something you run ON a quantum computer.

---

## When to Use ATLAS-Q

### **Good Fit:**

**1. Algorithm Research & Development**
- Testing quantum algorithms before running on real quantum hardware
- Exploring algorithm behavior at larger scales than hardware allows
- Rapid prototyping and iteration

**2. Quantum Chemistry & Materials Science**
- Variational Quantum Eigensolver (VQE) for molecular ground states
- Time evolution of quantum systems (TDVP)
- Systems with local interactions (not highly entangled everywhere)

**3. Optimization Problems**
- QAOA (Quantum Approximate Optimization Algorithm)
- Combinatorial optimization
- Parameter tuning and training

**4. Education & Learning**
- Teaching quantum algorithms without requiring quantum hardware access
- Demonstrating quantum concepts at realistic scales
- Experimenting with quantum circuits interactively

**5. Condensed Matter Physics**
- Simulating quantum spin chains (1000+ sites)
- Studying phase transitions
- Computing correlation functions and entanglement entropy

### **Not Ideal For:**

**1. Highly Entangled Circuits**
- Random circuits with deep entanglement
- Systems where every qubit strongly couples to every other qubit
- Circuits specifically designed to be classically hard

**2. Exact Simulation Requirements**
- When you need bit-perfect accuracy with zero approximation
- Benchmarking against specific statevector results (< 30 qubits)
- Applications where any approximation error is unacceptable

**3. CPU-Only Environments**
- ATLAS-Q is optimized for CUDA GPUs
- CPU fallbacks exist but performance is significantly lower
- Best results need NVIDIA graphics cards

**4. Absolute Beginners**
- If you're just learning quantum computing basics, Qiskit or Cirq have better tutorials
- ATLAS-Q assumes some quantum computing background
- Learning curve is steeper than industry-standard tools

---

## Technical Approach (Simplified)

### **Tensor Networks (Matrix Product States / MPS)**

Instead of storing all 2^n quantum amplitudes, ATLAS-Q represents the quantum state as a chain of smaller tensors connected together.

**Think of it like:**
- **Traditional approach:** Store every email address that matches a pattern (millions of entries)
- **ATLAS-Q approach:** Store the regex pattern itself (`/[a-z]+@gmail\.com/`)

Or:
- **Traditional:** Store a 1-million-entry phonebook
- **ATLAS-Q:** Store patterns + rules to generate entries on-demand

This works when the quantum state has structure. Systems with low-to-moderate entanglement compress well; maximally entangled random states don't (just like random data doesn't compress well).

### **Stabilizer Formalism**

For certain types of quantum operations (Clifford gates: H, S, CNOT), there's a mathematical trick that lets you track the evolution using only n² numbers instead of 2^n.

ATLAS-Q automatically detects when this applies and switches to this ultra-fast mode (20× speedup). When non-Clifford gates appear (T, Toffoli), it seamlessly switches to the general tensor network method.

### **Adaptive Truncation**

As circuits get deeper, some patterns become less important. ATLAS-Q automatically identifies and discards negligible patterns while tracking the approximation error.

You can control the accuracy vs. speed tradeoff - need higher accuracy? It keeps more patterns. Need faster simulation? It aggressively truncates.

---

## For Technical Users: Detailed Capabilities

### **Performance Numbers (What to Expect)**

**Memory Efficiency:**
- 30 qubits with low-to-moderate entanglement: **0.03 MB** (vs 16 GB for statevector)
- 50 qubits with bond dimension χ=16: **~1 MB** (vs 16 petabytes for statevector)
- 100,000 qubits with χ=64 (low-to-moderate entanglement): **~120 GB** (vs impossible)

**Speed:**
- Gate operations: **77,000+ ops/sec** on NVIDIA GPUs
- Clifford circuits: **20× faster** than generic tensor network methods
- VQE optimization: Competitive with Qiskit Aer for equivalent system sizes

**Accuracy:**
- Adaptive truncation error: Controllable down to 10^-8 or better
- Error bounds tracked and reported throughout simulation
- Can trade accuracy for speed based on application needs

### **Technical Features**

**GPU Acceleration:**
- Custom Triton kernels for 2-qubit gate operations
- cuBLAS tensor core acceleration (TF32)
- Modular exponentiation kernels for period-finding
- Automatic CPU fallback if GPU unavailable

**Noise Models:**
- Depolarizing, dephasing, amplitude damping, Pauli noise
- Stochastic noise applicator with reproducible seeds
- NISQ-era simulation capabilities matching Qiskit Aer

**Advanced Algorithms:**
- TDVP (Time-Dependent Variational Principle) for time evolution
- VQE (Variational Quantum Eigensolver) for ground states
- QAOA (Quantum Approximate Optimization Algorithm)
- MPO (Matrix Product Operator) framework for Hamiltonians

**Molecular Chemistry (v0.6.0):**
- PySCF integration for electronic structure Hamiltonians
- Jordan-Wigner transformation for fermion-to-qubit mapping
- Support for H2, LiH, H2O, and custom molecular geometries
- Seamless VQE integration for ground state energy calculations

**Graph Optimization (v0.6.0):**
- MaxCut Hamiltonian builder for QAOA problems
- Weighted and unweighted graph support
- Automatic edge normalization for undirected graphs

**Advanced Tensor Networks (v0.6.0):**
- Circuit Cutting: Min-cut partitioning for large circuits
- PEPS: True 2D tensor networks for shallow circuits
- Distributed MPS: Multi-GPU bond-parallel decomposition
- cuQuantum Backend: Optional NVIDIA acceleration (2-10× speedup)

**Robustness:**
- Multi-driver SVD with fallback (gesdd → gesvd → jitter → precision promotion)
- Automatic precision promotion when numerical stability requires it
- Comprehensive error tracking and statistics
- Canonical form support (left, right, mixed)

### **Installation & Quick Start**

**Prerequisites:**
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support (optional but recommended)

**Installation:**
```bash
git clone https://github.com/followthsapper/ATLAS-Q.git
cd ATLAS-Q
pip install -r requirements.txt
pip install -e .
```

**Simple Example (for technical users):**

Creating and manipulating a 50-qubit system:
```python
from atlas_q import get_adaptive_mps
import torch

# Create MPS for 50 qubits
adaptive = get_adaptive_mps()
mps = adaptive['AdaptiveMPS'](50, bond_dim=16, device='cuda')

# Apply Hadamard gates to all qubits
H = torch.tensor([[1,1],[1,-1]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
for q in range(50):
 mps.apply_single_qubit_gate(q, H.to('cuda'))

# Check compression and statistics
print(mps.stats_summary())
# Shows: bond dimensions, memory usage, compression ratio, etc.
```

For more examples, see `COMPLETE_GUIDE.md`.

---

## Comparison with Major Simulators

| Feature | ATLAS-Q | Qiskit Aer | Cirq | ITensor | TeNPy |
|---------|---------|------------|------|---------|-------|
| **Max Qubits (typical)** | 100+ (structured) | ~40 | ~40 | 100+ (1D) | 100+ (1D) |
| **Memory Efficiency** | Excellent | Standard | Standard | Excellent | Excellent |
| **GPU Support** | Yes (Triton) | Yes (cuQuantum) | Limited | No | No |
| **Ease of Use** | Moderate | Excellent | Excellent | Moderate | Good |
| **Documentation** | Good | Excellent | Excellent | Excellent | Good |
| **Noise Models** | Yes | Yes | Yes | No | Limited |
| **Ecosystem** | Growing | Mature | Mature | Mature | Mature |
| **Best For** | Research, large TN | General purpose | Google stack | Physics | Condensed matter |

### **When to Choose ATLAS-Q:**
- You need to simulate 50+ qubit systems with low-to-moderate entanglement
- GPU acceleration is important to you
- Working with tensor network algorithms specifically
- Memory constraints are your bottleneck
- You're comfortable with tensor network concepts

### **When to Choose Alternatives:**
- **Qiskit Aer:** You want the most mature ecosystem, best tutorials, broadest community support
- **Cirq:** You're integrating with Google's quantum stack or need excellent documentation
- **ITensor/TeNPy:** You're doing condensed matter physics research exclusively and want physics-specific features
- **Exact simulators:** You need bit-perfect accuracy for < 30 qubits

---

## Real-World Use Cases (Detailed)

### **Use Case 1: Drug Discovery Research**
**Scenario:** Pharmaceutical company simulating molecular interactions for drug candidates.

**How ATLAS-Q helps:**
- Run VQE simulations for molecules with 50-100 atoms (100-200 qubits)
- Explore potential energy surfaces faster than traditional methods
- Iterate through thousands of configurations in days instead of months

**vs Other Simulators:**
- Qiskit Aer: Limited to ~30 qubits for VQE (memory constraints)
- ATLAS-Q: Handles 100+ qubit systems with local interactions (molecules naturally have low-to-moderate entanglement due to local coupling)

**Technical details:** Molecules have local electronic interactions, resulting in MPS bond dimensions that grow slowly with system size. This is exactly what ATLAS-Q is optimized for.

---

### **Use Case 2: Algorithm Development for NISQ Devices**
**Scenario:** Quantum computing startup developing algorithms for near-term quantum hardware.

**How ATLAS-Q helps:**
- Test algorithms with realistic noise models before paying for quantum hardware time
- Iterate quickly on algorithm design (hours vs weeks)
- Validate correctness at scale before deployment

**vs Other Simulators:**
- Similar to Qiskit Aer for noise simulation capabilities
- Advantage: 20× faster for Clifford-heavy circuits (common in error correction research)
- Can test larger systems than fit on current quantum hardware

---

### **Use Case 3: University Quantum Computing Course**
**Scenario:** Professor teaching quantum algorithms to 50 students.

**How ATLAS-Q helps:**
- Students run experiments on laptops with gaming GPUs
- Demonstrate Shor's algorithm, Grover's search, VQE on systems large enough to be interesting
- No cloud credits or hardware access management needed

**vs Other Simulators:**
- Qiskit/Cirq: Better tutorials and documentation (easier to learn)
- ATLAS-Q: Can run larger examples (100-qubit Grover vs 20-qubit in Qiskit)
- Trade-off: Steeper learning curve but more impressive demonstrations

---

### **Use Case 4: Condensed Matter Physics Research**
**Scenario:** Physicist studying quantum spin systems and phase transitions.

**How ATLAS-Q helps:**
- Simulate spin chains with 1,000+ sites
- Compute time evolution and correlation functions
- Study quantum phase transitions with adaptive truncation

**vs Other Simulators:**
- ITensor/TeNPy: Purpose-built for this, excellent domain-specific documentation
- ATLAS-Q: Competitive performance + GPU acceleration, more general-purpose

---

## Limitations & Honest Assessment

### **What ATLAS-Q Does Well**
 Memory efficiency (626,000× compression demonstrated)
 GPU acceleration (custom Triton kernels)
 Hybrid backend optimization (automatic Stabilizer switching)
 Production-ready robustness (fallbacks, error tracking)
 Large-scale structured systems (100K+ qubits)

### **Where ATLAS-Q Struggles**
 Highly entangled random circuits (same as all classical simulators)
 Ease of use for beginners (steeper learning curve than Qiskit)
 Ecosystem maturity (smaller community than industry leaders)
 Non-CUDA hardware (optimized for NVIDIA GPUs)
 Perfect accuracy requirements (uses controlled approximations)

### **Honest Competitive Assessment**

**Is it groundbreaking?**
No - tensor network methods have existed for decades. ATLAS-Q is excellent engineering and optimization of existing theory, not a fundamentally new approach.

**Is it valuable?**
Yes - the combination of features (GPU acceleration, hybrid backends, compression efficiency, production quality) makes it a strong tool for specific use cases where other simulators struggle.

**Should you use it?**
If you need large-scale tensor network simulation with GPU acceleration and you're comfortable with the learning curve: yes. If you're learning quantum computing basics or need maximum ecosystem support: start with Qiskit or Cirq.

### **The Bottom Line**

ATLAS-Q is **genuinely impressive engineering** that achieves compression ratios and qubit scales difficult or impossible with traditional simulators. The custom GPU kernels and hybrid backend switching show production-grade work.

**It's not magic:** The efficiency comes from exploiting structure in quantum systems. For highly entangled, unstructured circuits, it faces similar limitations as other simulators.

**Best use:** Research, education, and algorithm development where scaling beyond 40 qubits matters and you have quantum systems with low-to-moderate entanglement or structure. Think VQE for chemistry, QAOA for optimization, or quantum spin chains in physics.

**Realistic framing:** Technically strong, well-engineered, fills a genuine gap. Not yet as mature or easy-to-use as Qiskit/Cirq, but offers capabilities they can't match for specific workloads. Solid tool, not revolution.

---

## Common Questions

### **What makes this different from other quantum simulators?**
ATLAS-Q simulates quantum computers way more efficiently than traditional approaches. Most tools can handle 30-40 quantum bits; ATLAS-Q can handle 100,000+ for certain types of problems. This is achieved by being smart about compression - storing patterns instead of everything - which lets researchers test quantum algorithms without needing actual quantum computers.

### **Is this a major breakthrough?**
ATLAS-Q is a solid engineering achievement that makes quantum research more accessible. It's not going to revolutionize quantum computing itself, but it provides researchers with a significantly better tool to develop and test quantum algorithms. Think "really good tooling" rather than "fundamental breakthrough."

### **How does it compare to existing tools?**
For the specific types of quantum problems it's designed for - like simulating molecules, testing certain quantum algorithms, or studying quantum spin systems - ATLAS-Q can handle way larger systems than standard tools. It's not better at everything, but it fills a genuine gap that researchers needed for large-scale structured quantum systems.

---

## Current Status & Future

### **Version 0.6.0 (Current - October 2025)**
- Full tensor network implementation
- GPU acceleration with custom Triton kernels
- Noise models and NISQ simulation
- VQE, QAOA, TDVP algorithms
- **NEW:** Molecular Hamiltonians (PySCF integration)
- **NEW:** MaxCut QAOA Hamiltonians
- **NEW:** Circuit Cutting & partitioning
- **NEW:** PEPS 2D tensor networks
- **NEW:** Distributed MPS (multi-GPU ready)
- **NEW:** cuQuantum 25.x backend integration
- All 46/46 integration tests passing
- Production-ready for research use

### **Planned Improvements**
- Integration adapters for Qiskit/Cirq circuits
- Additional tutorial notebooks
- PyPI package for easier installation
- Expanded documentation and examples

---

## Learning Resources

- **This document:** High-level overview and comparisons
- **Whitepaper** (`WHITEPAPER.md`): Technical architecture and implementation
- **Research Paper** (`RESEARCH_PAPER.md`): Mathematical foundations and algorithms
- **Complete Guide** (`COMPLETE_GUIDE.md`): Installation, tutorials, API reference, and examples
- **Test Suite:** 75+ examples of working code in `/tests/`
- **Demos:** Working examples in `/scripts/demos/`
- **Benchmarks:** Reproducible performance tests in `/scripts/benchmarks/`

---

## Who Should Use ATLAS-Q?

### **Ideal Users:**
- Quantum algorithm researchers exploring large-scale behavior
- Graduate students working on quantum computing theses
- Physicists studying quantum many-body systems
- Algorithm developers prototyping before hardware deployment
- Educators teaching advanced quantum computing concepts

### **May Want Alternatives:**
- Beginners learning quantum computing basics → Qiskit has better tutorials
- Teams requiring commercial support → No company backing ATLAS-Q yet
- Maximum ecosystem integration needed → Qiskit/Cirq more mature
- CPU-only environments → GPU optimization won't help

---

## Getting Help

- **Issues:** https://github.com/followthsapper/ATLAS-Q/issues
- **Discussions:** https://github.com/followthsapper/ATLAS-Q/discussions
- **Documentation:** See `/docs/` folder
- **Examples & Demos:** See `/scripts/demos/` and `/scripts/benchmarks/` folders

---

**License:** MIT (Free and open source)
**Version:** 0.6.0 (October 2025)
**Contact:** https://github.com/followthsapper/ATLAS-Q

---

**P.S. for non-technical readers:** If you made it this far and still find this confusing, here's the ultra-simple version: "Special software that lets regular computers simulate quantum computers way better than usual by being smart about compression. Helps researchers test quantum stuff without needing expensive quantum hardware." That's it!
