# ATLAS-Q Math Guide

*A step-by-step walkthrough to the math under the hood, with symbols explained and worked examples.*

## How to use this guide

Each concept appears in three layers:
1. **What & why**
2. **Symbols & definitions**
3. **Step-by-step math** (talk-through derivations + worked examples)

Figures are sprinkled throughout to make the ideas visual.

## Notation & Symbols (quick reference)

| Symbol | Meaning |
|--------|---------|
| $\vert 0 \rangle, \vert 1 \rangle$ | Computational basis states of a single qubit (bit-like "basis vectors"). |
| $\alpha, \beta \in \mathbb{C}$ | Complex numbers (like 2D numbers with real + imaginary parts). |
| $\vert \psi \rangle$ | A quantum state (unit vector in a complex Hilbert space). |
| $n$ | Number of qubits. |
| $2^n$ | Dimension of $n$-qubit state space (number of amplitudes in statevector). |
| $A^{[k]}_i$ | MPS site-$k$ tensors; $i \in \{0,1\}$ is the physical index. |
| $\chi$ | Bond dimension: how wide the MPS "links" are; controls expressive power & memory. |
| $\sigma_s$ | Schmidt singular values across a cut (measure of entanglement "weight"). |
| $S$ | Entanglement entropy (bits): $S = -\sum_s \sigma_s^2 \log_2 \sigma_s^2$. |
| $U, G$ | Quantum gates: single-qubit $U \in \mathbb{C}^{2 \times 2}$; two-qubit $G \in \mathbb{C}^{4 \times 4}$. |
| $\text{QFT}_N$ | Quantum Fourier Transform on $N = 2^n$ points. |
| $a, r, N$ | Period-finding parameters: base $a$, period $r$, modulus $N$. |
| $\gcd(x, y)$ | Greatest common divisor. |
| $H$ | Hamiltonian (energy operator) for time evolution. |
| $\Pi_{T_\Psi}$ | Projector onto the tangent space of the MPS manifold at $\vert \Psi \rangle$ (TDVP). |

## 1. Qubits, Statevectors, and Why Memory Explodes

### 1.1 What & why
A bit is 0 or 1. A qubit can be a blend (superposition) of both. For $n$ qubits, the list of "weights" (amplitudes) you'd need grows as $2^n$. That's why naive simulation explodes in memory.

### 1.2 Symbols & definitions

**Single qubit:**

$$\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle, \quad \alpha, \beta \in \mathbb{C}, \quad |\alpha|^2 + |\beta|^2 = 1.$$

**$n$-qubit statevector:**

$$\vert \Psi \rangle = \sum_{x \in \{0,1\}^n} c_x \vert x \rangle, \quad \sum_x |c_x|^2 = 1.$$

There are $2^n$ amplitudes $c_x$.

**Memory (statevector) with complex64 (16 bytes each):**

$$\text{bytes} = 16 \cdot 2^n.$$

### 1.3 Step-by-step math (talk-through)

- For each qubit you add, the number of basis strings doubles (append 0 or 1).
- So after $n$ qubits, the count is $2^n$.
- Each amplitude is 16 bytes (complex64), so memory is $16 \cdot 2^n$ bytes.
- Check: $n = 30 \Rightarrow 16 \cdot 2^{30} \approx 16$ GB. That's already desktop-sized max.

**Figure 1** — Memory scaling (log scale)

## 2. Matrix Product States (MPS): The Compression Engine

### 2.1 What & why
MPS breaks the huge state into a chain of small tensors. If entanglement is moderate, the chain is narrow (small $\chi$), and memory is near linear in $n$ instead of exponential.

### 2.2 Symbols & definitions

**MPS form:**

$$\vert \Psi \rangle = \sum_{i_1, \ldots, i_n \in \{0,1\}} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} \vert i_1 \cdots i_n \rangle,$$

where each $A^{[k]}_{i_k} \in \mathbb{C}^{\chi_{k-1} \times \chi_k}$.

$\chi = \max_k \chi_k$ is the bond dimension.

**Rough memory with complex64:**

$$\text{bytes} \approx \underbrace{2n\chi^2}_{\text{\# complex entries}} \times 16.$$

### 2.3 Step-by-step math (talk-through)

- Each site has two matrices $A^{[k]}_0, A^{[k]}_1$, each roughly $\chi \times \chi$.
- That's $2\chi^2$ complex numbers per site $\Rightarrow 2n\chi^2$ across $n$ sites.
- Multiply by 16 bytes: $32n\chi^2$ bytes.
- Compare to $16 \cdot 2^n$ — when $\chi$ is small, MPS is dramatically smaller.

**ASCII sketch (Figure A):**
```
[A^[1]]--χ--[A^[2]]--χ--[A^[3]]-- ... --[A^[n]]
```
Each "--χ--" is a virtual link of width χ; bigger χ = more entanglement capacity.

## 3. Entanglement: Why $\chi$ Matters

### 3.1 What & why
Entanglement is "how much two parts depend on each other." More entanglement needs wider links ($\chi$).

**Intuitive analogy:** Think of $\chi$ as "how many stories" you keep about the relationship between neighboring qubits. Calm, independent relationships = few stories needed. Drama everywhere = lots of stories to track.

### 3.2 Symbols & definitions

**Schmidt decomposition across a cut:**

$$\vert \Psi \rangle = \sum_{s=1}^{\chi} \sigma_s \vert L_s \rangle \otimes \vert R_s \rangle, \quad \sum_s \sigma_s^2 = 1.$$

**Entropy (bits):**

$$S = -\sum_s \sigma_s^2 \log_2(\sigma_s^2).$$

### 3.3 Step-by-step (talk-through)

- Any bipartition can be written as a sum of "links" weighted by $\sigma_s$.
- The $\sigma_s^2$ act like probabilities and sum to 1.
- If only one $\sigma$ is nonzero → $S = 0$ (no entanglement).
- If many are comparable → larger $S$ → need larger $\chi$.

**Figure 3** — Entropy examples

## 4. Gates on MPS: Merge → Apply → SVD → Truncate

### 4.1 What & why
Single-qubit gates are easy (local). Two-qubit neighbor gates can increase entanglement; we compress right away so $\chi$ stays under control, while tracking the approximation error.

### 4.2 Symbols & definitions

**Single-qubit gate $U$ at site $k$:**

$$\tilde{A}^{[k]}_i = \sum_{j \in \{0,1\}} U_{ij} A^{[k]}_j.$$

**Two-qubit gate $G$ on $k, k+1$:**
1. Merge $\Rightarrow \Theta \in \mathbb{C}^{(2\chi) \times (2\chi)}$
2. Apply $G$ to the physical $2 \times 2$ space
3. SVD $\tilde{\Theta} = USV^\dagger$
4. Truncate to cap $\chi$, then reshape back into two sites.

### 4.3 Step-by-step (talk-through)

- **Merge:** build a matrix $\Theta$ that mixes physical indices $(i, j)$ with virtual links $(\alpha, \beta)$.
- **Apply gate:** multiply $\Theta$ by the $4 \times 4$ gate on the physical indices → $\tilde{\Theta}$.
- **SVD:** factor $\tilde{\Theta} = USV^\dagger$; singular values $S$ tell us how much entanglement this bond needs.
- **Truncate:** keep the largest $k$ singular values (cap $\chi$ and meet error tolerance), drop the rest, and split back into two tensors.

**Figure 4a** — Singular values; **Figure 4b** — Cumulative energy

## 5. Truncation & Error: Keep the Signal, Quantify the Loss

### 5.1 What & why
We compress by dropping tiny singular values and track how much we dropped (local error). Summarizing these gives a safe global error bound.

### 5.2 Symbols & definitions

Keep smallest $k$ such that

$$\sum_{s=1}^k \sigma_s^2 \geq 1 - \varepsilon_{\text{bond}}^2.$$

**Local error:**

$$\varepsilon_{\text{local}} = \sum_{s>k} \sigma_s^2.$$

**Global bound over $M$ truncations:**

$$\varepsilon_{\text{global}} \leq \sqrt{\sum_{m=1}^M \varepsilon_m^2}.$$

### 5.3 Step-by-step (talk-through)

- The singular values squared behave like "energy" or "probability mass."
- We pick a tolerance $\varepsilon_{\text{bond}}$ ("acceptable per-bond loss").
- Keep as many top singular values as needed to reach that cumulative target.
- The leftover is the local truncation error; sum-in-quadrature gives a safe global bound.

## 6. Stabilizer (Clifford) Fast Path

### 6.1 What & why
If a circuit uses only Clifford gates (H, S, CNOT, etc.), we can simulate much faster with a tableau (no big tensors). ATLAS-Q automatically uses this when it applies; when a non-Clifford (e.g., $T$) appears, it hands off to MPS.

### 6.2 Symbols & definitions

- **Tableau:** a binary matrix $(X|Z) \in \{0,1\}^{n \times 2n}$ + phases describes the stabilizer group.
- Clifford gates are linear updates on $(X|Z)$; cost ~ $\tilde{O}(n^2)$.

### 6.3 Step-by-step (talk-through)

- Stabilizer states are the joint +1 eigenstates of $n$ Pauli strings.
- We track those strings compactly in a tableau.
- Each Clifford gate is a structured row/column operation.
- This avoids the tensor machinery and is dramatically faster for large Clifford subcircuits.

## 7. QFT & Periodic States (Shor's Subroutine)

### 7.1 What & why
A periodic state has nonzero amplitudes at positions $a, a+r, a+2r, \ldots$. The QFT turns "spacings in time" into "spikes in frequency": it peaks at multiples of $N/r$.

### 7.2 Symbols & definitions

**QFT on $N = 2^n$:**

$$\text{QFT}_N \vert x \rangle = \frac{1}{\sqrt{N}} \sum_{m=0}^{N-1} e^{2\pi i mx/N} \vert m \rangle.$$

**Periodic state:**

$$\vert \psi_{a,r} \rangle = \frac{1}{\sqrt{k}} \sum_{j=0}^{k-1} \vert a + jr \rangle, \quad k = \left\lfloor \frac{N-a-1}{r} \right\rfloor + 1.$$

**Closed-form amplitude (Dirichlet-type kernel):**

$$\langle m \vert \text{QFT}_N \vert \psi_{a,r} \rangle = \frac{e^{2\pi i am/N}}{\sqrt{Nk}} \cdot \frac{1 - e^{2\pi i rmk/N}}{1 - e^{2\pi i rm/N}}.$$

### 7.3 Step-by-step (talk-through)

- Start with the definition of QFT; plug in a sum over $x = a + jr$.
- You get a geometric series $\sum_{j=0}^{k-1} z^j$ with $z = e^{2\pi i rm/N}$.
- Sum it with $\sum_{j=0}^{k-1} z^j = \frac{1-z^k}{1-z}$ (when $z \neq 1$).
- The magnitude squares produce peaks whenever the denominator $\sin(\pi rm/N)$ is near zero → $m \approx \ell \cdot N/r$.

**Figure 2** — QFT peaks for a periodic state

## 8. From Period to Factors (Shor's Logic)

### 8.1 What & why
Once you know the period $r$ of $a^x \mod N$, you can usually recover the prime factors of $N$ using gcd tricks.

### 8.2 Symbols & definitions

Find smallest $r > 0$ such that $a^r \equiv 1 \pmod{N}$.

If $r$ is even and $a^{r/2} \not\equiv -1 \pmod{N}$, then

$$p = \gcd(a^{r/2} - 1, N), \quad q = \gcd(a^{r/2} + 1, N).$$

### 8.3 Step-by-step (talk-through)

- If $a^r \equiv 1$, then $a^{r/2}$ is a square root of 1 modulo $N$.
- The nontrivial square roots of 1 modulo $N$ reveal factors via gcd with $N$.
- Compute two gcds: one with $a^{r/2} - 1$, one with $a^{r/2} + 1$.
- With high probability, you get both factors $p, q$.

**Micro example:** $N = 21, a = 2$. Powers: $2, 4, 8, 16, 11, 1 \Rightarrow r = 6$.

$2^3 = 8$. $\gcd(8-1, 21) = 7$, $\gcd(8+1, 21) = 3$. So $21 = 3 \cdot 7$.

## 9. MPOs, Expectation Values, and Correlations (TN 101)

### 9.1 What & why
To measure energy or observables efficiently, we use Matrix Product Operators (MPOs) and contract them with the MPS using left/right environments.

### 9.2 Symbols & definitions (sketch)

- MPO tensors $W^{[k]}$ with virtual bond $D$.
- **Left environment recursion:**

$$E^{[k+1]} = \sum_{i_k, j_k} (A^{[k]}_{i_k})^\dagger E^{[k]} W^{[k]}_{i_k j_k} A^{[k]}_{j_k}.$$

- **Expectation** $\langle O \rangle = \text{Tr}(E^{[n+1]})$.

### 9.3 Step-by-step (talk-through)

- Start with $E^{[1]} = 1$ (scalar).
- At each site, "pull through" the MPS bra, the MPO tensor, and the MPS ket to get the next environment.
- After the last site, take the trace to get the scalar expectation value.
- GPU acceleration makes these contractions fast at scale.

## 10. Time Evolution with TDVP: Stay on the MPS Manifold

### 10.1 What & why
We want to evolve $\vert \Psi(t) \rangle$ by Schrödinger's equation but inside the MPS family (so it stays compressible). TDVP does this by projecting the time derivative onto the MPS tangent space.

### 10.2 Symbols & definitions

**Schrödinger:**

$$\frac{d}{dt} \vert \Psi(t) \rangle = -i H \vert \Psi(t) \rangle.$$

**TDVP projection:**

$$\frac{d}{dt} \vert \Psi(t) \rangle = -i \Pi_{T_\Psi}(H \vert \Psi(t) \rangle).$$

- **1-site TDVP:** keeps $\chi$ fixed
- **2-site TDVP:** allows $\chi$ to grow (via SVD in each step)

### 10.3 Step-by-step (talk-through)

- Compute $H \vert \Psi \rangle$: tendency of the state to move.
- Project this vector onto the tangent space of MPS with current $\chi$: that's the TDVP step direction.
- Euler step: $\vert \Psi(t + \Delta t) \rangle \approx \text{normalize}(\vert \Psi \rangle - i\Delta t \cdot \text{projected})$.
- 2-site TDVP periodically allows a 2-site block update + SVD so $\chi$ can increase where needed.

**Intuitive picture:** You're hiking along a ridge (the MPS manifold). TDVP keeps you on the ridge while moving forward in time, rather than falling off into the exponentially-large valley below.

## 11. Complexity Cheat-Sheet (when you win)

| Op / Model | Memory | Cost per op (dominant) |
|------------|--------|------------------------|
| Statevector (any gate) | $O(2^n)$ | $O(2^n)$ |
| MPS single-qubit | $O(n\chi^2)$ | $O(\chi^2)$ |
| MPS two-qubit (neighbors) | $O(n\chi^2)$ | $O(\chi^3)$ (SVD) |
| Stabilizer Clifford | $\tilde{O}(n^2)$ | $\tilde{O}(n^2)$ |

**Rule of thumb:** If your circuits keep entanglement moderate, $\chi$ stays small (e.g., $\leq 64$) → huge gains.

## 12. GPU & Triton: Same Math, Fewer Memory Trips

### 12.1 What & why
The math is the same; we just do it efficiently:
- Fuse "merge → apply gate → reshape" into one GPU kernel (Triton).
- Use tensor cores (TF32) for fast contractions (cuBLAS).
- Fewer global-memory round-trips = 1.5–3× faster 2-qubit hotspots.

**Analogy:** Instead of walking to the pantry three times for flour, sugar, and eggs, you carry a basket and grab everything in one trip.

### 12.2 Step-by-step (talk-through)

- Build $\Theta$ from neighboring MPS cores, apply $G$, reshape → one fused kernel.
- SVD still happens (potentially with robust fallbacks if ill-conditioned).
- Use TF32 where allowed for speed; auto-promote precision if numerics demand it.

## 13. Fully Worked Mini Examples

### 13.1 Bell pair & entropy

Start $\vert 00 \rangle$. Apply $H$ on qubit 0:

$$H \vert 0 \rangle = \frac{1}{\sqrt{2}}(\vert 0 \rangle + \vert 1 \rangle)$$

$$\Rightarrow \vert \psi \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 10 \rangle).$$

Apply CNOT $0 \to 1$:

$$\vert \Phi^+ \rangle = \frac{1}{\sqrt{2}}(\vert 00 \rangle + \vert 11 \rangle).$$

Schmidt values across the cut are $\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$ → entropy $S = 1$ bit.

MPS needs $\chi \geq 2$ to represent this exactly.

### 13.2 Two-site gate update (shapes)

- Suppose the left/right bonds are $\chi$.
- Merge → $(2\chi) \times (2\chi)$ matrix.
- Apply $G \in \mathbb{C}^{4 \times 4}$ (acts on the physical 2×2).
- SVD → truncate to $k^* \leq 2\chi$ to respect $\chi_{\max}$ and tolerance.
- Reshape factors back into two MPS cores.

### 13.3 Truncation numerics

Example singular values: $0.80, 0.58, 0.14, 0.05, 0.02, 0.01$.

Energy (squared) sums to $\approx 1$.

With $\varepsilon_{\text{bond}} = 0.05$, keep top $k$ so that cumulative $\geq 1 - 0.05^2 = 0.9975$.

Read the cumulative curve (Figure 4b) to pick $k$; the leftover gives local error.

### 13.4 QFT peaks (toy)

$N = 8, r = 2$. State has spikes at $0, 2, 4, 6$.

QFT gives peaks at $m = \ell \cdot N/r = \ell \cdot 4$ → $m = 0, 4$.

The Dirichlet kernel form makes these peaks precise (Figure 2).

### 13.5 Memory comparison (concrete numbers)

**Why "626,000× compression" is realistic:**

Consider $n = 50$ qubits:
- **Statevector:** $2^{50} \times 16$ bytes $\approx 18$ petabytes
- **MPS with $\chi = 16$:** $50 \times 16^2 \times 32$ bytes $\approx 0.4$ MB
- **Compression ratio:** $\approx 45,000,000×$

Even for $n = 40$ qubits:
- **Statevector:** $2^{40} \times 16$ bytes $\approx 17.6$ TB
- **MPS with $\chi = 32$:** $40 \times 32^2 \times 32$ bytes $\approx 1.3$ MB
- **Compression ratio:** $\approx 13,500,000×$

The key: as long as entanglement stays moderate ($\chi$ small), MPS memory grows linearly while statevector grows exponentially.

## 14. Tuning Knobs (math-guided)

### 14.1 Key Parameters
- **$\chi_{\max}$ (memory):** bytes $\approx 32n\chi_{\max}^2$.
 - Lower = faster/smaller memory, but fewer patterns
 - Higher = more accurate, but slower
- **Tolerance $\varepsilon_{\text{bond}}$ (accuracy):** global $\lesssim \sqrt{M} \varepsilon_{\text{bond}}$ (very rough) with $M$ truncations.
 - Smaller = keep more detail
 - Larger = prune more, go faster
 - **Practical tip:** Start with strict tolerance (e.g., $10^{-6}$). If too slow, relax it gradually.
- **Layout:** Prefer neighbor couplings (1D/2D locality) to keep $\chi$ modest.
- Use stabilizer when Clifford-heavy; handoff to MPS when needed.

### 14.2 When MPS Wins vs. Struggles

**MPS excels at:**
- 1D or 2D circuits with local interactions
- Shallow circuits
- Variational algorithms (VQE/QAOA) with moderate entanglement
- Clifford-heavy circuits (via stabilizer fast path)
- Structured problems with natural locality

**MPS struggles with:**
- Deep random circuits designed for maximum entanglement
- Extensive long-range entangling gates between distant qubits
- Circuits that violate area-law entanglement scaling

**Key insight:** If your problem has natural structure/locality, you're in the sweet spot for massive compression.

## Figures Index

1. **Memory scaling (statevector vs MPS)** — exponential vs near-linear with small $\chi$.
2. **QFT of a periodic state** — Dirichlet-like peaks at multiples of $N/r$.
3. **Entanglement entropy examples** — how spectra map to bits of entanglement.
4. **SVD truncation visuals** — singular values & cumulative energy.

## Closing Mental Model

- **Statevector:** keep everything → $2^n$ blows up.
- **MPS:** keep patterns (links of width $\chi$) → $\approx n\chi^2$ stays tame.
- **Entanglement drives $\chi$:** more drama → bigger $\chi$.
- **Two-qubit gates:** create entanglement; compress via SVD immediately; track error.
- **Clifford?** Take the stabilizer highway.
- **Periodic states + QFT:** rhythms in → peaks out; gcd finishes the job.
- **GPUs:** same math, fewer memory trips.

## Quick Reference: Key Formulas & Rules of Thumb

### Memory Requirements
- **Statevector:** $16 \cdot 2^n$ bytes
- **MPS:** $\approx 32n\chi^2$ bytes
- **Rule:** Every qubit doubles statevector memory; MPS stays linear if $\chi$ controlled

### Practical Limits
- **30 qubits statevector:** ~16 GB (desktop limit)
- **40 qubits statevector:** ~16 TB (server farm)
- **50 qubits statevector:** ~18 PB (impossible)
- **100+ qubits MPS with $\chi \leq 64$:** Often < 100 MB

### Bond Dimension Guidelines
- $\chi = 1$: Product states (no entanglement)
- $\chi \leq 8$: Weakly entangled states
- $\chi = 16-32$: Moderate entanglement (sweet spot for many algorithms)
- $\chi = 64-128$: Significant entanglement (still tractable)
- $\chi > 256$: Approaching limits, consider if MPS is right approach

### Error Management
- **Local truncation:** $\varepsilon_{\text{local}} = \sum_{s>\text{kept}} \sigma_s^2$
- **Global bound:** $\varepsilon_{\text{global}} \leq \sqrt{\sum_m \varepsilon_m^2}$
- **Practical starting point:** $\varepsilon_{\text{bond}} = 10^{-6}$

### Performance Tips
- **Circuit design:** Keep gates between neighbors when possible
- **Stabilizer circuits:** Can be ~20× faster than MPS
- **GPU advantage:** Most pronounced for $\chi \geq 32$ and two-qubit heavy circuits
- **Compression achieved:** Often $10^6$-$10^8$× for structured problems with $n > 40$