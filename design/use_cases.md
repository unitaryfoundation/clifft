
# UCC — Use Cases & Application Workflows

This document outlines the core use cases enabled by the Unitary Compiler Collection (UCC) framework. Because UCC fundamentally decouples the $\mathcal{O}(n^2)$ topological tracking of a quantum circuit from its $\mathcal{O}(1)$ probabilistic execution via a multi-level Ahead-Of-Time (AOT) compiler, it exposes multiple layers of program representation. This allows UCC to function simultaneously as an ultra-fast simulator, an algebraic optimizer, and a hardware routing compiler.

For each use case, we define the integration point within the pipeline, the minimum functionality required for implementation, relevant references, and our strategy for demonstrating state-of-the-art performance.

---

## 1. Ground-Truth Simulation of Magic State Cultivation (MSC)

### Overview

Generating high-fidelity logical $|T\rangle$ states is a massive bottleneck in fault-tolerant quantum computing. Evaluating the exact non-Clifford cultivation circuit at scale (e.g., code distance $d=5$, involving 42 qubits and 72 $T/T^\dagger$ gates) has historically been computationally intractable.

- **Program Representation Level:** Level 4 (VM Runtime). This exercises the full AOT pipeline, compiling the circuit down to hardware-aligned bytecode and sampling it over millions of shots.
- **Minimum Functionality Needed:**
  - *Phase 1 (MVP) Fast-Path:* Because the $d=5$ MSC circuit uses exactly 42 qubits, it fits entirely within the MVP's $\le 64$ qubit inline uint64\_t execution path.
  - *Phase 2 (Control Flow & Noise):* OP\_POSTSELECT is strictly required. The $d=5$ circuit has a $>99\%$ discard rate under noise. Aborting the SVM loop the moment a parity check fails—rather than evaluating the remaining non-Clifford branches—is essential for tracking scale. Geometric gap sampling (NoiseSchedule) is also required to implement the uniform depolarizing models used in the papers.
- **References:**
  - Gidney, Shutty, Jones (2024): "Magic state cultivation: growing T states as cheap as CNOT gates"
  - Li et al. (2025): "SOFT: a high-performance simulator for universal fault-tolerant quantum circuits"

#### Demonstration & Evaluation Strategy

We will use a two-step benchmarking strategy to prove both correctness and absolute performance:

1. **The S-Gate Baseline (vs. Stim / Gidney):** Gidney evaluated $d=5$ cultivation by replacing $T$ gates with $S$ gates to make it simulatable in Stim, and extrapolated the results. We will simulate this exact $S$-gate proxy circuit through both Stim and UCC.
   - *Correctness:* Because $S$-gates are Cliffords, UCC's Front-End will absorb them entirely. If our logical error rates and discard rates perfectly match Stim, it mathematically proves our trace generation, Heisenberg rewinding, and gap-sampled noise models are flawless before introducing $T$-gate complexity.
   - *Speed Benchmark:* Since the AOT compiler eliminates all tableau math, the SVM simply executes fast integer XORs for noise and measurements. This will benchmark our SVM's raw execution speed against stim::TableauSimulator's dynamic tracking.
2. **The True T-Gate Benchmark (vs. SOFT):** The SOFT simulator recently achieved the first exact simulation of the $d=5$ $T$-gate circuit, but it required 16 NVIDIA H800 GPUs running for 20 days to collect 200 billion shots. We will run the true $T$-gate circuit end-to-end. By leveraging *Dominant Term Factoring* (where the heaviest identity branch requires zero FLOPs and zero memory writes) and OP\_POSTSELECT, we aim to show that UCC running on a conventional CPU cluster can achieve dramatically higher throughput than SOFT's GPU brute-force approach.

---

## 2. Best-in-Class Noise-Aware T-Count Reduction ("Heisenberg-TOHPE")

### Overview

Minimizing non-Clifford gate counts (such as the T-count) is a primary objective in quantum compilation. The 2025 state-of-the-art parity-table algorithms (such as the TOHPE and FastTODD procedures) achieve mathematically optimal reductions by mapping subcircuits to phase polynomials and solving the Third-Order Symmetric Tensor Rank (3-STR) problem.

However, these traditional solvers suffer from two fatal structural bottlenecks:

1. **The Ancilla & Partitioning Penalty:** To gain global visibility on mixed-Clifford circuits, they force "Hadamard Gadgetization," massively bloating the physical ancilla qubit count. Without gadgets, compilers must rely on bulky graph-partitioning heuristics to chop the circuit into Hadamard-free zones.
2. **The Extraction / Synthesis Penalty:** Converting an optimized phase polynomial back into a physical, executable unitary circuit requires synthesizing complex CNOT networks. This destroys hardware routing constraints and inflates circuit depth.

**UCC natively bypasses both bottlenecks.** Because the Front-End's TableauSimulator mathematically *absorbs* all Clifford gates and rewinds everything to the $t=0$ topological basis, the resulting Heisenberg IR (HIR) is natively a global phase polynomial. By embedding the TOHPE algorithm directly into the UCC Middle-End, we can optimize the HIR and export it directly to Pauli-Based Computation (PBC) hardware routers like tqec—or execute it directly in the SVM. **We never have to extract a CNOT unitary circuit**.

- **Program Representation Level:** Level 2 (Optimized Heisenberg IR).
- **Minimum Functionality Needed:** Phase 1 (MVP). Requires Front-End HIR emission and the Middle-End applying the TOHPE GF(2) null-space extraction over the pairwise ANDs of the topological masks within commuting cliques.
- **References:** Vandaele (2025): "Lower T-count with faster algorithms" (arXiv:2407.08695v2); op-T-mize benchmark suite.

### The Algorithm: L1-Resident Heisenberg-TOHPE

Because T-gates generate phase polynomials over $\mathbb{Z}_8$, linear dependencies must be found in the degree-3 Reed-Muller code space. UCC implements the TOHPE algorithm natively using CPU bitwise arithmetic:

1. **Native Slicing via Symplectic Physics:** The Middle-End sweeps the HIR sequentially, grouping T_GATE and T_DAG_GATE nodes into "cliques" using the $\mathcal{O}(1)$ popcount symplectic inner product. Unlike competitors that use manual partitioning algorithms, UCC naturally halts cliques when a rewound mask anti-commutes (e.g., a T-gate rewound through a Hadamard becomes an X-mask and naturally breaks the clique).
2. **Noise-Aware Barrier Enforcement:** The sweep halts strictly at any MEASURE or NOISE node (including coherent noise LCU branches) that anti-commutes with the active clique. This mathematically guarantees that the optimization never conjugates physical error models into new bases, making the compiler strictly FTQC-safe.
3. **Build the Pairwise $L$-Matrix:** For a clique of $m$ T-gates, the spatial stab_masks form a parity table $P$. Following TOHPE, the optimizer constructs the binary matrix $L$ where each row is the bitwise AND of two spatial dimensions: $L_{\alpha\beta} = P_\alpha \wedge P_\beta$.
4. **3-STR Null-Space Extraction:** The optimizer runs fast GF(2) Gaussian elimination to find a vector $y$ in the right null-space of $L$ ($Ly = 0$). This identifies a linear dependence that preserves the third-order tensor signature.
5. **Greedy Substitution & Clifford Fold-Back:** Using the extracted $y$ and a deterministically chosen vector $z$, the optimizer substitutes $P' = P \oplus zy^T$. Redundant T_GATE nodes are physically deleted from the HIR, and residual lower-order phases are converted to Clifford S gates, which are flagged to be mathematically absorbed by the Front-End during the final fold-back pass.

### Computational Scaling & The Arbitrary Qubit ($N$) Rescue

Vandaele reduced the classical complexity of TODD to $\mathcal{O}(n^2 m^3)$ for TOHPE. UCC fundamentally alters how this executes in memory, decoupling optimization time from both Clifford depth and physical qubit count.

- **Scaling with Clifford Gates ($G_C$): $\mathcal{O}(1)$**
  In UCC, the Front-End absorbs all $G_C$ Cliffords dynamically. Cliffords literally do not exist in the HIR, decoupling optimization time from Clifford depth entirely.
- **Scaling with Qubits ($N$): Bounded by $\mathcal{O}(m^2)$ (The Algebraic Compression)**
  Naively, the pairwise matrix $L$ has $\binom{N}{2}$ rows, which would cause a catastrophic memory explosion for large $N$. However, because UCC operates in GF(2), the right null-space of $L$ is a mathematical invariant under row-basis projection. Regardless of whether $N = 64$ or $N = 10,000$, the optimizer extracts the linearly independent row-basis of $P$ (which can never exceed $m$, the number of T-gates). The constructed $L_{compressed}$ matrix has at most $\binom{m}{2} + m$ rows.
- **Scaling with Non-Clifford Gates ($m$): Polynomial $\mathcal{O}(m^3)$**
  Finding the null-space of the bounded $L_{compressed}$ matrix takes strictly $\mathcal{O}(m^3)$ bitwise operations. Combined with the "Chunked Rewinding" extension that strictly limits lightcone smearing, the entire working matrix is permanently trapped at a microscopic size, fitting completely inside the CPU's L1 cache and executing in microseconds.

### Demonstration & Evaluation Strategy (The Dual-Pronged Benchmark)

We will process standard benchmark datasets (such as the 2025 Vandaele suites) through UCC and demonstrate state-of-the-art performance by validating three core claims for the MVP release:

1. **The Table 2 Benchmark (Generic, Ancilla-Free Circuits):** On generic, Hadamard-rich circuits, UCC natively matches the best-in-class ancilla-free T-counts. We will prove that while traditional compilers must run complex partitioning heuristics to isolate Hadamard-free zones and synthesize CNOT boundaries, UCC's AOT Front-End naturally isolates non-commuting cliques via pure symplectic geometry in **microseconds**.
2. **The Table 3 Benchmark (Massive Absolute Reductions):** On naturally Hadamard-free arithmetic circuits (e.g., Galois Field Multipliers, $h=0$), the entire circuit forms a massive commuting clique. We will prove that UCC matches the massive absolute T-count reductions of AlphaTensor / FastTODD, with **zero CNOT synthesis penalty** because we route directly to physical PBC instructions.
3. **FTQC-Safe Noise Preservation:** Purely logical synthesizers destroy physical error models by blindly commuting non-Clifford gates past noise. We will inject a parameterized error model (both stochastic Pauli and coherent over-rotations) into a magic-state distillation factory and mathematically prove via our statevector oracle that UCC uniquely optimizes T-counts while perfectly preserving the syndromic structure and physics of the QEC barriers.

---

## 3. Physical Hardware Routing (tqec / Surface Codes)

### Overview

Modern fault-tolerant architectures (like surface/color codes compiled via tools like tqec) do not execute physical $T$ and $H$ gates; they route sequences of multi-qubit Pauli measurements via lattice surgery (Pauli-Based Computation).

- **Program Representation Level:** Level 2 (Optimized HIR Export).
- **Minimum Functionality Needed:** Phase 4. Export of the HirModule to JSON or text, capturing explicit Pauli masks, non-Clifford complex weights, and the global\_weight accumulator.

#### Demonstration Strategy

Traditional workflows force compilers to output deep circuits of CNOTs and T-gates, which the router must then awkwardly reverse-engineer back into Pauli strings. UCC acts as the ultimate "frontend" for hardware routing. We will demonstrate UCC ingesting a messy algorithmic circuit, cancelling redundant T-gates globally, and exporting a clean, barrier-aware list of algebraically independent Pauli operations directly into a router like tqec.

---

## 4. Error Mitigation: Pauli Twirling & Noise Characterization

### Overview

Error mitigation strategies (like Pauli twirling or Probabilistic Error Cancellation) convert coherent errors into stochastic Pauli noise by inserting random Pauli gates into the circuit. Simulating complex twirled noise models typically crushes standard dynamic simulators due to the sheer volume of random branching.

- **Program Representation Level:** Level 3 (Back-End) & Level 4 (VM Runtime).
- **Minimum Functionality Needed:** Phase 2. Multi-Pauli noise channels in the AST and HIR, compiled into the Constant Pool's NoiseSchedule.

#### Demonstration Strategy

Instead of twirling the circuit text (which exponentially bloats the number of circuits to track), the user compiles the circuit *once*. During code generation, the Back-End compiles the twirled Pauli channels into the NoiseSchedule. The SVM then samples these errors via branchless geometric gap sampling. We will demonstrate that executing a heavily twirled circuit in UCC takes effectively the exact same amount of time as simulating the bare circuit, fundamentally solving the simulation bottleneck for error mitigation research.

---

## 5. Variational Quantum Algorithms (VQAs / QAOA) at Scale

### Overview

Variational algorithms (VQE, QAOA) require executing the exact same circuit topology thousands of times, iteratively updating the rotation angles of non-Clifford gates (e.g., $R_Z(\theta)$). Standard simulators must re-compile the circuit or re-traverse the AST for every angle change.

- **Program Representation Level:** Level 4 (VM Constant Pool Mutation).
- **Minimum Functionality Needed:** Phase 2 & 3. Generic GATE OpType in the HIR, LCU fast-math payloads in the Back-End, and the OP\_BRANCH\_LCU opcodes. Template monomorphization (>64 qubit support) is needed for scale.

#### Demonstration Strategy

In UCC, the geometric structure of a VQA circuit never changes—only the complex weights do. The Back-End lowers arbitrary rotations into pre-computed memory shifts (x_mask) and Constant Pool references. We can expose a Python API to directly mutate the floating-point values in the ConstantPool (lcu_pool). This bypasses AOT compilation entirely across training epochs, dropping the per-iteration compilation latency to zero and achieving unprecedented simulation speeds for VQA parameter sweeps.
