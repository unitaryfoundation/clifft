# UCC — Use Cases & Application Workflows

This document outlines the core use cases enabled by the Unitary Compiler Collection (UCC) framework. By explicitly factoring universal quantum circuits into a deterministic coordinate frame and a localized, probabilistic Active Statevector ($|\psi\rangle = \gamma U_C P (|\phi\rangle_A \otimes |0\rangle_D)$), UCC achieves a strict separation of $\mathcal{O}(n^2)$ topological tracking from $\mathcal{O}(1)$ execution.

This multi-level Ahead-Of-Time (AOT) pipeline allows UCC to function simultaneously as an ultra-fast Virtual Machine, an algebraic optimizer, and a hardware routing compiler. For each use case, we define the integration point within the pipeline, the required functionality, relevant references, and our demonstration strategy.

---

## 1. Ground-Truth Simulation of Magic State Cultivation (MSC)

This is already underway in [magic.plan.md](./magic.plan.md).
### Overview

Generating high-fidelity logical $|T\rangle$ states is a massive bottleneck in fault-tolerant quantum computing. Evaluating exact non-Clifford cultivation circuits at scale has historically been computationally intractable. With the Factored State architecture, the Active array size dynamically expands during $T$-gates and physically compacts during measurements. For circuits where non-Clifford entanglement is aggressively cooled by syndrome checks, the peak Active dimension ($k_{\max}$) remains bounded, enabling exact simulation at massive physical scales.

- **Program Representation Level:** Level 4 (VM Runtime). This exercises the full AOT pipeline, compiling the circuit down to localized RISC bytecode and sampling it over millions of shots.
- **Minimum Functionality Needed:**
  - *Wide Pauli Trackers:* The circuit scales to hundreds of qubits, requiring template monomorphization of the Pauli Frame ($P$) trackers (`p_x`, `p_z`) to use `stim::bitword<512>` and AVX-512 instructions.
  - *Array Compaction:* When an Active qubit is measured, the Back-End must emit localized virtual `SWAP` instructions to route the target axis to the top of the array, allowing the VM to physically halve the continuous statevector memory ($k \to k-1$) without strided fragmentation.
  - *Fast-Fail Post-Selection:* The `OP_POSTSELECT` instruction is strictly required. Distillation circuits have extreme discard rates under noise. Aborting the SVM loop the moment a parity check fails prevents wasting deep non-Clifford FLOPs on doomed logical runs.
- **References:**
  - Gidney, Shutty, Jones (2024): "Magic state cultivation: growing T states as cheap as CNOT gates"

#### Demonstration & Evaluation Strategy

We target the exact 463-qubit, trillion-shot curve required to escape the physical noise floor.
To survive processing $10^{12}$ shots without Out-Of-Memory (OOM) crashes, we deploy **Dense Survivor Sampling**. The C++ core strictly limits memory allocation to shots that survive post-selection. By passing Sinter's `postselection_mask` into the UCC compiler, doomed shots instantly fast-fail via `OP_POSTSELECT`. With a $>99\%$ discard rate, a 10-million shot batch returns a tiny megabyte-scale array to Python instead of a massive gigabyte blowout, allowing seamless distributed orchestration via AWS spot clusters.

---

## 2. Best-in-Class Noise-Aware T-Count Reduction (FastTODD)
This is already underway in [optimizer.plan.md](./optimizer.plan.md).
### Overview

Minimizing non-Clifford gate counts (such as the T-count) is a primary objective in quantum compilation. State-of-the-art parity-table algorithms (such as FastTODD) achieve mathematically optimal reductions by mapping subcircuits to phase polynomials and solving the Third-Order Symmetric Tensor Rank (3-STR) problem.

However, traditional solvers suffer from two structural bottlenecks:
1. **The Ancilla Penalty:** To gain global visibility on mixed-Clifford circuits, they force "Hadamard Gadgetization," massively bloating the physical ancilla qubit count.
2. **The Synthesis Penalty:** Converting an optimized phase polynomial back into a physical unitary circuit requires synthesizing complex CNOT networks, destroying hardware routing constraints.

**UCC natively bypasses both bottlenecks.** Because the Front-End mathematically rewinds all non-Clifford operations to the $t=0$ vacuum natively, the resulting Heisenberg IR (HIR) is inherently a state-agnostic, global phase polynomial. By embedding the FastTODD algorithm directly into the UCC Middle-End, we can optimize the HIR and either export it directly to hardware routers or lower it to the SVM—never synthesizing a CNOT ladder.

- **Program Representation Level:** Level 2 (Optimized Heisenberg IR).
- **Minimum Functionality Needed:** Front-End HIR emission, $\mathcal{O}(1)$ symplectic inner product checks, and the FastTODD $\mathcal{O}(n^4 m^3)$ 3-STR null-space extraction pass.
- **References:** Vandaele (2025): "Lower T-count with faster algorithms" (arXiv:2407.08695v2).

### The Algorithm: State-Agnostic FastTODD

Because T-gates generate phase polynomials over $\mathbb{Z}_8$, linear dependencies must be found in the degree-3 Reed-Muller code space.

1. **Native Slicing:** The Middle-End sweeps the HIR, grouping `T_GATE` nodes into cliques using the $\mathcal{O}(1)$ popcount symplectic inner product of their rewound masks.
2. **Noise-Aware Barrier Enforcement:** The sweep halts strictly at any `MEASURE` or `NOISE` node that anti-commutes with the active clique. This mathematically guarantees that optimization never conjugates physical error models into new bases, making the compiler strictly FTQC-safe.
3. **Null-Space Extraction:** For a clique of $m$ T-gates, the optimizer constructs a binary matrix $L$ encoding pairwise ANDs of the spatial masks, and extracts the right null-space to find linear redundancies.
4. **Organic Clifford Absorption:** Redundant T-gates are cancelled. Residual lower-order phases form Clifford $S$ gates. Rather than attempting complex algebraic fold-back, the optimizer simply emits generic `CLIFFORD` nodes. The Back-End's $V_{cum}$ virtual frame compressor mathematically absorbs these new Cliffords natively during bytecode lowering.

---

## 3. End-to-End Hardware Execution Integrations

### Overview

While UCC acts as an ultra-fast simulator, its AOT architecture uniquely positions it as a powerful compiler frontend for physical hardware. By flattening the circuit into an algebraically independent list of multi-qubit Pauli operations (the Optimized HIR), UCC breaks away from rigid DAG-based routing. Translating this mathematically pure IR back into physical hardware requires architectures that natively support high-weight Pauli operations or feature flexible connectivity.

- **Program Representation Level:** Level 2 (Optimized HIR Export).
- **Minimum Functionality Needed:** Export of the HIR to JSON or text, capturing explicit Pauli masks, complex weights, and the `global_weight` accumulator.

### 3.1 The NISQ Target: Trapped Ions (Quantinuum H2)
Trapped ions utilize an all-to-all QCCD architecture. Their native entangling gates map naturally to multi-qubit Pauli exponentials ($e^{-i \theta (Z \otimes Z \dots)}$).
Standard compilers get trapped in local minima trying to commute rotations through dense Clifford networks, synthesizing massive CNOT ladders. The UCC Front-End *absorbs 100% of the Clifford routing*, revealing the true global phase polynomial. We export the Optimized HIR as a minimal list of pure Phase Gadgets. The Quantinuum compiler folds these gadgets directly into native $ZZPhase$ entangling pulses with zero SWAP overhead.

### 3.2 The FTQC Target: Surface Codes (tqec)
Modern superconducting FTQC architectures execute logic via Pauli-Based Computation (PBC) using lattice surgery, where a weight-15 logical Pauli string can be measured in the exact same logical time step as a weight-1 string.
Translating IRs into ZX-calculus graphs before lowering to lattice surgery is unnecessary with UCC. Because the Optimized HIR is *already* a mathematically pure list of multi-qubit Pauli correlation operations, it acts as the native AST for lattice surgery routers like `tqec`.

---

## 4. Error Mitigation: Pauli Twirling & Noise Characterization

### Overview

Error mitigation strategies (like Pauli twirling or Probabilistic Error Cancellation) convert coherent errors into stochastic Pauli noise by inserting random Pauli gates into the circuit. Simulating complex twirled noise models typically crushes standard dynamic simulators due to the sheer volume of random branching and matrix updates.

- **Program Representation Level:** Level 4 (VM Runtime).
- **Minimum Functionality Needed:** The `OP_APPLY_PAULI` localized RISC instruction and Constant Pool mask fetching.

#### Demonstration Strategy
In the Factored State architecture ($|\psi\rangle = \gamma U_C P (|\phi\rangle_A \otimes |0\rangle_D)$), stochastic Pauli noise is completely decoupled from the continuous Active Statevector ($|\phi\rangle_A$). During AOT compilation, twirled Pauli channels are mapped to $n$-bit masks. At runtime, the VM executes `OP_APPLY_PAULI`, which fetches the mask from the Constant Pool and XORs it directly into the 1D Pauli Frame trackers (`p_x`, `p_z`). Because this requires zero floating-point operations and completely ignores the dense Active array, executing a heavily twirled circuit in UCC takes effectively the exact same amount of time as simulating the bare circuit.

---

## 5. Variational Quantum Algorithms (VQAs / QAOA) at Scale

### Overview

Variational algorithms (VQE, QAOA) require executing the exact same circuit topology thousands of times, iteratively updating the rotation angles of continuous non-Clifford gates (e.g., $R_{ZZ}(\theta)$) via a classical optimizer. Standard simulators must re-compile the circuit or re-traverse the AST for every angle change, creating unacceptable latency loops.

- **Program Representation Level:** Level 4 (VM Bytecode Mutation).
- **Minimum Functionality Needed:** Dominant Term Factoring in the Front-End, `OP_PHASE_LCU` opcodes, and inline `weight_re` / `weight_im` structs within the 32-byte VM Instruction.

#### Demonstration Strategy

In UCC, the geometric structure and basis compression of a VQA circuit never changes—only the complex weights do. The Back-End geometrically compresses arbitrary continuous rotations into localized virtual axes operations. Crucially, the relative complex weights of these rotations are stored *inline* directly inside the 24-byte payload union of the `OP_PHASE_LCU` bytecode instruction.

We expose a Python API to iterate over the contiguous C++ `Program.bytecode` array and directly mutate the floats in-place. This bypasses AOT compilation entirely across training epochs. The in-memory payload mutation drops per-iteration compilation latency to exactly zero while preserving optimal L1 cache alignment for the execution hot-loop, achieving unprecedented simulation speeds for VQA parameter sweeps.

## 6. Correct-by-Construction FT Gadget Synthesis & Verification
### Overview
Designing and optimizing fault-tolerant (FT) gadgets—such as syndrome extraction circuits or logical state preparations—is traditionally a process of heuristic trial-and-error followed by heavy post-hoc verification. The 2025 paritea framework (Rodatz et al.) revolutionized this by formalizing "$w$-fault-equivalence," allowing idealized mathematical gadgets to be refined into physical circuits that are guaranteed to preserve their error-correcting properties by construction.

However, applying this via graphical ZX-calculus assumes infinite topological flexibility (ignoring hardware connectivity constraints, directed time, and gate scheduling) and relies on expensive graph traversals. UCC natively solves this by executing $w$-fault-equivalence algebraically. Because the UCC Ahead-Of-Time (AOT) compiler already tracks the deterministic geometric propagation of Pauli operators via the Heisenberg IR (HIR), it can instantly compute gadget distances, enforce strict hardware routing, and dynamically minimize simulation overhead ($k_{\max}$).

- **Program Representation Level**: Level 1 (Front-End AOT) & Level 2 (Optimized Heisenberg IR).
- **Minimum Functionality Needed**: Phase 1 (MVP) for $\mathcal{O}(1)$ symplectic inner products and native detecting set extraction via dormant measurements (OP_MEAS_STATIC); Phase 2 (Control Flow & Noise) to inject adversarial HIR_NOISE nodes for offline distance checking.

References:
Rodatz, Poór, Kissinger (2025): "Fault Tolerance by Construction" (and the associated paritea tool).

### The Algorithm: Algebraic FT Synthesis & Distance Checking
UCC implements fault-equivalence checks purely through symplectic geometry and deterministic frame updates, completely bypassing graphical Pauli-web traversals:

1. Ultra-Fast $\mathcal{O}(1)$ Distance Verification: To find the exact fault distance $d$ of a compiled gadget, the UCC compiler injects adversarial physical errors (weight-$w$ HIR_NOISE nodes) and maps them to the virtual basis via the offline Clifford frame: $\tilde{E} = (U_C^{(t)})^\dagger E_{\text{lab}} U_C^{(t)}$. A fault is verified as detectable offline if it anti-commutes with a deterministic dormant measurement (HIR_MEASURE where $v \in D$). UCC evaluates the entire parity-check null space algebraically using $\mathcal{O}(1)$ bitwise symplectic inner products.
2. FT-Safe Global Optimization: Traditional logical optimizers destroy physical error models by blindly commuting non-Clifford gates past stochastic barriers. UCC explicitly makes its optimization pass FT-safe. The compiler will only permit Clifford absorptions, gate cancellations, and node commutations if the resulting mutated HIR graph is mathematically proven to remain $d$-fault-equivalent to the original target.
3. Hardware-Aware Constraint Filtering: When unrolling a theoretical fault-equivalent rewrite into a physical circuit, UCC acts as a strict hardware filter. It instantly rejects any candidate rewrite whose virtual basis transformations require physical interactions outside of the target QPU's specific connectivity graph or violates strict time-domain stochastic barriers.
4. $k_{\max}$ Minimization (Simulation Optimization): Because there are many topologically distinct ways to unroll a fault-tolerant gadget, the UCC synthesizer uses $k_{\max}$ (the peak number of simultaneously active virtual qubits) as its primary optimization cost function. By discovering the specific physical arrangement that uncomputes active qubits as early as possible, UCC minimizes the spatial spread of non-Clifford entanglement, exponentially reducing the memory and runtime required by the VM.
### Demonstration & Evaluation Strategy
We will validate UCC as a state-of-the-art FT synthesis engine through a two-step demonstration:
The Verification Benchmark (vs. paritea / Stim): We will replicate the distance calculations for standard Steane-style and Shor-style syndrome extraction circuits. We will show that UCC evaluates the exact same undetectable fault null-spaces as paritea, but does so orders of magnitude faster by utilizing dense bitwise matrix operations over the HIR instead of topological graph traversals.
The Constrained Synthesis Benchmark: We will input an idealized logical state preparation (e.g., the Steane $|0\rangle$ state) and a target distance $d$. We will task UCC with synthesizing a physical circuit constrained to a heavy-hex hardware topology. We will prove that UCC can autonomously output a valid, $d$-fault-equivalent physical circuit that not only meets the strict hardware routing requirements but also achieves a strictly minimal $k_{\max}$ footprint for ultra-fast classical simulation.
