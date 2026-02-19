
# UCC — Use Cases & Application Workflows

This document outlines the core use cases enabled by the Universal Compiler Collection (UCC) framework. Because UCC fundamentally decouples the $\mathcal{O}(n^2)$ topological tracking of a quantum circuit from its $\mathcal{O}(1)$ probabilistic execution via a multi-level Ahead-Of-Time (AOT) compiler, it exposes multiple layers of program representation. This allows UCC to function simultaneously as an ultra-fast simulator, an algebraic optimizer, and a hardware routing compiler.

For each use case, we define the integration point within the pipeline, the minimum functionality required for implementation, relevant references, and our strategy for demonstrating state-of-the-art performance.

---

## 1. Ground-Truth Simulation of Magic State Cultivation (MSC)

### Overview

Generating high-fidelity logical $|T\rangle$ states is a massive bottleneck in fault-tolerant quantum computing. Evaluating the exact non-Clifford cultivation circuit at scale (e.g., code distance $d=5$, involving 42 qubits and 72 $T/T^\dagger$ gates) has historically been computationally intractable.

- **Program Representation Level:** Level 4 (VM Runtime). This exercises the full AOT pipeline, compiling the circuit down to hardware-aligned bytecode and sampling it over millions of shots.
- **Minimum Functionality Needed:**
  - *Phase 1 (MVP) Fast-Path:* Because the $d=5$ MSC circuit uses exactly 42 qubits, it fits entirely within the MVP's $\le 64$ qubit inline uint64\_t execution path.
  - *Phase 2 (Control Flow & Noise):* OP\_POSTSELECT is strictly required. The $d=5$ circuit has a $>99\%$ discard rate under noise. Aborting the VM loop the moment a parity check fails—rather than evaluating the remaining non-Clifford branches—is essential for tracking scale. Geometric gap sampling (NoiseSchedule) is also required to implement the uniform depolarizing models used in the papers.
- **References:**
  - Gidney, Shutty, Jones (2024): "Magic state cultivation: growing T states as cheap as CNOT gates"
  - Li et al. (2025): "SOFT: a high-performance simulator for universal fault-tolerant quantum circuits"

#### Demonstration & Evaluation Strategy

We will use a two-step benchmarking strategy to prove both correctness and absolute performance:

1. **The S-Gate Baseline (vs. Stim / Gidney):** Gidney evaluated $d=5$ cultivation by replacing $T$ gates with $S$ gates to make it simulatable in Stim, and extrapolated the results. We will simulate this exact $S$-gate proxy circuit through both Stim and UCC.
   - *Correctness:* Because $S$-gates are Cliffords, UCC's Front-End will absorb them entirely. If our logical error rates and discard rates perfectly match Stim, it mathematically proves our trace generation, Heisenberg rewinding, and gap-sampled noise models are flawless before introducing $T$-gate complexity.
   - *Speed Benchmark:* Since the AOT compiler eliminates all tableau math, the VM simply executes fast integer XORs for noise and measurements. This will benchmark our VM's raw execution speed against stim::TableauSimulator's dynamic tracking.
2. **The True T-Gate Benchmark (vs. SOFT):** The SOFT simulator recently achieved the first exact simulation of the $d=5$ $T$-gate circuit, but it required 16 NVIDIA H800 GPUs running for 20 days to collect 200 billion shots. We will run the true $T$-gate circuit end-to-end. By leveraging *Dominant Term Factoring* (where the heaviest identity branch requires zero FLOPs and zero memory writes) and OP\_POSTSELECT, we aim to show that UCC running on a conventional CPU cluster can achieve dramatically higher throughput than SOFT's GPU brute-force approach.

---

## 2. Physical Circuit Optimization & Last-Mile Compilation

### Overview

Minimizing non-Clifford gate counts (such as the $T$-count) is a primary objective in quantum compilation. Finding the absolute theoretical minimum $T$-count for an arbitrary abstract quantum circuit is an NP-hard problem, typically addressed by offline logical synthesis tools utilizing phase polynomials or ZX-calculus graph rewriting (e.g., PyZX, tket). While mathematically optimal for pure, unitary algorithms, these tools frequently face computational bottlenecks and semantic limitations when applied to the stochastic constraints of physical fault-tolerant quantum computing (FTQC) circuits.

UCC approaches this optimization with a different objective. The UCC Middle-End acts as an $\mathcal{O}(1)$ greedy, algebraic cancellation engine. It is not designed to replace offline logical synthesis, but rather to complement it as a noise-aware, "last-mile" physical compiler. Operating natively in the Heisenberg picture, UCC functions as a global peephole optimizer that sees through Clifford obfuscation to identify and eliminate topological redundancies introduced during physical quantum error correction (QEC) routing, while strictly respecting stochastic memory barriers.

Furthermore, the Heisenberg IR (HIR) provides a flat, algebraically independent representation of the circuit. While the current UCC optimizer implements a fast, single-pass sweep, the HIR serves as a highly structured foundation upon which future researchers could build more exhaustive, globally optimal synthesis algorithms—acting as a highly efficient mathematical alternative to ZX-diagrams. Developing such algorithms, however, remains outside the scope of the immediate UCC MVP.

- **Program Representation Level:** Level 2 (Optimized Heisenberg IR). The circuit is parsed into the AST, passed through the Front-End to generate the HIR, and then optimized by the Middle-End. It is *not* lowered to VM bytecode.
- **Minimum Functionality Needed:** Phase 1 (MVP). Requires Front-End HIR emission, and the Middle-End's $\mathcal{O}(1)$ bitwise commutation, fusion, and cancellation passes using symplectic inner products. Must include rigorous barrier awareness (MEASURE and NOISE).
- **References:** `op-T-mize` dataset (Kottmann, 2025); PyZX (Kissinger & van de Wetering, 2020); TODD Algorithm (Heyfron & Campbell, 2017).

#### Demonstration & Evaluation Strategy

We will evaluate the UCC optimizer by characterizing its performance profile across both logical abstraction layers and physical FTQC regimes:

1. **The Speed vs. Optimality Trade-off (`op-T-mize`):** We will process the standard `op-T-mize` dataset through UCC and compare compilation times and $T$-count reductions against tools like TODD and PyZX. Because UCC relies on exact Pauli matches rather than phase polynomial re-synthesis, it will mathematically leave some multi-axis cancellations unexploited. However, this benchmark will explicitly quantify the value of the trade-off: UCC trades absolute theoretical optimality for execution speeds that are orders of magnitude faster than graph-rewriting methods, establishing its utility as a Just-In-Time (JIT) heuristic.
2. **Preservation of Physical Error Models:** Purely logical synthesizers are designed for unitary circuits and can inadvertently invalidate physical error models—for instance, by commuting a $T$ gate past an $X$-error channel, thereby conjugating the error into a coherent rotation. We will process a noisy physical circuit (e.g., a magic state distillation factory) to demonstrate that UCC safely cancels redundant non-Clifford gates separated by commuting noise or measurements, but strictly halts at anti-commuting barriers, preserving the physical validity of the simulation.
3. **Scalability in the Fault-Tolerant Regime:** Heavy topological synthesizers frequently experience exponential slowdowns or memory exhaustion when processing fully unrolled QEC circuits. We will take a logical block that has been provably optimized by PyZX, route it into a physical surface code via Pauli-Based Computation (PBC), and feed the unrolled physical trace into UCC. We will demonstrate UCC sweeping and fusing the boundary redundancies introduced by the routing process in milliseconds, highlighting its viability as a scalable physical compiler.

#### Metric & Proving Equivalence

- **Metric (Re-synthesis vs. Counting):** Synthesizing optimized Pauli operations back into a standard `.stim` or OpenQASM text file requires solving the Clifford synthesis problem, which adds unnecessary computational overhead for modern FTQC execution workflows. Therefore, our standard metric will simply be the count of non-Clifford operations (`hir.num_gate_ops`) before and after optimization.
- **Proving Equivalence to Readers:** To mathematically guarantee that the $\mathcal{O}(1)$ optimized HIR is functionally equivalent to the original circuit without reverse-synthesis, we will employ two automated proofs:
  1. *Statevector Oracle (for circuits $\le 20$ qubits):* We compile the unoptimized input circuit into a dense statevector using our Python prototype, then run the Optimized HIR through the UCC Back-End and execute 1 shot on the VM. We assert that `np.allclose(expanded_vm_state, pure_python_state * global_weight)` holds true.
  2. *Inverse Annihilation (for large circuits):* We concatenate the Optimized HIR with the *inverse* of the unoptimized original circuit. We run this combined program through the UCC VM. If the optimization is flawlessly correct, the VM must deterministically measure the vacuum state $|00..0\rangle$ with exactly 100% probability.

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

Instead of twirling the circuit text (which exponentially bloats the number of circuits to track), the user compiles the circuit *once*. During code generation, the Back-End compiles the twirled Pauli channels into the NoiseSchedule. The VM then samples these errors via branchless geometric gap sampling. We will demonstrate that executing a heavily twirled circuit in UCC takes effectively the exact same amount of time as simulating the bare circuit, fundamentally solving the simulation bottleneck for error mitigation research.

---

## 5. Variational Quantum Algorithms (VQAs / QAOA) at Scale

### Overview

Variational algorithms (VQE, QAOA) require executing the exact same circuit topology thousands of times, iteratively updating the rotation angles of non-Clifford gates (e.g., $R_Z(\theta)$). Standard simulators must re-compile the circuit or re-traverse the AST for every angle change.

- **Program Representation Level:** Level 4 (VM Constant Pool Mutation).
- **Minimum Functionality Needed:** Phase 2 & 3. Generic GATE OpType in the HIR, LCU fast-math payloads in the Back-End, and the OP\_BRANCH\_LCU opcodes. Template monomorphization (>64 qubit support) is needed for scale.

#### Demonstration Strategy

In UCC, the geometric structure of a VQA circuit never changes—only the complex weights do. The Back-End lowers arbitrary rotations into pre-computed memory shifts (x_mask) and Constant Pool references. We can expose a Python API to directly mutate the floating-point values in the ConstantPool (lcu_pool). This bypasses AOT compilation entirely across training epochs, dropping the per-iteration compilation latency to zero and achieving unprecedented simulation speeds for VQA parameter sweeps.
