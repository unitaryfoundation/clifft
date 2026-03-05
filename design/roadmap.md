# UCC Paper 1 Roadmap: Shattering the Exponential Walls of FTQC

## The Core Narrative
Every exact quantum simulator hits an exponential complexity wall. Statevector simulators scale as $\mathcal{O}(2^N)$ (The Physical Wall). ZX-Calculus and Stabilizer-Rank simulators scale as $\mathcal{O}(2^{\alpha t})$ (The Gate Wall).

UCC introduces the **Factored State Architecture**, which mathematically decouples geometric topology from probability. This shifts the exponential wall strictly to the *peak active rank* ($k_{\max}$). Because Fault-Tolerant Quantum Computing (FTQC) naturally bounds $k_{\max}$ via continuous syndrome measurements, UCC easily bypasses both traditional walls. We prove this architecture acts as a true compiler by natively fusing non-Clifford operations to dynamically shrink the VM's memory footprint, culminating in the exact simulation of a 463-qubit Magic State Cultivation protocol over a trillion shots on standard CPU cores.

---

## Phase 1: Standard Gate Set Expansion
**Reference:** `design/missing_gates.plan.md`
**Goal:** Implement the vast majority of Stim's standard library to ensure UCC is a robust, drop-in replacement for any benchmark circuit.
* **Tasks:**
  * Implement Stim parser aliases (`ZCX` -> `CX`, etc.) and No-Ops (`I`, `I_ERROR`).
  * Implement classical padding (`MPAD`) and Syntactic Sugar (Pair measurements like `MXX`, `MZZ`, and Y-basis resets `RY`, `MRY`).
  * Implement the generic Clifford expansion (30+ gates) using a generic Stim tableau inverse fallback to absorb them all AOT.
  * Implement multi-parameter noise channels (`PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`) required by Sinter's noise models.

## Phase 2: The Toolchain Proof (Lite Optimizer)
**Reference:** `design/optimizer.plan.md`
**Goal:** Prove UCC is a *compiler*, not just a simulator. We establish that the Heisenberg IR (HIR) allows for state-agnostic algebraic reductions that physically shrink the VM's memory footprint.
* **Tasks:**
  * Expand the HIR to support `CLIFFORD_PHASE` nodes.
  * Implement the `PassManager` and `PeepholeFusionPass`.
  * Scan the HIR left-to-right to algebraically fuse adjacent $T$ and $T^\dagger$ gates acting on the same rewound Pauli axis into $S$ or $S^\dagger$ gates.
  * Implement the Back-End lowering so the compiler mathematically absorbs these new Cliffords into the offline coordinate frame at zero runtime cost.
  * *Deferred:* Complex $\mathcal{O}(n^4)$ FastTODD / TOHPE GF(2) linear algebra solvers.

## Phase 3: The Three Walls Benchmarks
**Reference:** `design/performance_comparison.plan.md`
**Goal:** Visually and mathematically prove the core thesis of the paper against industry standards (Qiskit/Qulacs for Statevector, `tsim` for Stabilizer-Rank).
* **Tasks:**
  * Build the Phase Space Generator to isolate $N$, $t$, and $k$ scaling.
  * Build the unified, isolated subprocess runner to safely capture exact compilation times, execution times, and Peak Memory (RAM/VRAM) usage.
  * Execute Panel A ($N$ Wall), Panel B ($k$ Wall), Panel C ($t$ Wall), and Panel D (FTQC Throughput).
  * Generate the definitive 4-panel matplotlib dashboard.

## Phase 4: The FTQC Climax (Magic State Cultivation)
**Reference:** `design/magic.plan.md`
**Goal:** Execute a world-class physics benchmark that previously required a 16-GPU cluster, entirely on standard CPU cores, proving UCC's massive scale capabilities.
* **Tasks:**
  * Scale the C++ core to 512 qubits using `stim::bitword<W>` templates and AVX-512 vectorization.
  * Implement Sinter-native fast-fail compilation (`OP_POSTSELECT`) to instantly abort doomed shots.
  * Implement Dense Survivor Sampling ($\mathcal{O}(1)$ Discard Memory) to safely return data to Python without RAM blowouts.
  * Execute the 42-qubit SOFT baseline to prove $\mathcal{O}(1)$ CPU superiority over GPUs.
  * Execute the 463-qubit, trillion-shot end-to-end circuit via AWS spot instances.

---

## Explicitly Deferred to Future Work (Paper 2)
To maintain strict narrative focus and bound engineering time, the following are excluded from this release:
* **Arbitrary Coherent Noise Simulation** (`coherent_noise.plan.md`)
* **Global T-Count Reduction** (`optimizer.plan.md` Phases 2-6)
* **WebAssembly Compiler Explorer** (`explorer.plan.md`)

> **STATUS: DEFERRED (POST-PAPER 1)**
> This plan is explicitly excluded from the debut UCC paper to maintain narrative focus and bound engineering time. The core narrative of Paper 1 relies on the fact that FTQC measurements naturally cool the active dimension ($k_{\max}$). Coherent noise introduces continuous rank explosion that contradicts this clean narrative. This will form the basis of a dedicated follow-up paper.
