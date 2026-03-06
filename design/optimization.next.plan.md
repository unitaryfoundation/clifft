# UCC Implementation Plan: Advanced State-Agnostic Optimization

## Executive Summary & Constraints

This plan builds upon the foundational optimizer infrastructure (which established the `PassManager`, `CLIFFORD_PHASE` HIR nodes, and the basic adjacent `PeepholeFusionPass`).

While state-of-the-art parity-table solvers (like TODD/FastTODD) achieve optimal T-counts via complex $\mathcal{O}(n^4)$ GF(2) linear algebra, they are exceptionally difficult to implement and debug. However, because UCC's Front-End rewinds all operations to the $t=0$ vacuum, the Heisenberg IR (HIR) natively eliminates the physical routing graph (CNOT ladders). This uniquely allows us to implement highly powerful, state-agnostic algebraic passes using simple $\mathcal{O}(n)$ bitwise popcounts that mimic the power of ZX-Calculus rewrites.

This document outlines three advanced, LLM-friendly optimization passes that exploit the HIR to dramatically increase simulation performance and reduce T-counts, culminating in a benchmark against the industry-standard `op-T-mize` dataset.

**Strict Constraints:**

1. **Pre-requisites Assumed:** The basic `PassManager` and `CLIFFORD_PHASE` node support are already implemented and tested.
2. **FTQC Safety (Stochastic Barriers):** None of these passes are permitted to blindly commute non-Clifford gates past stochastic barriers (`MEASURE`, `NOISE`, `DETECTOR`). Every optimization must actively check the $\mathcal{O}(1)$ symplectic inner product against barriers to guarantee the physical error model is perfectly preserved.
3. **Pure Bitwise Math:** Do NOT attempt to implement Gaussian elimination or matrix null-space solvers. Rely strictly on `stim::bitword` XORs, ANDs, and popcounts.
4. **Reproducible:** Benchmarking scripts must live in `paper/optimize/` and generate publication-ready plots comparing UCC to standard tools. This should run standalone and clearly reference which dependencies.

---

## Phase 1: Global Phase Folding (Distance-Agnostic Tunneling)

**Goal:** Implement a distance-agnostic hash map to fold redundant phases across the entire circuit, effortlessly tunneling through intermediate routing operations to collapse deep compute/uncompute ladders.

* **Task 1.1 (The $\mathbb{Z}_8$ Map):** Implement `PhaseFoldingPass` inheriting from `Pass`. Iterate left-to-right through the HIR, building a new `std::vector<HeisenbergOp> optimized_ops`. Maintain a hash map to track active phase gadgets over $\mathbb{Z}_8$. The key is the combined 128-bit `destab_mask` and `stab_mask`. Map operations to their phase values ($T=1, S=2, Z=4, T^\dagger=-1 \pmod 8$).
* **Task 1.2 (Phase Accumulation):** When encountering a phase gate ($T$, $T^\dagger$, or a `CLIFFORD_PHASE`), do not emit it. Add its phase to the map. If the accumulated phase hits $0 \pmod 8$, completely erase the key from the map.
* **Task 1.3 (Stochastic Barriers & Tunneling):** When encountering a barrier node (`MEASURE`, `NOISE`, `DETECTOR`):
1. Iterate through the active masks in the hash map.
2. For each mask, evaluate the $\mathcal{O}(1)$ symplectic inner product against the barrier's mask: `popcount((A.X & B.Z) ^ (A.Z & B.X)) % 2`.
3. If they **anti-commute** (result is 1): The phase cannot cross the barrier. Resolve its $\mathbb{Z}_8$ sum, emit the corresponding phase node to `optimized_ops`, and erase it from the map.
4. If they **commute** (result is 0): Leave it in the map! It mathematically tunnels through the barrier.
5. Emit the barrier node to `optimized_ops`.


* **Task 1.4 (Circuit Flush):** At the end of the HIR sweep, sequentially emit any remaining masks in the hash map.
* **DoD:** A Catch2 unit test feeds `H 0; T 0; CX 0 1; M 1; CX 0 1; T_DAG 0;`. The pass stores `Z0: 1`, ignores `M 1` (because $Z_0$ commutes with $Z_0 \otimes Z_1$), matches the $T^\dagger$ gate, and safely annihilates them both, leaving an HIR with zero phase gates.

## Phase 2: Greedy Spider-Nest Reduction ($\mathcal{O}(m^2)$ Collision Search)

**Goal:** Catch deep multi-controlled Toffoli redundancies (ubiquitous in Galois Field multipliers) by finding 4-term linear dependencies ($P_A \oplus P_B = P_C \oplus P_D$) in the phase polynomial using a fast collision search.

* **Task 2.1 (Clique Extraction):** Implement `SpiderNestPass` inheriting from `Pass`. Sweep the HIR to collect maximal cliques of mutually commuting $T$-gates and $T^\dagger$-gates. Halt clique expansion at any anti-commuting stochastic barriers. Let the clique size be $m$.
* **Task 2.2 (The $\mathcal{O}(m^2)$ Collision Hash):** Within a clique, execute a nested loop over all unique pairs of T-gates $(A, B)$. Store the bitwise XOR of their $t=0$ masks in a hash map: `collision_map[A.mask ^ B.mask] = std::pair(A, B)`.
* **Task 2.3 (Dependency Resolution):** If `collision_map` already contains the XOR mask when evaluating pair $(C, D)$ (and the four gates are strictly disjoint), you have discovered a weight-4 linear dependency: $P_A \oplus P_B \oplus P_C \oplus P_D = 0$.
* **Task 2.4 (Algebraic Substitution):**
1. Verify the $\mathbb{Z}_8$ phase sums of the 4 gates align with known PyZX Spider-Nest identities (e.g., $T \otimes T \otimes T^\dagger \otimes T^\dagger$).
2. Replace these 4 independent T-gates with exactly **2 T-gates** (acting on any two of the original overlapping axes) and the required residual `CLIFFORD_PHASE` ($S$/$Z$) nodes to balance the polynomial.
3. Overwrite the HIR nodes and restart the clique sweep until no collisions remain.


* **DoD:** The optimizer natively parses a decomposed Galois Field multiplier circuit (e.g., `gf2^4_mult`). It discovers the geometric overlaps in the $CCZ$ ladders and mathematically reduces the overall T-count substantially without ever building a DAG.

## Phase 3: Statevector Squeezing (Minimizing $k_{\max}$)

**Goal:** Temporally separate entanglement generation from state collapse to exponentially shrink the Virtual Machine's physical RAM usage ($\mathcal{O}(2^k)$). This reorganizes the causal structure to evaluate uncomputation measurements as early as physically possible.

* **Task 3.1 (Eager Compaction - Backward Sweep):** Implement `StatevectorSqueezePass` inheriting from `Pass`. Iterate *backward* through the HIR. When encountering a `MEASURE` node, bubble it as far *left* (as early) as possible. Swap it with preceding nodes sequentially, halting only when it hits a node it anti-commutes with (using the $\mathcal{O}(1)$ check) or another barrier.
* **Task 3.2 (Lazy Expansion - Forward Sweep):** Iterate *forward* through the HIR. When encountering a non-Clifford node (`T_GATE` or `GATE` for LCUs), bubble it as far *right* (as late) as possible, swapping it with succeeding nodes as long as they commute.
* **DoD:** A Catch2 unit test feeds the interleaving sequence: `H 0; T 0; H 1; T 1; M 0; M 1;`. The compiler safely bubbles this into `H 0; T 0; M 0; H 1; T 1; M 1;`. The VM reports that $k_{\max}$ strictly drops from 2 (requiring 4 complex amplitudes) to 1 (requiring 2 amplitudes) because the array now "breathes" optimally.

## Phase 4: The `op-T-mize` Benchmark Suite

**Goal:** Quantify UCC's optimization superiority and memory footprint against industry standards using canonical quantum algorithms.

* **Task 4.1 (Benchmark Harness):** Create `paper/optimize/benchmark_optimize.py`. Download the standard `op-T-mize` benchmark circuits highlighted in the literature (e.g., `Adder16`, `csla_mux3`, `barenco_tof_10`, `gf2^4_mult`).
* *Critical Modification:* Standard `op-T-mize` circuits are strictly unitary (`.qasm` without measurements). To benchmark Statevector Squeezing accurately, write a utility to inject mid-circuit `MEASURE` instructions onto the "garbage" ancillas of the arithmetic adders immediately following their uncomputation blocks.



*   **Task 4.2 (The Multi-Tool Benchmarking Shootout):** Create a comparative execution script (`paper/optimize/benchmark_shootout.py`) to systematically evaluate UCC against the industry-standard T-count optimizers highlighted in the literature. Route the modified `op-T-mize` circuits through the following frameworks:
    0. You may need to research this libraries online, as well as the https://pennylane.ai/datasets/op-t-mize dataset. Please first come up with a plan on how to author this. YOu can ask for human help in finding sites/details of the packages.
    1.  **TKET (`pytket`):** Apply `pytket.passes.FullPeepholeOptimise()`. This represents the state-of-the-art in DAG-based spatial peephole optimization and will demonstrate how rigid CNOT ladders block standard routing compilers from seeing distant redundancies.
    2.  **PyZX (`pyzx`):** Apply `pyzx.full_reduce()`. This represents the state-of-the-art in heavy, graph-based ZX-Calculus rewrites (which natively search for Spider-Nest identities and phase polynomials).
    3.  **TODD:** If available via CLI (e.g., `todd-rs`), use this as the baseline for heavy $\mathcal{O}(n^4)$ phase-polynomial extraction solvers.
    4.  **UCC:** Natively apply the `PhaseFoldingPass`, `SpiderNestPass`, and `StatevectorSqueezePass` over the Heisenberg IR.
    *Execution Protocol:* For each modified circuit in the `op-T-mize` dataset, route it through UCC and the baseline tools. Run each compiler inside an isolated Python subprocess to accurately capture compilation latency without garbage collection drift. Strictly record the starting T-count, the final optimized T-count, and the wall-clock compilation time in milliseconds.
