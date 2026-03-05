# UCC Implementation Plan: State-Agnostic Global Optimization

## Executive Summary & Constraints

In the Factored State architecture, the Front-End mathematically rewinds all non-Clifford operations back to the t=0 vacuum. As a result, the Heisenberg IR (HIR) is inherently a **global phase polynomial** completely devoid of hardware timing or memory layout. This allows the Middle-End to bypass "Hadamard Gadgetization" and synthesize T-count optimizations directly using Third-Order Symmetric Tensor Rank (3-STR) reductions.

This plan implements Vandaele's (2025) algorithms. We prioritize **TOHPE** (Third Order Homogeneous Polynomials Elimination, O(n^2 m^3)) as the core pass, followed by **FastTODD** (O(n^4 m^3)) as a deeper extension.

**Strict Constraints:**
1. **State-Agnosticism:** The optimizer must manipulate the HIR purely using mathematical masks (destab_mask, stab_mask). It must never require knowledge of an initial state or the Virtual Machine's Active/Dormant sets.
2. **Virtual Diagonalization Bridge:** Vandaele's binary parity tables strictly require Z-basis operators. The optimizer must dynamically compute local Clifford transformations to temporarily diagonalize arbitrary commuting Pauli cliques into pure Z-strings before executing the linear algebra reductions.
3. **Clifford Remainders:** T-count reduction alters the phase polynomial at the 3rd order, leaving residual degree-1 and degree-2 phase polynomials (S and CZ equivalents). The optimizer must emit these back into the HIR as explicit nodes. The Back-End will naturally absorb these into the V_cum tracker at zero runtime cost.
4. **Reproducible** When it comes to actually optimize the circuits (so not the changes to UCC itselef), write scripts to do them in `paper/optimize` directory. The scripts should also generate any plots we would plan to use in a paper.
---

## Phase 1: HIR Expansion & Peephole Fusion

**Goal:** Expand the HIR to support arbitrary Clifford phase residuals and implement fast adjacent algebraic cancellation.

*   **Task 1.1 (HIR Update):** Update `OpType` in `src/ucc/frontend/hir.h` to include `CLIFFORD_PHASE`. Update `HeisenbergOp` with a factory method to define a Clifford phase rotation (e.g., S, Z, S_dag) applied to a specific Pauli mask.
*   **Task 1.2 (Pass Interface):** Create `src/ucc/optimizer/pass.h`. Define a pure abstract base class `Pass { virtual void run(HirModule& hir) = 0; }`. Define `PassManager` to execute them sequentially.
*   **Task 1.3 (Peephole Pass):** Implement `PeepholeFusionPass`. Iterate left-to-right. For each `T_GATE`, scan ahead for another `T_GATE` with identical `destab_mask` and `stab_mask`. Halt the scan immediately if you hit an anti-commuting `MEASURE` or `NOISE` node (using the O(1) symplectic inner product).
*   **Task 1.4 (Algebraic Fusion):**
    *   If T * T_dag = I: Delete both nodes.
    *   If T * T = S: Replace the two nodes with a single `CLIFFORD_PHASE` HIR node on the shared Pauli axis.
*   **DoD:** A Catch2 test feeds H 0; T 0; T 0; H 0; T 0. The pass fuses the first two T's into a `CLIFFORD_PHASE`, leaving the third T intact. The Back-End absorbs the new node correctly without expanding the active statevector.

## Phase 2: Virtual Diagonalization & The Parity Table Bridge

**Goal:** Translate a clique of arbitrary commuting HIR Paulis into Vandaele's binary Z-basis Parity Table P, and translate back.

*   **Task 2.1 (Clique Extraction):** Implement the first phase of `TohpePass`. Sweep the HIR to collect maximal cliques of mutually commuting `T_GATE`s bounded by stochastic barriers. Let the clique size be m.
*   **Task 2.2 (Simultaneous Diagonalization):** For a clique of m Paulis, use a local GF(2) solver (or a temporary `stim::Tableau`) to find a Clifford transformation C that maps all m generators to pure Z-strings (i.e., destab_mask becomes 0 for all of them in the temporary frame).
*   **Task 2.3 (Parity Table Construction):** Construct the binary Parity Table P (size n x m) where column i is the stab_mask of the diagonalized Pauli i.
*   **Task 2.4 (The Remainder Extraction):** Write a utility that, given an original Parity Table P and an optimized table P', computes the degree-2 and degree-1 difference in the phase polynomial (Vandaele Eq. 14). Emit this difference as diagonal CZ and S operations, map them back through C_dag, and emit them into the HIR as `CLIFFORD_PHASE` nodes alongside the optimized T gates.
*   **DoD:** A test extracts a commuting clique of Y_0 Z_1 and X_0 X_1, perfectly diagonalizes them to Z_0 and Z_1, builds table P, and translates identical columns back to the original physical masks.

## Phase 3: The TOHPE Pass (Algorithm 2)

**Goal:** Implement the O(n^2 m^3) Third Order Homogeneous Polynomials Elimination algorithm.

*   **Task 3.1 (GF2 Utilities):** Implement a fast GF(2) right-null-space extraction utility using 64-bit word operations.
*   **Task 3.2 (Matrix L & Z-set):** Inside `TohpePass`, for a given Parity Table P, compute the binary matrix L where L_{alpha, beta} = P_alpha AND P_beta. Compute the candidate set Z.
*   **Task 3.3 (Null-Space Search):** Find a vector y in the null-space of L such that y != 0 and (y != 1 or |y| == 0 mod 2).
*   **Task 3.4 (Objective Maximization):** If y is found, iterate through the Z-set to find the vector z that maximizes the column reduction objective function (Equation 35 from Vandaele 2025).
*   **Task 3.5 (Substitution):** Execute P' = P XOR (z * y^T). Remove duplicate/zero columns. Call Task 2.4 to overwrite the HIR with the optimized P' and the Clifford remainders.
*   **DoD:** A synthetic CCZ circuit (7 T-gates) compiles. The TOHPE pass natively isolates the clique, executes Algorithm 2, and outputs an optimized HIR containing exactly 4 T-gates and the correct residual Clifford phases.

## Phase 4: Optimization Verification & Oracles

**Goal:** Establish multi-tiered mathematical guarantees that the optimizer does not corrupt quantum probability.

*   **Task 4.1 (Tier 1: Signature Tensor Invariant):** Write a C++ Catch2 test `test_opt_tensor.cc`. Given a clique P and its optimized replacement P', calculate A_{alpha, beta, gamma} = |P_alpha AND P_beta AND P_gamma| mod 2. Assert that A == A' for all triplets. This mathematically proves 3rd-order equivalence natively in C++.
*   **Task 4.2 (Tier 2: Exact Statevector Fuzzing):** Write a Python test that generates random 8-qubit Clifford+T circuits. Compile and execute them with the optimizer OFF and ON. Assert that `ucc.get_statevector()` yields identical arrays (fidelity > 0.9999).
*   **Task 4.3 (Tier 3: The Identity Uncompute Test):** For massive circuits where statevectors OOM (e.g., a 50-qubit Adder). Generate the optimized HIR. Programmatically append the exact syntactic inverse (U_dag) of the *unoptimized* circuit to the HIR. Simulate 100,000 shots. Assert every single shot deterministically measures 0 across all qubits, proving U_opt * U_dag = I.

## Phase 5: The op-T-mize Benchmark Suite

**Goal:** Quantify UCC's optimization superiority in the Python layer against industry standards.

*   **Task 5.1 (Benchmark Harness):** Create `paper/optimize/test_bench_optimize.py`. Download the standard `op-T-mize` QASM circuits (Adder8, csla_mux3, barenco_tof_10, gf2^4_mult, etc.).
*   **Task 5.2 (Execution & Metrics):** Route the circuits through UCC, `pyzx.full_reduce()`, and `pytket.passes.FullPeepholeOptimise`.
*   **Task 5.3 (Reporting):** For each framework, report:
    1. Final T-Count.
    2. Compilation Latency (ms).
    3. **Peak Active Rank (k_max):** Highlight how UCC's logical T-count reduction directly shrinks the Virtual Machine's physical RAM footprint, a metric completely unique to the Factored State architecture.

## Phase 6: The FastTODD Upgrade (Algorithm 3) [Optional]

**Goal:** Extend TOHPE with the deeper O(n^4 m^3) search space.

*   **Task 6.1 (Sub-solver Matrices):** Implement the `FastToddPass`. It first calls TOHPE until no reductions remain. It then constructs the matrices X^(z) and v^(z) to search for linear dependencies in the expanded space.
*   **Task 6.2 (Gaussian Elimination):** Implement the block GF(2) solver to extract the right null-space of the combined matrices.
*   **Task 6.3 (Substitution):** Apply the exact same objective maximization and P' substitution as TOHPE.
*   **DoD:** FastTODD finds edge-case reductions that TOHPE missed on specific deep circuit topologies.
