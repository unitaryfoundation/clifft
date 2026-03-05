# UCC Implementation Plan: Performance Benchmarking & Scaling Analysis

## Executive Summary & Constraints

To demonstrate UCC's utility as a high-performance simulation tool, we must benchmark it across three distinct topological regimes to isolate its architectural advantages. Because standard statevector (SV) simulators scale based on the physical qubit count $\mathcal{O}(2^N)$, while UCC scales based on the active non-Clifford dimension $\mathcal{O}(2^k)$, a standard benchmark suite only tells half the story.

This plan establishes a standalone, automated benchmarking suite comparing UCC against industry standards: **Stim** (for stabilizer circuits), and **Qiskit Aer** and **Qulacs** (for dense statevector simulation).

**Strict Constraints:**
1.  **Single-Core Execution:** To accurately compare algorithmic efficiency and memory footprint, all benchmarks must restrict execution to a single physical CPU core. Multi-threading masks fundamental architectural scaling limits and makes memory tracking unreliable.
2.  **Correctness First:** Before capturing any timing metrics, every benchmark circuit topology must be mathematically verified against Qiskit Aer (fidelity $> 0.9999$) or Stim (exact measurement distributions) at small scales ($N \le 8$).
3.  **Strict Timing Separation:** For UCC, the profiling harness MUST explicitly report "AOT Compilation Time" separately from "VM Execution Time" to highlight the $\mathcal{O}(1)$ execution loop.
4.  **Noiseless SV Baseline:** To provide a clean computer-science comparison of memory and FLOPs scaling without the confounding overhead of Monte Carlo trajectory wrappers or density matrix switching, the statevector baseline comparisons must be strictly noiseless.
5. **Reproducible** Write the benchmark scripts (so not changes to UCC itself) as a set of python files in a `paper/benchmark` directory. The scripts should also generate any plots we would plan to use in a paper.

---

## Phase 1: Input Format & Benchmark Gate Support

**Goal:** Extend the UCC parser to support the parameterized gates required for standard QFT and RQC benchmarks, while maintaining a pure extended-Stim input format and minimizing C++ AST bloat.

*   **Task 1.1 (Continuous Rotation Gates):** Add native support for parameterized rotations to `gate_data.h` and the parser.
    *   **Add `RZ(theta)`:** The Front-End evaluates this mathematically as $e^{-i \frac{\theta}{2} Z} = \cos(\frac{\theta}{2}) I - i \sin(\frac{\theta}{2}) Z$. Utilize the Dominant Term Factoring logic (established in `coherent_noise.plan.md`) to extract the massive identity branch offline, multiplying $\cos(\frac{\theta}{2})$ into the global scalar, and emitting the localized $Z$ interference via a generic `GATE` LCU node into the HIR.
    *   **Add `RX(theta)` and `RY(theta)`:** Instead of complex native lowering, have the parser or Front-End immediately desugar these into Clifford conjugates surrounding an `RZ` gate (e.g., `RX(theta) q` $\to$ `H q; RZ(theta) q; H q`).
*   **Task 1.2 (QASM Transpiler):** Statevector benchmarks traditionally use OpenQASM. Instead of building a complex OpenQASM AST natively in C++, write a lightweight Python utility in `paper/benchmark/qasm_to_ucc.py`. This script parses standard OpenQASM 2.0 benchmark files (using `qiskit.qasm2`) and unrolls them into the extended `.stim` text format.
    *   Map standard unparameterized gates ($H$, $S$, $X$, $CX$, etc.) directly to their Stim equivalents.
    *   Decompose parameterized controlled-phase gates ($CPHASE(\theta)$) into $CNOT$ and `RZ` gates using standard analytical expansions before writing to the `.stim` file.
*   **DoD:** The UCC compiler successfully ingests and compiles an unrolled QFT circuit containing $H$, $CNOT$, and `RZ(theta)` instructions, outputting valid localized RISC bytecode without requiring any new multi-qubit VM opcodes.

## Phase 2: Topology Verification (The Correctness Gates)

**Goal:** Establish an absolute guarantee that the benchmarked circuits compute correct quantum probabilities before profiling them for speed or memory.

*   **Task 2.1 (Statevector Equivalence Fuzzing):** Create `paper/benchmark/verify_sv.py`. Generate the exact Dense QFT and Synthetic Factored State circuits at scales $N \in [4, 8, 12]$.
    *   Load the circuit into Qiskit Aer and extract the dense complex statevector.
    *   Compile the generated `.stim` file in UCC, execute 1 shot, and use the C++ bridge `ucc.get_statevector()` to expand the factored state into a dense $2^N$ array.
    *   Assert that the absolute fidelity $|\langle \psi_{\text{Aer}} | \psi_{\text{UCC}} \rangle|^2 > 0.9999$.
*   **Task 2.2 (Distribution Equivalence Fuzzing):** Create `paper/benchmark/verify_stim.py`. Generate the exact Unrotated Surface Code memory circuit used for the Stim baseline at distance $d=3$.
    *   Sample 10,000 shots in Stim and 10,000 shots in UCC.
    *   Assert that the classical measurement parity distributions fall within strict $5\sigma$ binomial bounds of each other.
*   **DoD:** The test suite passes automatically with zero errors, mathematically validating the compiler's geometric rewinding and AOT compression for the exact continuous and dynamic topologies we are about to benchmark.

## Phase 3: The Profiling Harness

**Goal:** Build a robust, automated Python benchmarking harness capable of cleanly extracting Peak Memory and CPU Time while preventing garbage-collection drift.

*   **Task 3.1 (Subprocess Isolation):** Create `paper/benchmark/runner.py`. To prevent Python memory bloat from polluting the metrics, the harness must spawn each simulation (Aer, Qulacs, Stim, UCC) in a completely fresh, isolated `subprocess`.
*   **Task 3.2 (Resource Tracking):** Implement a resource tracker using the OS-level `resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss` to capture the absolute peak physical RAM utilized by the C++ extensions during execution.
*   **Task 3.3 (Data Export):** The runner must execute a parameter grid and export a normalized CSV containing `Framework`, `Regime`, `N_Qubits`, `k_max` (for UCC), `Peak_Memory_MB`, `Time_Compile_s`, and `Time_Exec_s`.

## Phase 4: Regime A - The Stabilizer Baseline ($k = 0$)

**Goal:** Prove UCC’s generic array compaction and virtual basis updates incur minimal overhead compared to the industry’s most optimized SIMD Pauli-frame tracker.

*   **Task 4.1 (Circuit Generation):** Use `stim.Circuit.generated("surface_code:unrotated_memory_z", ...)` to generate noiseless distance $d \in [3, 5, 7, 9, 11]$ circuits.
*   **Task 4.2 (Execution):**
    *   Target: 10,000 bulk shots (to trigger batch-processing loops).
    *   Measure the total runtime to return the 10,000 shots for both Stim and UCC.
*   **Task 4.3 (Evaluation):** Plot Execution Time vs Distance $d$.
*   **DoD:** UCC's curve scales polynomially and remains within the same order of magnitude as Stim, proving that dynamic array compaction (`OP_MEAS_ACTIVE_INTERFERE` / `OP_FRAME_SWAP`) for mid-circuit measurements resolves natively in $\mathcal{O}(1)$ time without bloating the VM execution loop.

## Phase 5: Regime B - The Dense Baseline ($k \to N$)

**Goal:** Establish UCC as a competitively performant dense statevector engine when topological compression provides no dimensional advantage.

*   **Task 5.1 (The Scrambled QFT):** Generate QFT circuits for $N \in [10, 15, 20, 25, 28]$.
    *   *Critical Step:* Prepend a layer of random $X$ and $Z$ gates, followed by a global layer of $H$ gates to all qubits. This heavily scrambles the initial computational basis, forcing Qiskit and Qulacs to evaluate massive dense entanglement and defeating any "zero-state" optimization heuristics.
*   **Task 5.2 (Execution):** Measure the **Time to 1st Sample** (the time required to execute the gates and compute the final statevector).
*   **Task 5.3 (Evaluation):** Plot Execution Time (log scale) vs $N$.
*   **DoD:** The plot shows Qiskit, Qulacs, and UCC's VM Execution Time all following the same $\mathcal{O}(2^N)$ exponential scaling curve. UCC's execution latency should be closely comparable to Qiskit Aer, proving the 32-byte RISC array logic traverses the CPU L1 cache efficiently on strictly dense workloads.

## Phase 6: Regime C - The Factored State Boundary ($N \gg k$)

**Goal:** The definitive showcase. Visually demonstrate *why* the Clifford Frame was decoupled from the Active Statevector by scaling physical qubits massively while keeping non-Clifford entanglement bounded.

*   **Task 6.1 (The Synthetic Sparse Circuit):** Write a generator that strictly caps non-Clifford complexity while exploding physical complexity.
    1.  Parameterize physical qubits $N \in [20, 30, 40, 60, 80, 100]$.
    2.  Lock the active rank to exactly $k=15$ by applying `T` gates to only 15 specific qubits.
    3.  Apply massive, deeply entangling ladders of global Cliffords (e.g., $CNOT(i, i+1)$, $H$, $S$) across *all* $N$ qubits to ensure standard simulators must entangle the entire physical state.
*   **Task 6.2 (The OOM Barrier):** Run Qiskit, Qulacs, and UCC. Set a hard system memory limit (e.g., 32 GB) to simulate standard hardware bounds.
*   **Task 6.3 (Memory vs. Time Evaluation):**
    *   **Plot A (Peak Memory vs $N$):** Qiskit and Qulacs will attempt to allocate a $2^N$ complex array. They will exhibit an exponential vertical asymptote and OOM crash around $N \approx 30$. UCC must display a perfectly flat horizontal line at $\sim 512$ KB (the exact size of a $2^{15}$ array).
    *   **Plot B (Time vs $N$):** Qiskit and Qulacs execution times will scale exponentially. UCC's VM execution time will remain a perfectly flat horizontal line. UCC's AOT Compilation Time will scale gracefully at $\mathcal{O}(N^2)$ polynomial time.
*   **DoD:** The generated plots explicitly prove that standard statevector simulators hit an exponential physical wall, while UCC scales effortlessly based on logical entanglement rather than physical footprint.
