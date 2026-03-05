# UCC Implementation Plan: Invariant-Based Native Fuzzing

**Status: COMPLETE** — All 5 phases implemented and merged (PRs #61-#65).

## Executive Summary & Constraints

We want to implement more robust testing of the UCC toolchain, since random circuits tend to decohere to the uniform superposition state. We want to be sure we are catching edge cases in other states.

**Strict Constraints:**
1. **Tier 1 First:** You must prove the raw array math and basis compressor native in C++ before attempting Python integration.
2. **No Text Roundtripping:** Do not test the compiler by formatting the HIR back into `.stim` text. The Front-End destructively absorbs Cliffords, rendering syntactic roundtrips impossible.
3. **C++ Native Oracle:** The factored state must be verifiable directly in C++ via a new bridge function expanding the active statevector.

---

## Phase 1: The C++ Statevector Bridge ✅

*Merged: PR #61*

**Goal:** Prove the core mathematical equation $|\psi\rangle = \gamma U_C P (|\phi\rangle_A \otimes |0\rangle_D)$ works natively in C++, anchoring all future validation.

*   **Task 1.1 (Expansion Math):** Rewrite `ucc::get_statevector` in `svm.cc`.
    1. Allocate a dense $2^n$ array.
    2. Expand the $2^k$ elements of $v[]$ into the dense array, interleaving zeros for Dormant dimensions.
    3. Apply the Pauli frame $P$ (using the `p_x`, `p_z` trackers).
    4. Apply the offline compiled Clifford Frame $U_C$ (`final_tableau` from Constant Pool).
    5. Multiply by the global scalar $\gamma$.
*   **Task 1.2 (Static Validation):** Write a Catch2 test that manually populates a `SchrodingerState` with known values, pairs it with a known $U_C$, and verifies the expansion perfectly matches a hand-calculated numpy-style statevector.

## Phase 2: Fuzzing the Virtual Compressor (Back-End) ✅

*Merged: PR #62*

**Goal:** Prove the greedy $\mathcal{O}(n)$ basis reduction algorithm correctly localizes massive multi-qubit Pauli interference strings.

*   **Task 2.1 (Heavy Mask Generator):** Write a C++ fuzzer that generates chaotic, highly entangled multi-qubit `stim::PauliString` masks ($X$ and $Z$ components spread across 20 qubits).
*   **Task 2.2 (Compression Oracle):** Feed the masks into the Back-End's compressor.
*   **Task 2.3 (Validation):** Extract the emitted sequence of virtual `OP_FRAME_CNOT` and `OP_FRAME_CZ` instructions ($V$). Assert mathematically that conjugating the input Pauli string by this sequence ($V P_{in} V^\dagger$) results in an operator supported on exactly **one** virtual qubit axis.

## Phase 3: Fuzzing Array Compaction (VM) ✅

*Merged: PR #63*

**Goal:** Prove that division-free array folding perfectly preserves quantum probability when an active axis is measured and demoted to Dormant.

*   **Task 3.1 (Compaction Fuzzer):** In `test_svm.cc`, write a test that bypasses the compiler entirely. Manually construct a `SchrodingerState` with $k=4$ (16 complex elements initialized randomly).
*   **Task 3.2 (Interference Execution):** Manually execute `OP_MEAS_ACTIVE_INTERFERE` on a target axis $v$. This physically halves the array size ($k \to 3$).
*   **Task 3.3 (Norm & Expectation Validation):** Assert that the scalar $\gamma$ correctly deferred the $\sqrt{2 P_m}$ normalization. Assert that the absolute physical probabilities calculated via the statevector bridge identically match theoretical projective matrices.
*   **Task 3.4 (Zero-Probability Branch Stability):** Construct an active array initialized to a specific state that is mathematically deterministic in the measurement basis (e.g., the exact $|-\rangle$ state measured via `OP_MEAS_ACTIVE_INTERFERE` in the $X$-basis). Assert that the VM gracefully handles the $P=0$ branch. The deferred normalization ($\gamma \leftarrow \gamma / \sqrt{2 P_m}$) must resolve without `NaN` or `Infinity` corruption from floating-point noise.
*   **Task 3.5 (Active/Dormant Boundary Straddling):** In `test_backend.cc`, feed the compressor heavy Pauli strings that target both Active and Dormant axes simultaneously. Assert that the compiler emits `OP_EXPAND` only when structurally required, maintaining the Zero-Cost Dormant property where possible.

## Phase 4: Python End-to-End Oracles ✅

*Merged: PR #64*

**Goal:** Validate the integrated 4-stage pipeline directly against Qiskit-Aer.

*   **Task 4.1 (Unitary Fuzzer):** Update `test_qiskit_aer.py`. Generate random circuits containing dense Clifford entanglement interspersed with `T` and `T_DAG` gates.
*   **Task 4.2 (Execution):** Compile the circuit in UCC, execute 1 shot natively, and use `ucc.get_statevector` to extract the full dense state.
*   **Task 4.3 (Validation):** Assert fidelity $> 0.9999$ against `qiskit_aer.StatevectorSimulator`.
*   **Task 4.4 (Stochastic Fuzzer):** Update `test_statistical_equivalence.py`. Run circuits with mid-circuit measurements and noise channels. Verify that the distributions of the classical `meas_record` fall within strict $5\sigma$ binomial bounds compared to Stim.

## Phase 5: Structural Oracles & Memory Lifecycle ✅

*Merged: PR #65*

**Goal:** Bypass random-circuit decoherence by testing exact destructive interference, extreme memory lifecycles, and biased probabilities.

*   **Task 5.1 (The Bounded-T Mirror Fuzzer):** Overcomes the Qiskit 20-qubit memory wall without OOMing the VM.
    *   **The Generator:** Write a Python fuzzer that generates a massive 40-qubit circuit ($U$) containing thousands of random Clifford gates (`H`, `S`, `CX`), but strictly limits the total number of `T` gates to exactly 12.
    *   **The Inverse:** Programmatically compute the exact syntactic inverse ($U^\dagger$) by reversing gate order and swapping $T \leftrightarrow T^\dagger$, $S \leftrightarrow S^\dagger$.
    *   **The Bypass:** Combine them into $U U^\dagger$. You **must disable or bypass the Middle-End optimizer** for this test (otherwise it will statically cancel the $t=0$ HIR nodes and the VM will do no work).
    *   **Execution:** Compile and simulate 10,000 shots natively in UCC.
    *   **Assertion:** The peak rank (`prog.peak_rank`) must be $\le 12$ (using $\sim 64$ KB of RAM). Every single shot must deterministically measure the all-zeros bitstring ($|00\dots0\rangle$). This proves the Back-End compressor and VM array logic can perfectly evaluate and uncompute 40-qubit global entanglement without floating-point drift or topological corruption.

*   **Task 5.2 (The "Breathing" Memory Lifecycle Test):** Stresses the `VirtualRegisterManager` and deferred normalization.
    *   Write a circuit loop that repeatedly injects a non-Clifford state (`H`, then `T`), entangles it with a persistent logical register, and measures the injected qubit to force array compaction. Repeat 500 times.
    *   **Assertion:** The peak rank ($k_{\max}$) must remain strictly bounded (e.g., $k$ breathes $1 \to 2 \to 1$). The final $\gamma$ scalar must not underflow to zero despite hundreds of consecutive $1/\sqrt{2}$ divisions.

*   **Task 5.3 (Biased Amplitude Statistics):** Fuzz with circuits possessing known analytical biases (e.g., `H 0; T 0; H 0; M 0` yields $P(0) = \cos^2(\pi/8) \approx 85.35\%$). Assert the sampled distribution falls within strict binomial bounds (e.g., $5\sigma$) to prove the RNG branch selection logic works perfectly on asymmetric splits.

---

## Open Items & Future Work

All planned phases are complete. The test suite now contains 255 C++ tests and 194 Python tests (449 total), running in ~5s. The following areas are not covered by this plan and may warrant future testing investment:

1.  **Middle-End Optimizer Testing:** The `skip_optimizer` parameter was added to `ucc.compile()` as future-proofing. When a middle-end optimizer is implemented (e.g., T-gate cancellation, dead code elimination), dedicated tests should verify it preserves semantics and that `skip_optimizer=True` correctly bypasses it.

2.  **Higher Qubit Count Statevector Validation:** The Qiskit-Aer statevector oracle is capped at 6 qubits due to the $O(4^n)$ cost of `get_statevector`'s tableau-to-unitary expansion. An optimized expansion (e.g., column-by-column Clifford simulation) could push this to 15-20 qubits.

3.  **Gate Coverage Gaps:** The test suite exercises H, S, S_DAG, T, T_DAG, X, Y, Z, CX, CY, CZ, M, MR, R, RX, and noise channels. Gates listed in `design/missing_gates.plan.md` (e.g., ISWAP, SQRT_X, etc.) will need tests when implemented.

4.  **Adversarial Floating-Point Circuits:** While the breathing test validates gamma renormalization over 1000 rounds, circuits that produce catastrophic cancellation in the array amplitudes (near-zero norms from destructive interference mid-circuit) are not explicitly tested.

5.  **Performance Regression Tests:** The current suite validates correctness only. A separate benchmark harness (e.g., pytest-benchmark or C++ Google Benchmark) could track compile time, sample throughput, and memory usage across commits.
