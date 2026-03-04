# UCC Implementation Plan: Invariant-Based Native Fuzzing

## Executive Summary & Constraints

We want to implement more robust testing of the UCC toolchain, since random circuits tend to decohere to the uniform superposition state. We want to be sure we are catching edge cases in other states.

**Strict Constraints:**
1. **Tier 1 First:** You must prove the raw array math and basis compressor native in C++ before attempting Python integration.
2. **No Text Roundtripping:** Do not test the compiler by formatting the HIR back into `.stim` text. The Front-End destructively absorbs Cliffords, rendering syntactic roundtrips impossible.
3. **C++ Native Oracle:** The factored state must be verifiable directly in C++ via a new bridge function expanding the active statevector.

---

## Phase 1: The C++ Statevector Bridge

**Goal:** Prove the core mathematical equation $|\psi\rangle = \gamma U_C P (|\phi\rangle_A \otimes |0\rangle_D)$ works natively in C++, anchoring all future validation.

*   **Task 1.1 (Expansion Math):** Rewrite `ucc::get_statevector` in `svm.cc`.
    1. Allocate a dense $2^n$ array.
    2. Expand the $2^k$ elements of $v[]$ into the dense array, interleaving zeros for Dormant dimensions.
    3. Apply the Pauli frame $P$ (using the `p_x`, `p_z` trackers).
    4. Apply the offline compiled Clifford Frame $U_C$ (`final_tableau` from Constant Pool).
    5. Multiply by the global scalar $\gamma$.
*   **Task 1.2 (Static Validation):** Write a Catch2 test that manually populates a `SchrodingerState` with known values, pairs it with a known $U_C$, and verifies the expansion perfectly matches a hand-calculated numpy-style statevector.

## Phase 2: Fuzzing the Virtual Compressor (Back-End)

**Goal:** Prove the greedy $\mathcal{O}(n)$ basis reduction algorithm correctly localizes massive multi-qubit Pauli interference strings.

*   **Task 2.1 (Heavy Mask Generator):** Write a C++ fuzzer that generates chaotic, highly entangled multi-qubit `stim::PauliString` masks ($X$ and $Z$ components spread across 20 qubits).
*   **Task 2.2 (Compression Oracle):** Feed the masks into the Back-End's compressor.
*   **Task 2.3 (Validation):** Extract the emitted sequence of virtual `OP_FRAME_CNOT` and `OP_FRAME_CZ` instructions ($V$). Assert mathematically that conjugating the input Pauli string by this sequence ($V P_{in} V^\dagger$) results in an operator supported on exactly **one** virtual qubit axis.

## Phase 3: Fuzzing Array Compaction (VM)

**Goal:** Prove that division-free array folding perfectly preserves quantum probability when an active axis is measured and demoted to Dormant.

*   **Task 3.1 (Compaction Fuzzer):** In `test_svm.cc`, write a test that bypasses the compiler entirely. Manually construct a `SchrodingerState` with $k=4$ (16 complex elements initialized randomly).
*   **Task 3.2 (Interference Execution):** Manually execute `OP_MEAS_ACTIVE_INTERFERE` on a target axis $v$. This physically halves the array size ($k \to 3$).
*   **Task 3.3 (Norm & Expectation Validation):** Assert that the scalar $\gamma$ correctly deferred the $\sqrt{2 P_m}$ normalization. Assert that the absolute physical probabilities calculated via the statevector bridge identically match theoretical projective matrices.

## Phase 4: Python End-to-End Oracles

**Goal:** Validate the integrated 4-stage pipeline directly against Qiskit-Aer.

*   **Task 4.1 (Unitary Fuzzer):** Update `test_qiskit_aer.py`. Generate random circuits containing dense Clifford entanglement interspersed with `T` and `T_DAG` gates.
*   **Task 4.2 (Execution):** Compile the circuit in UCC, execute 1 shot natively, and use `ucc.get_statevector` to extract the full dense state.
*   **Task 4.3 (Validation):** Assert fidelity $> 0.9999$ against `qiskit_aer.StatevectorSimulator`.
*   **Task 4.4 (Stochastic Fuzzer):** Update `test_statistical_equivalence.py`. Run circuits with mid-circuit measurements and noise channels. Verify that the distributions of the classical `meas_record` fall within strict $5\sigma$ binomial bounds compared to Stim.
