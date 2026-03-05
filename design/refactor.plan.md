# UCC Refactor Plan: Factored State Architecture

## Status: COMPLETE ✅

All four phases (0-4) are implemented and verified. 208 C++ Catch2 tests and
129 Python integration tests pass, including exact statevector oracle validation
against Qiskit-Aer and statistical equivalence against Stim for noisy QEC circuits.

Completed on branch `feat/phase4-pipeline-integration` across 3 commits:
- `bf8ee00` Phase 4 Task 4.1: Remove AG pivot infrastructure from front-end
- `484c747` Phase 4 Tasks 4.2-4.3: Full lower() pipeline wiring with new opcodes
- `4878e5a` Phase 4 Task 4.4: Fix 8 pipeline integration bugs (U_C order,
  measurement normalization, OP_ARRAY_S, T gate lowering, sign tracking)

### Open Items / Known Limitations

1. **No `REPEAT` support.** The parser rejects `REPEAT` blocks. QEC circuits
   must be fully unrolled (Stim's `circuit.flattened()`) before compilation.
2. **10-qubit statevector cap.** `get_statevector()` rejects circuits with >10
   qubits to prevent accidental 2^n blowup. This is a safety guard, not a
   fundamental limit — sampling works at any qubit count within the 64-qubit
   frame width.
3. **No `OP_POSTSELECT`.** Post-selection is not yet implemented in the VM.
   Required for Magic State Cultivation (see `design/magic.plan.md` Phase 2).
4. **Single frame width (W=64).** The Pauli frame uses `stim::bitword<64>`,
   limiting circuits to 64 qubits. Scaling to 512 qubits requires template
   monomorphization (see `design/magic.plan.md` Phase 1).
5. **No S-dagger array opcode.** `OP_ARRAY_S` was added but `OP_ARRAY_S_DAG`
   is not yet needed (compress_pauli only emits S, not S_DAG). May be needed
   for future gate decompositions.
6. **Greedy compression is O(n) but not optimal.** The Pauli compressor uses a
   single-pass greedy pivot selection. It always succeeds but may emit more
   CNOTs than a global optimization would.

---

## Executive Summary & Constraints

This plan overhauls the UCC architecture to implement the explicit Factored State Representation defined in `theory.tex`:
$$|\psi^{(t)}\rangle = \gamma^{(t)} U_C^{(t)} P^{(t)} (|\phi^{(t)}\rangle_A \otimes |0\rangle_D)$$

By proving that multi-qubit Aaronson-Gottesman (AG) measurement pivots can be synthesized via ahead-of-time (AOT) virtual coordinate compression, we are completely removing $\mathcal{O}(n^2)$ matrix math, `x_mask`, `commutation_mask`, and `GF2Basis` tracking from the Virtual Machine (VM). The VM is transitioning to a hyper-fast, localized RISC instruction set operating on a dynamically scaling active statevector.

**Strict Constraints for this Phase:**
1. **The Clean Slate:** You MUST begin this refactor by deleting obsolete code. The Python prototype (`prototype/`), old tests, and the C++ GF(2) basis logic (`src/ucc/backend/gf2_basis.cc/h`) must be entirely removed before writing new logic.
2. **64-Qubit MVP, 512-Qubit Prep:** We restrict the system to 64 qubits for now, but you MUST use `stim::bitword<kStimWidth>` for the Pauli frame ($P$) in `SchrodingerState`. Use `uint16_t` for virtual axes in the `Instruction` payload. This guarantees the 32-byte struct and frame trackers won't need architectural changes when scaling to 512+ qubits.
3. **No Commutation in the VM:** The VM must never evaluate basis spans or commutations. All multi-qubit Pauli interference must be geometrically compressed into local 1-qubit and 2-qubit virtual axis operations AOT by the Back-End.
4. **Contiguous Array Compaction:** When an active qubit is measured and demoted to Dormant, the array halves. To prevent strided/fragmented indexing, the compiler MUST emit virtual `SWAP` instructions to route the target axis to the highest active dimension ($k-1$) immediately before measurement.
5. **Incremental C++ TDD:** Because we are abandoning the Python prototype, you must validate the math layer-by-layer using native C++ Catch2 micro-tests before attempting full pipeline integration.

---

## Phase 0: The Clean Slate ✅

**Goal:** Eradicate the old GF(2) theory, legacy prototypes, and obsolete opcodes to prevent hallucination of old architecture patterns.

*   **Task 0.1 (Delete Legacy):**
    *   Execute `git rm -rf prototype/` to permanently delete the Python prototype. Remove any references to it in `pyproject.toml`, `.pre-commit-config.yaml`, and CI workflows.
    *   Delete `src/ucc/backend/gf2_basis.cc` and `gf2_basis.h`. Remove them from `CMakeLists.txt`.
*   **Task 0.2 (Gut the Backend API):** In `src/ucc/backend/backend.h`:
    *   Delete the old `Opcode` enum.
    *   Strip out the old `AGMatrix` class entirely.
    *   Remove `x_mask`, `commutation_mask`, and the old union payloads from the `Instruction` struct. (Leave the struct definition empty/stubbed out but preserve `static_assert(sizeof(Instruction) == 32)`).
*   **Task 0.3 (Gut the SVM):** In `src/ucc/svm/svm.h` and `svm.cc`, strip out all old opcode execution switch cases.
*   **DoD:** The project compiles (even if the SVM doesn't do anything useful yet) and the codebase is completely purged of `GF2Basis`, `AGMatrix`, and `commutation_mask`.

---

## Phase 1: VM State & RISC Instruction Set ✅

**Goal:** Define the new execution state and implement the array/frame math for the localized RISC instructions. Test them in absolute isolation.

*   **Task 1.1 (SchrodingerState):** Update `SchrodingerState` in `svm.h`:
    *   Replace `uint64_t destab_signs` and `stab_signs` with `stim::bitword<kStimWidth> p_x, p_z;` (representing the Pauli frame $P$).
    *   Add `std::complex<double> gamma = {1.0, 0.0};` to track the global phase and deferred normalization.
    *   The $2^{k_{\max}}$ coefficient array `v_` remains the same.
*   **Task 1.2 (RISC Instruction Set):** Define the new `Opcode` enum and update the `Instruction` union payload in `backend.h`. Since operations are now highly localized, use `uint16_t` for axis indices.
    *   **Frame Opcodes:** `OP_FRAME_CNOT`, `OP_FRAME_CZ`, `OP_FRAME_H`, `OP_FRAME_S`, `OP_FRAME_SWAP`. (Payload: `uint16_t axis_1, axis_2`). These execute pure bitwise operations on `p_x` and `p_z` and do not touch the complex array.
    *   **Array Opcodes:** `OP_ARRAY_CNOT`, `OP_ARRAY_CZ`, `OP_ARRAY_SWAP`. (Payload: `uint16_t axis_1, axis_2`). These update `p_x` and `p_z` AND loop over `v_` to swap/negate amplitudes.
    *   **Local Fast-Paths:** `OP_EXPAND` (Virtual $H_v$ on dormant, doubles active dimension $k \to k+1$, divides `gamma` by $\sqrt{2}$), `OP_PHASE_T`, `OP_PHASE_T_DAG` (diagonal phase on active axis).
    *   **Measurement:** `OP_MEAS_DORMANT_STATIC`, `OP_MEAS_DORMANT_RANDOM`, `OP_MEAS_ACTIVE_DIAGONAL`, `OP_MEAS_ACTIVE_INTERFERE`. (Payload: target axis, classical target index).
*   **Task 1.3 (SVM Execution):** Implement the math for these opcodes in `svm.cc` based on `theory.tex` Sections 4 and 5.
    *   *Note:* `OP_MEAS_ACTIVE_INTERFERE` projects onto the $X$-basis. It must perform division-free addition/subtraction (folding $v_{i}$ and $v_{i \oplus 2^{k-1}}$), physically halve the array size ($k \to k-1$), and defer the normalization by updating `gamma`.
*   **Task 1.4 (Catch2 Micro-Tests):** Write `tests/test_svm_risc.cc`. Instantiate a dummy `SchrodingerState`. Manually construct a `std::vector<Instruction>` (e.g., `OP_EXPAND`, `OP_PHASE_T`, `OP_ARRAY_CNOT`). Execute them and assert the resulting complex amplitudes and `gamma` updates are mathematically exact.
*   **DoD:** Micro-tests pass, proving the raw SVM array/frame math perfectly mirrors theoretical equivalents without any AOT compiler involvement.

---

## Phase 2: The Statevector Expansion Bridge ✅

**Goal:** Prove the core mathematical equation $|\psi\rangle = \gamma U_C P (|\phi\rangle_A \otimes |0\rangle_D)$ works natively in C++, providing an anchor for future validation.

*   **Task 2.1 (Bridge Function):** Rewrite `ucc::get_statevector` in `svm.cc`.
    1. Expand the $2^k$ elements of $|\phi\rangle_A$ into a dense $2^n$ array, inserting zeros for all qubits in the Dormant set ($D$).
    2. Apply the Pauli frame $P$ (using `p_x`, `p_z`) to the dense state.
    3. Apply the offline compiled $U_C$ (the `final_tableau` from the Constant Pool) using Stim's `VectorSimulator` or native permutations.
    4. Multiply every element by the global scalar $\gamma$.
*   **Task 2.2 (Math Test):** Write a Catch2 test that manually populates a `SchrodingerState` with a known $P$, $\gamma$, and $v[]$, pairs it with a known Stim tableau for $U_C$, and calls `get_statevector`. Assert it perfectly matches a manual numpy-style statevector expansion.
*   **DoD:** We can accurately decode the factored state representation back into a standard dense $2^n$ statevector for exact validation.

---

## Phase 3: The Virtual Frame Compressor (AOT Back-End) ✅

**Goal:** Build the $\mathcal{O}(n)$ greedy Pauli reduction algorithm that translates global physical Pauli operations into localized virtual ones.

*   **Task 3.1 (Virtual Register Tracker):** In `backend.cc`, create a `VirtualRegisterManager`. It tracks which virtual qubits are in the Active ($A$) set vs Dormant ($D$) set, and maintains a bijective mapping from active virtual qubits to contiguous array axes ($0$ to $k-1$).
*   **Task 3.2 (Greedy Compression Logic):** Implement the algorithmic logic from Section 4.1 of `theory.tex`. Given a mapped virtual Pauli:
    *   **$X$-Compression:** Iterate through $X$ supports. Pivot preferentially on $D$. Append virtual `CNOT` sequences.
    *   **$Z$-Compression:** Iterate through $Z$ supports. Append virtual `CZ` or `CNOT` sequences.
    *   Maintain a tracking `stim::Tableau<kStimWidth> V_cum` (init to Identity). As you determine virtual gates, append them to $V_{cum}$ via `TableauTransposedRaii`.
*   **Task 3.3 (Opcode Router):** As the compressor decides on a virtual gate (e.g., CNOT from $v_c$ to $v_t$), consult the `VirtualRegisterManager`:
    *   If $v_c \in D$: Emit `OP_FRAME_CNOT` (Zero-cost dormant property).
    *   If $v_c \in A$ and $v_t \in A$: Emit `OP_ARRAY_CNOT`.
*   **Task 3.4 (Compression Micro-Test):** Write Catch2 tests that feed random, heavy `stim::PauliString` masks into the compressor. Verify that the resulting $V_{cum}$ successfully maps the heavy Pauli to a localized single-qubit $Z_v$ or $X_v$.
*   **DoD:** The compressor deterministically reduces any multi-qubit Pauli into a single-qubit virtual operation while emitting the exact sequence of RISC opcodes required to simulate it.

---

## Phase 4: Full Pipeline Integration & Oracles ✅

**Goal:** Wire the Front-End, Back-End, and VM together, utilizing the array compaction technique, and validate against Qiskit/Stim.

*   **Task 4.1 (Front-End Simplification):** In `frontend.cc`, rip out `fwd_after.then(inv_before)` and all AG pivot extraction. The Front-End no longer computes AG matrices. Measurements simply extract the $t=0$ Heisenberg mask and emit a `MEASURE` node to the HIR.
*   **Task 4.2 (Pipeline Wiring):** In `backend.cc` `lower()`, iterate through the HIR.
    *   Map the $t=0$ HIR Pauli mask to the *current* virtual frame: $P_v = V_{cum} P_{t=0} V_{cum}^\dagger$.
    *   Run the compressor on $P_v$.
    *   If non-Clifford ($T$-gate): emit `OP_EXPAND` (if dormant) and `OP_PHASE_T`. Update the Active set and register mapping.
    *   If Measurement: **Array Compaction**. Check the target's mapped axis. If it is in $A$ but is not at axis $k-1$, emit `OP_ARRAY_SWAP` and `OP_FRAME_SWAP` to route it to the top. Emit the appropriate measurement opcode, then demote the qubit to $D$.
*   **Task 4.3 (Final Frame computation):** At the end of lowering, compute $U_C = U_{phys} V_{cum}^\dagger$ and store it in the `ConstantPool` for use by `get_statevector`.
*   **Task 4.4 (Integration Tests):** Build the Python extensions (`uv pip install -e .`) and run the Python integration tests: `uv run pytest tests/python/ -v`.
*   **DoD:** `test_qiskit_aer.py` (Exact statevector equivalence) and `test_statistical_equivalence.py` (Stim distribution equivalence) pass 100%. The system correctly simulates universal circuits using the new factored theory.
