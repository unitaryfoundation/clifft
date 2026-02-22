# UCC MVP Implementation Plan

## Executive Summary & MVP Constraints
You are building the Phase 1 Minimum Viable Product (MVP) of the Unitary Compiler Collection (UCC).

**Strict Constraints for this specific MVP phase (Overrides general guidelines if in conflict):**
1. **64-Qubit Fast-Path ONLY:** Limit all Pauli masks to ≤64 qubits using `stim::bitword<64>` (zero-overhead wrapper around `uint64_t` with cleaner bitwise API). Do NOT use `stim::simd_bits<W>` for fixed-size masks—that type is for variable-length heap-allocated bit arrays. For future scaling beyond 64 qubits, upgrade to `bitword<256>` or `bitword<512>` (AVX/AVX-512).
2. **No C++ CLI:** **DO NOT** write a `main.cc` or a command-line executable. The engine is invoked exclusively via the Python API.
3. **Stub the Optimizer:** Pass the Heisenberg IR (HIR) directly from the Front-End to the Back-End. We are skipping Middle-End commutation/fusion for this milestone.
4. **Features:** Noiseless simulation only. Do not implement gap sampling, error models, or complex annotations.
5. **Circuit Scope:** Support basic Cliffords (`H`, `S`, `S_DAG`, `X`, `Y`, `Z`, `CX`, `CY`, `CZ`), $T$ and $T^\dagger$ gates, Measurements (`M`, `MX`, `MPP`), and classical Pauli feedback (`CX rec[-k] q`, `CZ rec[-k] q`).

---

## Phase 1: Project Setup & Build Infrastructure
**Goal:** Establish the CMake and Python build pipeline using `scikit-build-core` for seamless local Python development.
*   **Task 1.0 (Code Quality Scaffolding):**
    *   Create a robust `.gitignore` for C++ (CMake build dirs), Python (`__pycache__`, `.venv`, `*.so`), and IDEs.
    *   Configure `ruff` (lint/format) and `mypy` (strict typing) sections inside `pyproject.toml`. Add `pre-commit` to the dev dependencies.
    *   Create a standard `.clang-format` file (e.g., BasedOnStyle: Google, ColumnLimit: 100).
    *   Create `.pre-commit-config.yaml` hooking into `clang-format`, `ruff`, and `mypy`.
*   **Task 1.1:** Create `CMakeLists.txt` to compile a `ucc_core` library. Use `FetchContent` to pull exactly `v1.15.0` of `https://github.com/quantumlib/Stim.git` (https://github.com/quantumlib/Stim/tree/v1.15.0), and Catch2 (v3).
*   **Task 1.2:** Create `pyproject.toml` configured with `scikit-build-core` and `nanobind`.
*   **Task 1.3:** Create a minimal C++ function, bind it in `src/python/bindings.cc`, and expose it in `src/python/ucc/__init__.py`.
*   **Definition of Done (DoD):** Running `uv pip install -e .` succeeds, and the dummy function can be called from Python without errors. unning `uv run pre-commit run --all-files` passes completely with zero formatting, linting, or typing errors. The `git status` tree is completely free of untracked build artifacts, respecting the `.gitignore`.

## Phase 2: Circuit AST & Parser
**Goal:** Read a stripped `.stim` file into an abstract syntax tree.
*   **Task 2.1:** Define `ucc::Circuit`, `ucc::AstNode`, and the `ucc::GateType` enum. Include fields for targets and measurement record indices (`rec[-k]`).
*   **Task 2.2:** Write a line-by-line parser.
    *   *Resets Trick:* AOT simulators cannot branch the Clifford frame. Mathematically, a reset is a measurement followed by a conditionally applied Pauli. Decompose `R q` into `M q` followed by `CX rec[-1] q` (apply X if measure was 1). Decompose `RX q` into `MX q` followed by `CZ rec[-1] q`. This requires zero new SVM opcodes!
    *   Parse `MPP` into a list of Pauli targets.
    *   Parse `CX rec[-k] q` as a feedback operation.
*   **DoD:** A C++ Catch2 test parses a string containing gates, measurements, and a reset, verifying the AST matches the decomposed structure.

## Phase 3: Front-End (Trace Generation)
**Goal:** Drive Stim's `TableauSimulator` to absorb Cliffords and emit the Heisenberg IR (HIR).
*   **Task 3.1 (HIR Structs):** Define `HeisenbergOp` using `stim::bitword<64>` for `destab_mask` and `stab_mask` (32-byte cache-aligned layout). Define `HirModule` with `stim::Tableau<64>` for AG pivot matrices.
*   **Task 3.2 (Clifford & T Rewinding):** Iterate over the AST. When a Clifford is hit, apply it directly via `sim.inv_state.prepend_*()` for O(1) performance. When `T` or `T_DAG` is hit, extract the rewound Pauli Z from `sim.inv_state.zs[q]`, and emit a `HeisenbergOp` with `type=T_GATE` and `is_dagger=false` or `is_dagger=true` respectively.
    *   *(Debug Optional):* At the end of trace generation, extract and save the forward tableau (`sim.inv_state.inverse()`) to `HirModule::final_tableau` for statevector debugging.
*   **Task 3.3 (Measurements & AG Pivots):** When an `M`, `MX`, or `MPP` is hit, rewind the observable. If the measurement commutes with the tableau, record it. If it anti-commutes, compute the Aaronson-Gottesman (AG) pivot matrix using `fwd_after.then(inv_before)`, update the tableau, append the matrix to `HirModule::ag_matrices`, and emit `HeisenbergOp::MEASURE`.
*   **Task 3.4 (Classical Control):** For `CX rec[-k] q` (and the decomposed resets), rewind the target Pauli ($X_q$ or $Z_q$). Emit an `OP_CONDITIONAL_PAULI` operation in the HIR containing the rewound mask and the resolved absolute index of the measurement record.
*   **DoD:** A test feeding `H 0; T 0` verifies the emitted HIR mask corresponds to the $+X$ axis. A test feeding `H 0; M 0` verifies an AG matrix is generated.

## Phase 4: Compiler Back-End (Code Generation)
**Goal:** Lower the HIR into execution-engine bytecode (Skipping the Optimizer for now).
*   **Task 4.1:** Define the `Instruction` struct (union). **Assert that it is exactly 32 bytes.** Define the `ConstantPool`.
*   **Task 4.2 (GF2 Tracking):** Iterate through the HIR. Maintain the active GF(2) basis $V$ (up to 64 `bitword<64>` vectors). For `T_GATE` (regardless of `is_dagger`), evaluate the spatial shift $\beta$ against $V$. Emit `OP_BRANCH` (if new dimension), `OP_COLLIDE` (if in basis), or `OP_SCALAR_PHASE` (if $\beta=0$). Compute `x_mask` and `commutation_mask`. Track `peak_rank`. Store the final $V$ basis in `ConstantPool::gf2_basis` if debugging/targetting state vector output.
*   **Task 4.3 (Measurements & Feedback):** Lower `MEASURE` operations into `OP_MEASURE_MERGE`, `OP_MEASURE_FILTER`, or `OP_MEASURE_DETERMINISTIC` by evaluating against $V$. Append `OP_AG_PIVOT` if the Front-End provided an AG matrix index. Lower `OP_CONDITIONAL_PAULI` directly into a bytecode instruction.
*   **DoD:** A test showing 4 independent $T$ gates emits 4 `OP_BRANCH` opcodes and yields a `peak_rank` of 4.

## Phase 5: Schrodinger Virtual Machine Runtime
**Goal:** Execute the bytecode and track the complex coefficient array.
*   **Task 5.1:** Define `SchrodingerState` containing a dynamically allocated `std::complex<double>` array of size $2^{\text{peak\_rank}}$, `uint64_t destab_signs`, `uint64_t stab_signs`, and a `std::vector<uint8_t> meas_record`. Include a deterministic PRNG.
*   **Task 5.2:** Implement the SVM switch statement for all opcodes.
    *   `OP_BRANCH`, `OP_COLLIDE`, `OP_SCALAR_PHASE`.
    *   `OP_MEASURE_MERGE`, `OP_MEASURE_FILTER`, `OP_MEASURE_DETERMINISTIC`, `OP_AG_PIVOT`.
    *   `OP_CONDITIONAL_PAULI` (if `meas_record[idx] == 1`, XOR the masks into `destab_signs`/`stab_signs`).
*   **Task 5.3:** Expose `ucc.compile` and `ucc.sample(program, shots)` via nanobind.
*   **DoD:** Python can successfully call `ucc.sample()` on a compiled program and receive a numpy array of results.

## Phase 6: Exact AG_PIVOT Geometry & Physics Closure
**Goal:** Close the technical debt on Aaronson-Gottesman measurement pivots and scalar phase commutation. The MVP SVM currently uses a simplified XOR to propagate measurement divergences, which breaks entanglement correlations in complex circuits. It must execute the full $\mathcal{O}(n^2)$ matrix-vector multiplication. Additionally, `OP_SCALAR_PHASE` must correctly evaluate commutation masks.


*   **Task 6.1 (Frontend AG Slot Extraction):**
    *   Update `HeisenbergOp::measure_` and its factory `make_measure` to store a `uint8_t ag_stab_slot`.
    *   In `frontend.cc` (`collapse_z_measurement`, `collapse_x_measurement`, `collapse_y_measurement`, `MPP`), after computing the AG pivot matrix (`fwd_after.then(inv_before)`), find which stabilizer row in the post-collapse tableau matches the measured observable (ignoring sign).
    *   *Algorithm:* Iterate $i$ from $0$ to `sim.num_qubits - 1`. Extract $s =$ `forward_after.z_output(i)`. Set `s.sign = false`. If $s$ matches the measured observable (also with `sign = false`), store $i$ as `ag_stab_slot` and emit it in the HIR.
    *   **Add this exact comment above the loop in `frontend.cc`:**
        `// TODO(perf): Stim's internal collapse_qubit_z returns the pivot index, but do_MZ discards it.`
        `// If upstream Stim ever exposes this, or if we bypass do_MZ to call collapse_qubit_z directly,`
        `// we can drop this O(n) scan and achieve O(1) retrieval.`
*   **Task 6.2 (Backend Propagation & SCALAR_PHASE Fix):**
    *   In `backend.h`, update `Instruction::meta` to explicitly include `uint32_t ag_stab_slot` (can be safely aliased in the union with `controlling_meas`).
    *   In `backend.cc`, map the HIR's `ag_stab_slot` into the emitted `OP_AG_PIVOT` instruction's meta payload.
    *   *SCALAR_PHASE Fix:* In `backend.cc`, when $\beta = 0$ (emitting `OP_SCALAR_PHASE`), **do not** hardcode `commutation_mask = 0`. Compute it using `compute_commutation_mask(basis.vectors(), destab, stab)` because diagonal operations can still anti-commute with active superposition dimensions.
*   **Task 6.3 (SVM SCALAR_PHASE Execution):**
    *   In `svm.cc`, update `op_scalar_phase`. Precompute `factors[2] = {1.0 + i_tan * base_phase, 1.0 - i_tan * base_phase}`.
    *   Inside the loop, compute `parity = compute_sign_parity(i, instr.commutation_mask)` and multiply `v[i] *= factors[parity]`.
*   **Task 6.4 (SVM AG_PIVOT Matrix Math):**
    *   In `svm.cc` (`op_ag_pivot`), completely remove the MVP simplification that just XORs `destab_mask`/`stab_mask`.
    *   Implement the full $64 \times 64$ bitwise matrix-vector multiplication. Map the old `state.destab_signs` and `state.stab_signs` into `new_destab` and `new_stab` by evaluating the symplectic inner product against the columns of `mat.xs[i]` and `mat.zs[i]`.
    *   If the physical outcome diverges from `ag_ref_outcome` (i.e., `divergence == 1`), inject the localized error by flipping the sign of the new stabilizer exactly at the target slot: `new_stab ^= (1ULL << instr.meta.ag_stab_slot);`
*   **DoD:** All C++ tests pass. Write a new C++ Catch2 test verifying `OP_SCALAR_PHASE` correctly branches phases when `commutation_mask != 0` (e.g. `H 0; T 0; S 0; H 0; T 0`). Write a new C++ test verifying `OP_AG_PIVOT` correctly propagates prior Pauli errors through an asymmetric frame collapse.

## Phase 7: Validation & Integration Testing
**Goal:** Prove the system produces mathematically correct physics.
*   **Task 7.1 (Statevector Expansion Utility):** Write a Python utility `extract_statevector` that:
    1. Allocates a dense $2^N$ numpy array.
    2. For each GF(2) index $\alpha$ (from 0 to $2^{\text{rank}}-1$), compute the physical basis index by XORing the columns of `gf2_basis` selected by the bits of $\alpha$.
    3. Apply the physical $\pm 1, \pm i$ signs from `destab_signs`/`stab_signs` using the `final_tableau` columns to determine which stabilizer/destabilizer generators contribute.
    4. Place `v[α] * sign * global_weight` at the computed physical index.
*   **Task 7.2 (Statevector Micro-Test):** Use the utility from 7.1 to validate a 4-qubit pure unitary circuit (Cliffords + $T$, no measurements) against a known statevector oracle using `np.allclose`. *(Note: because the VM uses Dominant Term Factoring, the VM's raw state is unnormalized; ensure the utility accounts for this).*
*   **Task 7.3 (Statistical Macro-Test):** Load the manually stripped Gidney proxy circuit (S gates only). Run 10,000 shots in `stim` and 10,000 shots in `ucc`. Assert that the probability distributions of the resulting measurement bitstrings match within a strict statistical tolerance (e.g., $< 0.02$ divergence).
*   **Task 7.4 (Statevector Debugging API):** Add a `debug_mode=False` flag to `ucc.compile()`. When true, retain `final_tableau` and `gf2_basis` on the returned `Program`. Expose `ucc.to_statevector(program, schrodinger_state)` that leverages the math from Task 7.1.
*   **DoD:** All tests pass reliably via `pytest`.

---

## Follow-ups (Post-MVP)

Items identified during implementation review for future improvement:

### API Encapsulation
- **`v()` accessor removal**: The current `SchrodingerState::v()` exposes raw coefficient array for testing convenience. Refactor to encapsulate execution: `sample()` should accept a caller-provided output buffer or return results directly, eliminating public access to internal state.
- **ndarray protocol for results**: Current Python `sample()` returns `list[list[int]]`. Investigate returning a numpy ndarray with zero-copy semantics (allocate numpy array, pass data pointer to C++). Reference Stim's approach for guidance.

### Robustness
- **Allocation size limits**: Consider adding configurable limits on `peak_rank` to prevent accidental multi-GB allocations.
